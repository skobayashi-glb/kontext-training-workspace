"""Local controller. AWS CLI supplies short-lived credentials; SSH verifies keys."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {'completed', 'failed', 'cancelled', 'submission_failed'}


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default


def valid_job(job: str) -> str:
    if not re.fullmatch('[a-f0-9]{32}', job):
        raise ValueError('ジョブIDは32文字の16進数で指定してください。')
    return job


class Settings:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.data = json.loads(self.path.read_text(encoding='utf-8-sig'))
        for name in ['instance_id', 'region', 'profile', 'ssh_alias', 'host_key_alias',
                     'remote_project', 'remote_python', 'remote_base']:
            setattr(self, name, self.data[name])
        for name in ['remote_project', 'remote_python', 'remote_base']:
            value = getattr(self, name)
            if not value.startswith('/') or any(c in value for c in ['\n', '\r', '\x00']):
                raise ValueError('無効なリモートパス: ' + name)
        self.project = (self.path.parent / self.data['local_project']).resolve()
        self.cache = (self.path.parent / self.data['cache_dir']).resolve()
        self.setup_timeout = int(self.data.get('setup_timeout_seconds', 360))
        self.grace = int(self.data.get('collection_grace_seconds', 90))
        if not 60 <= self.setup_timeout <= 900 or not 15 <= self.grace <= 600:
            raise ValueError('タイムアウト設定の範囲が不正です。')

    def remote_dir(self, job):
        return self.remote_base + '/jobs/' + valid_job(job)


class AwsClient:
    def __init__(self, settings):
        self.settings = settings

    def call(self, operation, *args):
        executable = shutil.which('aws')
        if not executable:
            raise RuntimeError('AWS CLI v2が必要です。')
        command = [executable, 'ec2', operation, *args, '--region', self.settings.region,
                   '--profile', self.settings.profile, '--output', 'json', '--no-cli-pager',
                   '--cli-connect-timeout', '5', '--cli-read-timeout', '15']
        result = subprocess.run(command, capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace').strip())
        return json.loads(result.stdout)

    def instance(self):
        return self.call('describe-instances', '--instance-ids', self.settings.instance_id)['Reservations'][0]['Instances'][0]

    def verify_stop(self):
        if not self.settings.data.get('dedicated_instance'):
            raise RuntimeError('自動停止には dedicated_instance=true が必要です。')
        if self.instance().get('RootDeviceType') != 'ebs':
            raise RuntimeError('EBSルートインスタンスだけに対応しています。')
        behavior = self.call('describe-instance-attribute', '--instance-id', self.settings.instance_id,
                             '--attribute', 'instanceInitiatedShutdownBehavior')
        if behavior['InstanceInitiatedShutdownBehavior']['Value'] != 'stop':
            raise RuntimeError('OSシャットダウン設定がstopではありません。設定変更は行いません。')

    def wait(self, target, timeout=240):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            item = self.instance()
            if item['State']['Name'] == target:
                return item
            if item['State']['Name'] in {'terminated', 'shutting-down'}:
                raise RuntimeError('対象EC2は削除状態です。')
            time.sleep(5)
        raise TimeoutError('EC2状態待ちがタイムアウトしました: ' + target)

    def start(self):
        current = self.instance()['State']['Name']
        if current == 'stopping':
            print('前回のEC2停止完了を待っています。', flush=True)
            self.wait('stopped', 900)
            current = 'stopped'
        if current == 'stopped':
            self.call('start-instances', '--instance-ids', self.settings.instance_id)
            return True
        if current not in {'pending', 'running'}:
            raise RuntimeError('起動できないEC2状態: ' + current)
        return False

    def stop(self):
        state = self.instance()['State']['Name']
        if state not in {'stopped', 'stopping', 'terminated', 'shutting-down'}:
            self.call('stop-instances', '--instance-ids', self.settings.instance_id)


class SshClient:
    def __init__(self, settings, ip):
        self.settings, self.ip = settings, ip
        if not ip:
            raise RuntimeError('パブリックIPがありません。ネットワーク設定を確認してください。')

    def command(self, remote):
        if os.name == 'nt':
            executable = str(Path(os.environ['WINDIR']) / 'System32/OpenSSH/ssh.exe')
        else:
            executable = shutil.which('ssh')
        if not executable:
            raise RuntimeError('OpenSSHが必要です。')
        return [executable, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ConnectTimeout=5', '-o', 'ConnectionAttempts=1',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                '-o', 'HostName=' + self.ip, '-o', 'HostKeyAlias=' + self.settings.host_key_alias,
                self.settings.ssh_alias, remote]

    def run(self, remote, data=None, timeout=45):
        result = subprocess.run(self.command(remote), input=data, capture_output=True, timeout=timeout)
        if result.returncode:
            error = result.stderr.decode('utf-8', errors='replace').strip()
            raise RuntimeError(error or result.stdout.decode('utf-8', errors='replace').strip())
        return result.stdout

    def wait(self, timeout=180):
        deadline, last = time.monotonic() + timeout, ''
        while time.monotonic() < deadline:
            try:
                self.run('true', timeout=12)
                return self
            except RuntimeError as error:
                last = str(error)
                if 'Host key verification failed' in last or 'IDENTIFICATION HAS CHANGED' in last:
                    raise
            except subprocess.TimeoutExpired:
                pass
            time.sleep(3)
        raise TimeoutError('SSH接続待ちがタイムアウトしました: ' + last)

    def agent_command(self, job, action, weights=False):
        directory = self.settings.remote_dir(job)
        args = [self.settings.remote_python, directory + '/agent.py', action, directory]
        if weights:
            args.append('--weights')
        return shlex.join(args)

    def agent(self, job, action):
        return json.loads(self.run(self.agent_command(job, action)))

    def collect(self, job, output, weights=False):
        evaluation = output.parent / 'training/evaluation'
        skip = [p.relative_to(output.parent).as_posix() for p in evaluation.rglob('*') if p.is_file() and p.suffix.lower() in {'.png', '.jpg'}] if evaluation.exists() else []
        with output.open('wb') as stream:
            result = subprocess.run(self.command(self.agent_command(job, 'collect', weights)),
                                    input=json.dumps({'skip': skip}).encode(), stdout=stream, stderr=subprocess.PIPE, timeout=300 if weights else 60)
        if result.returncode:
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))


class SourceBundle:
    EXCLUDE = {'.git', '.venv', '.venv312', '__pycache__', 'datasets', 'models', 'runs', 'data',
               'node_modules', 'dist', 'build', '.cache', '.agents', '.codex', 'ui'}
    SUFFIXES = {'.py', '.toml', '.yaml', '.yml', '.json', '.sh', '.md'}

    def __init__(self, settings):
        self.settings = settings

    def files(self):
        if not (self.settings.project / 'src/flux_kontext_training/remote_cli.py').is_file():
            raise RuntimeError('ローカル学習コードが見つかりません。local_projectを確認してください。')
        result = []
        for directory, dirs, names in os.walk(self.settings.project, followlinks=False):
            dirs[:] = [name for name in dirs if name not in self.EXCLUDE and not name.startswith('.')
                       and not name.endswith('.egg-info') and not (Path(directory) / name).is_symlink()]
            for name in names:
                path = Path(directory) / name
                if not name.startswith('.') and path.suffix in self.SUFFIXES and not path.is_symlink():
                    result.append(path)
        if sum(p.stat().st_size for p in result) > 20_000_000:
            raise ValueError('コード同期対象が20 MBを超えています。データ混入を確認してください。')
        return sorted(result)

    def build(self, request, config=None):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            for path in self.files():
                archive.add(path, arcname='workspace/' + path.relative_to(self.settings.project).as_posix(), recursive=False)
            for name, content in [('agent.py', (Path(__file__).parent / 'agent.py').read_bytes()),
                                  ('request.json', json.dumps(request).encode('utf-8'))]:
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(content), 0o600
                archive.addfile(info, io.BytesIO(content))
            if config is not None:
                content = json.dumps(config, ensure_ascii=False).encode('utf-8')
                info = tarfile.TarInfo('workspace/_aws_training.json')
                info.size, info.mode = len(content), 0o600
                archive.addfile(info, io.BytesIO(content))
        return buffer.getvalue()


class ResultCache:
    def __init__(self, settings):
        self.settings = settings

    def directory(self, job):
        return self.settings.cache / valid_job(job)

    def latest(self):
        latest = load_json(self.settings.cache / 'latest.json', {})
        if not latest.get('job_id'):
            raise RuntimeError('ジョブがありません。先にtrainを実行してください。')
        return valid_job(latest['job_id'])

    def fetch(self, ssh, job, weights=False):
        directory = self.directory(job)
        directory.mkdir(parents=True, exist_ok=True)
        # Background collection and an explicit fetch may overlap after sleep.
        with CacheLock(directory / 'fetch.lock'):
            return self._fetch(ssh, job, weights)

    def _fetch(self, ssh, job, weights):
        directory = self.directory(job)
        temporary = directory / 'incoming.tar.gz'
        ssh.collect(job, temporary, weights)
        # Remote archives are treated as untrusted paths, never extracted with extractall.
        with tarfile.open(temporary, 'r:gz') as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (not member.isfile() or name.is_absolute() or '..' in name.parts
                        or '\\' in member.name or ':' in member.name):
                    raise ValueError('不正な結果アーカイブのパスです。')
                if member.name not in {'status.json', 'request.json', 'log.txt', 'training.json'} and not member.name.startswith('training/'):
                    raise ValueError('想定外の結果ファイルです。')
                if not weights and member.size > 100_000_000:
                    raise ValueError('ログファイルが大きすぎます。EBS側の原本は保持されています。')
                target = directory.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_name(target.name + '.download')
                with archive.extractfile(member) as source, pending.open('wb') as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                pending.replace(target)
        temporary.unlink(missing_ok=True)
        status = load_json(directory / 'status.json', {})
        if status.get('state') in TERMINAL:
            save_json(directory / 'collection.json', {'complete': True, 'cached_at': time.time()})
            # Only release remote shutdown after the final logs/metrics are saved.
            ssh.agent(job, 'ack')
        return status


class CacheLock:
    def __init__(self, path):
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = self.path.open('a+b')
        self.file.write(b'0')
        self.file.flush()
        deadline = time.monotonic() + 330
        while True:
            try:
                self.file.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self.file.close()
                    raise TimeoutError('別の結果取得処理が終了しませんでした。')
                time.sleep(1)

    def __exit__(self, *args):
        self.file.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


class TrainingLauncher:
    def __init__(self, settings):
        self.settings = settings
        self.aws = AwsClient(settings)
        self.cache = ResultCache(settings)

    def connect(self):
        current = self.aws.wait('running', 240)
        if not current.get('PublicIpAddress'):
            raise RuntimeError('EC2にパブリックIPがありません。')
        return SshClient(self.settings, current['PublicIpAddress']).wait()

    def background(self, args, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('ab') as log:
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            subprocess.Popen([sys.executable, str(ROOT / 'aws_run.py'), '--settings', str(self.settings.path), *args],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=flags, start_new_session=os.name != 'nt')

    def deploy(self, ssh, job, bundle):
        directory = self.settings.remote_dir(job)
        script = '''import io,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1])
root.mkdir(parents=True,exist_ok=False,mode=0o700)
payload=sys.stdin.buffer.read(25000001)
if len(payload)>25000000:raise ValueError('Bundle too large')
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
 for member in archive:
  name=pathlib.PurePosixPath(member.name)
  if not member.isfile() or name.is_absolute() or '..' in name.parts or '\\\\' in member.name:raise ValueError('Unsafe archive')
  target=root.joinpath(*name.parts)
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(archive.extractfile(member).read())
'''
        ssh.run(shlex.join(['python3', '-c', script, directory]), bundle)

    def guard(self, ssh, job):
        directory = self.settings.remote_dir(job)
        ssh.run(shlex.join(['sudo', '-n', 'systemd-run', '--unit=kontext-setup-' + job,
                            '--on-active=' + str(self.settings.setup_timeout) + 's', '/usr/bin/python3',
                            directory + '/agent.py', 'guard', directory]))

    def recover_pending(self, ssh, current_job):
        # Reuse this training startup to collect logs missed while the PC was off.
        if not self.settings.cache.exists():
            return
        for directory in self.settings.cache.iterdir():
            if not directory.is_dir() or not re.fullmatch('[a-f0-9]{32}', directory.name) or directory.name == current_job:
                continue
            meta = load_json(directory / 'local.json', {})
            state = load_json(directory / 'status.json', {})
            collected = load_json(directory / 'collection.json', {})
            if (meta.get('accepted') or meta.get('submission_attempted')) and (state.get('state') not in TERMINAL or not collected.get('complete')):
                try:
                    recovered = self.cache.fetch(ssh, directory.name)
                    if recovered.get('state') in TERMINAL:
                        self.maybe_review(directory.name)
                except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
                    print('前回ログの補完を保留: ' + str(error), file=sys.stderr)

    def train(self, config_path=None, command=None, dry_run=False, evaluate_only=False, auto_review=True):
        config = None
        if config_path:
            # Schema validation is CPU-only and happens BEFORE any paid operation.
            sys.path.insert(0, str(self.settings.project / 'src'))
            from flux_kontext_training.remote_cli import CLITrainingConfig
            config = CLITrainingConfig.model_validate_json(config_path.read_text(encoding='utf-8-sig')).model_dump()
            module = 'flux_kontext_training.review_existing' if evaluate_only else 'flux_kontext_training.remote_cli'
            command = [self.settings.remote_python, '-m', module, '--config', '_aws_training.json']
        if not command:
            raise ValueError('学習設定かリモートコマンドを指定してください。')
        if command[0] in {'python', 'python3'}:
            command = [self.settings.remote_python, *command[1:]]
        job = uuid.uuid4().hex
        request = {'command': command, 'python': self.settings.remote_python,
                   'remote_project': self.settings.remote_project, 'auto_stop': True,
                   'max_runtime_seconds': self.settings.data.get('evaluation_timeout_seconds', 900) if evaluate_only else None,
                   'collection_grace_seconds': self.settings.grace,
                   'source_checkpoint_id': config.get('source_checkpoint_id') if config else None}
        bundle = SourceBundle(self.settings).build(request, config)
        if dry_run:
            print(json.dumps({'instance_id': self.settings.instance_id, 'command': command,
                              'sync_files': [p.relative_to(self.settings.project).as_posix() for p in SourceBundle(self.settings).files()],
                              'bundle_bytes': len(bundle), 'auto_stop': True,
                              'AWS_SSH_access': False}, ensure_ascii=False, indent=2))
            return job
        self.aws.verify_stop()
        directory = self.cache.directory(job)
        metadata = {'job_id': job, 'accepted': False, 'submission_attempted': False,
                    'created_at': time.time(), 'setup_deadline': time.time() + self.settings.setup_timeout,
                    'start_attempted': False, 'started_here': False,
                    'auto_review': auto_review and self.settings.data.get('auto_review', True),
                    'kind': 'evaluation' if evaluate_only else 'training'}
        save_json(directory / 'local.json', metadata)
        save_json(self.settings.cache / 'latest.json', {'job_id': job})
        started_here = False
        try:
            initial = self.aws.instance()['State']['Name']
            if initial in {'stopped', 'stopping'}:
                if initial == 'stopping':
                    self.aws.wait('stopped', 900)
                metadata['start_attempted'] = True
                metadata['setup_deadline'] = time.time() + self.settings.setup_timeout
                save_json(directory / 'local.json', metadata)
                self.background(['_setup-watch', job], directory / 'setup-watch.log')
                started_here = True  # Also clean up an ambiguous StartInstances timeout.
            self.aws.start()
            metadata['started_here'] = started_here
            save_json(directory / 'local.json', metadata)
            print('EC2とSSHの準備を待っています。', flush=True)
            ssh = self.connect()
            if isinstance(getattr(ssh, 'ip', None), str):
                metadata['public_ip'] = ssh.ip
            self.recover_pending(ssh, job)
            self.deploy(ssh, job, bundle)
            if started_here:
                self.guard(ssh, job)
            # Serialize submission with the local setup watchdog. A timeout
            # that wins the lock prevents submission rather than killing a
            # job that has just started behind the watchdog's earlier read.
            with CacheLock(directory / 'setup-state.lock'):
                latest = load_json(directory / 'local.json', {})
                if latest.get('setup_expired'):
                    raise TimeoutError('準備期限が過ぎたため学習を投入せず停止しました。')
                metadata['submission_attempted'] = True
                save_json(directory / 'local.json', metadata)
            accepted = ssh.agent(job, 'submit')
            if not accepted.get('accepted'):
                raise RuntimeError('ジョブ受付を確認できませんでした。')
            metadata['accepted'] = True
            save_json(directory / 'local.json', metadata)
            try:
                self.background(['_watch', job], directory / 'collector.log')
            except OSError as error:
                print('ログ取得プロセスを開始できませんでした。学習は継続中です: ' + str(error), file=sys.stderr)
            kind = '評価' if evaluate_only else '学習'
            print(f'{kind}ジョブを受け付けました: {job}\nここからはPCを閉じてもEC2で処理を継続します。')
            print('ログ・結果: python aws_run.py view')
            return job
        except BaseException:
            if started_here and not metadata['submission_attempted']:
                self.aws.stop()
            elif metadata['submission_attempted']:
                print('受付状況が不明なため学習を強制停止していません。statusで確認してください。', file=sys.stderr)
            raise

    def setup_watch(self, job):
        path = self.cache.directory(job) / 'local.json'
        while True:
            meta = load_json(path, {})
            if meta.get('accepted') or meta.get('submission_attempted'):
                return  # Remote guard owns ambiguous submissions; never cut off a real training job.
            if time.time() >= meta.get('setup_deadline', 0):
                with CacheLock(path.parent / 'setup-state.lock'):
                    meta = load_json(path, {})
                    if meta.get('accepted') or meta.get('submission_attempted'):
                        return
                    meta['setup_expired'] = True
                    save_json(path, meta)
                    if meta.get('start_attempted'):
                        self.aws.stop()
                return
            time.sleep(3)

    def maybe_review(self, job):
        meta = load_json(self.cache.directory(job) / 'local.json', {})
        if not meta.get('auto_review', False):
            return
        review = self.cache.directory(job) / 'review'
        review.mkdir(exist_ok=True)
        with CacheLock(review / 'dispatch.lock'):
            state_file = review / 'state.json'
            if state_file.exists():
                return
            save_json(state_file, {'state': 'queued', 'queued_at': time.time()})
            self.background(['_review', job], review / 'process.log')

    def watch(self, job):
        directory = self.cache.directory(job)
        # The address stays fixed while the instance runs. SSH collection must
        # survive AWS login expiry during multi-hour training.
        ip = load_json(directory / 'local.json', {}).get('public_ip')
        failures = 0
        while True:
            try:
                if ip:
                    try:
                        status = self.cache.fetch(SshClient(self.settings, ip), job)
                        failures = 0
                        if status.get('state') in TERMINAL:
                            self.maybe_review(job)
                            return
                        time.sleep(10)
                        continue
                    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as error:
                        failures += 1
                        print(str(error), flush=True)
                item = self.aws.instance()
                if item['State']['Name'] in {'stopped', 'stopping'}:
                    save_json(directory / 'instance.json', {'state': item['State']['Name'], 'checked_at': time.time()})
                    return
                if item['State']['Name'] == 'running' and item.get('PublicIpAddress'):
                    ip = item['PublicIpAddress']
                    status = self.cache.fetch(SshClient(self.settings, item['PublicIpAddress']), job)
                    if status.get('state') in TERMINAL:
                        self.maybe_review(job)
                        return
            except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as error:
                print(str(error), flush=True)
                if failures >= 18:
                    save_json(directory / 'collection_deferred.json', {'reason': str(error), 'time': time.time()})
                    return
            time.sleep(10)

    def fetch(self, job, start=False, weights=False):
        owned = False
        try:
            item = self.aws.instance()
            if item['State']['Name'] != 'running':
                if not start:
                    print('EC2は停止中です。取得済みログはviewで閲覧できます。最終結果の取得は fetch --start を指定してください。')
                    return
                self.aws.verify_stop()
                owned = True
                self.aws.start()
            ssh = self.connect()
            if owned:
                self.guard(ssh, job)
            status = self.cache.fetch(ssh, job, weights)
            if status.get('state') in TERMINAL:
                self.maybe_review(job)
            print(json.dumps(status, ensure_ascii=False, indent=2))
            print('取得先: ' + str(self.cache.directory(job)))
        finally:
            if owned:
                self.aws.stop()

    def status(self, job):
        data = load_json(self.cache.directory(job) / 'status.json', {'state': 'not_cached'})
        try:
            item = self.aws.instance()
            data['instance_state'] = item['State']['Name']
            if item['State']['Name'] == 'running':
                data.update(SshClient(self.settings, item['PublicIpAddress']).agent(job, 'status'))
        except (RuntimeError, subprocess.SubprocessError) as error:
            data['connection_error'] = str(error)
        print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--':
        argv = ['run', *argv]
    parser = argparse.ArgumentParser(description='ローカル編集・EC2オンデマンド学習')
    parser.add_argument('--settings', type=Path, default=ROOT / 'aws_config.json')
    sub = parser.add_subparsers(dest='action', required=True)
    train = sub.add_parser('train', help='設定JSONを使って学習を投入')
    train.add_argument('--config', type=Path, default=ROOT / 'training_config.json')
    train.add_argument('--dry-run', action='store_true')
    train.add_argument('--no-auto-review', action='store_true')
    evaluation = sub.add_parser('evaluate', help='既存モデルとベースを画像評価。新規学習はしません')
    evaluation.add_argument('--config', type=Path, default=ROOT / 'training_config.json')
    evaluation.add_argument('--dry-run', action='store_true')
    evaluation.add_argument('--no-auto-review', action='store_true')
    run = sub.add_parser('run', help='任意の有限コマンドをリモートで実行')
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('command', nargs=argparse.REMAINDER)
    for name in ['status', 'cancel', '_watch', '_setup-watch']:
        p = sub.add_parser(name)
        p.add_argument('job', nargs='?')
    fetch = sub.add_parser('fetch', help='結果取得。--start指定時だけ起動を許可')
    fetch.add_argument('job', nargs='?')
    fetch.add_argument('--start', action='store_true')
    fetch.add_argument('--weights', action='store_true')
    control = sub.add_parser('control', help='画像評価に基づく学習率変更・早期終了の指示')
    control.add_argument('job', nargs='?')
    control.add_argument('--file', type=Path, required=True)
    logs = sub.add_parser('logs', help='ローカルに取得済みのログを表示')
    logs.add_argument('job', nargs='?')
    logs.add_argument('--follow', action='store_true')
    view = sub.add_parser('view', help='ローカルの結果・loss・ログをブラウザーで閲覧')
    view.add_argument('--port', type=int, default=8765)
    for name in ['review', '_review']:
        review = sub.add_parser(name, help='取得済み結果を診断。EC2起動・学習はしません')
        review.add_argument('job', nargs='?')
        review.add_argument('--force', action='store_true')
    sub.add_parser('jobs', help='ローカルにあるジョブ一覧')
    args = parser.parse_args(argv)
    settings = Settings(args.settings)
    launcher = TrainingLauncher(settings)
    try:
        if args.action == 'train':
            launcher.train(args.config, dry_run=args.dry_run, auto_review=not args.no_auto_review)
        elif args.action == 'evaluate':
            launcher.train(args.config, dry_run=args.dry_run, evaluate_only=True, auto_review=not args.no_auto_review)
        elif args.action == 'run':
            command = args.command[1:] if args.command and args.command[0] == '--' else args.command
            launcher.train(command=command, dry_run=args.dry_run)
        elif args.action == 'view':
            from .viewer import ResultViewer
            ResultViewer(settings.cache).serve(args.port)
        elif args.action == 'jobs':
            if settings.cache.exists():
                for directory in sorted(settings.cache.iterdir(), reverse=True):
                    if directory.is_dir() and re.fullmatch('[a-f0-9]{32}', directory.name):
                        data = load_json(directory / 'status.json', {})
                        print(directory.name, data.get('state', 'not_cached'))
        else:
            job = valid_job(args.job) if args.job else launcher.cache.latest()
            if args.action == 'status':
                launcher.status(job)
            elif args.action in {'review', '_review'}:
                from .post_review import PostTrainingReview
                print(PostTrainingReview(settings, job).run(force=args.force))
            elif args.action == 'fetch':
                launcher.fetch(job, args.start, args.weights)
            elif args.action == 'cancel':
                item = launcher.aws.instance()
                if item['State']['Name'] != 'running':
                    raise RuntimeError('EC2は実行中ではありません。')
                print(SshClient(settings, item['PublicIpAddress']).agent(job, 'cancel'))
            elif args.action == 'control':
                item = launcher.aws.instance()
                if item['State']['Name'] != 'running':
                    raise RuntimeError('EC2は実行中ではありません。')
                ssh = SshClient(settings, item['PublicIpAddress'])
                print(ssh.run(ssh.agent_command(job, 'control'), args.file.read_bytes()).decode('utf-8'))
            elif args.action == '_watch':
                launcher.watch(job)
            elif args.action == '_setup-watch':
                launcher.setup_watch(job)
            elif args.action == 'logs':
                last = ''
                while True:
                    path = launcher.cache.directory(job) / 'log.txt'
                    content = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
                    if content != last:
                        print(content[len(last):] if content.startswith(last) else content, end='', flush=True)
                        last = content
                    if not args.follow:
                        if not content:
                            print('ログはまだ取得されていません。')
                        break
                    time.sleep(2)
        return 0
    except KeyboardInterrupt:
        print('ローカル操作を終了しました。受付済み学習ジョブは継続します。', file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
