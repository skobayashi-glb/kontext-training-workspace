"""Standalone Linux job agent. Runs training as the SSH user, under systemd."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time
import traceback


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


class JobAgent:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.job_id = self.directory.name
        if not re.fullmatch('[a-f0-9]{32}', self.job_id):
            raise ValueError('Invalid job id')
        self.request = json.loads((self.directory / 'request.json').read_text())
        self.unit = 'kontext-train-' + self.job_id
        self.root = self.directory.parent.parent
        self.status_path = self.directory / 'status.json'
        self.boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip() if os.name != 'nt' else 'test'

    def state(self):
        result = subprocess.run(['systemctl', 'show', self.unit, '--property=ActiveState', '--value'],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip()

    def status(self):
        data = json.loads(self.status_path.read_text()) if self.status_path.exists() else {'state': 'prepared'}
        data['unit_state'] = self.state()
        data['job_id'] = self.job_id
        return data

    def submit(self):
        import fcntl
        import pwd
        lock_path = self.root / 'submit.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker = self.root / 'active.json'
            if marker.exists():
                active = json.loads(marker.read_text())
                if active.get('boot_id') == self.boot:
                    raise RuntimeError('Another job owns this boot: ' + active['job_id'])
            gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=15)
            if gpu.returncode or gpu.stdout.strip():
                raise RuntimeError('GPU is unavailable or another GPU process is running')
            workspace = self.directory / 'workspace'
            workspace.mkdir(exist_ok=True)
            project = Path(self.request['remote_project'])
            for name in ['datasets', 'models']:
                target = project / name
                link = workspace / name
                if link.exists() or link.is_symlink():
                    raise RuntimeError('Unexpected existing data link: ' + name)
                if target.exists():
                    link.symlink_to(target, target_is_directory=True)
            (workspace / 'runs').mkdir(exist_ok=True)
            checkpoint = self.request.get('source_checkpoint_id')
            if checkpoint:
                if not re.fullmatch('[a-f0-9]{32}', checkpoint):
                    raise ValueError('Invalid checkpoint id')
                source = project / 'runs' / checkpoint
                if not source.is_dir():
                    raise ValueError('Source checkpoint does not exist')
                (workspace / 'runs' / checkpoint).symlink_to(source, target_is_directory=True)
            write_json(self.status_path, {'job_id': self.job_id, 'state': 'submitted', 'submitted_at': time.time()})
            # systemd opens StandardOutput as root before applying --uid.
            # Precreate the file as the SSH user so collection stays readable.
            (self.directory / 'log.txt').touch(mode=0o600, exist_ok=True)
            write_json(marker, {'boot_id': self.boot, 'job_id': self.job_id})
            user = pwd.getpwuid(os.getuid()).pw_name
            command = ['sudo', '-n', 'systemd-run', '--unit=' + self.unit, '--service-type=exec',
                       '--uid=' + user, '--working-directory=' + str(workspace),
                       '--property=UMask=0077', '--property=TimeoutStopSec=180',
                       '--property=StandardOutput=append:' + str(self.directory / 'log.txt'),
                       '--property=StandardError=inherit', '--setenv=PYTHONUNBUFFERED=1']
            if self.request.get('auto_stop', True):
                command += ['--property=ExecStopPost=+/usr/bin/systemctl poweroff']
            if self.request.get('max_runtime_seconds'):
                command += ['--property=RuntimeMaxSec=' + str(int(self.request['max_runtime_seconds']))]
            command += [self.request['python'], str(self.directory / 'agent.py'), 'worker', str(self.directory)]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            except BaseException:
                write_json(self.status_path, {'job_id': self.job_id, 'state': 'submission_failed'})
                marker.unlink(missing_ok=True)
                raise
            return {'job_id': self.job_id, 'unit': self.unit, 'accepted': True}

    def worker(self):
        child = None
        stopping = False

        def cancel(signum, frame):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

        previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
        status = {'job_id': self.job_id, 'state': 'running', 'started_at': time.time()}
        write_json(self.status_path, status)
        rc = 1
        try:
            env = dict(os.environ)
            workspace = self.directory / 'workspace'
            env['PYTHONPATH'] = str(workspace / 'src')
            env['KONTEXT_REMOTE_JOB_DIR'] = str(self.directory)
            env['PYTHONUNBUFFERED'] = '1'
            child = subprocess.Popen(self.request['command'], cwd=workspace, env=env)
            rc = child.wait()
            status.update(state='cancelled' if stopping or rc in {130, 143, -15} else ('completed' if rc == 0 else 'failed'),
                          exit_code=rc, finished_at=time.time())
        except BaseException as error:
            traceback.print_exc()
            status.update(state='failed', exit_code=1, error=str(error), finished_at=time.time())
        finally:
            write_json(self.status_path, status)
            sys.stdout.flush()
            sys.stderr.flush()
            if hasattr(os, 'sync'):
                os.sync()
            # Give an online collector time to cache final metrics, without
            # depending on that PC being online to eventually stop the EC2.
            grace = self.request.get('collection_grace_seconds', 90) if self.request.get('auto_stop', True) else 0
            deadline = time.monotonic() + grace
            while not stopping and time.monotonic() < deadline and not (self.directory / 'collected').exists():
                time.sleep(1)
            if not self.request.get('auto_stop', True):
                marker = self.root / 'active.json'
                if marker.exists() and json.loads(marker.read_text()).get('job_id') == self.job_id:
                    marker.unlink()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return rc

    def collect(self, weights=False):
        try:
            options = json.loads(sys.stdin.buffer.read() or b'{}')
        except (ValueError, OSError):
            options = {}
        skip = set(options.get('skip', []))
        files = [(self.directory / name, name) for name in ['status.json', 'request.json', 'log.txt', 'training.json']]
        training = self.directory / 'training.json'
        if training.exists():
            data = json.loads(training.read_text())
            training_id = data['training_id']
            if not re.fullmatch('[a-f0-9]{32}', training_id):
                raise ValueError('Invalid training id')
            output = self.directory / 'workspace' / 'runs' / training_id
            for name in ['job.json', 'config.json', 'metrics.jsonl', 'initial_config.json', 'progress.json', 'decisions.jsonl', 'selected_checkpoint.json', 'review_mode.json']:
                files.append((output / name, 'training/' + name))
            evaluation = output / 'evaluation'
            if evaluation.is_dir():
                for path in evaluation.rglob('*'):
                    if path.is_file() and not path.is_symlink() and path.suffix.lower() in {'.png', '.jpg', '.json'}:
                        name = 'training/evaluation/' + path.relative_to(evaluation).as_posix()
                        if path.suffix.lower() in {'.png', '.jpg'} and name in skip:
                            continue
                        files.append((path, name))
            if weights and (output / 'weights').is_dir():
                selected = output / 'weights'
                selection = output / 'selected_checkpoint.json'
                if selection.is_file():
                    candidate = Path(json.loads(selection.read_text())['checkpoint']).resolve()
                    if output.resolve() not in candidate.parents or not candidate.is_dir():
                        raise ValueError('Invalid selected checkpoint path')
                    selected = candidate
                for path in selected.rglob('*'):
                    if path.is_file() and not path.is_symlink():
                        files.append((path, 'training/weights/' + path.relative_to(selected).as_posix()))
        with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
            for path, name in files:
                if path.is_file():
                    archive.add(path, arcname=name, recursive=False)

    def cancel(self):
        (self.directory / 'cancel').touch()
        subprocess.run(['sudo', '-n', 'systemctl', 'stop', '--no-block', self.unit], check=True, timeout=15)
        return {'job_id': self.job_id, 'cancel_requested': True}

    def acknowledge(self):
        if self.status_path.exists() and json.loads(self.status_path.read_text()).get('state') in {'completed', 'failed', 'cancelled'}:
            (self.directory / 'collected').touch()
        return {'acknowledged': True}

    def control(self):
        values = json.loads(sys.stdin.buffer.read())
        if set(values) - {'learning_rate', 'stop', 'reason', 'preferred_epoch'}:
            raise ValueError('Unknown control field')
        if values.get('learning_rate') is not None and not 1e-8 <= float(values['learning_rate']) <= 1e-3:
            raise ValueError('Learning rate is out of range')
        if values.get('preferred_epoch') is not None and not 1 <= int(values['preferred_epoch']) <= 30:
            raise ValueError('Preferred epoch is out of range')
        write_json(self.directory / 'control.json', values)
        return {'control_saved': True, **values}

    def guard(self):
        # This is a setup timeout, never a maximum training duration.
        if self.state() not in {'active', 'activating', 'reloading', 'deactivating'}:
            gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=15)
            if gpu.returncode or gpu.stdout.strip():
                return
            other = subprocess.run(['systemctl', 'list-units', 'kontext-train-*', '--state=active,activating,deactivating',
                                    '--no-legend', '--no-pager'], capture_output=True, text=True, timeout=15)
            if other.returncode or other.stdout.strip():
                return
            subprocess.run(['/usr/bin/systemctl', 'poweroff'], check=True, timeout=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['submit', 'worker', 'status', 'collect', 'cancel', 'ack', 'guard', 'control'])
    parser.add_argument('directory')
    parser.add_argument('--weights', action='store_true')
    args = parser.parse_args()
    agent = JobAgent(args.directory)
    if args.action == 'worker':
        return agent.worker()
    if args.action == 'collect':
        agent.collect(args.weights)
        return 0
    if args.action == 'guard':
        agent.guard()
        return 0
    result = agent.acknowledge() if args.action == 'ack' else getattr(agent, args.action)()
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
