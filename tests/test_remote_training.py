import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import base64
import threading
import types
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'flux-kontext-training/src'))
from kontext_remote.agent import JobAgent
from kontext_remote.client import Settings, SourceBundle, TrainingLauncher, ResultCache, save_json
from kontext_remote.viewer import ResultViewer
from flux_kontext_training import remote_cli
from flux_kontext_training.job_store import JobStore
from flux_kontext_training.schemas import TrainingConfig, MetricEvent


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = json.loads((ROOT / 'aws_config.json').read_text())
        data['local_project'] = str(ROOT / 'flux-kontext-training')
        data['cache_dir'] = str(self.root / 'cache')
        self.settings_path = self.root / 'settings.json'
        save_json(self.settings_path, data)
        self.settings = Settings(self.settings_path)

    def tearDown(self):
        self.temp.cleanup()

    def agent(self, command=None, auto_stop=False):
        directory = self.root / 'remote/jobs' / ('a' * 32)
        directory.mkdir(parents=True, exist_ok=True)
        request = {'command': command or [sys.executable, '-c', 'pass'], 'python': sys.executable,
                   'remote_project': str(self.root / 'legacy'), 'auto_stop': auto_stop,
                   'collection_grace_seconds': 0}
        (directory / 'workspace').mkdir(exist_ok=True)
        save_json(directory / 'request.json', request)
        return JobAgent(directory)

    def test_dry_run_never_contacts_aws_or_ssh(self):
        launcher = TrainingLauncher(self.settings)
        with patch.object(launcher.aws, 'call', side_effect=AssertionError('Paid/network access')), contextlib.redirect_stdout(io.StringIO()):
            job = launcher.train(ROOT / 'training_config.json', dry_run=True)
        self.assertEqual(len(job), 32)
        self.assertFalse(self.settings.cache.exists())

    def test_previous_final_logs_are_recovered_without_an_extra_start(self):
        launcher = TrainingLauncher(self.settings)
        pending, done, current = 'a' * 32, 'b' * 32, 'c' * 32
        for job in [pending, done, current]:
            save_json(launcher.cache.directory(job) / 'local.json', {'accepted': True})
        save_json(launcher.cache.directory(done) / 'status.json', {'state': 'completed'})
        save_json(launcher.cache.directory(done) / 'collection.json', {'complete': True})
        ssh = Mock()
        with patch.object(launcher.cache, 'fetch') as fetch, \
             patch.object(launcher.aws, 'call', side_effect=AssertionError('Extra AWS access')):
            launcher.recover_pending(ssh, current)
        fetch.assert_called_once_with(ssh, pending)

    def test_final_status_alone_is_not_mistaken_for_complete_log_collection(self):
        launcher = TrainingLauncher(self.settings)
        job = 'a' * 32
        save_json(launcher.cache.directory(job) / 'local.json', {'accepted': True})
        save_json(launcher.cache.directory(job) / 'status.json', {'state': 'completed'})
        with patch.object(launcher.cache, 'fetch') as fetch:
            launcher.recover_pending(Mock(), 'b' * 32)
        self.assertEqual(fetch.call_count, 1)

    def test_bundle_excludes_data_secrets_environments_and_outputs(self):
        project = self.root / 'project'
        for name in ['src/flux_kontext_training/remote_cli.py', 'main.py', '.env',
                     'datasets/private.json', 'models/model.json', 'runs/private.json',
                     '.venv/lib/private.py', '.git/config.json', 'ui/private.json', 'data/private.py']:
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('x')
        self.settings.project = project
        bundle = SourceBundle(self.settings).build({'command': ['python']})
        with tarfile.open(fileobj=io.BytesIO(bundle), mode='r:gz') as archive:
            names = archive.getnames()
        self.assertEqual(set(names), {'workspace/main.py', 'workspace/src/flux_kontext_training/remote_cli.py', 'agent.py', 'request.json'})

    def test_invalid_training_config_fails_before_paid_operations(self):
        config = self.root / 'bad.json'
        launcher = TrainingLauncher(self.settings)
        with patch.object(launcher.aws, 'call', side_effect=AssertionError('Network')):
            for data in [{'dataset': 'test', 'epochs': 0}, {'dataset': 'test', 'epoch': 1}]:
                save_json(config, data)
                with self.assertRaises(ValueError):
                    launcher.train(config)

    def test_ambiguous_start_timeout_still_requests_stop(self):
        launcher = TrainingLauncher(self.settings)
        launcher.aws = Mock()
        launcher.aws.instance.return_value = {'State': {'Name': 'stopped'}}
        launcher.aws.start.side_effect = TimeoutError('start response lost')
        with patch.object(launcher, 'background'):
            with self.assertRaises(TimeoutError):
                launcher.train(ROOT / 'training_config.json')
        launcher.aws.stop.assert_called_once()

    def test_ambiguous_submit_never_force_stops_training(self):
        launcher = TrainingLauncher(self.settings)
        launcher.aws = Mock()
        launcher.aws.instance.return_value = {'State': {'Name': 'stopped'}}
        ssh = Mock()
        ssh.agent.side_effect = TimeoutError('SSH disconnected after submission')
        with patch.object(launcher, 'background'), patch.object(launcher, 'connect', return_value=ssh), \
             patch.object(launcher, 'deploy'), patch.object(launcher, 'guard') as guard:
            with self.assertRaises(TimeoutError):
                launcher.train(ROOT / 'training_config.json')
            guard.assert_called_once()
        launcher.aws.stop.assert_not_called()

    def test_setup_watch_never_stops_an_accepted_or_ambiguous_job(self):
        launcher = TrainingLauncher(self.settings)
        launcher.aws = Mock()
        for flag in ['accepted', 'submission_attempted']:
            path = launcher.cache.directory('a' * 32) / 'local.json'
            save_json(path, {flag: True, 'setup_deadline': 0, 'start_attempted': True})
            launcher.setup_watch('a' * 32)
        launcher.aws.stop.assert_not_called()

    def test_collection_and_review_survive_expired_aws_credentials(self):
        launcher = TrainingLauncher(self.settings)
        job = 'a' * 32
        save_json(launcher.cache.directory(job) / 'local.json', {'accepted': True, 'public_ip': '192.0.2.1'})
        with patch.object(launcher.aws, 'instance', side_effect=AssertionError('Expired AWS must not be needed')), \
             patch.object(launcher.cache, 'fetch', return_value={'state': 'completed'}), \
             patch.object(launcher, 'maybe_review') as review:
            launcher.watch(job)
        review.assert_called_once_with(job)

    def test_expired_setup_prevents_submission(self):
        launcher = TrainingLauncher(self.settings)
        launcher.aws = Mock()
        launcher.aws.instance.return_value = {'State': {'Name': 'stopped'}}
        ssh = Mock()
        def expire(*args):
            latest = json.loads((self.settings.cache / 'latest.json').read_text())['job_id']
            path = launcher.cache.directory(latest) / 'local.json'
            state = json.loads(path.read_text())
            state['setup_expired'] = True
            save_json(path, state)
        with patch.object(launcher, 'background'), patch.object(launcher, 'connect', return_value=ssh), \
             patch.object(launcher, 'deploy', side_effect=expire), patch.object(launcher, 'guard'):
            with self.assertRaises(TimeoutError):
                launcher.train(ROOT / 'training_config.json')
        ssh.agent.assert_not_called()
        launcher.aws.stop.assert_called_once()

    def test_remote_bootstrap_is_valid_python_and_binary_payload(self):
        launcher = TrainingLauncher(self.settings)
        ssh = Mock()
        launcher.deploy(ssh, 'a' * 32, b'archive')
        command, payload = ssh.run.call_args.args
        remote_args = shlex.split(command)
        compile(remote_args[2], '<bootstrap>', 'exec')
        self.assertEqual(payload, b'archive')
        self.assertIn("'\\\\' in member.name", remote_args[2])

    def test_worker_preserves_success_and_failure_exit_codes(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            for code in [0, 7]:
                agent = self.agent([sys.executable, '-c', f'raise SystemExit({code})'])
                self.assertEqual(agent.worker(), code)
                state = json.loads(agent.status_path.read_text())
                self.assertEqual(state['exit_code'], code)
                self.assertEqual(state['state'], 'completed' if code == 0 else 'failed')
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def test_submit_precreates_readable_log_before_systemd_opens_it(self):
        agent = self.agent(auto_stop=True)
        fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        pwd = types.SimpleNamespace(getpwuid=lambda uid: types.SimpleNamespace(pw_name='ec2-user'))

        def run(command, **kwargs):
            if command[0] == 'sudo':
                self.assertTrue((agent.directory / 'log.txt').exists())
                self.assertIn('--property=ExecStopPost=+/usr/bin/systemctl poweroff', command)
                self.assertIn('--uid=ec2-user', command)
            return subprocess.CompletedProcess(command, 0, '', '')

        with patch.dict(sys.modules, {'fcntl': fcntl, 'pwd': pwd}), patch('os.getuid', return_value=1000, create=True), \
             patch('kontext_remote.agent.subprocess.run', side_effect=run):
            self.assertTrue(agent.submit()['accepted'])

    def test_guard_does_not_stop_a_running_training_unit(self):
        agent = self.agent()
        with patch.object(agent, 'state', return_value='active'), patch('kontext_remote.agent.subprocess.run') as run:
            agent.guard()
        run.assert_not_called()

    def test_guard_does_not_stop_another_gpu_process(self):
        agent = self.agent()
        with patch.object(agent, 'state', return_value='inactive'), \
             patch('kontext_remote.agent.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '1234\n', '')) as run:
            agent.guard()
        self.assertEqual(run.call_count, 1)

    def test_guard_stops_failed_setup_when_no_jobs_exist(self):
        agent = self.agent()
        with patch.object(agent, 'state', return_value='inactive'), \
             patch('kontext_remote.agent.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')) as run:
            agent.guard()
        self.assertEqual(run.call_args.args[0], ['/usr/bin/systemctl', 'poweroff'])

    def test_image_collection_skips_already_cached_immutable_images(self):
        agent = self.agent()
        training_id = 'b' * 32
        save_json(agent.directory / 'training.json', {'training_id': training_id})
        output = agent.directory / 'workspace/runs' / training_id / 'evaluation'
        output.mkdir(parents=True)
        (output / 'epoch_0002.jpg').write_bytes(b'immutable-image-fixture')
        save_json(output / 'timeline.json', {'updated': True})
        buffer = io.BytesIO()
        stdin = types.SimpleNamespace(buffer=io.BytesIO(json.dumps({'skip': ['training/evaluation/epoch_0002.jpg']}).encode()))
        with patch('sys.stdin', stdin), patch('sys.stdout', types.SimpleNamespace(buffer=buffer)):
            agent.collect()
        with tarfile.open(fileobj=io.BytesIO(buffer.getvalue()), mode='r:gz') as archive:
            names = archive.getnames()
        self.assertNotIn('training/evaluation/epoch_0002.jpg', names)
        self.assertIn('training/evaluation/timeline.json', names)

    def test_result_archive_cannot_escape_cache(self):
        def collect(job, path, weights):
            with tarfile.open(path, 'w:gz') as archive:
                content = b'unsafe'
                info = tarfile.TarInfo('../../escaped.txt')
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        ssh = Mock()
        ssh.collect.side_effect = collect
        with self.assertRaises(ValueError):
            ResultCache(self.settings).fetch(ssh, 'a' * 32)
        self.assertFalse((self.settings.cache / 'escaped.txt').exists())
        self.assertFalse((self.root / 'escaped.txt').exists())

    def test_final_collection_acknowledges_only_after_files_are_saved(self):
        def collect(job, path, weights):
            with tarfile.open(path, 'w:gz') as archive:
                for name, content in [('status.json', b'{"state":"completed","exit_code":0}'), ('log.txt', '完了\n'.encode())]:
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
        ssh = Mock()
        ssh.collect.side_effect = collect
        def acknowledge(job, action):
            self.assertEqual((self.settings.cache / job / 'log.txt').read_text(encoding='utf-8'), '完了\n')
            self.assertTrue(json.loads((self.settings.cache / job / 'collection.json').read_text())['complete'])
            self.assertEqual(action, 'ack')
        ssh.agent.side_effect = acknowledge
        state = ResultCache(self.settings).fetch(ssh, 'a' * 32)
        self.assertEqual(state['state'], 'completed')
        ssh.agent.assert_called_once()

    def test_new_store_never_rewrites_historical_running_jobs(self):
        runs = self.root / 'runs'
        old = runs / ('b' * 32)
        old.mkdir(parents=True)
        content = '{"status":"running","do_not_touch":true}'
        (old / 'job.json').write_text(content)
        store = JobStore(runs_dir=runs, load_existing=False)
        store.create(TrainingConfig(dataset='test'))
        self.assertEqual((old / 'job.json').read_text(), content)

    def test_training_adapter_records_metrics_and_success_without_cuda(self):
        self.exercise_training('success', 0, 'completed')

    def test_training_adapter_records_error_without_cuda(self):
        self.exercise_training('failure', None, 'failed')

    def test_training_adapter_records_cancellation_without_cuda(self):
        self.exercise_training('cancel', 130, 'cancelled')

    def exercise_training(self, outcome, expected_code, expected_state):
        job_dir = self.root / 'job'
        job_dir.mkdir()
        runs = self.root / 'runs'
        class FakeTrainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def run(self):
                self.kwargs['on_metric'](MetricEvent(step=1, train_loss=0.5, val_loss=0.4))
                if outcome == 'failure':
                    raise RuntimeError('simulated CUDA failure')
                if outcome == 'cancel':
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        command = remote_cli.TrainingCommand(TrainingConfig(dataset='test'), trainer_factory=FakeTrainer)
        with patch.object(command, 'validate'), patch.object(remote_cli, 'RUNS_DIR', runs), \
             patch.dict(os.environ, {'KONTEXT_REMOTE_JOB_DIR': str(job_dir)}), contextlib.redirect_stdout(io.StringIO()):
            if outcome == 'failure':
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    command.run()
            else:
                self.assertEqual(command.run(), expected_code)
        job = json.loads((job_dir / 'training.json').read_text())['training_id']
        data = json.loads((runs / job / 'job.json').read_text())
        self.assertEqual(data['status'], expected_state)
        self.assertIn('0.5', (runs / job / 'metrics.jsonl').read_text())

    def test_offline_viewer_serves_logs_metrics_and_escapes_no_html(self):
        job = 'a' * 32
        directory = self.settings.cache / job
        save_json(directory / 'local.json', {'created_at': 1})
        save_json(directory / 'status.json', {'state': 'completed'})
        (directory / 'log.txt').write_text('<script>private-log</script>', encoding='utf-8')
        image = directory / 'training/evaluation/base.jpg'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'example-image-bytes')
        viewer = ResultViewer(self.settings.cache)
        with viewer.server(0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = 'http://127.0.0.1:' + str(server.server_port)
                self.assertEqual(json.load(urlopen(base + '/api/jobs'))[0]['state'], 'completed')
                self.assertEqual(json.load(urlopen(base + '/api/jobs/' + job))['log'], '<script>private-log</script>')
                self.assertIn('textContent', urlopen(base).read().decode())
                self.assertEqual(urlopen(base + '/images/' + job + '/base.jpg').read(), b'example-image-bytes')
            finally:
                server.shutdown()
                thread.join()


if __name__ == '__main__':
    unittest.main()
