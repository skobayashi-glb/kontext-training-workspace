"""Read-only viewer for locally cached results. No AWS or ML dependencies."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import urlsplit

from .client import load_json


class ResultViewer:
    def __init__(self, cache):
        self.cache = Path(cache)

    def jobs(self):
        if not self.cache.exists():
            return []
        jobs = []
        for directory in self.cache.iterdir():
            if directory.is_dir() and re.fullmatch('[a-f0-9]{32}', directory.name):
                local = load_json(directory / 'local.json', {})
                status = load_json(directory / 'status.json', {})
                jobs.append({'id': directory.name, 'created_at': local.get('created_at', 0),
                             'state': status.get('state', 'not_cached'), 'exit_code': status.get('exit_code'),
                             'last_sync': (directory / 'status.json').stat().st_mtime if (directory / 'status.json').exists() else None})
        return sorted(jobs, key=lambda j: j['created_at'], reverse=True)

    def details(self, job):
        if not re.fullmatch('[a-f0-9]{32}', job):
            raise ValueError('Invalid job id')
        directory = self.cache / job
        metrics = []
        path = directory / 'training/metrics.jsonl'
        if path.exists():
            for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
                try:
                    metrics.append(json.loads(line))
                except ValueError:
                    continue
        log = directory / 'log.txt'
        return {'status': load_json(directory / 'status.json', {}),
                'config': load_json(directory / 'training/config.json', {}),
                'progress': load_json(directory / 'training/progress.json', {}),
                'evaluation': load_json(directory / 'training/evaluation/timeline.json', {}),
                'boards': [p.relative_to(directory).as_posix() for p in sorted((directory / 'training/evaluation').rglob('*.jpg'))] if (directory / 'training/evaluation').exists() else [],
                'review_state': load_json(directory / 'review/state.json', {}),
                'ai_review': (directory / 'review/ai_review.md').read_text(encoding='utf-8', errors='replace') if (directory / 'review/ai_review.md').exists() else '',
                'metrics': metrics[-10000:],
                'log': log.read_text(encoding='utf-8', errors='replace')[-150000:] if log.exists() else '',
                'files': [str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and not p.name.endswith('.tmp')]}

    def server(self, port):
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urlsplit(self.path).path
                try:
                    if path == '/':
                        content = (Path(__file__).parent / 'viewer.html').read_bytes()
                        mime = 'text/html; charset=utf-8'
                    elif path == '/api/jobs':
                        content = json.dumps(viewer.jobs(), ensure_ascii=False).encode()
                        mime = 'application/json; charset=utf-8'
                    elif path.startswith('/images/'):
                        parts = path.split('/')
                        if len(parts) < 4 or not re.fullmatch('[a-f0-9]{32}', parts[2]):
                            raise ValueError('Invalid image path')
                        base = (viewer.cache / parts[2] / 'training/evaluation').resolve()
                        target = (base / '/'.join(parts[3:])).resolve()
                        if base not in target.parents or target.suffix.lower() not in {'.png', '.jpg'}:
                            raise ValueError('Invalid image path')
                        content = target.read_bytes()
                        mime = 'image/png' if target.suffix.lower() == '.png' else 'image/jpeg'
                    elif path.startswith('/api/jobs/'):
                        content = json.dumps(viewer.details(path.rsplit('/', 1)[-1]), ensure_ascii=False).encode()
                        mime = 'application/json; charset=utf-8'
                    else:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', mime)
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                except (ValueError, OSError):
                    self.send_error(400)

            def log_message(self, format, *args):
                pass

        return ThreadingHTTPServer(('127.0.0.1', port), Handler)

    def serve(self, port=8765):
        with self.server(port) as server:
            print(f'結果・loss・ログ: http://127.0.0.1:{server.server_port}\n終了: Ctrl+C（EC2や学習には影響しません）', flush=True)
            server.serve_forever()
