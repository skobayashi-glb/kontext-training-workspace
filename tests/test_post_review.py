import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kontext_remote.client import Settings, save_json
from kontext_remote.post_review import PostTrainingReview


class PostReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        settings = json.loads((ROOT / 'aws_config.json').read_text())
        settings['cache_dir'] = str(root / 'results')
        save_json(root / 'settings.json', settings)
        self.settings = Settings(root / 'settings.json')
        self.review = PostTrainingReview(self.settings, 'a' * 32)
        save_json(self.review.directory / 'status.json', {'state': 'completed'})

    def tearDown(self):
        self.temp.cleanup()

    def test_review_is_read_only_and_has_no_training_tools(self):
        self.review.output.mkdir()
        command = self.review.command('codex', Path('prompt'), Path('report'), [Path('comparison.jpg')])
        self.assertIn('read-only', command)
        self.assertIn('shell_tool', command)
        self.assertIn('unified_exec', command)
        self.assertIn('--ignore-user-config', command)
        self.assertIn('web_search="disabled"', command)
        self.assertNotIn('aws', command)
        self.assertEqual(command[-2:], ['--', '-'])
        self.assertIn('ユーザーが決め', self.review.prompt({}, []))

    def test_review_runs_once_after_completion_and_saves_report(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            report = Path(command[command.index('-o') + 1])
            report.write_text('画像とlossを確認。次の学習はユーザーが決定。', encoding='utf-8')
            return subprocess.CompletedProcess(command, 0, b'', b'')
        with patch('kontext_remote.post_review.shutil.which', return_value='codex'), \
             patch('kontext_remote.post_review.subprocess.run', side_effect=run):
            first = self.review.run()
            second = self.review.run()
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads((self.review.output / 'state.json').read_text())['state'], 'completed')

    def test_missing_codex_keeps_numeric_summary_without_starting_anything(self):
        with patch('kontext_remote.post_review.shutil.which', return_value=None), \
             patch('kontext_remote.post_review.subprocess.run') as run:
            result = self.review.run()
        run.assert_not_called()
        self.assertEqual(result.name, 'summary.json')
        self.assertEqual(json.loads((self.review.output / 'state.json').read_text())['state'], 'requires_codex')

    def test_notes_refresh_preserves_findings_and_replaces_only_generated_section(self):
        notes = self.settings.path.parent / 'TRAINING_NOTES.md'
        notes.write_text('# 調査\n確定事項は保持\n\n<!-- AUTO_REVIEW_START -->\nold\n<!-- AUTO_REVIEW_END -->\n', encoding='utf-8')
        summary = {'status': {'state': 'completed'}}
        self.review.update_notes(summary)
        self.review.update_notes(summary)
        result = notes.read_text(encoding='utf-8')
        self.assertIn('確定事項は保持', result)
        self.assertEqual(result.count('<!-- AUTO_REVIEW_START -->'), 1)
        self.assertIn(self.review.job, result)
        self.assertNotIn('\nold\n', result)

    def test_review_includes_train_output_and_warns_against_unverified_overfitting_claims(self):
        base = self.review.directory / 'training/evaluation'
        base.mkdir(parents=True)
        (base / 'train_final_selected.jpg').touch()
        self.assertIn(base / 'train_final_selected.jpg', self.review.images())
        self.assertIn('過学習の有無を断定しない', self.review.prompt({}, []))


if __name__ == '__main__':
    unittest.main()
