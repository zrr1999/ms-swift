import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('ci_source_copy.sh')


class SourceCopyTest(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / 'source with spaces'
        self.source.mkdir()
        (self.source / 'input.txt').write_text('original')
        (self.source / '.git').mkdir()
        (self.source / '.git/config').write_text('original git configuration')
        (self.source / 'linked.txt').symlink_to('input.txt')

    def run_copy(self, *command, source=None):
        return subprocess.run(['bash', str(SCRIPT), str(source or self.source), *command],
                              capture_output=True,
                              text=True,
                              check=False)

    def test_private_copy_preserves_inputs_and_command_arguments(self):
        result = self.run_copy(
            sys.executable, '-c', '''
import json
import sys
from pathlib import Path

cwd = Path.cwd()
assert (cwd / 'input.txt').read_text() == 'original'
assert (cwd / 'linked.txt').is_symlink()
assert (cwd / 'linked.txt').read_text() == 'original'
(cwd / 'input.txt').write_text('modified')
(cwd / '.git/config').write_text('modified git configuration')
(cwd / 'build-output').write_text('output')
print(json.dumps({'cwd': str(cwd), 'args': sys.argv[1:]}))
''', 'argument with spaces', '*')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['args'], ['argument with spaces', '*'])
        self.assertNotEqual(Path(report['cwd']), self.source)
        self.assertFalse(Path(report['cwd']).exists())
        self.assertEqual((self.source / 'input.txt').read_text(), 'original')
        self.assertEqual((self.source / '.git/config').read_text(), 'original git configuration')
        self.assertFalse((self.source / 'build-output').exists())

    def test_failure_is_preserved_and_private_copy_is_removed(self):
        result = self.run_copy(
            sys.executable, '-c', '''
import sys
from pathlib import Path

print(Path.cwd())
Path('build-output').write_text('partial output')
sys.exit(23)
''')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertFalse(Path(result.stdout.strip()).exists())
        self.assertFalse((self.source / 'build-output').exists())

    def test_copy_failure_prevents_command_execution(self):
        marker = Path(self.directory.name) / 'command-ran'
        result = self.run_copy(
            sys.executable,
            '-c',
            'from pathlib import Path; import sys; Path(sys.argv[1]).touch()',
            str(marker),
            source=self.source / 'missing')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())

    def test_missing_command_is_rejected(self):
        result = self.run_copy()
        self.assertEqual(result.returncode, 2)
        self.assertIn('Usage:', result.stderr)


if __name__ == '__main__':
    unittest.main()
