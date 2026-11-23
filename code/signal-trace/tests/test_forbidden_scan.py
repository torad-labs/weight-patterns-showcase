"""The installed byte prefilter preserves the verifier's original file/text scan."""
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.research_guard import verify_round2 as guard


def original_scan(root, method):
    """The pre-integration text interpretation and traversal, as a fixture oracle."""
    problems = []
    pattern = re.compile('|'.join(re.escape(s) for s in method['forbidden_references']))
    for place in (root / 'code/signal-trace/pipeline', root / method['runs_root']):
        if not place.exists():
            continue
        for path in place.rglob('*'):
            if not path.is_file() or path.suffix not in {'.py', '.json', '.md'} or path.resolve() in guard.SELF:
                continue
            if path.stat().st_size > 2_000_000:
                continue
            hit = pattern.search(path.read_text(errors='ignore'))
            if hit:
                problems.append(f'{path.relative_to(root)}: references deleted first-attempt work ({hit.group()})')
    return problems


class InstalledForbiddenScanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = self.root / 'code/signal-trace/pipeline'
        self.runs = self.root / 'runs/example'
        self.pipeline.mkdir(parents=True)
        self.runs.mkdir(parents=True)

    def test_real_files_keep_decoding_matches_and_refusal_order(self):
        fixtures = {
            self.pipeline / 'clean.py': b'ordinary text',
            self.pipeline / 'unicode.md': 'ordinary café'.encode(),
            self.pipeline / 'hit.json': b'blocked-name and blocked-other',
            self.runs / 'invalid.json': b'blocked-\xffname',
            self.runs / 'truncated.md': b'blocked-\xe2\x82name',
            self.runs / 'lines.json': b'two\r\nlines\rfinal',
            self.runs / 'punctuation.py': '[!] café'.encode(),
        }
        for path, data in fixtures.items():
            path.write_bytes(data)
        for references in (['blocked-name', 'blocked-other'], ['blocked', 'blocked-name'],
                           ['two\nlines'], ['café'], ['[!]'], [], ['']):
            with self.subTest(references=references):
                method = dict(runs_root='runs', forbidden_references=references)
                found = []
                guard.check_forbidden(self.root, method, found)
                self.assertEqual(found, original_scan(self.root, method))

    def test_roots_suffix_size_boundary_and_self_exclusion_stay_the_same(self):
        needle = b'blocked-name'
        allowed = self.runs / 'at-boundary.json'
        allowed.write_bytes(b' ' * (2_000_000 - len(needle)) + needle)
        (self.runs / 'too-large.json').write_bytes(b' ' * (2_000_001 - len(needle)) + needle)
        (self.runs / 'ignored.bin').write_bytes(needle)
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'ignored.json').write_bytes(needle)
        own = self.pipeline / 'self.py'
        own.write_bytes(needle)
        method = dict(runs_root='runs', forbidden_references=['blocked-name'])
        with patch.object(guard, 'SELF', guard.SELF | {own.resolve()}):
            found = []
            guard.check_forbidden(self.root, method, found)
            self.assertEqual(found, original_scan(self.root, method))
            self.assertEqual(found, ['runs/example/at-boundary.json: references deleted first-attempt work (blocked-name)'])


if __name__ == '__main__':
    unittest.main()
