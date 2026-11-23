"""A verified bucket copy standing for a declared input (Astra's contract, records/retrospectives/2026-11-13-declared-input-archives-astra.md,
and its four boundary repairs of November 13): what the verifier accepts and refuses, what `make` refuses before it lists or reads
anything, what the bucket is asked for, and what a native run needs back on this disk. Files in a temporary tree and a stand-in for the
bucket tool only: no bucket, no file of the study."""
import builtins
import importlib.util
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from pipeline.native_records import run_declaration
from pipeline.research_guard import archive_inputs as tool
from pipeline.research_guard import input_archives as archives
from pipeline.research_guard.verify_round2 import ROOT, verify
from tests.guard_fixture import copy_method

STUDY = 'studies/04-round-2-overlap'
RUNS = f'{STUDY}/runs'
DATA = f'{RUNS}/overlap/data'
ANSWERS = f'{STUDY}/inputs/answers.json'

BUCKET_TOOL = '''#!/usr/bin/env python3
import os, shutil, sys
from pathlib import Path
store, bucket = Path(os.environ['TORAD_RESEARCH_TRACES']), Path(os.environ['STAND_IN_BUCKET'])
_, kind, verb, *paths = sys.argv
with open(os.environ['STAND_IN_CALLS'], 'a') as calls:
    calls.write(' '.join([verb] + paths) + '\\n')
failed = 0
for rel in paths:
    here, there = store / rel, bucket / rel
    if verb == 'upload':
        there.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(here, there)
        print(f'sent      {rel}')
    elif verb == 'verify':
        right = there.is_file() and there.read_bytes() == here.read_bytes()
        print(f'verified  {rel}  {here.stat().st_size} B' if right else f'FAILED    {rel}: the bucket holds other bytes')
        failed += not right
    elif verb == 'download':                      # a folder brings everything the bucket holds below it, as the real tool does
        for source in [there] if there.is_file() else sorted(p for p in there.rglob('*') if p.is_file()):
            target = store / source.relative_to(bucket)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        print(f'downloaded {rel}')
sys.exit(1 if failed else 0)
'''


def declaration(**extra):
    d = dict(method_step=[1, 2, 3, 4], why_this_direction='w', repeating_pattern_check='p', hypothesis='h', what_went_wrong_before='b',
             rule='r', prediction='pr', would_weaken_it='ww', inputs=[f'{STUDY}/inputs/in.json'], outputs=[f'{RUNS}/overlap/OVERLAP.md'])
    d.update(extra)
    return d


@contextmanager
def untouched(*places):
    """Fail when anything at or below a place is listed, walked or opened (looking at a path's own name and links is not that)."""
    real = [os.path.realpath(p) for p in places]

    def guarded(call):
        def checked(path, *rest, **more):
            if isinstance(path, (str, bytes, os.PathLike)):
                at = os.path.realpath(os.fsdecode(path))
                if any(at == place or at.startswith(place + os.sep) for place in real):
                    raise AssertionError(f'{call.__name__} of {at}: listed or read before the refusal')
            return call(path, *rest, **more)
        return checked

    with ExitStack() as stack:
        for owner, name in ((os, 'walk'), (os, 'scandir'), (os, 'listdir'), (os, 'open'), (io, 'open'), (builtins, 'open')):
            stack.enter_context(patch.object(owner, name, guarded(getattr(owner, name))))
        yield


class InputArchives(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve(); self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / STUDY / 'inputs').mkdir(parents=True)
        copy_method(self.tmp)
        (self.tmp / STUDY / 'inputs/in.json').write_text('{}')
        (self.tmp / ANSWERS).write_text(json.dumps(dict(answers=['fit-01', 'fit-02'])))      # the unsealed index a membership cites
        (self.tmp / STUDY / 'ROUND-2.md').write_text('overlap edit other')
        self.registry = self.tmp / archives.REGISTRY; self.registry.mkdir()
        (self.registry / archives.EXCLUDED).write_text(json.dumps(dict(sealed_name_prefixes=['reserved'], no_fit_answers=['nofit-01'])))
        self.declare('overlap', declaration())
        self.data = self.tmp / DATA; (self.data / 'sub').mkdir(parents=True)
        (self.data / 'a.bin').write_bytes(b'aaaa'); (self.data / 'sub/b.bin').write_bytes(b'bbbbbb'); (self.data / 'index.json').write_text('{}')
        self.declare('edit', declaration(method_step=5, inputs=[DATA], target_source=f'{DATA}/index.json', outputs=[f'{RUNS}/edit/x']))
        self.members = dict(sources=[ANSWERS], answers=['fit-01', 'fit-02'], pooled={'.': ['fit-01', 'fit-02']})   # the whole input pools the two

    def declare(self, name, d):
        folder = self.tmp / RUNS / name; folder.mkdir(parents=True, exist_ok=True)
        (folder / 'declaration.json').write_text(json.dumps(d))
        return folder

    def archive(self, item=DATA, verified=True, members=None):
        name, problems = tool.make(self.tmp, item, members or self.members)
        self.assertEqual(problems, [])
        if verified:
            self.assertEqual(self.verify_passed(name), [])
        return name

    def verify_passed(self, name, leave_out=(), failed=False):
        record = self.record(name)
        lines = [f'verified  {archives.below(record["bucket"]["location"], sub)}  4 B' for sub in record['files'] if sub not in leave_out]
        lines += ['FAILED  x: sizes differ'] if failed else []
        return tool.attach(self.tmp, name, '\n'.join(lines + [f'{len(lines)} verified']), 'torad-research traces verify', '2026-11-13 21:00:00 CST')

    def record(self, name):
        return json.loads((self.registry / name).read_text())

    def rewrite(self, name, **changes):
        record = self.record(name); record.update(changes)
        (self.registry / name).write_text(json.dumps(record))

    def problems(self, text):
        found = verify(self.tmp)
        self.assertTrue(any(text in p for p in found), f'{text!r} not among {found}')

    def native(self, run='edit'):
        with patch.object(run_declaration, 'ROOT', self.tmp):
            return run_declaration.check_native_run(self.tmp / RUNS / run, self.tmp / RUNS / run / 'declaration.json')

    def fetch_right(self, root, record, missing):
        right = {'a.bin': b'aaaa', 'sub/b.bin': b'bbbbbb', 'index.json': b'{}'}
        for sub in missing:
            file = Path(root) / record['input'] / sub
            file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(right[sub])

    def test_without_a_record_every_check_is_what_it_was(self):
        self.assertEqual(verify(self.tmp), [])
        shutil.rmtree(self.data)
        self.problems(f'edit: input {DATA} does not exist')
        self.problems('edit steps need target_source')

    def test_a_verified_record_stands_for_released_files_and_a_folder_still_here_is_checked(self):
        self.archive()
        (self.data / 'a.bin').unlink(); (self.data / 'sub/b.bin').unlink()      # the folder and its metadata stay
        self.assertEqual(verify(self.tmp), [])
        shutil.rmtree(self.data)                                                # the target is in the inventory too
        self.assertEqual(verify(self.tmp), [])

    def test_an_unverified_record_stands_for_nothing_missing(self):
        self.archive(verified=False)
        self.assertEqual(verify(self.tmp), [])
        (self.data / 'a.bin').unlink()
        self.problems('carries no remote verification')

    def test_a_bare_true_or_another_inventorys_verification_is_refused(self):
        name = self.archive()
        self.rewrite(name, remote_bytes_verified=True)
        self.problems('its remote verification is not of this inventory')
        name2 = self.record(name)
        self.rewrite(name, remote_bytes_verified=dict(passed_ct='t', command='c', files=3, bytes=12, inventory_sha256='0' * 64))
        self.problems('its remote verification is not of this inventory')
        files = dict(name2['files'], **{'late.bin': dict(bytes=1, sha256='0' * 64)})
        self.rewrite(name, files=files, remote_bytes_verified=None)
        self.problems('inventory_sha256 is not the hash of its files')

    def test_a_local_file_that_is_not_its_inventorys_is_reported(self):
        self.archive()
        (self.data / 'a.bin').write_bytes(b'longer than before')
        self.problems('has 18 bytes here and 4 in its inventory')

    def test_a_missing_child_is_covered_only_by_its_own_entry(self):
        (self.data / 'sub/b.bin').unlink()
        self.archive()                                                          # the inventory is made without sub/b.bin
        (self.data / 'sub/b.bin').write_bytes(b'bbbbbb')
        self.declare('edit', declaration(method_step=5, inputs=[f'{DATA}/sub/b.bin', f'{DATA}/a.bin'], target_source=f'{DATA}/index.json', outputs=[f'{RUNS}/edit/x']))
        self.assertEqual(verify(self.tmp), [])
        (self.data / 'a.bin').unlink()                                          # in the inventory: covered
        self.assertEqual(verify(self.tmp), [])
        (self.data / 'sub/b.bin').unlink()                                      # not in it: missing
        self.problems(f'edit: input {DATA}/sub/b.bin does not exist')

    def test_a_declared_folder_above_an_archived_one_must_itself_be_here(self):
        self.archive(f'{DATA}/sub')
        (self.data / 'sub/b.bin').unlink()
        self.assertEqual(verify(self.tmp), [])
        self.declare('edit', declaration(method_step=5, inputs=[DATA], target_source=f'{RUNS}/overlap/declaration.json', outputs=[f'{RUNS}/edit/x']))
        shutil.rmtree(self.data)
        self.problems('is not on this disk and no archive record holds the whole of it')

    def test_paths_that_escape_links_and_other_locations_are_refused(self):
        name = self.archive()
        good = self.record(name)
        for change, text in ((dict(input=f'{RUNS}/overlap/../overlap/data'), 'names no clean repository-relative input'),
                             (dict(input='/etc'), 'names no clean repository-relative input'),
                             (dict(input='code/elsewhere'), 'outside the round-two runs and the read-only inputs'),
                             (dict(bucket=dict(traces_root='studies', location='04-round-2-overlap/runs/other/data')), 'its bucket location is not the input'),
                             (dict(files={'../../declaration.json': good['files']['a.bin']}), 'is not a clean path below the input'),
                             (dict(files={'/etc/passwd': good['files']['a.bin']}), 'is not a clean path below the input'),
                             (dict(files={'a.bin': dict(bytes=4, sha256='short')}), 'needs its bytes and its sha256'),
                             (dict(form='declared-input-archive/1'), f'is not a {archives.FORM} record')):
            self.rewrite(name, **dict(good, **change))
            self.problems(text)
        os.symlink(self.tmp / RUNS / 'overlap', self.tmp / RUNS / 'link')
        self.rewrite(name, **dict(good, input=f'{RUNS}/link/data', bucket=dict(traces_root='studies', location='04-round-2-overlap/runs/link/data')))
        self.problems('is not the path it resolves to')

    def test_what_may_not_be_an_input_is_refused_before_anything_is_listed_or_read(self):
        outside = self.tmp / 'elsewhere'; (outside / 'reserved-40b').mkdir(parents=True); (outside / 'reserved-40b/s.bin').write_bytes(b's')
        (self.tmp / 'code/elsewhere').mkdir(parents=True); (self.tmp / 'code/elsewhere/c.bin').write_bytes(b'c')
        sealed = self.tmp / RUNS / 'overlap/reserved-40b'; sealed.mkdir(); (sealed / 'case.json').write_text('{}')
        os.symlink(outside, self.tmp / RUNS / 'overlap/linked')                  # a link on the way that leaves the repository
        os.symlink(self.tmp / RUNS / 'overlap', self.tmp / RUNS / 'alias')       # a link on the way that stays inside it
        os.symlink(sealed, self.data / 'quiet')                                  # a link whose target is sealed
        held = self.archive(f'{DATA}/sub', verified=False)
        refused = ((f'{RUNS}/overlap/reserved-40b', 'names sealed or no-fit material'),
                   (f'{RUNS}/overlap/reserved-40b/case.json', 'names sealed or no-fit material'),
                   (f'{RUNS}/overlap/linked', 'is not the path it resolves to'),
                   (f'{RUNS}/overlap/linked/reserved-40b', 'names sealed or no-fit material'),
                   (f'{RUNS}/alias/data', 'is not the path it resolves to'),
                   (f'{DATA}/quiet', 'is not the path it resolves to'),
                   (f'{DATA}/nofit-01.own', 'names sealed or no-fit material'),
                   ('code/elsewhere', 'outside the round-two runs and the read-only inputs'),
                   (f'{RUNS}/overlap/declaration.json', 'stay on this disk'),
                   (archives.REGISTRY, 'stay on this disk'),
                   (f'{RUNS}/overlap/../overlap/data', 'not a clean repository-relative path'),
                   (DATA, f'{held} already holds {DATA}/sub'))
        with untouched(self.data, sealed, outside, self.tmp / 'code/elsewhere'):
            for item, why in refused:
                name, problems = tool.make(self.tmp, item, self.members)
                self.assertIsNone(name, item)
                self.assertTrue(any(why in p for p in problems), f'{item}: {why!r} not among {problems}')
                self.assertTrue(all('nothing was listed or read' in p for p in problems), problems)
        self.assertEqual(sorted(p.name for p in self.registry.glob('*.json')), sorted([archives.EXCLUDED, held]))
        (self.registry / held).unlink()
        with self.assertRaisesRegex(AssertionError, 'listed or read before the refusal'):   # the watch does see a read: an allowed input is walked
            with untouched(self.data):
                tool.make(self.tmp, DATA, self.members)

    def test_sealed_and_no_fit_names_are_never_archived(self):
        (self.data / 'reserved-40b').mkdir(); (self.data / 'reserved-40b/x.bin').write_bytes(b'x')
        (self.data / 'nofit-01.own').mkdir(); (self.data / 'nofit-01.own/y.npy').write_bytes(b'y')
        (self.data / 'nofit-01').mkdir(); (self.data / 'nofit-01/z.npy').write_bytes(b'z')
        (self.data / 'nofit-01-output-norms.npy').write_bytes(b'n')             # the id inside a longer name
        (self.data / 'sub/000-fit-01--nofit-01.json').write_text('{}')          # a pair that names it
        (self.data / 'old-reserved-set').mkdir(); (self.data / 'old-reserved-set/r.bin').write_bytes(b'r')
        (self.data / 'preserved.bin').write_bytes(b'kept')                      # 'reserved' inside another word is not the sealed name
        with untouched(self.data / 'reserved-40b', self.data / 'nofit-01.own', self.data / 'nofit-01', self.data / 'old-reserved-set',
                       self.data / 'nofit-01-output-norms.npy', self.data / 'sub/000-fit-01--nofit-01.json'):
            name = self.archive()
        self.assertEqual(sorted(self.record(name)['files']), ['a.bin', 'index.json', 'preserved.bin', 'sub/b.bin'])
        self.assertEqual(self.record(name)['left_on_this_disk'], {'a sealed or no-fit name': 6})
        good = self.record(name)
        for sub in ('reserved-40b/x.bin', 'nofit-01.own/y.npy', 'nofit-01/z.npy', 'nofit-01-output-norms.npy', 'sub/000-fit-01--nofit-01.json',
                    'old-reserved-set/r.bin'):
            self.rewrite(name, files=dict(good['files'], **{sub: dict(bytes=1, sha256='0' * 64)}))
            self.problems('names sealed or no-fit material')
        self.rewrite(name, **good)
        (self.registry / archives.EXCLUDED).unlink()
        self.problems('no archive record is accepted without it')

    def test_a_file_of_mixed_or_unresolved_membership_stays_on_this_disk(self):
        (self.data / 'pool.f32').write_bytes(b'pppp')                           # pools answers no index states
        (self.data / 'fit-01-output-norms.npy').write_bytes(b'n')               # names its answer
        members = dict(self.members, pooled={'sub': ['fit-01', 'fit-02']})
        with untouched(self.data / 'pool.f32', self.data / 'a.bin', self.data / 'index.json'):
            name = self.archive(members=members, verified=False)                # what its membership does not resolve is not even read
        good = self.record(name)
        self.assertEqual(sorted(good['files']), ['fit-01-output-norms.npy', 'sub/b.bin'])
        self.assertEqual(good['left_on_this_disk'], {'membership mixed or unresolved': 3})
        self.assertEqual(good['membership']['sources'], {ANSWERS: archives.digest(self.tmp / ANSWERS)})
        self.rewrite(name, files=dict(good['files'], **{'pool.f32': dict(bytes=4, sha256='0' * 64)}))
        self.problems('hold answers its membership does not resolve')
        for membership, text in ((dict(good['membership'], answers=['fit-01', 'fit-02', 'nofit-01']), 'answer(s) no fit uses'),
                                 (dict(good['membership'], pooled={'sub': ['fit-01', 'someone-else']}), 'pooled entry'),
                                 (dict(good['membership'], sources={}), 'its membership needs sources'),
                                 (dict(good['membership'], sources={f'{RUNS}/overlap/reserved-40b/cases.json': '0' * 64}), 'unsealed repository path'),
                                 (dict(good['membership'], sources={ANSWERS: '0' * 64}), 'is not the file the record hashed')):
            self.rewrite(name, **dict(good, membership=membership))
            self.problems(text)
        self.rewrite(name, **{k: v for k, v in good.items() if k != 'membership'}, membership=None)
        self.problems('its membership needs sources')
        self.rewrite(name, **good)
        self.assertEqual(verify(self.tmp), [])
        (self.tmp / ANSWERS).write_text('{"answers": ["rewritten"]}')            # the index changed after the record cited it
        self.problems('is not the file the record hashed')
        for stated, text in ((dict(self.members, answers=['fit-01', 'nofit-01']), 'answer(s) no fit uses'),
                             (dict(answers=['fit-01'], pooled={}), 'a membership needs its sources'),
                             (dict(self.members, sources=[f'{STUDY}/inputs/missing.json']), 'unsealed repository file')):
            made, problems = tool.make(self.tmp, f'{RUNS}/edit', stated)
            self.assertIsNone(made); self.assertTrue(any(text in p for p in problems), problems)

    def test_declarations_results_parameter_records_and_membership_sources_stay_on_this_disk(self):
        (self.tmp / RUNS / 'overlap/measured.json').write_text('{}')
        (self.tmp / RUNS / 'overlap/index-of-answers.json').write_text('{}')
        self.declare('edit', declaration(method_step=5, inputs=[DATA], target_source=f'{DATA}/index.json', outputs=[f'{RUNS}/edit/x'],
                                     parameters=dict(step=dict(measured_from=f'{RUNS}/overlap/measured.json', derivation='d', scales_with='s'))))
        members = dict(self.members, sources=[ANSWERS, f'{RUNS}/overlap/index-of-answers.json'])
        name = self.archive(f'{RUNS}/overlap', members=members)
        good = self.record(name)
        self.assertEqual(sorted(good['files']), ['data/a.bin', 'data/index.json', 'data/sub/b.bin'])
        self.assertEqual(good['left_on_this_disk'], {'a record that stays on this disk': 3})
        for sub, text in (('declaration.json', 'stays on this disk'), ('measured.json', 'stays on this disk'), ('index-of-answers.json', 'a membership source')):
            self.rewrite(name, files=dict(good['files'], **{sub: dict(bytes=2, sha256='0' * 64)}))
            self.problems(text)

    def test_where_an_input_comes_from_is_checked_for_an_archived_path_too(self):
        name = self.archive()
        good = self.record(name)
        ghost = f'{RUNS}/ghost/data'
        self.rewrite(name, **dict(good, input=ghost, bucket=dict(traces_root='studies', location='04-round-2-overlap/runs/ghost/data')))
        self.declare('edit', declaration(method_step=5, inputs=[ghost], target_source=f'{ghost}/index.json', outputs=[f'{RUNS}/edit/x']))
        self.problems(f'edit: input {ghost} comes from ghost, which is not a run')
        self.problems('edit steps need target_source inside a step-3/4 overlap run')

    def test_two_records_over_one_path_are_refused(self):
        name = self.archive(f'{DATA}/sub')
        self.assertEqual(tool.make(self.tmp, DATA, self.members)[1], [f'{name} already holds {DATA}/sub (nothing was listed or read)'])
        whole = dict(self.record(name), input=DATA, bucket=dict(traces_root='studies', location='04-round-2-overlap/runs/overlap/data'),
                     files={'sub/b.bin': self.record(name)['files']['b.bin']}, remote_bytes_verified=None)
        whole['inventory_sha256'] = archives.inventory_sha256(whole['files'])
        (self.registry / 'second.json').write_text(json.dumps(whole))
        self.problems('two records over one path')

    def test_a_verification_binds_only_when_it_names_every_file_and_the_files_stood_still(self):
        name = self.archive(verified=False)
        self.assertIn('does not say verified for 1 inventoried file', self.verify_passed(name, leave_out=('sub/b.bin',))[0])
        self.assertIn('reports a failure', self.verify_passed(name, failed=True)[0])
        os.utime(self.data / 'a.bin', ns=(1, 1))
        self.assertIn('are not the ones its inventory hashed', self.verify_passed(name)[0])
        self.assertIsNone(self.record(name)['remote_bytes_verified'])
        self.assertEqual(archives.restore(self.tmp, name), [])
        self.assertEqual(self.verify_passed(name), [])
        self.assertEqual(self.record(name)['remote_bytes_verified']['inventory_sha256'], self.record(name)['inventory_sha256'])

    def test_the_bucket_is_asked_for_the_inventorys_files_and_for_nothing_else(self):
        bucket, calls, shelf = self.tmp / 'bucket', self.tmp / 'calls.txt', self.tmp / 'bin'
        shelf.mkdir(); (shelf / tool.TOOL).write_text(BUCKET_TOOL); (shelf / tool.TOOL).chmod(0o755)
        (self.data / 'loose.f32').write_bytes(b'llll')                          # unresolved: stays here
        (self.data / 'nofit-01.own').mkdir(); (self.data / 'nofit-01.own/y.npy').write_bytes(b'y')
        (self.data / 'fit-02-output-norms.npy').write_bytes(b'n')
        members = dict(self.members, pooled={'sub': ['fit-01'], 'a.bin': ['fit-02'], 'index.json': ['fit-01', 'fit-02']})
        inventory = ['a.bin', 'fit-02-output-norms.npy', 'index.json', 'sub/b.bin']
        place = '04-round-2-overlap/runs/overlap/data'
        with patch.dict(os.environ, dict(PATH=f'{shelf}{os.pathsep}{os.environ["PATH"]}', STAND_IN_BUCKET=str(bucket), STAND_IN_CALLS=str(calls))):
            name = self.archive(members=members, verified=False)
            record = dict(self.record(name), name=name)
            self.assertEqual(sorted(record['files']), inventory)
            self.assertEqual(tool.bucket(self.tmp, 'upload', record, inventory)[1], 0)
            self.assertEqual(sorted(p.relative_to(bucket / place).as_posix() for p in (bucket / place).rglob('*') if p.is_file()), inventory)
            for stranger, content in (('loose.f32', b'llll'), ('nofit-01.own/y.npy', b'y'), ('reserved-40b/s.bin', b's'), ('someone-elses.bin', b'e')):
                (bucket / place / stranger).parent.mkdir(parents=True, exist_ok=True); (bucket / place / stranger).write_bytes(content)
            self.assertEqual(tool.verify(self.tmp, name), [])
            bound = self.record(name)['remote_bytes_verified']
            self.assertEqual((bound['files'], bound['inventory_sha256']), (4, record['inventory_sha256']))
            self.assertTrue((self.registry / tool.VERIFICATIONS / (name[:-5] + '.txt')).read_text().count('verified  '), 4)
            for gone in ('a.bin', 'sub/b.bin', 'loose.f32', 'nofit-01.own/y.npy'):
                (self.data / gone).unlink()
            self.assertEqual(archives.restore(self.tmp, name, tool.download), [])
            self.assertEqual(((self.data / 'a.bin').read_bytes(), (self.data / 'sub/b.bin').read_bytes()), (b'aaaa', b'bbbbbb'))
            for stranger in ('loose.f32', 'nofit-01.own/y.npy', 'reserved-40b', 'someone-elses.bin'):
                self.assertFalse((self.data / stranger).exists(), stranger)       # in the bucket, below the same folder, and never pulled
            with self.assertRaisesRegex(ValueError, 'is not in the inventory'):
                tool.bucket(self.tmp, 'download', record, ['loose.f32'])
            (bucket / place / 'a.bin').write_bytes(b'AAAA')                       # the bucket's bytes are not the local file's
            self.assertTrue(any('verify call(s) failed' in p for p in tool.verify(self.tmp, name)))
        asked = [line.split() for line in calls.read_text().splitlines()]
        self.assertEqual({path for _, *paths in asked for path in paths}, {f'{place}/{sub}' for sub in inventory})
        self.assertEqual(sorted(paths for verb, *paths in asked if verb == 'download'), [[f'{place}/a.bin'], [f'{place}/sub/b.bin']])   # one file a call
        os.link(self.data / 'a.bin', self.data / 'second-name.bin')               # two names of one file never go in one call
        twice = dict(record, files=dict(record['files'], **{'second-name.bin': record['files']['a.bin']}))
        space = tool.room([tool.TOOL, 'traces', 'upload'], dict(os.environ))
        self.assertEqual(list(tool.calls(self.tmp, twice, ['a.bin', 'second-name.bin', 'index.json'], space)), [['a.bin'], ['second-name.bin', 'index.json']])
        self.assertEqual(list(tool.calls(self.tmp, twice, inventory, space)), [inventory])   # what fits in one call's arguments goes in one call
        self.assertEqual(list(tool.calls(self.tmp, twice, inventory, space, alone=True)), [[sub] for sub in inventory])
        cost = lambda sub: len(f'{place}/{sub}') + 1 + tool.POINTER                          # a call takes the files its arguments have room for
        self.assertEqual(list(tool.calls(self.tmp, twice, inventory, cost('a.bin') + cost('fit-02-output-norms.npy'))),
                         [['a.bin', 'fit-02-output-norms.npy'], ['index.json', 'sub/b.bin']])
        self.assertLess(space, os.sysconf('SC_ARG_MAX') - sum(len(k) + len(v) + 2 for k, v in os.environ.items()))
        with self.assertRaisesRegex(ValueError, 'does not fit in one argument'):
            list(tool.calls(self.tmp, twice, ['a.bin'], 8))
        self.assertEqual(tool.streams(), len(os.sched_getaffinity(0)))            # as many streams of calls as this process has processor threads
        sized = dict(record, files={sub: dict(bytes=size) for sub, size in (('big', 10), ('s1', 3), ('s2', 3), ('s3', 3), ('s4', 2))})
        self.assertEqual(tool.shares(sized, ['big', 's1', 's2', 's3', 's4'], 2), [['big'], ['s1', 's2', 's3', 's4']])   # each file to the share with the fewest bytes so far, in order
        self.assertEqual(tool.shares(sized, ['big', 's1'], 6), [['big'], ['s1']])
        self.assertEqual((tool.shares(sized, [], 6), tool.bucket(self.tmp, 'upload', record, [])), ([], ('', 0)))

    def test_release_removes_only_the_files_of_a_bound_inventory_that_are_still_the_ones_it_hashed(self):
        bucket, calls, shelf = self.tmp / 'bucket', self.tmp / 'calls.txt', self.tmp / 'bin'
        shelf.mkdir(); (shelf / tool.TOOL).write_text(BUCKET_TOOL); (shelf / tool.TOOL).chmod(0o755)
        (self.data / 'loose.f32').write_bytes(b'llll')                          # unresolved: no inventory holds it
        members = dict(self.members, pooled={'sub': ['fit-01'], 'a.bin': ['fit-02'], 'index.json': ['fit-01', 'fit-02']})
        here = lambda: sorted(p.relative_to(self.data).as_posix() for p in self.data.rglob('*') if p.is_file())
        refused = lambda text: (self.assertIn(text, tool.release(self.tmp, name)[1][0]), self.assertEqual(here(), whole))
        with patch.dict(os.environ, dict(PATH=f'{shelf}{os.pathsep}{os.environ["PATH"]}', STAND_IN_BUCKET=str(bucket), STAND_IN_CALLS=str(calls))):
            name = self.archive(members=members, verified=False)
            whole = here()
            self.assertEqual(whole, ['a.bin', 'index.json', 'loose.f32', 'sub/b.bin'])
            refused('no read-back')                                             # unverified
            record = dict(self.record(name), name=name)
            self.assertEqual(tool.bucket(self.tmp, 'upload', record, sorted(record['files']))[1], 0)
            self.assertEqual(self.verify_passed(name), [])
            refused('does not keep the verification')                           # bound, and what the bucket tool said is not on record
            self.rewrite(name, remote_bytes_verified=None)
            self.assertEqual(tool.verify(self.tmp, name), [])
            a = self.data / 'a.bin'
            was = os.lstat(a)
            twin = self.data.parent / 'twin.bin'; twin.write_bytes(b'AAAA')
            os.utime(twin, ns=(was.st_atime_ns, was.st_mtime_ns)); os.replace(twin, a)   # another file under an inventoried name
            refused('are not the ones its inventory hashed')
            a.write_bytes(b'aaaa')
            self.assertEqual(archives.restore(self.tmp, name), [])
            (self.tmp / ANSWERS).write_text(json.dumps(dict(answers=['fit-01', 'fit-02', 'fit-03'])))   # the index the membership rests on moved
            refused('is not the file the record hashed')
            (self.tmp / ANSWERS).write_text(json.dumps(dict(answers=['fit-01', 'fit-02'])))
            done, problems = tool.release(self.tmp, name)
            self.assertEqual((problems, done['files_released'], done['bytes_released'], done['already_gone'], done['still_here']), ([], 3, 12, 0, 0))
            self.assertEqual(here(), ['loose.f32'])                             # the inventory's files and nothing else
            self.assertTrue((self.data / 'sub').is_dir() and (self.tmp / ANSWERS).is_file())
            self.assertFalse((self.registry / archives.RECEIPTS / name).exists())
            self.assertEqual(verify(self.tmp), [])                              # the record stands for them
            with self.assertRaisesRegex(ValueError, 'in the bucket, not on this disk'):
                self.native()
            again, problems = tool.release(self.tmp, name)                      # nothing is left to take
            self.assertEqual((problems, again['files_released'], again['already_gone']), ([], 0, 3))
            self.assertEqual(archives.restore(self.tmp, name, tool.download), [])
            self.assertEqual(here(), whole)
            self.assertEqual(self.native()['method_step'], 5)
            (self.data / 'sub/b.bin').unlink()                                  # one file gone by another hand: the rest still go
            done, problems = tool.release(self.tmp, name)
            self.assertEqual((problems, done['files_released'], done['already_gone']), ([], 2, 1))
        kept = json.loads((self.registry / tool.RELEASES / name).read_text())
        self.assertEqual([r['files_released'] for r in kept['releases']], [3, 2])
        self.assertEqual((kept['inventory_sha256'], kept['input']), (record['inventory_sha256'], DATA))

    @unittest.skipIf(os.geteuid() == 0, 'directory permission refusal requires a non-root user')
    def test_release_records_completed_removals_when_a_later_file_cannot_be_removed(self):
        bucket, calls, shelf = self.tmp / 'bucket', self.tmp / 'calls.txt', self.tmp / 'bin'
        shelf.mkdir(); (shelf / tool.TOOL).write_text(BUCKET_TOOL); (shelf / tool.TOOL).chmod(0o755)
        with patch.dict(os.environ, dict(PATH=f'{shelf}{os.pathsep}{os.environ["PATH"]}', STAND_IN_BUCKET=str(bucket), STAND_IN_CALLS=str(calls))):
            name = self.archive(verified=False)
            record = dict(self.record(name), name=name)
            self.assertEqual(tool.bucket(self.tmp, 'upload', record, sorted(record['files']))[1], 0)
            self.assertEqual(tool.verify(self.tmp, name), [])
            folder = self.data / 'sub'
            mode = folder.stat().st_mode
            folder.chmod(0o555)                                      # files can still be read and identified, but not removed
            try:
                done, problems = tool.release(self.tmp, name)
            finally:
                folder.chmod(mode)
            self.assertEqual((done['files_released'], done['bytes_released'], done['still_here']), (2, 6, 1))
            self.assertEqual(len(problems), 1)
            self.assertIn('sub/b.bin', problems[0])
            self.assertIn('PermissionError', problems[0])
            self.assertFalse((self.data / 'a.bin').exists() or (self.data / 'index.json').exists())
            self.assertEqual((folder / 'b.bin').read_bytes(), b'bbbbbb')
            kept = json.loads((self.registry / tool.RELEASES / name).read_text())
            self.assertEqual(kept['releases'], [done])
            self.assertEqual(done['problems'], problems)
            self.assertTrue((self.registry / archives.RECEIPTS / name).is_file())
            again, problems = tool.release(self.tmp, name)
            self.assertEqual((problems, again['files_released'], again['already_gone'], again['still_here']), ([], 1, 2, 0))
            kept = json.loads((self.registry / tool.RELEASES / name).read_text())
            self.assertEqual([entry['files_released'] for entry in kept['releases']], [2, 1])
            self.assertFalse((self.registry / archives.RECEIPTS / name).exists())

    def test_a_native_run_needs_its_own_archived_inputs_back_and_held_to_their_inventory(self):
        self.declare('other', declaration(outputs=[f'{RUNS}/other/x']))
        name = self.archive()
        self.assertEqual(self.native()['method_step'], 5)                       # here and verified when the record was made
        (self.data / 'a.bin').unlink(); (self.data / 'index.json').unlink()
        self.assertEqual(verify(self.tmp), [])                                  # the record stands for them
        with self.assertRaisesRegex(ValueError, 'in the bucket, not on this disk'):
            self.native()
        self.assertEqual(self.native('other')['method_step'], [1, 2, 3, 4])     # a run that does not read them is not held
        wrong = lambda root, record, missing: (Path(root) / record['input'] / 'a.bin').write_bytes(b'AAAA')
        self.assertTrue(any('another sha256' in p for p in archives.restore(self.tmp, name, wrong)))
        with self.assertRaisesRegex(ValueError, 'refusing to run'):
            self.native()
        (self.data / 'a.bin').unlink()
        self.assertEqual(archives.restore(self.tmp, name, self.fetch_right), [])
        self.assertEqual(self.native()['method_step'], 5)
        os.utime(self.data / 'sub/b.bin', ns=(5, 5))                            # touched since it was verified
        with self.assertRaisesRegex(ValueError, 'not verified against its inventory since it came back'):
            self.native()
        self.assertEqual(archives.restore(self.tmp, name), [])
        self.assertEqual(self.native()['method_step'], 5)

    def test_a_replacement_of_the_same_size_and_time_ends_the_receipt_and_no_start_hashes_an_input(self):
        name = self.archive()
        never = lambda *a, **k: self.fail('a native start hashed a file of an input')
        with patch.object(archives, 'read_held', never):
            self.assertEqual(self.native()['method_step'], 5)                   # ready by the receipt alone
        a = self.data / 'a.bin'
        was = os.lstat(a)
        twin = self.data.parent / 'twin.bin'; twin.write_bytes(b'AAAA')
        os.utime(twin, ns=(was.st_atime_ns, was.st_mtime_ns)); os.replace(twin, a)   # another file: same name, size and modification time
        self.assertEqual((os.lstat(a).st_size, os.lstat(a).st_mtime_ns), (was.st_size, was.st_mtime_ns))
        with patch.object(archives, 'read_held', never), self.assertRaisesRegex(ValueError, 'not verified against its inventory since it came back'):
            self.native()
        self.assertEqual(verify(self.tmp), [])                                  # the record check looks at sizes: it is restore that hashes
        self.assertTrue(any('another sha256' in p for p in archives.restore(self.tmp, name)))
        a.write_bytes(b'aaaa')
        self.assertEqual(archives.restore(self.tmp, name), [])
        self.assertEqual(self.native()['method_step'], 5)
        was = os.lstat(a)
        time.sleep(0.02)
        with open(a, 'r+b') as same:                                            # the same inode rewritten in place, its modification time put back
            same.write(b'aaaA')
        os.utime(a, ns=(was.st_atime_ns, was.st_mtime_ns))
        self.assertEqual((os.lstat(a).st_ino, os.lstat(a).st_size, os.lstat(a).st_mtime_ns), (was.st_ino, was.st_size, was.st_mtime_ns))
        with patch.object(archives, 'read_held', never), self.assertRaisesRegex(ValueError, 'not verified against its inventory since it came back'):
            self.native()
        receipt = json.loads((self.registry / archives.RECEIPTS / name).read_text())['files']
        self.assertEqual(sorted(receipt['a.bin']), sorted(archives.IDENTITY))

    def test_restore_overwrites_nothing(self):
        name = self.archive()
        (self.data / 'a.bin').write_bytes(b'mine, not the archive\'s')
        (self.data / 'sub/b.bin').unlink()
        problems = archives.restore(self.tmp, name, self.fetch_right)
        self.assertTrue(any('a.bin has 23 bytes here and 4 in its inventory; it is left as it is' in p for p in problems))
        self.assertEqual((self.data / 'a.bin').read_bytes(), b'mine, not the archive\'s')
        self.assertFalse((self.registry / archives.RECEIPTS / name).exists())    # a failed restore is not a ready input
        with self.assertRaisesRegex(ValueError, 'refusing to run'):
            self.native()

    def test_what_travels_to_a_rented_box_with_the_preflight_names_nothing_the_sender_refuses(self):
        spec = importlib.util.spec_from_file_location('solve_files', ROOT / 'records/development/2026-11-13-rented-solve/solve_files.py')
        sender = importlib.util.module_from_spec(spec); spec.loader.exec_module(sender)
        found, _ = sender.closure(ROOT / 'code/signal-trace', ['pipeline.native_records.run_declaration'])
        self.assertIn('pipeline.research_guard.input_archives', found)           # the checks travel with the preflight
        self.assertNotIn('pipeline.research_guard.archive_inputs', found)        # the operator's file, which names the bucket tool, does not
        for module in ('pipeline.research_guard.input_archives', 'pipeline.research_guard.verify_round2', 'pipeline.native_records.run_declaration'):
            text = found[module].read_text()
            self.assertEqual([what for what, pattern in sender.PRIVATE if pattern.search(text)], [], module)

    def test_the_repository_registry_is_whole_and_its_no_fit_list_is_the_members_rule(self):
        self.assertEqual(archives.registry(ROOT)[1], [])
        listed, problem = archives.exclusions(ROOT)
        self.assertIsNone(problem)
        members = json.loads((ROOT / RUNS / 'wide-overlap/members.json').read_text())
        fitting = {m for role in ('correct', 'failure') for m in [x['id'] for x in members if x['role'] == role][0::2]}
        self.assertEqual(listed['no_fit'], {x['id'] for x in members} - fitting)
        self.assertEqual((len(fitting), len(listed['no_fit'])), (30, 30))
        for sealed in ('reserved-40b', 'reserved-40-original', 'inputs/acquisition/reserved-40b/cases.json'):
            self.assertIsNotNone(archives.excluded_part(sealed, listed), sealed)
        one = sorted(listed['no_fit'])[0]
        for named in (one, f'{one}.own', f'members/{one}-output-norms.npy', f'pairs/000-block-base--{one}.json'):
            self.assertIsNotNone(archives.excluded_part(named, listed), named)
        for free in (sorted(fitting)[0], f'{one}0', 'preserved/capture-reservations.json', 'group-observation/gpu0-failure/vector-sums.f32'):
            self.assertIsNone(archives.excluded_part(free, listed), free)


if __name__ == '__main__':
    unittest.main()
