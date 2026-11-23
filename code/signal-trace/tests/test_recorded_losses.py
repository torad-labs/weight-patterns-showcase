"""Recorded losses (pipeline/research_guard/recorded_losses.py): a path no store holds excuses its own absence, to the declaration and
the archive record its entry names as the bytes it names, and nothing else. Made-up files in a temporary tree only: no bucket, no
file of the study, nothing sealed."""
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.native_records import run_declaration
from pipeline.research_guard import archive_inputs as tool
from pipeline.research_guard import input_archives as archives
from pipeline.research_guard import recorded_losses, storage_archives as storage
from pipeline.research_guard.verify_round2 import ROOT, verify
from tests.guard_fixture import copy_method

STUDY = 'studies/04-round-2-overlap'
RUNS = f'{STUDY}/runs'
LIST = 'records/losses.json'
GONE = f'{RUNS}/source/capture/capture.json'
RECORD = 'model.json'
ORIGINAL = 'a' * 64


def sha(data):
    return hashlib.sha256(data).hexdigest()


def declaration(**extra):
    """A complete step 1-4 declaration; run_folder gives it its one output inside its own folder."""
    d = dict(method_step=[1, 2, 3, 4], why_this_direction='w', repeating_pattern_check='p', hypothesis='h',
             what_went_wrong_before='b', rule='r', prediction='pr', would_weaken_it='ww', inputs=['shared/in.json'])
    d.update(extra)
    return d


class RecordedLosses(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve(); self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / STUDY).mkdir(parents=True)
        self.method = copy_method(self.tmp, read_only_inputs='shared')
        self.put('shared/in.json', b'{}')
        self.put(f'{STUDY}/ROUND-2.md', b'source reader later folder')
        self.run_folder('source', declaration())
        self.run_folder('reader', declaration(inputs=['shared/in.json', GONE]))

    def put(self, rel, data):
        file = self.tmp / rel; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(data)

    def save_method(self):
        (self.tmp / archives.METHOD).write_text(json.dumps(self.method))

    def run_folder(self, name, declared):
        declared = dict(declared)
        declared.setdefault('outputs', [f'{RUNS}/{name}/out.json'])
        self.put(f'{RUNS}/{name}/declaration.json', json.dumps(declared).encode())

    def declared_now(self, *runs):
        """{run: sha256 of its declaration.json as it is on this disk now}: what a list binds a declared loss to."""
        return {run: sha((self.tmp / RUNS / run / 'declaration.json').read_bytes()) for run in runs}

    def lose(self, paths, **changed):
        body = dict(form=recorded_losses.FORM, paths=paths)
        body.update(changed)
        raw = json.dumps(body).encode()
        self.put(LIST, raw)
        self.method['recorded_losses'] = dict(record=LIST, sha256=sha(raw), why='a made-up deletion')
        self.save_method()

    def entry(self, digest='b' * 64, declared_by=None, bound_by=None):
        """An entry known to the reader's declaration as it is now, unless told otherwise."""
        if declared_by is None and bound_by is None:
            declared_by = self.declared_now('reader')
        return dict(sha256=digest, why='a capture receipt that was never committed', declared_by=declared_by or {}, bound_by=bound_by or {})

    # a declared input that is lost

    def test_without_a_list_a_missing_input_is_drift(self):
        self.assertEqual(verify(self.tmp), [f'reader: input {GONE} does not exist'])

    def test_a_listed_loss_excuses_that_input_and_only_it(self):
        other = f'{RUNS}/source/capture/other.json'
        self.run_folder('reader', declaration(inputs=['shared/in.json', GONE, other]))
        self.lose({GONE: self.entry()})
        self.assertEqual(verify(self.tmp), [f'reader: input {other} does not exist'])
        for spelling in (GONE + '/', f'{RUNS}/source/capture/./capture.json', f'{RUNS}/source/capture/x/../capture.json'):
            self.run_folder('reader', declaration(inputs=['shared/in.json', spelling]))
            self.lose({GONE: self.entry()})
            self.assertEqual(verify(self.tmp), [f'reader: input {spelling} does not exist'])       # the listed spelling only

    def test_a_loss_is_known_only_to_the_declaration_the_list_names_as_the_bytes_it_names(self):
        self.lose({GONE: self.entry()})
        self.assertEqual(verify(self.tmp), [])
        said = recorded_losses.NOT_THIS_DECLARATION                                         # and the gate says why the list did not help
        self.run_folder('later', declaration(inputs=['shared/in.json', GONE]))              # a new declaration gets nothing from the list
        self.assertEqual(verify(self.tmp), [f'later: input {GONE} does not exist{said}'])
        shutil.rmtree(self.tmp / RUNS / 'later')
        self.run_folder('reader', declaration(inputs=['shared/in.json', GONE], rule='another rule'))   # nor does a changed one
        self.assertEqual(verify(self.tmp), [f'reader: input {GONE} does not exist{said}'])
        self.run_folder('reader', declaration(inputs=['shared/in.json', GONE]))             # its listed bytes again
        self.assertEqual(verify(self.tmp), [])

    def test_a_loss_excuses_no_target_and_no_parameter_citation(self):
        self.run_folder('reader', declaration(method_step=5, inputs=['shared/in.json'], target_source=GONE,
                                              parameters=dict(step=dict(measured_from=GONE, derivation='d', scales_with='s'))))
        self.lose({GONE: self.entry()})
        found = verify(self.tmp)
        self.assertIn('reader: edit steps need target_source inside a step-3/4 overlap run', found)
        self.assertIn(f'reader: parameter step cites a missing record {GONE}', found)

    def test_where_a_lost_input_comes_from_is_still_checked(self):
        stray = f'{RUNS}/nobody/capture.json'
        self.run_folder('reader', declaration(inputs=['shared/in.json', stray]))
        self.lose({stray: self.entry()})
        self.assertEqual(verify(self.tmp), [f'reader: input {stray} comes from nobody, which is not a run'])

    # the list itself

    def test_the_list_is_the_bytes_the_method_file_hashed(self):
        self.lose({GONE: self.entry()})
        self.put(LIST, json.dumps(dict(form=recorded_losses.FORM, paths={GONE: self.entry(), f'{RUNS}/source/x.json': self.entry()})).encode())
        found = verify(self.tmp)
        self.assertTrue(any('is not the file the method file hashed' in p for p in found), found)
        self.assertIn(f'reader: input {GONE} does not exist', found)

    def test_a_listed_path_that_is_here_is_a_problem(self):
        self.put(GONE, b'{}')
        self.lose({GONE: self.entry()})
        self.assertTrue(any('is on this disk; take it off the list' in p for p in verify(self.tmp)))

    @unittest.skipIf(os.geteuid() == 0, 'root looks into any folder')
    def test_a_path_that_cannot_be_looked_at_is_a_problem_and_not_a_loss(self):
        closed = self.tmp / RUNS / 'source/capture'
        closed.mkdir(parents=True); self.addCleanup(os.chmod, closed, 0o755)
        self.lose({GONE: self.entry()})
        os.chmod(closed, 0o000)
        lost, problems = recorded_losses.load(self.tmp, self.method)
        self.assertEqual(lost, {})
        self.assertTrue(any('cannot be looked at' in p for p in problems), problems)

    def test_entries_are_clean_unsealed_paths_below_the_runs_with_a_reason_and_a_record_that_names_them(self):
        whole = self.entry()
        for paths, word in (({'shared/in2.json': whole}, 'is not a clean path below'),
                            ({f'{RUNS}/source/../x': whole}, 'is not a clean path below'),
                            ({f'{RUNS}/source/reserved-40b/x.json': whole}, 'names sealed material'),
                            ({GONE: {k: v for k, v in whole.items() if k != 'why'}}, 'needs why'),
                            ({GONE: {k: v for k, v in whole.items() if k != 'sha256'}}, 'needs why'),
                            ({GONE: dict(whole, sha256='short')}, 'needs why'),
                            ({GONE: {k: v for k, v in whole.items() if k != 'declared_by'}}, 'needs declared_by and bound_by'),
                            ({GONE: dict(whole, declared_by={}, bound_by={})}, 'needs declared_by and bound_by'),
                            ({GONE: dict(whole, declared_by={'reader': 'short'})}, 'needs declared_by and bound_by'),
                            ({GONE: dict(whole, bound_by=['model.json'])}, 'needs declared_by and bound_by')):
            self.lose(paths)
            found = verify(self.tmp)
            self.assertTrue(any(word in p for p in found), (paths, found))
            self.assertIn(f'reader: input {GONE} does not exist', found)       # a list with a problem stands for nothing

    def test_one_bad_entry_makes_the_whole_list_stand_for_nothing(self):
        second = f'{RUNS}/source/capture/second.json'
        self.run_folder('reader', declaration(inputs=['shared/in.json', GONE, second]))
        self.lose({GONE: self.entry(), second: dict(self.entry(), why='')})
        found = verify(self.tmp)
        self.assertIn(f'reader: input {GONE} does not exist', found)
        self.assertIn(f'reader: input {second} does not exist', found)
        self.assertEqual(recorded_losses.named(self.tmp), {})

    def test_a_path_with_no_recorded_hash_says_null(self):
        self.lose({GONE: self.entry(None)})
        self.assertEqual(verify(self.tmp), [])

    # no run computes on a loss

    def test_no_native_run_reads_a_loss(self):
        self.lose({GONE: self.entry()})
        with patch.object(run_declaration, 'ROOT', self.tmp):
            with self.assertRaisesRegex(ValueError, 'recorded as lost'):
                run_declaration.check_native_run(self.tmp / RUNS / 'reader', self.tmp / RUNS / 'reader/declaration.json')
            source = self.tmp / RUNS / 'source/declaration.json'
            self.assertEqual(run_declaration.check_native_run(source.parent, source), json.loads(source.read_text()))

    def test_no_native_run_reads_a_folder_that_holds_a_loss(self):
        folder = f'{RUNS}/source/capture'
        self.put(f'{folder}/kept.json', b'{}')                                  # the folder is here; one file of it is lost
        self.run_folder('folder', declaration(inputs=['shared/in.json', folder]))
        self.lose({GONE: self.entry()})
        self.assertEqual(verify(self.tmp), [])
        with patch.object(run_declaration, 'ROOT', self.tmp):
            with self.assertRaisesRegex(ValueError, 'recorded as lost'):
                run_declaration.check_native_run(self.tmp / RUNS / 'folder', self.tmp / RUNS / 'folder/declaration.json')

    def test_no_native_run_reads_a_folder_that_holds_a_loss_under_another_spelling_or_a_link(self):
        folder = f'{RUNS}/source/capture'
        self.put(f'{folder}/kept.json', b'{}')
        (self.tmp / RUNS / 'folder').mkdir(parents=True)
        os.symlink(self.tmp / folder, self.tmp / RUNS / 'folder/link')
        for spelling in (f'{folder}/.', f'{RUNS}//source/capture', f'{RUNS}/reader/../source/capture', f'{RUNS}/source/./capture',
                         str(self.tmp / folder), f'{RUNS}/folder/link', f'{RUNS}/folder/link/capture.json', RUNS, STUDY, '.'):
            self.run_folder('folder', declaration(inputs=['shared/in.json', spelling]))
            self.lose({GONE: self.entry()})
            with patch.object(run_declaration, 'ROOT', self.tmp):
                try:
                    run_declaration.check_native_run(self.tmp / RUNS / 'folder', self.tmp / RUNS / 'folder/declaration.json')
                except ValueError as refused:                   # by the verifier or by the loss check: either way it does not run
                    self.assertRegex(str(refused), 'recorded as lost|Round-two drift', spelling)
                else:
                    self.fail(f'a native run was let read {spelling}')
            self.assertEqual(recorded_losses.touching(recorded_losses.named(self.tmp), self.tmp, spelling), [GONE], spelling)

    def test_touching_is_the_path_what_lies_below_it_and_a_listed_folder_above_it(self):
        lost, root = {GONE: {}, f'{RUNS}/source/rows': {}}, self.tmp
        self.assertEqual(recorded_losses.touching(lost, root, GONE), [GONE])
        self.assertEqual(recorded_losses.touching(lost, root, f'{RUNS}/source/capture'), [GONE])
        self.assertEqual(recorded_losses.touching(lost, root, f'{RUNS}/source/capture/'), [GONE])
        self.assertEqual(recorded_losses.touching(lost, root, f'{RUNS}/source'), [GONE, f'{RUNS}/source/rows'])
        self.assertEqual(recorded_losses.touching(lost, root, f'{RUNS}/source/rows/a/x.bin'), [f'{RUNS}/source/rows'])
        self.assertEqual(recorded_losses.touching(lost, root, f'{RUNS}/source/capture/capture.json.sha256'), [])
        self.assertEqual(recorded_losses.touching(lost, root, f'{RUNS}/source/cap'), [])
        self.assertEqual(recorded_losses.touching(lost, root, f'{RUNS}/sourcery'), [])
        self.assertEqual(recorded_losses.touching(lost, root, 'shared/in.json'), [])
        self.assertEqual(recorded_losses.touching({}, root, GONE), [])

    def test_a_target_above_a_loss_stops_a_native_run(self):
        folder = f'{RUNS}/source/capture'
        self.put(f'{folder}/kept.json', b'{}')
        self.run_folder('folder', declaration(method_step=5, target_source=folder))
        self.lose({GONE: self.entry()})
        with patch.object(run_declaration, 'ROOT', self.tmp):
            with self.assertRaisesRegex(ValueError, 'recorded as lost'):
                run_declaration.check_native_run(self.tmp / RUNS / 'folder', self.tmp / RUNS / 'folder/declaration.json')

    def test_a_loss_named_in_any_field_or_shape_stops_a_native_run(self):
        folder = f'{RUNS}/source/capture'
        self.put(f'{folder}/kept.json', b'{}')
        for extra in (dict(resume_checkpoint=folder), dict(reference=dict(folder=f'{folder}/')), dict(target_source=[GONE]),
                      dict(sources=[dict(definition=f'{RUNS}/reader/../source/capture/capture.json')])):
            self.run_folder('folder', declaration(**extra))
            self.lose({GONE: self.entry()})
            self.assertEqual(verify(self.tmp), [], extra)                      # the record gate has nothing to say of these fields
            with patch.object(run_declaration, 'ROOT', self.tmp):
                with self.assertRaisesRegex(ValueError, 'recorded as lost', msg=extra):
                    run_declaration.check_native_run(self.tmp / RUNS / 'folder', self.tmp / RUNS / 'folder/declaration.json')
        lost = recorded_losses.named(self.tmp)
        said = declaration(note='n/a', why_this_direction=f'the receipt {GONE} is gone and/or was never sent', rule='/', hypothesis='')
        self.assertEqual(recorded_losses.named_in(lost, self.tmp, said), [])   # prose that mentions a loss names no path
        self.assertEqual(recorded_losses.named_in(lost, self.tmp, declaration(inputs=['.'])), [f'inputs[0]: .: {GONE}'])
        self.assertEqual(recorded_losses.named_in(lost, self.tmp, declaration(outputs=[folder, GONE])), [])   # written, not read

    def test_a_run_named_without_a_path_stops_a_native_run_when_its_folder_holds_a_loss(self):
        for extra in (dict(matched_work=dict(run='source')), dict(predecessor='source', file='capture/capture.json')):
            self.run_folder('folder', declaration(**extra))
            self.lose({GONE: self.entry()})
            with patch.object(run_declaration, 'ROOT', self.tmp):
                with self.assertRaisesRegex(ValueError, 'recorded as lost', msg=extra):
                    run_declaration.check_native_run(self.tmp / RUNS / 'folder', self.tmp / RUNS / 'folder/declaration.json')
        self.run_folder('source', declaration(name='source', note='reader'))    # its own name, and a run that holds no loss
        self.lose({GONE: self.entry()})
        with patch.object(run_declaration, 'ROOT', self.tmp):
            source = self.tmp / RUNS / 'source/declaration.json'
            self.assertEqual(run_declaration.check_native_run(source.parent, source), json.loads(source.read_text()))

    def test_a_file_where_a_listed_folder_stood_is_seen_as_gone(self):
        self.put(f'{RUNS}/source/capture', b'a file, not the folder')               # the lost file's parent is now a plain file
        self.lose({GONE: self.entry()})
        self.assertEqual(recorded_losses.load(self.tmp, self.method), ({GONE: dict(sha256='b' * 64, declared_by=self.declared_now('reader'), bound_by={})}, []))

    def test_a_broken_list_stops_a_native_run(self):
        self.lose({GONE: dict(self.entry(), why='')})
        with patch.object(run_declaration, 'ROOT', self.tmp):
            with self.assertRaisesRegex(ValueError, 'Round-two drift'):
                run_declaration.check_native_run(self.tmp / RUNS / 'source', self.tmp / RUNS / 'source/declaration.json')

    # a model archive whose `written` source is lost

    def model_record(self, name=RECORD, model=f'{RUNS}/source/model/edited.bin', weights=b'weights', written=GONE, stated_by=None):
        declared = f'{RUNS}/source/declaration.json'
        self.put(model, weights)
        self.put(f'{STUDY}/input-archives/excluded.json', json.dumps(dict(sealed_name_prefixes=['reserved'], no_fit_answers=['nofit-01'])).encode())
        self.run_folder('source', declaration(model_sha256=ORIGINAL))
        self.run_folder('reader', declaration())
        receipt = json.dumps(dict(model_sha256=sha(weights), original=ORIGINAL)).encode()
        files = {'.': dict(bytes=len(weights), sha256=sha(weights))}
        sources = {declared: sha((self.tmp / declared).read_bytes()), written: sha(receipt)}
        record = dict(form=storage.FORM, input=model, kind='file', artifact='model', storage_role='declared-input',
                      bucket=dict(traces_root='studies', location=model[len('studies/'):]), files=files,
                      inventory_sha256=archives.inventory_sha256(files), remote_bytes_verified=None, restore='restore',
                      provenance=dict(answer_rows=storage.NO_ROWS, produced_by=declared, written=written,
                                      original=dict(sha256=ORIGINAL, stated_by=stated_by or declared), sources=sources))
        raw = json.dumps(record).encode()
        self.put(f'{STUDY}/input-archives/{name}', raw)
        return sha(receipt), receipt, raw

    def known_to(self, raw, name=RECORD):
        return {name: sha(raw)}

    def test_a_lost_source_is_accepted_only_with_the_hash_the_record_bound(self):
        bound, receipt, raw = self.model_record()
        self.assertTrue(any('provenance source' in p and 'is not on this disk' in p for p in verify(self.tmp)))
        self.lose({GONE: self.entry('c' * 64, bound_by=self.known_to(raw))})
        self.assertTrue(any('is not on this disk' in p for p in verify(self.tmp)))            # another hash: not the bound file
        self.lose({GONE: self.entry(bound, bound_by=self.known_to(raw))})
        self.assertEqual(verify(self.tmp), [])
        records, problems, _, _ = archives.registry(self.tmp)                                 # the tools see what the verifier sees
        self.assertEqual((problems, [r['name'] for r in records]), ([], [RECORD]))
        self.method.pop('recorded_losses'); self.save_method()
        self.put(GONE, receipt)                                                               # back on disk: checked as ever
        self.assertEqual(verify(self.tmp), [])
        self.put(GONE, b'{"model_sha256": "something else"}')
        self.assertTrue(any('is not the file the record hashed' in p for p in verify(self.tmp)))

    def test_a_lost_source_is_known_only_to_the_record_file_the_list_names_as_the_bytes_it_names(self):
        bound, _, raw = self.model_record()
        self.lose({GONE: self.entry(bound, bound_by=self.known_to(raw))})
        self.assertEqual(verify(self.tmp), [])
        # The same record with another inventory, consistent in itself: before the list bound the record's bytes, nothing held it.
        edited = json.loads(raw)
        edited['files']['.']['sha256'] = 'e' * 64
        edited['inventory_sha256'] = archives.inventory_sha256(edited['files'])
        self.put(f'{STUDY}/input-archives/{RECORD}', json.dumps(edited).encode())
        found = verify(self.tmp)
        self.assertTrue(any(RECORD in p and 'is not on this disk as a plain file' in p for p in found), found)
        self.assertIn(f'input archive {RECORD}: {recorded_losses.NOT_THIS_RECORD}', found)   # and the gate says why the list did not help
        self.assertEqual(archives.registry(self.tmp)[0], [])                                 # and it stands for nothing
        self.put(f'{STUDY}/input-archives/{RECORD}', raw)
        self.assertEqual(verify(self.tmp), [])
        # A second record, of another file, that cites the same lost receipt with its bound hash.
        self.model_record(name='second.json', model=f'{RUNS}/source/model/second.bin', weights=b'other weights')
        self.put(f'{STUDY}/input-archives/{RECORD}', raw)
        found = verify(self.tmp)
        self.assertTrue(any('second.json' in p and 'is not on this disk as a plain file' in p for p in found), found)
        self.assertFalse(any(RECORD in p for p in found), found)
        # A record file of the listed name under another name's bytes, and the listed bytes under another name.
        self.assertEqual(recorded_losses.bound(recorded_losses.named(self.tmp), 'second.json', raw), {})
        self.assertEqual(recorded_losses.bound(recorded_losses.named(self.tmp), RECORD, raw + b' '), {})
        self.assertEqual(recorded_losses.bound(recorded_losses.named(self.tmp), RECORD, raw), {GONE: bound})

    def test_the_source_that_states_the_original_is_never_excused(self):
        bound, _, raw = self.model_record(stated_by=GONE)
        self.lose({GONE: self.entry(bound, bound_by=self.known_to(raw))})
        self.assertTrue(any("does not state the original's sha256" in p for p in verify(self.tmp)))

    def test_a_record_being_made_gets_nothing_from_the_list(self):
        bound, _, raw = self.model_record()
        self.lose({GONE: self.entry(bound, bound_by=self.known_to(raw))})
        records, problems, method, keep = archives.registry(self.tmp)
        names, _ = archives.exclusions(self.tmp)
        made = archives.check_record(self.tmp, RECORD, json.loads(raw), method, keep, names)      # as archive_inputs asks of a new record
        self.assertTrue(any('is not on this disk as a plain file' in p for p in made), made)
        declared, second = f'{RUNS}/source/declaration.json', f'{RUNS}/source/model/second.bin'
        self.put(second, b'other weights')                                                     # the operator's command, with the lost receipt
        name, problems = tool.make_model(self.tmp, second, dict(sources=[declared, GONE], produced_by=declared, written=GONE,
                                                                original=dict(sha256=ORIGINAL, stated_by=declared)))
        self.assertIsNone(name)
        self.assertTrue(any(GONE in p for p in problems), problems)
        self.assertEqual(sorted(p.name for p in (self.tmp / archives.REGISTRY).glob('*.json')), ['excluded.json', RECORD])

    # the repository's own list

    def test_the_repository_list_is_whole(self):
        method = json.loads((ROOT / archives.METHOD).read_text())
        lost, problems = recorded_losses.load(ROOT, method)
        self.assertEqual(problems, [])
        self.assertEqual(bool(lost), 'recorded_losses' in method)

    def test_every_record_the_repository_list_names_is_here_as_the_bytes_it_names(self):
        method = json.loads((ROOT / archives.METHOD).read_text())
        for path, entry in recorded_losses.load(ROOT, method)[0].items():
            for run, digest in entry['declared_by'].items():
                self.assertEqual(sha((ROOT / method['runs_root'] / run / 'declaration.json').read_bytes()), digest, (path, run))
            for name, digest in entry['bound_by'].items():
                self.assertEqual(sha((ROOT / archives.REGISTRY / name).read_bytes()), digest, (path, name))


if __name__ == '__main__':
    unittest.main()
