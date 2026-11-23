"""The model archive and the storage-only backup (Astra's contract, records/retrospectives/2026-11-14-storage-only-archives-astra.md):
what a form-3 record must be, checked on the record alone. Made-up files in a temporary tree only: no bucket, no file of the study,
nothing sealed and no answer of the study's no-fit list."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline.research_guard import input_archives as archives
from pipeline.research_guard import storage_archives as storage
from tests.guard_fixture import copy_method

STUDY = 'studies/04-round-2-overlap'
RUNS = f'{STUDY}/runs'
POOL = f'{RUNS}/pool/rows'
MODEL = f'{RUNS}/edit/model/edited.gguf'
INDEX = f'{RUNS}/pool/index.json'
ORIGINAL = 'a' * 64


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Form3(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve(); self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / STUDY / 'inputs').mkdir(parents=True)
        copy_method(self.tmp)
        self.registry = self.tmp / archives.REGISTRY; self.registry.mkdir()
        (self.registry / archives.EXCLUDED).write_text(json.dumps(dict(sealed_name_prefixes=['reserved'], no_fit_answers=['nofit-01', 'nofit-02'])))
        (self.registry / storage.NAMED).write_text(json.dumps(dict(sources=[f'{RUNS}/pool'], decided_in='a made-up decision')))
        self.put(INDEX, json.dumps(dict(answers=['fit-01', 'nofit-01'])).encode())
        self.rows = {'x-0.f32': b'rows of both', 'nofit-01/own.bin': b'one answer'}
        for sub, data in self.rows.items():
            self.put(f'{POOL}/{sub}', data)
        self.weights = b'weights'
        self.put(MODEL, self.weights)
        self.put(f'{RUNS}/edit/declaration.json', json.dumps(dict(model_sha256=ORIGINAL)).encode())
        self.put(f'{RUNS}/edit/result.json', b'{}')
        self.put(f'{RUNS}/edit/edit.json', json.dumps(dict(model_sha256=sha(self.weights))).encode())
        self.method = json.loads((self.tmp / archives.METHOD).read_text())
        self.names, problem = archives.exclusions(self.tmp)
        self.assertIsNone(problem)
        self.keep = {archives.METHOD, archives.REGISTRY, f'{RUNS}/edit/declaration.json', f'{RUNS}/edit/result.json'}

    def put(self, rel, data):
        file = self.tmp / rel; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(data)

    def bound(self, *sources):
        return {source: sha((self.tmp / source).read_bytes()) for source in sources}

    def inventory(self, item, files):
        made = {sub: dict(bytes=len(data), sha256=sha(data)) for sub, data in files.items()}
        return dict(form=storage.FORM, input=item, bucket=dict(traces_root='studies', location=item[len('studies/'):]), files=made,
                    inventory_sha256=archives.inventory_sha256(made), remote_bytes_verified=None, restore='restore')

    def backup(self, **changed):
        record = dict(self.inventory(POOL, self.rows), kind='folder', artifact='answer-rows', storage_role='storage-only',
                      eligibility=storage.NO_GRANT,
                      membership=dict(sources=self.bound(INDEX), answers=['fit-01', 'nofit-01'], pooled={'.': ['fit-01', 'nofit-01']}, cohort='mixed'))
        record.update(changed)
        return record

    def model(self, **changed):
        declared, result, edit = f'{RUNS}/edit/declaration.json', f'{RUNS}/edit/result.json', f'{RUNS}/edit/edit.json'
        record = dict(self.inventory(MODEL, {'.': self.weights}), kind='file', artifact='model', storage_role='declared-input',
                      provenance=dict(answer_rows=storage.NO_ROWS, sources=self.bound(declared, result, edit), produced_by=declared,
                                      written=edit, original=dict(sha256=ORIGINAL, stated_by=declared)))
        record.update(changed)
        return record

    def problems(self, record):
        return storage.check_record(archives, self.tmp, 'r.json', record, self.method, self.keep, self.names)

    def refused(self, record, *words):
        found = self.problems(record)
        self.assertTrue(found, 'the record was accepted')
        for word in words:
            self.assertTrue(any(word in problem for problem in found), f'{word!r} not in {found}')

    # the storage-only backup

    def test_a_mixed_pool_with_a_no_fit_folder_is_a_backup(self):
        self.assertEqual(self.problems(self.backup()), [])

    def test_a_backup_is_never_excused_a_lost_membership_source(self):
        record = self.backup()
        known = {INDEX: record['membership']['sources'][INDEX]}                 # a recorded loss, with the very hash the record bound
        check = lambda: storage.check_record(archives, self.tmp, 'r.json', record, self.method, self.keep, self.names, known)
        self.assertEqual(check(), [])                                           # the source is here: the record is whole
        (self.tmp / INDEX).unlink()
        self.assertTrue(check())                                                # gone: refused, listed or not (only a model's is excused)

    def test_the_same_files_are_no_form_2_record(self):
        record = dict(self.backup(), form=archives.FORM)
        found = archives.check_record(self.tmp, 'r.json', record, self.method, self.keep, self.names)
        self.assertTrue(any('no fit uses' in problem for problem in found), found)

    def test_a_backup_says_it_grants_nothing(self):
        self.refused(self.backup(eligibility='inputs of the next fit'), 'eligibility')
        record = self.backup(); del record['eligibility']
        self.refused(record, 'eligibility')

    def test_the_cohort_is_what_the_exclusion_list_makes_it(self):
        record = self.backup(); record['membership']['cohort'] = 'fitting'
        self.refused(record, "makes it 'mixed'")
        del record['membership']['cohort']
        self.refused(record, "makes it 'mixed'")

    def test_answers_no_exclusion_names_are_an_ordinary_archive(self):
        self.put(f'{POOL}/x-0.f32', self.rows['x-0.f32'])
        record = dict(self.inventory(POOL, {'x-0.f32': self.rows['x-0.f32']}), kind='folder', artifact='answer-rows', storage_role='storage-only',
                      eligibility=storage.NO_GRANT, membership=dict(sources=self.bound(INDEX), answers=['fit-01'], pooled={'.': ['fit-01']}, cohort='fitting'))
        self.refused(record, 'ordinary archive')

    def test_only_a_named_source_is_backed_up(self):
        (self.registry / storage.NAMED).write_text(json.dumps(dict(sources=[f'{RUNS}/another'])))
        self.refused(self.backup(), 'not at or below a source')
        (self.registry / storage.NAMED).unlink()
        self.refused(self.backup(), 'cannot be read')

    def test_a_sealed_name_is_refused_in_a_backup_too(self):
        record = self.backup()
        entry = record['files'].pop('nofit-01/own.bin'); record['files']['reserved-40b/own.bin'] = entry
        record['inventory_sha256'] = archives.inventory_sha256(record['files'])
        self.refused(record, 'sealed')
        sealed = dict(self.backup(), input=f'{RUNS}/pool/reserved-40b', bucket=dict(traces_root='studies', location='04-round-2-overlap/runs/pool/reserved-40b'))
        self.refused(sealed, "'reserved-40b' names sealed")

    def test_a_file_no_index_resolves_is_not_backed_up(self):
        record = self.backup(); record['membership']['pooled'] = {}
        self.refused(record, 'unresolved membership stays')

    def test_a_membership_source_stays_and_is_the_file_hashed(self):
        record = self.backup(); record['membership']['sources'] = {INDEX: 'b' * 64}
        self.refused(record, 'not the file the record hashed')
        inside = self.backup()
        inside['membership']['sources'] = self.bound(f'{POOL}/x-0.f32')
        self.refused(inside, 'stays on this disk')

    def test_a_read_back_of_another_inventory_is_refused(self):
        record = self.backup(remote_bytes_verified=dict(passed_ct='t', command='c', files=2, bytes=1, inventory_sha256='c' * 64))
        self.refused(record, 'not of this inventory')
        right = self.backup()
        right['remote_bytes_verified'] = dict(passed_ct='t', command='c', files=2, bytes=sum(e['bytes'] for e in right['files'].values()),
                                              inventory_sha256=right['inventory_sha256'])
        self.assertEqual(self.problems(right), [])

    def test_a_shape_that_is_neither_is_refused(self):
        self.refused(self.backup(storage_role='declared-input'), 'is neither')
        self.refused(self.model(storage_role='storage-only'), 'is neither')
        self.refused(self.backup(artifact='rows'), 'is neither')

    # the model

    def test_a_model_with_its_provenance_is_an_archive(self):
        self.assertEqual(self.problems(self.model()), [])

    def test_a_model_states_no_membership(self):
        self.refused(self.model(membership=dict(sources=self.bound(INDEX), answers=[], pooled={})), 'states no membership')
        self.refused(self.model(membership=dict(sources=self.bound(INDEX), answers=['fit-01'], pooled={'.': ['fit-01']})), 'states no membership')

    def test_a_model_says_answer_rows_are_not_applicable(self):
        record = self.model(); record['provenance']['answer_rows'] = []
        self.refused(record, 'word for word')

    def test_a_model_is_one_file(self):
        self.refused(self.model(kind='folder'), 'holds one file')

    def test_a_model_is_produced_by_the_run_it_lies_in(self):
        self.put(f'{RUNS}/other/declaration.json', b'{}')
        other = f'{RUNS}/other/declaration.json'
        record = self.model(); record['provenance']['sources'].update(self.bound(other)); record['provenance']['produced_by'] = other
        self.refused(record, 'produced_by')

    def test_the_written_record_states_the_file_inventoried(self):
        self.put(f'{RUNS}/edit/edit.json', json.dumps(dict(model_sha256='d' * 64)).encode())
        self.refused(self.model(), 'does not state the sha256 of the file inventoried')

    def test_the_original_is_stated_by_a_source(self):
        record = self.model(); record['provenance']['original']['sha256'] = 'e' * 64
        self.refused(record, 'does not state the original')
        record = self.model(); del record['provenance']['original']
        self.refused(record, 'needs the original')

    def test_a_provenance_source_changed_since_is_refused(self):
        record = self.model()
        self.put(f'{RUNS}/edit/edit.json', json.dumps(dict(model_sha256=sha(self.weights), note='edited')).encode())
        self.refused(record, 'not the file the record hashed')

    def test_a_model_under_a_no_fit_or_sealed_name_is_refused(self):
        for part in ('nofit-01', 'reserved-40b'):
            item = f'{RUNS}/edit/{part}/edited.gguf'
            self.put(item, self.weights)
            record = self.model(input=item, bucket=dict(traces_root='studies', location=item[len('studies/'):]))
            self.refused(record, f'{part!r} names sealed or no-fit')

    def test_sources_of_reads_either_shape(self):
        self.assertEqual(set(storage.sources_of(self.backup())), {INDEX})
        self.assertEqual(len(storage.sources_of(self.model())), 3)
        self.assertEqual(storage.sources_of(dict(form=archives.FORM)), {})


if __name__ == '__main__':
    unittest.main()
