import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.native_records import run_declaration
from pipeline.research_guard.verify_round2 import ROOT, verify
from tests.guard_fixture import copy_method

METHOD = 'studies/04-round-2-overlap/round-2-method.json'


def full_declaration(**extra):
    d = dict(method_step=[1, 2, 3, 4], why_this_direction='w', why_it_might_be_wrong='x',
             repeating_pattern_check='p', hypothesis='h', what_went_wrong_before='b', rule='r',
             prediction='pr', would_weaken_it='ww', inputs=['shared/in.json'],
             outputs=['studies/04-round-2-overlap/runs/overlap/OVERLAP.md'], run_by='astra')
    d.update(extra)
    return d


class Round2Verifier(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp)
        copy_method(self.tmp, read_only_inputs='shared')
        (self.tmp / 'shared').mkdir(); (self.tmp / 'shared/in.json').write_text('{}')
        (self.tmp / 'studies/04-round-2-overlap/ROUND-2.md').write_text('overlap')
        self.runs = self.tmp / 'studies/04-round-2-overlap/runs'

    def run_folder(self, name, declaration, result=None):
        folder = self.runs / name; folder.mkdir(parents=True)
        (folder / 'declaration.json').write_text(json.dumps(declaration))
        if result is not None:
            (folder / 'result.json').write_text(json.dumps(result))
        return folder

    def test_complete_overlap_declaration_passes(self):
        self.run_folder('overlap', full_declaration())
        self.assertEqual(verify(self.tmp), [])

    def test_the_runs_folder_itself_is_not_an_input(self):
        for spelling in ('studies/04-round-2-overlap/runs', 'studies/04-round-2-overlap/runs/'):
            (self.runs / 'overlap').mkdir(parents=True, exist_ok=True)
            (self.runs / 'overlap/declaration.json').write_text(json.dumps(full_declaration(inputs=['shared/in.json', spelling])))
            self.assertEqual(verify(self.tmp), [f'overlap: input {spelling} is the whole runs folder, not a run or a file of one'])

    def test_a_target_or_an_input_that_is_no_path_is_reported_not_raised(self):
        self.run_folder('overlap', full_declaration())
        (self.tmp / 'studies/04-round-2-overlap/ROUND-2.md').write_text('overlap edit-a')
        for target in ('studies/04-round-2-overlap/runs', 'studies/04-round-2-overlap/runs/.', ['studies/04-round-2-overlap/runs/overlap'], 7):
            folder = self.runs / 'edit-a'; folder.mkdir(exist_ok=True)
            (folder / 'declaration.json').write_text(json.dumps(full_declaration(
                method_step=5, target_source=target, outputs=['studies/04-round-2-overlap/runs/edit-a/x'])))
            self.assertEqual(verify(self.tmp), ['edit-a: edit steps need target_source inside a step-3/4 overlap run'], target)
        (folder / 'declaration.json').write_text(json.dumps(full_declaration(
            inputs=['shared/in.json', ['shared/in.json'], 3, ''], outputs=['studies/04-round-2-overlap/runs/edit-a/x'])))
        self.assertEqual(verify(self.tmp), ["edit-a: input ['shared/in.json'] is not a path", 'edit-a: input 3 is not a path',
                                            "edit-a: input '' is not a path"])

    def test_missing_reasoning_fields_fail(self):
        d = full_declaration(); del d['why_this_direction']; del d['repeating_pattern_check']
        self.run_folder('overlap', d)
        problems = verify(self.tmp)
        self.assertTrue(any('why_this_direction' in p for p in problems))
        self.assertTrue(any('repeating_pattern_check' in p for p in problems))

    def test_bare_or_underived_parameter_fails(self):
        self.run_folder('overlap', full_declaration(parameters=dict(cap=0.072, step=dict(measured_from='shared/in.json'))))
        problems = verify(self.tmp)
        self.assertTrue(any('cap is a bare number' in p for p in problems))
        self.assertTrue(any('step lacks derivation' in p for p in problems))
        self.assertTrue(any('step lacks scales_with' in p for p in problems))

    def test_edit_needs_an_overlap_target(self):
        self.run_folder('edit-a', full_declaration(method_step=5, target_source='studies/04-round-2-overlap/runs/overlap/declaration.json',
                                                   outputs=['studies/04-round-2-overlap/runs/edit-a/x']))
        (self.tmp / 'studies/04-round-2-overlap/ROUND-2.md').write_text('overlap edit-a')
        self.assertTrue(any('step-3/4 overlap run' in p for p in verify(self.tmp)))
        self.run_folder('overlap', full_declaration())
        self.assertEqual(verify(self.tmp), [])

    def test_first_attempt_reference_and_unlisted_run_fail(self):
        self.run_folder('other', full_declaration(rule='reuse joint-a2 targets', outputs=['studies/04-round-2-overlap/runs/other/x']))
        problems = verify(self.tmp)
        self.assertTrue(any('joint-a2' in p for p in problems))
        self.assertTrue(any('does not mention run other' in p for p in problems))

    def test_original_model_read_needs_declared_hash(self):
        method = json.loads((self.tmp / METHOD).read_text())
        model = self.tmp / method['read_only_model']['path']
        model.parent.mkdir(parents=True)
        model.write_bytes(b'gguf')
        inputs = ['shared/in.json', method['read_only_model']['path']]
        self.run_folder('overlap', full_declaration(inputs=inputs))
        self.assertTrue(any('model_sha256' in p for p in verify(self.tmp)))
        (self.runs / 'overlap/declaration.json').write_text(json.dumps(
            full_declaration(inputs=inputs, model_sha256=method['read_only_model']['sha256'])))
        self.assertEqual(verify(self.tmp), [])

    def test_uncommitted_declaration_and_result_are_valid_records(self):
        method = json.loads((self.tmp / METHOD).read_text())
        result = {field: 'recorded outcome' for field in method['result_fields']}
        self.run_folder('overlap', full_declaration(), result)
        self.assertEqual(verify(self.tmp), [])

    def test_sweep_record_does_not_require_a_separate_step_5_run(self):
        folder = self.run_folder('overlap', full_declaration())
        self.run_folder('sweep', full_declaration(
            method_step=6,
            target_source=str((folder / 'declaration.json').relative_to(self.tmp)),
            outputs=['studies/04-round-2-overlap/runs/sweep/result.json']))
        (self.tmp / 'studies/04-round-2-overlap/ROUND-2.md').write_text('overlap sweep')
        self.assertEqual(verify(self.tmp), [])

    def test_native_record_can_be_updated_after_outputs_exist(self):
        declaration = full_declaration()
        folder = self.run_folder('overlap', declaration)
        output = folder / 'capture.f32'
        output.write_bytes(b'recorded output')
        os.utime(output, ns=(1, 1))
        with patch.object(run_declaration, 'ROOT', self.tmp):
            self.assertEqual(run_declaration.check_native_run(folder, folder / 'declaration.json'), declaration)

    def test_repository_itself_verifies(self):
        self.assertEqual(verify(ROOT), [])


if __name__ == '__main__':
    unittest.main()
