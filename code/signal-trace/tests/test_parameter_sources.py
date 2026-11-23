"""A declared parameter's citation narrowed by a refinement bound to its declaration (Astra's contract,
records/retrospectives/2026-11-14-parameter-source-refinement-astra.md): complete and incomplete coverage of a cited folder, stale
bindings, sources that are gone, a newly added citation, sealed and no-fit names, and a refinement against an archive. Made-up files in
a temporary tree only: no file of the study, nothing sealed, no answer no fit uses."""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.research_guard import archive_inputs as tool
from pipeline.research_guard import input_archives as archives
from pipeline.research_guard import parameter_sources as sources
from pipeline.research_guard.verify_round2 import ROOT, verify
from tests.guard_fixture import copy_method
from tests.test_input_archives import ANSWERS, RUNS, STUDY, declaration, untouched

ACQ = f'{STUDY}/inputs/acq'                       # the folder two parameters cite whole
CASES = [f'{ACQ}/cases/a/case.json', f'{ACQ}/cases/b/case.json']
QUALIFICATION = f'{ACQ}/qualification'            # what an archive would take once nothing cites it
POPULATION = f'{ACQ}/population'
INDEX = f'{RUNS}/overlap/rows-index.json'         # the producer's index that names the two cases
DATA = f'{RUNS}/overlap/data'
CITING = dict(pairs=dict(measured_from=ACQ, derivation='d', scales_with='s'), regime=dict(measured_from=ACQ, derivation='d', scales_with='s'))


@contextlib.contextmanager
def unseen(*places):
    """Fail when a place, or anything below it, is so much as looked at: no status, no listing, no open."""
    real, lstat, stat_ = [os.path.realpath(p) for p in places], os.lstat, os.stat

    def guarded(call):
        def checked(path, *rest, **more):
            at = os.path.abspath(os.fsdecode(path)) if isinstance(path, (str, bytes, os.PathLike)) else None
            if at and any(at == place or at.startswith(place + os.sep) for place in real):
                raise AssertionError(f'{call.__name__} of {at}: looked at')
            return call(path, *rest, **more)
        return checked

    with untouched(*places), patch.object(os, 'lstat', guarded(lstat)), patch.object(os, 'stat', guarded(stat_)):
        yield


class ParameterSources(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve(); self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / STUDY / 'inputs').mkdir(parents=True)
        copy_method(self.tmp)
        self.put(f'{STUDY}/inputs/in.json', '{}')
        self.put(ANSWERS, json.dumps(dict(answers=['fit-01', 'fit-02'])))
        self.put(f'{STUDY}/ROUND-2.md', 'overlap edit later')
        self.registry = self.tmp / archives.REGISTRY; self.registry.mkdir()
        self.put(f'{archives.REGISTRY}/{archives.EXCLUDED}', json.dumps(dict(sealed_name_prefixes=['reserved'], no_fit_answers=['nofit-01'])))
        for path, text in zip(CASES, ('{"mass": 4}', '{"mass": 5}')):
            self.put(path, text)
        self.put(f'{QUALIFICATION}/q1.bin', 'qqqq'); self.put(f'{QUALIFICATION}/sub/q2.bin', 'qq')
        self.put(f'{POPULATION}/p1.json', '{}'); self.put(f'{POPULATION}/p2.json', '{}')
        self.put(INDEX, json.dumps(dict(sources=CASES)))
        self.put(f'{RUNS}/overlap/measured.json', '{}')
        self.put(f'{DATA}/a.bin', 'aaaa'); self.put(f'{DATA}/index.json', '{}')
        self.overlap = declaration(parameters=dict(CITING, step=dict(measured_from=f'{RUNS}/overlap/measured.json', derivation='d', scales_with='s')))
        self.declare('overlap', self.overlap)
        self.members = dict(sources=[ANSWERS], answers=['fit-01', 'fit-02'], pooled={'.': ['fit-01', 'fit-02']})

    def put(self, rel, text):
        file = self.tmp / rel; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(text)

    def declare(self, name, d):
        self.put(f'{RUNS}/{name}/declaration.json', json.dumps(d))

    def stated(self, **changes):
        return dict(dict(sources=CASES, subtrees={}, evidence=[INDEX], why='the two cases the producer\'s index names'), **changes)

    def refine(self, parameter, run='overlap', **changes):
        return tool.refine(self.tmp, f'{RUNS}/{run}/declaration.json', parameter, self.stated(**changes))

    def both(self, **changes):
        for parameter in ('pairs', 'regime'):
            name, problems = self.refine(parameter, **changes)
            self.assertEqual(problems, [])
        return name

    def keep(self):
        return archives.registry(self.tmp)[3]

    def said(self, deep=False):
        return tool.refinements(self.tmp, deep)[1]

    def record(self, name):
        return json.loads((self.registry / sources.FOLDER / name).read_text())

    def write(self, name, record):
        self.put(f'{archives.REGISTRY}/{sources.FOLDER}/{name}', json.dumps(record))

    def whole(self):
        """The cited folder is kept whole: nothing below it may be an archive's input, and `make` says so before it lists anything."""
        keep = self.keep()
        self.assertIn(ACQ, keep)
        self.assertTrue(archives.stays(f'{QUALIFICATION}/q1.bin', keep))
        with untouched(self.tmp / ACQ):
            made, problems = tool.make(self.tmp, QUALIFICATION, self.members)
        self.assertIsNone(made)
        self.assertTrue(any('parameter record' in p and 'nothing was listed or read' in p for p in problems), problems)

    def test_with_no_refinement_a_citation_is_kept_whole_and_nothing_is_read(self):
        with untouched(self.tmp / ACQ):
            self.assertEqual(tool.refinements(self.tmp), ({}, []))
        self.whole()
        self.assertEqual(verify(self.tmp), [])

    def test_a_folder_every_citing_parameter_narrows_is_kept_as_their_sources_and_evidence(self):
        self.both()
        keep = self.keep()
        self.assertNotIn(ACQ, keep)
        for path in CASES + [INDEX]:
            self.assertTrue(archives.stays(path, keep), path)
        self.assertFalse(archives.stays(f'{QUALIFICATION}/q1.bin', keep))
        self.assertEqual(self.said(deep=True), [])
        made, problems = tool.make(self.tmp, QUALIFICATION, self.members)
        self.assertEqual(problems, [])
        self.assertEqual(sorted(json.loads((self.registry / made).read_text())['files']), ['q1.bin', 'sub/q2.bin'])
        self.assertEqual(verify(self.tmp), [])

    def test_one_parameter_left_whole_keeps_the_whole_folder(self):
        self.assertEqual(self.refine('pairs')[1], [])
        self.whole()
        self.assertTrue(archives.stays(INDEX, self.keep()))     # what the one refinement names stays as well: the union

    def test_a_new_declaration_that_cites_the_folder_keeps_it_whole_again(self):
        self.both()
        self.assertNotIn(ACQ, self.keep())
        self.declare('later', declaration(outputs=[f'{RUNS}/later/x'], parameters=dict(count=dict(measured_from=ACQ, derivation='d', scales_with='s'))))
        self.whole()
        self.declare('later', declaration(outputs=[f'{RUNS}/later/x'], parameters=dict(count=dict(measured_from=QUALIFICATION, derivation='d', scales_with='s'))))
        keep = self.keep()
        self.assertEqual((ACQ in keep, QUALIFICATION in keep), (False, True))      # a narrower new citation keeps what it cites

    def test_a_refinement_bound_to_other_declaration_bytes_narrows_nothing_until_the_bytes_are_those_again(self):
        self.both()
        self.declare('overlap', dict(self.overlap, revisions=['a later word']))      # the parameters are the same; the bytes are not
        self.whole()
        self.assertEqual(len([p for p in self.said() if 'is not the bytes the refinement was bound to' in p and 'stays whole' in p]), 2)
        self.assertEqual(verify(self.tmp), [])                 # a refinement that narrows nothing is the operator's check to say, not a run's
        self.declare('overlap', self.overlap)
        self.assertNotIn(ACQ, self.keep())

    def test_a_refinement_of_another_citation_or_of_a_parameter_that_is_gone_narrows_nothing(self):
        name = self.both()
        for parameters in (dict(CITING, regime=dict(measured_from=QUALIFICATION, derivation='d', scales_with='s')), dict(pairs=CITING['pairs'])):
            changed = dict(self.overlap, parameters=parameters)
            self.declare('overlap', changed)
            for each in ('overlap--pairs.json', 'overlap--regime.json'):      # even bound again to the new bytes by hand
                self.write(each, dict(self.record(each), declaration_sha256=archives.digest(self.tmp / RUNS / 'overlap/declaration.json')))
            self.assertTrue(any('overlap--regime.json' in p and 'does not cite what the refinement states' in p for p in self.said()), self.said())
            self.assertFalse(any('overlap--pairs.json' in p for p in self.said()))
        self.assertEqual(name, 'overlap--regime.json')

    def test_a_source_that_is_gone_or_another_size_narrows_nothing_and_the_operators_check_reads_its_bytes(self):
        self.both()
        case = self.tmp / CASES[0]
        right = case.read_text()
        case.unlink()
        self.whole()
        case.write_text(right + ' ')
        self.whole()
        self.assertTrue(all('is not on this disk as the file the refinement names' in p for p in self.said()) and len(self.said()) == 2)
        case.write_text('{"mass": 9}')                         # the same size, other bytes
        self.assertNotIn(ACQ, self.keep())                     # a run's check looks at the size; the file itself still stays
        self.assertTrue(archives.stays(CASES[0], self.keep()))
        self.assertTrue(all('is not the bytes the refinement names' in p for p in self.said(deep=True)) and len(self.said(deep=True)) == 2)
        (self.tmp / INDEX).write_text(json.dumps(dict(sources=CASES[:1])))
        self.whole()                                           # the evidence is bound by its bytes at every check

    def test_an_empty_set_a_wildcard_a_path_outside_the_citation_a_link_and_no_evidence_are_refused(self):
        os.symlink(self.tmp / ACQ / 'cases', self.tmp / ACQ / 'again')
        for changes, text in ((dict(sources=[]), 'names no source'),
                              (dict(sources=[f'{ACQ}/cases/*/case.json']), 'a wildcard names no record'),
                              (dict(sources=[f'{ACQ}/cases/../cases/a/case.json']), 'not one path in its one spelling'),
                              (dict(sources=[f'{RUNS}/overlap/measured.json']), 'is not at or below the citation'),
                              (dict(sources=[f'{ACQ}/again/a/case.json']), 'is not the path it resolves to'),
                              (dict(sources=[f'{ACQ}/cases/a']), 'cannot be read'),
                              (dict(sources=[], subtrees={CASES[0]: 'a file named as a folder'}), 'is not a folder on this disk'),
                              (dict(sources=[], subtrees={f'{ACQ}/cases': ' '}), 'neither a file with its bytes and sha256 nor a named subtree'),
                              (dict(evidence=[]), 'names no producer or index record'),
                              (dict(why=''), 'does not say what the derivation needs')):
            made, problems = self.refine('pairs', **changes)
            self.assertIsNone(made)
            self.assertTrue(any(text in p for p in problems), (text, problems))
        self.assertFalse((self.registry / sources.FOLDER).exists() and any((self.registry / sources.FOLDER).iterdir()))
        self.assertIn("states no parameter 'absent' with a measured_from", tool.refine(self.tmp, f'{RUNS}/overlap/declaration.json', 'absent', self.stated())[1][0])
        self.whole()

    def test_a_sealed_or_no_fit_name_is_refused_from_its_spelling_and_never_looked_at(self):
        self.put(f'{ACQ}/reserved-40b/held.json', '{}'); self.put(f'{ACQ}/cases/nofit-01/case.json', '{}')
        sealed, no_fit = self.tmp / ACQ / 'reserved-40b', self.tmp / ACQ / 'cases/nofit-01'
        entry = dict(bytes=2, sha256=archives.digest(sealed / 'held.json'))
        self.both()
        with unseen(sealed, no_fit):
            for changes in (dict(sources=[f'{ACQ}/reserved-40b/held.json']), dict(sources=[f'{ACQ}/cases/nofit-01/case.json']),
                            dict(sources=CASES, subtrees={f'{ACQ}/reserved-40b': 'kept whole'}), dict(evidence=[f'{ACQ}/cases/nofit-01/case.json'])):
                made, problems = self.refine('pairs', **changes)
                self.assertIsNone(made)
                self.assertTrue(all('names sealed or no-fit material' in p and 'nothing was opened' in p for p in problems) and problems, problems)
            for named in (f'{ACQ}/reserved-40b/held.json', f'{ACQ}/cases/nofit-01/case.json'):   # a record written by hand is not looked at either
                self.write('overlap--pairs.json', dict(self.record('overlap--regime.json'), parameter='pairs', sources={named: entry}))
                self.assertIn(ACQ, self.keep())
                self.assertTrue(any('names sealed or no-fit material' in p for p in self.said(deep=True)))
            self.assertEqual(verify(self.tmp), [])

    def test_a_named_subtree_stays_whole_and_the_check_never_lists_it(self):
        self.both(subtrees={POPULATION: 'the screened population: its ledger names no file one by one'})
        with untouched(self.tmp / POPULATION):
            keep = self.keep()
            self.assertEqual(self.said(deep=True), [])
        self.assertTrue(archives.stays(f'{POPULATION}/p1.json', keep))
        self.assertFalse(archives.stays(f'{QUALIFICATION}/q1.bin', keep))
        made, problems = tool.make(self.tmp, ACQ, self.members)       # the folder itself: what stays in it is left where it is
        self.assertEqual(problems, [])
        record = json.loads((self.registry / made).read_text())
        self.assertEqual(sorted(record['files']), ['qualification/q1.bin', 'qualification/sub/q2.bin'])
        self.assertEqual(record['left_on_this_disk'], {'a record that stays on this disk': 4})
        self.assertEqual(verify(self.tmp), [])

    def test_an_archive_record_written_over_a_refined_source_stands_for_nothing(self):
        self.both()
        made, _ = tool.make(self.tmp, QUALIFICATION, self.members)
        record = json.loads((self.registry / made).read_text())
        case = dict(bytes=(self.tmp / CASES[0]).stat().st_size, sha256=archives.digest(self.tmp / CASES[0]))
        files = {f'qualification/{sub}': entry for sub, entry in record['files'].items()} | {'cases/a/case.json': case}
        (self.registry / made).write_text(json.dumps(dict(record, input=ACQ, bucket=dict(record['bucket'], location=ACQ[len('studies/'):]), files=files,
                                                          inventory_sha256=archives.inventory_sha256(files))))
        records, problems, _, keep = archives.registry(self.tmp)
        self.assertEqual(records, [])
        self.assertTrue(any('stay on this disk' in p for p in problems), problems)
        self.assertIn(ACQ, keep)      # a source in an inventory: the refinements give way, the folder is whole again, and the record over it is refused

    def test_a_refinement_takes_nothing_back_from_an_archive(self):
        made, problems = tool.make(self.tmp, DATA, self.members)
        self.assertEqual(problems, [])
        for changes, text in ((dict(evidence=[f'{DATA}/index.json']), 'evidence'), (dict(evidence=[INDEX, f'{DATA}/a.bin']), 'evidence')):
            refused, problems = self.refine('pairs', **changes)
            self.assertIsNone(refused)
            self.assertTrue(any(text in p and 'takes nothing back from an archive' in p for p in problems), problems)
        self.both()
        self.write('overlap--pairs.json', dict(self.record('overlap--pairs.json'), evidence={f'{DATA}/index.json': archives.digest(self.tmp / DATA / 'index.json')}))
        self.assertEqual(archives.registry(self.tmp)[1], [])          # the record is whole; it is the refinement that gives way
        self.assertIn(ACQ, self.keep())
        self.assertEqual(verify(self.tmp), [])

    def test_two_refinements_of_one_parameter_narrow_nothing(self):
        self.both()
        self.write('again.json', self.record('overlap--pairs.json'))
        self.whole()
        self.assertTrue(any('more than one refinement' in p for p in self.said()))

    def test_every_other_record_stays_as_it_did(self):
        before = self.keep()
        self.both()
        after = self.keep()
        self.assertEqual(before - after, {ACQ})
        self.assertEqual(after - before, set(CASES) | {INDEX})
        for path in (f'{RUNS}/overlap/measured.json', f'{RUNS}/overlap/declaration.json', archives.METHOD, f'{archives.REGISTRY}/{sources.FOLDER}/overlap--pairs.json'):
            self.assertTrue(archives.stays(path, after), path)

    def test_a_refinement_is_not_overwritten_and_the_citations_are_counted_from_the_declarations(self):
        self.assertEqual(self.refine('pairs'), ('overlap--pairs.json', []))
        again, problems = self.refine('pairs', why='another word')
        self.assertIsNone(again); self.assertIn('a refinement is not overwritten', problems[0])
        self.assertEqual(self.record('overlap--pairs.json')['why'], self.stated()['why'])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(tool.citing(self.tmp, f'{QUALIFICATION}/q1.bin'), [])
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('overlap / pairs', lines[0]); self.assertIn('narrowed to 3', lines[0])
        self.assertIn('overlap / regime', lines[1]); self.assertTrue(lines[1].endswith('whole'))
        self.assertIn('stays whole, 1 citation(s) are not narrowed', lines[2])


if __name__ == '__main__':
    unittest.main()
