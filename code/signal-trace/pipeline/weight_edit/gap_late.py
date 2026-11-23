"""A sweep's passes started late: what the driver binds before a pass takes rows the pass before it recorded, and how each kind of step
is settled on the sweep's own writes (code/signal-trace/recorder/STREAM-RECORDER.md, "observe-stream-late"; the admission and its
conditions: records/retrospectives/2026-11-14-late-start-admission-astra.md).

A matrix's rows are two recorded outputs, its input and the state its output is added to (ladder.labels_of), and both lie before the
matrix. When the matrix has been written, the pass that reads the next matrix's rows need not compute the model up to there again: seeded
with those two outputs, the late recorder computes from the written matrix on. What that recorder cannot know is whether a seed belongs
to the file it is reading: it holds a seed to the engine at one position and takes every other row from the seed's file. The driver
binds it, here.

  a seed    one output of a pass, every answer's rows in position order in one file, with what it is bound to: the sha256 of the
            bytes the pass gave, the pass and the recorder that made it, the state that pass read (gap_state's identity, and each
            changed tensor's own sha256), and the place in the path order of the matrix whose output it is. Everything a seed was
            computed from lies before that place
  belongs   a seed belongs to a pass of the file as it now stands when the file is of the same original, every matrix before the
            seed's place holds the bytes it held when the seed was recorded, the seed's file holds the bytes the pass gave, and the
            pass reads the matrix right after the seed's in the path order. The answers, their text and positions, the two
            recorders, the engine's libraries and the card are one `scope` for the life of this object, and a seed never outlives
            the process that wrote it: a sweep taken up begins with an ordinary pass
  its bytes a seed's file is hashed whole where it is consumed, not known by its size and time: before a late pass is started, and
            again when that pass has ended, since the recorder holds a seed to the engine at one position only and reads every other
            row from the file. A late pass whose seed held other bytes at either end is not taken (`Changed`): the ordinary pass
            supplies the rows, nothing is said of the step's kinds, and the record says why
  a step    the matrix written since the seeds were recorded, and the matrix now read (late_start.kinds). The late pass alone
            supplies a step's rows only when each kind of the step has been compared whole with the ordinary pass across a real
            write of this sweep, and held (late_start.Held). Until then the ordinary pass supplies them; where the step crosses a
            real write and a kind is not yet compared, both passes are made and compared, once. A kind that did not hold stays
            failed, its two raw streams are left where they are, and its steps keep the ordinary pass
  anchors   from layer 15 on a late pass is also given the two states the shared caches are made from. They are seeds like any other,
            bound the same way and kept for as long as they belong; a process that begins past them asks its first pass for them

Whenever a seed is missing or does not belong, the pass is the ordinary one and the reason is in the matrix's record. Nothing here
chooses a matrix, an order, a row or a request: it decides which recorder computes rows that are the same bytes either way, and every
record says which did."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from pipeline.acquisition import late_start
from pipeline.acquisition.run_a_baseline import BACKENDS
from pipeline.acquisition.run_a_observation import RECORDERS, STREAM_RECORDERS
from pipeline.weight_edit import ladder, tube
from pipeline.weight_edit.gap_rows import of_name, unused

ORDINARY = STREAM_RECORDERS[0]
# The incoming states of the two layers whose caches every layer from late_start.SHARED_FROM on reads, each with the place of the matrix
# whose incoming state it is (that layer's attention output).
ANCHORS = {f'l_out-{layer - 1}': 2 * layer for layer in (late_start.SHARED_FROM - 2, late_start.SHARED_FROM - 1)}


def place(name):
    """A matrix's place in the path order: every layer's attention output and then its feed-forward, first layer to last."""
    layer, kind = of_name(name)
    return 2 * layer + list(tube.KINDS).index(kind)


class Changed(RuntimeError):
    """A seed's file did not hold the bytes recorded for it when a late pass had read it: what that pass computed from is not known."""


def changed_of(working):
    """Every tensor of the working file that is no longer the original's, with the sha256 of its bytes."""
    return {name: each['sha256'] for name, each in working.manifest['changed'].items()}


def libraries(folder=BACKENDS):
    """The engine's shared libraries as they are now: one sha256 over each file's name and sha256."""
    files = sorted(each for each in Path(folder).iterdir() if each.is_file() and not each.is_symlink() and '.so' in each.name)
    if not files:
        raise ValueError(f'no library in {folder}')
    return hashlib.sha256(''.join(f'{each.name} {tube.digest(each)}\n' for each in files).encode()).hexdigest()


def declared(run, asked, chosen, original_sha256, starts_from=None, kept_under=None):
    """The late start a run's declaration names (`program` and its `sha256`), held to the installed file, with the scope its comparisons
    are good for: the two recorders, the engine's libraries, card 0 and its driver, the original, the answers with their text, and how
    the passes read it. A sweep that starts from a state another sweep reached says so (`starts_from`, that state's file's sha256):
    it is then part of the scope, with the answers that state wrote. `kept_under` is the folder such a sweep's comparisons are kept
    in, a pass of a repeated correction its own; without it they lie under the run, as a single sweep's do."""
    if asked['program'] not in STREAM_RECORDERS or asked['program'] == ORDINARY:
        raise ValueError(f"{asked['program']!r} is no installed recorder with a late start")
    installed = RECORDERS/asked['program']
    if tube.digest(installed) != asked['sha256'] and not tube.same_program(installed, asked['sha256']):
        raise ValueError(f"The installed {asked['program']} is not the program this run declares for its late passes")
    card = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
    text = hashlib.sha256(json.dumps([[m['id'], m['tokens'], m['recorded']] for m in chosen], separators=(',', ':')).encode()).hexdigest()
    scope = dict(ordinary=ORDINARY, ordinary_sha256=tube.digest(RECORDERS/ORDINARY), late=asked['program'], late_sha256=tube.digest(installed), libraries_sha256=libraries(),
                 card=card, original_sha256=original_sha256, answers=[m['id'] for m in chosen], text_sha256=text, passes_read=tube.READING, stop_after_last_label=tube.STOPS)
    if starts_from is not None:
        scope['starts_from'] = starts_from
    return Late(run, chosen, asked['program'], scope, kept_under=kept_under)


class Late:
    """The passes of one sweep in one process. `read` gives a matrix's two outputs as the file now computes them, and how they were read."""

    def __init__(self, run, chosen, program, scope, passed=None, kept_under=None):
        self.run, self.chosen, self.program = Path(run), chosen, program
        self.held = late_start.Held((self.run/'late-start' if kept_under is None else Path(kept_under))/'held.json', scope)
        self.passed = passed or tube.passed
        self.seeds, self.last = {}, None   # the seeds this process holds by label, and the matrix its last pass read
        self.first_row = dict(zip((m['id'] for m in chosen), np.cumsum([0] + [m['question'] + m['steps'] for m in chosen]).tolist()))
        self.folder = self.run/'passes'/'seeds'
        if self.folder.exists():   # a stopped process's seeds are nobody's: they are bound to nothing this process knows
            shutil.rmtree(self.folder)
        self.folder.mkdir(parents=True)

    def close(self):
        """The seeds are this process's and serve nothing after it: their files go with it."""
        shutil.rmtree(self.folder, ignore_errors=True)
        self.seeds, self.last = {}, None

    def unbound(self, seed, working):
        """Why a seed does not belong to a pass of the file as it stands, or None when it does."""
        if working.identity()['original_sha256'] != seed['original_sha256']:
            return 'it was recorded from a file of another original'
        now = changed_of(working)
        for name in now:
            try:
                place(name)
            except ValueError:
                return f'{name} is changed and is no matrix of the path order: what it reaches is not known here'
        before = lambda changed: {name: sha256 for name, sha256 in changed.items() if place(name) < seed['place']}
        if before(now) != before(seed['changed']):
            return 'a matrix before it holds other bytes than when it was recorded'
        return self.other_bytes(seed)

    def other_bytes(self, seed):
        """Why a seed's file is not the bytes its pass gave, or None when it is: the whole file hashed, as it lies there now."""
        try:
            found = tube.digest(seed['file'])
        except FileNotFoundError:
            return 'its file is gone'
        return None if found == seed['sha256'] else 'its file is not as it was written: it holds other bytes'

    def plan(self, name, working, labels):
        """Which pass supplies this matrix's rows, and why: 'late' (with its seeds), 'both' (the ordinary pass supplies them and the
        late pass is compared with it) or 'ordinary'."""
        ordinary = lambda why: dict(take='ordinary', why=why)
        if tube.READING != 'as generated' or not tube.STOPS:
            return ordinary('the run\'s passes are not read as generated with the stop after the last label: the late recorder reads no other case')
        if self.last is None:
            return ordinary('no pass of this process before this one: there are no seeds')
        if place(name) != place(self.last) + 1:
            return ordinary(f'{name} is not the matrix after {self.last} in the path order')
        before = list(ladder.labels_of(*of_name(self.last)))
        given = before + ([label for label in ANCHORS if label not in before and label not in labels] if of_name(name)[0] >= late_start.SHARED_FROM else [])
        for label in given:
            if label not in self.seeds:
                return ordinary(f'no seed for {label}')
            why = self.unbound(self.seeds[label], working)
            if why:
                return ordinary(f'the seed {label}: {why}')
        step, seeds = late_start.kinds(self.last, name), [self.seeds[label] for label in given]
        if self.held.may(step):
            return dict(take='late', step=step, seeds=seeds)
        failed = [kind for kind in step if kind in self.held.record['kinds'] and not self.held.record['kinds'][kind]['held']]
        if changed_of(working).get(self.last) == seeds[0]['changed'].get(self.last):
            return ordinary(f'{self.last} was not written, so nothing is compared across it, and a kind of this step has not held')
        if not self.held.open(step):
            return ordinary(f'a kind of this step did not hold when it was compared ({"; ".join(failed)}): its steps keep the ordinary pass')
        return dict(take='both', step=step, seeds=seeds)

    def one(self, name, working, of, wanted, seeds=None, keep=False, kept=True):
        """One pass of the working file. With `seeds` it is started late. With `kept`, each wanted output becomes a seed, bound to
        the state the pass read. Returns the wanted outputs (tube.passed's) and the seeds made."""
        state, files = working.identity(), {}

        def each(member, label, rows):
            if label not in files:
                files[label] = [(self.folder/f'{label}.f32.next').open('wb'), hashlib.sha256(), int(rows.shape[1]), 0]
            raw = np.ascontiguousarray(rows, dtype='<f4').tobytes()
            files[label][0].write(raw)
            files[label][1].update(raw)
            files[label][3] += int(rows.shape[0])

        started = (self.program, {m['id']: [dict(label=seed['label'], vectors=str(seed['file']), width=seed['width'], first_row=self.first_row[m['id']]) for seed in seeds]
                                  for m in self.chosen}) if seeds else None
        try:
            arrays = self.passed(working.file, self.run, name, wanted, self.chosen, started=started, each=each if kept else None, keep=keep)
            read = [f"{seed['label']} ({why})" for seed in seeds or [] for why in [self.other_bytes(seed)] if why]   # the bytes the pass has just consumed
            if read:
                raise Changed(f"when the late pass {name} had ended, a seed it read was no longer the bytes recorded for it: {'; '.join(read)}")
        except BaseException:
            for out, *_ in files.values():
                out.close()
                Path(out.name).unlink(missing_ok=True)
            raise
        made = {}
        for label, (out, found, width, rows) in files.items():
            out.flush()
            os.fsync(out.fileno())
            out.close()
            file = self.folder/f'{label}.f32'
            os.replace(str(file) + '.next', file)
            made[label] = dict(label=label, file=file, width=width, rows=rows, sha256=found.hexdigest(),
                               place=ANCHORS[label] if label in ANCHORS else place(of), state=state['sha256'], original_sha256=state['original_sha256'], changed=changed_of(working),
                               recorded_by=dict(passed=name, recorder=self.program if seeds else ORDINARY, read=of))
        return arrays, made

    def read(self, name, working):
        """The matrix's input and the state its output is added to, as the file now computes them (tube.passed's two arrays), and how
        they were read: the recorder, the pass, why it was the ordinary one or which seeds it started from, and a comparison made."""
        labels = list(ladder.labels_of(*of_name(name)))
        plan = self.plan(name, working, labels)
        state, began = working.identity(), time.time()
        told = lambda seed: dict(label=seed['label'], sha256=seed['sha256'], rows=seed['rows'], width=seed['width'], recorded_in_state=seed['state'], recorded_by=seed['recorded_by'])
        if plan['take'] == 'late':
            passed = unused(self.run, name, state, '-late')
            try:
                arrays, made = self.one(passed, working, name, labels, plan['seeds'])
                how = dict(recorder=self.program, passed=passed, step=plan['step'], seeds=[told(seed) for seed in plan['seeds']])
            except Changed as changed:   # what that pass read is not known: its rows are not taken, and the ordinary pass supplies them
                plan = dict(take='ordinary', why=f'{changed}. Its rows are not taken')
        if plan['take'] != 'late':
            # a process that begins past an anchor's own pass asks this pass for it: the stop after the last label lies after both
            lacking = [label for label, at in ANCHORS.items() if plan['take'] == 'ordinary' and tube.STOPS and tube.READING == 'as generated'
                       and at < place(name) and label not in labels and (label not in self.seeds or self.unbound(self.seeds[label], working))]
            passed = unused(self.run, name, state)
            arrays, made = self.one(passed, working, name, labels + lacking, keep=plan['take'] == 'both')
            how = dict(recorder=ORDINARY, passed=passed, **({'why': plan['why']} if plan['take'] == 'ordinary' else {}))
            if plan['take'] == 'both':
                late, ids, refused, unread = unused(self.run, name, state, '-late'), [m['id'] for m in self.chosen], None, None
                try:
                    self.one(late, working, name, labels, plan['seeds'], keep=True, kept=False)
                    compared = late_start.same(self.run/'passes'/passed, self.run/'passes'/late, ids)
                except subprocess.CalledProcessError as error:   # the late recorder threw (a seed that is not the file's own output): the kind did not hold
                    compared, refused = [], f'the late recorder ended with {error.returncode}: {Path(self.run/"passes"/late).with_suffix(".stderr.log").name}'
                except Changed as changed:   # no comparison of the recorder was made: the step's kinds stay as they were
                    unread = str(changed)
                if unread is None:
                    held = self.held.note(plan['step'], self.last, name, compared, state_read=state['sha256'], passes=dict(ordinary=passed, late=late),
                                          seeds=[told(seed) for seed in plan['seeds']], **({'late_pass_refused': refused} if refused else {}))
                    how.update(step=plan['step'], compared=dict(with_pass=late, recorder=self.program, held=held, seeds=[told(seed) for seed in plan['seeds']]))
                else:
                    how.update(step=plan['step'], not_compared=dict(with_pass=late, why=unread))
                if unread is not None or held:   # two streams that were compared and differ stay where they are, whole, for whoever looks at the difference
                    for folder in (passed, late):
                        for each in ids:
                            shutil.rmtree(self.run/'passes'/folder/each, ignore_errors=unread is not None)
        for label in [label for label in self.seeds if label not in made and label not in ANCHORS]:   # the pass before's seeds have served
            self.seeds.pop(label)['file'].unlink(missing_ok=True)
        self.seeds.update(made)
        self.last = name
        return arrays[:2], dict(how, seconds=round(time.time() - began, 1))
