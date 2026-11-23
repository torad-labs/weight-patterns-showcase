"""What a capture of a model's own answers belongs to, is made from and is registered on. own_text.py's, importable, files only.

Until November 15, 2026 own_text.py took a cohort and an output folder and nothing else. It called tube.declared on the COHORT's
declaration (how that cohort's old passes read a text, and its allowance), so a capture for a new run was checked against no
declaration of its own before the recorder started; and it bound and registered through the frozen axis of the first overlap
(loop_requests.capture_sources, tube.registered), whose axis answer is not one of the thirty fitting answers. Astra named both
(her message of November 15, 2026; records/retrospectives/2026-11-15-first-pass-acquisition-astra.md): a capture is admitted by the run
that declares it, and a loop that fits and measures on the thirty fitting answers opens no other answer's arrays. Three functions:

  admitted           a capture is a DECLARED run's. own_text.py calls the research checker on that run's declaration first
                     (pipeline.native_records.run_declaration.check_native_run: the verifier, and the run's archived inputs back
                     on this disk) and then this, before a folder is made or the recorder started. The output lies inside the
                     run. For a pass of a repeated correction (PASS=<n>) the output is that pass's capture (gap_run.of_pass), the
                     cohort is the declaration's, the model is the state the pass starts from BY ITS SHA256 (gap_run.begins: the
                     declared start for pass 1, the state the pass before swept to after it), and the reference history is the
                     one the declaration's measurement names. Returns what the capture's record says it belongs to.
  native_events      what a declared functional loop's acquisition is: `native-events-only`, said by the declaration before anything
                     is generated, or the capture is refused. Such a capture emits each answer's own history, its ending and the
                     native states, and pairs nothing with anything.
  one_behind         the order of a capture's work in time, and nothing of its content: each batch of answers is finished (exports,
                     files, registrations) while the recorder writes the next batch, so the card does not wait on the processor.

Until the evening of November 15, 2026 this file also held the registration of forms 1 to 3 (original_sources, unchanged_since,
laid_on_original: a generated answer laid on its own original recording by the unit turns). It is in capture_registration.py now, the
same bytes, for the old captures alone: a capture that only acquires imports this file and no pairing with it.

What none of this is: a judgment that a path pairs rows of the same reasoning, or an admission of a run. The checker decides whether
a declared run may execute; this only refuses a capture that is not the declared one."""
import hashlib
import os
from pathlib import Path
from typing import Any

NATIVE_EVENTS = 'native-events-only'   # functional_loop.acquisition: the answer's own history, its ending and the native states, nothing paired


def lease_fds(state):
    """Keep the sole controller's ownership lock alive in its native child."""
    if state is None:
        return ()
    said = os.environ.get('WP_WORKING_MODEL_OWNER_FD', '')
    if not said.isdigit():
        raise ValueError('working-model acquisition requires its inherited ownership descriptor')
    fd = int(said)
    actual, expected = os.fstat(fd), (Path(state['model_file']).parent / 'owner.lock').stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('the inherited ownership lock is for another working model')
    return (fd,)


def digest(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def pass_number(said):
    """PASS as a pass's number, or None where none is given."""
    if said is None or said == '':
        return None
    if not str(said).isdigit() or int(said) < 1:
        raise ValueError(f'a pass is numbered from 1, and PASS is {said!r}')
    return int(said)


def history(before, root):
    """How a record names the reference history: the recordings, or the folder of an earlier capture from the repository's root."""
    if before == 'recordings':
        return 'recordings'
    at = Path(before).resolve()
    return str(at.relative_to(root)) if at.is_relative_to(root) else at.name


def admitted(root, declared, declaration, cohort_run, model_sha256, before, out, number=None, kept_streams=None,
             *, model_identity=None, model=None):
    """What a capture belongs to, or ValueError where it is not the declared run's. `declaration` is what the checker returned for
    `declared` (the caller has run it); `model_sha256` the sha256 of the file the capture will read; `before` as `history` names it;
    `kept_streams` the folder KEEP_RAW names, if any."""
    root, declared, out = Path(root).resolve(), Path(declared).resolve(), Path(out).resolve()
    if not out.is_relative_to(declared) or out == declared:
        raise ValueError(f'a capture is written inside the run that declares it: {out} is not under {declared}')
    if kept_streams is not None and not Path(kept_streams).resolve().is_relative_to(declared):
        raise ValueError('the kept streams (KEEP_RAW) lie outside the declared run: a run writes only in its own folder')
    said = dict(run=declared.name, declaration_sha256=digest(declared/'declaration.json'), declared_step=declaration.get('method_step'))
    if 'functional_loop' in declaration:
        import os
        from pipeline.weight_edit import functional_capture
        if os.environ.get('TUBE'):
            raise ValueError('a fitting functional capture cannot substitute its bound source')
        return dict(**said, **functional_capture.admitted(
            root, declared, declaration, cohort_run, model_sha256, before, out, number,
            case_manifest=os.environ.get('CASES'), model_identity=model_identity, model=model))
    if model_identity is not None or declaration.get('working_model') is not None:
        raise ValueError('a working-model capture requires its declared functional loop')
    if number is None:
        return said
    from pipeline.weight_edit import gap_run
    asked = declaration.get('sweep')
    if not isinstance(asked, dict):
        raise ValueError('a pass belongs to a run whose declaration names its sweep')
    at = gap_run.of_pass(declared, number)
    if out != (at['capture']/'own').resolve():
        raise ValueError(f"pass {number}'s capture is {at['capture']/'own'}, and {out} was named")
    if (root/asked['cohort']).resolve() != Path(cohort_run).resolve():
        raise ValueError(f"the declaration reads its answers' questions from {asked['cohort']}: a capture of another cohort is not this run's")
    start: Any = gap_run.begins(declared, asked, number, root)
    wanted = declaration['model_sha256'] if start is None else start['kept']['model_sha256']
    if model_sha256 != wanted:
        raise ValueError(f'the model is not the state pass {number} starts from: its sha256 is {model_sha256[:12]}… and that state is {wanted[:12]}…')
    measured = declaration.get('set_measurement')
    if measured is not None and measured.get('history_before') != before:
        raise ValueError(f"the declaration's measurement names the reference history {measured.get('history_before')!r}, and the capture was given {before!r}")
    return dict(**said, pass_number=number,
                starts_from=dict(model_sha256=wanted, state='the original' if start is None else start['kept'].get('identity'),
                                 file=None if start is None else str(Path(start['file']).resolve().relative_to(root))))


def native_events(declaration):
    """Whether a capture of this declared run only acquires: True for a functional loop that says `acquisition: native-events-only`,
    False for a run that declares no functional loop, ValueError for a functional loop that says anything else. A functional loop's
    capture never registers an answer: the one-pair loop pairs events by its own relation, after the capture."""
    loop = declaration.get('functional_loop')
    if loop is None:
        return False
    if not isinstance(loop, dict) or loop.get('acquisition') != NATIVE_EVENTS:
        raise ValueError(f"a functional loop's capture is {NATIVE_EVENTS!r} and its declaration says so before anything is generated: "
                         f"functional_loop.acquisition is {loop.get('acquisition') if isinstance(loop, dict) else loop!r}")
    return True


def one_behind(batches, start, ended, finish):
    """Takes `batches` through the recorder one call at a time and finishes each batch (its exports, its answers' files, its
    registrations: work that needs no card) while the NEXT batch's call runs, in this process and on no thread. `start(batch)` begins
    a call and returns it; `ended(call, batch)` waits for its end and returns what it made, raising if it failed; `finish(batch,
    made)` is the rest. Until November 15, 2026 own_text.py finished a batch before it started the next call, so the card stood idle
    for the exports and the registrations (61.9 of pass 1's 273.7 s). What each `finish` returned comes back in the batches' order.
    Nothing about a batch's files changes; only when they are made does. If a batch's finishing fails while a call runs, the call is
    let end (`call.settle()`: the card is not left with a process this one no longer watches) and the failure is raised."""
    done, behind = [], None
    for batch in batches:
        call = start(batch)
        try:
            if behind is not None:
                done.append(finish(*behind))
        except BaseException:
            call.settle()
            raise
        behind = (batch, ended(call, batch))
    if behind is not None:
        done.append(finish(*behind))
    return done
