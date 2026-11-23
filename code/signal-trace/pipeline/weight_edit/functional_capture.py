"""A functional loop's acquisition belongs to its actual complete reached state."""
import json
from pathlib import Path
from typing import Any

from . import functional_install, working_identity
from .functional_reading import digest
from .functional_reference import checked


def working_state(root, declared, declaration, number, *, model=None):
    """The ready revision of this run's single native working file."""
    configuration = declaration.get('working_model')
    if configuration is None:
        return None
    if not isinstance(configuration, dict) or configuration.get('format') != working_identity.FORMAT:
        raise ValueError('unknown declared working-model format')
    if type(number) is not int or number < 1:
        raise ValueError('a working capture needs its numbered observation')
    root, declared = Path(root).resolve(), Path(declared).resolve()
    file = declared / 'working/model.gguf'
    if model is not None and Path(model).absolute() != file:
        raise ValueError('the native reader must use the declared single working file')
    state = working_identity.load_state(declared / 'working/state.json', model=file)
    if type(state['generation']) is not int or state['generation'] != number - 1:
        raise ValueError('the ready working generation is not this observation\'s state')
    original = declaration['original_model']
    if (state['original_file'] != str((root / original['record']).resolve())
            or state['model_identity']['original_sha256'] != original['sha256']
            or state['model_identity']['run'] != str(declared.relative_to(root))):
        raise ValueError('the working state is not based on the declared original')
    return state


def admitted(root, declared, declaration, cohort_run, model_sha256, before, out, number,
             *, case_manifest=None, model_identity=None, model=None) -> dict[str, Any]:
    root, declared, out = Path(root).resolve(), Path(declared).resolve(), Path(out).resolve()
    specification = declaration.get('functional_loop')
    if not isinstance(specification, dict) or type(number) is not int or number < 1:
        raise ValueError('a functional capture needs its declared loop and numbered state')
    if out != declared / 'loop' / str(number) / 'capture/own':
        raise ValueError('the capture must be in this functional state\'s declared output')
    if (root / specification['cohort']).resolve() != Path(cohort_run).resolve():
        raise ValueError('the capture cohort is not the declared fitting cohort')
    if before != specification['history_before']:
        raise ValueError('the capture history is not the declared acquisition reference')
    protocol = checked(root, specification['reader_protocol'])
    selection = {}
    bound_cases = specification.get('cases')
    if bound_cases is not None:
        if not case_manifest or Path(case_manifest).resolve() != (root / bound_cases['record']).resolve():
            raise ValueError('the capture must use its declared fitting manifest')
        cases = checked(root, bound_cases)
        members = specification.get('training_members')
        if (cases.get('format') != 'whole-gap-cases/1' or cases.get('role') != 'fitting-named'
                or cases.get('cohort') != Path(cohort_run).name):
            raise ValueError('the capture manifest must name fitting answers from its declared cohort')
        if (not isinstance(members, list) or not members or len(set(members)) != len(members)
                or cases.get('ids') != members):
            raise ValueError('the capture members are not exactly the declared training members')
        if (protocol.get('cases', {}).get('role') != 'fitting-named'
                or protocol['cases'].get('manifest_sha256') != bound_cases['sha256']):
            raise ValueError('the reader protocol must bind the same fitting manifest')
        selection = dict(training_members=members, cases_sha256=bound_cases['sha256'])
    elif case_manifest:
        raise ValueError('an undeclared manifest cannot replace the fitting population')
    for path, sha256 in protocol['instruments'].items():
        checked(root, dict(record=path, sha256=sha256), parse=False)
    working = working_state(root, declared, declaration, number, model=model)
    if working is not None:
        if working_identity.KIND not in protocol.get('model_identity_formats', []):
            raise ValueError('the reader protocol must declare working-model identity support before acquisition')
        if model_sha256 is not None or model_identity != working['model_identity']:
            raise ValueError('the captured model needs its complete ready revision identity')
        return dict(pass_number=number,
                    starts_from=dict(**working_identity.fields(model_identity), state=model_identity,
                                     file=str(Path(working['model_file']).relative_to(root))), functional_state=True,
                    reader_protocol_sha256=specification['reader_protocol']['sha256'], **selection)
    if model_identity is not None:
        raise ValueError('a typed model identity cannot replace a legacy file sha256')
    if number == 1:
        source = specification['first_state']
        state = 'the declared initial state'
        checked(root, dict(record=source['file'], sha256=source['model_sha256']), parse=False)
    else:
        pointer = declared / 'loop' / str(number - 1) / 'reached.json'
        if not pointer.is_file():
            raise ValueError('no complete preceding reached state is published')
        source = json.loads(pointer.read_text())
        if source.get('format') != 'functional-reached-state/1' or source.get('step') != number - 1:
            raise ValueError('the preceding reached state belongs to another step')
        file = (root / source['file']).resolve()
        if not file.is_relative_to(declared):
            raise ValueError('a reached model must remain inside its declared run')
        if digest(functional_install.manifest(file)) != source['joint_sha256']:
            raise ValueError('the reached joint manifest differs from its published sha256')
        joint = functional_install.load(file, source['model_sha256'], specification['matrices'])
        state = joint['kept']['identity']
    if model_sha256 != source['model_sha256']:
        raise ValueError('the captured model is not this state\'s declared model')
    return dict(pass_number=number, starts_from=dict(model_sha256=model_sha256, state=state, file=source['file']),
                functional_state=True, reader_protocol_sha256=specification['reader_protocol']['sha256'], **selection)
