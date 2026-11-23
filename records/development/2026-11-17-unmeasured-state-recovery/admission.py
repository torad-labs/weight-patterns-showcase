"""Admit a completed installed update whose fresh capture never started.

The older measurement is evidence for the installed transition, never a loss at
its endpoint. Native/master payload hashes and finite/layout checks remain the
existing FullState.restore contract, executed before fresh baseline acquisition.
"""
from pathlib import Path

FORMAT = 'installed-unmeasured-recovery/1'
RESERVE_FAILURE = 'OSError: complete capture/checkpoint reserve needs '
RECEIPT_KEYS = ('format', 'status', 'iteration', 'model_identity', 'base_identity',
                'inventory_identity', 'transition')


def validate(root, declaration, candidate, previous, checked, continuation):
    recovery = declaration['unmeasured_recovery']
    if (recovery.get('format') != FORMAT or 'start_fresh' in declaration
            or recovery.get('first_action') != 'fresh-native-baseline-before-any-proposal'):
        raise ValueError('unmeasured recovery requires its explicit first fresh baseline')
    installed = checked(root, recovery['installed'])
    proposal = checked(root, recovery['proposal'])
    stopped = checked(root, recovery['launcher_exit'])
    source = checked(root, recovery['predecessor_declaration'])
    old = declaration['provenance_last_measurement']
    iteration = candidate.get('iteration')
    identity = candidate.get('model_identity', {})
    transition = candidate.get('transition', {})
    expected = dict(identity, generation=iteration - 1) if type(iteration) is int else None
    folder = (root / recovery['predecessor_declaration']['record']).parent
    if (not continuation.is_full(candidate) or candidate.get('status') != 'complete'
            or type(iteration) is not int or iteration < 1
            or identity.get('generation') != iteration
            or identity.get('run') != str(folder.relative_to(root))
            or identity.get('original_sha256') != declaration['original_model']['sha256']
            or candidate.get('last_measurement') != old
            or previous.get('model_identity') != expected
            or previous.get('generation') != iteration - 1 or previous.get('step') != iteration - 1
            or previous.get('complete') is not True or previous.get('ended') is not True
            or candidate.get('loss_identity') != declaration['loss_identity']
            or previous.get('loss_identity') != declaration['loss_identity']
            or candidate.get('eta') != declaration['eta'] or source.get('eta') != candidate.get('eta')):
        raise ValueError('installed state and its preceding measured provenance differ')
    for key, record in [('final-state.json', declaration['resume_checkpoint']),
                        ('launcher-exit.json', recovery['launcher_exit']),
                        (f'loop/{iteration}/reached.json', recovery['installed']),
                        (f'loop/{iteration}/proposal.json', recovery['proposal']),
                        (f'loop/{iteration-1}/measurement.json', old)]:
        if root / record['record'] != folder / key:
            raise ValueError('recovery evidence must belong to the exact predecessor transition')
    if (any(installed.get(k) != candidate.get(k) for k in RECEIPT_KEYS)
            or installed.get('source_observation') != old
            or transition.get('source_identity') != expected
            or transition.get('next_identity') != identity
            or transition.get('iteration') != iteration or transition.get('in_progress') is not None
            or proposal.get('status') != 'ready' or proposal.get('source_observation') != old
            or proposal.get('source_model_identity') != expected
            or proposal.get('target_model_identity') != identity
            or proposal.get('loss_identity') != declaration['loss_identity']):
        raise ValueError('installed receipt/proposal do not prove the saved completed transition')
    names = declaration['functional_loop']['matrices']
    committed = transition.get('committed', {})
    if (len(names) != len(set(names)) or set(committed) != set(names)
            or any(row.get('complete') is not True for row in committed.values())):
        raise ValueError('every eligible tensor must have a complete installation receipt')
    reason = candidate.get('reason', '')
    if (not reason.startswith(RESERVE_FAILURE) or stopped.get('exit_code') != 1
            or stopped.get('reason') != reason or stopped.get('installed_updates') != iteration
            or stopped.get('state_status') != 'complete' or stopped.get('last_measurement') != old):
        raise ValueError('recovery is limited to the retained pre-capture storage reserve failure')
    if ((folder / f'loop/{iteration}/measurement.json').exists()
            or (folder / f'loop/{iteration}/capture').exists()):
        raise ValueError('predecessor endpoint already has capture data; do not label it unmeasured')
    historical = continuation.historical_step(source, candidate)
    if (declaration['counter_semantics']['historical_predecessor'] != historical
            or recovery.get('historical_installed_step') != historical
            or declaration.get('resume_predecessor_generation') != iteration):
        raise ValueError('installed cumulative/local counters differ')
    if (source['reference'] != declaration['reference']
            or source['tolerance'] != declaration['tolerance']
            or source['loss_identity'] != declaration['loss_identity']
            or source['original_model'] != declaration['original_model']):
        raise ValueError('recovery must preserve the exact reference/calibration/model')
    manifest = continuation.checkpoint_record(root, candidate, checked)
    if set(row['name'] for row in manifest['layout']) != set(names):
        raise ValueError('typed checkpoint does not contain the complete eligible inventory')
    for path, kind in zip(continuation.payloads(root, candidate, folder, checked)[1:], ('native', 'remainder')):
        if path.stat().st_size != manifest[kind]['bytes']:
            raise ValueError('saved typed payload is absent or truncated')
    return manifest
