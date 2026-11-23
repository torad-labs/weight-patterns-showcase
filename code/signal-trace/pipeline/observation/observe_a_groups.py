"""Run A's four separate within-group observations on saved natural executions.

No model execution, forced history, fitting, or changed weights. R2 restoration
uses torad-research; only a restored copy with an already verified archive can
be released after its complete contribution. Pre-existing local files stay local.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from pipeline.native_records.group_observation import GroupObservation, publish
from pipeline.acquisition.run_a_baseline import ROOT, RUN
from pipeline.acquisition.run_a_cohort import reserve_space, release_space


OUTPUT = RUN/'group-observation'


def note(state, **fields):
    row = dict(at_ct=datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(), state=state, **fields)
    text = json.dumps(row)
    with (OUTPUT/'progress.jsonl').open('a') as f:
        f.write(text+'\n')
    print(text, flush=True)


@contextmanager
def storage_reservation(output, required):
    """Share capacity reservations with capture and other analysis workers."""
    key = 'observation:'+str(output)
    waiting = False
    required += 4*1024**3  # Metadata, filesystem overhead and free operating space.
    while not reserve_space(key, output, required):
        if not waiting:
            note('waiting_for_analysis_space', output=str(output), required_bytes=required)
            waiting = True
        time.sleep(10)
    try:
        yield
    finally:
        release_space(key)


def groups():
    index = json.loads((RUN/'response-correspondence/index.json').read_text())
    cases = {c['id']: c for c in json.loads((ROOT/'research/experiments/signal-trace-repair/A-cohort.json').read_text())['cases']}
    result = {}
    for p in index['pairs']:
        if p['relation'] not in ('failure_overlay', 'natural_numerical_success_overlay'):
            continue
        if not p['same_hardware'] or p['left_gpu'] != p['right_gpu']:
            raise ValueError('A within-group observation crossed physical cards')
        label = f"gpu{p['left_gpu']}-"+('failure' if p['relation']=='failure_overlay' else 'success')
        g = result.setdefault(label, dict(label=label, gpu=p['left_gpu'], role=p['relation'], reference=p['left'], members=[]))
        if g['reference'] != p['left']:
            raise ValueError('Multiple index references in one group')
        c = cases[p['right']]
        if c.get('gpu', 0)!=g['gpu'] or c['role'] not in ('failure','natural_numerical_success'):
            raise ValueError('Construction membership or same-card baseline differs')
        g['members'].append(dict(case=c, correspondence=str(RUN/'execution-correspondence'/Path(p['file']).name)))
    if sum(len(g['members']) for g in result.values()) != 45:
        raise ValueError('The frozen 18 failure / 27 success population was changed')
    return result, cases


def source(case):
    cid = case['id']
    metadata = json.loads((RUN/'context-index/compressed-events'/(cid+'.json')).read_text())
    audit = json.loads((ROOT/case['audit']).read_text())
    context = json.loads((RUN/'context-index'/(cid+'.json')).read_text())
    if not (audit['tokens_and_all_score_hashes_equal'] and audit['recorded_producer_and_cache_links_valid']):
        raise ValueError('An unqualified source cannot enter a group observation')
    if metadata['source_sha256'] != audit['events_sha256'] or context['events_sha256'] != audit['events_sha256']:
        raise ValueError('Source audit and native context index disagree')
    return ROOT/metadata['compressed'], ROOT/metadata['native_tensor_data'], metadata, audit


def ensure_native(case, native):
    if native.exists():
        prefetch = OUTPUT/'prefetched-native.json'
        owned = json.loads(prefetch.read_text()) if prefetch.exists() else []
        return case['id'] in owned
    archive_path = RUN/'cohort-archives'/(case['id']+'.json')
    if case['id'] in ('block-changed','block-base'):
        archive_path = ROOT/'records/A-pilot-raw-archive-2026-11-08.json'
        archive = json.loads(archive_path.read_text())
        if archive['status'] != '8 files uploaded and remote byte readback verified':
            raise ValueError('The pilot source archive is not verified')
    else:
        archive = json.loads(archive_path.read_text())
        if archive['verify_exit_code'] != 0 or archive['input_sha256'] != case['input_sha256']:
            raise ValueError('The source archive is not verified for this question')
    entry = next(e for e in archive['files'] if ROOT/e['path']==native)
    env = dict(os.environ, XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',
               DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
    with storage_reservation(native.parent, entry['bytes']):
        note('restoring_native_recording', case=case['id'], bytes=entry['bytes'])
        with (OUTPUT/(case['id']+'-download.log')).open('a') as log:
            subprocess.run(['torad-research','traces','download',str(native)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if native.stat().st_size != entry['bytes']:
            raise ValueError('Restored source length differs from the archive manifest')
    note('native_recording_restored', case=case['id'], bytes=entry['bytes'])
    return True


def run_group(specification, cases, first_member_only=False):
    label = specification['label']
    output = OUTPUT/label
    reference = cases[specification['reference']]
    events, _, _, audit = source(reference)
    if output.exists():
        observation = GroupObservation(output)
        if observation.state['complete']:
            note('group_already_complete', group=label)
            return
    else:
        note('indexing_group_reference', group=label, reference=reference['id'])
        identity = dict(gpu=specification['gpu'], role=specification['role'], reference=reference['id'],
                        expected_members=[m['case']['id'] for m in specification['members']],
                        reference_events_sha256=audit['events_sha256'],
                        method='research/experiments/signal-trace-repair/A-RECURRENCE.md')
        observation = GroupObservation.create(output, events, identity, storage=storage_reservation)
        note('group_reference_indexed', group=label, vector_elements=observation.layout['elements'],
             locations=observation.layout['locations'])
    members = specification['members'][:1] if first_member_only else specification['members']
    for member in members:
        case = member['case']; cid=case['id']
        if cid in observation.state['members']:
            continue
        events, native, metadata, audit = source(case)
        restored = ensure_native(case, native)
        mapping = json.loads(Path(member['correspondence']).read_text())
        if mapping['right_events_sha256'] != audit['events_sha256']:
            raise ValueError('Execution correspondence names a different source recording')
        note('observing_member', group=label, case=cid,
             completed_members=len(observation.state['members']), members=len(specification['members']))
        observation.add(cid, events, native, mapping, source=dict(case=case, audit=case['audit'],
            compressed_events=metadata, original_graph_and_native_vectors='All original inputs, outputs, native weight bindings and producer/cache references retained at their recorded event IDs'),
            progress=lambda **kw: note('member_progress', group=label, **kw))
        note('member_observed', group=label, case=cid, completed_members=len(observation.state['members']),
             members=len(specification['members']))
        # Only this analysis's restored working copies enter this branch. The
        # verified archive and exact source paths/event IDs/chunk hashes remain.
        if restored:
            native.unlink()
            note('restored_working_copy_released', case=cid)
    if len(observation.state['members'])==len(specification['members']):
        report = observation.finish()
        note('group_observation_complete', group=label, members=len(report['members']),
             observed_locations=report['observed_locations'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--workers', type=int, default=1, choices=(1,2,3,4),
                        help='Independent group processes; all share disk-space reservations')
    parser.add_argument('--first-member-only', action='store_true',
                        help='Process the reference member; explicitly partial, never a completed group')
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    available, cases = groups()
    selected = args.groups or ['gpu0-failure','gpu0-success','gpu1-failure','gpu1-success']
    if len(set(selected))!=len(selected) or any(x not in available for x in selected):
        raise ValueError('Each existing group may run once')
    publish(OUTPUT/'group-index.json', dict(groups=available, method='Separate observed groups before contrasts',
            full_source_case_count=47, eligible_cases=45, other_error_cases=2,
            scope='Observation only; no new model response, fitted edit, or selected layer'))
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_group, available[g], cases, args.first_member_only) for g in selected]
        for future in futures:
            future.result()
    note('selected_group_pass_finished', groups=selected, partial=args.first_member_only, new_weight_edits=0)


if __name__=='__main__':
    main()
