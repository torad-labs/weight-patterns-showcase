"""Index complete native executions for individual inspection of A's two calculation passages.

All operations and all recorded input/output chunks in after_loss and destination
are retained, across every layer and every group member. Complete original captures
remain the source for other passages. This is retrieval scope, not an edit selector.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

from pipeline.native_records.group_observation import frames, publish
from pipeline.acquisition.run_a_baseline import ROOT, RUN


OUTPUT = RUN/'individual-observation'


def prepare(member_path):
    member_path = Path(member_path)
    member = json.loads(member_path.read_text())
    cid = member['member']
    destination = OUTPUT/cid
    if (destination/'selection.json').exists():
        return json.loads((destination/'selection.json').read_text())
    destination.mkdir(parents=True, exist_ok=False)
    annotation = json.loads((RUN/'response-correspondence/cases'/(cid+'.json')).read_text())
    context = json.loads((RUN/'context-index'/(cid+'.json')).read_text())
    roles = {s['index']:s['role'] for s in annotation['sections']}
    predictions = [t['prediction'] for t in annotation['tokens'] if roles.get(t['section']) in ('after_loss','destination')]
    selected = {context['prediction_executions'][str(p)] for p in predictions}
    archive = (ROOT/'records/A-pilot-raw-archive-2026-11-08.json' if cid in ('block-changed','block-base')
               else RUN/'cohort-archives'/(cid+'.json'))
    archive_record = json.loads(archive.read_text())
    native = Path(member['native_data'])
    archived = next(a for a in archive_record['files'] if (ROOT/a['path']).resolve()==native.resolve())
    chunk_map = {}
    found = set()
    event_count = 0
    selected_hash = hashlib.sha256()
    event_path = destination/'events.jsonl.zst'
    with event_path.open('xb') as compressed:
        compressor = subprocess.Popen(['zstd','-q','-c','-T1'], stdin=subprocess.PIPE, stdout=compressed)
        try:
            for execution, events in frames(member['events'], context['events_sha256']):
                if execution not in selected:
                    continue
                found.add(execution)
                for event in events:
                    line = (json.dumps(event,separators=(',',':'))+'\n').encode()
                    selected_hash.update(line)
                    compressor.stdin.write(line)
                    event_count += 1
                    snapshots = [event['output']] + [i.get('snapshot') for i in event['inputs'] if i]
                    for snapshot in snapshots:
                        if not snapshot:
                            continue
                        for chunk in snapshot.get('chunks',[]):
                            key = (chunk['offset'], chunk['bytes'])
                            if key in chunk_map and chunk_map[key]['sha256'] != chunk['sha256']:
                                raise ValueError('One source byte interval has conflicting recorded hashes')
                            chunk_map[key] = chunk
        finally:
            compressor.stdin.close()
            status = compressor.wait()
    if status or found != selected:
        raise ValueError('Selected complete executions could not be recovered')
    request = dict(version=1,source=dict(path=str(native.relative_to(ROOT/'runs')),bytes=archived['bytes'],sha256=archived['sha256']),
                   chunks=sorted(chunk_map.values(),key=lambda c:(c['offset'],c['bytes'])))
    publish(destination/'request.json',request)
    intervals = []
    for chunk in request['chunks']:
        a,b=chunk['offset'],chunk['offset']+chunk['bytes']
        if intervals and a<=intervals[-1][1]: intervals[-1][1]=max(intervals[-1][1],b)
        else: intervals.append([a,b])
    record = dict(case=cid,group=member_path.parent.parent.name,member_record=str(member_path),
        source_events_sha256=context['events_sha256'],selected_events_sha256=selected_hash.hexdigest(),
        selected_predictions=predictions,selected_executions=sorted(selected),events=event_count,chunks=len(chunk_map),
        unique_requested_bytes=sum(b-a for a,b in intervals),source_bytes=archived['bytes'],
        scope='Every recorded operation and input/output chunk in the complete after_loss and destination passages; all layers. Other passages remain in complete source recordings. No edit location or repair direction is selected.')
    publish(destination/'selection.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--case',action='append',dest='cases')
    args=p.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    paths=[]
    for group in ['gpu0-failure','gpu0-success','gpu1-failure','gpu1-success']:
        state=json.loads((RUN/'group-observation'/group/'state.json').read_text())
        if not state['complete']:raise ValueError('Group observation is incomplete')
        for cid in state['members']:
            if not args.cases or cid in args.cases: paths.append(RUN/'group-observation'/group/'members'/(cid+'.json'))
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(prepare,path) for path in paths]):
            r=future.result();records.append(r)
            print(json.dumps({k:r[k] for k in ['case','events','chunks','unique_requested_bytes','source_bytes']}),flush=True)
    publish(OUTPUT/'selection-index.json',dict(cases=records,complete_cohort=len(records)==45,
        unique_requested_bytes=sum(r['unique_requested_bytes'] for r in records),source_bytes=sum(r['source_bytes'] for r in records)))


if __name__=='__main__': main()
