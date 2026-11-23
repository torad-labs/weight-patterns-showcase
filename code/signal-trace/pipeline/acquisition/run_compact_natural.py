"""Capture natural responses in bounded batches, qualifying and compacting each.

The declared prompts, cards, settings and recorder are unchanged. Each response
must match its own plain natural tokens and every score hash before its working
stream is released. All105 connected turns and gains remain available.
"""
import argparse
import json
from pathlib import Path

from pipeline.acquisition.run_a_observation import run
from pipeline.acquisition.compact_release import release_stream
from pipeline.native_records.group_observation import publish
from pipeline.observation.export_a_edited import export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
    parser.add_argument('--batch-size',type=int,choices=[1,2,3,4],default=3)
    args=parser.parse_args();out=args.run_dir.resolve()
    declaration=json.loads((out/'declaration.json').read_text())
    members={m['case']:m for m in declaration['members'] if m['gpu']==args.gpu}
    source=out/f'natural-stream-inputs-gpu{args.gpu}.json'
    requests=json.loads(source.read_text())
    if len(requests)!=len(members) or {r['id'] for r in requests}!=set(members):
        raise ValueError('Batching must cover every declared question exactly once on its card')
    natural=out/f'natural-gpu{args.gpu}';natural.mkdir(parents=True,exist_ok=True)
    executions=[];rows=[]
    for start in range(0,len(requests),args.batch_size):
        batch=requests[start:start+args.batch_size]
        path=out/f'acquisition-gpu{args.gpu}'/f'batch-{start:03d}'
        inputs=path.with_suffix('.json');inputs.parent.mkdir(parents=True,exist_ok=True)
        if inputs.exists() and json.loads(inputs.read_text())!=batch:
            raise ValueError('Existing batch has different source requests')
        if not inputs.exists():publish(inputs,batch)
        if not (path/'execution.json').exists():
            run(Path(declaration['checkpoint']),inputs,path,args.gpu,'natural','capture',declaration['retention'])
        execution=json.loads((path/'execution.json').read_text())
        if execution['cases']!=[r['id'] for r in batch] or execution['gpu']!=args.gpu:
            raise ValueError('Completed batch has a different population or card')
        executions.append(str(path/'execution.json'))
        for request in batch:
            cid=request['id'];source_case=path/cid;link=natural/cid
            if link.exists() and link.resolve()!=source_case.resolve():
                raise ValueError('Natural case is already bound to another execution')
            if not link.exists():link.symlink_to(source_case,target_is_directory=True)
            row=export(members[cid],out)
            release_stream(link/'stream.f32',out/'natural-turns'/cid)
            rows.append(row)
        publish(natural/'batch-progress.json',dict(completed=[r['case'] for r in rows],executions=executions))
    publish(natural/'execution.json',dict(status='complete',kind='natural',gpu=args.gpu,
        checkpoint=declaration['checkpoint'],cases=[r['id'] for r in requests],executions=executions,
        retention=declaration['retention'],scope='Same-card native natural batches; complete tokens and scores qualified against each plain run. Working streams released after all105 connected turns and norms were exported.'))
    print(json.dumps(dict(gpu=args.gpu,completed=len(rows),predictions=sum(r['predictions'] for r in rows))),flush=True)


if __name__=='__main__':main()
