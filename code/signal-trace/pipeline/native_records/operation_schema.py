"""Compare recorded operation/port topology; this is not a capability measure."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from pipeline.native_records.native_trace import events


def describe_frame(fid, frame):
    ids={e['event']:i for i,e in enumerate(frame)}
    schema=[]
    unresolved=0
    for e in frame:
        ports=[]
        for x in e['inputs']:
            if x is None:
                ports.append(None)
            elif 'weight' in x:
                ports.append({'weight':x['weight']['name']})
            elif x.get('cache_state') is not None:
                ports.append({'cache':x['cache_storage']})
            elif x.get('producer_event') in ids:
                ports.append({'producer':ids[x['producer_event']]})
            else:
                ports.append({'unresolved':x['layout']['name']})
                unresolved+=1
        schema.append({'name':e['name'],'operation':e['operation'],'ports':ports})
    digest=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
    return {'execution':fid,'phase':frame[0]['phase'],'events':len(frame),
            'ordered_operation_port_schema_sha256':digest,'unresolved_inputs':unresolved}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('events',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--scope',required=True)
    a=p.parse_args()
    rows=[describe_frame(fid,list(frame)) for fid,frame in itertools.groupby(events(a.events),key=lambda e:e['graph_execution'])]
    report={'scope':a.scope+'; matching architecture does not imply a capability pattern',
            'frames':rows,'distinct_decode_schemas':len({r['ordered_operation_port_schema_sha256'] for r in rows if r['phase']=='decode'})}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'frames':len(rows),'distinct_decode_schemas':report['distinct_decode_schemas']}))


if __name__=='__main__':
    main()
