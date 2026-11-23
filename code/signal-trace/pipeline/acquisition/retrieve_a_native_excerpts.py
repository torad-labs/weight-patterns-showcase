"""Retrieve A's exact individual passage recordings through torad-research."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime
import json
from itertools import zip_longest
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from pipeline.native_records.group_observation import publish
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT
from pipeline.acquisition.run_a_baseline import RUN
from pipeline.acquisition.run_a_cohort import reserve_space, release_space


def retrieve(directory):
    directory = Path(directory)
    selection = json.loads((directory/'selection.json').read_text())
    cid = selection['case']
    destination = directory/'native'
    key = 'native-excerpt:'+cid
    while not reserve_space(key, destination, selection['source_bytes']+4*1024**3):
        time.sleep(5)
    try:
        env = dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
        with (directory/'download.log').open('a') as log:
            subprocess.run(['torad-research','traces','extract',str(directory/'request.json'),'--to',str(destination)],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt=json.loads((destination/'excerpt.json').read_text())
        result=dict(case=cid,at_ct=datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(),bytes=receipt['packed_bytes'],
                    source_bytes=receipt['source']['bytes'],source_sha256=receipt['source']['sha256'],
                    complete_source=False,request_sha256=receipt['manifest_sha256'])
        publish(directory/'retrieved.json',result)
        return result
    finally:
        release_space(key)


def main():
    groups=[]
    for group in ['gpu0-failure','gpu0-success','gpu1-failure','gpu1-success']:
        groups.append(json.loads((RUN/'group-observation'/group/'state.json').read_text())['members'])
    expected=[cid for row in zip_longest(*groups) for cid in row if cid is not None]
    complete={c for c in expected if (OUTPUT/c/'retrieved.json').exists()}
    pending={}
    last_progress=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        while len(complete)<len(expected):
            for cid in expected:
                if len(pending)>=4:break
                if cid in complete or cid in pending.values() or not (OUTPUT/cid/'selection.json').exists():continue
                pending[pool.submit(retrieve,OUTPUT/cid)]=cid
                print(json.dumps(dict(state='retrieving',case=cid)),flush=True)
            if pending:
                done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    result=future.result();complete.add(pending.pop(future));last_progress=time.monotonic()
                    print(json.dumps(dict(state='retrieved',completed=len(complete),total=len(expected),**result)),flush=True)
            else:
                if time.monotonic()-last_progress>1800:raise RuntimeError('No completed passage preparation became available for 30 minutes')
                time.sleep(1)
    publish(OUTPUT/'retrieval-complete.json',dict(cases=expected,complete=True,new_inference=False,new_weight_edits=0))


if __name__=='__main__':main()
