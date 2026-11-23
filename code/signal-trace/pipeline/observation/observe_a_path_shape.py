"""Describe the geometry of each recorded residual progression in its local frame.

No common absolute direction across questions is assumed. All native vectors
remain in the linked branch views. These scalar descriptors help inspect the
connected paths; they cannot identify a capability or specify weight edits alone.
"""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from pipeline.native_records.group_observation import publish
from pipeline.observation.inspect_a_connected_progressions import View,branches,OUTPUT


FIELDS=['turn_angle_radians','relative_step_norm','step_incoming_cosine','previous_turn_cosine']


def collect(cid):
    out=OUTPUT/cid/'path-shape'
    if (out/'index.json').exists() and json.loads((out/'index.json').read_text()).get('version')==2:return cid
    out.mkdir(exist_ok=True)
    v=View(cid);records=[];values=[];connections=None
    for p,frame in v.frames.items():
        cs=branches(v.template(p));steps=[]
        for c in cs:
            steps.append(dict(layer=c['layer'],kind=c['kind'],incoming=c['incoming'],output=c['merged']))
            if c['kind']=='per_layer_embedding':
                steps.append(dict(layer=c['layer'],kind='layer_output_scale',incoming=c['merged'],output=c['layer_exit']))
        if connections is None:connections=steps
        assert connections==steps
        a=np.stack([v.vector(p,s['incoming']) for s in steps]);b=np.stack([v.vector(p,s['output']) for s in steps])
        norm=lambda x:np.linalg.norm(x,axis=1)
        na,nb=norm(a),norm(b);delta=b-a;nd=norm(delta)
        au=a/np.maximum(na[:,None],1e-30);bu=b/np.maximum(nb[:,None],1e-30);turn=bu-au
        angle=np.arccos(np.clip(np.einsum('ij,ij->i',au,bu),-1,1))
        relative=nd/np.maximum(na,1e-30)
        parallel=np.divide(np.einsum('ij,ij->i',delta,a),nd*na,out=np.full_like(nd,np.nan),where=nd*na!=0)
        nt=norm(turn);previous=np.full_like(nt,np.nan)
        denominator=nt[1:]*nt[:-1]
        # A scalar layer scale can change length without turning. Do not assign
        # an angle to its float32 rounding residue.
        valid=(nt[1:]>8*np.finfo(np.float32).eps)&(nt[:-1]>8*np.finfo(np.float32).eps)
        previous[1:]=np.divide(np.einsum('ij,ij->i',turn[1:],turn[:-1]),denominator,out=np.full_like(denominator,np.nan),where=valid)
        values.append(np.stack([angle,relative,parallel,previous],axis=1).astype('<f4'))
        records.append(dict(prediction=p,execution=frame['execution'],role=frame['role'],piece=frame['piece']))
    np.save(out/'shape.npy',np.asarray(values))
    publish(out/'index.json',dict(version=2,case=cid,records=records,connections=connections,fields=FIELDS,
                                 source=str(v.path/'view.json'),
                                 definition='Each ADD and intervening layer-output scaling is an actual directed step. Angle and length use its own incoming/output pair. Previous-turn cosine follows the successive recorded residual path in the shared native hidden coordinates.',
                                 limit='Geometric descriptors of the recorded progression, not a complete capability signature, semantic axis, proof of causation or edit target.'))
    return cid


def main():
    cases=sorted(p.parent.parent.name for p in OUTPUT.glob('*/branch-view/view.json'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(collect,c) for c in cases]):print(f.result(),flush=True)


if __name__=='__main__':main()
