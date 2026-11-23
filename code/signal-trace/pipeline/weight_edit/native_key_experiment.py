"""Prepare, inspect or run the installed learner with measured native pre-cache keys."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from pipeline.weight_edit import final_experiment as baseline, native_key_anchor, native_key_driver, native_key_finite, trace_diagnostic

MODE='final-experiment-native-key-anchored-training'
ORIGINAL_MODE='final-experiment-full-computational-training'
SIZING=baseline.ROOT/'records/development/2026-11-18-installed-native-key-anchor/sizing.json'
MODULE='pipeline.weight_edit.native_key_experiment'


def configuration(declaration):
    architecture=baseline.read(baseline.ROOT/declaration['architecture']['record'])
    tensors=baseline.read(baseline.ROOT/declaration['tensors']['record'])
    producers=architecture['layers']-architecture['layers_sharing_keys']
    count=declaration['native_context']
    calibration=baseline.read(SIZING)
    if producers!=calibration['layers'] or count!=calibration['context']:
        raise ValueError('Recalibrate native K metadata sizing for this producer inventory')
    packed_per_event=sum(4*tensors[f'blk.{layer}.attn_k.weight']['shape'][1] for layer in range(producers))
    payload=count*packed_per_event
    metadata=count*calibration['max_widened_sum_per_position']
    return dict(format='native-precache-key-anchoring/1',driver=baseline.binding(Path(native_key_driver.__file__)),
                wrapper=baseline.binding(Path(native_key_anchor.__file__)),sizing=baseline.binding(SIZING),
                capture_labels=native_key_driver.LABELS,producers=producers,
                max_context=count,native_f32_payload_bytes=payload,additional_index_allowance_bytes=metadata,
                additional_capture_allowance_bytes=payload+metadata,
                persistent_cpu_anchor_payload_bytes=2*payload,scope='Exact maximum K data payload for this context; '
                'index reserve scales the largest complete integer-widened metadata charge per position. Full training peak host/device memory remains unmeasured.')


def prepare(run, prior, review, intent, stopping=None, diagnostic=None, prefix_objective=None, evaluate_after=None,
            resume_evaluation_interval=False, *, development_origin=None):
    dec=baseline.prepare(run,prior,review,intent,stopping,diagnostic,prefix_objective,evaluate_after,resume_evaluation_interval,
                         development_origin=development_origin)
    anchor=configuration(dec)
    dec['operation']=MODE
    dec['native_key_anchor']=anchor
    bindings=[baseline.binding(Path(__file__)),anchor['driver'],anchor['wrapper'],
              baseline.binding(Path(native_key_finite.__file__)),baseline.binding(Path(trace_diagnostic.__file__))]
    if dec.get('trace_diagnostic'):
        dec['observability']['additional_metrics']=list(dict.fromkeys([*dec['observability']['additional_metrics'],native_key_finite.METRIC]))
    dec['implementation']=list({r['record']:r for r in [*dec['implementation'],*bindings]}.values())
    dec['observability']['entrypoint']=(dec.get('prefix_objective') or anchor)['driver']['record']
    dec['observability']['sources']=list({r['record']:r for r in [*dec['observability']['sources'],*bindings]}.values())
    dec['inputs']=sorted(set(dec['inputs'])|{baseline.read(SIZING)['measurement']['record']})
    storage=dec['storage']
    # Future continuations are measured from the ordinary controller's retained
    # residual budget; don't add the key payload a second time on resumption.
    ordinary=baseline.read(baseline.ROOT/storage['sizing']['record'])['capture_reserve_bytes']
    storage['capture_reserve_bytes']=ordinary+anchor['additional_capture_allowance_bytes']
    capacity=dec['parameters']['storage']['value']
    capacity['free_required_bytes']+=storage['capture_reserve_bytes']-capacity['capture_reserve_bytes']
    capacity['capture_reserve_bytes']=storage['capture_reserve_bytes']
    dec['parameters']['storage']['derivation']+=' Include native K payload and an index allowance scaled to the measured all-producer capture.'
    dec['commands']=[shlex.join(['env', f"PYTHONPATH={baseline.ENV['PYTHONPATH']}", *baseline.RUNTIME,
                                '-m',MODULE,'run', str(run)])]
    return dec


def check(run,dec):
    if dec.get('operation')!=MODE or dec.get('program')!=baseline.binding(baseline.MEASURED):
        raise ValueError('A measured native-key continuation is required')
    anchor=configuration(dec)
    if dec.get('native_key_anchor')!=anchor:
        raise ValueError('Changed native key implementation, architecture or storage calibration')
    ordinary=baseline.read(baseline.ROOT/dec['storage']['sizing']['record'])['capture_reserve_bytes']
    if dec['storage']['capture_reserve_bytes']!=ordinary+anchor['additional_capture_allowance_bytes']:
        raise ValueError('The capture reserve omits or double-counts the native keys')
    cap=dec['parameters']['storage']['value']
    if cap['capture_reserve_bytes']!=dec['storage']['capture_reserve_bytes'] or cap['free_required_bytes']!=sum(
            cap[k] for k in ('native_checkpoint_bytes','remainder_checkpoint_bytes','capture_reserve_bytes','free_floor_bytes')):
        raise ValueError('Checkpoint preflight and native-key reserve disagree')
    return baseline.inspect_run(run,{**dec,'operation':ORIGINAL_MODE})|{'native_key_anchor':anchor}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    cmd=parser.add_subparsers(dest='action',required=True)
    prep=cmd.add_parser('prepare')
    prep.add_argument('run',type=Path)
    prep.add_argument('--from-run',type=Path,required=True)
    prep.add_argument('--review',type=Path,required=True)
    prep.add_argument('--intent',type=Path,required=True)
    prep.add_argument('--stopping-policy',type=Path)
    prep.add_argument('--trace-diagnostic',type=Path)
    prep.add_argument('--prefix-objective',type=Path,help='train toward a frozen common-prefix turn displacement')
    prep.add_argument('--evaluate-after',type=Path,help='exit at the evaluation interval derived from bound fitted history')
    prep.add_argument('--resume-evaluation-interval',action='store_true',
                      help='complete the interrupted prefix evaluation interval from its last measured state')
    prep.add_argument('--write',action='store_true')
    validate=cmd.add_parser('check')
    validate.add_argument('run',type=Path)
    validate.add_argument('--declaration',type=Path)
    cmd.add_parser('run').add_argument('run',type=Path)
    args=parser.parse_args()
    run=baseline.run_path(args.run)
    if args.action=='prepare':
        dec=prepare(run,baseline.run_path(args.from_run),args.review,baseline.read(args.intent),
                    args.stopping_policy,args.trace_diagnostic,args.prefix_objective,args.evaluate_after,
                    args.resume_evaluation_interval)
        if args.write:
            check(run,dec)
            encoded=json.dumps(dec,indent=2,allow_nan=False)+'\n'
            run.mkdir()
            (run/'declaration.json').write_text(encoded)
            print(run/'declaration.json')
        else:
            print(json.dumps(dec,indent=2))
        return
    dec=baseline.read(getattr(args,'declaration',None) or run/'declaration.json')
    receipt=check(run,dec)
    if args.action=='check':
        print(json.dumps(receipt,indent=2))
        return
    if any((run/file).exists() for file in ('working-state.json','progress.jsonl','loop','launcher-exit.json','observability','owner.lock')):
        parser.error('Continue a used output from its measured checkpoint in a new run directory.')
    if not receipt['storage_ready']:
        parser.error(f"Native-key continuation needs {receipt['storage_needed_bytes']} bytes; {receipt['storage_available_bytes']} available.")
    os.chdir(baseline.ROOT)
    entry=(dec.get('prefix_objective') or dec['native_key_anchor'])['driver']['record']
    os.execve(sys.executable,[sys.executable,str(baseline.ROOT/entry),str(run)],os.environ|baseline.ENV)


if __name__=='__main__':
    main()
