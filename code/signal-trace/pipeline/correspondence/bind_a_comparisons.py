"""Attach response correspondences to verified native executions and history."""
import json
import re

from pipeline.acquisition.run_a_baseline import ROOT, RUN


def bind(pair, left, right, left_prompt, right_prompt):
    template_equal = re.sub(r'\d+', '<number>', left_prompt['prompt']) == re.sub(r'\d+', '<number>', right_prompt['prompt'])
    if not template_equal or len(left_prompt['tokens']) != len(right_prompt['tokens']):
        raise ValueError('This A binding requires the verified common prompt template; no guessed row map')
    prompt_pairs = [[i, i] for i in range(len(left_prompt['tokens']))]
    frames = []
    right_prefill = {tuple(f['input_positions']): f for f in right['frames']
                     if f['phase'] == 'prefill' and f['prediction'] is None}
    for frame in left['frames']:
        if frame['phase'] != 'prefill' or frame['prediction'] is not None:
            continue
        other = right_prefill.get(tuple(frame['input_positions']))
        if other is None:
            raise ValueError('Prefill partitions differ; explicit row-level remapping is needed')
        frames.append(dict(left_execution=frame['execution'], right_execution=other['execution'],
                           role='prefill', left_input_positions=frame['input_positions'],
                           right_input_positions=other['input_positions'],
                           reason='Same prompt-template row positions, independently verified against native token and rotary-position hashes; differing numeral values retained'))
    for prediction in pair['predictions']:
        lp, rp = prediction['left_prediction'], prediction['right_prediction']
        frames.append(dict(left_execution=left['prediction_executions'][str(lp)],
                           right_execution=right['prediction_executions'][str(rp)],
                           left_decision=lp, right_decision=rp, role=prediction['section'],
                           reason=prediction['reason'],
                           identical_incoming_history=prediction['identical_incoming_history']))
    return dict(relation=pair['relation'], left=pair['left'], right=pair['right'],
                left_gpu=pair.get('left_gpu', 0), right_gpu=pair.get('right_gpu', 0),
                same_hardware=pair.get('same_hardware', True),
                left_events_sha256=left['events_sha256'], right_events_sha256=right['events_sha256'],
                prompt_token_pairs=prompt_pairs, frames=frames,
                unmatched_left_predictions=pair['unmatched_left'], unmatched_right_predictions=pair['unmatched_right'],
                history_limit='A paired selected token enters cache only when subsequently supplied; actual writers/versions and masks determine each read')


def prepare():
    source=RUN/'response-correspondence'
    index=json.loads((source/'index.json').read_text())
    output=RUN/'execution-correspondence';output.mkdir(exist_ok=True)
    done=[];pending=[]
    for specification in index['pairs']:
        lp=RUN/'context-index'/(specification['left']+'.json')
        rp=RUN/'context-index'/(specification['right']+'.json')
        if not lp.exists() or not rp.exists():
            pending.append(specification['file']);continue
        pair=json.loads((source/specification['file']).read_text())
        prompts=[json.loads((ROOT/index['cases'][cid]['input']).read_text())[0]
                 for cid in [specification['left'],specification['right']]]
        record=bind(pair,json.loads(lp.read_text()),json.loads(rp.read_text()),*prompts)
        record['response_correspondence']=str((source/specification['file']).relative_to(ROOT))
        name=specification['file'].split('/')[-1]
        (output/name).write_text(json.dumps(record,indent=2)+'\n')
        done.append(dict(file=name,left=pair['left'],right=pair['right'],executions=len(record['frames'])))
    (output/'index.json').write_text(json.dumps(dict(bound=done,pending=pending),indent=2)+'\n')
    print(json.dumps(dict(bound=len(done),pending=len(pending))))


if __name__ == '__main__':
    prepare()
