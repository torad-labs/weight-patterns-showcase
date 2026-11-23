// Whole-token navigation over one recorded globaltimer domain.
// Follow last-participating-block boundaries; retain overlap in the block view.
import {stepMetrics as sequenceMetrics} from "./internal-model.mjs";
export function decodeSequence(replay,includeFree=false){
  const layers=replay.layers.map(layer=>{
    const substeps=replay.steps.filter(s=>s.layer===layer.id);
    const steps=substeps.filter(s=>s.parent==null).map(step=>{
      const m=sequenceMetrics(step,includeFree);
      return {id:step.id,step,start:m.end-m.critical,end:m.end,earliest:m.start,blocks:m.blocks};
    }).filter(s=>s.blocks).sort((a,b)=>a.start-b.start||a.end-b.end);
    const start=Math.min(...steps.map(s=>s.start)),end=Math.max(...steps.map(s=>s.end));
    return {...layer,start,end,earliest:Math.min(...steps.map(s=>s.earliest)),steps,substeps};
  });
  const start=Math.min(...layers.map(l=>l.start)),end=Math.max(...layers.map(l=>l.end));
  const steps=layers.flatMap(l=>l.steps).sort((a,b)=>a.start-b.start||a.end-b.end);
  return {layers,steps,start,end,duration:end-start};
}
export function decodePosition(sequence,time){
  const complete=time>=sequence.end;
  const layer=sequence.layers.find(l=>time<l.end)??sequence.layers.at(-1);
  const gap=time<layer.start;
  const step=layer.steps.findLast(s=>s.start<=time&&s.end>time)??(complete?layer.steps.at(-1):null);
  return {layer:layer.id,stepId:step?.id??null,gap,complete};
}
export function seekDecodeStep(sequence,time,direction){
  return direction>0?sequence.steps.find(s=>s.start>time)?.start??sequence.end:
    sequence.steps.findLast(s=>s.start<time)?.start??sequence.start;
}
