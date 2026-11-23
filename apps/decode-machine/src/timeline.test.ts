import {describe,expect,test} from 'bun:test';
import {Timeline,LAYER_COUNT,PROMPT_TOKENS,storyStages,operationActive,samplePath} from './timeline';
import {CARRIER_LABELS,carrierState,carrierPassPosition,weightBank} from './representation';
import {ConnectedJourney,journeyPartState} from './connected-journey';
describe('scripted question-to-answer journey',()=>{
 test('Gemma retains input, all 35 layers and answer while inspection uses the same reversible story position',()=>{
  const j=new ConnectedJourney(35),t=j.timeline;
  for(let layer=0;layer<35;layer++){t.seek(.11+(.43-.11)*(layer+.4)/35);expect(t.layerIndex).toBe(layer);expect(journeyPartState(t,layer,'attention').active).toBe(true);}
  j.inspect(4,'attention');const position=t.progress;j.tick(6);expect(j.progress).toBeCloseTo(.5);expect(t.layerIndex).toBe(4);expect(t.progress).toBeGreaterThan(position);
  const inside=t.progress;j.continue();expect(t.progress).toBe(inside);j.tick(1);expect(t.progress).toBeGreaterThan(inside);
  t.seek(.75);expect(t.answer).toBe('Paris.');j.inspect(34,'down');expect(t.layerIndex).toBe(34);j.setDuration(20);j.tick(20);expect(j.progress).toBeCloseTo(1);expect(t.playing).toBe(false);
  j.continue();t.seek(.99);expect(t.answer).toBe('Paris.');t.seek(.01);expect(t.answer).toBe('');expect(t.currentToken).toBe('question');
 });
 test('prefill predicts Paris, feeding Paris predicts punctuation, then the end signal stops playback',()=>{
  const t=new Timeline();
  t.seek(.44);expect(t.stage.id).toBe('choose-paris');expect(t.answer).toBe('');expect(t.cacheCount(31)).toBe(PROMPT_TOKENS.length);
  t.seek(.49);expect(t.answer).toBe('Paris');expect(t.currentToken).toBe('Paris');
  t.seek(.70);expect(t.stage.id).toBe('choose-period');expect(t.answer).toBe('Paris');expect(t.cacheCount(31)).toBe(PROMPT_TOKENS.length+1);
  t.seek(.76);expect(t.answer).toBe('Paris.');
  t.seek(.95);expect(t.stage.id).toBe('choose-end');expect(t.cacheCount(31)).toBe(PROMPT_TOKENS.length+2);
  t.playing=true;t.step(120);expect(t.progress).toBe(1);expect(t.playing).toBe(false);expect(t.answer).toBe('Paris.');expect(t.stage.id).toBe('done');
  t.seek(.02);expect(t.answer).toBe('');expect(t.cacheCount(0)).toBe(0);
 });
 test('all stages and parallel work share the chosen total duration, and pause and seeking preserve state',()=>{
  for(const seconds of [1,10,30,60,120,300,3600]){
   const t=new Timeline();t.setDuration(seconds);t.step(seconds*.2);expect(t.progress).toBeCloseTo(.2,12);
   t.playing=false;const before=t.progress;t.setDuration(300);t.step(10);expect(t.progress).toBe(before);
   for(const stage of storyStages.filter(s=>s.pass!==null)){
    t.seek(stage.start+(stage.end-stage.start)*(16+.34)/LAYER_COUNT);
    expect(t.layerIndex).toBe(16);expect(t.layerActive).toBe(true);expect(t.candidatesReady).toBe(false);
    expect(['head-a','head-b','head-c'].every(id=>operationActive(id as 'head-a',t.layerProgress))).toBe(true);
    expect(operationActive('join',t.layerProgress)).toBe(false);
    t.next();expect(t.phase.name).toBe('Join');
   }
  }
 });
 test('cache retention and answer reconstruction depend on position, including reverse scrubbing',()=>{
  const t=new Timeline();
  for(const [start,end,base]of [[.11,.43,0],[.54,.70,7],[.81,.94,8]])for(let layer=0;layer<LAYER_COUNT;layer++){
   t.seek(start+(end-start)*(layer+.34)/LAYER_COUNT);expect(t.cacheCount(layer)).toBe(base===0?7:base+1);
   if(layer+1<LAYER_COUNT)expect(t.cacheCount(layer+1)).toBe(base);
  }
  t.seek(.9);expect(t.answer).toBe('Paris.');expect(t.candidatesReady).toBe(false);t.seek(.95);expect(t.candidatesReady).toBe(true);t.seek(.3);expect(t.answer).toBe('');expect(t.cacheCount(31)).toBe(0);
  t.seekLayer(31);expect(t.layerIndex).toBe(31);expect(t.playing).toBe(false);
  t.setDuration(NaN);expect(t.durationSeconds).toBe(120);t.seek(NaN);expect(Number.isFinite(t.progress)).toBe(true);
 });
 test('carriers travel by distance along unequal pipe segments',()=>{expect(samplePath([[0,0,0],[1,0,0],[1,3,0]],.5)).toEqual([1,1,0]);});
 test('a carrier keeps its identity while its layer values change, with continuous handoffs and reversible time',()=>{
  const t=new Timeline();
  for(let token=0;token<CARRIER_LABELS.length;token++){
   t.seek(carrierPassPosition(token,12,.1));const first=carrierState(t,token),position=t.progress;
   expect(first.label).toBe(CARRIER_LABELS[token].trim());expect(first.values).not.toEqual(first.input);
   carrierState(t,(token+1)%9);expect(t.progress).toBe(position);
   t.seek(carrierPassPosition(token,12,.8));expect(carrierState(t,token).values).not.toEqual(first.values);
   t.setDuration(3600);t.seek(position);expect(carrierState(t,token)).toEqual(first);
   t.seek(carrierPassPosition(token,12,1-1e-8));const end=carrierState(t,token);
   t.seek(carrierPassPosition(token,13,0));const next=carrierState(t,token);
   end.values.forEach((v,k)=>expect(v).toBeCloseTo(next.input[k],6));
   expect(next.layer).toBe(13);
  }
  t.seek(.6);const retained=carrierState(t,5);expect(retained.finished).toBe(true);expect(retained.label).toBe('France');expect(carrierState(t,7).finished).toBe(false);
  t.seek(.9);expect(carrierState(t,5).values).toEqual(retained.values);
  t.seek(.01);expect(carrierState(t,5).assembled).toBe(false);
 });
 test('each layer owns distinct stored weights, unchanged across passes and duration changes',()=>{
  const banks=Array.from({length:LAYER_COUNT},(_,i)=>weightBank(i));
  expect(new Set(banks.flat().map(w=>w.id)).size).toBe(384);
  expect(new Set(banks.map(bank=>JSON.stringify(bank.map(w=>w.value)))).size).toBe(32);
  const t=new Timeline();for(const progress of [.2,.6,.9,.1]){t.seek(progress);t.setDuration(300);expect(weightBank(12)).toEqual(banks[12]);}
 });
});
