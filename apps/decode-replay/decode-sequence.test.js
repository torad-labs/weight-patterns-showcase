import {test,expect} from "bun:test";
import {normalizeInternalReplay} from "./internal-model.mjs";
import {decodeSequence,decodePosition,seekDecodeStep} from "./decode-sequence.mjs";
const fixture=(count=3,shift=0)=>{
  const layers=Array.from({length:count},(_,i)=>({id:String(i),kind:i%2?"F":"S"}));
  const steps=layers.flatMap((l,i)=>[0,1].map(j=>({id:`l${i}:s${j}`,layer:l.id,parent:null,predecessors:j?[`l${i}:s0`]:[],name:j?"FFN":"Attention",kind:"work",fromTap:"A",toTap:"B",observations:[{block:"0",sm:"2",startNs:String(shift+i*300+j*100),endNs:String(shift+i*300+(j+1)*100),cycles:"100",participating:true}]})));
  steps.push({...steps[0],id:"inner",parent:steps[0].id,name:"Score product",observations:[{...steps[0].observations[0],startNs:String(shift+20),endNs:String(shift+80)}]});
  return normalizeInternalReplay({schema:"qgre.phase-replay.v1",clock:"globaltimer",layers,steps});
};
test("whole decode includes every layer and uses one capture clock, not summed envelopes",()=>{
  const s=decodeSequence(fixture());
  expect(s.layers.map(l=>l.id)).toEqual(["0","1","2"]);
  expect([s.start,s.end,s.duration]).toEqual([0,800,800]);
  expect(s.steps.length).toBe(6);
  expect(s.layers[0].substeps.length).toBe(3);
  expect(s.layers[1].start).toBe(300);
  expect(decodePosition(s,250)).toMatchObject({layer:"1",gap:true,complete:false});
});
test("playback crosses layers, follows internal steps, and stops at the final layer",()=>{
  const s=decodeSequence(fixture());
  expect(decodePosition(s,50)).toMatchObject({layer:"0",stepId:"l0:s0",complete:false});
  expect(decodePosition(s,350)).toMatchObject({layer:"1",stepId:"l1:s0",complete:false});
  expect(decodePosition(s,750)).toMatchObject({layer:"2",stepId:"l2:s1",complete:false});
  expect(decodePosition(s,800)).toMatchObject({layer:"2",complete:true});
  expect(seekDecodeStep(s,100,1)).toBe(300);
  expect(seekDecodeStep(s,300,-1)).toBe(100);
});
test("generated token lengths visit every declared layer without resetting the clock",()=>{
  for(let count=1;count<=48;count++){
    const shift=count*10003,s=decodeSequence(fixture(count,shift));
    expect(s.layers.length).toBe(count);
    for(let i=0;i<count;i++)expect(decodePosition(s,shift+i*300+50).layer).toBe(String(i));
    expect(s.end-s.start).toBe((count-1)*300+200);
    expect(decodePosition(s,s.end).complete).toBe(true);
  }
});
test("bystander early arrival does not jump the followed layer ahead of its completion",()=>{
  const r=fixture();r.steps[2].observations.push({...r.steps[2].observations[0],block:"1",start:50,end:70,startNs:"50",endNs:"70"});
  const s=decodeSequence(r);
  expect(decodePosition(s,60).layer).toBe("0");
  expect(s.layers[1].start).toBe(300);
  expect(s.layers[1].earliest).toBe(50);
});
