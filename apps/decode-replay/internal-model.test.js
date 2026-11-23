import { test, expect } from "bun:test";
import { normalizeInternalReplay, normalizeInternalCapture, layerGraph, stepMetrics, rankedSteps } from "./internal-model.mjs";

const raw=()=>({
  schema:"qgre.phase-replay.v1", clock:"globaltimer", token:"1067",
  layers:[{id:"0",kind:"S"},{id:"1",kind:"F"}],
  steps:[
    {id:"l0:work",layer:"0",name:"Output projection",kind:"work",fromTap:"A",toTap:"B",parent:null,predecessors:[],description:"Project attention output",observations:[
      {block:"0",sm:"4",startNs:"100",endNs:"180",cycles:"200",participating:true},
      {block:"1",sm:"7",startNs:"110",endNs:"200",cycles:"230",participating:true}]},
    {id:"l0:wait",layer:"0",name:"Wait for output",kind:"wait",fromTap:"B",toTap:"C",parent:null,predecessors:["l0:work"],description:"Consume the published output",observations:[
      {block:"0",sm:"4",startNs:"180",endNs:"220",cycles:"100",participating:true},
      {block:"1",sm:"7",startNs:"200",endNs:"225",cycles:"60",participating:true}]},
    {id:"l0:inner",layer:"0",name:"Weight stream",kind:"work",fromTap:"W0",toTap:"W1",parent:"l0:work",predecessors:[],description:"Read weights and perform the matvec",observations:[
      {block:"0",sm:"4",startNs:"120",endNs:"160",cycles:"100",participating:true}]}
  ]
});
test("a layer is a connected graph of timed work and waits, with drill-down children",()=>{
  const r=normalizeInternalReplay(raw()),g=layerGraph(r,"0");
  expect(g.nodes.map(n=>n.id)).toEqual(["l0:work","l0:wait"]);
  expect(g.edges).toEqual([{from:"l0:work",to:"l0:wait"}]);
  expect(layerGraph(r,"0","l0:work").nodes.map(n=>n.id)).toEqual(["l0:inner"]);
  expect(g.nodes[0].name).toBe("Output projection");
});
test("step metrics retain critical span, block duration, skew and wait tail as different quantities",()=>{
  const r=normalizeInternalReplay(raw()),work=stepMetrics(r.steps[0]),wait=stepMetrics(r.steps[1]);
  expect(work).toMatchObject({start:100,end:200,critical:90,envelope:100,minBlock:80,maxBlock:90,blocks:2});
  expect(wait).toMatchObject({start:180,end:225,critical:25,envelope:45,maxBlock:40,arrivalSkew:20});
  expect(wait.observations[0]).toMatchObject({beforeLastProducer:20,afterLastProducer:20});
  expect(wait.observations[1]).toMatchObject({beforeLastProducer:0,afterLastProducer:25});
});
test("ranking never double-counts child phases as independent top-level costs",()=>{
  const r=normalizeInternalReplay(raw());
  expect(rankedSteps(r,"work").map(s=>s.id)).toEqual(["l0:work"]);
  expect(rankedSteps(r,"wait").map(s=>s.id)).toEqual(["l0:wait"]);
});
test("block overlap is preserved instead of arranging parallel work into a fake serial timeline",()=>{
  const input=raw();input.steps[0].observations[1].startNs="150";
  const r=normalizeInternalReplay(input),m=stepMetrics(r.steps[0]);
  expect(m.observations[0].end).toBe(180);
  expect(m.observations[1].start).toBe(150);
  expect(m.critical).toBe(50);
});
test("generated clock translations and block ordering preserve measured durations",()=>{
  for(let i=0;i<80;i++){
    const input=raw(),shift=(i%2?-1n:1n)*(2n**60n+BigInt(i)*10007n);
    for(const s of input.steps){for(const o of s.observations){o.startNs=String(BigInt(o.startNs)+shift);o.endNs=String(BigInt(o.endNs)+shift);}if(i%3===0)s.observations.reverse();}
    const r=normalizeInternalReplay(input);
    expect(stepMetrics(r.steps[0])).toMatchObject({critical:90,envelope:100,minBlock:80,maxBlock:90,blocks:2});
    expect(stepMetrics(r.steps[1])).toMatchObject({critical:25,arrivalSkew:20,maxBlock:40});
    expect(layerGraph(r,"0").edges).toEqual([{from:"l0:work",to:"l0:wait"}]);
  }
});
test("capture metadata and renderable root coverage are admitted before replacing the current view",()=>{
  const valid=raw();valid.layers=valid.layers.slice(0,1);
  const capture={source:{path:"saved-tape.bin"},internalReplay:valid};
  expect(normalizeInternalCapture(capture).replay.steps.length).toBe(3);
  expect(()=>normalizeInternalCapture({...capture,source:undefined})).toThrow();
  expect(()=>normalizeInternalCapture({...capture,source:{path:42}})).toThrow();
  const absentRoot=raw();absentRoot.layers.unshift({id:"missing",kind:"S"});
  expect(()=>normalizeInternalCapture({...capture,internalReplay:absentRoot})).toThrow();
  const noParticipants=structuredClone(valid);
  for(const s of noParticipants.steps)for(const o of s.observations)o.participating=false;
  expect(()=>normalizeInternalCapture({...capture,internalReplay:noParticipants})).toThrow();
});
test("generated invalid render-field types are rejected before mounting",()=>{
  const valid=raw();valid.layers=valid.layers.slice(0,1);
  const admit=value=>normalizeInternalCapture({source:{path:"saved.bin"},internalReplay:value});
  for(const field of ["id","name","kind","fromTap","toTap"]){
    for(const value of [null,false,0,[],{},""]){
      const input=structuredClone(valid);input.steps[0][field]=value;
      expect(()=>admit(input)).toThrow();
    }
  }
  for(const value of ["",false,0,[],{}]){
    const input=structuredClone(valid);for(const step of input.steps.filter(s=>s.parent==null))step.parent=value;
    expect(()=>admit(input)).toThrow();
  }
  for(const field of ["id","kind"])for(const value of [null,false,0,[],{},""]){
    const input=structuredClone(valid);input.layers[0][field]=value;
    expect(()=>admit(input)).toThrow();
  }
  // Missing and explicit-null parents are the same root convention everywhere.
  const missing=structuredClone(valid);delete missing.steps[0].parent;
  expect(admit(missing).replay.steps[0].id).toBe("l0:work");
});
test("invalid dependencies, missing timestamps and duplicate steps are refused",()=>{
  const input=raw();input.steps[1].predecessors=["missing"];
  expect(()=>normalizeInternalReplay(input)).toThrow();
  input.steps[1].predecessors=["l0:work"];input.steps[0].observations[0].startNs="unknown";
  expect(()=>normalizeInternalReplay(input)).toThrow();
  const empty=raw();empty.layers=[];empty.steps=[];
  expect(()=>normalizeInternalReplay(empty)).toThrow();
  const duplicate=raw();duplicate.steps.push(duplicate.steps[0]);
  expect(()=>normalizeInternalReplay(duplicate)).toThrow();
});
