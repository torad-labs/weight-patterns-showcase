import {test,expect} from "bun:test";
import {normalizeInternalReplay} from "./internal-model.mjs";
import {worldPlan,worldView,worldNodeState,worldBounds} from "./world-model.mjs";
function recording(count=3){
  const layers=Array.from({length:count},(_,i)=>({id:String(i),kind:i%2?"D":"S"}));
  const steps=layers.flatMap((l,i)=>[
    {id:`l${i}:mixer:0`,name:"Projection",layer:l.id,parent:null,kind:"work",fromTap:"A",toTap:"B",predecessors:[],observations:[{block:"0",startNs:String(i*300),endNs:String(i*300+200),cycles:"100",participating:true}]},
    {id:`l${i}:q:0`,name:"Q product",layer:l.id,parent:`l${i}:mixer:0`,kind:"work",fromTap:"A",toTap:"Q",predecessors:[],observations:[{block:"0",startNs:String(i*300+20),endNs:String(i*300+80),cycles:"30",participating:true}]}
  ]);
  return normalizeInternalReplay({schema:"qgre.phase-replay.v1",clock:"globaltimer",layers,steps});
}
test("the world contains every recorded layer and every step, with distinct spatial addresses",()=>{
  for(let count=1;count<=48;count++){
    const replay=recording(count),plan=worldPlan(replay);
    expect(plan.layers.map(l=>l.id)).toEqual(replay.layers.map(l=>l.id));
    expect(plan.layers.flatMap(l=>l.steps.map(s=>s.step.id)).sort()).toEqual(replay.steps.map(s=>s.id).sort());
    expect(new Set(plan.layers.map(l=>l.origin.join(","))).size).toBe(count);
    expect(plan.links.length).toBe(count-1);
    expect(worldBounds(plan.layers).every(Number.isFinite)).toBe(true);
  }
});
test("opening a measured housing keeps the actual children and a shared operation has no invented duration",()=>{
  const plan=worldPlan(recording()),view=worldView(plan,"0","mixer");
  expect(view.nodes.map(n=>n.id)).toEqual(["l0:q:0"]);
  expect(view.nodes[0].evidence.duration).toBe(60);
  const leaf=worldView(plan,"0","l0:q:0");
  for(const n of leaf.nodes)expect(n.evidence.duration).toBe(null);
});
test("activity follows recorded block intervals, not explanatory wires or shared invented timings",()=>{
  const plan=worldPlan(recording()),node=plan.layers[0].nodes.find(n=>n.id==="l0:q:0");
  expect(worldNodeState(node,10).active).toBe(false);
  expect(worldNodeState(node,40).active).toBe(true);
  expect(worldNodeState(node,90).active).toBe(false);
  expect(worldNodeState(plan.layers[0].nodes.find(n=>n.id==="weights"),40).active).toBe(false);
});
test("two views can inspect different layers without changing the recording or its clock",()=>{
  const plan=worldPlan(recording()),before=JSON.stringify(plan.sequence);
  const left=worldView(plan,"0"),right=worldView(plan,"1");
  expect(left.nodes.some(n=>n.id==="kv-cache")).toBe(true);
  expect(right.nodes.some(n=>n.id==="recurrent-state")).toBe(true);
  expect(right.nodes.some(n=>n.id==="kv-cache")).toBe(false);
  expect(JSON.stringify(plan.sequence)).toBe(before);
});
