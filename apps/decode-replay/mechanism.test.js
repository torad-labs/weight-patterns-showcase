import {test,expect} from "bun:test";
import {mechanismGraph,mechanismEvidence} from "./mechanism.mjs";
import {normalizeInternalReplay} from "./internal-model.mjs";
import {fitAssembly} from "./model.mjs";
function fixture(kind="S"){
  const steps=[
    ["mixer:0","Q / K / V + RoPE",null],["mixer:2","Attention",null],
    ["main:0","Output projection",null],["main:2","Output fold + residual",null],
    ["main:8","Gate / up projection",null],["main:10","Down projection",null],
    ["main:12","Down fold + residual",null],
    ["gate-up:0","Input stage + norm","main:8"],["gate-up:1","Gate / up weight streams","main:8"],["gate-up:2","Activation + publish","main:8"],
    ["attention-pass0:2","Attend atom","mixer:2"]
  ].map(([id,name,parent],i)=>({id:"l0:"+id,layer:"0",name,kind:"work",description:name,parent:parent?"l0:"+parent:null,predecessors:[],fromTap:"A",toTap:"B",observations:[{block:"0",sm:"2",participating:true,startNs:String(i*100),endNs:String(i*100+50),cycles:"100"}]}));
  return normalizeInternalReplay({schema:"qgre.phase-replay.v1",clock:"globaltimer",layers:[{id:"0",kind}],steps});
}
test("cutaway exposes connected computational parts, storage and residual bypasses",()=>{
  const g=mechanismGraph(fixture(),"0");
  expect(g.nodes.some(n=>n.id==="attention")).toBe(true);
  expect(g.nodes.some(n=>n.id==="kv-cache"&&n.shape==="storage")).toBe(true);
  expect(g.edges.some(e=>e.from==="input"&&e.to==="post-attention"&&e.payload==="residual")).toBe(true);
  expect(g.nodes.find(n=>n.id==="ffn").expandable).toBe(true);
  const ids=new Set(g.nodes.map(n=>n.id));
  for(const edge of g.edges){expect(ids.has(edge.from)).toBe(true);expect(ids.has(edge.to)).toBe(true);expect(edge.payload.length).toBeGreaterThan(0);}
});
test("the default cutaway already exposes inner parts and fits their actual rotated extents",()=>{
  const nodes=mechanismGraph(fixture(),"0").nodes;
  expect(nodes.some(n=>n.parentPart==="ffn"&&n.stepId==="l0:gate-up:1")).toBe(true);
  for(const [width,height] of [[390,560],[1440,900],[3840,1150]])for(let i=0;i<20;i++){
    const {items}=fitAssembly(nodes,width,height,{yaw:i/10-1,pitch:i/20-.5,explode:i/20,zoom:1});
    for(const item of items)for(const p of item.corners){expect(p.x).toBeGreaterThanOrEqual(0);expect(p.x).toBeLessThanOrEqual(width);expect(p.y).toBeGreaterThanOrEqual(0);expect(p.y).toBeLessThanOrEqual(height);}
  }
});
test("opening a component reveals real timed substeps before explanatory math",()=>{
  const r=fixture(),g=mechanismGraph(r,"0","ffn");
  expect(g.nodes.filter(n=>n.stepId).map(n=>n.stepId)).toContain("l0:gate-up:1");
  const exact=g.nodes.find(n=>n.stepId==="l0:gate-up:1");
  expect(mechanismEvidence(r,exact)).toMatchObject({kind:"measured",duration:50});
});
test("unseparated softmax operations share the parent observation, never invented individual durations",()=>{
  const r=fixture(),g=mechanismGraph(r,"0","l0:attention-pass0:2");
  expect(g.nodes.map(n=>n.id)).toEqual(expect.arrayContaining(["maximum","exponential","probability-sum","weighted-values","normalize"]));
  for(const n of g.nodes){const e=mechanismEvidence(r,n);expect(e.kind).toBe("shared");expect(e.duration).toBeNull();expect(e.span).toBe(50);}
});
test("DeltaNet exposes recurrent state instead of inventing an attention cache",()=>{
  const g=mechanismGraph(fixture("D"),"0");
  expect(g.nodes.some(n=>n.id==="recurrent-state")).toBe(true);
  expect(g.nodes.some(n=>n.id==="kv-cache")).toBe(false);
  expect(g.nodes.find(n=>n.id==="mixer").name).toBe("DeltaNet");
});
test("opening storage keeps codes and scales visible without inventing memory timings",()=>{
  const r=fixture(),g=mechanismGraph(r,"0","kv-cache");
  expect(g.nodes.map(n=>n.id)).toEqual(expect.arrayContaining(["key-codes","key-scales","value-codes","value-scales","position-map"]));
  for(const n of g.nodes)expect(mechanismEvidence(r,n).kind).toBe("untimed");
});
test("a missing stamp is untimed, not zero",()=>{
  const r=fixture();r.steps=[];r.byId=new Map();
  const g=mechanismGraph(r,"0");
  for(const n of g.nodes)expect(mechanismEvidence(r,n).duration).toBeNull();
});
