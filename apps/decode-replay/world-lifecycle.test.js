import {test,expect} from "bun:test";
import {normalizeInternalReplay} from "./internal-model.mjs";
import {worldPlan,worldView,worldNodeState} from "./world-model.mjs";
const record=(kinds="SSSSFSSSSFSSSSFSSSSFSSSSFSSSSFSSSSF")=>normalizeInternalReplay({
  schema:"qgre.phase-replay.v1",clock:"globaltimer",token:"1067",
  layers:[...kinds].map((kind,i)=>({id:String(i),kind})),
  steps:[...kinds].flatMap((_,i)=>["mixer:0","mixer:1","mixer:2","mixer:3","main:0","main:1","main:2","main:5","main:8","main:9","main:10","main:11","main:12","main:15"].map((role,j)=>({
    id:`l${i}:${role}`,layer:String(i),name:role,parent:null,kind:[1,3,5,7,9,11,13].includes(j)?"wait":"work",fromTap:"a",toTap:"b",predecessors:[],
    observations:[{block:"0",startNs:String(i*2000+j*100),endNs:String(i*2000+(j+1)*100),cycles:"100",participating:true}]
  })))
});
const topology=(kinds="SSSSFSSSSFSSSSFSSSSFSSSSFSSSSFSSSSF",shared=20,ple=true)=>({
  schema:"qgre.capture-topology.v1",bindingSha:"captured-table-digest",
  modelFacts:{layer_is_sliding:kinds,num_kv_shared_layers:String(shared)},perLayerInput:ple,
  windows:[...kinds].map(k=>k==="S"?"512":"131072")
});
test("token intake, output and return loop are present but never assigned invented time",()=>{
  const r=record(),p=worldPlan(r,topology());
  expect(p.stations.map(s=>s.id)).toEqual(["entry","exit"]);
  expect(worldView(p,"entry").nodes.map(n=>n.id)).toEqual(["token-in","embedding","ple-prepare","seed-residual"]);
  expect(worldView(p,"exit").nodes.map(n=>n.id)).toEqual(["final-norm","lm-head","selection","stop-check","advance","token-out"]);
  expect(p.routes.some(e=>e.kind==="feedback"&&e.from.area==="exit"&&e.to.area==="entry")).toBe(true);
  for(const s of p.stations)for(const n of s.nodes){expect(n.evidence.duration).toBe(null);expect(worldNodeState(n,p.sequence.start).active).toBe(false);}
  expect(p.sequence.duration).toBe(69400);
});
test("every real layer exposes its entry, output publication and recorded wait machinery",()=>{
  const p=worldPlan(record(),topology());
  for(const l of p.layers){
    for(const id of ["input-norm","activation-stage","publish-residual","residual-ready","weight-transport"])expect(l.nodes.some(n=>n.id===id)).toBe(true);
    const wait=l.nodes.find(n=>n.id==="residual-ready");
    expect(wait.evidence.step.id).toBe("l"+l.id+":main:15");
    expect(wait.evidence.duration).toBe(100);
  }
});
test("E2B PLE is a side branch inside the recorded tail, not a second timed cost",()=>{
  const p=worldPlan(record(),topology()),l=p.layers[0],ple=l.nodes.find(n=>n.id==="ple-inject");
  expect(ple.evidence.kind).toBe("shared");
  expect(ple.evidence.duration).toBe(null);
  expect(ple.evidence.step.id).toBe("l0:main:12");
  const view=worldView(p,"0","ple-inject");
  expect(view.nodes.map(n=>n.id)).toEqual(["ple-gate","ple-input","ple-activate","ple-project","ple-norm","ple-add"]);
  expect(view.nodes.every(n=>n.evidence.duration===null)).toBe(true);
});
test("shared KV consumers point to the captured owner's bank and do not pretend to write their own",()=>{
  const p=worldPlan(record(),topology());
  const routes=p.routes.filter(e=>e.kind==="kv-share");
  expect(routes.length).toBe(20);
  for(const l of p.layers.slice(15)){
    const source=l.kind==="F"?"14":"13",cache=l.nodes.find(n=>n.id==="kv-cache");
    expect(cache.targetLayer).toBe(source);
    expect(cache.shape).toBe("portal");
    expect(l.edges.some(e=>e.from==="mixer"&&e.to==="kv-cache")).toBe(false);
    expect(routes.some(e=>e.from.area===source&&e.to.area===l.id)).toBe(true);
  }
});
test("generated shared tails always use the last unshared owner of the same kind",()=>{
  for(let n=4;n<=36;n++)for(let shared=0;shared<n-2;shared++){
    const kinds=Array.from({length:n},(_,i)=>i%3===2?"F":"S").join(""),p=worldPlan(record(kinds),topology(kinds,shared,false));
    const boundary=n-shared;
    for(const r of p.routes.filter(e=>e.kind==="kv-share")){
      const owner=Number(r.from.area),consumer=Number(r.to.area);
      expect(owner).toBeLessThan(boundary);expect(consumer).toBeGreaterThanOrEqual(boundary);
      expect(kinds[owner]).toBe(kinds[consumer]);
      expect(owner).toBe(kinds.slice(0,boundary).lastIndexOf(kinds[consumer]));
    }
  }
});
test("missing topology stays unknown and malformed captured topology refuses before rendering",()=>{
  const r=record(),p=worldPlan(r);
  expect(p.layers[0].nodes.some(n=>n.id==="ple-inject")).toBe(false);
  expect(p.routes.filter(e=>e.kind==="kv-share")).toEqual([]);
  for(const bad of [{...topology(),perLayerInput:"yes"},{...topology(),windows:["512"]},{...topology(),modelFacts:{layer_is_sliding:"F",num_kv_shared_layers:"20"}},{...topology(),modelFacts:{layer_is_sliding:r.layers.map(l=>l.kind).join(""),num_kv_shared_layers:"35"}}])expect(()=>worldPlan(r,bad)).toThrow();
});
