import { test, expect } from "bun:test";
import { normalizeBundle, activeEvents, seekEvent, filteredEvents, advancePlayback, fitAssembly } from "./model.mjs";

const bundle = (events) => ({
  schema: "qgre.decode-replay.v1", label: "Test", models: [], counters: [],
  timelines: [{ id: "nsys", label: "Recorded launch", unit: "ns", events }],
});
test("large profiler timestamps retain nanosecond differences before conversion", () => {
  const data = normalizeBundle(bundle([
    { id:"b", name:"kernel B", kind:"kernel", startNs:"9007199254741100", endNs:"9007199254741500" },
    { id:"a", name:"kernel A", kind:"kernel", startNs:"9007199254740993", endNs:"9007199254741093" },
  ]));
  expect(data.timelines[0].events.map(e=>[e.id,e.start,e.end])).toEqual([["a",0,100],["b",107,507]]);
  expect(activeEvents(data.timelines[0], 100)).toHaveLength(0);
  expect(activeEvents(data.timelines[0], 107).map(e=>e.id)).toEqual(["b"]);
});
test("independent clocks are normalized separately and counters do not become events", () => {
  const input=bundle([{id:"a",name:"A",kind:"kernel",startNs:"100",endNs:"200"}]);
  input.timelines.push({ id:"phase",label:"GPU cycles",unit:"cycles",events:[{id:"p",name:"phase",kind:"phase",startNs:"1000000",endNs:"1000099"}]});
  input.counters=[{name:"aggregate",timelineBinding:null,metrics:[{name:"sm",value:"82",unit:"%"}]}];
  const data=normalizeBundle(input);
  expect(data.timelines.map(t=>t.duration)).toEqual([100,99]);
  expect(data.counters[0].timelineBinding).toBeNull();
});
test("invalid timelines fail visibly instead of silently producing a replay", () => {
  expect(()=>normalizeBundle({schema:"different"})).toThrow();
  for(const [startNs,endNs] of [["8","3"],["NaN","9"],["1.5","9"]]) {
    expect(()=>normalizeBundle(bundle([{id:"a",name:"A",kind:"kernel",startNs,endNs}]))).toThrow();
  }
});
test("phase replay preserves measured layer periods and never compresses inter-layer gaps", () => {
  const input=bundle([]);
  input.timelines[0].phaseReport={phases:[{phase:"A->B",class:"work"},{phase:"B->C",class:"wait"}],per_layer_phase_ns:[["3","4"],["8","5"]],layer_period:{per_layer_ns:["10"]}};
  input.timelines[0].launchBinding={position:"17"};
  const events=normalizeBundle(input).timelines[0].events;
  expect(events.map(e=>[e.start,e.end,e.kind])).toEqual([[0,3,"work"],[3,7,"wait"],[7,10,"inter-layer"],[10,18,"work"],[18,23,"wait"]]);
  expect(events[3].layer).toBe("1");
  expect(events[3].token).toBe("17");
});
test("focused playback advances within its selected interval instead of skipping to the full recording", () => {
  expect(advancePlayback(100,100,10,100,200)).toEqual({time:101,ended:false});
  expect(advancePlayback(199.9,100,10,100,200)).toEqual({time:200,ended:true});
  for(let span=1;span<1000;span+=17) {
    const next=advancePlayback(50,100,30,50,50+span);
    expect(next.time).toBeGreaterThanOrEqual(50);
    expect(next.time).toBeLessThanOrEqual(50+span);
  }
});
test("reset camera fits every rotated cuboid inside the viewport at all supported sizes", () => {
  for(const width of [354,1000,2900])for(const height of [280,640])for(let count=1;count<=16;count+=3)for(const yaw of [-1,-.25,.7]) {
    const nodes=Array.from({length:count},(_,i)=>({id:String(i),level:i%4+1}));
    const layout=fitAssembly(nodes,width,height,{yaw,pitch:.4,explode:.5,zoom:1});
    for(const item of layout.items)for(const point of item.corners){
      expect(point.x).toBeGreaterThanOrEqual(31.999);
      expect(point.x).toBeLessThanOrEqual(width-31.999);
      expect(point.y).toBeGreaterThanOrEqual(39.999);
      expect(point.y).toBeLessThanOrEqual(height-39.999);
    }
  }
});
test("malformed architecture or metric records fail before replacing the visible session", () => {
  const invalid=bundle([]);
  invalid.models=[{id:"broken",nodes:null,edges:[],layers:[]}];
  expect(()=>normalizeBundle(invalid)).toThrow();
  invalid.models=[];
  invalid.counters=[{metrics:{}}];
  expect(()=>normalizeBundle(invalid)).toThrow();
  invalid.counters=[];
  invalid.architecture={revision:123};
  expect(()=>normalizeBundle(invalid)).toThrow();
});
test("generated overlapping captures preserve interval membership and navigation", () => {
  const events=Array.from({length:200},(_,i)=>({id:String(i),name:`launch ${i}`,kind:"kernel",startNs:String(1000+i*7),endNs:String(1020+i*7)}));
  const timeline=normalizeBundle(bundle(events)).timelines[0];
  for(let t=0;t<1400;t+=13) {
    expect(activeEvents(timeline,t).map(e=>e.id)).toEqual(events.filter((_,i)=>i*7<=t&&i*7+20>t).map(e=>e.id));
  }
  expect(seekEvent(timeline,20,1)).toBe(21);
  expect(seekEvent(timeline,20,-1)).toBe(14);
  expect(filteredEvents(timeline,"launch 199")).toHaveLength(1);
});
