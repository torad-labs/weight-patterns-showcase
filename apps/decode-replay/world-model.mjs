// Scene geometry is logical composition; timing is always an observation in the tape.
import {mechanismGraph as anatomyForWorld,mechanismEvidence as evidenceForWorld} from "./mechanism.mjs";
import {stepMetrics as metricsForWorld} from "./internal-model.mjs";
import {decodeSequence as sequenceForWorld} from "./decode-sequence.mjs";
import {admitWorldTopology,completeWorldLayer,lifecycleDetail,tokenStations,lifecycleRoutes} from "./world-lifecycle.mjs";
function worldAnatomy(replay,id,focus,topology){
  const graph=focus?(lifecycleDetail(replay,id,focus,topology)??anatomyForWorld(replay,id,focus)):completeWorldLayer(replay,id,anatomyForWorld(replay,id),topology);
  return decorateWorld(replay,graph);
}
function decorateWorld(replay,graph){
  return {...graph,nodes:graph.nodes.map(n=>{
    const size=n.size??(n.shape==="storage"?[210,100,65]:[220,105,40]);
    return {...n,position:[n.position[0],n.position[1]*1.6,n.position[2]],size:[size[0],size[1]*1.6,size[2]],evidence:evidenceForWorld(replay,n)};
  })};
}
export function worldPlan(replay,captureTopology=null){
  const topology=admitWorldTopology(replay,captureTopology),sequence=sequenceForWorld(replay);
  const layers=replay.layers.map((l,i)=>({
    ...l,index:i,origin:[(i%5)*3600,Math.floor(i/5)*2900,0],
    ...worldAnatomy(replay,l.id,null,topology),
    steps:replay.steps.filter(s=>s.layer===l.id).map(step=>({step,metrics:metricsForWorld(step)})),
    timing:sequence.layers.find(x=>x.id===l.id)
  }));
  const stations=tokenStations(topology,layers).map(s=>({...s,...decorateWorld(replay,s)}));
  return {replay,sequence,topology,layers,stations,areas:[stations[0],...layers,stations[1]],routes:lifecycleRoutes(layers,topology),links:layers.slice(1).map((l,i)=>({from:layers[i].id,to:l.id}))};
}
export function worldView(plan,id,focus=null){
  const layer=plan.areas.find(l=>l.id===id);
  if(!layer)throw new Error("Unknown world area.");
  return focus&&layer.kind!=="station"?{...layer,...worldAnatomy(plan.replay,id,focus,plan.topology),origin:[0,0,0]}:{...layer,origin:[0,0,0]};
}
export function worldBounds(layers){
  let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
  for(const l of layers)for(const n of l.nodes){
    const x=l.origin[0]+n.position[0],y=l.origin[1]+n.position[1],size=n.size;
    x0=Math.min(x0,x-size[0]/2-100);x1=Math.max(x1,x+size[0]/2+100);
    y0=Math.min(y0,y-size[1]/2-180);y1=Math.max(y1,y+size[1]/2+180);
  }
  return [x0,y0,x1,y1];
}
export function worldNodeState(node,time){
  const observed=node.evidence.metrics?.observations??[];
  const live=observed.filter(o=>o.start<=time&&o.end>time);
  const m=node.evidence.metrics;
  return {active:live.length>0,blocks:live.length,progress:m?.envelope?Math.max(0,Math.min(1,(time-m.start)/m.envelope)):0,kind:node.evidence.step?.kind??"untimed"};
}
