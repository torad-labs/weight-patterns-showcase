// Recorded device intervals only. Integer clocks stay exact until their bounded difference is displayed.
export function normalizeInternalReplay(raw) {
  if(raw?.schema!=="qgre.phase-replay.v1" || raw.clock!=="globaltimer" || !Array.isArray(raw.layers) || !raw.layers.length || !Array.isArray(raw.steps) || !raw.steps.length) throw new Error("Missing internal phase replay.");
  const integer=v=>{if(typeof v!=="string" || !/^-?\d+$/.test(v))throw new Error("Phase timestamps must be decimal text.");return BigInt(v);};
  const text=v=>typeof v==="string"&&v.length>0;
  if(raw.layers.some(l=>!l||!text(l.id)||!text(l.kind)))throw new Error("Layer identifiers and kinds must be nonempty text.");
  const layers=new Set(raw.layers.map(l=>l.id)), ids=new Set();
  if(layers.size!==raw.layers.length)throw new Error("Duplicate layers.");
  // The reader emits signed offsets from tap A, so attention can legitimately precede zero.
  // Keep a zero origin for small offsets; subtract large origins before Number conversion.
  let origin=0n, earliest=null;
  for(const s of raw.steps)for(const o of s.observations??[]){const t=integer(o.startNs);if(earliest===null||t<earliest)earliest=t;}
  if(earliest!==null && (earliest>BigInt(Number.MAX_SAFE_INTEGER)||earliest< -BigInt(Number.MAX_SAFE_INTEGER)))origin=earliest;
  const number=v=>{const n=integer(v)-origin;if(n>BigInt(Number.MAX_SAFE_INTEGER)||n< -BigInt(Number.MAX_SAFE_INTEGER))throw new Error("Phase span is too large.");return Number(n);};
  const steps=raw.steps.map(s=>{
    if(!s||!text(s.id)||ids.has(s.id)||!layers.has(s.layer)||!text(s.name)||!text(s.kind)||!text(s.fromTap)||!text(s.toTap)||!Array.isArray(s.predecessors)||!Array.isArray(s.observations))throw new Error("Invalid internal step identity, boundaries or graph.");
    if(s.parent!=null&&!text(s.parent))throw new Error("Invalid phase parent.");
    ids.add(s.id);const blocks=new Set();
    const observations=s.observations.map(o=>{
      if(typeof o.block!=="string"||blocks.has(o.block)||typeof o.participating!=="boolean")throw new Error("Invalid block observation.");
      blocks.add(o.block);integer(o.cycles);
      return {...o,start:number(o.startNs),end:number(o.endNs),anomaly:integer(o.endNs)<integer(o.startNs)};
    });
    return {...s,name:String(s.name),description:String(s.description??""),observations};
  });
  const byId=new Map(steps.map(s=>[s.id,s]));
  for(const s of steps){
    for(const id of [...s.predecessors,...(s.parent?[s.parent]:[])]){
      if(!byId.has(id)||id===s.id||byId.get(id).layer!==s.layer)throw new Error("Invalid phase dependency or parent.");
    }
  }
  const visiting=new Set(),done=new Set();
  function visit(s){if(visiting.has(s.id))throw new Error("Cyclic phase graph.");if(done.has(s.id))return;visiting.add(s.id);for(const p of s.predecessors)visit(byId.get(p));if(s.parent)visit(byId.get(s.parent));visiting.delete(s.id);done.add(s.id);}
  steps.forEach(visit);
  return {...raw,origin:String(origin),steps,byId};
}
// Admission for the mounted page is stricter than a partial graph model.
// Validate everything the initial render needs before stopping the old controller.
export function normalizeInternalCapture(capture){
  if(typeof capture?.source?.path!=="string"||!capture.source.path.trim())throw new Error("Internal capture is missing its source path.");
  const replay=normalizeInternalReplay(capture.internalReplay);
  for(const layer of replay.layers){
    const roots=replay.steps.filter(s=>s.layer===layer.id&&s.parent==null);
    if(!roots.length||!roots.some(s=>s.observations.some(o=>o.participating)))throw new Error(`Layer ${layer.id} has no recorded participating root step.`);
  }
  return {...capture,replay};
}
export function layerGraph(replay,layer,parent=null){
  const nodes=replay.steps.filter(s=>s.layer===layer&&(s.parent??null)===parent),ids=new Set(nodes.map(n=>n.id));
  return {nodes,edges:nodes.flatMap(s=>s.predecessors.filter(p=>ids.has(p)).map(p=>({from:p,to:s.id})))};
}
export function stepMetrics(step,includeFree=false){
  const observations=step.observations.filter(o=>includeFree||o.participating);
  if(!observations.length)return {start:0,end:0,critical:0,envelope:0,minBlock:0,maxBlock:0,blocks:0,arrivalSkew:0,observations:[],anomalies:0};
  const starts=observations.map(o=>o.start),ends=observations.map(o=>o.end),durations=observations.map(o=>o.end-o.start);
  const start=Math.min(...starts),lastStart=Math.max(...starts),end=Math.max(...ends);
  return {start,end,critical:end-lastStart,envelope:end-start,minBlock:Math.min(...durations),maxBlock:Math.max(...durations),blocks:observations.length,arrivalSkew:lastStart-start,anomalies:observations.filter(o=>o.anomaly).length,
    observations:observations.map(o=>({...o,
      // Arithmetic split at the last observed entry, NOT a claim of causal producer identity.
      beforeLastProducer:Math.max(0,Math.min(o.end,lastStart)-o.start),
      afterLastProducer:Math.max(0,o.end-Math.max(o.start,lastStart))
    }))};
}
export function rankedSteps(replay,kind){
  return replay.steps.filter(s=>!s.parent&&s.kind===kind).sort((a,b)=>stepMetrics(b).critical-stepMetrics(a).critical);
}
