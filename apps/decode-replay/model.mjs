// Presentation only: exact capture clocks are subtracted as integers before display conversion.
// No model tensors, numerical inference, or inferred execution schedule lives here.
export function normalizeBundle(raw) {
  if (!raw || raw.schema !== "qgre.decode-replay.v1") throw new Error("Not a QGRE replay bundle. Generate bundle.json with qgre dev decode-replay.");
  const data = structuredClone(raw);
  for (const key of ["models", "timelines", "counters"]) if (!Array.isArray(data[key])) throw new Error(`Missing ${key} array.`);
  if(data.architecture?.revision!==undefined && typeof data.architecture.revision!=="string") throw new Error("Source revision must be text.");
  for(const model of data.models){
    if(!model||typeof model.id!=="string"||!Array.isArray(model.nodes)||!Array.isArray(model.edges)||!Array.isArray(model.layers)) throw new Error("Invalid source model: expected identity, nodes, edges and layers.");
    if(model.label!==undefined&&typeof model.label!=="string")throw new Error("Model label must be text.");
    const nodeIds=new Set();
    for(const node of model.nodes){
      if(!node||typeof node.id!=="string"||nodeIds.has(node.id))throw new Error("Source nodes need unique text identities.");
      nodeIds.add(node.id);
      if(node.label!==undefined&&typeof node.label!=="string")throw new Error("Node label must be text.");
      if(node.level!==undefined&&(!Number.isFinite(node.level)||node.level<0||node.level>4))throw new Error("Invalid kernel tier.");
    }
  }
  for(const capture of data.counters){
    if(!capture||!Array.isArray(capture.metrics))throw new Error("Counter capture needs a metrics array.");
    for(const metric of capture.metrics)if(!metric||typeof metric.name!=="string"||typeof metric.value!=="string")throw new Error("Metrics must preserve their names and decimal text values.");
  }
  const ids = new Set();
  data.timelines = data.timelines.map((timeline) => {
    if (!timeline.id || ids.has(timeline.id)) throw new Error("Capture IDs must be unique.");
    ids.add(timeline.id);
    if (timeline.phaseReport && !timeline.internalReplay) {
      const r=timeline.phaseReport, rows=r.per_layer_phase_ns, phases=r.phases, periods=r.layer_period?.per_layer_ns;
      if(!Array.isArray(rows)||!Array.isArray(phases)||!Array.isArray(periods)||periods.length!==Math.max(0,rows.length-1)) throw new Error("Phase report has incomplete measured layer periods.");
      const integer=v=>{if(typeof v!=="string"||!/^\d+$/.test(v))throw new Error("Phase report contains a negative or invalid interval.");return BigInt(v);};
      const events=[];let layerStart=0n;
      rows.forEach((row,layer)=>{
        if(!Array.isArray(row)||row.length!==phases.length)throw new Error("Phase row does not match its declared phases.");
        let at=layerStart;
        row.forEach((duration,i)=>{
          const end=at+integer(duration),phase=phases[i];
          events.push({id:`phase:${layer}:${i}`,name:`Layer ${layer} · ${phase.phase}`,kind:phase.class,layer:String(layer),token:timeline.launchBinding?.position,startNs:String(at),endNs:String(end),recordingNodeId:`phase:${layer}:${i}`});at=end;
        });
        if(layer<rows.length-1){const next=layerStart+integer(periods[layer]);if(next<at)throw new Error("Phase durations exceed the measured layer period.");if(next>at)events.push({id:`phase:${layer}:link`,name:`Layer ${layer} → ${layer+1}`,kind:"inter-layer",layer:String(layer),startNs:String(at),endNs:String(next)});layerStart=next;}
      });
      timeline.events=events;
    }
    if (!Array.isArray(timeline.events)) throw new Error("Capture events must be an array.");
    const eventIds = new Set();
    const parsed = timeline.events.map((event) => {
      if (!event.id || eventIds.has(event.id)) throw new Error("Event IDs must be unique within a capture.");
      eventIds.add(event.id);
      if (![event.startNs, event.endNs].every(v => typeof v === "string" && /^\d+$/.test(v))) throw new Error(`Invalid timestamp in ${event.id}.`);
      const begin = BigInt(event.startNs), end = BigInt(event.endNs);
      if (end < begin) throw new Error(`Reversed interval in ${event.id}.`);
      return { event, begin, end };
    }).sort((a,b)=>a.begin < b.begin ? -1 : a.begin > b.begin ? 1 : 0);
    const origin = parsed[0]?.begin ?? 0n;
    let duration = 0n;
    const events = parsed.map(({event,begin,end}) => {
      if (end-origin > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("Capture span exceeds exact browser range; split the capture.");
      duration = end-origin > duration ? end-origin : duration;
      return {...event, name:String(event.name ?? event.id), start:Number(begin-origin), end:Number(end-origin)};
    });
    return {...timeline, events, origin:String(origin), duration:Number(duration)};
  });
  return data;
}
export function activeEvents(timeline, time) {
  return timeline?.events.filter(e => e.start <= time && e.end > time) ?? [];
}
export function filteredEvents(timeline, query) {
  const term = query.trim().toLowerCase();
  return timeline?.events.filter(e => `${e.name} ${e.kind} ${e.id}`.toLowerCase().includes(term)) ?? [];
}
export function seekEvent(timeline, time, direction) {
  const times=timeline?.events.map(e=>e.start) ?? [];
  return direction > 0 ? times.find(t=>t>time) ?? timeline?.duration ?? 0 : times.findLast(t=>t<time) ?? 0;
}
export function fitAssembly(nodes, width, height, {yaw,pitch,explode,zoom}) {
  const columns=Math.min(width<600?2:7,Math.max(2,nodes.length)),rows=Math.ceil(nodes.length/columns);
  const offsets=([width,height,depth])=>[[-width/2,-height/2,0],[width/2,-height/2,0],[width/2,height/2,0],[-width/2,height/2,0],[-width/2,-height/2,-depth],[width/2,-height/2,-depth],[width/2,height/2,-depth],[-width/2,height/2,-depth]];
  const rotate=([x,y,z])=>{const a=x*Math.cos(yaw)-z*Math.sin(yaw),b=x*Math.sin(yaw)+z*Math.cos(yaw);return {x:a,y:y*Math.cos(pitch)-b*Math.sin(pitch),z:y*Math.sin(pitch)+b*Math.cos(pitch)};};
  const positioned=nodes.map((node,i)=>({node,offsets:offsets(node.size??[140,56,18]),center:node.position?[node.position[0],node.position[1],node.position[2]*(.4+explode*1.6)]:[(i%columns-(columns-1)/2)*175,(Math.floor(i/columns)-(rows-1)/2)*105,(Number(node.level??3)-2)*explode*55]}));
  const points=positioned.flatMap(({center:[x,y,z],offsets})=>offsets.map(([a,b,c])=>rotate([x+a,y+b,z+c])));
  const xs=points.map(p=>p.x),ys=points.map(p=>p.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const scale=Math.min((width-64)/Math.max(1,maxX-minX),(height-80)/Math.max(1,maxY-minY))*zoom;
  const project=point=>{const p=rotate(point);return {x:width/2+(p.x-(minX+maxX)/2)*scale,y:height/2+(p.y-(minY+maxY)/2)*scale,z:p.z};};
  const items=positioned.map(item=>({...item,p:project(item.center),corners:item.offsets.map(([a,b,c])=>project([item.center[0]+a,item.center[1]+b,item.center[2]+c]))}));
  return {items,scale,project};
}
export function advancePlayback(time, elapsedMs, seconds, start, end) {
  if(seconds<=0 || end<start) throw new Error("Invalid playback range or speed.");
  const next=Math.min(end,Math.max(start,time)+Math.min(Math.max(0,elapsedMs),100)/1000*(end-start)/seconds);
  return {time:next,ended:next>=end};
}
export function formatTime(value, unit = "ns") {
  if (unit !== "ns") return `${Math.round(value).toLocaleString()} ${unit}`;
  if (value >= 1e9) return `${(value/1e9).toFixed(3)} s`;
  if (value >= 1e6) return `${(value/1e6).toFixed(3)} ms`;
  if (value >= 1e3) return `${(value/1e3).toFixed(3)} µs`;
  return `${Math.round(value)} ns`;
}
