// Logical lifecycle over hash-bound capture facts, never tensor simulation.
// world-model decorates timing; world-lifecycle.test.js pins the behavior.
const lifePart=(id,name,position,more={})=>({id,name,label:name,position,size:[260,110,45],shape:"operation",inputs:[],outputs:[],...more});
const lifeWire=(from,to,payload)=>({from,to,payload,kind:"operand"});
export function admitWorldTopology(replay,raw){
  const owners=replay.layers.map(l=>l.id);
  if(raw==null)return {known:false,perLayerInput:false,owners,windows:null};
  if(raw.schema!=="qgre.capture-topology.v1"||typeof raw.bindingSha!=="string"||!raw.bindingSha.trim()||typeof raw.perLayerInput!=="boolean"||!raw.modelFacts||typeof raw.modelFacts!=="object")throw new Error("Invalid captured topology.");
  const count=raw.modelFacts.num_kv_shared_layers??"0";
  if(typeof count!=="string"||!/^\d+$/.test(count)||!Number.isSafeInteger(Number(count))||Number(count)>=replay.layers.length)throw new Error("Invalid captured KV-sharing extent.");
  const kinds=raw.modelFacts.layer_is_sliding,shared=Number(count),boundary=replay.layers.length-shared;
  if(kinds!=null&&(typeof kinds!=="string"||kinds.length!==replay.layers.length||[...kinds].some((k,i)=>k!==replay.layers[i].kind)))throw new Error("Captured layer kinds do not match this tape.");
  if(shared&&!kinds)throw new Error("KV-sharing sources need captured layer kinds.");
  for(let i=boundary;i<replay.layers.length;i++){const index=kinds.slice(0,boundary).lastIndexOf(kinds[i]);if(index<0)throw new Error("Shared KV has no owner kind.");owners[i]=replay.layers[index].id;}
  const windows=raw.windows??null;
  if(windows!==null&&(!Array.isArray(windows)||windows.length!==replay.layers.length||windows.some(w=>typeof w!=="string"||!/^\d+$/.test(w))))throw new Error("Invalid captured cache windows.");
  return {known:true,perLayerInput:raw.perLayerInput,owners,windows,bindingSha:raw.bindingSha};
}
export function completeWorldLayer(replay,layer,graph,topology){
  const nodes=graph.nodes.map(n=>({...n})),edges=graph.edges.map(e=>({...e})),index=replay.layers.findIndex(l=>l.id===layer);
  const step=role=>replay.byId.has("l"+layer+":"+role)?"l"+layer+":"+role:null;
  const add=(id,name,p,more={})=>nodes.push(lifePart(id,name,p,more));
  add("input-norm","Input RMSNorm",[-1150,-570,0],{expandable:true,description:"n1 reads residual and published sumsq partials. These stamps begin after its ready wait."});
  add("activation-stage","Activation tile",[-650,-570,0],{shape:"storage",expandable:true,description:"Activation codes and block scales stage together. Transfer duration is not separately recorded."});
  add("weight-transport","TMA / weight ring",[400,-470,60],{shape:"storage",expandable:true,description:"Logical transfer and operand staging. Ring occupancy and transfer latency are not recorded."});
  for(const e of edges){if(e.from==="input"&&e.to==="mixer")e.from="activation-stage";if(e.from==="weights")e.from="weight-transport";}
  edges.push(lifeWire("input","input-norm","residual + sumsq"),lifeWire("input-norm","activation-stage","codes + scales"),lifeWire("weights","weight-transport","weight codes + scales"));
  const waits=[
    ["qkv-ready","QKV ready",[-310,-160,40],"mixer:1","mixer","attention"],
    ["attention-ready","Attention ready",[430,-160,40],"mixer:3","attention","output-projection"],
    ["output-ready","Partials ready",[650,20,40],"main:1","output-projection","post-attention"],
    ["norm-ready","Norm inputs",[350,440,40],"main:5","post-attention","ffn"],
    ["ffn-ready","FFN ready",[-310,480,40],"main:9","ffn","down"],
    ["down-ready","Down ready",[-760,480,40],"main:11","down","residual-out"]
  ];
  for(const [id,name,p,role,from,to] of waits){const sid=step(role);if(!sid||!nodes.some(n=>n.id===from)||!nodes.some(n=>n.id===to))continue;add(id,name,p,{shape:"gate",size:[150,82,85],stepId:sid,description:"Recorded wait, with a logical input connection. No individual producer delay is inferred."});const e=edges.find(e=>e.from===from&&e.to===to);if(e)e.to=id;edges.push(lifeWire(id,to,"ready"));}
  let tail="residual-out";
  if(topology.perLayerInput){
    const residual=nodes.find(n=>n.id===tail);if(residual){delete residual.stepId;residual.scopeId=step("main:12");residual.name=residual.label="FFN residual";}
    add("ple-inject","Per-layer input",[-430,850,50],{scopeId:step("main:12"),expandable:true,description:"Gated PLE projection and normalization inside the down-consumer span."});
    add("layer-scale","Layer scale",[-950,850,20],{scopeId:step("main:12"),description:"Scale the closing residual sum; no independent timing."});
    edges.push(lifeWire(tail,"ple-inject","post-FFN hidden"),lifeWire("ple-inject","layer-scale","hidden + PLE"));tail="layer-scale";
  }
  add("publish-residual","Publish residual",[-1150,1080,0],{scopeId:step("main:12"),expandable:true,description:"Publish residual hi/lo, sumsq partials and readiness counters. These stores share the tail span."});
  add("residual-ready","Residual ready",[-1550,1080,0],{stepId:step("main:15"),shape:"gate",expandable:true,description:"Recorded residual wait before the next layer's n1 reader."});
  edges.push(lifeWire(tail,"publish-residual","residual + partials"),lifeWire("publish-residual","residual-ready","publish / wait"));
  const cache=nodes.find(n=>n.id==="kv-cache"),owner=topology.owners[index];
  if(cache){
    cache.expandable=true;cache.cacheWindow=topology.windows?.[index]??null;cache.tokenPosition=replay.token??null;
    if(owner!==layer){cache.shape="portal";cache.name=cache.label="KV from L"+owner;cache.targetLayer=owner;cache.description="Read the owner's cache; this layer does not append its own K/V.";
      for(let i=edges.length-1;i>=0;i--)if(edges[i].from==="mixer"&&edges[i].to==="kv-cache")edges.splice(i,1);
      const mixer=nodes.find(n=>n.id==="mixer");if(mixer)mixer.name=mixer.label="Q / norm / RoPE";}
  }
  // Separate supply, compute and publication tracks; preserve child offsets inside housings.
  const positions={input:[-1350,-570,0],"input-norm":[-900,-570,0],"activation-stage":[-450,-570,0],weights:[180,-650,-100],"weight-transport":[750,-650,60],mixer:[-900,-100,0],"qkv-ready":[-470,-100,40],"kv-cache":[-900,180,100],"recurrent-state":[-900,180,100],attention:[0,-100,0],"attention-ready":[470,-100,40],"output-projection":[900,-100,0],"output-ready":[900,200,40],"post-attention":[900,500,0],"norm-ready":[470,500,40],ffn:[0,500,0],"ffn-ready":[-470,500,40],down:[-900,500,0],"down-ready":[-1300,500,40],"residual-out":[-1300,850,0],"ple-inject":[-700,850,50],"layer-scale":[-150,850,20],"publish-residual":[400,850,0],"residual-ready":[950,850,0]};
  const originals=new Map(nodes.map(n=>[n.id,n.position]));
  for(const n of nodes){const p=positions[n.id];if(p)n.position=p;else if(n.parentPart&&positions[n.parentPart]){const a=originals.get(n.parentPart),b=positions[n.parentPart];n.position=n.position.map((v,i)=>v+b[i]-a[i]);}}
  return {...graph,nodes,edges};
}
export function lifecycleDetail(replay,layer,focus,topology){
  const scope=replay.byId.has("l"+layer+":main:12")?"l"+layer+":main:12":null;
  const shared=(id,name,p)=>lifePart(id,name,p,{scopeId:scope});let nodes;
  if(focus==="ple-inject"&&topology.perLayerInput)nodes=[
    shared("ple-gate","Gate projection",[-600,-120,0]),lifePart("ple-input","Prepared PLE slice",[-600,150,0],{shape:"storage"}),
    shared("ple-activate","GELU × PLE",[-230,-120,0]),shared("ple-project","Project to hidden",[140,-120,0]),shared("ple-norm","Post-PLE norm",[510,-120,0]),shared("ple-add","Add to hidden",[510,160,0])];
  else if(focus==="weight-transport")nodes=[lifePart("weight-bank","Packed weight bank",[-600,0,0],{shape:"storage"}),lifePart("tma","TMA transfer",[-200,0,0]),lifePart("ring","Shared-memory ring",[200,0,0],{shape:"storage"}),lifePart("operand","Tensor-core operands",[600,0,0])];
  else if(focus==="activation-stage")nodes=[lifePart("activation-codes","Activation codes",[-400,-120,0],{shape:"storage"}),lifePart("activation-scales","Block scales",[-400,120,0],{shape:"storage"}),lifePart("activation-tile","Codes + scales tile",[100,0,0],{shape:"storage"}),lifePart("activation-ready","Ready for mixer",[500,0,0],{shape:"gate"})];
  else if(focus==="input-norm")nodes=[lifePart("residual-values","Incoming residual",[-500,-100,0],{shape:"storage"}),lifePart("sumsq","Published sumsq",[-500,150,0],{shape:"storage"}),lifePart("norm-fold","Fold + inverse RMS",[-80,0,0]),lifePart("gamma","Apply gamma",[330,0,0])];
  else if(focus==="publish-residual"||focus==="residual-ready")nodes=[shared("residual-hi","Residual hi",[-500,-160,0]),shared("residual-lo","Residual lo",[-500,80,0]),shared("sumsq-publish","Sumsq partials",[-100,220,0]),shared("counter-publish","Publish counter",[-100,-100,0]),lifePart("counter-wait","Wait for residual",[350,-100,0],{shape:"gate",stepId:replay.byId.has("l"+layer+":main:15")?"l"+layer+":main:15":null}),lifePart("next-norm","Next layer n1",[750,-100,0])];
  else if(focus==="kv-cache"){
    const index=replay.layers.findIndex(l=>l.id===layer),owner=topology.owners[index],window=topology.windows?.[index];
    nodes=[lifePart("cache-owner",topology.known?"KV owner L"+owner:"KV owner unknown",[-550,0,0],{shape:owner===layer?"storage":"portal",targetLayer:owner===layer?null:owner}),lifePart("key-storage","K codes + scales",[-150,-140,0],{shape:"storage"}),lifePart("value-storage","V codes + scales",[-150,140,0],{shape:"storage"}),lifePart("history-window",window?"Window "+window:"Window unknown",[300,0,0],{shape:"storage",cacheWindow:window??null,tokenPosition:replay.token??null,description:"Captured window capacity and token position, not a recording of cache contents or earlier updates."}),lifePart("cache-read","Attention reads",[750,0,0])];
  }else return null;
  let edges=nodes.slice(1).map((n,i)=>lifeWire(nodes[i].id,n.id,"logical dependency"));
  if(focus==="ple-inject")edges=[lifeWire("ple-gate","ple-activate","gate"),lifeWire("ple-input","ple-activate","PLE"),lifeWire("ple-activate","ple-project","gated input"),lifeWire("ple-project","ple-norm","projected input"),lifeWire("ple-norm","ple-add","correction")];
  if(focus==="kv-cache")edges=[lifeWire("cache-owner","key-storage","K"),lifeWire("cache-owner","value-storage","V"),lifeWire("key-storage","history-window","keys"),lifeWire("value-storage","history-window","values"),lifeWire("history-window","cache-read","history")];
  if(focus==="activation-stage")edges=[lifeWire("activation-codes","activation-tile","codes"),lifeWire("activation-scales","activation-tile","scales"),lifeWire("activation-tile","activation-ready","publish")];
  if(focus==="publish-residual"||focus==="residual-ready")edges=[lifeWire("residual-hi","counter-publish","store before publish"),lifeWire("residual-lo","counter-publish","store before publish"),lifeWire("sumsq-publish","counter-publish","store before publish"),lifeWire("counter-publish","counter-wait","release / acquire"),lifeWire("counter-wait","next-norm","ready")];
  if(focus==="input-norm")edges=[lifeWire("residual-values","norm-fold","x"),lifeWire("sumsq","norm-fold","sumsq"),lifeWire("norm-fold","gamma","normalized")];
  return {nodes,edges,title:focus,meaning:"Logical structure; only bracketed operations carry their own duration."};
}
export function tokenStations(topology,layers){
  const entry=[lifePart("token-in","Input token",[-850,0,0],{shape:"storage"}),lifePart("embedding","Embedding lookup",[-430,0,0]),...(topology.perLayerInput?[lifePart("ple-prepare","Prepare all PLE slices",[-120,280,0],{shape:"storage",description:"Projected embedding, normalization and token-table addition, before the layer loop."})]:[]),lifePart("seed-residual","Seed residual",[400,0,0],{shape:"storage"})];
  const output=[lifePart("final-norm","Final RMSNorm",[-900,0,0]),lifePart("lm-head","Vocabulary projection",[-450,0,0]),lifePart("selection","Token selection",[0,0,0],{description:"Greedy argmax or the selected sampler. This tape has no selected-token ID or isolated selection duration."}),lifePart("stop-check","Stop-set check",[450,0,0],{shape:"gate",description:"A stop-set match freezes the next-step inputs; otherwise position and KV count advance. The tape does not record which branch was taken."}),lifePart("advance","Advance position / KV",[450,280,0]),lifePart("token-out","Next token",[0,280,0],{shape:"storage",description:"Logical feedback. Its value and next execution are not recorded here."})];
  const e=[lifeWire("token-in","embedding","token ID"),lifeWire("embedding","seed-residual","hidden")];if(topology.perLayerInput)e.push(lifeWire("embedding","ple-prepare","embedding + token row"));
  const o=[lifeWire("final-norm","lm-head","normalized hidden"),lifeWire("lm-head","selection","logits"),lifeWire("selection","stop-check","selected ID"),lifeWire("selection","token-out","selected ID"),lifeWire("stop-check","advance","advance unless stopped"),lifeWire("advance","token-out","publish next-step state")];
  const last=layers.at(-1);
  return [{id:"entry",title:"TOKEN INTAKE",kind:"station",name:"Untimed entry",origin:[-3300,0,0],nodes:entry,edges:e,timing:null,steps:[]},{id:"exit",title:"TOKEN OUTPUT",kind:"station",name:"Untimed output",origin:[last.origin[0]+3300,last.origin[1],0],nodes:output,edges:o,timing:null,steps:[]}];
}
export function lifecycleRoutes(layers,topology){
  const end=(area,node)=>({area,node}),routes=[];
  for(let i=1;i<layers.length;i++)routes.push({from:end(layers[i-1].id,"residual-ready"),to:end(layers[i].id,"input"),kind:"handoff",label:"Residual + readiness"});
  routes.push({from:end("entry","seed-residual"),to:end(layers[0].id,"input"),kind:"handoff",label:"Seed residual"},{from:end(layers.at(-1).id,"residual-ready"),to:end("exit","final-norm"),kind:"handoff",label:"Final hidden"},{from:end("exit","token-out"),to:end("entry","token-in"),kind:"feedback",label:"Next token · logical return"});
  for(const [i,l] of layers.entries()){if(topology.owners[i]!==l.id)routes.push({from:end(topology.owners[i],"kv-cache"),to:end(l.id,"kv-cache"),kind:"kv-share",label:"Shared KV from L"+topology.owners[i]});if(topology.perLayerInput)routes.push({from:end("entry","ple-prepare"),to:end(l.id,"ple-inject"),kind:"ple",label:"Prepared PLE slice"});}
  return routes;
}
