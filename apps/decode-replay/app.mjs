// UI state stays in this offline page. No network requests or profiler uploads.
const $=id=>document.getElementById(id);
let data, model=null, timeline=null, time=0, playing=false, lastFrame=0, page=0, selectedEvent=null, selectedNode=null, focusNode=null, nodePage=0;
let rangeStart=0, rangeEnd=1;
let timelinePaint=null, timelinePaintKey='', recordedEvents=[];
const groupNames={index:'Indexing',stage:'Shared memory',project:'Projection',embed:'Embedding',elementwise:'Elementwise',norm:'Normalization',instr:'Instructions',kv:'KV cache',rope:'RoPE',attention:'Attention',cpt:'Training blocks',gather:'Gather',sample:'Sampling'};
const pageSize=()=>innerWidth<700?4:16;
const nodeLabel=n=>n.group?(groupNames[n.label]??n.label):n.label??n.id;
const modelLabel=m=>(m?.label??m?.id??"Source composition").replace(/^gemma4 /,"Gemma 4 ").replace(/^qwen38 /,"Qwen3.8 ").replace(/\b(e?\d+)b\b/gi,s=>s.toUpperCase());
const captureLabel=(t,i)=>`${/qwen/i.test(t.label)?"Qwen":/gemma/i.test(t.label)?"Decode":"Trace"} recording ${i+1}`;
const scene=new AssemblyScene($("scene"),selectNode);
const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;};
function option(select,value,label){const o=el("option",label);o.value=value;select.append(o);}
function explainError(error){$("error").textContent=error.message??String(error);$("error").hidden=false;}
function load(raw){
  const next=normalizeBundle(raw);mountInternal(next);pause();data=next;time=0;page=0;selectedEvent=null;selectedNode=null;focusNode=null;$("error").hidden=true;
  $("run-title").textContent=data.label||"Recorded decode";$("revision").textContent=data.architecture?.dirty?"Source snapshot · includes working changes":"Source snapshot · committed";
  $("model").replaceChildren();for(const [i,m] of data.models.entries())option($("model"),String(i),modelLabel(m));
  if(!data.models.length)option($("model"),"","No source composition");
  $("capture").replaceChildren();for(const [i,t] of data.timelines.entries())option($("capture"),String(i),captureLabel(t,i));
  if(!data.timelines.length)option($("capture"),"","No recorded timeline");
  $("counter").replaceChildren();for(const [i,c] of data.counters.entries())option($("counter"),String(i),`${c.source?.path?.split("/").pop()??"NCU"} · launch ${c.launchId} · ${c.name}`);
  if(!data.counters.length)option($("counter"),"","No NCU capture");
  const limits=$("limitations");limits.replaceChildren();
  const descriptions=[
    ["Architecture",data.architecture?.binding??"Source snapshot only."],
    ["Recorded execution","The timeline shows only events present in the selected recording. Different clock domains are not merged. Playback speed is a viewing control, not a performance estimate."],
    ["Counter evidence","Inspect SM/DRAM throughput, stalls and issue metrics in NCU counters. Missing metrics are not zero. Aggregates do not identify the critical path by themselves."],
    ["Inside one launch","Without device timestamps, the interior of a persistent megakernel is unobserved. A launch is not necessarily one token. Phase tapes can expose instrumented intervals, but not production savings."],
    ...((data.limitations??[]).map((x,i)=>[`Evidence limit ${i+1}`,x]))
  ];
  for(const [title,body] of descriptions){const section=el("section");section.append(el("h3",title),el("p",body));limits.append(section);}
  chooseModel();chooseTimeline();showCounters();showArchitecture();
}
function currentNodes(){
  if($("view").value==="recorded"){
    const all=recordedEvents;
    let index=all.findIndex(e=>e.end>time);if(index<0)index=Math.max(0,all.length-12);
    return all.slice(Math.max(0,index-2),Math.max(0,index-2)+12).map(e=>({id:e.id,label:e.name.replace(/^.*silicon_l4_/,"").replace(/_decode_step.*$/," decode"),level:0,kind:e.kind,event:e}));
  }
  if(!model)return[];
  return focusNode?[focusNode,...model.nodes.filter(n=>n.parent===focusNode.id)]:model.nodes.filter(n=>!n.parent);
}
function refreshScene(){
  const all=currentNodes(), recorded=$("view").value==="recorded";const size=pageSize();nodePage=Math.min(nodePage,Math.max(0,Math.ceil(all.length/size)-1));const nodes=(recorded?all.slice(0,size):all.slice(nodePage*size,(nodePage+1)*size)).map(n=>({...n,label:nodeLabel(n),childCount:n.group?model.nodes.filter(c=>c.parent===n.id).length:undefined}));
  scene.selected=selectedNode?.id;scene.setData(nodes,recorded?[]:model?.edges??[]);
  $("graph-count").textContent=recorded?`${nodes.length} nearby recorded events`:`${nodes.length} shown / ${all.length} in this group · ${model?.nodes?.length??0} total nodes`;
  $("assembly-subtitle").textContent=recorded?"Measured event order · not SM placement":"Static composition · not SM placement";
  $("scene-caption").textContent=recorded?"Each block is a recorded interval; active intervals are filled.":"Source-derived pieces. No invented internal timings.";
  $("components").replaceChildren();
  if(focusNode&&!recorded){const back=el("button","Back to assembly");back.onclick=()=>{focusNode=model.nodes.find(n=>n.id===focusNode.parent)??null;nodePage=0;refreshScene();showArchitecture();};$("components").append(back);}
  if(!recorded&&all.length>size){for(const [label,delta] of [["Previous pieces",-1],["Next pieces",1]]){const b=el("button",label);b.disabled=delta<0?nodePage===0:(nodePage+1)*size>=all.length;b.onclick=()=>{nodePage+=delta;refreshScene();};$("components").append(b);}}
  for(const node of nodes){const b=el("button",node.label??node.id);b.setAttribute("aria-pressed",String(node.id===selectedNode?.id));b.onclick=()=>selectNode(node);$("components").append(b);}
}
function chooseModel(){model=data.models[Number($("model").value)]??null;selectedNode=null;focusNode=null;nodePage=0;$("assembly-title").textContent=modelLabel(model);$("layer").replaceChildren();
  // Only layers explicitly present in the source config are listed. Selecting one does not invent a dynamic schedule.
  const layers=model?.layers??[];option($("layer"),"","All declared slots");for(const [i,layer] of layers.entries())option($("layer"),String(i),`Layer ${i} · ${layer.kind??layer}`);
  refreshScene();showArchitecture();
}
function detail(label,value){const d=$("inspection").querySelector("dl")??$("inspection").appendChild(el("dl"));d.append(el("dt",label),el("dd",value===undefined||value===null?"Not recorded":String(value)));}
function sourceLink(path,line){const node=el("a",`${path}${line?":"+line:""}`,"source-link");node.href=`vscode://file/${String(path).replace(/^\//,"")}${line?":"+line:""}`;return node;}
function showArchitecture(){
  $("evidence-label").textContent="Static source";$("inspection").replaceChildren(el("h3",modelLabel(model),"inspection-title"),el("p","Choose a module group to open its typed building blocks."));
  if(model){
    const levels=[3,2,1].map(level=>[level,model.nodes.filter(n=>n.level===level).length]);
    const max=Math.max(1,...levels.map(([,n])=>n));
    const counts=el("div",undefined,"tier-counts");
    for(const [level,count] of levels){const row=el("div"),label=el("span",`L${level} · ${count} pieces`),bar=el("i");bar.style.width=`${count/max*100}%`;row.append(label,bar);counts.append(row);}
    $("inspection").append(counts);detail("Recipe",model.id);detail("Policy",model.policy);detail("Production entry",model.entry);detail("Declared layer pattern",model.layers?.length?`${model.layers.length} layers; choose a layer to inspect its declared kind.`:"Not included in this snapshot");
    detail("Source revision",data.architecture?.revision);const provenance=el("details"),provenanceTitle=el("summary","Source identity and policy");provenance.append(provenanceTitle,$("inspection").querySelector("dl"));$("inspection").append(provenance);
    const d=el("details"),s=el("summary","Model configuration");d.append(s,el("pre",JSON.stringify(model.config??{},null,2)));$("inspection").append(d);}
}
function selectNode(node){
  if(node.event){selectEvent(node.event);return;}
  selectedNode=node;selectedEvent=null;scene.selected=node.id;scene.draw();refreshScene();$("evidence-label").textContent="Source operation";
  $("inspection").replaceChildren(el("h3",node.label??node.id,"inspection-title"),el("p",node.group?"A source-directory grouping, not an execution stage. Open it to inspect its typed invocations.":"A source invocation. Its presence is not evidence of when it ran."));
  detail("Instance",node.id);detail("Module",node.module??node.source);detail("Pyramid tier",`L${node.level??"?"}`);detail("Source location",node.location);
  if(node.source)$("inspection").append(sourceLink(node.source,node.line));
  const bindings=node.bindings??[];if(bindings.length){$("inspection").append(el("h3","Typed ports"));const ul=el("ul");for(const b of bindings)ul.append(el("li",typeof b==="string"?b:`${b.direction} ${b.port}: ${b.type} ← ${b.value}`));$("inspection").append(ul);}
  const children=model.nodes.filter(n=>n.parent===node.id);if(children.length){const b=el("button",`Open composition · ${children.length} pieces`);b.onclick=()=>{focusNode=node;nodePage=0;refreshScene();};$("inspection").append(b);}
}
function chooseTimeline(){pause();timeline=data.timelines[Number($("capture").value)]??null;time=0;page=0;selectedEvent=null;rangeStart=0;rangeEnd=timeline?.duration||1;timelinePaintKey="";recordedEvents=timeline?.events.filter(e=>!["api","driver"].includes(e.kind))??[];
  $("clock-note").textContent=timeline?`${timeline.evidence} · independent ${timeline.unit==="ns"?"nanosecond":timeline.unit} clock`:"No measured timeline. Source composition remains inspectable; NCU counters, if imported, are available below.";
  $("event-count").textContent=timeline?`${timeline.events.length.toLocaleString()} recorded events`:"No events";
  $("duration").textContent=timeline?formatTime(timeline.duration,timeline.unit):"—";
  for(const id of ["play","previous","next","scrubber"])$(id).disabled=!timeline?.events.length;
  renderEvents();renderTime();
}
function pause(){playing=false;$("play").textContent="Play";}
function selectEvent(event){selectedEvent=event;selectedNode=null;time=event.start;if(time<rangeStart||time>rangeEnd){rangeStart=0;rangeEnd=timeline.duration||1;}pause();renderTime();renderEvents();$("evidence-label").textContent="Measured interval";
  $("inspection").replaceChildren(el("h3",event.name,"inspection-title"),el("p",timeline?.evidence));
  detail("Duration",formatTime(event.end-event.start,timeline.unit));detail("Clock origin",`${timeline.origin} ${timeline.unit}`);detail("Absolute start",event.startNs);detail("Absolute end",event.endNs);
  for(const key of ["kind","device","stream","process","thread","correlation","gridX","gridY","gridZ","blockX","blockY","blockZ","registersPerThread","staticSharedMemory","dynamicSharedMemory","token","layer","tap","block"])if(event[key]!==undefined&&event[key]!==null)detail(key,event[key]);
  detail("Capture",timeline.source?.path);detail("Source binding",event.operationId??"Not established — no automatic name match.");
}
function renderEvents(){const list=filteredEvents(timeline,$("search").value);if($("event-order").value==="duration")list.sort((a,b)=>(b.end-b.start)-(a.end-a.start));page=Math.min(page,Math.max(0,Math.ceil(list.length/20)-1));$("events").replaceChildren();$("table-count").textContent=`${list.length.toLocaleString()} matches`;
  $("page-back").disabled=page===0;$("page-next").disabled=(page+1)*20>=list.length;
  for(const e of list.slice(page*20,(page+1)*20)){const tr=el("tr");tr.dataset.selected=String(e.id===selectedEvent?.id);const cell=el("td"),b=el("button",e.name);b.onclick=()=>selectEvent(e);cell.append(b);tr.append(cell,el("td",`${e.kind} / ${e.stream??e.block??e.thread??"—"}`),el("td",formatTime(e.start,timeline.unit)),el("td",formatTime(e.end-e.start,timeline.unit)),el("td","Recorded"));$("events").append(tr);}
  if(!list.length){const row=el("tr"),cell=el("td",timeline?"No events match this search.":"No event recording loaded. Import an nsys capture or phase tape; NCU is available in its own tab.","empty");cell.colSpan=5;row.append(cell);$("events").append(row);}
}
function showCounters(){const c=data.counters[Number($("counter").value)];$("metrics").replaceChildren();for(const m of c?.metrics??[]){const r=el("tr");r.append(el("td",m.name),el("td",m.value||"Not collected"),el("td",m.unit||"Unspecified"));$("metrics").append(r);}
  if(!c){const r=el("tr"),cell=el("td","No NCU capture loaded.","empty");cell.colSpan=3;r.append(cell);$("metrics").append(r);}
}
function renderTime(){
  $("time").textContent=timeline?`${formatTime(time,timeline.unit)} / ${formatTime(timeline.duration,timeline.unit)}`:"No timeline";
  $("scrubber").value=String((time-rangeStart)/(rangeEnd-rangeStart)*10000);
  $("focus-time").disabled=!selectedEvent;$("full-time").disabled=!timeline;
  $("duration").textContent=timeline?formatTime(rangeEnd,timeline.unit):"—";document.querySelector(".axis span").textContent=timeline?formatTime(rangeStart,timeline.unit):"0";
  // Only an explicit exported operation identity may animate an internal slot.
  const active=activeEvents(timeline,time);
  scene.active=new Set(active.map(e=>$("view").value==="recorded"?e.id:e.operationId).filter(Boolean));
  if($("view").value==="recorded")refreshScene();else scene.draw();drawTimeline();
}
function drawTimeline(){
  const canvas=$("timeline"),rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);canvas.width=rect.width*ratio;canvas.height=rect.height*ratio;const c=canvas.getContext("2d");c.scale(ratio,ratio);const w=rect.width,h=rect.height;const left=Math.min(130,w*.22);
  c.fillStyle="#ebe7dc";c.fillRect(0,0,w,h);c.font="12px system-ui";c.fillStyle="#596466";
  if(!timeline?.events.length){c.fillText("No recorded time axis",16,h/2);return;}
  const kinds=[...new Set(timeline.events.map(e=>e.kind))],lane=(h-14)/kinds.length,span=Math.max(1,rangeEnd-rangeStart);
  const key=[timeline.id,rangeStart,rangeEnd,w,h,ratio,selectedEvent?.id].join("|");
  if(timelinePaintKey===key&&timelinePaint){c.drawImage(timelinePaint,0,0,w,h);}else{
    for(const [i,kind] of kinds.entries()){c.fillStyle="#252d31";c.fillText(kind,12,12+lane*i+lane/2);c.strokeStyle="#d6d3c8";c.beginPath();c.moveTo(left,lane*(i+1)+6);c.lineTo(w,lane*(i+1)+6);c.stroke();}
    for(const e of timeline.events){if(e.end<rangeStart||e.start>rangeEnd)continue;const i=kinds.indexOf(e.kind),x=left+(w-left-8)*(Math.max(rangeStart,e.start)-rangeStart)/span,width=Math.max(1,(w-left-8)*(Math.min(rangeEnd,e.end)-Math.max(rangeStart,e.start))/span);const isApi=["api","driver"].includes(e.kind);c.fillStyle=e.id===selectedEvent?.id?"#a7442c":isApi?"#7f9eae":"#286b9c";c.fillRect(x,9+i*lane+(isApi?lane*.2:0),width,Math.max(8,lane*(isApi?.45:1)-10));}
    timelinePaint=document.createElement("canvas");timelinePaint.width=canvas.width;timelinePaint.height=canvas.height;timelinePaint.getContext("2d").drawImage(canvas,0,0);timelinePaintKey=key;
  }
  const x=left+(w-left-8)*(time-rangeStart)/span;c.strokeStyle="#a7442c";c.lineWidth=2;c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke();
}
function frame(now){if(playing&&timeline){if(lastFrame){const next=advancePlayback(time,now-lastFrame,Number($("speed").value),rangeStart,rangeEnd);time=next.time;if(next.ended)pause();}renderTime();}lastFrame=now;if(playing)requestAnimationFrame(frame);}
$("play").onclick=()=>{if(playing)pause();else if(timeline){$("view").value="recorded";if(time<rangeStart||time>=rangeEnd)time=rangeStart;playing=true;lastFrame=0;$("play").textContent="Pause";requestAnimationFrame(frame);}};
$("previous").onclick=()=>{pause();time=seekEvent(timeline,time,-1);renderTime();};
$("next").onclick=()=>{pause();time=seekEvent(timeline,time,1);renderTime();};
$("scrubber").oninput=()=>{pause();time=rangeStart+Number($("scrubber").value)/10000*(rangeEnd-rangeStart);renderTime();};
$("timeline").onclick=e=>{if(!timeline)return;pause();const r=e.currentTarget.getBoundingClientRect(),left=Math.min(130,r.width*.22);time=rangeStart+Math.max(0,Math.min(1,(e.clientX-r.left-left)/(r.width-left-8)))*(rangeEnd-rangeStart);renderTime();const active=activeEvents(timeline,time);if(active.length)selectEvent(active[0]);};
$("timeline").onmousemove=e=>{if(!timeline)return;const r=e.currentTarget.getBoundingClientRect(),left=Math.min(130,r.width*.22);const t=rangeStart+Math.max(0,(e.clientX-r.left-left)/(r.width-left-8))*(rangeEnd-rangeStart);const hits=activeEvents(timeline,t);e.currentTarget.title=hits.slice(0,6).map(x=>`${x.name} · ${formatTime(x.end-x.start,timeline.unit)}`).join("\n")||formatTime(t,timeline.unit);};
$("view").onchange=()=>{nodePage=0;refreshScene();renderTime();};
$("model").onchange=chooseModel;$("capture").onchange=chooseTimeline;$("counter").onchange=showCounters;
$("layer").onchange=()=>{showArchitecture();const i=$("layer").value;if(i!==""){const layer=model.layers[Number(i)];detail("Selected layer",i);detail("Declared mixer",layer.kind??layer);$("inspection").append(el("p","This is the checkpoint's declared layer kind. The 3D view retains the compiled slot inventory; unrecorded branches are not animated."));}};
$("explode").oninput=()=>{scene.explode=Number($("explode").value)/100;scene.draw();};$("reset").onclick=()=>scene.reset();
$("focus-time").onclick=()=>{if(!selectedEvent||!timeline)return;pause();const pad=Math.max(1,selectedEvent.end-selectedEvent.start);rangeStart=Math.max(0,selectedEvent.start-pad);rangeEnd=Math.min(timeline.duration,selectedEvent.end+pad);renderTime();};
$("full-time").onclick=()=>{rangeStart=0;rangeEnd=timeline?.duration||1;renderTime();};
$("event-order").onchange=()=>{page=0;renderEvents();};
$("search").oninput=()=>{page=0;renderEvents();};$("page-back").onclick=()=>{page--;renderEvents();};$("page-next").onclick=()=>{page++;renderEvents();};
$("help-toggle").onclick=()=>{$("help").hidden=!$("help").hidden;$("help-toggle").setAttribute("aria-expanded",String(!$("help").hidden));};
for(const id of ["events","counters","limits"])$(id+"-tab").onclick=()=>{for(const other of ["events","counters","limits"]){$(other+"-tab").setAttribute("aria-selected",String(id===other));$(other+"-panel").hidden=id!==other;}};
$("open-bundle").onclick=()=>$("file").click();
$("file").onchange=async()=>{try{const f=$("file").files[0];if(!f)return;if(f.size>100*1024*1024)throw new Error("Bundle exceeds 100 MiB. Export a shorter capture.");load(JSON.parse(await f.text()));}catch(e){explainError(e);}finally{$("file").value="";}};
new ResizeObserver(()=>{drawTimeline();if(data)refreshScene();}).observe($("timeline"));
try{load(JSON.parse($("bundle").textContent));}catch(e){explainError(e);}
