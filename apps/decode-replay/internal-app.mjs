// One navigable world over recorded observations. Evidence opens only on request.
let internalController=null;
function mountInternal(bundle){
  const captures=bundle.timelines.filter(t=>t.internalReplay).map(normalizeInternalCapture);
  const plans=captures.map(c=>{if(c.captureTopology&&c.launchBinding?.launch_tables_sha256&&c.captureTopology.bindingSha!==c.launchBinding.launch_tables_sha256)throw new Error("Captured topology does not match the launch binding.");return worldPlan(c.replay,c.captureTopology);});
  internalController?.stop();
  const host=document.getElementById("internal");host.replaceChildren();document.body.classList.add("decode-world");
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  if(!plans.length){const empty=el("div","No internal recording. Source and launch evidence is available below.","world-empty"),open=el("button","Open recording");open.onclick=()=>document.getElementById("file").click();empty.append(open);host.append(empty);document.querySelector(".supporting").open=true;return;}
  let plan=plans[0],replay=plan.replay,time=plan.sequence.start,layer=replay.layers[0].id,other=replay.layers[1]?.id??layer;
  let playing=false,raf=0,last=0,comparison=false,follow=true,overview=false,pane=0,focus=[null,null],history=[[],[]];
  const abort=new AbortController();
  host.innerHTML=`
  <div class="world-bar"><div class="world-brand">QGRE <span>Decode world</span></div><label class="world-recording"><span class="sr-only">Recording</span><select id="world-recording"></select></label><div class="world-bar-actions"><button id="world-evidence">Evidence</button><button id="world-import">Open recording</button></div></div>
  <div class="world-tools"><nav aria-label="Scene path" id="world-path"></nav><div class="world-view-tools"><button id="world-entry">Intake</button><button id="world-home">World</button><button id="world-exit">Output</button><button id="world-snapshots">Snapshots</button><button id="world-follow" aria-pressed="true">Follow token</button><button id="world-compare" aria-pressed="false">Compare layers</button><button id="world-minus" aria-label="Zoom out">−</button><button id="world-plus" aria-label="Zoom in">+</button><label class="world-explode">Explode<input id="world-explode" type="range" min="0" max="2" step=".05" value="1"></label><button id="world-help" aria-label="Scene controls">Controls</button></div></div>
  <div class="world-viewport" id="world-viewport">
    <section class="world-pane" id="world-pane-a"><div class="world-pane-label"><label>Area <select id="world-layer-a" aria-label="Left layer"></select></label><output id="world-pane-time-a"></output><button id="world-parts-a">Parts</button></div><canvas id="world-canvas-a" tabindex="0" aria-label="Decode world. Drag to pan, wheel to zoom, Shift-drag to orbit, double-click to enter."></canvas><div class="world-phase-line" id="world-phases-a"></div></section>
    <section class="world-pane" id="world-pane-b" hidden><div class="world-pane-label"><label>Area <select id="world-layer-b" aria-label="Right layer"></select></label><output id="world-pane-time-b"></output><button id="world-parts-b">Parts</button></div><canvas id="world-canvas-b" tabindex="0" aria-label="Second mechanism for side-by-side comparison."></canvas><div class="world-phase-line" id="world-phases-b"></div></section>
    <div class="world-map"><div>Layer map <span>click to travel</span></div><canvas id="world-minimap" width="250" height="160" aria-label="All recorded layers. Layer selectors provide keyboard navigation."></canvas></div>
    <div class="world-key"><span class="world-key-work">Recorded work</span><span class="world-key-wait">Wait</span><span>Dashed · supply / shared / return</span><span>Geometry ≠ chip placement</span></div>
    <div id="world-help-panel" hidden><strong>Move through the mechanism</strong><span>Drag / arrows · pan</span><span>Wheel / + − · zoom</span><span>Shift + drag · orbit</span><span>Double-click · enter part</span><span>Escape · go back</span><span>Space · play / pause</span></div>
    <div id="world-part" class="world-part" hidden></div>
  </div>
  <div class="world-transport"><div class="world-play-controls"><button id="world-previous" aria-label="Previous recorded step">Previous</button><button id="world-play" class="world-primary">Play decode</button><button id="world-next" aria-label="Next recorded step">Next</button></div><output id="world-status"></output><output id="world-clock"></output><label class="world-speed"><span class="sr-only">Playback speed</span><select id="world-speed"><option value="30">30 s / token</option><option value="10">10 s / token</option><option value="60">60 s / token</option><option value="120">120 s / token</option></select></label><button id="world-steps">Steps</button><input id="world-scrub" type="range" min="0" max="10000" step="1" value="0" aria-label="Recorded token time"><div id="world-layer-strip" class="world-layer-strip"></div><div class="world-footnote"><span>Recorded globaltimer · instrumented</span><span>Intake / output / transfer: untimed</span><span id="world-count"></span></div></div>
  <aside id="world-drawer" hidden><div class="world-drawer-head"><h2 id="world-drawer-title">Recorded steps</h2><button id="world-drawer-close">Close</button></div><div id="world-drawer-body"></div></aside>`;
  const $w=id=>host.querySelector("#world-"+id),btn=(text,fn,cls)=>{const b=el("button",text,cls);b.onclick=fn;return b;};
  const scenes=["a","b"].map((k,i)=>new DecodeWorldScene($w("canvas-"+k),{select:h=>choosePart(h,i),enter:h=>enterPart(h,i),manual:()=>{follow=false;overview=false;syncTools();}}));
  const map=$w("minimap"),mc=map.getContext("2d"),mapHits=[];
  const options=(select,items)=>{select.replaceChildren();for(const [value,label] of items){const o=el("option",label);o.value=value;select.append(o);}};
  options($w("recording"),captures.map((c,i)=>[String(i),"Token "+c.replay.token+" · "+(bundle.label??"recorded decode").split(" · ")[0]]));
  function selectedLayer(i){return i===0?layer:other;}
  function stop(){playing=false;cancelAnimationFrame(raf);$w("play").textContent="Play decode";}
  function syncTools(){for(const [id,value] of [["follow",follow],["compare",comparison],["home",overview]])$w(id).setAttribute("aria-pressed",String(value));}
  function fillSelectors(){for(const k of ["a","b"])options($w("layer-"+k),plan.areas.map(l=>[l.id,l.kind==="station"?l.title:l.id+" · "+(l.name??l.kind)]));$w("layer-a").value=layer;$w("layer-b").value=other;$w("count").textContent=replay.layers.length+" layers · "+replay.steps.length+" recorded spans";}
  function frameScene(i){
    scenes[i].detail=!overview||comparison||!!focus[i];
    if(innerWidth<700&&!overview&&!focus[i]){
      const area=scenes[i].layers.find(l=>l.id===selectedLayer(i)),anchor=area.nodes.find(n=>n.id===(area.timing?"mixer":area.id==="entry"?"embedding":"selection"));
      if(anchor){scenes[i].frame(worldBounds([{...area,nodes:area.nodes.filter(n=>n.id===anchor.id||n.parentPart===anchor.id)}]));return;}
    }
    if(i===1||comparison||focus[i]){scenes[i].frame(worldBounds(scenes[i].layers));return;}
    if(overview){const b=worldBounds(plan.areas);scenes[0].frame([b[0]-400,b[1],b[2],b[3]+450]);return;}
    const index=plan.areas.findIndex(l=>l.id===layer),count=innerWidth>1900&&plan.areas[index].timing?2:1;
    scenes[0].frame(worldBounds(plan.areas.slice(index,index+count)));
  }
  function renderScenes(frame=true){
    $w("pane-b").hidden=!comparison;$w("viewport").classList.toggle("comparing",comparison);
    scenes[0].set(comparison||focus[0]?[worldView(plan,layer,focus[0])]:plan.areas,comparison||focus[0]?[]:plan.routes,overview?null:layer);scenes[1].set([worldView(plan,other,focus[1])]);
    for(const [i,k] of ["a","b"].entries()){$w("layer-"+k).value=selectedLayer(i);const l=plan.areas.find(l=>l.id===selectedLayer(i));$w("pane-time-"+k).textContent=l.timing?formatTime(l.timing.end-l.timing.start):"Untimed structure";if(frame)frameScene(i);}
    renderPath();renderPhases();syncTools();paint();
  }
  function renderPath(){const nav=$w("path");nav.replaceChildren(btn("Decode",goWorld));nav.append(el("span","/"),btn(plan.stations.find(s=>s.id===selectedLayer(pane))?.title??"Layer "+selectedLayer(pane),()=>{focus[pane]=null;history[pane]=[];overview=false;renderScenes();}));for(const [i,h] of history[pane].entries())nav.append(el("span","/"),btn(h.name,()=>{focus[pane]=h.id;history[pane]=history[pane].slice(0,i+1);renderScenes();}));}
  function goWorld(){stop();comparison=false;focus=[null,null];history=[[],[]];$w("part").hidden=true;pane=0;overview=true;follow=false;renderScenes();}
  function chooseLayer(id,i=0,seek=true){stop();pane=i;if(i===0)layer=id;else other=id;focus[i]=null;history[i]=[];overview=false;follow=false;$w("part").hidden=true;const timing=plan.areas.find(l=>l.id===id).timing;if(seek&&timing)time=timing.start;renderScenes(false);frameScene(i);paint();}
  function choosePart(hit,i){
    pane=i;if(i===0)layer=hit.layer;else other=hit.layer;$w("layer-"+(i?"b":"a")).value=hit.layer;scenes[i].selected={layer:hit.layer,id:hit.node.id};renderPath();
    const box=$w("part"),n=hit.node,e=n.evidence;box.replaceChildren(btn("×",()=>{box.hidden=true;scenes[i].selected=null;scenes[i].draw();},"world-part-close"),el("strong",n.name),el("span",e.kind==="measured"&&e.duration!==null?formatTime(e.duration)+" · recorded":e.kind==="shared"?"Shared timing · not separately measured":"Untimed"));
    if(n.expandable)box.append(btn("Enter part",()=>enterPart(hit,i),"world-primary"));if(n.targetLayer!=null)box.append(btn("Travel to owner L"+n.targetLayer,()=>chooseLayer(n.targetLayer,i,false),"world-primary"));box.append(btn(e.step?"Block evidence":"Structure",()=>showEvidence(n)));box.hidden=false;scenes[i].draw();
  }
  function enterPart(hit,i){if(!hit.node.expandable){choosePart(hit,i);return;}stop();pane=i;if(i===0)layer=hit.layer;else other=hit.layer;focus[i]=hit.node.id;history[i].push({id:hit.node.id,name:hit.node.name});follow=false;overview=false;$w("part").hidden=true;renderScenes();}
  function back(){if(history[pane].length){history[pane].pop();focus[pane]=history[pane].at(-1)?.id??null;renderScenes();}else goWorld();}
  function renderPhases(){
    const max=Math.max(1,...(comparison?[layer,other]:[layer]).map(id=>{const t=plan.areas.find(l=>l.id===id).timing;return t?t.end-t.start:0;}));
    for(const [i,k] of ["a","b"].entries()){const rail=$w("phases-"+k),l=plan.areas.find(l=>l.id===selectedLayer(i));rail.replaceChildren();rail.hidden=!l.timing;for(const item of l.timing?.steps??[]){const b=btn("",()=>{stop();time=item.start;chooseStep(item.step,i);},item.step.kind);b.style.left=(item.start-l.timing.start)/max*100+"%";b.style.width=Math.max(.15,(item.end-item.start)/max*100)+"%";b.title=item.step.name+" · "+formatTime(item.end-item.start);b.setAttribute("aria-label",b.title);rail.append(b);}rail.title=comparison?"Same time scale in both panes":"Recorded root-step spans";}
  }
  function chooseStep(s,i=0){pane=i;if(i===0)layer=s.layer;else other=s.layer;focus[i]=s.parent??null;history[i]=s.parent?[{id:s.parent,name:replay.byId.get(s.parent).name}]:[];follow=false;overview=false;renderScenes();const view=focus[i]||comparison?scenes[i].layers[0]:scenes[i].layers.find(l=>l.id===s.layer),found=view.nodes.find(n=>n.stepId===s.id);choosePart({layer:s.layer,node:found??{id:s.id,name:s.name,evidence:mechanismEvidence(replay,{stepId:s.id})}},i);}
  function showSnapshots(){
    const body=showDrawer("Independent recorded snapshots");
    body.append(el("p","Each tile opens its own clock. No execution or cache updates are reconstructed between recordings."));
    for(const [i,c] of captures.entries()){
      const p=plans[i],row=btn("",()=>{$w("recording").value=String(i);$w("recording").onchange();},"world-snapshot");
      row.append(el("strong","Position "+c.replay.token),el("span",p.layers.length+" layers · "+p.replay.steps.length+" measured spans"));
      const windows=[...new Set(p.topology.windows??[])];
      row.append(el("span",p.topology.known?"Cache windows "+(windows.join(" / ")||"not supplied")+" · "+p.routes.filter(r=>r.kind==="kv-share").length+" shared-KV consumers":"Captured cache topology unavailable"));
      row.append(el("small",c.source.path));body.append(row);
    }
  }
  function showDrawer(title){$w("drawer-title").textContent=title;$w("drawer-body").replaceChildren();$w("drawer").hidden=false;return $w("drawer-body");}
  for(const [i,k] of ["a","b"].entries())$w("parts-"+k).onclick=()=>{
    pane=i;const area=worldView(plan,selectedLayer(i),focus[i]),body=showDrawer("Parts · "+(area.kind==="station"?area.title:"Layer "+area.id));
    for(const node of area.nodes.filter(n=>!n.parentPart))body.append(btn(node.name,()=>{$w("drawer").hidden=true;choosePart({layer:area.id,node},i);scenes[i].frame(worldBounds([{...area,origin:comparison||focus[i]?[0,0,0]:plan.areas.find(l=>l.id===area.id).origin,nodes:[node]}]));},"world-step-row"));
  };
  function showSteps(){const area=plan.areas.find(l=>l.id===selectedLayer(pane)),body=showDrawer(area.timing?"Layer "+area.id+" · recorded steps":area.title+" · untimed parts");if(!area.timing)for(const node of area.nodes)body.append(btn(node.name,()=>choosePart({layer:area.id,node},pane),"world-step-row"));for(const {step,metrics:m} of area.steps){const b=btn("",()=>{stop();time=m.end-m.critical;chooseStep(step,pane);});b.className="world-step-row"+(step.parent?" child":"");b.append(el("span",step.name),el("code",formatTime(m.critical)));body.append(b);}}
  function showEvidence(n){const e=n.evidence,body=showDrawer(n.name);body.append(el("p",n.description??e.step?.description??"Logical structure; this tape has no separate timing for this part."));if(n.cacheWindow)body.append(el("p","Captured window: "+n.cacheWindow+" positions · snapshot position "+n.tokenPosition+". Contents and earlier updates are not recorded."));if(!e.step)return;body.append(el("p",e.kind==="shared"?"These operations share the enclosing span.":"Duration is latest exit minus latest entry; not an exclusive cost."));if(n.equation)body.append(el("code",n.equation));const table=el("table"),head=el("tr");for(const t of ["Block / SM","Start","End","Cycles"])head.append(el("th",t));table.append(head);for(const o of e.metrics?.observations??[]){const row=el("tr");for(const t of [o.block+" / "+(o.sm??"—"),o.startNs,o.endNs,o.cycles])row.append(el("td",t));table.append(row);}body.append(table);}
  function renderStrip(){const strip=$w("layer-strip");strip.replaceChildren();for(const l of plan.layers){const b=btn(l.id,()=>chooseLayer(l.id));b.dataset.layer=l.id;b.title="Layer "+l.id+" · "+formatTime(l.timing.end-l.timing.start);b.setAttribute("aria-label",b.title);b.style.flexGrow=String(Math.max(.25,(l.timing.end-l.timing.start)/plan.sequence.duration*plan.layers.length));strip.append(b);}}
  function paintMap(){
    mc.clearRect(0,0,250,160);mapHits.length=0;const bounds=worldBounds(plan.areas),scale=Math.min(225/(bounds[2]-bounds[0]),130/(bounds[3]-bounds[1])),point=(x,y)=>[12+(x-bounds[0])*scale,12+(y-bounds[1])*scale];mc.strokeStyle="#435e69";mc.beginPath();for(const [i,l] of plan.areas.entries()){const p=point(...l.origin);if(i)mc.lineTo(...p);else mc.moveTo(...p);}mc.stroke();
    for(const l of plan.areas){const b=worldBounds([l]),[x,y]=point(b[0],b[1]),w=(b[2]-b[0])*scale,h=(b[3]-b[1])*scale;mc.fillStyle=l.id===layer?"#259f9d":comparison&&l.id===other?"#bf8134":"#3d535e";mc.fillRect(x,y,w,h);mc.fillStyle="#fff";mc.font="10px system-ui";mc.textAlign="center";mc.fillText(l.id==="entry"?"IN":l.id==="exit"?"OUT":l.id,x+w/2,y+h/2+3);mapHits.push({x,y,w,h,id:l.id});}
    if(!comparison&&!focus[0]){const a=scenes[0],points=[[0,0],[a.w,0],[a.w,a.h],[0,a.h]].map(p=>point(...a.unproject(...p)));mc.beginPath();points.forEach((p,i)=>i?mc.lineTo(...p):mc.moveTo(...p));mc.closePath();mc.strokeStyle="#e1edf1";mc.stroke();}
  }
  map.onclick=e=>{const r=map.getBoundingClientRect(),x=(e.clientX-r.left)*250/r.width,y=(e.clientY-r.top)*160/r.height,h=mapHits.find(v=>x>=v.x&&x<=v.x+v.w&&y>=v.y&&y<=v.y+v.h);if(h)chooseLayer(h.id);};
  function followClock(){const p=decodePosition(plan.sequence,time),target=p.complete?"exit":p.layer;if(follow&&!comparison&&target!==layer){layer=target;focus[0]=null;history[0]=[];$w("part").hidden=true;renderScenes();}}
  function paint(){const s=plan.sequence,p=decodePosition(s,time);$w("clock").textContent=formatTime(time-s.start)+" / "+formatTime(s.duration);$w("scrub").value=String((time-s.start)/Math.max(1,s.duration)*10000);$w("status").textContent=p.complete?"Layer tape complete · output untimed":p.gap?"Between layers":"L"+p.layer+" · "+(replay.byId.get(p.stepId)?.name??"recorded boundary");for(const b of $w("layer-strip").children){const l=plan.layers.find(l=>l.id===b.dataset.layer);b.dataset.state=time>=l.timing.end?"complete":p.layer===l.id?"active":"pending";b.setAttribute("aria-pressed",String(l.id===layer));}for(const [i,scene] of scenes.entries()){scene.time=time;if(i===0||comparison)scene.draw();}paintMap();}
  function tick(t){if(!playing)return;if(last)time=Math.min(plan.sequence.end,time+Math.min(t-last,100)/1000*plan.sequence.duration/Number($w("speed").value));last=t;followClock();paint();if(time>=plan.sequence.end)stop();else raf=requestAnimationFrame(tick);}
  function play(){if(playing){stop();return;}if(time>=plan.sequence.end)time=plan.sequence.start;if(!comparison){follow=true;overview=false;focus[0]=null;history[0]=[];layer=decodePosition(plan.sequence,time).layer;renderScenes();}playing=true;last=0;$w("play").textContent="Pause";raf=requestAnimationFrame(tick);}
  $w("entry").onclick=()=>chooseLayer("entry");$w("exit").onclick=()=>chooseLayer("exit");$w("snapshots").onclick=showSnapshots;
  $w("play").onclick=play;$w("home").onclick=goWorld;$w("follow").onclick=()=>{follow=!follow;if(follow){comparison=false;overview=false;focus=[null,null];history=[[],[]];layer=decodePosition(plan.sequence,time).layer;renderScenes();}syncTools();};
  $w("compare").onclick=()=>{stop();if(!plan.layers.some(l=>l.id===layer))layer=decodePosition(plan.sequence,time).layer;if(!plan.layers.some(l=>l.id===other))other=plan.layers[0].id;comparison=!comparison;overview=false;follow=false;focus=[null,null];history=[[],[]];if(layer===other)other=plan.layers[(plan.layers.findIndex(l=>l.id===layer)+1)%plan.layers.length].id;pane=0;renderScenes();};
  $w("layer-a").onchange=()=>chooseLayer($w("layer-a").value,0);$w("layer-b").onchange=()=>chooseLayer($w("layer-b").value,1,false);
  for(const [key,d] of [["previous",-1],["next",1]])$w(key).onclick=()=>{stop();time=seekDecodeStep(plan.sequence,time,d);followClock();paint();};
  $w("scrub").oninput=()=>{stop();time=plan.sequence.start+Number($w("scrub").value)/10000*plan.sequence.duration;followClock();paint();};
  $w("minus").onclick=()=>{follow=false;syncTools();scenes[pane].zoom(.8);};$w("plus").onclick=()=>{follow=false;syncTools();scenes[pane].zoom(1.25);};
  $w("explode").oninput=()=>{for(const scene of scenes){scene.explode=Number($w("explode").value);scene.draw();}};
  $w("help").onclick=()=>{$w("help-panel").hidden=!$w("help-panel").hidden;};$w("steps").onclick=showSteps;$w("drawer-close").onclick=()=>{$w("drawer").hidden=true;};
  $w("import").onclick=()=>document.getElementById("file").click();$w("evidence").onclick=()=>{const support=document.querySelector(".supporting");support.open=!support.open;};
  $w("recording").onchange=()=>{stop();plan=plans[Number($w("recording").value)];replay=plan.replay;time=plan.sequence.start;layer=replay.layers[0].id;other=replay.layers[1]?.id??layer;focus=[null,null];history=[[],[]];comparison=false;overview=false;follow=true;pane=0;$w("part").hidden=true;$w("drawer").hidden=true;fillSelectors();renderStrip();renderScenes();};
  document.addEventListener("keydown",e=>{if(["INPUT","SELECT","TEXTAREA","BUTTON"].includes(e.target.tagName))return;if(e.code==="Space"){e.preventDefault();play();}else if(e.key==="Escape"){if(!$w("drawer").hidden)$w("drawer").hidden=true;else if(document.querySelector(".supporting").open)document.querySelector(".supporting").open=false;else back();}},{signal:abort.signal});
  window.addEventListener("resize",()=>{if(scenes[0].frameBounds)frameScene(0);if(comparison&&scenes[1].frameBounds)frameScene(1);},{signal:abort.signal});
  for(const scene of scenes)scene.onDraw=paintMap;internalController={stop:()=>{stop();abort.abort();for(const scene of scenes)scene.dispose();}};
  fillSelectors();renderStrip();renderScenes();
}
