import './style.css';
import {Engine} from '@babylonjs/core/Engines/engine';
import {Scene} from '@babylonjs/core/scene';
import {ArcRotateCamera} from '@babylonjs/core/Cameras/arcRotateCamera';
import {Vector3,Quaternion,Matrix} from '@babylonjs/core/Maths/math.vector';
import {Color3,Color4} from '@babylonjs/core/Maths/math.color';
import {MeshBuilder} from '@babylonjs/core/Meshes/meshBuilder';
import {ShaderMaterial} from '@babylonjs/core/Materials/shaderMaterial';
import {StandardMaterial} from '@babylonjs/core/Materials/standardMaterial';
import {TransformNode} from '@babylonjs/core/Meshes/transformNode';
import {Mesh} from '@babylonjs/core/Meshes/mesh';
import {ImportMeshAsync} from '@babylonjs/core/Loading/sceneLoader';
import {PointerEventTypes} from '@babylonjs/core/Events/pointerEvents';
import '@babylonjs/core/Rendering/outlineRenderer';
import '@babylonjs/core/Shaders/ShadersInclude/instancesDeclaration';
import '@babylonjs/core/Shaders/ShadersInclude/instancesVertex';
import '@babylonjs/loaders/glTF';
import {Timeline,LAYER_COUNT,PROMPT_TOKENS,operationProgress,samplePath,type PartId} from './timeline';
import {createStory,storyCopy,inputPoint,outputPoint,supplyPoint,returnPath} from './story';
import {CARRIER_LABELS,carrierState,carrierPassPosition,weightBank} from './representation';
import {createWeights} from './weights';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const clock=new Timeline(),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
clock.playing=!reduced;
const B=(x:number,y:number,z:number)=>new Vector3(x,z,-y);
const canvas=$<HTMLCanvasElement>('scene');
for(const [j,text]of PROMPT_TOKENS.entries()){const chip=document.createElement('button');chip.className='prompt-token';chip.textContent=text;chip.setAttribute('aria-label',`Inspect ${text.trim()} carrier`);chip.addEventListener('click',()=>inspectCarrier(j));$('prompt-tokens').append(chip);}
for(const [j,text]of CARRIER_LABELS.entries()){const option=document.createElement('option');option.value=String(j);option.textContent=`${j+1} · ${text.trim()}`;$('carrier-select').append(option);}
$<HTMLSelectElement>('carrier-select').value='5';
$('inspector').append($('concurrency'));
let engine:Engine,scene:Scene,camera:ArcRotateCamera;
let selection:PartId='assembly',selectedLayer=0,following=false,userMovedCamera=false,casing=0,casingTarget=0;
let selectedCarrier=5,drawerOpen=0;
function revealCarrierReadout(){if(innerWidth<=650)return;const details=$('carrier-details'),inspector=$('inspector');if(details.getBoundingClientRect().bottom>inspector.getBoundingClientRect().bottom)inspector.scrollTop=details.offsetTop-16;}
function inspectCarrier(index:number){selectedCarrier=index;$<HTMLSelectElement>('carrier-select').value=String(index);$<HTMLDetailsElement>('carrier-details').open=true;requestAnimationFrame(revealCarrierReadout);}
let transition=false,desiredTarget=B(-1,0,.5),desiredRadius=32,desiredAlpha=1.35,desiredBeta=.56;
const heads=[-.82,0,.82];
const centers=Array.from({length:LAYER_COUNT},(_,i)=>{const row=Math.floor(i/8),col=row%2?7-i%8:i%8;return {x:(col-3.5)*4.2,y:(row-1.5)*4,reverse:row%2===1};});
function point(index:number,x:number,y:number,z=1.03){const c=centers[index],sign=c.reverse?-1:1;return B(c.x+x*sign,c.y+y*sign,z);}
const layers:{root:TransformNode;rotors:TransformNode[];cover:TransformNode;coverBase:Vector3;foot:Mesh;label:HTMLButtonElement;jaw:TransformNode;press:TransformNode;jawBase:number;pressBase:number}[]=[];
const marks:Mesh[]=[],headLabels:HTMLDivElement[]=[];
const landmarks:{element:HTMLDivElement;point:Vector3}[]=[];
const routes:{mesh:Mesh;layer:number}[]=[],contextMeshes:Mesh[]=[],routeLabels:HTMLDivElement[]=[];
let residual:Mesh;
const incoming:Vector3[][]=[];
const vertexSource=`precision highp float;
attribute vec3 position;attribute vec3 normal;uniform mat4 viewProjection;
varying vec3 vNormal;
#include<instancesDeclaration>
void main(){
#include<instancesVertex>
vNormal=normalize(mat3(finalWorld)*normal);gl_Position=viewProjection*finalWorld*vec4(position,1.0);
}`;
const fragmentSource=`precision highp float;
varying vec3 vNormal;uniform vec3 baseColor;
void main(){float light=dot(normalize(vNormal),normalize(vec3(-0.4,0.85,0.35)));
float band=light>0.55?1.0:(light>0.02?0.81:0.62);gl_FragColor=vec4(baseColor*band,1.0);}`;
function cel(name:string,color:string){
 const m=new ShaderMaterial(name,scene,{vertexSource,fragmentSource},{attributes:['position','normal'],uniforms:['world','viewProjection','baseColor']});
 m.setColor3('baseColor',Color3.FromHexString(color));return m;
}
function outline(mesh:Mesh,width=.012){mesh.renderOutline=true;mesh.outlineColor=Color3.FromHexString('#33494c');mesh.outlineWidth=width;}
function polyline(name:string,points:Vector3[],material:ShaderMaterial|StandardMaterial,radius=.035){
 const tube=MeshBuilder.CreateTube(name,{path:points,radius,tessellation:8,cap:Mesh.CAP_ALL},scene);tube.material=material;tube.isPickable=false;return tube;
}
function along(path:Vector3[],progress:number){const p=samplePath(path.map(v=>v.asArray()),progress);return new Vector3(...p as [number,number,number]);}
function headPath(i:number,h:number){return [point(i,-.92,0),point(i,-.5,heads[h]),point(i,.55,heads[h])];}
function joinPath(i:number,h:number){return [point(i,.55,heads[h]),point(i,.92,0)];}
function residualPath(i:number){return [point(i,-.92,0),point(i,-1.08,-1.48),point(i,1.08,-1.48),point(i,.92,0)];}
function overviewRadius(){return innerWidth<=650?36:Math.max(27,44/(innerWidth/innerHeight));}
function fitOverview(){
 const previous={target:camera.target.clone(),radius:camera.radius,alpha:camera.alpha,beta:camera.beta};
 const workbench=$('workbench'),prevDetail=workbench.dataset.detail;
 workbench.dataset.detail='false';
 camera.setTarget(B(-4,0,.6));camera.radius=overviewRadius();camera.alpha=1.48;camera.beta=.56;
 const rect=canvas.getBoundingClientRect();
 const top=innerWidth<=650?14:210,bottom=innerWidth<=650?rect.height-14:$('lower-interface').getBoundingClientRect().top-rect.top-28;
 if(prevDetail===undefined)delete workbench.dataset.detail;else workbench.dataset.detail=prevDetail;
 const left=innerWidth<=650?16:50,right=rect.width-left;
 const corners:number[][]=[];for(const x of [-25,17.5])for(const y of [-8.2,8.2])for(const z of [.1,2.5])corners.push([x,y,z]);
 for(let iteration=0;iteration<4;iteration++){
  camera.getViewMatrix(true);scene.updateTransformMatrix(true);
  const projected=corners.map(([x,y,z])=>Vector3.Project(B(x,y,z),Matrix.Identity(),scene.getTransformMatrix(),camera.viewport.toGlobal(rect.width,rect.height)));
  const minX=Math.min(...projected.map(p=>p.x)),maxX=Math.max(...projected.map(p=>p.x)),minY=Math.min(...projected.map(p=>p.y)),maxY=Math.max(...projected.map(p=>p.y));
  const worldPerPixel=2*camera.radius*Math.tan(camera.fov/2)/rect.height;
  const dx=(minX+maxX-left-right)/2,dy=(minY+maxY-top-bottom)/2;
  const {alpha,beta,radius}=camera;
  camera.setTarget(camera.target.add(camera.getDirection(Vector3.Right()).scale(dx*worldPerPixel)).add(camera.getDirection(Vector3.Up()).scale(-dy*worldPerPixel)));
  camera.alpha=alpha;camera.beta=beta;camera.radius=radius*Math.max((maxX-minX)/(right-left),(maxY-minY)/(bottom-top));
 }
 const fitted={target:camera.target.clone(),radius:camera.radius};
 camera.setTarget(previous.target);camera.radius=previous.radius;camera.alpha=previous.alpha;camera.beta=previous.beta;return fitted;
}
function updatePlayback(){
 $('play').setAttribute('aria-label',clock.playing?'Pause':clock.progress===1?'Replay':'Play');
 $('play-icon').innerHTML=clock.playing?'<path d="M8 5v14M16 5v14"/>':'<path d="m8 4 11 8-11 8V4Z"/>';
 $('status').textContent=clock.progress===1?'Complete':clock.playing?'Running':'Paused';
}
function updateInspector(){
 $('workbench').dataset.selection=selection;
 const texts={
  assembly:[clock.stage.name,storyCopy[clock.stage.id]],
  supply:['The model’s weight store.','The reservoir stands for the loaded model. Each layer has its own numbered drawer of sample weights. Read copies travel to its devices; the stored balls stay unchanged.'],
  layer:[`Inside layer ${String(selectedLayer+1).padStart(2,'0')}.`,'Sizing jaws adjust scale. Three heads use stored context in parallel. A feed-forward press reshapes their result; bypasses preserve the input.'],
  input:['From words to numbers.','The cutter makes illustrative tokens. The drawer gives each a numerical carrier; pin heights represent example vector values.'],
  output:['Choose, print, repeat.','The rising bars represent authored candidate scores. The highest one is selected, printed, and fed back. The end signal closes the return gate.'],
 };
 $('part-title').textContent=texts[selection][0];$('part-description').textContent=texts[selection][1];
 $('case-toggle').hidden=selection!=='layer';$('concurrency').hidden=selection!=='layer';
 $('weight-details').hidden=selection!=='layer';
 $('weight-title').textContent=`Layer ${String(selectedLayer+1).padStart(2,'0')} weight drawer`;
 $('weight-values').replaceChildren(...weightBank(selectedLayer).map(w=>{const span=document.createElement('span');span.textContent=`${w.id.split('.')[1]}  ${w.value>=0?'+':''}${w.value.toFixed(2)}`;return span;}));
 $('layer-select').setAttribute('aria-label','Inspect a layer');
 document.querySelectorAll<HTMLElement>('[data-part]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.part===selection)));
}
function selectPart(id:PartId,index=clock.layerIndex,seek=false){
 selection=id;selectedLayer=index;following=false;userMovedCamera=false;$('follow-toggle').setAttribute('aria-pressed','false');
 if(id==='layer'&&seek){clock.seekLayer(index);updatePlayback();}
 $<HTMLSelectElement>('layer-select').value=String(index);
 updateInspector();
 desiredAlpha=1.48;desiredBeta=.56;
 if(id==='assembly'){const fit=fitOverview();desiredTarget=fit.target;desiredRadius=fit.radius;}
 else if(id==='supply'){desiredTarget=supplyPoint;desiredRadius=innerWidth<=650?6:5;desiredBeta=.7;}
 else if(id==='input'){desiredTarget=inputPoint;desiredRadius=innerWidth<=650?10:13;desiredBeta=.65;}
 else if(id==='output'){desiredTarget=outputPoint.add(B(-.8,0,0));desiredRadius=innerWidth<=650?11:14;desiredBeta=.7;}
 else {desiredTarget=point(index,.3,-.2,.8);desiredRadius=innerWidth<=650?9:12.5;desiredAlpha=centers[index].reverse?1.35+Math.PI:1.35;desiredBeta=.62;}
 transition=true;
}
function toggleCase(){casingTarget=casingTarget?0:1;$('case-toggle').setAttribute('aria-pressed',String(!!casingTarget));$('case-toggle').querySelector('span')!.textContent=casingTarget?'Lower layer cover':'Lift layer cover';}
function arrangeOverviewLabels(elements:HTMLElement[]){
 const canvasRect=canvas.getBoundingClientRect();
 const obstacles=[...document.querySelectorAll<HTMLElement>('.scene-caption,#inspector,.masthead,.transport,.layer-label:not([hidden])')].filter(el=>el.getClientRects().length).map(el=>el.getBoundingClientRect());
 const intersects=(a:{left:number;right:number;top:number;bottom:number},b:{left:number;right:number;top:number;bottom:number})=>a.left<b.right+5&&a.right>b.left-5&&a.top<b.bottom+5&&a.bottom>b.top-5;
 for(const el of elements){
  if(el.hidden)continue;
  const original=el.getBoundingClientRect();let placed=false;
  for(const [dx,dy]of [[0,0],[0,24],[0,-24],[0,48],[0,-48],[48,0],[-48,0],[0,72]]){
   const candidate={left:original.left+dx,right:original.right+dx,top:original.top+dy,bottom:original.bottom+dy};
   if(candidate.left<canvasRect.left+8||candidate.right>canvasRect.right-8||candidate.top<canvasRect.top+8||candidate.bottom>canvasRect.bottom-8||obstacles.some(b=>intersects(candidate,b)))continue;
   el.style.transform+=` translate(${dx}px,${dy}px)`;obstacles.push(candidate as DOMRect);placed=true;break;
  }
  if(!placed)el.hidden=true;
 }
}

async function start(){
 engine=new Engine(canvas,true,{preserveDrawingBuffer:true,stencil:true},false);
 scene=new Scene(engine);scene.useRightHandedSystem=true;scene.clearColor=Color4.FromHexString('#e8e8e1ff');
 camera=new ArcRotateCamera('Observer',1.35,.56,overviewRadius(),B(-1.3,0,.6),scene);
 camera.fov=.68;camera.minZ=.05;camera.maxZ=250;camera.lowerRadiusLimit=2.6;camera.upperRadiusLimit=120;camera.lowerBetaLimit=.15;camera.upperBetaLimit=1.4;
 camera.wheelDeltaPercentage=.014;camera.pinchDeltaPercentage=.014;camera.panningSensibility=100;camera.inertia=.7;camera.attachControl(canvas,true);
 const materials:Record<string,ShaderMaterial|StandardMaterial>={};
 for(const [name,color]of Object.entries({steel:'#91a4a7',brass:'#c59854',ink:'#344b50',ivory:'#e4e3d5',base:'#c6ceca',amber:'#dc9c4a'}))materials[name]=cel(name,color);
 const glass=new StandardMaterial('clear hopper',scene);glass.diffuseColor=Color3.FromHexString('#bdd3d0');glass.emissiveColor=Color3.FromHexString('#bdd3d0');glass.alpha=.09;glass.disableLighting=true;glass.backFaceCulling=false;materials.glass=glass;
 const active=cel('Active layer','#94b6ae'),complete=cel('Complete layer','#acbcb5');
 const pipe=cel('Carrier rails','#8da29d'),weightPipe=cel('Weight feed','#bca779'),teal=cel('Token information','#378b82'),preserved=cel('Preserved information','#6c9e95');
 async function load(group:string){
  const result=await ImportMeshAsync(`/assets/${group}.glb`,scene);
  for(const m of result.meshes)if(m instanceof Mesh&&m.getTotalVertices()){
   const old=m.material?.name??'steel';m.material=materials[old]??materials.steel;
   if(old!=='glass')outline(m);
  }
  return result.meshes[0] as Mesh;
 }
 const template=await load('layer');
 for(let i=0;i<LAYER_COUNT;i++){
  const root=template.clone(`layer_${i}`,null,false)!;
  root.position=B(centers[i].x,centers[i].y,0);root.rotationQuaternion=Quaternion.RotationAxis(Vector3.Up(),centers[i].reverse?Math.PI:0);
  const meshes=root.getChildMeshes();for(const m of meshes)m.metadata={part:'layer',layer:i};
  const nodes=root.getChildTransformNodes(false);
  const rotors=[0,1,2].map(h=>nodes.find(n=>n.name.endsWith(`roller_${h}`))!);
  const cover=nodes.find(n=>n.name.endsWith('layer_cover'))!;
  if(rotors.some(n=>!n)||!cover)throw Error('The layer asset is missing a moving pivot.');
  const foot=meshes.find(m=>m.name.endsWith('layer_foot')) as Mesh;
  const label=document.createElement('button');label.className='layer-label';label.textContent=String(i+1).padStart(2,'0');label.setAttribute('aria-label',`Inspect layer ${i+1}`);label.addEventListener('click',()=>selectPart('layer',i,true));$('part-labels').append(label);
  const jaw=nodes.find(n=>n.name.endsWith('normalizer_jaw'))!,press=nodes.find(n=>n.name.endsWith('ffn_press'))!;
  layers.push({root,rotors,cover,coverBase:cover.position.clone(),foot,label,jaw,press,jawBase:jaw.position.y,pressBase:press.position.y});
  for(let h=0;h<3;h++)routes.push({mesh:polyline(`head_rail_${i}_${h}`,[...headPath(i,h),point(i,.92,0)],pipe,.04),layer:i});
  routes.push({mesh:polyline(`residual_${i}`,residualPath(i),preserved,.045),layer:i});
  routes.push({mesh:polyline(`ffn_route_${i}`,[point(i,.92,0),point(i,2.22,0)],pipe,.045),layer:i});
  routes.push({mesh:polyline(`ffn_residual_${i}`,[point(i,.92,0),point(i,1.05,-1.3),point(i,2.22,-1.3),point(i,2.22,0)],preserved,.035),layer:i});
  const rescale=MeshBuilder.CreateTorus(`Second sizing ring ${i}`,{diameter:.3,thickness:.06,tessellation:16},scene);rescale.position=point(i,1.08,0);rescale.material=materials.steel;routes.push({mesh:rescale,layer:i});
  for(const [name,x]of [['Split',-.92],['Join',.92]] as [string,number][]){const hub=MeshBuilder.CreateSphere(`${name}_${i}`,{diameter:.15,segments:12},scene);hub.position=point(i,x,0);hub.material=teal;hub.isPickable=false;routes.push({mesh:hub,layer:i});}
  const previous=i?point(i-1,2.22,0):inputPoint.add(B(1.7,0,0));
  const entry=point(i,-1.65,0);
  const connection=[previous,entry];
  // The row turn is an actual pipe; stage order does not change when the screen layout folds.
  if(i&&i%8===0){const outer=centers[i].reverse?17.5:-17.5;connection.splice(1,0,new Vector3(outer,1.03,previous.z),new Vector3(outer,1.03,entry.z));}
  incoming.push([...connection,point(i,-.92,0)]);contextMeshes.push(polyline(`between_layers_${i}`,connection,pipe,.055));
 }
 template.setEnabled(false);
 const hopper=await load('supply');hopper.position=B(-19,0,0);for(const m of hopper.getChildMeshes())m.metadata={part:'supply'};contextMeshes.push(hopper);
 const story=await createStory(scene,clock,point,{...materials,token:teal,preserved},load,incoming);
 const weights=createWeights(scene,clock,layers,point,materials.amber,weightPipe);
 const drawerLabel=document.createElement('div');drawerLabel.className='head-label';$('part-labels').append(drawerLabel);
 for(const text of ['Split','Join','Residual','Normalize','Feed-forward']){const label=document.createElement('div');label.className='head-label route-label';label.textContent=text;$('part-labels').append(label);routeLabels.push(label);}
 for(const [text,point]of [['Weights',B(-19,0,2)]] as [string,Vector3][]){
  const element=document.createElement('div');element.className='head-label';element.textContent=text;$('part-labels').append(element);landmarks.push({element,point});
 }
 function carrier(name:string,material:ShaderMaterial,size=.19){const m=MeshBuilder.CreateBox(name,{size},scene);m.material=material;outline(m,.009);m.isPickable=false;return m;}
 residual=carrier('Residual carrier',preserved,.16);
 for(let h=0;h<3;h++){
  marks.push(carrier(`Head ${h+1} carrier`,teal));
  const label=document.createElement('div');label.className='head-label';label.textContent=`Head ${h+1}`;$('part-labels').append(label);headLabels.push(label);
 }
 for(let i=0;i<LAYER_COUNT;i++){const option=document.createElement('option');option.value=String(i);option.textContent=`Layer ${String(i+1).padStart(2,'0')}`;$('layer-select').append(option);}
 scene.onPointerObservable.add(info=>{
  if(info.type===PointerEventTypes.POINTERDOWN){transition=false;following=false;userMovedCamera=true;$('follow-toggle').setAttribute('aria-pressed','false');}
  if(info.type===PointerEventTypes.POINTERPICK&&info.pickInfo?.hit){const meta=info.pickInfo.pickedMesh?.metadata;if(meta?.carrier!==undefined){inspectCarrier(meta.carrier);clock.playing=false;updatePlayback();}else if(meta?.part){selectPart(meta.part,meta.layer??clock.layerIndex,meta.part==='layer');if(meta.weight)$<HTMLDetailsElement>('weight-details').open=true;}}
 });
 canvas.addEventListener('wheel',()=>{transition=false;following=false;userMovedCamera=true;$('follow-toggle').setAttribute('aria-pressed','false');},{passive:true});
 engine.resize();selectPart('assembly');if(reduced){camera.setTarget(desiredTarget);camera.radius=desiredRadius;camera.alpha=desiredAlpha;camera.beta=desiredBeta;transition=false;}
 $('loading').hidden=true;for(const id of ['play','next','scrubber'])($(id) as HTMLButtonElement).disabled=false;updatePlayback();
 let previous=performance.now(),uiTime=0,lastLayer='';
 engine.runRenderLoop(()=>{
  const now=performance.now(),dt=Math.max(0,(now-previous)/1000);previous=now;if(document.hidden)return;
  const wasPlaying=clock.playing;clock.step(dt);if(wasPlaying!==clock.playing)updatePlayback();const i=clock.layerIndex,p=clock.layerProgress,running=clock.layerActive;
  const layerKey=`${clock.stage.id}:${i}`;
  if(lastLayer!==layerKey){
   for(let n=0;n<layers.length;n++)layers[n].foot.material=running&&n===i?active:(clock.progress>=.11&&(n<i||!running))?complete:materials.base;
   lastLayer=layerKey;
  }
  for(let n=0;n<layers.length;n++){
   layers[n].root.setEnabled(selection==='assembly'||(selection==='layer'&&n===selectedLayer));
   const phase=n<i?1:n===i?operationProgress('head-a',p):0;
   layers[n].rotors.forEach((node,h)=>node.rotationQuaternion=Quaternion.RotationAxis(new Vector3(0,0,1),phase*Math.PI*6*(h%2?-1:1)));
   const lift=n===selectedLayer&&selection==='layer'?casing:0;layers[n].cover.position.copyFrom(layers[n].coverBase.add(new Vector3(0,lift*.8,0)));
   layers[n].jaw.position.y=layers[n].jawBase-(running&&n===i&&p<.18?Math.sin(p/.18*Math.PI)*.35:0);
   layers[n].press.position.y=layers[n].pressBase-(running&&n===i?Math.sin(operationProgress('ffn',p)*Math.PI)*.48:0);
  }
  casing=reduced?casingTarget:casing+(casingTarget-casing)*Math.min(1,dt*8);
  const drawerTarget=$<HTMLDetailsElement>('weight-details').open?1:0;
  drawerOpen=reduced?drawerTarget:drawerOpen+(drawerTarget-drawerOpen)*Math.min(1,dt*8);
  for(let h=0;h<3;h++){
   marks[h].setEnabled(running&&p>=.18&&p<.62);
   marks[h].position.copyFrom(p<.52?along(headPath(i,h),operationProgress('head-a',p)):along(joinPath(i,h),operationProgress('join',p)));
   marks[h].scaling.set(1,.65+operationProgress('head-a',p)*.55,1);
  }
  residual.setEnabled(running&&p>=.18&&p<.98);residual.position.copyFrom(p<.62?along(residualPath(i),Math.min(1,Math.max(0,(p-.18)/.24))):along([point(i,.92,0),point(i,1.05,-1.3),point(i,2.22,-1.3),point(i,2.22,0)],Math.min(1,(p-.62)/.34)));
  if(following){
   if(selection!=='assembly'){selection='assembly';updateInspector();}
   if(selectedLayer!==i){selectedLayer=i;$<HTMLSelectElement>('layer-select').value=String(i);}
   desiredTarget=running?(p<.18?along(incoming[i],p/.18):p<.52?along([point(i,-.92,0),point(i,.55,0)],(p-.18)/.34):p<.62?along([point(i,.55,0),point(i,.92,0)],(p-.52)/.1):p<.9?along([point(i,.92,0),point(i,2.05,0)],(p-.62)/.28):along([point(i,2.05,0),i===31?outputPoint:point(i,2.22,0)],(p-.9)/.1)):clock.progress<.11?inputPoint:clock.stage.id.startsWith('return-')?along(returnPath,clock.stageProgress):outputPoint.add(B(-.8,0,0));
   desiredRadius=innerWidth<=650?11:15;desiredAlpha=1.48;desiredBeta=.62;transition=true;
  }
  routes.forEach(({mesh,layer})=>mesh.setEnabled(selection==='assembly'||(selection==='layer'&&layer===selectedLayer)));
  contextMeshes.forEach(mesh=>mesh.setEnabled(selection==='assembly'||(selection==='supply'&&mesh===hopper)));
  if(selection!=='assembly'&&(selection!=='layer'||selectedLayer!==i)){residual.setEnabled(false);marks.forEach(m=>m.setEnabled(false));}
  weights.update(selection,selectedLayer,drawerOpen);
  story.update(selection,selectedLayer,selectedCarrier,$<HTMLDetailsElement>('carrier-details').open);
  if(transition){const f=reduced?1:1-Math.exp(-Math.min(dt,.1)*5);const prevAlpha=camera.alpha,prevBeta=camera.beta,prevRadius=camera.radius;const da=Math.atan2(Math.sin(desiredAlpha-prevAlpha),Math.cos(desiredAlpha-prevAlpha));camera.setTarget(Vector3.Lerp(camera.target,desiredTarget,f));camera.radius=prevRadius+(desiredRadius-prevRadius)*f;camera.alpha=prevAlpha+da*f;camera.beta=prevBeta+(desiredBeta-prevBeta)*f;camera.inertialAlphaOffset=0;camera.inertialBetaOffset=0;camera.inertialRadiusOffset=0;if(!following&&Math.abs(camera.radius-desiredRadius)<.005&&Vector3.Distance(camera.target,desiredTarget)<.005)transition=false;}
  const detail=following||selection!=='assembly'||camera.radius<overviewRadius()*.65;
  $('workbench').dataset.detail=String(detail);scene.render();
  const rect=canvas.getBoundingClientRect(),inspector=$('inspector').getBoundingClientRect();
  function project(v:Vector3){const p=Vector3.Project(v,Matrix.Identity(),scene.getTransformMatrix(),camera.viewport.toGlobal(engine.getRenderWidth(),engine.getRenderHeight()));return {x:p.x/engine.getRenderWidth()*rect.width,y:p.y/engine.getRenderHeight()*rect.height,z:p.z};}
  for(let n=0;n<layers.length;n++){
   const l=layers[n].label,q=project(point(n,0,-1.28,.3));l.style.transform=`translate(${q.x}px,${q.y+canvas.offsetTop}px) translate(-50%,0)`;l.textContent=innerWidth<=650?`Layer ${n+1}`:String(n+1).padStart(2,'0');
   const behind=q.x+18>inspector.left&&q.y+rect.top>inspector.top&&q.y+rect.top<inspector.bottom;
   l.hidden=!layers[n].root.isEnabled()||q.z<0||q.z>1||q.x<15||q.x>rect.width-15||q.y<(innerWidth<=650?8:60)||q.y>rect.height-(innerWidth<=650?20:160)||behind||(detail&&n!==selectedLayer)||(innerWidth<=650&&n!==i);
   l.classList.toggle('active',running&&n===i);l.classList.toggle('finished',clock.progress>=.11&&(n<i||!running));
  }
  for(let h=0;h<3;h++){const q=project(point(selectedLayer,0,heads[h],1.35));const l=headLabels[h];l.style.transform=`translate(${q.x}px,${q.y}px) translate(-50%,-100%)`;l.hidden=selection!=='layer'||q.z<0||q.z>1||q.x<0||q.x>rect.width||innerWidth<=650;}
  for(const {element,point}of landmarks){const q=project(point);element.style.transform=`translate(${q.x}px,${q.y+canvas.offsetTop}px) translate(-50%,-140%)`;element.hidden=detail||q.z<0||q.z>1;}
  for(const [n,label]of routeLabels.entries()){const xy=[[-.92,-.35],[.92,-.3],[0,-1.48],[-1.35,.6],[1.62,.9]][n];const q=project(point(selectedLayer,xy[0],xy[1],1.1));label.style.transform=`translate(${q.x}px,${q.y+canvas.offsetTop}px) translate(-50%,-140%)`;label.hidden=selection!=='layer'||q.z<0||q.z>1||innerWidth<=650;}
  for(const a of story.annotations){const q=project(a.position());a.element.style.transform=`translate(${q.x}px,${q.y+canvas.offsetTop}px) translate(-50%,-100%)`;a.element.hidden=!a.views.includes(selection)||a.visible?.()===false||q.z<0||q.z>1||q.x<10||q.x>rect.width-10||q.y<8||q.y>rect.height-12;}
  const drawerPoint=project(point(selectedLayer,0,-2.13-drawerOpen*.5,.4));drawerLabel.style.transform=`translate(${drawerPoint.x}px,${drawerPoint.y+canvas.offsetTop}px) translate(-50%,0)`;drawerLabel.textContent=`Weights ${String(selectedLayer+1).padStart(2,'0')} · stored`;drawerLabel.hidden=selection!=='layer';
  if(selection==='assembly')arrangeOverviewLabels([...story.annotations.map(a=>a.element),...landmarks.map(a=>a.element)]);
  else if(selection==='layer')arrangeOverviewLabels([...story.annotations.map(a=>a.element).reverse(),drawerLabel,...routeLabels,...headLabels]);
  if(now-uiTime>80){
   $('phase-name').textContent=running?`Layer ${String(i+1).padStart(2,'0')} · ${clock.phase.name}`:clock.stage.name;
   $('phase-brief').textContent=running?clock.stage.name:'Question → Paris → . → End';
   if($('answer-text').textContent!==(clock.answer||'…'))$('answer-text').textContent=clock.answer||'…';$('answer-text').classList.toggle('complete',clock.stage.id==='done');
   $('cache-status').textContent=`${clock.cacheCount(selectedLayer)} positions stored in this layer’s K/V rack`;
   for(const [j,chip]of [...document.querySelectorAll<HTMLElement>('.prompt-token')].entries()){chip.classList.toggle('ready',clock.progress>=.065*(j+1)/7);}
   for(const button of document.querySelectorAll<HTMLElement>('[data-stage]'))button.setAttribute('aria-current',String(clock.progress>=Number(button.dataset.stage)&&clock.progress<Number(button.dataset.end)));
   if(selection==='assembly')updateInspector();
   const state=carrierState(clock,selectedCarrier);
   $('carrier-state').textContent=!state.assembled?'Waiting for pin assembly':`${state.finished?'Retained after':'Layer'} ${String(state.layer+1).padStart(2,'0')} · ${state.operation}`;
   $('carrier-identity').textContent=`“${state.label}” stays token ${selectedCarrier+1}`;
   $('carrier-values').hidden=!state.assembled;
   for(const [k,row]of [...document.querySelectorAll<HTMLElement>('.pin-reading')].entries()){
    const input=state.input[k],value=state.values[k];row.querySelector('.pin-before')!.textContent=`${input>=0?'+':''}${input.toFixed(2)}`;row.querySelector('.pin-now')!.textContent=`${value>=0?'+':''}${value.toFixed(2)}`;
    row.style.setProperty('--pin-before',`${(input+1)*50}%`);row.style.setProperty('--pin-now',`${(value+1)*50}%`);
   }
   for(const [j,chip]of [...document.querySelectorAll<HTMLElement>('.prompt-token')].entries())chip.setAttribute('aria-pressed',String(j===selectedCarrier&&$<HTMLDetailsElement>('carrier-details').open));
   const scrub=$<HTMLInputElement>('scrubber');scrub.value=String(clock.progress);scrub.style.setProperty('--progress',`${clock.progress*100}%`);scrub.setAttribute('aria-valuetext',`Layer ${i+1} of ${LAYER_COUNT}, ${clock.elapsedSeconds.toFixed(1)} of ${clock.durationSeconds} seconds`);
   $('clock').textContent=`${clock.elapsedSeconds.toFixed(1)} / ${clock.durationSeconds} s`;
   $('active-layer').textContent=running?`${String(i+1).padStart(2,'0')} / ${LAYER_COUNT}`:clock.stage.id==='done'?'Finished':'Between passes';
   const lp=selectedLayer===i?p:selectedLayer<i?1:0;
   const track=document.querySelector<HTMLElement>('.lane-row>div')!;
   $('lane-cursor').style.left=`${track.offsetLeft+lp*track.offsetWidth}px`;
   for(const bar of document.querySelectorAll<HTMLElement>('.lane-bar'))bar.classList.toggle('running',lp>=Number(bar.dataset.start)&&lp<Number(bar.dataset.end));
   uiTime=now;
  }
 });
 document.addEventListener('visibilitychange',()=>{previous=performance.now();});
 let fittedW=0,fittedH=0;
 new ResizeObserver(()=>{const {width,height}=canvas.getBoundingClientRect();if(Math.abs(width-fittedW)<1&&Math.abs(height-fittedH)<1)return;fittedW=width;fittedH=height;engine.resize();if(selection==='assembly'&&!following&&!userMovedCamera)selectPart('assembly');}).observe(canvas);
 if(import.meta.env.DEV)Object.assign(window,{decodeMachine:{clock,scene,camera,layers,weights,weightBalls:weights.copies,marks,story,selectPart,get selection(){return selection;},get selectedCarrier(){return selectedCarrier;},get selectedLayer(){return selectedLayer;},get following(){return following;},get casing(){return casing;}}});
}
function togglePlay(){if(clock.progress===1)clock.restart();else clock.playing=!clock.playing;updatePlayback();}
$('play').addEventListener('click',togglePlay);
$('replay').addEventListener('click',()=>{clock.restart();selectPart('assembly');updatePlayback();});
document.querySelectorAll<HTMLElement>('[data-stage]').forEach(el=>el.addEventListener('click',()=>{clock.playing=false;clock.seek(Number(el.dataset.stage));updatePlayback();}));
$('next').addEventListener('click',()=>{clock.next();updatePlayback();});
$('scrubber').addEventListener('input',e=>{clock.playing=false;clock.seek(Number((e.target as HTMLInputElement).value));updatePlayback();});
$('duration').addEventListener('change',e=>{const input=e.target as HTMLInputElement;clock.setDuration(input.valueAsNumber);input.value=String(clock.durationSeconds);});
$('layer-select').addEventListener('change',e=>selectPart('layer',Number((e.target as HTMLSelectElement).value),true));
$('case-toggle').addEventListener('click',toggleCase);
$('carrier-select').addEventListener('change',e=>inspectCarrier(Number((e.target as HTMLSelectElement).value)));
$('carrier-details').addEventListener('toggle',()=>{if($<HTMLDetailsElement>('carrier-details').open)requestAnimationFrame(revealCarrierReadout);});
$('find-carrier').addEventListener('click',()=>{const state=carrierState(clock,selectedCarrier);if(state.waiting||state.finished){clock.seek(carrierPassPosition(selectedCarrier,selection==='layer'?selectedLayer:0));}clock.playing=false;selectPart('layer',clock.layerIndex);updatePlayback();});
$('follow-toggle').addEventListener('click',()=>{following=!following;userMovedCamera=false;$('follow-toggle').setAttribute('aria-pressed',String(following));if(following){selection='assembly';updateInspector();transition=true;}else transition=false;});
$('reset-view').addEventListener('click',()=>selectPart('assembly'));
document.querySelectorAll<HTMLElement>('[data-part]').forEach(el=>el.addEventListener('click',()=>selectPart(el.dataset.part as PartId,clock.layerIndex,el.dataset.part==='layer')));
function help(show:boolean){$('help').hidden=!show;$('help-toggle').setAttribute('aria-expanded',String(show));if(show)$('help-close').focus();else $('help-toggle').focus();}
$('help-toggle').addEventListener('click',()=>help($('help').hidden===true));$('help-close').addEventListener('click',()=>help(false));
window.addEventListener('keydown',e=>{
 if(e.key==='Escape'&&!$('help').hidden){help(false);return;}
 if(e.target instanceof HTMLInputElement||e.target instanceof HTMLSelectElement||e.target instanceof HTMLButtonElement||e.target instanceof HTMLAnchorElement)return;
 if(e.code==='Space'){e.preventDefault();togglePlay();}
 else if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();clock.playing=false;clock.seek(clock.progress+(e.key==='ArrowRight'?.5:-.5)/clock.durationSeconds);updatePlayback();}
 else if(e.key==='Home'){e.preventDefault();selectPart('assembly');}
 else if(e.key.toLowerCase()==='c'&&selection==='layer')toggleCase();
});
start().catch(error=>{$('loading').hidden=true;$('error').hidden=false;$('error-message').textContent=error instanceof Error?error.message:String(error);console.error(error);});
