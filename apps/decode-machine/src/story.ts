import {Scene} from '@babylonjs/core/scene';
import {Mesh} from '@babylonjs/core/Meshes/mesh';
import {MeshBuilder} from '@babylonjs/core/Meshes/meshBuilder';
import {TransformNode} from '@babylonjs/core/Meshes/transformNode';
import {Vector3} from '@babylonjs/core/Maths/math.vector';
import type {Material} from '@babylonjs/core/Materials/material';
import {Timeline,PROMPT_TOKENS,LAYER_COUNT,samplePath,type PartId} from './timeline';
import {CARRIER_LABELS,carrierState} from './representation';

export const B=(x:number,y:number,z:number)=>new Vector3(x,z,-y);
export const inputPoint=B(-20,-5.6,1.03),outputPoint=B(-20,5.6,1.03),supplyPoint=B(-19,0,1.45);
export const returnPath=[outputPoint,B(-24,5.6,1.03),B(-24,-5.6,1.03),B(-18.3,-5.6,1.03)];
export const storyCopy={
 intake:'The cutter separates the question into illustrative tokens. The words are still labels here.',
 embed:'A drawer assembles a numerical carrier for each token. Pin heights stand for example vector values.',
 prefill:'The prompt positions move together through each layer. K/V racks retain information for later attention.',
 'choose-paris':'The final prompt position produces candidate scores. The scripted selector chooses Paris.',
 'return-paris':'Paris is printed and fed back as the next input. The question remains in every layer’s K/V rack.',
 'decode-paris':'One new position crosses the stack, using the stored question. This pass predicts the next token.',
 'choose-period':'The selector chooses a period. The visible answer is now Paris.',
 'return-period':'The period returns to the input. Its pass will decide whether the answer should continue.',
 'decode-period':'Each layer reads the retained context and adds this new position to its cache.',
 'choose-end':'The end signal wins. It closes the return gate instead of printing another word.',
 done:'Paris. The end signal has stopped the machine. Replay, or scrub backward to inspect any step.',
};
export async function createStory(scene:Scene,clock:Timeline,point:(i:number,x:number,y:number,z?:number)=>Vector3,materials:Record<string,Material>,load:(name:string)=>Promise<Mesh>,incoming:Vector3[][]){
 const annotations:{element:HTMLDivElement;position:()=>Vector3;views:PartId[];visible?:()=>boolean}[]=[];
 const label=(text:string,position:()=>Vector3,views:PartId[],visible?:()=>boolean)=>{const element=document.createElement('div');element.className='head-label story-label';element.textContent=text;document.getElementById('part-labels')!.append(element);annotations.push({element,position,views,visible});return element;};
 const box=(name:string,at:Vector3,size:[number,number,number],material:Material,parent?:TransformNode)=>{const mesh=MeshBuilder.CreateBox(name,{width:size[0],height:size[2],depth:size[1]},scene);mesh.position=at;mesh.material=material;mesh.isPickable=false;if(parent)mesh.parent=parent;return mesh;};
 const tube=(name:string,path:Vector3[],material:Material,radius=.045)=>{const mesh=MeshBuilder.CreateTube(name,{path,radius,tessellation:8,cap:Mesh.CAP_ALL},scene);mesh.material=material;mesh.isPickable=false;return mesh;};
 const along=(path:Vector3[],t:number)=>new Vector3(...samplePath(path.map(p=>p.asArray()),t) as [number,number,number]);
 const intake=await load('intake');intake.position=B(-20,-5.6,0);intake.getChildMeshes().forEach(m=>m.metadata={part:'input'});
 const selector=await load('selector');selector.position=B(-20,5.6,0);selector.getChildMeshes().forEach(m=>m.metadata={part:'output'});
 const blade=intake.getChildTransformNodes().find(n=>n.name==='cutter_blade')!;
 const arm=selector.getChildTransformNodes().find(n=>n.name==='selector_arm')!;
 const stop=selector.getChildTransformNodes().find(n=>n.name==='stop_gate')!;
 const bladeY=blade.position.y,stopY=stop.position.y;
 const belt=tube('Answer feedback pipe',returnPath,materials.token,.06);
 const link=tube('Embedding to first layer',[B(-18.3,-5.6,1.03),point(0,-1.65,0)],materials.token);
 const outlet=tube('Final layer to candidates',[point(31,2.22,0),outputPoint],materials.token);
 const cutterPaper=box('Question strip',B(-21,-5.6,.8),[1.7,2.8,.05],materials.ivory);
 const chips=PROMPT_TOKENS.map((text,j)=>{const mesh=box(`Paper token ${j}`,B(-20.2,-6.77+j*.39,.84),[.65,.3,.06],materials.ivory);label(text.trim(),()=>mesh.position.add(B(0,0,.06)),['input'],()=>mesh.isEnabled());return mesh;});
 const pins:Mesh[]=[];
 const carriers=CARRIER_LABELS.map((_,j)=>{const root=new TransformNode(`Token carrier ${j}`,scene);const base=box(`Carrier base ${j}`,Vector3.Zero(),[.38,.27,.07],materials.token,root);base.isPickable=true;base.metadata={carrier:j};for(let k=0;k<4;k++){const pin=box(`Vector pin ${j}_${k}`,B((k-1.5)*.085,0,.15),[.045,.15,.18],materials.token,root);pin.isPickable=true;pin.metadata={carrier:j};pins.push(pin);}return root;});
 const cacheRoots:TransformNode[]=[],cachePins:Mesh[][]=[];
 for(let i=0;i<LAYER_COUNT;i++){
  const root=new TransformNode(`Layer ${i+1} K/V rack`,scene);cacheRoots.push(root);const pegs:Mesh[]=[];
  for(const y of [1.55,1.8]){const rack=tube(`Cache shelf ${i}_${y}`,[point(i,-.8,y,.38),point(i,.8,y,.38)],materials.steel,.035);rack.parent=root;}
  for(let n=0;n<9;n++)for(let row=0;row<2;row++){const peg=box(`Cache ${i} position ${n} ${row?'V':'K'}`,point(i,-.72+n*.18,1.55+row*.25,.5),[.1,.16,.2],row?materials.preserved:materials.token,root);pegs.push(peg);}
  cachePins.push(pegs);
 }
 const bars=[0,1,2,3].map(j=>{const mesh=box(`Candidate score ${j}`,B(-20,4.25+j*.9,.6),[.43,.48,.1],j===0?materials.token:materials.steel);return mesh;});
 const candidateLabels=bars.map((mesh,j)=>label('',()=>B(-19.4,4.25+j*.9,.7),['output'],()=>mesh.isEnabled()));
 const cacheReads=[0,1,2].map(j=>box(`Cached context read ${j}`,Vector3.Zero(),[.12,.12,.18],materials.preserved));
 const feedback=box('Selected token on return track',returnPath[0],[.3,.25,.3],materials.token);
 label('Token cutter',()=>B(-21,-5.6,1.9),['assembly','input']);
 label('Pin assembly',()=>B(-19.2,-5.6,1.3),['input']);
 label('Candidate selector',()=>B(-20,5.6,2.4),['assembly','output']);
 label('Return track',()=>B(-24,0,1.05),['assembly']);
 const printed=label('',()=>B(-22.4,5.6,.8),['assembly','output'],()=>!!clock.answer);printed.classList.add('printed-answer');
 const cacheLabel=label('K / V · stored context',()=>point(selectedLayer,0,1.85,.7),['layer']);
 let selectedLayer=0;
 let selectedCarrier=5;
 let inspecting=false;
 const carrierLabel=label('',()=>carriers[selectedCarrier].position.add(B(0,0,.9)),['assembly','layer','input'],()=>inspecting&&carriers[selectedCarrier].isEnabled());
 return {
  annotations,carriers,pins,cachePins,cacheReads,feedback,bars,intake,selector,
  update(selection:PartId,index:number,carrier:number,inspectCarrier:boolean){
   selectedCarrier=carrier;inspecting=inspectCarrier;carrierLabel.textContent=`“${CARRIER_LABELS[carrier].trim()}” · changing pins`;
   selectedLayer=index;const id=clock.stage.id,t=clock.stageProgress,active=clock.layerActive,i=clock.layerIndex,p=clock.layerProgress;
   const global=selection==='assembly',showIntake=global||selection==='input',showSelector=global||selection==='output';intake.setEnabled(showIntake);selector.setEnabled(showSelector);belt.setEnabled(global);link.setEnabled(global);outlet.setEnabled(global);
   blade.position.y=bladeY-(id==='intake'?Math.abs(Math.sin(t*Math.PI*7))*.7:0);
   cutterPaper.setEnabled(showIntake&&clock.progress<.065);cutterPaper.scaling.x=id==='intake'?Math.max(.04,1-t):1;
   chips.forEach((mesh,j)=>mesh.setEnabled(showIntake&&clock.progress>=.065*(j+1)/7));
   for(let j=0;j<carriers.length;j++){
    const root=carriers[j];let show=false;
    if(id==='embed'&&j<7){show=t>=j/9;root.position.copyFrom(B(-19.2,-6.77+j*.39,.9));}
    else if(active&&(clock.stage.pass===0?j<7:j===6+clock.stage.pass!)){
     show=true;
     const path=p<.18?incoming[i]:p<.52?[point(i,-.92,0),point(i,.55,0)]:p<.62?[point(i,.55,0),point(i,.92,0)]:p<.9?[point(i,.92,0),point(i,2.05,0)]:[point(i,2.05,0),i===31?outputPoint:point(i,2.22,0)];
     const local=p<.18?p/.18:p<.52?(p-.18)/.34:p<.62?(p-.52)/.1:p<.9?(p-.62)/.28:(p-.9)/.1;
     root.position.copyFrom(along(path,local));root.position.z+=(j<7?j-3:0)*.16;root.position.y+=.3+(j<7?j*.035:0);
    }
    root.scaling.setAll(inspectCarrier?(j===carrier?1.3:.7):1);
    root.setEnabled(show&&(global||(id==='embed'&&selection==='input')||(selection==='layer'&&index===i)));
   }
   for(let j=0;j<carriers.length;j++){const state=carrierState(clock,j);state.values.forEach((value,k)=>{const pin=pins[j*4+k],height=.05+(value+1)*.12;pin.scaling.y=height/.18;pin.position.y=.035+height/2;});}
   for(let n=0;n<LAYER_COUNT;n++){cacheRoots[n].setEnabled(global||(selection==='layer'&&n===index));const count=clock.cacheCount(n);cachePins[n].forEach((peg,k)=>peg.setEnabled(Math.floor(k/2)<count));}
   cacheReads.forEach((mesh,h)=>{mesh.setEnabled(active&&p>=.18&&p<.52&&(global||(selection==='layer'&&index===i)));mesh.position.copyFrom(along([point(i,-.72+h*.54,1.55,.6),point(i,-.5,1.5,1.5),point(i,-.25,[-.82,0,.82][h],1.1)],(p-.18)/.34));});
   cacheLabel.textContent=`K / V · ${clock.cacheCount(index)} positions kept`;
   const choosing=id.startsWith('choose-'),returned=id.startsWith('return-');
   bars.forEach((bar,j)=>{bar.setEnabled(showSelector&&clock.candidatesReady);const height=(choosing?Math.min(1,t*2):1)*[1.5,.65,.43,.28][j];bar.scaling.y=Math.max(.01,height/.1);bar.position.y=.5+height/2;candidateLabels[j].textContent=clock.candidateLabels[j];});
   arm.position.z=-(-1.35+(choosing?(1-Math.min(1,t*2))*2.7:0));
   stop.position.y=stopY-(clock.progress>=.985?.65:0);
   feedback.setEnabled(global&&returned);if(returned)feedback.position.copyFrom(along(returnPath,t));
   printed.textContent=clock.answer;
  },
 };
}
