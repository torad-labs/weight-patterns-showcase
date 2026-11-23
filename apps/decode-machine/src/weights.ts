import {MeshBuilder} from '@babylonjs/core/Meshes/meshBuilder';
import {Mesh} from '@babylonjs/core/Meshes/mesh';
import type {TransformNode} from '@babylonjs/core/Meshes/transformNode';
import type {Scene} from '@babylonjs/core/scene';
import type {Material} from '@babylonjs/core/Materials/material';
import {Vector3} from '@babylonjs/core/Maths/math.vector';
import {Timeline,samplePath,type PartId} from './timeline';
import {weightBank} from './representation';

export function createWeights(scene:Scene,clock:Timeline,layers:{root:TransformNode}[],point:(i:number,x:number,y:number,z?:number)=>Vector3,amber:Material,pipe:Material){
 const banks=layers.map(({root},i)=>{
  const drawer=root.getChildTransformNodes().find(n=>n.name.endsWith('weight_drawer'))!;
  const weights=weightBank(i),balls=weights.map((weight,k)=>{
   const ball=drawer.getChildMeshes().find(n=>n.name.endsWith(`stored_weight_${k}`))!;
   ball.scaling.scaleInPlace(.72+Math.abs(weight.value)*.55);
   ball.metadata={part:'layer',layer:i,weight:weight.id};return ball;
  });return {drawer,base:drawer.position.clone(),weights,balls};
 });
 const copies=[0,1,2,3].map(h=>{const m=MeshBuilder.CreateSphere(`Weight read copy ${h}`,{diameter:.17,segments:12},scene);m.material=amber;m.isPickable=false;return m;});
 let feeds:Mesh[]=[],lastLayer=-1;
 const destinations=[[-.25,-.82],[-.25,0],[-.25,.82],[1.62,0]];
 const path=(i:number,h:number)=>[point(i,[-.63,-.21,.21,.63][h],-1.35,.52),point(i,-.75,-1.1,1.55),point(i,destinations[h][0],destinations[h][1],1.55)];
 return {banks,copies,
  update(selection:PartId,selectedLayer:number,open:number){
   const i=clock.layerIndex,p=clock.layerProgress,visible=clock.layerActive&&(selection==='assembly'||(selection==='layer'&&selectedLayer===i));
   banks.forEach((bank,n)=>{bank.drawer.position.copyFrom(bank.base);if(selection==='layer'&&selectedLayer===n)bank.drawer.position.z+=open*.5;});
   if(lastLayer!==i){feeds.forEach(m=>m.dispose());feeds=copies.map((_,h)=>{const m=MeshBuilder.CreateTube(`Local weight read pipe ${i}_${h}`,{path:path(i,h),radius:.025,tessellation:8},scene);m.material=pipe;m.isPickable=false;return m;});lastLayer=i;}
   copies.forEach((mesh,h)=>{
    const start=h<3?.10:.64,ready=h<3?.18:.70,done=h<3?.52:.90,end=h<3?.62:1;
    const on=visible&&p>=start&&p<end;mesh.setEnabled(on);feeds[h].setEnabled(visible&&selection==='layer');
    const progress=p<ready?(p-start)/(ready-start):p<done?1:1-(p-done)/(end-done);
    mesh.position.set(...samplePath(path(i,h).map(p=>p.asArray()),progress) as [number,number,number]);
    mesh.metadata={weight:banks[i].weights[[0,4,8,11][h]].id,layer:i,readCopy:true};
   });
  },
 };
}
