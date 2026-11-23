import {LAYER_COUNT,PROMPT_TOKENS,Timeline,phases,storyStages} from './timeline';

// Four-component teaching model, NOT an extracted embedding or model trace.
// Every sample is reconstructed from position in the shared story clock.
export const CARRIER_LABELS=[...PROMPT_TOKENS,'Paris','.'];
export const weightBank=(layer:number)=>Array.from({length:12},(_,slot)=>({
 id:`L${String(layer+1).padStart(2,'0')}.W${String(slot+1).padStart(2,'0')}`,
 value:Math.round(Math.sin((layer+1)*2.17+(slot+1)*1.31)*90)/100,
}));
const normalize=(v:number[])=>{const scale=Math.sqrt(v.reduce((sum,x)=>sum+x*x,0)/v.length+.00001);return v.map(x=>x/scale*.5);};
const samples=CARRIER_LABELS.map((_,token)=>{
 let value=Array.from({length:4},(_,k)=>Math.sin((token+1)*.91+(k+1)*1.7)*.85);
 return Array.from({length:LAYER_COUNT},(_,layer)=>{
  const input=value,w=weightBank(layer).map(w=>w.value),norm=normalize(input);
  const mixed=norm.map((v,k)=>Math.tanh(v*w[k]+norm[(k+1)%4]*w[k+4]+norm[(k+3)%4]*w[k+8]));
  const joined=input.map((v,k)=>(v+mixed[k])*.5),rescaled=normalize(joined);
  const pressed=rescaled.map((v,k)=>Math.tanh(v*(1+w[k+4])+rescaled[(k+2)%4]*w[k+8]));
  const output=joined.map((v,k)=>(v+pressed[k])*.5);
  value=output;
  return [input,norm,mixed,joined,rescaled,pressed,output];
 });
});
export function carrierState(clock:Timeline,token:number){
 const pass=token<7?0:token-6,stage=storyStages.find(s=>s.pass===pass)!;
 const waiting=clock.progress<stage.start,finished=clock.progress>=stage.end;
 const layer=waiting?0:finished?31:clock.layerIndex;
 const progress=waiting?0:finished?1:clock.layerProgress;
 const phaseIndex=phases.findIndex(p=>progress<p.end),phase=phaseIndex<0?5:phaseIndex;
 const frames=samples[token][layer],a=frames[phase],b=frames[phase+1];
 const t=Math.max(0,Math.min(1,(progress-phases[phase].start)/(phases[phase].end-phases[phase].start)));
 return {token,label:CARRIER_LABELS[token].trim(),layer,pass,waiting,finished,
  assembled:clock.progress>=(token<7?.065:token===7?.49:.76),
  input:frames[0],values:a.map((v,k)=>v+(b[k]-v)*t),
  operation:waiting?'Awaiting stack':finished?'Stack complete':phases[phase].name};
}
export function carrierPassPosition(token:number,layer:number,phase=.34){
 const stage=storyStages.find(s=>s.pass===(token<7?0:token-6))!;
 return stage.start+(stage.end-stage.start)*(Math.min(31,Math.max(0,layer))+phase)/LAYER_COUNT;
}
