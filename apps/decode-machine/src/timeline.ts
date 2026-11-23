// Authored teaching schedule. These are neither measured timings nor real tokenizer results.
export const LAYER_COUNT=32;
export const PROMPT_TOKENS=['What',"’s",' the',' capital',' of',' France','?'] as const;
export const QUESTION='What’s the capital of France?';
export const phases=[
 {start:0,end:.18,name:'Normalize'}, {start:.18,end:.52,name:'Parallel'},
 {start:.52,end:.62,name:'Join'}, {start:.62,end:.70,name:'Rescale'},
 {start:.70,end:.90,name:'Feed-forward'}, {start:.90,end:1,name:'Output'},
] as const;
export const operations=[
 {id:'prepare',label:'Normalize',start:0,end:.18},
 {id:'head-a',label:'Head 1',start:.18,end:.52},{id:'head-b',label:'Head 2',start:.18,end:.52},{id:'head-c',label:'Head 3',start:.18,end:.52},
 {id:'bypass',label:'Residual',start:0,end:.62},{id:'join',label:'Join',start:.52,end:.62},
 {id:'rescale',label:'Rescale',start:.62,end:.70},{id:'ffn',label:'Feed-forward',start:.70,end:.90},
 {id:'output',label:'Output',start:.90,end:1},
] as const;
export const storyStages=[
 {id:'intake',start:0,end:.065,name:'Cut the question',pass:null},
 {id:'embed',start:.065,end:.11,name:'Assemble the carriers',pass:null},
 {id:'prefill',start:.11,end:.43,name:'Read the question',pass:0},
 {id:'choose-paris',start:.43,end:.49,name:'Choose “Paris”',pass:null},
 {id:'return-paris',start:.49,end:.54,name:'Feed “Paris” back',pass:null},
 {id:'decode-paris',start:.54,end:.70,name:'Predict what follows “Paris”',pass:1},
 {id:'choose-period',start:.70,end:.76,name:'Choose “.”',pass:null},
 {id:'return-period',start:.76,end:.81,name:'Feed “.” back',pass:null},
 {id:'decode-period',start:.81,end:.94,name:'Predict what follows “.”',pass:2},
 {id:'choose-end',start:.94,end:.985,name:'Choose the end signal',pass:null},
 {id:'done',start:.985,end:1,name:'Answer complete',pass:null},
] as const;
export type OperationId=typeof operations[number]['id'];
export type StoryId=typeof storyStages[number]['id'];
export type PartId='assembly'|'supply'|'layer'|'input'|'output';
const clamp=(n:number)=>Math.max(0,Math.min(1,n));
export function operationProgress(id:OperationId,progress:number){const op=operations.find(o=>o.id===id)!;return clamp((progress-op.start)/(op.end-op.start));}
export function operationActive(id:OperationId,progress:number){const op=operations.find(o=>o.id===id)!;return progress>=op.start&&progress<op.end;}
export class Timeline{
 progress=0;playing=true;durationSeconds=120;
 constructor(public layerCount=LAYER_COUNT){}
 setDuration(seconds:number){if(Number.isFinite(seconds)&&seconds>=1&&seconds<=3600)this.durationSeconds=seconds;}
 step(delta:number){if(this.playing&&Number.isFinite(delta)){this.progress=clamp(this.progress+Math.max(0,delta)/this.durationSeconds);if(this.progress===1)this.playing=false;}return this.progress;}
 seek(progress:number){if(Number.isFinite(progress)){this.progress=clamp(progress);if(this.progress===1)this.playing=false;}}
 get stage(){return storyStages.find(s=>this.progress<s.end)??storyStages[storyStages.length-1];}
 get stageProgress(){return clamp((this.progress-this.stage.start)/(this.stage.end-this.stage.start));}
 get layerActive(){return this.stage.pass!==null;}
 get passIndex(){return this.progress<.54?0:this.progress<.81?1:2;}
 get layerPosition(){return Math.round(this.stageProgress*this.layerCount*1e10)/1e10;}
 get layerIndex(){return this.layerActive?Math.min(this.layerCount-1,Math.floor(this.layerPosition)):this.progress<.11?0:this.layerCount-1;}
 get layerProgress(){return this.layerActive?Math.round((this.layerPosition-this.layerIndex)*1e10)/1e10:this.progress<.11?0:1;}
 get phase(){return phases.find(p=>this.layerProgress<p.end)??phases[phases.length-1];}
 get elapsedSeconds(){return this.progress*this.durationSeconds;}
 get answer(){return this.progress<.475?'':this.progress<.745?'Paris':'Paris.';}
 get currentToken(){return this.progress<.49?'question':this.progress<.76?'Paris':'.';}
 get candidateLabels(){return this.progress<.54?['Paris','Lyon','London','Rome']:this.progress<.81?['.',',',' is',' France']:['End','!',' the','?'];}
 get candidatesReady(){return this.stage.id.startsWith('choose-')||this.stage.id.startsWith('return-')||this.stage.id==='done';}
 cacheCount(layer:number){
  if(this.progress<.11)return 0;
  const pass=this.stage.pass;
  if(pass===null)return this.progress<.54?7:this.progress<.81?8:9;
  const base=pass===0?0:6+pass;
  return layer<this.layerIndex||(layer===this.layerIndex&&this.layerProgress>=.18)?(pass===0?7:base+1):base;
 }
 seekLayer(index:number,phase=.34){const pass=storyStages.find(s=>s.pass===this.passIndex)!;this.playing=false;this.seek(pass.start+(pass.end-pass.start)*(Math.max(0,Math.min(this.layerCount-1,index))+phase)/this.layerCount);}
 next(){this.playing=false;if(this.layerActive){const next=phases.find(p=>p.start>this.layerProgress+.00001);this.seek(next?this.stage.start+(this.stage.end-this.stage.start)*(this.layerIndex+next.start)/this.layerCount:this.stage.start+(this.stage.end-this.stage.start)*(this.layerIndex+1)/this.layerCount);}else this.seek(this.stage.end);}
 restart(){this.progress=0;this.playing=true;}
}
export function samplePath(points:readonly (readonly number[])[],t:number):number[]{
 const lengths=points.slice(1).map((p,i)=>Math.hypot(...p.map((v,j)=>v-points[i][j])));
 let distance=clamp(t)*lengths.reduce((a,b)=>a+b,0);
 for(let i=0;i<lengths.length;i++){if(distance<=lengths[i]||i===lengths.length-1){const f=lengths[i]===0?0:distance/lengths[i];return points[i].map((v,j)=>v+(points[i+1][j]-v)*f);}distance-=lengths[i];}
 return [...points[0]];
}
