export interface TraceMeasure { name: string; relative: number | null; absolute: number; originalNorm: number; editedNorm: number; cosine: number | null; original: {event:number;node:number;offset:number;width:number}; edited: {event:number;node:number;offset:number;width:number} }
export interface TraceDecision {step:number; selected:number; piece:string; top:{id:number;piece:string;logit:number}[]}
export interface TracePrediction {step:number;position:number;inputCoefficient:number;measures:TraceMeasure[];original:TraceDecision;edited:TraceDecision}
export interface TraceCase {id:string;label:string;before:string;after:string;interpretation:string;prompt:string;expected:Record<string,unknown>;boundary:{common:number;divergence:number|null;eligible:number};predictions:TracePrediction[];original:{content:string;tokens:number[];steps:TraceDecision[]};edited:{content:string;tokens:number[];steps:TraceDecision[]};sources:{original:string;edited:string};fidelity:{tokens:boolean;fullScoreHashes:boolean}}
export interface TraceStudy {schema:string;title:string;engine:string;checkpoint:{tensor:string;original:string;edited:string;changedValues:number;relativeWeightChange:number};stages:string[];cases:TraceCase[];scope:string}
export function readTraceStudy(value: unknown): TraceStudy {
  const s=value as TraceStudy;
  if(s?.schema!=='connected-weight-trace-v1'||!Array.isArray(s.cases)||!s.cases.length)throw Error('This file does not contain connected weight traces.');
  for(const c of s.cases){
    if(!c.fidelity?.tokens||!c.fidelity.fullScoreHashes||c.predictions.length!==c.boundary.eligible)throw Error('Trace capture has not matched its ordinary execution.');
    let common=0;while(common<Math.min(c.original.tokens.length,c.edited.tokens.length)&&c.original.tokens[common]===c.edited.tokens[common])common++;
    if(common!==c.boundary.common||c.boundary.eligible!==Math.min(common+1,c.original.steps.length,c.edited.steps.length))throw Error('Trace crosses its shared input history.');
    for(const [i,p] of c.predictions.entries()){
      if(p.step!==i||p.measures.length!==s.stages.length||p.measures.some((m,j)=>m.name!==s.stages[j]||!Number.isFinite(m.absolute)||m.absolute<0||m.relative!==null&&(!Number.isFinite(m.relative)||m.relative<0)))throw Error('Trace measurements are incomplete.');
    }
  }
  return s;
}
export const tracePercent=(x:number|null)=>x===null?'undefined':x===0?'0%':`${(100*x).toFixed(3)}%`;
export function traceScale(study:TraceStudy){return Math.max(.001,...study.cases.flatMap(c=>c.predictions.flatMap(p=>p.measures.map(m=>m.relative??0))))*1.15;}
export function decisionAlternatives(p:TracePrediction){
  const ids=[...new Set([p.original.selected,p.edited.selected])];
  return ids.map(id=>({id,piece:p.original.top.find(t=>t.id===id)?.piece??p.edited.top.find(t=>t.id===id)?.piece??String(id),original:p.original.top.find(t=>t.id===id)?.logit??null,edited:p.edited.top.find(t=>t.id===id)?.logit??null}));
}
