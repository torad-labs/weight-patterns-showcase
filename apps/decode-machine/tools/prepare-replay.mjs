// Reuse the replay reader's admission and metrics; this is a display projection, not a second decoder.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {createHash} from 'node:crypto';
import {normalizeInternalCapture,stepMetrics} from '../../decode-replay/internal-model.mjs';
import {admitWorldTopology} from '../../decode-replay/world-lifecycle.mjs';
import {decodeSequence} from '../../decode-replay/decode-sequence.mjs';

const project=resolve(import.meta.dirname,'..'),repo=resolve(project,'../..');
const source=process.argv[2]??'runs/legs/internal-decode-replay/lifecycle-ready/bundle.json';
const bytes=await readFile(resolve(repo,source));
const bundle=JSON.parse(bytes),model=bundle.models.find(m=>m.id==='gemma4_e2b_q4_decode_production');
if(!model)throw new Error('Gemma 4 E2B source anatomy is absent.');
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const captures=[];
for(const raw of bundle.timelines.filter(t=>t.source?.kind==='qgre-phase-report')){
 const {replay}=normalizeInternalCapture(raw),topology=admitWorldTopology(replay,raw.captureTopology);
 if(!topology.known)throw new Error('A recorded architecture needs captured topology.');
 const tape=await readFile(resolve(repo,raw.source.path));
 const tableBytes=await readFile(resolve(repo,dirname(raw.source.path),'launch_tables.json'));
 const binding=raw.launchBinding;
 if(hash(tape)!==binding.tape_sha256||hash(tableBytes)!==binding.launch_tables_sha256||topology.bindingSha!==binding.launch_tables_sha256)throw new Error('Capture binding changed; rebuild the source replay before importing.');
 const launch=JSON.parse(tableBytes),facts=raw.captureTopology.modelFacts,journey=decodeSequence(replay);
 const steps=replay.steps.map(s=>{
  const m=stepMetrics(s);
  return {id:s.id,layer:Number(s.layer),role:s.id.slice(s.id.indexOf(':')+1),name:s.name,kind:s.kind,parent:s.parent,predecessors:s.predecessors,description:s.description,fromTap:s.fromTap,toTap:s.toTap,start:m.start,lastStart:m.end-m.critical,end:m.end,anomalies:m.anomalies,observations:m.observations.map(o=>[Number(o.block),o.start,o.end,String(o.sm)])};
 });
 const layers=replay.layers.map((l,i)=>{
  const roots=steps.filter(s=>s.layer===i&&s.parent===null&&s.kind!=='sentinel'&&s.observations.length);
  return {id:i,kind:l.kind,owner:Number(topology.owners[i]),window:Number(topology.windows[i]),headDim:Number(launch.tables.HeadDimOf[i]),ffnWidth:Number(launch.tables.FfnKBlocksOf[i])*32,start:Math.min(...roots.map(s=>s.start)),journeyStart:journey.layers[i].start,end:Math.max(...roots.map(s=>s.end))};
 });
 const start=Math.min(...layers.map(l=>l.start)),end=Math.max(...layers.map(l=>l.end));
 captures.push({id:String(binding.position),position:Number(binding.position),grid:Number(binding.grid_blocks),queryHeads:Number(facts.num_q_heads),kvHeads:Number(facts.sliding_kv_heads),hidden:Number(facts.hidden_size),outputSlices:Number(launch.values.o_k_slices),downSlices:Number(launch.values.down_k_slices),source:raw.source.path,tapeSha:binding.tape_sha256,tablesSha:binding.launch_tables_sha256,start,end,duration:end-start,layers,steps,coverage:replay.coverage,semantics:replay.semantics});
 const capture=captures.at(-1),captureDir=resolve(repo,dirname(raw.source.path));
 try {
  const prompt=JSON.parse(await readFile(resolve(captureDir,'prompt.json'),'utf8'));
  const records=(await readFile(resolve(captureDir,'steps.jsonl'),'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  const token=records.find(r=>r.position===capture.position);
  if(typeof prompt.prompt==='string')capture.promptText=prompt.prompt;
  if(token&&Number.isSafeInteger(token.token_in)&&Number.isSafeInteger(token.token_out))capture.tokenRecord={input:token.token_in,output:token.token_out,source:resolve(captureDir,'steps.jsonl')};
 } catch(error) { if(error.code!=='ENOENT')throw error; }
}
if(!captures.length)throw new Error('This machine view needs at least one bound phase recording.');
const out=process.argv[3]?resolve(process.argv[3]):resolve(project,'public/replay');await mkdir(out,{recursive:true});
const anatomy={revision:bundle.architecture.revision,binding:bundle.architecture.binding,entry:model.entry,policy:model.policy,roles:model.config.roles,shape:model.config.model_shape,nodes:model.nodes.map(n=>({...n,source:n.source?.replace(repo+'/',''),location:n.location?.replace(repo+'/','')})),edges:model.edges};
await writeFile(resolve(out,'anatomy.json'),JSON.stringify(anatomy));
const result={schema:'qgre.mechanical-replay.v1',label:'Gemma 4 E2B',source,sourceSha:hash(bytes),sourceRevision:bundle.architecture.revision,captures};
await writeFile(resolve(out,'gemma4-e2b.json'),JSON.stringify(result));
console.log(captures.map(c=>`${c.position}: ${c.layers.length} layers, ${c.steps.length} records, ${c.duration/1e6} ms instrumented envelope`).join('\n'));
