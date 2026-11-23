// Project saved observations into the workbench; never invent cross-pass timestamps.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { Database } from 'bun:sqlite';
import { existsSync } from 'node:fs';

const root = resolve(process.argv[2]), out = resolve(process.argv[3]);
await mkdir(out, {recursive:true});
const read = async p => readFile(resolve(root,p),'utf8');
const json = async p => JSON.parse(await read(p));
const digest = async p => createHash('sha256').update(await readFile(resolve(root,p))).digest('hex');
const binary = await digest('bin/qgre-infer'), serverBinary = await digest('bin/qgre-serve');
const captureData = JSON.parse(await readFile(resolve(out,'gemma4-e2b.json'),'utf8'));
const sourceBundle = JSON.parse(await readFile(resolve(import.meta.dirname,'../../..',captureData.source),'utf8'));
const captureSource = c => c.source.includes('/later-context/') ? 'taps:later-context' : 'taps:first-reply';
const stageRole = {n1:'mixer:0', k_raw:'mixer:0',k_normed:'mixer:0',k_rope:'mixer:0',v_normed:'mixer:0',q_raw:'mixer:0',q_normed:'mixer:0',q_rope:'mixer:0',attn_out:'mixer:2',o_proj:'main:0',post_attn:'main:2',n2:'gate-up:0',geglu:'main:8',mlp_down:'main:10',layer_out:'main:12'};
for (const c of captureData.captures) {
  const ranges = await json(`range-${c.position}.json`), values=[];
  for(const s of ranges.stages) {
    if(!stageRole[s.stage]) continue;
    for(const row of s.rows) {
      const recordId=`l${row.layer}:${stageRole[s.stage]}`;
      if(!c.steps.some(step=>step.id===recordId)) throw Error(`Missing operation ${recordId}`);
      const counts=[Number(row.zero_scale_active_blocks),...row.exponent_histogram.map(Number)];
      if(counts.reduce((a,b)=>a+b,0)!==Number(row.active_blocks)) throw Error('Histogram population mismatch');
      values.push({id:`value:${c.id}:${s.stage}:${row.layer}`,recordId,worker:null,instrumentId:'tap-range',label:s.stage,boundary:`Stored ${s.stage}, layer ${Number(row.layer)+1}`,edges:Array.from({length:18},(_,i)=>i-1),counts,zeros:Number(row.zero_scale_active_blocks),saturated:null,population:'32-element blocks; bin -1 is the encoded zero pair, other bins are maximum code exponents'});
    }
  }
  const bundle={schema:'qgre.profiling-evidence.v1',id:`e2b-${captureSource(c)==='taps:first-reply'?'first-turn':'later-context'}-${c.id}`,label:`E2B stored values at position ${c.position}`,origin:'recorded',captureId:c.id,tapeSha256:c.tapeSha,clockDomain:`globaltimer-offsets:${c.id}`,
    instruments:[{id:'tap-range',label:'Stored FP8 blocks',method:'QGRE tap-range v3 reader; active blocks only, padding excluded',source:resolve(root,`range-${c.position}.json`),pass:captureSource(c)==='taps:first-reply'?'First reply · diagnostic':'Saved later prompt · cold prefill · diagnostic',cache:'Not controlled',clocks:'Not locked',sampleNs:null,binary}],
    tasks:[],dependencies:[],instructions:[],work:[],buffers:[],samples:[],values,tokens:[],experiments:[]};
  await writeFile(resolve(out,`profiling-${c.id}.json`),JSON.stringify(bundle));
}
const turnsRaw=(await read('conversation-normal/turns.jsonl')).trim().split('\n').filter(Boolean).map(JSON.parse);
const replies=await Promise.all(turnsRaw.map(t=>json(`conversation-normal/turn-${String(t.turn).padStart(3,'0')}.response.json`)));
const seenPhrases=new Set();
const textReuse=turnsRaw.map((t,i)=>{
  const ids=replies[i].choices[0].token_ids,phrases=new Set();
  for(let j=0;j+8<=ids.length;j++) phrases.add(ids.slice(j,j+8).join(','));
  const novel=[...phrases].filter(p=>!seenPhrases.has(p)).length;
  for(const phrase of phrases) seenPhrases.add(phrase);
  return {id:`text-reuse:${t.turn}`,turnId:`turn:${t.turn}`,windowTokens:8,uniquePhrases:phrases.size,newPhrases:novel,newFraction:phrases.size?novel/phrases.size:null};
});
const log=await read('conversation-normal/server.stderr.txt');
const requestLogs=log.split('[qgre-serve] POST /v1/completions -> Ok');
const offset=(await read('host-utc-offset.txt')).trim();
const telemetry=(await read('conversation-normal/gpu.csv')).trim().split('\n').slice(1).map(line=>{
  const cells=line.split(',').map(s=>s.trim());
  return {ms:Date.parse(cells[0].replaceAll('/','-').replace(' ','T')+offset),smMHz:parseFloat(cells[7]),memoryMiB:parseFloat(cells[4]),powerW:parseFloat(cells[9]),temperatureC:parseFloat(cells[10])};
}).filter(row=>Object.values(row).every(Number.isFinite));
const summary=rows=>Object.fromEntries(['smMHz','memoryMiB','powerW','temperatureC'].map(key=>[key,{min:Math.min(...rows.map(r=>r[key])),max:Math.max(...rows.map(r=>r[key])),mean:rows.reduce((sum,r)=>sum+r[key],0)/rows.length}]));
function fit(rows){
  if(rows.length<2) return null;
  const n=rows.length,x=rows.reduce((s,p)=>s+p[0],0)/n,y=rows.reduce((s,p)=>s+p[1],0)/n;
  const denominator=rows.reduce((s,p)=>s+(p[0]-x)**2,0);
  if(!denominator) return null;
  const slope=rows.reduce((s,p)=>s+(p[0]-x)*(p[1]-y),0)/denominator,intercept=y-slope*x;
  const variance=rows.reduce((s,p)=>s+(p[1]-y)**2,0);
  return {n,slope,intercept,rSquared:variance?1-rows.reduce((s,p)=>s+(p[1]-intercept-slope*p[0])**2,0)/variance:null};
}
const turns=turnsRaw.map((t,i)=>{
  const launch=requestLogs[i]?.match(/launch: kind=megagen .*?gen_tokens=(\d+) gpu_us=(\d+)/);
  if(launch && Number(launch[1])!==t.usage.completion_tokens-1) throw Error('Persistent launch does not match reply token count');
  const prefills=[...requestLogs[i].matchAll(/launch: kind=prefill rows=(\d+) .*?gpu_us=(\d+)/g)];
  const prefillNs=prefills.reduce((sum,p)=>sum+Number(p[2])*1000,0),prefillRows=prefills.reduce((sum,p)=>sum+Number(p[1]),0);
  const samples=telemetry.filter(s=>s.ms>=Number(BigInt(t.startUnixNs)/1000000n)&&s.ms<=Number(BigInt(t.endUnixNs)/1000000n));
  return {id:`turn:${t.turn}`,sourceId:'production:conversation',ordinal:t.turn,promptTokens:t.usage.prompt_tokens,cachedTokens:t.usage.prompt_tokens_details?.cached_tokens??0,generatedTokens:t.usage.completion_tokens,cumulativeTokens:t.cumulative_generated,hostNs:Number(BigInt(t.endUnixNs)-BigInt(t.startUnixNs)),prefillNs,prefillRows,gpuNs:launch?Number(launch[2])*1000:null,decodeTokens:launch?Number(launch[1]):null,finish:t.finish_reason,telemetry:samples.length?{sourceId:'telemetry:conversation',samples:samples.length,...summary(samples)}:null,request:resolve(root,`conversation-normal/turn-${String(t.turn).padStart(3,'0')}.request.json`),response:resolve(root,`conversation-normal/turn-${String(t.turn).padStart(3,'0')}.response.json`)};
});
function readTrace(path,sourceId) {
  const db = new Database(resolve(root,path),{readonly:true});
  const launches=db.query('SELECT k.rowid AS ordinal,s.value AS name,k.start AS startNs,k.end AS endNs,k.registersPerThread AS registers,k.gridX AS grid,k.blockX AS threads,k.dynamicSharedMemory AS dynamicSharedBytes,k.staticSharedMemory AS staticSharedBytes,k.dynamicSharedMemory+k.staticSharedMemory AS sharedBytes FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').all().map(l=>({...l,id:`${sourceId}:launch:${l.ordinal}`,sourceId}));
  const last=launches.at(-1);
  if(!last) throw Error(`No CUDA launch in ${path}`);
  const metrics=db.query('SELECT n.metricName AS name,count(*) AS samples,min(g.value) AS min,avg(g.value) AS mean,max(g.value) AS max FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS n USING(typeId,metricId) WHERE g.timestamp BETWEEN ? AND ? AND n.metricId IN (7,8,9,16,19,20) GROUP BY n.metricName').all(last.startNs,last.endNs).map((m,i)=>({...m,id:`${sourceId}:metric:${i}`,sourceId,launchId:last.id,unit:'%'}));
  const warnings=db.query('SELECT text FROM DIAGNOSTIC_EVENT WHERE severity=2').all().map(r=>r.text);
  const memoryEvents=db.query('SELECT e.rowid AS ordinal,e.start AS timestampNs,e.globalPid AS process,e.contextId,e.address,e.bytes,k.label AS memoryKind,o.label AS operation,e.correlationId FROM CUDA_GPU_MEMORY_USAGE_EVENTS e JOIN ENUM_CUDA_MEM_KIND k ON e.memKind=k.id JOIN ENUM_CUDA_DEV_MEM_EVENT_OPER o ON e.memoryOperationType=o.id ORDER BY e.start').all().map(e=>({...e,id:`${sourceId}:memory:${e.ordinal}`,sourceId,address:String(e.address)}));
  warnings.push('GPC clock export is excluded after invalid negative samples in the early trace. nvidia-smi telemetry is retained separately.','Metric means use samples whose timestamps fall inside the final decode launch. They do not describe the preceding prefill.');
  db.close(); return {launches,metrics,warnings,memoryEvents};
}
const sourceId='nsys:first-reply';
const {launches,metrics,warnings,memoryEvents}=readTrace('nsys-attempt-1.sqlite',sourceId);
const later=existsSync(resolve(root,'later-context/trace.sqlite'))?readTrace('later-context/trace.sqlite','nsys:later-diagnostic'):null;
if(later){launches.push(...later.launches);metrics.push(...later.metrics);memoryEvents.push(...later.memoryEvents);}
const generatedTokens=turns.at(-1)?.cumulativeTokens??0;
const laterTurn=existsSync(resolve(root,'later-context/source-request.txt'))?Number((await read('later-context/source-request.txt')).match(/turn-(\d+)\.request/)?.[1]):null;
const tokenCorrespondence=new Map();
for(const c of captureData.captures) {
  const turn=captureSource(c)==='taps:first-reply'?1:laterTurn;
  if(!turn || !c.tokenRecord) continue;
  const reply=await json(`conversation-normal/turn-${String(turn).padStart(3,'0')}.response.json`),index=c.position-reply.usage.prompt_tokens;
  tokenCorrespondence.set(c.id,{turnId:`turn:${turn}`,productionOutputIndex:index,inputMatches:c.tokenRecord.input===reply.choices[0].token_ids[index],outputMatches:c.tokenRecord.output===reply.choices[0].token_ids[index+1],scope:'Two token IDs at this selected position; not a full trajectory or quality comparison.'});
}
const run={schema:'qgre.run-evidence.v1',id:'gemma4-e2b-30k-2026-11-05',label:'E2B · a conversation growing to 30k',targetTokens:30000,generatedTokens,
  configuration:{checkpoint:(await read('checkpoint.sha256')).trim().split('\n').map(line=>{const [sha256,path]=line.split(/\s+/);return{path,sha256};}),device:'NVIDIA GeForce RTX 5080',deviceId:'GPU-b6d3869b-4794-4ff4-0398-47df517a8c82',deviceDetails:resolve(root,'device.xml'),policySha:log.match(/policy_sha=([a-f0-9]+)/)?.[1]??null,geometry:Object.fromEntries([...log.matchAll(/launch: kind=megagen ([^\n]+)/g)][0][1].split(' ').map(s=>s.split('=')).filter(([key])=>!['rows','gen_tokens','gpu_us'].includes(key))),sampling:{greedy:true,speculation:false,parallelRequests:1,retainedConversation:true},clockControl:'None; observed, not locked'},
  sources:[{id:'production:conversation',method:'QGRE production server; greedy persistent decode, natural turn endings',artifact:resolve(root,'conversation-normal'),binarySha256:serverBinary,warnings:['Exact source revision of the saved executable is unknown. The binary hash and emitted policy identify this run.','GPU per-token values are reply averages, not individual-token measurements.']},{id:'telemetry:conversation',method:'nvidia-smi every 500 ms; samples inside each complete request wall interval',artifact:resolve(root,'conversation-normal/gpu.csv'),binarySha256:null,warnings:[`Host timestamp offset ${offset}; per-request summaries include prefill and host work as well as decode.`]},{id:sourceId,method:'Nsight Systems 2026.1.3 · first reply, separate pass',artifact:resolve(root,'nsys-attempt-1.nsys-rep'),binarySha256:binary,warnings},{id:'taps:first-reply',method:'QGRE instrumented single-token launches · first reply',artifact:resolve(root,'first-turn-tapes'),binarySha256:binary,warnings:['Diagnostic timings are not production timings. Stored top-bin codes cannot establish clipping.']}],turns,launches,metrics,
  captures:captureData.captures.map(c=>{const meanTail=kind=>{const steps=c.steps.filter(s=>s.role==='mixer:2'&&c.layers[s.layer].kind===kind);return steps.reduce((a,s)=>a+s.end-s.lastStart,0)/steps.length;};return{id:c.id,sourceId:captureSource(c),position:c.position,durationNs:c.duration,profiling:`/evidence/e2b-30k/profiling-${c.id}.json`,slidingTailNs:meanTail('S'),fullTailNs:meanTail('F'),tokenCorrespondence:tokenCorrespondence.get(c.id)??null};}),
  gaps:[{area:'Dependency readiness',missing:'No generation-specific producer-ready events; phase waits do not identify their cause.'},{area:'Buffer lifetime',missing:'No per-buffer first use, last use and release events; CUDA allocation lifetime is a different scope.'},{area:'Useful work',missing:'No matched valid/issued-element and requested/transferred-byte denominators per operation.'},{area:'Instruction supply',missing:'Needs exact-binary PC samples and instruction counters; aggregate activity is insufficient.'},{area:'Token timing',missing:'Production records whole persistent launches. No individual-token timestamps were captured.'},{area:'Later-context internals',missing:'These seven instrumented slices belong to the first reply, not the complete 30k conversation.'}],
  observations:[{claim:'The first reply kept roughly three quarters of SMs active while instruction issue, tensor activity and DRAM activity remained much lower.',evidenceIds:metrics.filter(m=>m.sourceId===sourceId).map(m=>m.id),next:'Bind exact-binary warp-stall PCs to operations and producer-readiness events to separate synchronization, instruction overhead and supply latency.'},{claim:'Continuation replies report cached prompt tokens; the conversation is not simply prefilling the entire history at each turn.',evidenceIds:turns.filter(t=>t.cachedTokens>0).map(t=>t.id),next:'Compare later reply averages against context length and inspect full-context attention if its share grows.'}]};
if(later) run.sources.push({id:'nsys:later-diagnostic',method:'Nsight Systems + instrumented decode after cold prefill of a saved later prompt',artifact:resolve(root,'later-context/trace.nsys-rep'),binarySha256:binary,warnings:[...later.warnings,'Diagnostic single-token launch; not the original server cache state or production timing.']});
if(run.captures.some(c=>c.sourceId==='taps:later-context')) {
  run.sources.push({id:'taps:later-context',method:'QGRE instrumented decode after cold prefill of a saved later prompt',artifact:resolve(root,'later-context/tapes'),binarySha256:binary,warnings:['A separate diagnostic pass; not the production server cache state or timing.']});
  run.gaps=run.gaps.filter(g=>g.area!=='Later-context internals');
}
if(existsSync(resolve(root,'capture-failures.json'))) {
  run.unavailableCaptures=await json('capture-failures.json');
  if(!run.sources.some(s=>s.id==='taps:later-context')) run.sources.push({id:'taps:later-context',method:'Later prompt diagnostic: stored values admitted, phase timing rejected',artifact:resolve(root,'later-context/tapes'),binarySha256:binary,warnings:run.unavailableCaptures.map(c=>c.reason)});
  run.valueCaptures=await Promise.all(run.unavailableCaptures.map(async c=>({id:`stored-values:${c.position}`,sourceId:c.sourceId,position:c.position,artifact:resolve(root,`range-${c.position}.json`),timingBinding:null,report:await json(`range-${c.position}.json`)})));
  run.gaps=run.gaps.map(g=>g.area==='Later-context internals'?{...g,missing:'Later stored values and the Nsight launch trace were captured. The internal phase timeline is unavailable because its recorded clock bound failed validation.'}:g);
}
run.counterPasses=(sourceBundle.counters??[]).map((c,i)=>({...c,id:`ncu:early:${i}`,sourceId:'ncu:early',timelineBinding:null}));
if(run.counterPasses.length) {
  run.sources.push({id:'ncu:early',method:'Nsight Compute application replay · bounded early decode launch',artifact:resolve(root,'ncu-early.ncu-rep'),binarySha256:binary,warnings:['Clock and cache control disabled. Counters describe the early fixture only; no timestamp or PC binding to an internal phase is established.']});
  run.gaps=run.gaps.map(g=>g.area==='Instruction supply'?{...g,missing:'Early-launch counters are captured. Exact PC-to-operation attribution and later-context counters remain unbound.'}:g);
  for(const pass of run.counterPasses) {
    const metric=name=>pass.metrics.find(m=>m.name===name)?.value;
    const issue=metric('smsp__issue_active.avg.pct_of_peak_sustained_active'), registers=metric('launch__occupancy_limit_registers'), shared=metric('launch__occupancy_limit_shared_mem');
    if(issue!==undefined&&registers!==undefined&&shared!==undefined) run.observations.push({claim:`The early counter replay measures ${issue}% instruction issue activity. Registers limit residency to ${registers} block(s) per SM and shared memory independently limits it to ${shared}.`,evidenceIds:[pass.id],next:'Bind sampled sleep and memory-dependency PCs to producer/consumer operations. Evaluate register and shared-memory budgets together; these counters do not establish a speedup or a later-context bottleneck.'});
  }
}
const early=run.captures.filter(c=>c.sourceId==='taps:first-reply');
run.memoryEvents=memoryEvents;
run.memoryScope='CUDA allocation and release events. They do not establish application buffer ownership, first use, last use, or reclaimable capacity.';
run.textReuse=textReuse;
run.textReuseScope='Distinct contiguous 8-token sequences in each reply compared with earlier replies. Measures token-phrase reuse, not semantic correctness or reasoning quality.';
const pinned=memoryEvents.filter(e=>e.sourceId==='nsys:first-reply'&&e.memoryKind==='Pinned'&&e.bytes===2147483648);
if(pinned.length) run.observations.push({claim:'The first trace records a 2 GiB pinned-host allocation retained through decode, matching the size the QGRE log calls its conversation lot.',evidenceIds:pinned.map(e=>e.id),next:'Measure occupied bytes and reuse within this reserved lot before deciding whether less capacity or a later allocation would help.'});
run.contextGrowth={
  production:{sourceId:'production:conversation',evidenceIds:turns.map(t=>t.id),x:'Mean context position inside a reply',y:'Mean GPU decode ns per token',fit:fit(turns.filter(t=>t.decodeTokens>0&&t.gpuNs!==null).map(t=>[t.promptTokens+(t.decodeTokens-1)/2,t.gpuNs/t.decodeTokens]))},
  fullAttention:{sourceId:'taps:first-reply',evidenceIds:early.map(c=>c.id),x:'Captured context position',y:'Mean full-attention last-entry-to-last-exit tail ns',fullLayerCount:captureData.captures[0].layers.filter(l=>l.kind==='F').length,fit:fit(early.map(c=>[c.position,c.fullTailNs]))},
  slidingAttention:{sourceId:'taps:first-reply',evidenceIds:early.filter(c=>c.position>=512).map(c=>c.id),x:'Captured context position after the window fills',y:'Mean sliding-attention last-entry-to-last-exit tail ns',fit:fit(early.filter(c=>c.position>=512).map(c=>[c.position,c.slidingTailNs]))},
  interpretation:'Cross-pass agreement is a hypothesis discriminator, not causal attribution or a measured speedup. Investigate full-history attention partitioning in policy; verify by controlled same-path runs and the existing coherence gate.'
};
run.observations.push({claim:'Production context-growth slope closely tracks the combined growth of full-history attention tails in the early diagnostic pass; sliding-window tails plateau.',evidenceIds:[...turns.map(t=>t.id),...early.map(c=>c.id)],next:'Test the attention position-slice policy as a controlled bundle; retain the same precision, token behavior and full megakernel path. Compare the context slope, not only a short-prompt average.'});
run.relations=[...turns.map(t=>({from:t.id,to:t.sourceId,kind:'measured_by'})),...turns.slice(1).map((t,i)=>({from:turns[i].id,to:t.id,kind:'retained_in_context'})),...launches.map(l=>({from:l.id,to:l.sourceId,kind:'measured_by'})),...metrics.map(m=>({from:m.id,to:m.launchId,kind:'sampled_during'})),...run.captures.map(c=>({from:c.id,to:c.sourceId,kind:'measured_by'})),...run.captures.filter(c=>c.tokenCorrespondence).map(c=>({from:c.id,to:c.tokenCorrespondence.turnId,kind:'replays_context_of'})),...run.counterPasses.map(c=>({from:c.id,to:c.sourceId,kind:'measured_by'}))];
run.relations.push(...memoryEvents.map(e=>({from:e.id,to:e.sourceId,kind:'measured_by'})));
run.relations.push(...textReuse.map(e=>({from:e.id,to:e.turnId,kind:'derived_from'})));
run.relations.push(...(run.unavailableCaptures??[]).map(c=>({from:c.id,to:c.sourceId,kind:'rejected_timing_from'})),...(run.valueCaptures??[]).map(c=>({from:c.id,to:c.sourceId,kind:'measured_by'})));
if(run.counterPasses.length&&existsSync(resolve(root,'instruction-sites.json'))) {
  run.instructionEvidence=await json('instruction-sites.json');
  run.instructionEvidence.source=resolve(root,run.instructionEvidence.source);
  run.relations.push(...run.instructionEvidence.sites.map(s=>({from:s.id,to:s.counterPassId,kind:'sampled_in'})));
}
await writeFile(resolve(out,'run.json'),JSON.stringify(run));
await writeFile(resolve(root,'run-evidence.json'),JSON.stringify(run,null,2));
console.log(`${generatedTokens} new tokens, ${turns.length} replies, ${captureData.captures.length} diagnostic slices; separate pass identities preserved.`);
