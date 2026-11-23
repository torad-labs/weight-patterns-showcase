import {readTraceStudy,tracePercent,traceScale,decisionAlternatives,type TraceStudy,type TraceCase} from './trace-model';
import {appUrl} from './replay-model';
import './trace-panel.css';
const el=<K extends keyof HTMLElementTagNameMap>(tag:K,text='',cls='')=>{const n=document.createElement(tag);n.textContent=text;n.className=cls;return n;};
const button=(text:string,action:()=>void)=>{const b=el('button',text);b.type='button';b.onclick=action;return b;};
const NS='http://www.w3.org/2000/svg';
function svgNode(tag:string,attrs:Record<string,string|number>,text=''){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));n.textContent=text;return n;}
const stageLabels=['Incoming signal','Weight projection','Normalization','Layer 16 exit','Layer 24 exit','Layer 34 exit','Final readout'];
function energyDrawing(before:string,after:string){
 const outcomes=[before,after].map(value=>value.split(' / ')),ceiling=Math.max(...outcomes.flatMap(row=>row.slice(0,2).map(Number)));
 const drawing=el('div','','trace-energy');
 for(const [index,row]of outcomes.entries()){
  const group=el('div','','trace-energy-arm');group.append(el('b',`${index?'Edited':'Original'} · ${row[2]}`));
  for(const [i,name]of ['Available','Required'].entries()){const line=el('div','','trace-energy-line'),track=el('div','','trace-energy-track'),bar=el('i');bar.style.width=`${Number(row[i])/ceiling*100}%`;track.append(bar);line.append(el('span',`${name} ${row[i]} J`),track);group.append(line);}
  drawing.append(group);
 }
 return drawing;
}
class TracePanel {
 private study?:TraceStudy;private selected=0;private step=0;private playing=false;private frame=0;private started=0;private duration=20;private ceiling=1;
 private jump=button('First changed token',()=>{this.stop();this.step=this.selectedCase!.boundary.divergence??this.selectedCase!.predictions.length-1;this.draw();});
 private host=el('section','','trace-workbench');private content=el('div');private nav=el('nav','','trace-cases');private slider=el('input');private play=button('Play shared history',()=>this.toggle());private readout=el('output');private plot=el('div','','trace-plot');private details=el('div','','trace-details');private caseInfo=el('div','','trace-case-info');private boundary=el('p','','trace-boundary');private selectedCase?:TraceCase;
 constructor(){
  this.host.id='weight-traces';this.host.setAttribute('aria-label','Connected weight edit experiment');
  this.host.addEventListener('keydown',event=>event.stopPropagation());
  const head=el('div','','trace-heading'),title=el('div');title.append(el('h2','Follow the weight edit'),el('p','One stored change. Watch its effect travel through four different responses.'));
  const download=el('a','Download connected evidence');download.href=appUrl('evidence/weight-traces/connected-traces.json');download.download='connected-weight-traces.json';head.append(title,download);this.host.append(head,el('p','Loading saved native recordings…','trace-status'),this.content);document.querySelector('main')!.after(this.host);
  const jump=document.querySelector<HTMLAnchorElement>('[data-open-traces]');if(jump)jump.onclick=e=>{e.preventDefault();this.host.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});};
  void fetch(download.href).then(r=>{if(!r.ok)throw Error('Saved traces are not available.');return r.json();}).then(readTraceStudy).then(s=>{this.study=s;this.host.querySelector('.trace-status')!.textContent=s.engine+' · saved executions';this.mount();const query=new URLSearchParams(location.search),id=query.get('trace');if(id){const i=s.cases.findIndex(c=>c.id===id);this.select(i<0?0:i);requestAnimationFrame(()=>this.host.scrollIntoView());}}).catch(e=>{this.host.querySelector('.trace-status')!.textContent=String(e);this.content.append(button('Retry loading',()=>location.reload()));});
 }
 private mount(){
  this.ceiling=traceScale(this.study!);
  this.nav.setAttribute('aria-label','Compare repaired, preserved and damaged responses');
  for(const [i,c]of this.study!.cases.entries()){const b=button(c.label,()=>this.select(i));b.append(energyDrawing(c.before,c.after));this.nav.append(b);}
  const controls=el('div','','trace-controls'),label=el('label','Shared-history prediction');this.slider.type='range';this.slider.min='0';this.slider.step='1';this.slider.setAttribute('aria-label','Shared-history prediction');this.slider.oninput=()=>{this.stop();this.step=Number(this.slider.value);this.draw();};label.append(this.slider);this.readout.setAttribute('aria-live','off');
  const duration=el('label','Whole shared history '),seconds=el('input');seconds.type='number';seconds.min='2';seconds.max='600';seconds.value=String(this.duration);seconds.setAttribute('aria-label','Trace playback duration in seconds');seconds.onchange=()=>{this.stop();this.duration=Math.max(2,Math.min(600,Number(seconds.value)||20));seconds.value=String(this.duration);};duration.append(seconds,document.createTextNode(' seconds'));
  const more=el('details','','trace-timing');more.append(el('summary','Timing & first change'),this.jump,duration);controls.append(this.play,label,this.readout,more);
  const body=el('div','','trace-body');body.append(this.plot,this.details);
  this.content.append(this.nav,this.caseInfo,controls,this.boundary,body,el('p','Line height = change relative to the original vector at each measured boundary. One fixed scale across all four cases. The route is a schematic of connected computation; it is not the hidden geometry itself.','trace-caption'));
  this.select(0);
 }
 private select(index:number){this.stop();this.selected=index;this.selectedCase=this.study!.cases[index];const c=this.selectedCase;this.step=c.boundary.divergence??Math.min(235,c.predictions.length-1);this.slider.max=String(c.predictions.length-1);Array.from(this.nav.children).forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
  this.jump.textContent=c.boundary.divergence===null?'End of shared history':'First changed token';
  this.caseInfo.replaceChildren();const prompt=el('details'),sum=el('summary','Read the problem & interpretation');prompt.append(sum,el('strong',c.interpretation),el('p',c.prompt));const result=el('div'),selectedEnergy=el('div','','trace-selected-energy');selectedEnergy.append(energyDrawing(c.before,c.after));result.append(selectedEnergy,el('p',`Expected: ${c.expected.availableJ} J available · ${c.expected.requiredJ} J required · ${c.expected.answer}`));this.caseInfo.append(result,prompt);this.draw();}
 private draw(){
  const c=this.selectedCase!,p=c.predictions[this.step];this.slider.value=String(this.step);this.readout.value=`${this.step} / ${c.predictions.length-1}`;
  this.boundary.textContent=c.boundary.divergence===null?`All ${c.boundary.common} generated tokens stay identical. The recorded internal signals still change.`:`Same incoming tokens through prediction ${c.boundary.divergence}. At that decision the continuations separate; later internals are not paired here.`;
  const svg=svgNode('svg',{viewBox:'0 0 1100 340',role:'img','aria-label':`Connected measured changes at prediction ${this.step}. ${p.measures.map((m,i)=>stageLabels[i]+' '+tracePercent(m.relative)).join('; ')}`});
  const ceiling=this.ceiling,baseline=205,range=105,xs=p.measures.map((_,i)=>80+i*155);
  for(const tick of [0,.5,1]){const y=baseline-tick*range;svg.append(svgNode('path',{d:`M60 ${y}H1050`,stroke:'#b7c0b8','stroke-width':1,'stroke-dasharray':'3 5',fill:'none'}),svgNode('text',{x:8,y:y-6,'font-size':13,fill:'#53645f'},tracePercent(tick*ceiling)));}
  svg.append(svgNode('path',{d:`M${xs[0]} ${baseline}H${xs.at(-1)}`,stroke:'#567875','stroke-width':3,fill:'none'}));
  const points=p.measures.map((m,i)=>`${xs[i]},${baseline-(m.relative??0)/ceiling*range}`);
  svg.append(svgNode('polyline',{points:points.join(' '),fill:'none',stroke:'#896029','stroke-width':4,'stroke-linejoin':'round'}));
  for(const [i,m]of p.measures.entries()){
   const x=xs[i],y=baseline-(m.relative??0)/ceiling*range;
   if(i===1){svg.append(svgNode('rect',{x:x-33,y:27,width:66,height:36,rx:3,fill:'#e1c181',stroke:'#896029','stroke-width':1.5}));for(let j=0;j<5;j++)svg.append(svgNode('circle',{cx:x-23+j*11.5,cy:40,r:3.7,fill:'#dc9c4a',stroke:'#896029'}));svg.append(svgNode('path',{d:`M${x-21} 63V76H${x+21}V63`,fill:'none',stroke:'#896029','stroke-width':2}));}
   else svg.append(svgNode('rect',{x:x-23,y:40,width:46,height:18,rx:3,fill:'#c6ceca',stroke:'#567875'}));
   svg.append(svgNode('path',{d:`M${x} 78V${Math.min(y,baseline)-12}`,stroke:'#a8bcb3','stroke-width':1,fill:'none'}),svgNode('circle',{cx:x,cy:baseline,r:5,fill:'#567875'}),svgNode('rect',{x:x-5,y:y-5,width:10,height:10,fill:'#e1c181',stroke:'#896029','stroke-width':2}),svgNode('text',{x,y:255,'text-anchor':'middle','font-size':15,fill:'#263e3d'},stageLabels[i]),svgNode('text',{x,y:279,'text-anchor':'middle','font-size':16,'font-weight':650,fill:'#896029'},tracePercent(m.relative)));
  }
  svg.append(svgNode('circle',{cx:80,cy:315,r:4,fill:'#567875'}),svgNode('text',{x:92,y:320,'font-size':14,fill:'#53645f'},'Original baseline'),svgNode('rect',{x:229,y:311,width:8,height:8,fill:'#e1c181',stroke:'#896029'}),svgNode('text',{x:245,y:320,'font-size':14,fill:'#53645f'},'Edited signal · change / original norm at each boundary'));
  this.plot.replaceChildren(el('p','Scroll the route sideways to follow every measured boundary.','trace-scroll-hint'),svg);const input=el('p',`Incoming signal: ${p.measures[0].absolute===0?'byte-verified unchanged':'different'} · ${this.study!.checkpoint.changedValues.toLocaleString()} BF16 values changed in the layer 16 projection.`,'trace-caption');this.plot.append(input);
  const before=c.original.steps.slice(Math.max(0,this.step-28),this.step).map(s=>s.piece).join('');const context=el('div','','trace-context');context.append(el('h3','Same incoming passage'),el('pre',before||'The prompt has been read. This is the first output decision.'));
  const decisions=el('div','','trace-decisions');for(const [name,decision]of [['Original',p.original],['Edited',p.edited]] as const){const d=el('div');d.append(el('span',name),el('strong',JSON.stringify(decision.piece)));decisions.append(d);}
  const table=el('table');const caption=el('caption','Recorded scores of the selected alternatives');table.append(caption);const tr=el('tr');for(const name of ['Token','Original','Edited'])tr.append(el('th',name));const thead=el('thead');thead.append(tr);table.append(thead);const tbody=el('tbody');for(const a of decisionAlternatives(p)){const row=el('tr');for(const v of [JSON.stringify(a.piece),a.original?.toFixed(5)??'Outside saved top 8',a.edited?.toFixed(5)??'Outside saved top 8'])row.append(el('td',v));tbody.append(row);}table.append(tbody);
  const continuations=el('details');continuations.append(el('summary','Read both complete continuations'));const branches=el('div','','trace-branches');for(const [name,r]of [['Original',c.original],['Edited',c.edited]] as const){const branch=el('div');branch.append(el('h4',name),el('pre',r.content));branches.append(branch);}continuations.append(branches);
  const provenance=el('details');provenance.append(el('summary','Inspect source events and saved evidence'),el('p','Every capture matches its ordinary execution: identical generated tokens and full-vocabulary score hashes.'),el('pre',JSON.stringify({prediction:p.step,position:p.position,sources:c.sources,measures:p.measures,checkpoint:this.study!.checkpoint},null,2)));
  this.details.replaceChildren(context,decisions,table,continuations,provenance);
 }
 private stop(){this.playing=false;cancelAnimationFrame(this.frame);this.play.textContent='Play shared history';}
 private toggle(){if(this.playing){this.stop();return;}if(matchMedia('(prefers-reduced-motion: reduce)').matches){this.step=Math.min(this.step+1,this.selectedCase!.predictions.length-1);this.draw();return;}this.playing=true;this.play.textContent='Pause';if(this.step>=this.selectedCase!.predictions.length-1)this.step=0;this.started=performance.now()-this.step/(this.selectedCase!.predictions.length-1)*this.duration*1000;const tick=(now:number)=>{if(!this.playing)return;const next=Math.min(this.selectedCase!.predictions.length-1,Math.floor((now-this.started)/(this.duration*1000)*(this.selectedCase!.predictions.length-1)));if(next!==this.step){this.step=next;this.draw();}if(next>=this.selectedCase!.predictions.length-1)this.stop();else this.frame=requestAnimationFrame(tick);};this.frame=requestAnimationFrame(tick);}
}
new TracePanel();
