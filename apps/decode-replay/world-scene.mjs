// Continuous oblique mechanism world. Geometry describes data flow, not chip coordinates.
class DecodeWorldScene{
  constructor(canvas,{select,enter,manual}){
    this.canvas=canvas;this.ctx=canvas.getContext("2d");this.select=select;this.enter=enter;this.manual=manual;
    this.camera={x:0,y:0,scale:1,yaw:-.12,pitch:.58};this.layers=[];this.time=0;this.selected=null;this.hits=[];this.explode=1;this.drag=null;
    this.abort=new AbortController();const on=(type,fn,opts={})=>canvas.addEventListener(type,fn,{...opts,signal:this.abort.signal});
    this.resize=new ResizeObserver(()=>{this.measure();if(this.frameBounds)this.frame(this.frameBounds);else this.draw();});this.resize.observe(canvas);
    on("pointerdown",e=>{this.drag={x:e.clientX,y:e.clientY,sx:e.clientX,sy:e.clientY,orbit:e.shiftKey};canvas.setPointerCapture(e.pointerId);});
    on("pointermove",e=>{if(!this.drag){const n=this.hit(e);canvas.style.cursor=n?"pointer":"grab";canvas.title=n?n.node.name+(n.node.evidence.duration===null?" · "+n.node.evidence.kind:" · "+formatTime(n.node.evidence.duration)):"Drag to pan · wheel to zoom · Shift-drag to orbit";return;}
      const d=this.drag,dx=e.clientX-d.x,dy=e.clientY-d.y;d.x=e.clientX;d.y=e.clientY;
      if(Math.hypot(e.clientX-d.sx,e.clientY-d.sy)>4){this.manual();if(d.orbit){this.camera.yaw+=dx*.004;this.camera.pitch=Math.max(.15,Math.min(1.1,this.camera.pitch+dy*.003));}else this.pan(dx,dy);this.draw();}});
    on("pointerup",e=>{if(this.drag&&Math.hypot(e.clientX-this.drag.sx,e.clientY-this.drag.sy)<5){const h=this.hit(e);if(h)this.select(h);}this.drag=null;});
    on("pointercancel",()=>this.drag=null);
    on("dblclick",e=>{const h=this.hit(e);if(h)this.enter(h);});
    on("wheel",e=>{e.preventDefault();this.manual();const r=canvas.getBoundingClientRect();this.zoom(Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top);},{passive:false});
    on("keydown",e=>{const delta={ArrowLeft:[90,0],ArrowRight:[-90,0],ArrowUp:[0,90],ArrowDown:[0,-90]}[e.key];if(delta){e.preventDefault();this.manual();this.pan(...delta);this.draw();}else if(["+","=","-"].includes(e.key)){e.preventDefault();this.manual();this.zoom(e.key==="-"?.8:1.25);}});
    this.measure();
  }
  measure(){const r=this.canvas.getBoundingClientRect();this.w=r.width;this.h=r.height;const dpr=Math.min(devicePixelRatio||1,2);if(this.canvas.width!==Math.round(r.width*dpr)||this.canvas.height!==Math.round(r.height*dpr)){this.canvas.width=Math.round(r.width*dpr);this.canvas.height=Math.round(r.height*dpr);}this.ctx.setTransform(dpr,0,0,dpr,0,0);}
  set(layers,routes=[],routeFocus=null){this.layers=layers;this.routes=routes;this.routeFocus=routeFocus;this.draw();}
  project(x,y,z=0){const a=this.camera,dx=x-a.x,dy=y-a.y,rx=dx*Math.cos(a.yaw)-dy*Math.sin(a.yaw),ry=dx*Math.sin(a.yaw)+dy*Math.cos(a.yaw);return {x:this.w/2+rx*a.scale,y:this.h/2+(ry*Math.cos(a.pitch)-z*Math.sin(a.pitch))*a.scale,depth:ry};}
  unproject(x,y){const a=this.camera,rx=(x-this.w/2)/a.scale,ry=(y-this.h/2)/a.scale/Math.cos(a.pitch);return [a.x+rx*Math.cos(a.yaw)+ry*Math.sin(a.yaw),a.y-rx*Math.sin(a.yaw)+ry*Math.cos(a.yaw)];}
  pan(dx,dy){this.frameBounds=null;const a=this.camera,rx=dx/a.scale,ry=dy/a.scale/Math.cos(a.pitch);a.x-=rx*Math.cos(a.yaw)+ry*Math.sin(a.yaw);a.y-= -rx*Math.sin(a.yaw)+ry*Math.cos(a.yaw);}
  zoom(factor,x=this.w/2,y=this.h/2){this.frameBounds=null;const before=this.unproject(x,y);this.camera.scale=Math.max(.035,Math.min(4,this.camera.scale*factor));const after=this.unproject(x,y);this.camera.x+=before[0]-after[0];this.camera.y+=before[1]-after[1];this.draw();}
  frame(bounds){this.frameBounds=bounds;this.measure();if(!this.w||!this.h)return;const [x0,y0,x1,y1]=bounds,a=this.camera;a.x=(x0+x1)/2;a.y=(y0+y1)/2;a.scale=1;const corners=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]].map(p=>this.project(...p));const width=Math.max(...corners.map(p=>p.x))-Math.min(...corners.map(p=>p.x)),height=Math.max(...corners.map(p=>p.y))-Math.min(...corners.map(p=>p.y));a.scale=Math.max(this.detail?.45:.035,Math.min((this.w-90)/width,(this.h-100)/height,2));this.draw();}
  hit(e){const r=this.canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;return this.hits.findLast(h=>x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h);}
  polygon(points,fill,stroke,width=1){const c=this.ctx;c.beginPath();points.forEach((p,i)=>i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y));c.closePath();if(fill){c.fillStyle=fill;c.fill();}if(stroke){c.strokeStyle=stroke;c.lineWidth=width;c.stroke();}}
  box(x,y,z,w,h,d,fill,stroke){const p=(a,b,k)=>this.project(a,b,k);const top=[p(x-w/2,y-h/2,z+d),p(x+w/2,y-h/2,z+d),p(x+w/2,y+h/2,z+d),p(x-w/2,y+h/2,z+d)],base=[p(x-w/2,y-h/2,z),p(x+w/2,y-h/2,z),p(x+w/2,y+h/2,z),p(x-w/2,y+h/2,z)];
    this.polygon([top[1],base[1],base[2],top[2]],"#1b2c33",stroke);this.polygon([top[2],base[2],base[3],top[3]],"#233740",stroke);this.polygon(top,fill,stroke);return [...top,...base];}
  line(points,color,width=1,dashed=false){const c=this.ctx;c.beginPath();points.forEach((p,i)=>i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y));c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dashed?[4,5]:[]);c.stroke();c.setLineDash([]);}
  draw(){
    if(!this.w||!this.h)return;const c=this.ctx,w=this.w,h=this.h,scale=this.camera.scale;c.clearRect(0,0,w,h);c.fillStyle="#10191f";c.fillRect(0,0,w,h);this.hits=[];
    const corners=[[0,0],[w,0],[0,h],[w,h]].map(p=>this.unproject(...p)),x0=Math.floor(Math.min(...corners.map(p=>p[0]))/200)*200,x1=Math.max(...corners.map(p=>p[0])),y0=Math.floor(Math.min(...corners.map(p=>p[1]))/200)*200,y1=Math.max(...corners.map(p=>p[1]));
    const spacing=scale<.2?800:200;for(let x=x0;x<x1;x+=spacing)this.line([this.project(x,y0),this.project(x,y1)],"#1b2930");for(let y=y0;y<y1;y+=spacing)this.line([this.project(x0,y),this.project(x1,y)],"#1b2930");
    const visible=this.layers.filter(l=>{const p=this.project(l.origin[0],l.origin[1]+200);return p.x>-1600*scale&&p.x<w+1600*scale&&p.y>-1800*scale&&p.y<h+1800*scale;});
    for(const route of this.routes??[])this.drawRoute(route);
    for(const l of visible)this.drawLayer(l);
    this.onDraw?.();
  }
  drawRoute(route){
    const special=["kv-share","ple"].includes(route.kind);
    if(special&&route.to.area!==this.routeFocus&&route.from.area!==this.routeFocus)return;
    const from=this.layers.find(l=>l.id===route.from.area),to=this.layers.find(l=>l.id===route.to.area);
    const a=from?.nodes.find(n=>n.id===route.from.node),b=to?.nodes.find(n=>n.id===route.to.node);if(!a||!b)return;
    const s=this.camera.scale,p=this.project(from.origin[0]+a.position[0],from.origin[1]+a.position[1]),q=this.project(to.origin[0]+b.position[0],to.origin[1]+b.position[1]);
    let points;
    if(route.kind==="feedback"){
      const bounds=worldBounds(this.layers),bottom=bounds[3]+350,left=bounds[0]-350;
      points=[p,this.project(from.origin[0]+a.position[0],bottom),this.project(left,bottom),this.project(left,to.origin[1]+b.position[1]),q];
    }else points=[p,{x:p.x,y:p.y+180*s},{x:q.x,y:p.y+180*s},q];
    this.line(points,special?"#829ba7":"#506773",Math.max(1,(special?3:5)*s),route.kind!=="handoff");
    if(s>.17){const c=this.ctx,r=points[1];c.font=`${Math.max(11,Math.min(16,16*s))}px system-ui`;c.fillStyle="#b6cbd4";c.textAlign="left";c.fillText(route.label,r.x+8,r.y-8);}
    const before=points.at(-2),angle=Math.atan2(q.y-before.y,q.x-before.x),c=this.ctx;c.save();c.translate(q.x,q.y);c.rotate(angle);c.fillStyle="#b0c9d4";c.beginPath();c.moveTo(0,0);c.lineTo(-8,-4);c.lineTo(-8,4);c.closePath();c.fill();c.restore();
  }
  drawLayer(l){
    const c=this.ctx,s=this.camera.scale,ox=l.origin[0],oy=l.origin[1],project=(x,y,z=0)=>this.project(ox+x,oy+y,z);
    const station=l.kind==="station",title=project(Math.min(...l.nodes.map(n=>n.position[0]-n.size[0]/2)),Math.min(...l.nodes.map(n=>n.position[1]-n.size[1]/2))-150);c.textAlign="left";c.fillStyle="#e4edf1";c.font=`600 ${Math.max(12,Math.min(26,26*s))}px system-ui`;c.fillText(station?l.title:"L"+String(l.id).padStart(2,"0"),title.x,title.y);
    if(s>.24){c.font=`${Math.max(11,Math.min(17,17*s))}px system-ui`;c.fillStyle="#a6bdc7";const x=station?title.x:title.x+Math.max(48,65*s),y=station?title.y+Math.max(20,30*s):title.y;if(!station)c.fillText(l.name??({S:"SLIDING ATTENTION",F:"FULL ATTENTION",D:"DELTANET"}[l.kind]??l.kind),x,y);c.font=`${Math.max(11,Math.min(17,17*s))}px ui-monospace,monospace`;c.fillText(l.timing?formatTime(l.timing.end-l.timing.start):"Structure · not timed by this tape",x,y+(station?0:Math.max(15,24*s)));}
    const byId=new Map(l.nodes.map(n=>[n.id,n]));
    for(const e of l.edges){const a=byId.get(e.from),b=byId.get(e.to);if(!a||!b)continue;const p=project(...a.position),q=project(...b.position),selected=this.selected?.layer===l.id&&(this.selected.id===a.id||this.selected.id===b.id);const bend=e.payload==="residual"?120*s:0;
      this.line([p,{x:(p.x+q.x)/2,y:p.y+bend},{x:(p.x+q.x)/2,y:q.y+bend},q],selected?"#d5e3e8":"#506773",selected?2:Math.max(1,2*s),e.payload?.includes("weight"));
      if(s>.28){const x=(p.x+q.x)/2,y=q.y+bend;c.fillStyle=selected?"#e6eff3":"#758e99";c.beginPath();c.moveTo(x+5,y);c.lineTo(x-3,y-3);c.lineTo(x-3,y+3);c.fill();}
    }
    const ordered=[...l.nodes].sort((a,b)=>(a.shape==="housing"?-1:0)-(b.shape==="housing"?-1:0)||a.position[1]-b.position[1]);
    for(const n of ordered){
      if(n.parentPart&&s<.32)continue;
      const [nx,ny,nz]=n.position,x=ox+nx,y=oy+ny,z=(nz??0)*this.explode,[nw,nh,nd]=n.size,active=worldNodeState(n,this.time),selected=this.selected?.layer===l.id&&this.selected.id===n.id;
      const pigment=active.kind==="wait"?"#bf8134":"#259f9d",fill=active.active?pigment:n.shape==="storage"?"#34434a":"#344d58",stroke=selected?"#f4f7f9":active.active?"#b9e6df":"#627e89";
      const points=this.box(x,y,z,nw,nh,nd,n.shape==="housing"?"#1e3039":fill,stroke);
      const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x)),minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));
      this.hits.push({layer:l.id,node:n,x:minX,y:minY,w:maxX-minX,h:maxY-minY});
      if(selected)this.polygon(points.slice(0,4),null,"#f4f7f9",2.5);
      if(n.shape==="housing"){
        if(s>.25){const p=project(nx-nw/2+12,ny-nh/2+20,z+nd);c.fillStyle=active.active?"#a2dcd6":"#afc5cf";c.textAlign="left";c.font=`600 ${Math.max(11,Math.min(20,19*s))}px system-ui`;c.fillText(n.name,p.x,p.y);}
        continue;
      }
      if(s<.26)continue;
      const p=project(nx,ny,z+nd),font=Math.max(this.w>1600?15:12,Math.min(22,18*s)),width=Math.max(70,nw*s*.86);
      c.textAlign="center";c.fillStyle="#f1f5f7";c.font=`500 ${font}px system-ui`;
      const words=n.name.split(" "),lines=[];let line="";for(const word of words){if(line&&c.measureText(line+" "+word).width>width){lines.push(line);line=word;}else line+=(line?" ":"")+word;}if(line)lines.push(line);
      const displayed=lines.slice(0,2);for(let i=0;i<displayed.length;i++)c.fillText(displayed[i],p.x,p.y-8+(i-(displayed.length-1)/2)*(font+1));
      const e=n.evidence,label=e.kind==="shared"?"shared":e.duration===null?"untimed":formatTime(e.duration);c.font=`${Math.max(this.w>1600?13:11,Math.min(17,14*s))}px ui-monospace,monospace`;c.fillStyle=active.active?"#f5f9f6":"#b1c6ce";c.fillText(label,p.x,p.y+font+4);
      if(n.shape==="storage"){for(let j=0;j<3;j++)this.line([project(nx-nw/2,ny+nh/2,z+j*nd/3),project(nx+nw/2,ny+nh/2,z+j*nd/3)],"#829ba5");}
      else if(n.shape==="portal"){const a=project(nx-nw/2,ny-nh/2,z+nd),b=project(nx+nw/2,ny+nh/2,z+nd);c.strokeStyle="#c3d9e0";c.lineWidth=2;c.setLineDash([5,4]);c.strokeRect(Math.min(a.x,b.x)+4,Math.min(a.y,b.y)+4,Math.abs(b.x-a.x)-8,Math.abs(b.y-a.y)-8);c.setLineDash([]);}
      else if(n.shape==="gate"){for(const dx of [-nw/2+8,nw/2-8])this.line([project(nx+dx,ny-nh/2,z+nd),project(nx+dx,ny+nh/2,z+nd)],"#d0c5a7",3);}
      else if(s>.6){const q=project(nx-nw/2+12,ny-nh/2+7,z+nd+1);c.fillStyle=active.active?"#e8f8f2":"#789ca8";
        if(/product|projection|weight|dot/i.test(n.name)){for(let a=0;a<4;a++)for(let b=0;b<2;b++)c.fillRect(q.x+a*4,q.y+b*4,2.5,2.5);}
        else if(/norm|gather|fold|sum/i.test(n.name)){for(let a=0;a<3;a++)this.line([{x:q.x,y:q.y+a*4},{x:q.x+12,y:q.y+4}],"#9cbbc3");}
        else {this.line([{x:q.x,y:q.y+6},{x:q.x+5,y:q.y+6},{x:q.x+9,y:q.y},{x:q.x+15,y:q.y}],"#9cbbc3");}}
      if(active.active){const bw=Math.max(24,nw*s*.65);c.fillStyle="#10282e";c.fillRect(p.x-bw/2,p.y+font+10,bw,3);c.fillStyle="#e1f4ec";c.fillRect(p.x-bw/2,p.y+font+10,bw*active.progress,3);}
    }
  }
  dispose(){this.resize.disconnect();this.abort.abort();}
}
