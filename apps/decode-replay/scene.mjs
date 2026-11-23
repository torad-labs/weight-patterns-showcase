// A small, dependency-free 3D orthographic renderer. Positions encode composition only.
export class AssemblyScene {
  constructor(canvas, onSelect) {
    this.canvas=canvas; this.ctx=canvas.getContext("2d"); this.onSelect=onSelect;
    this.yaw=-.25; this.pitch=.4; this.zoom=1; this.explode=.5; this.nodes=[]; this.edges=[]; this.selected=null; this.active=new Set(); this.hits=[];
    this.observer=new ResizeObserver(()=>this.draw()); this.observer.observe(canvas);
    let drag=null;
    canvas.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY};canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener("pointermove",e=>{if(!drag)return;this.yaw+=(e.clientX-drag.x)*.006;this.pitch=Math.max(-1.1,Math.min(1.1,this.pitch+(e.clientY-drag.y)*.006));drag.x=e.clientX;drag.y=e.clientY;this.draw();});
    canvas.addEventListener("pointerup",e=>{if(drag&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<5){const r=canvas.getBoundingClientRect();const p={x:e.clientX-r.left,y:e.clientY-r.top};const hit=this.hits.findLast(h=>p.x>=h.x&&p.x<=h.x+h.w&&p.y>=h.y&&p.y<=h.y+h.h);if(hit)this.onSelect(hit.node);}drag=null;});
    canvas.addEventListener("pointercancel",()=>{drag=null;});
    canvas.addEventListener("wheel",e=>{e.preventDefault();this.zoom=Math.max(.45,Math.min(3,this.zoom*Math.exp(-e.deltaY*.001)));this.draw();},{passive:false});
    canvas.addEventListener("keydown",e=>{if(e.key.startsWith("Arrow")){e.preventDefault();if(e.key==="ArrowLeft")this.yaw-=.1;if(e.key==="ArrowRight")this.yaw+=.1;if(e.key==="ArrowUp")this.pitch=Math.max(-1.1,this.pitch-.1);if(e.key==="ArrowDown")this.pitch=Math.min(1.1,this.pitch+.1);this.draw();}});
  }
  reset(){this.yaw=-.25;this.pitch=.4;this.zoom=1;this.draw();}
  setData(nodes,edges){this.nodes=nodes;this.edges=edges;this.draw();}
  draw(){
    const rect=this.canvas.getBoundingClientRect(), w=rect.width,h=rect.height;
    if(!w||!h)return;
    const ratio=Math.min(window.devicePixelRatio||1,2);this.canvas.width=w*ratio;this.canvas.height=h*ratio;
    const c=this.ctx;c.setTransform(ratio,0,0,ratio,0,0);c.clearRect(0,0,w,h);this.hits=[];
    if(!this.nodes.length){c.fillStyle="#596466";c.font="16px Georgia";c.textAlign="center";c.fillText("No composed source in this bundle",w/2,h/2);return;}
    const nodes=this.mechanism&&w<700?this.nodes.filter(n=>!n.parentPart).map((n,i)=>({...n,size:[200,84,28],shape:n.shape==="housing"?"operation":n.shape,position:[(i%2)*250,Math.floor(i/2)*145,(i%2)*50]})):this.nodes;
    const {items,scale,project}=fitAssembly(nodes,w,h,{yaw:this.yaw,pitch:this.pitch,explode:this.explode,zoom:this.zoom});
    const byId=new Map(items.map(n=>[n.node.id,n]));
    // Ordered connections are drawn only when supplied by the composition; a grid is not an execution order.
    c.lineWidth=1.2;c.strokeStyle="#a1acae";
    for(const edge of this.edges){
      const a=byId.get(edge.from),b=byId.get(edge.to);if(!a||!b)continue;
      const selected=edge.from===this.selected||edge.to===this.selected;
      c.strokeStyle=selected?"#a7442c":"#9ba6a4";c.lineWidth=selected?2.5:1.3;
      const bend=this.mechanism&&edge.payload==="residual"?90*scale:0,mid={x:(a.p.x+b.p.x)/2,y:(a.p.y+b.p.y)/2+bend};
      c.beginPath();c.moveTo(a.p.x,a.p.y);c.quadraticCurveTo(mid.x,mid.y,b.p.x,b.p.y);c.stroke();
      if(this.mechanism){
        const t=.62,x=(1-t)*(1-t)*a.p.x+2*(1-t)*t*mid.x+t*t*b.p.x,y=(1-t)*(1-t)*a.p.y+2*(1-t)*t*mid.y+t*t*b.p.y;
        const angle=Math.atan2(b.p.y-a.p.y,b.p.x-a.p.x);c.fillStyle=c.strokeStyle;c.beginPath();c.moveTo(x+6*Math.cos(angle),y+6*Math.sin(angle));c.lineTo(x+5*Math.cos(angle+2.5),y+5*Math.sin(angle+2.5));c.lineTo(x+5*Math.cos(angle-2.5),y+5*Math.sin(angle-2.5));c.closePath();c.fill();
        if(selected&&edge.payload){c.font="12px system-ui";c.textAlign="center";const tw=c.measureText(edge.payload).width;c.fillStyle="#f5f1e8";c.fillRect(mid.x-tw/2-5,mid.y-20,tw+10,18);c.fillStyle="#252d31";c.fillText(edge.payload,mid.x,mid.y-7);}
      }
    }
    items.sort((a,b)=>a.p.z-b.p.z);
    const polygon=(points,fill,stroke)=>{c.beginPath();points.forEach((p,i)=>i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y));c.closePath();c.fillStyle=fill;c.fill();c.strokeStyle=stroke;c.stroke();};
    for(const item of items){
      const points=item.corners;
      const selected=item.node.id===this.selected, active=this.active.has(item.node.id);
      const fill=active?"#286b9c":selected?"#f0ddd0":item.node.shape==="storage"?"#e2e5d9":"#f7f4ec",stroke=selected?"#a7442c":"#929f9f";
      if(this.mechanism&&item.node.shape==="housing"){
        polygon(points.slice(0,4),"#e0dfd3",selected?"#a7442c":"#a4aba5");
        const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x)),minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));
        this.hits.push({node:item.node,x:minX,y:minY,w:maxX-minX,h:maxY-minY});
        c.textAlign="left";c.fillStyle="#252d31";c.font=`${w>2000?18:14}px system-ui`;c.fillText(item.node.label,minX+12,minY+24);
        c.font=`${w>2000?15:12}px ui-monospace,monospace`;c.fillText(item.node.subtitle??"",minX+12,minY+43);
        continue;
      }
      polygon([points[1],points[5],points[6],points[2]],active?"#1b5076":"#b9c4c5",stroke);
      polygon([points[2],points[6],points[7],points[3]],active?"#164563":"#d3d9d5",stroke);
      polygon(points.slice(0,4),fill,stroke);
      if(this.mechanism&&item.node.shape==="storage"){
        c.strokeStyle=stroke;for(const f of [.22,.45,.68]){c.beginPath();c.moveTo(points[1].x+(points[5].x-points[1].x)*f,points[1].y+(points[5].y-points[1].y)*f);c.lineTo(points[2].x+(points[6].x-points[2].x)*f,points[2].y+(points[6].y-points[2].y)*f);c.stroke();}
      }
      const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x)),minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));
      this.hits.push({node:item.node,x:minX,y:minY,w:maxX-minX,h:maxY-minY});
      c.textAlign="center";c.fillStyle=active?"#fff":"#252d31";const font=Math.max(this.mechanism&&w>2000?16:12,Math.min(21,14*scale));c.font=`${font}px system-ui`;
      let label=String(item.node.label??item.node.name??item.node.id);
      while(c.measureText(label).width>Math.max(60,((item.node.size?.[0]??140)-24)*scale)&&label.length>4)label=label.slice(0,-2);
      if(label!==String(item.node.label??item.node.name??item.node.id))label+="…";
      c.fillText(label,item.p.x,item.p.y-2);
      c.font=`${Math.max(12,Math.min(16,12*scale))}px ui-monospace,monospace`;c.fillStyle=active?"#e8f2f6":"#596466";c.fillText(item.node.subtitle??(item.node.group?`${item.node.childCount??0} invocations`:item.node.event?item.node.kind:`L${item.node.level??"?"}`),item.p.x,item.p.y+Math.max(16,15*scale));
      if(item.node.group){const maxCount=Math.max(1,...this.nodes.map(n=>n.childCount??0));c.fillStyle="#286b9c";c.fillRect(item.p.x-48*scale,item.p.y+23*scale,96*scale*(item.node.childCount??0)/maxCount,Math.max(2,3*scale));}
    }
  }
}
