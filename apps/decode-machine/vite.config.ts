import {defineConfig,type Plugin} from 'vite';
import {resolve} from 'node:path';
import {weightApi} from './tools/weight-api.ts';

const base=process.env.VITE_BASE??'/';
const mount=base==='/'?'':base.replace(/\/+$/,'');

function learningRoute():Plugin{
 const rewrite=(req:{url?:string},_res:unknown,next:()=>void)=>{
  if(req.url){
   const q=req.url.indexOf('?'),path=q<0?req.url:req.url.slice(0,q),search=q<0?'':req.url.slice(q);
   if(path===`${mount}/learning`||path===`${mount}/learning/`||path==='/learning'||path==='/learning/')req.url=`${mount}/teaching.html${search}`;
  }
  next();
 };
 return {
  name:'decode-machine-routes',
  configureServer(server){server.middlewares.use(rewrite);},
  configurePreviewServer(server){server.middlewares.use(rewrite);},
  transformIndexHtml(html){
   if(!mount)return html;
   return html.replaceAll('href="/"',`href="${mount}"`).replaceAll('href="/teaching.html"',`href="${mount}/learning"`);
  },
 };
}

export default defineConfig({base,plugins:[weightApi(),learningRoute()],build:{rollupOptions:{input:{replay:resolve(import.meta.dirname,'index.html'),teaching:resolve(import.meta.dirname,'teaching.html')}}}});
