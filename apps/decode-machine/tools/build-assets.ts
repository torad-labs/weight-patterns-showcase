/** Mechanical geometry is authored here; Blender translates the recipe to editable assets. */
import {mkdir,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
type V=[number,number,number];
type Part={name:string;group:string;kind:string;at:V;scale?:V;rotation?:V;material?:string;parent?:string};
const parts:Part[]=[];let group='layer';const half=Math.PI/2;
function part(name:string,kind:string,at:V,scale:V,material:string,rotation:V=[0,0,0],parent?:string){parts.push({name,group,kind,at,scale,material,rotation,parent});}
function box(name:string,at:V,scale:V,material='steel',parent?:string){part(name,'box',at,scale,material,[0,0,0],parent);}
function cyl(name:string,at:V,r:number,depth:number,material='steel',front=false,parent?:string){part(name,'cylinder',at,[r,r,depth],material,front?[half,0,0]:[0,0,0],parent);}
function sphere(name:string,at:V,r:number,material='amber'){part(name,'sphere',at,[r,r,r],material);}
function empty(name:string,at:V){parts.push({name,group,kind:'empty',at});}
// A reusable cutaway with three parallel roller channels. No decorative screws or engraving.
box('layer_foot',[0,0,.14],[2.1,2.65,.18],'base');
box('layer_left_rail',[-.82,0,.34],[.13,2.55,.28]);
box('layer_right_rail',[.82,0,.34],[.13,2.55,.28]);
for(const [i,y]of [-.82,0,.82].entries()){
 empty(`roller_${i}`,[0,y,.67]);
 cyl(`roller_body_${i}`,[0,0,0],.29,.5,'brass',true,`roller_${i}`);
 box(`roller_mark_${i}`,[0,-.255,0],[.46,.018,.07],'ink',`roller_${i}`);
 cyl(`roller_axle_${i}`,[0,y,.67],.065,.7,'ink',true);
 box(`channel_${i}`,[0,y,.41],[1.7,.55,.1],'ivory');
}
empty('layer_cover',[0,0,1.04]);
box('cover_back',[0,1.24,0],[1.9,.12,.12],'steel','layer_cover');
box('cover_front',[0,-1.24,0],[1.9,.12,.12],'steel','layer_cover');
for(const x of [-.88,.88])box(`cover_side_${x}`,[x,0,0],[.12,2.5,.12],'steel','layer_cover');
// A sizing jaw normalizes the carrier, followed by an expanding/compressing press.
box('normalizer_floor',[-1.35,0,.3],[.55,.9,.15],'base');
for(const y of [-.4,.4])box(`normalizer_post_${y}`,[-1.35,y,.72],[.1,.1,.85],'steel');
empty('normalizer_jaw',[-1.35,0,1.1]);box('sizing_jaw',[0,0,0],[.5,.85,.1],'ink','normalizer_jaw');
box('ffn_floor',[1.62,0,.3],[.95,1.3,.15],'base');
for(const y of [-.6,.6])box(`ffn_post_${y}`,[1.62,y,.8],[.12,.12,1.1],'steel');
empty('ffn_press',[1.62,0,1.2]);box('ffn_platen',[0,0,0],[.9,1.25,.12],'steel','ffn_press');
for(const x of [1.38,1.62,1.86])cyl(`ffn_spring_${x}`,[x,0,.65],.07,.5,'ink');
// A local weight drawer belongs to this layer; runtime assigns its sample values.
empty('weight_drawer',[0,-1.65,.35]);
box('weight_drawer_bed',[0,0,0],[1.85,.72,.12],'ink','weight_drawer');
box('weight_drawer_front',[0,-.37,.08],[1.85,.08,.24],'steel','weight_drawer');
box('weight_drawer_handle',[0,-.46,.08],[.48,.12,.08],'brass','weight_drawer');
for(let k=0;k<12;k++)part(`stored_weight_${k}`,'sphere',[-.63+(k%4)*.42,-.23+Math.floor(k/4)*.23,.17],[.095,.095,.095],'amber',[0,0,0],'weight_drawer');
// An open, transparent weight hopper: individual balls instead of a liquid volume.
group='supply';
cyl('weight_hopper_floor',[0,0,.3],.8,.2,'base');
cyl('weight_hopper_wall',[0,0,1.0],.78,1.35,'glass');
for(const z of [.4,1.65])part(`hopper_band_${z}`,'torus',[0,0,z],[.78,.78,.78],'steel');
for(let level=0;level<4;level++)for(let i=0;i<10;i++){
 const a=i/10*Math.PI*2+level*.24;
 sphere(`weight_${level}_${i}`,[Math.cos(a)*.51,Math.sin(a)*.51,.57+level*.25],.125);
}
for(let level=0;level<4;level++)for(let i=0;i<3;i++){
 const a=i/3*Math.PI*2;sphere(`weight_core_${level}_${i}`,[Math.cos(a)*.2,Math.sin(a)*.2,.57+level*.25],.125);
}
box('hopper_spout',[.9,0,.65],[.4,.36,.22],'steel');
// Open receiving tray and input dock share the same understated geometry.
group='output';
box('output_tray',[0,0,.3],[1.7,1.6,.15],'base');
for(const y of [-.75,.75])box(`output_edge_${y}`,[0,y,.48],[1.7,.1,.35],'steel');
box('output_stop',[-.8,0,.48],[.1,1.6,.35],'steel');
group='intake';
box('intake_bed',[0,0,.3],[3.4,3.2,.2],'base');
for(const y of [-1.45,1.45])box(`cutter_post_${y}`,[-.9,y,.9],[.15,.15,1.3],'steel');
empty('cutter_blade',[-.9,0,1.4]);box('blade',[0,0,0],[.1,2.9,.28],'ink','cutter_blade');
box('paper_roll',[-1.5,0,.55],[.5,2.8,.12],'ivory');
box('embedding_drawer',[.8,0,.55],[1,2.8,.35],'steel');
for(let i=0;i<7;i++)box(`drawer_slot_${i}`,[.8,-1.17+i*.39,.76],[.8,.25,.04],'ink');
group='selector';
box('selector_bed',[0,0,.3],[2.1,3.8,.2],'base');
for(let i=0;i<4;i++){box(`score_socket_${i}`,[0,-1.35+i*.9,.45],[.6,.65,.15],'ink');}
empty('selector_arm',[0,0,1.8]);box('selector_pointer',[0,0,0],[1.8,.1,.12],'steel','selector_arm');
box('printer_tray',[-2.4,0,.35],[1.5,2.3,.15],'base');
box('answer_paper',[-2.4,0,.46],[1.35,2.1,.04],'ivory');
empty('stop_gate',[-3.3,0,1.2]);box('stop_bar',[0,0,0],[.12,1.3,.12],'ink','stop_gate');
const materials={
 steel:{color:[.49,.58,.61,1],metal:0,rough:1},brass:{color:[.73,.50,.22,1],metal:0,rough:1},
 ink:{color:[.12,.19,.20,1],metal:0,rough:1},ivory:{color:[.85,.85,.78,1],metal:0,rough:1},
 base:{color:[.66,.72,.70,1],metal:0,rough:1},glass:{color:[.7,.8,.8,.13],metal:0,rough:1},
 amber:{color:[.88,.53,.19,1],metal:0,rough:1}
};
const root=resolve(import.meta.dirname,'..');
await mkdir(`${root}/public/assets`,{recursive:true});await mkdir(`${root}/assets`,{recursive:true});
await writeFile(`${root}/assets/recipe.json`,JSON.stringify({parts,materials,root}));
await writeFile(`${root}/public/assets/assembly.json`,JSON.stringify({groups:['layer','supply','output','intake','selector'],layerCount:32,headsPerLayer:3,timing:'scripted illustrative story',question:'What’s the capital of France?',answer:'Paris.',weightEncoding:'12 illustrative stored weights per layer; read copies travel to operations; stored values remain unchanged'}));
const bridge=String.raw`
import bpy,json,sys,os
data=json.load(open(sys.argv[sys.argv.index('--')+1]))
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
mats={}
for name,m in data['materials'].items():
 mat=bpy.data.materials.new(name);mat.use_nodes=True;p=mat.node_tree.nodes.get('Principled BSDF')
 p.inputs['Base Color'].default_value=m['color'];p.inputs['Metallic'].default_value=m['metal'];p.inputs['Roughness'].default_value=m['rough'];p.inputs['Alpha'].default_value=m['color'][3]
 if m['color'][3]<1:mat.surface_render_method='DITHERED'
 mats[name]=mat
objects={};groups={};cache={}
for p in data['parts']:
 group=p['group']
 if group not in groups:
  col=bpy.data.collections.new(group);bpy.context.scene.collection.children.link(col);groups[group]=col
 kind=p['kind'];name=p['name'];key=(kind,p.get('material'))
 if kind=='empty':obj=bpy.data.objects.new(name,None)
 elif key in cache:obj=bpy.data.objects.new(name,cache[key])
 else:
  if kind=='box':bpy.ops.mesh.primitive_cube_add(size=1)
  elif kind=='cylinder':bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=1,depth=1)
  elif kind=='sphere':bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=1)
  elif kind=='torus':bpy.ops.mesh.primitive_torus_add(major_segments=48,minor_segments=8,major_radius=1,minor_radius=.045)
  obj=bpy.context.object;obj.name=name
  for c in list(obj.users_collection):c.objects.unlink(obj)
  if kind!='box':
   for poly in obj.data.polygons:poly.use_smooth=True
  cache[key]=obj.data
 groups[group].objects.link(obj)
 if p.get('material') and hasattr(obj.data,'materials') and len(obj.data.materials)==0:obj.data.materials.append(mats[p['material']])
 obj.location=p['at'];obj.scale=p.get('scale',[1,1,1]);obj.rotation_euler=p.get('rotation',[0,0,0])
 if kind=='box':
  mod=obj.modifiers.new('Edge softness','BEVEL');mod.width=.04;mod.segments=1;obj.modifiers.new('Corner normals','WEIGHTED_NORMAL')
 if p.get('parent'):obj.parent=objects[p['parent']]
 objects[name]=obj
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(data['root'],'assets','layer-kit.blend'),compress=True)
for name,col in groups.items():
 bpy.ops.object.select_all(action='DESELECT')
 for obj in col.objects:obj.select_set(True)
 bpy.ops.export_scene.gltf(filepath=os.path.join(data['root'],'public','assets',name+'.glb'),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_animations=False,export_extras=True)
print('ASSET_BUILD_COMPLETE',len(objects),'parts',len(groups),'assemblies')
`;
const blender=process.env.BLENDER_BIN??'blender';
const result=spawnSync(blender,['--background','--factory-startup','--python-exit-code','1','--python-expr',bridge,'--',`${root}/assets/recipe.json`],{stdio:'inherit'});
if(result.error)throw result.error;process.exit(result.status??1);
