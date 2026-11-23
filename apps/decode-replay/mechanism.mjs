// Explanatory anatomy over recorded phase spans. This module never evaluates tensors.
// Data-flow edges explain operands, not instruction scheduling or measured transfers.
import {stepMetrics as metricsForMechanism} from "./internal-model.mjs";
const part=(id,name,position,more={})=>({id,name,label:name,position,shape:"operation",inputs:[],outputs:[],...more});
const wire=(from,to,payload)=>({from,to,payload,kind:"operand"});
const shared=(scope,id,name,position,inputs,outputs,description,equation)=>part(id,name,position,{scopeId:scope,inputs,outputs,description,equation});
export function mechanismEvidence(replay,node){
  const step=replay.byId.get(node.stepId??node.scopeId);
  if(!step)return {kind:"untimed",duration:null,span:null,step:null};
  const m=metricsForMechanism(step);
  if(node.stepId)return {kind:"measured",duration:m.blocks?m.critical:null,span:m.critical,step,metrics:m};
  return {kind:"shared",duration:null,span:m.blocks?m.critical:null,step,metrics:m};
}
export function mechanismGraph(replay,layer,focus=null){
  const id=role=>`l${layer}:${role}`,deltanet=replay.layers.find(l=>l.id===layer)?.kind==="D";
  const measured=(stepId,position)=>{const s=replay.byId.get(stepId);return part(stepId,s?.name??stepId,position,{stepId,description:s?.description??"This tape does not expose this step.",expandable:!!s,inputs:[s?.fromTap??"input"],outputs:[s?.toTap??"output"],kind:s?.kind??"work"});};
  if(!focus){
    const nodes=[
      part("input","Residual stream",[-700,0,0],{shape:"storage",inputs:["previous layer"],outputs:["x"],description:"The layer's incoming activation. The residual bypass carries it around each sublayer."}),
      part("mixer",deltanet?"DeltaNet":"Q / K / V",[-460,0,0],{stepId:id("mixer:0"),expandable:true,inputs:["x","projection weights"],outputs:deltanet?["recurrent output"]:["Q","K","V"],description:deltanet?"The recurrent mixer opens into its measured phase chain.":"Project the activation into queries, keys and values. Q normalization and RoPE sit inside the measured outer span."}),
      part(deltanet?"recurrent-state":"kv-cache",deltanet?"Recurrent state":"KV cache",[-460,145,110],{shape:"storage",expandable:true,inputs:deltanet?["state update"]:["K","V"],outputs:deltanet?["previous state"]:["cached K","cached V"],description:deltanet?"Persistent recurrence state, not an attention KV cache.":"Keys and values from the attention history. This is logical storage, not a measured memory address or chip location."}),
      ...(!deltanet?[part("attention","Attention",[-210,0,0],{stepId:id("mixer:2"),expandable:true,inputs:["Q","cached K","cached V"],outputs:["mixed values"],description:"Score query against keys, weight the values and combine channel/tile results."})]:[]),
      part("output-projection","Output projection",[40,0,0],{stepId:id("main:0"),expandable:true,inputs:["mixer output","output weights"],outputs:["output partials"],description:"Project the mixer's result back into the residual channel space."}),
      part("post-attention","Fold + residual",[280,0,0],{stepId:id("main:2"),expandable:true,inputs:["output partials","residual"],outputs:["updated residual"],description:"Fold projection partials and update the residual, with normalization/staging in the fused consumer."}),
      part("ffn","Gate / up",[520,0,0],{stepId:id("main:8"),expandable:true,inputs:["updated residual","gate / up weights"],outputs:["FFN activation"],description:"Normalize/stage the input, compute gate and up branches, activate and publish the intermediate."}),
      part("down","Down projection",[760,0,0],{stepId:id("main:10"),expandable:true,inputs:["FFN activation","down weights"],outputs:["down partials"],description:"Project the expanded FFN activation back into hidden channels."}),
      part("residual-out","Fold + residual",[1000,0,0],{stepId:id("main:12"),expandable:true,inputs:["down partials","residual"],outputs:["next layer"],description:"Fold down partials, update the residual and stage the next layer."}),
      part("weights","Packed weights",[280,-160,-100],{shape:"storage",expandable:true,inputs:["checkpoint"],outputs:["FP4 codes + block scales"],description:"Weight operands and their block scales. This diagram does not replay the contents of the weight memory."}),
    ];
    const edges=[wire("input","mixer","x"),wire("input","post-attention","residual"),wire("output-projection","post-attention","output partials"),wire("post-attention","ffn","normalized input"),wire("post-attention","residual-out","residual"),wire("ffn","down","activated intermediate"),wire("down","residual-out","down partials"),wire("weights","mixer","projection weights"),wire("weights","output-projection","output weights"),wire("weights","ffn","gate / up weights"),wire("weights","down","down weights")];
    if(deltanet)edges.push(wire("mixer","recurrent-state","state update"),wire("recurrent-state","mixer","previous state"),wire("mixer","output-projection","recurrent output"));
    else edges.push(wire("mixer","kv-cache","K / V"),wire("mixer","attention","Q"),wire("kv-cache","attention","cached K / V"),wire("attention","output-projection","mixed values"));
    // The default is already a cutaway: measured inner parts sit inside their
    // enclosing routine, rather than requiring another click to reveal any internals.
    const positions={input:[-1050,-160,0],mixer:[-650,-160,0],"kv-cache":[-650,150,100],"recurrent-state":[-650,150,100],attention:[20,-140,0],"output-projection":[650,-160,0],"post-attention":[650,200,0],ffn:[0,480,0],down:[-550,480,0],"residual-out":[-950,480,0],weights:[0,-460,-120]};
    const inner=[];
    for(const n of nodes){
      n.position=positions[n.id]??n.position;
      const children=replay.steps.filter(s=>s.parent===n.stepId);
      if(!children.length)continue;
      n.shape="housing";n.size=[children.length>2?650:450,100+Math.ceil(children.length/3)*110,30];
      for(const [i,s] of children.entries()){
        const columns=Math.min(3,children.length),rows=Math.ceil(children.length/columns);
        const p=measured(s.id,[n.position[0]+(i%columns-(columns-1)/2)*205,n.position[1]+(Math.floor(i/columns)-(rows-1)/2)*105+20,85]);
        p.parentPart=n.id;p.size=[185,72,22];inner.push(p);
      }
      const ids=new Set(children.map(s=>s.id));
      for(const s of children)for(const previous of s.predecessors)if(ids.has(previous))edges.push(wire(previous,s.id,"recorded boundary"));
    }
    nodes.push(...inner);
    return {title:deltanet?"Recurrent layer mechanism":"Attention + feed-forward mechanism",nodes,edges,meaning:"Open housings expose measured substeps. Tensor wires are logical connections, not physical chip wiring. Open a part for the next level."};
  }
  if(["kv-cache","weights","recurrent-state"].includes(focus)){
    const definitions=focus==="kv-cache"?[
      ["key-codes","Key codes","Encoded key vectors read by score products."],["key-scales","Key scales","Block scales travel with key codes."],["value-codes","Value codes","Encoded value vectors read by the probability/value product."],["value-scales","Value scales","Block scales travel with value codes."],["position-map","Position mapping","The recipe maps token positions to cache storage; addresses and capacity are not reconstructed here."]
    ]:focus==="weights"?[
      ["weight-codes","FP4 weight codes","Packed matrix operands."],["weight-scales","Block scales","The block-scale plane accompanies the packed weights."],["tile-stream","Operand tiles","The schedule streams the selected matrix tiles to the tensor-core operand path."]
    ]:[
      ["convolution-history","Convolution history","Recent projected tokens consumed by the causal convolution."],["recurrence","Recurrent state","The accumulated recurrence state, carried in the engine's narrow representation."],["decay","Decay / gates","The per-step decay and gating operands control the recurrent update."]
    ];
    return {title:focus==="kv-cache"?"KV cache":focus==="weights"?"Weight operands":"Recurrent state",nodes:definitions.map(([id,name,description],i)=>part(id,name,[(i%3-1)*260,Math.floor(i/3)*150,Math.floor(i/3)*80],{shape:"storage",description,inputs:["recorded model state"],outputs:["consumer operand"]})),edges:[],meaning:"Storage anatomy only. This tape records no individual storage latency or tensor contents."};
  }
  const role={mixer:"mixer:0",attention:"mixer:2","output-projection":"main:0","post-attention":"main:2",ffn:"main:8",down:"main:10","residual-out":"main:12"}[focus];
  const scope=role?id(role):focus,step=replay.byId.get(scope);
  const children=replay.steps.filter(s=>s.parent===scope);
  if(children.length){
    const nodes=children.map((s,i)=>measured(s.id,[(i%5-2)*240,Math.floor(i/5)*145,Math.floor(i/5)*55]));
    const ids=new Set(nodes.map(n=>n.id));const edges=children.flatMap(s=>s.predecessors.filter(p=>ids.has(p)).map(p=>wire(p,s.id,replay.byId.get(p)?.toTap.split("_")[0]??"published output")));
    return {title:step?.name??focus,nodes,edges,meaning:"Measured substeps. Open a part to see the operations within its span."};
  }
  const make=(i,name,p,ins,outs,desc,eq)=>shared(scope,i,name,p,ins,outs,desc,eq);
  let nodes=[],edges=[];
  if(/attention-pass\d+:2$/.test(scope)){
    nodes=[
      make("maximum","Maximum",[-390,0,0],["score tile"],["maximum"],"Find the tile maximum before the exponential transform.","m = max(scores)"),
      make("exponential","Exponentials",[-130,0,0],["score tile","maximum"],["unnormalized weights"],"Shift the scores and evaluate their exponentials.","p = exp(scores − m)"),
      make("probability-sum","Probability sum",[130,-95,80],["unnormalized weights"],["normalizer"],"Accumulate the probability mass for normalization.","s = Σ p"),
      make("weighted-values","Value product",[130,95,-80],["unnormalized weights","V"],["weighted sum"],"Multiply the probability weights by the value vectors.","u = p · V"),
      make("normalize","Finalize gather",[390,0,0],["weighted sum","normalizer"],["attention partial"],"Gather the probability sum and weighted values for the subsequent combine/finalization. The exact normalization is part of the fused attention pipeline.","carry (u, s, m)"),
    ];
    edges=[wire("maximum","exponential","maximum"),wire("exponential","probability-sum","p"),wire("exponential","weighted-values","p"),wire("probability-sum","normalize","s"),wire("weighted-values","normalize","u")];
  }else if(/gate-up:1$/.test(scope)){
    nodes=[
      make("input-tile","Activation tile",[-360,0,0],["normalized input"],["x"],"The staged narrow activation and its scales are shared by both matrix products.","codes + scales"),
      make("gate-weights","Gate weights",[-120,-155,100],["packed checkpoint"],["Wg"],"Gate weight codes and scales stream into the tensor-core operand path.","FP4 + scales"),
      make("up-weights","Up weights",[-120,155,-100],["packed checkpoint"],["Wu"],"Up weight codes and scales stream independently.","FP4 + scales"),
      make("gate-product","Gate matvec",[130,-65,55],["x","Wg"],["gate"],"Compute the gate branch; no separate gate duration is recorded by GU0/GU1.","gate = Wg · x"),
      make("up-product","Up matvec",[130,65,-55],["x","Wu"],["up"],"Compute the up branch within the same measured leaf span.","up = Wu · x"),
      make("branch-output","Gate + up",[390,0,0],["gate","up"],["activation inputs"],"Pass both branch outputs to the activation/store/publication stage.","two branch outputs"),
    ];
    edges=[wire("input-tile","gate-product","x"),wire("input-tile","up-product","x"),wire("gate-weights","gate-product","Wg"),wire("up-weights","up-product","Wu"),wire("gate-product","branch-output","gate"),wire("up-product","branch-output","up")];
    nodes.filter(n=>n.id.endsWith("weights")).forEach(n=>n.shape="storage");
  }else if(/qtail:1$|gate-up:0$/.test(scope)){
    nodes=[make("gather","Stage / gather",[-390,0,0],["x"],["tile"],"Gather/stage the input tile. Grouped with the normalization when no inner tap exists.","narrow tile"),
      make("sumsq","Sum of squares",[-130,0,0],["tile"],["sumsq"],"Reduce the squared input in the hardware reduction path.","Σ x²"),
      make("inverse","Inverse RMS",[130,0,0],["sumsq"],["normalization scalar"],"Form the mean and reciprocal square root.","rsqrt(mean(x²) + ε)"),
      make("scale","Scale + gamma",[390,0,0],["tile","scalar","gamma"],["normalized tile"],"Apply normalization and the learned scale.","x · inverse_RMS · γ")];
    edges=[wire("gather","sumsq","x"),wire("sumsq","inverse","sumsq"),wire("inverse","scale","inverse RMS"),wire("gather","scale","x")];
  }else if(/qtail:3$/.test(scope)){
    nodes=[make("q-read","Read Q",[-260,0,0],["normalized Q"],["Q pairs"],"Read the normalized query channels.","Q"),
      make("angles","Position angles",[0,-130,90],["position"],["cos / sin"],"Gather the rotary coefficients for the token position.","cos θ, sin θ"),
      make("rotate","Rotate pairs",[0,20,0],["Q pairs","cos / sin"],["rotated Q"],"Rotate pairs of channels before storing Q.","Q′ = rotation(θ) · Q"),
      make("store-q","Store + publish",[260,0,0],["rotated Q"],["Q visible"],"Store the narrow result and complete this tail's synchronization.","codes + scales")];
    edges=[wire("q-read","rotate","Q"),wire("angles","rotate","angles"),wire("rotate","store-q","Q′")];
  }else if(/gate-up:2$/.test(scope)){
    nodes=[make("activation","Gate activation",[-260,0,0],["gate"],["activated gate"],"Apply the recipe's gate activation. The phase alone does not certify a specific historical activation formula.","recipe activation(gate)"),
      make("multiply","Gate × up",[0,0,0],["activated gate","up"],["FFN intermediate"],"Multiply the activated gate by the up branch.","a = activation(gate) ⊙ up"),
      make("publish","Store + publish",[260,0,0],["FFN intermediate"],["C1 visible"],"Store the activation and publish the C1 dependency.","narrow activation + ready")];
    edges=[wire("activation","multiply","activated gate"),wire("multiply","publish","a")];
  }else if(/main:0$|main:10$|:q:0$/.test(scope)){
    nodes=[make("activation-read","Read activation",[-260,-80,50],["activation plane"],["x tile"],"Read the narrow activation and its block scales.","x"),
      make("weight-read","Stream weights",[-260,80,-50],["weight plane"],["W tile"],"Stream packed weight codes with their scales.","W"),
      make("matrix-product","Tensor-core product",[0,0,0],["W tile","x tile"],["partial result"],"The matrix product executes in the hardware path. This view never recomputes it.","partial = W · x"),
      make("narrow-store","Encode + store",[260,0,0],["partial result"],["published partial"],"Encode/store the narrow result and finish the partition.","narrow partial")];
    nodes[1].shape="storage";edges=[wire("activation-read","matrix-product","x"),wire("weight-read","matrix-product","W"),wire("matrix-product","narrow-store","partial")];
  }else if(/main:12$|main:2$/.test(scope)){
    nodes=[make("fold","Fold partials",[-260,0,0],["projection partials"],["combined result"],"Combine the partitioned projection results.","fold(partials)"),
      make("residual","Residual add",[0,0,0],["combined result","residual stream"],["new residual"],"Update the residual using the narrow accumulation carrier.","x′ = x + output"),
      make("stage-next","Stage + publish",[260,0,0],["new residual"],["next consumer"],"Stage the resulting state for the next consumer. Additional fused work stays within this shared span.","state + ready")];
    edges=[wire("fold","residual","output"),wire("residual","stage-next","x′")];
  }else if(/attention-pass\d+:0$/.test(scope)){
    nodes=[make("query","Query tile",[-260,-80,60],["Q"],["Q tile"],"The query operand for this channel pass.","Q"),
      make("keys","Key history",[-260,80,-60],["KV cache"],["K tile"],"Read the attended key history.","K"),
      make("dot","Score product",[0,0,0],["Q tile","K tile"],["scores"],"Compute query/key scores; earlier tiles may be included in the entry interval.","scores = Q · Kᵀ"),
      make("stash","Score stash",[260,0,0],["scores"],["score tile visible"],"Write the score tile before the warp-skew/stash barrier.","score tile")];
    nodes[1].shape="storage";edges=[wire("query","dot","Q"),wire("keys","dot","K"),wire("dot","stash","scores")];
  }else{
    nodes=[make("read","Acquire inputs",[-260,0,0],["upstream state"],["operands"],"Inputs to the selected operation. This conceptual split has no independent timestamps.","inputs"),
      make("operation",step?.name??"Operation",[0,0,0],["operands"],["result"],step?.description??"No more detailed stamp exists in this recording.","recorded routine"),
      make("publish","Result / handoff",[260,0,0],["result"],["next consumer"],"The operation's result and handoff. No independent duration is assigned.","result")];
    edges=[wire("read","operation","operands"),wire("operation","publish","result")];
  }
  return {title:step?.name??focus,nodes,edges,meaning:"Explained operations within ONE shared measured span. No individual duration or transfer time was recorded for these parts."};
}
