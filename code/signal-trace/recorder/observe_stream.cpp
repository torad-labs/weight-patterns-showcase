#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <openssl/sha.h>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using json=nlohmann::ordered_json;
namespace fs=std::filesystem;
static void save(const fs::path &p,const json &j){std::ofstream f(p);f<<j.dump(2,' ',false,json::error_handler_t::replace)<<'\n';if(!f)throw std::runtime_error("write failed: "+p.string());}
static std::string hex(const std::string &s){static const char *h="0123456789abcdef";std::string r;for(unsigned char c:s){r+=h[c>>4];r+=h[c&15];}return r;}
static std::string piece(const llama_vocab *v,llama_token id){char b[256];int n=llama_token_to_piece(v,id,b,sizeof(b),0,true);if(n>=0)return {b,size_t(n)};std::vector<char>a(-n);n=llama_token_to_piece(v,id,a.data(),a.size(),0,true);if(n<0)throw std::runtime_error("token piece");return {a.data(),size_t(n)};}
static std::vector<llama_token> tokenize(const llama_vocab *v,const std::string &text){
 const int n=llama_tokenize(v,text.data(),text.size(),nullptr,0,false,true);
 if(n>=0)throw std::runtime_error("empty or invalid teacher text");std::vector<llama_token> ids(-n);
 if(llama_tokenize(v,text.data(),text.size(),ids.data(),ids.size(),false,true)!=int(ids.size()))throw std::runtime_error("teacher tokenize mismatch");return ids;
}

#include "signal_trace.hpp"

struct observation {
 signal_trace trace;
 bool persist=false;
 std::string scope="full",retention="followup",phase;
 int call=0,position=0,micro=-1,micro_start=0,micro_count=0,prefill_next=0;
 uint64_t bytes=0,events=0;
 std::ofstream tensor_file,record_file;
 json nodes=json::array();
 std::unordered_map<const ggml_tensor*,int> pointers;
 std::unordered_map<std::string,int> structures;
 std::map<std::string,int> counts;
 std::vector<int> trace_input_positions;
 // Declared stream positions: when present, the stream is written only at these decode inputs and every
 // other token runs as ordinary inference (nothing is requested from the scheduler there).
 std::vector<int> stream_positions;
 // Declared stream labels: when present, the stream keeps only the outputs whose label matches this expression.
 // The rest of the pass is requested from nobody and runs as ordinary inference. Observation only.
 bool label_filter=false;std::regex stream_labels;
 bool wanted(const std::string &label)const{return !label_filter||std::regex_match(label,stream_labels);}
 // Patch: at one decode input position, add a vector to one recorded output and let the rest of the pass run.
 // During a patched pass nothing else is observed. A sweep repeats the last pass of a supplied answer once per patch.
 bool patch_on=false,sweeping=false;int patch_position=-1;
 std::unordered_map<std::string,std::vector<float>> patches;std::unordered_map<std::string,int> patched;
 // Stream retention: outputs only, every token, residual path (and optionally
 // projection inputs). No input preimages and no per-operation device
 // synchronization; the scheduler computes up to each requested node and the
 // after-callback reads its finished output. Declared trace positions still use
 // the complete signal_trace capture unchanged.
 std::string stream_scope="residual";
 std::ofstream stream_file,stream_records;
 uint64_t stream_bytes=0,stream_events=0;
 std::map<std::string,int> stream_counts;
 std::unordered_map<std::string,int> stream_decision;
 // Opt-in per case (stop_after_last_label): a compute is ended once the last output the pass asked for is written. The engine's graph is fixed by the
 // microbatch's token count and output count (llm_graph_params::allow_reuse), so the first compute of each such shape runs whole and lists its wanted
 // outputs in order; every later compute of that shape must write the same labels in the same order, and stops after the last of them.
 bool stop_early=false,stopped=false,recording=false;size_t seen=0;uint64_t computes=0,stops=0,after_stop=0;
 std::map<std::pair<int64_t,bool>,std::vector<std::string>> shapes;std::vector<std::string> *reference=nullptr;
 void end_compute(){   // at the start of the next compute and at the end of the case
  if(!reference)return;
  if(recording&&reference->empty())throw std::runtime_error("stop_after_last_label: a compute holds none of the wanted outputs");
  if(!recording&&seen!=reference->size())throw std::runtime_error("a compute wrote fewer wanted outputs than the first of its shape");
 }
 bool streamed(const ggml_tensor *t,std::string &label){
  label=t->name;
  if(t->op==GGML_OP_ADD&&t->src[0]&&t->src[1]){
   // The per-layer-embedding merge is an unnamed ADD of pe_in-N and its contribution.
   static const std::regex pe("^pe_in-([0-9]+)$");std::smatch m;
   const std::string a=t->src[0]->name,b=t->src[1]->name;
   if(std::regex_match(a,m,pe)||std::regex_match(b,m,pe)){label="pe_out-"+m[1].str();return wanted(label);}
  }
  auto it=stream_decision.find(label);
  if(it!=stream_decision.end())return it->second;
  static const std::regex residual("^((inp_scaled|attn_post_norm|attn_out|ffn_post_norm|pe_in|per_layer_embd_out|l_out)-[0-9]+|inp_scaled|result_norm)$");
  static const std::regex projection("^(attn_norm|ffn_norm|kqv_out|ffn_geglu|Qcur_pos|Kcur_pos|Vcur_normed)-[0-9]+$");
  const bool keep=(std::regex_match(label,residual)||(stream_scope=="projection"&&std::regex_match(label,projection)))&&wanted(label);
  stream_decision.emplace(label,keep);return keep;
 }
 int intern(const ggml_tensor *t){
  if(!t)return -1;
  if(auto it=pointers.find(t);it!=pointers.end())return it->second;
  json sources=json::array();for(auto *s:t->src)if(s)sources.push_back(intern(s));else sources.push_back(nullptr);
  json params=json::array();for(auto x:t->op_params)params.push_back(x);
  json node={{"name",t->name},{"operation",ggml_op_name(t->op)},{"type",ggml_type_name(t->type)},
   {"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"strides",{t->nb[0],t->nb[1],t->nb[2],t->nb[3]}},
   {"sources",sources},{"op_params_i32",params},{"view_source",t->view_src?json(intern(t->view_src)):json(nullptr)},{"view_offset",t->view_offs}};
  std::string key=node.dump();auto it=structures.find(key);int id;
  if(it!=structures.end())id=it->second;else{id=nodes.size();node["id"]=id;nodes.push_back(node);structures.emplace(std::move(key),id);}
  pointers[t]=id;return id;
 }
 bool retained(const std::string &name)const{
  static const std::regex connected("^(l_out-13|attn_post_norm-14|ffn_post_norm-14|l_out-14|attn_post_norm-15|ffn_post_norm-15|l_out-15|kqv_out-16|projection_out-16|attn_post_norm-16|ffn_post_norm-16|l_out-16|l_out-24|l_out-34|result_norm)$");
  if(retention=="connected")return std::regex_match(name,connected);
  static const std::regex names("^(inp_scaled|result_norm|l_out-[0-9]+|attn_out-1[2-6]|attn_post_norm-1[2-6]|ffn_post_norm-1[2-6]|pe_in-1[2-6]|per_layer_embd_out-1[2-6]|kqv_out-16|projection_out-16)$");
  return std::regex_match(name,names);
 }
 bool selected(const std::string &name)const{
  static const std::regex full("^(embd|inp_scaled|l_out|attn_norm|Qcur|Qcur_normed|Qcur_pos|Kcur|Kcur_normed|Kcur_pos|Vcur|Vcur_normed|kqv_out|kqv_merged|attn_out|attn_post_norm|ffn_norm|ffn_up|ffn_gate|ffn_geglu|ffn_out|ffn_post_norm|per_layer_embd_out|pe_in|result_norm)(-[0-9]+)?$");
  static const std::regex boundary("^(l_out|result_norm)(-[0-9]+)?$");
  return std::regex_match(name,scope=="boundary"?boundary:full);
 }
};

static void write_stream(observation &s,ggml_tensor *t,const std::string &label){
  if(!s.persist){++s.stream_events;return;}
  json r={{"event",s.stream_events},{"call",s.call},{"phase",s.phase},{"input_position",s.position},
   {"microbatch",s.micro},{"microbatch_start",s.micro_start},{"microbatch_count",s.micro_count},
   {"label",label},{"name",t->name},{"operation",ggml_op_name(t->op)},{"native_type",ggml_type_name(t->type)},
   {"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"strides",{t->nb[0],t->nb[1],t->nb[2],t->nb[3]}}};
  json sources=json::array();for(auto *x:t->src)if(x)sources.push_back(x->name);r["source_names"]=sources;
  if(t->type!=GGML_TYPE_F32||t->ne[3]!=1)throw std::runtime_error("stream tensor outside recorded native format: "+label);
  const bool qkv3=(label.rfind("Qcur_",0)==0||label.rfind("Kcur_",0)==0||label.rfind("Vcur_",0)==0);
  const int axis=qkv3?2:1;const int64_t rows=t->ne[axis];json positions=json::array();
  if(rows==s.micro_count){for(int64_t i=0;i<rows;i++)positions.push_back(s.micro_start+i);r["position_basis"]="materialized microbatch order";}
  else if(rows==1&&s.micro_count>0){positions.push_back(s.micro_start+s.micro_count-1);r["position_basis"]="last requested output row";}
  else r["position_basis"]="unresolved; retain native axes";
  r["token_axis"]=axis;r["positions"]=positions;
  std::vector<unsigned char> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());
  std::vector<float> values;values.reserve(ggml_nelements(t));
  for(int64_t z=0;z<t->ne[2];z++)for(int64_t y=0;y<t->ne[1];y++)for(int64_t x=0;x<t->ne[0];x++){
   float v;std::memcpy(&v,raw.data()+x*t->nb[0]+y*t->nb[1]+z*t->nb[2],sizeof(v));values.push_back(v);
  }
  // Stop mode computes no vocabulary scores, so their finite check is gone; the recorded outputs are checked instead.
  if(s.stop_early)for(float v:values)if(!std::isfinite(v))throw std::runtime_error("nonfinite stream value: "+label);
  r["offset_bytes"]=s.stream_bytes;r["values"]=values.size();
  s.stream_file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));s.stream_bytes+=values.size()*sizeof(float);
  s.stream_records<<r.dump()<<'\n';
  if(!s.stream_file||!s.stream_records)throw std::runtime_error("stream persistence failed");
  ++s.stream_events;++s.stream_counts[label.substr(0,label.find('-'))];
}
static void apply_patch(observation &s,ggml_tensor *t,const std::string &label,const std::vector<float> &add){
 const size_t n=add.size();
 if(t->type!=GGML_TYPE_F32||size_t(ggml_nelements(t))!=n||t->nb[0]!=sizeof(float))throw std::runtime_error("patch target has another layout: "+label);
 std::vector<float> v(n),back(n);ggml_backend_tensor_get(t,v.data(),0,n*sizeof(float));
 for(size_t i=0;i<n;i++)v[i]+=add[i];
 ggml_backend_tensor_set(t,v.data(),0,n*sizeof(float));ggml_backend_tensor_get(t,back.data(),0,n*sizeof(float));
 if(std::memcmp(v.data(),back.data(),n*sizeof(float)))throw std::runtime_error("patch was not written: "+label);
 if(++s.patched[label]!=1)throw std::runtime_error("patch target computed twice in one pass: "+label);
}
static std::vector<float> patch_row(const std::string &path,size_t row,size_t width){
 std::ifstream f(path,std::ios::binary);f.seekg(row*width*sizeof(float));std::vector<float> v(width);
 f.read(reinterpret_cast<char*>(v.data()),width*sizeof(float));if(!f)throw std::runtime_error("patch vector file is short: "+path);return v;
}
static bool callback(ggml_tensor *t,bool ask,void *opaque){
 auto &s=*static_cast<observation*>(opaque);std::string name=t->name;
 if(s.retention=="stream"){
  if(ask && name=="embd" && t->op==GGML_OP_GET_ROWS){
   ++s.micro;s.micro_count=t->ne[1];s.micro_start=s.phase=="prefill"?s.prefill_next:s.position;
   if(s.phase=="prefill")s.prefill_next+=s.micro_count;
   if(s.stop_early){
    s.end_compute();
    // A call's last microbatch carries its one output (llama_batch_get_one, no logits array): prefill_next has reached the call's last position + 1.
    const bool output=s.phase!="prefill"||s.prefill_next==s.position+1;
    const auto [shape,fresh]=s.shapes.try_emplace({s.micro_count,output});
    s.reference=&shape->second;s.recording=fresh;++s.computes;s.seen=0;s.stopped=false;
   }
  }
  else if(s.stop_early&&s.stopped)++s.after_stop;   // the scheduler went on after a stop: another split follows the one that holds the wanted outputs
  if(s.patch_on&&s.phase=="decode"&&s.position==s.patch_position){
   std::string label;s.streamed(t,label);
   const auto target=s.patches.find(label);
   if(target==s.patches.end())return !ask;
   if(!ask)apply_patch(s,t,label,target->second);
   return true;
  }
  if(s.sweeping)return !ask;
  const bool traced=!s.trace_input_positions.empty() && std::binary_search(s.trace_input_positions.begin(),s.trace_input_positions.end(),s.position);
  if(!s.stream_positions.empty() && !traced && !(s.phase=="decode" && std::binary_search(s.stream_positions.begin(),s.stream_positions.end(),s.position)))return !ask;
  std::string label;const bool keep=s.streamed(t,label);
  if(traced){
   const bool r=s.trace.observe(t,ask,s.call,s.phase,s.position);
   if(!ask&&keep)write_stream(s,t,label);
   return r;
  }
  if(!keep)return !ask;
  if(!ask){
   write_stream(s,t,label);
   if(s.stop_early){
    if(!s.reference)throw std::runtime_error("stop_after_last_label: a wanted output came before the first compute's embd");
    if(s.recording)s.reference->push_back(label);
    else{
     if(s.seen>=s.reference->size()||(*s.reference)[s.seen]!=label)throw std::runtime_error("a compute wrote other wanted outputs than the first of its shape: "+label);
     if(++s.seen==s.reference->size()){s.stopped=true;++s.stops;return false;}   // the last wanted output of this compute: the rest of the graph is not run
    }
   }
  }
  return true;
 }
 if(s.retention=="trace"){
  // A conditional calibration can record complete operations at declared input
  // positions without recording the preceding, already saved natural history.
  // This changes observation only. Every supplied token still runs normally.
  if(!s.trace_input_positions.empty() && !std::binary_search(s.trace_input_positions.begin(),s.trace_input_positions.end(),s.position))return false;
  return s.trace.observe(t,ask,s.call,s.phase,s.position);
 }
 // Read the native projection at its own producer boundary, before the buffer
 // can be reused. This adds observation only; no tensor or weight is changed.
 if(t->op==GGML_OP_MUL_MAT&&t->src[0]&&std::string(t->src[0]->name)=="blk.16.attn_output.weight")name="projection_out-16";
 if(ask){
  // The token embedding is the first captured materialized microbatch entry.
  if(name=="embd"){
   s.pointers.clear();++s.micro;
   s.micro_count=t->ne[1];s.micro_start=s.phase=="prefill"?s.prefill_next:s.position;
   if(s.phase=="prefill")s.prefill_next+=s.micro_count;
  }
  if(s.persist)s.intern(t);
  return name=="projection_out-16"||s.selected(name);
 }
 ++s.events;++s.counts[name];
 if(!s.persist)return true;
 json r={{"event",s.events-1},{"call",s.call},{"phase",s.phase},{"input_position",s.position},
  {"microbatch",s.micro},{"microbatch_start",s.micro_start},{"microbatch_count",s.micro_count},
  {"node",s.intern(t)},{"name",name},{"native_type",ggml_type_name(t->type)},
  {"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}};
 if(!s.retained(name)){r["capture"]="not-retained-by-followup-scope";s.record_file<<r.dump()<<'\n';return true;}
 if(t->type!=GGML_TYPE_F32 || t->ne[3]!=1){r["capture"]="unrecorded-native-format";s.record_file<<r.dump()<<'\n';return true;}
 const bool qkv3=(name.rfind("Qcur_",0)==0 || name.rfind("Kcur_",0)==0 || name.rfind("Vcur_",0)==0);
 const int axis=qkv3?2:1;const int64_t rows=t->ne[axis];json positions=json::array();
 if(rows==s.micro_count){for(int64_t i=0;i<rows;i++)positions.push_back(s.micro_start+i);r["position_basis"]="materialized microbatch order";}
 else if(rows==1 && s.micro_count>0){positions.push_back(s.micro_start+s.micro_count-1);r["position_basis"]="last requested output row; verify GET_ROWS ancestry";}
 else r["position_basis"]="unresolved; retain native axes";
 r["token_axis"]=axis;r["positions"]=positions;
 std::vector<unsigned char> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());
 std::vector<float> values;values.reserve(ggml_nelements(t));
 for(int64_t z=0;z<t->ne[2];z++)for(int64_t y=0;y<t->ne[1];y++)for(int64_t x=0;x<t->ne[0];x++){
  float v;std::memcpy(&v,raw.data()+x*t->nb[0]+y*t->nb[1]+z*t->nb[2],sizeof(v));values.push_back(v);
 }
 r["offset_bytes"]=s.bytes;r["values"]=values.size();r["bytes"]=values.size()*sizeof(float);
 s.tensor_file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));s.bytes+=values.size()*sizeof(float);
 if(!s.tensor_file)throw std::runtime_error("tensor persistence failed");
 s.record_file<<r.dump()<<'\n';if(!s.record_file)throw std::runtime_error("record persistence failed");return true;
}

int main(int argc,char **argv)try{
 if(argc<6)throw std::runtime_error("usage: observe MODEL OUT CASES BACKENDS plain|sham|capture [full|boundary] [followup|connected|trace] | capture residual|projection stream");
 const fs::path out=argv[2];if(fs::exists(out))throw std::runtime_error("output already exists");fs::create_directories(out);
 const std::string mode=argv[5];if(mode!="plain"&&mode!="sham"&&mode!="capture")throw std::runtime_error("bad mode");
 ggml_backend_load_all_from_path(argv[4]);auto mp=llama_model_default_params();mp.n_gpu_layers=99;
 auto *model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("model load");
 auto *vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);
 observation state;state.persist=mode=="capture";if(argc>6)state.scope=argv[6];
 if(argc>7)state.retention=argv[7];
 if(state.retention!="followup"&&state.retention!="connected"&&state.retention!="trace"&&state.retention!="stream")throw std::runtime_error("unknown tensor retention");
 if(state.retention=="stream"){state.stream_scope=state.scope;if(state.stream_scope!="residual"&&state.stream_scope!="projection")throw std::runtime_error("stream scope must be residual or projection");}
 if(state.retention=="trace"||state.retention=="stream")state.trace.bind_model(argv[1],out,argv[4]);
 auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=128;cp.n_ubatch=32;cp.n_threads=8;cp.n_threads_batch=8;
 cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.kv_unified=false;
 if(mode!="plain"){cp.cb_eval=callback;cp.cb_eval_user_data=&state;}
 auto *ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context");
 json cases;std::ifstream input(argv[3]);input>>cases;json results=json::array();
 for(auto &c:cases){
  const std::string id=c.at("id");if(id.find('/')!=std::string::npos)throw std::runtime_error("bad case id");const fs::path dir=out/id;fs::create_directory(dir);
  state.nodes=json::array();state.structures.clear();state.pointers.clear();state.counts.clear();state.bytes=state.events=0;state.call=0;state.micro=-1;state.prefill_next=0;
  state.stream_bytes=state.stream_events=0;state.stream_counts.clear();
  if(state.retention=="trace")state.trace.begin(dir,state.persist);
  else if(state.retention=="stream"){
   state.trace.begin(dir,state.persist);
   if(state.persist){state.stream_file.open(dir/"stream.f32",std::ios::binary);state.stream_records.open(dir/"stream.jsonl");}
  }
  else if(state.persist){state.tensor_file.open(dir/"tensors.bin",std::ios::binary);state.record_file.open(dir/"records.jsonl");}
  std::vector<llama_token> ids;
  if(c.contains("tokens"))ids=c.at("tokens").get<std::vector<llama_token>>();
  else{
   const std::string text="<bos><|turn>user\n"+c.at("prompt").get<std::string>()+"<turn|>\n<|turn>model\n";
   const int n=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
   if(n>=0)throw std::runtime_error("tokenize size");ids.resize(-n);
   const int m=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),false,true);
   if(m!=int(ids.size()))throw std::runtime_error("tokenize mismatch");c["tokens"]=ids;
  }if(ids.empty())throw std::runtime_error("empty prompt");
  state.trace_input_positions=c.value("trace_input_positions",std::vector<int>{});
  std::sort(state.trace_input_positions.begin(),state.trace_input_positions.end());
  if(std::adjacent_find(state.trace_input_positions.begin(),state.trace_input_positions.end())!=state.trace_input_positions.end())throw std::runtime_error("duplicate trace input position");
  for(auto p:state.trace_input_positions)if(p<int(ids.size()))throw std::runtime_error("selected trace positions must be single-token decode inputs");
  state.stream_positions=c.value("stream_positions",std::vector<int>{});
  std::sort(state.stream_positions.begin(),state.stream_positions.end());
  if(std::adjacent_find(state.stream_positions.begin(),state.stream_positions.end())!=state.stream_positions.end())throw std::runtime_error("duplicate stream position");
  for(auto p:state.stream_positions)if(p<int(ids.size()))throw std::runtime_error("stream positions must be single-token decode inputs");
  if(!state.stream_positions.empty()&&state.retention!="stream")throw std::runtime_error("stream positions require stream retention");
  state.label_filter=c.contains("stream_labels");state.stream_decision.clear();
  if(state.label_filter){
   if(state.retention!="stream")throw std::runtime_error("stream labels require stream retention");
   state.stream_labels=std::regex(c.at("stream_labels").get<std::string>());
  }
  state.stop_early=c.value("stop_after_last_label",false);state.stopped=state.recording=false;state.seen=0;state.computes=state.stops=state.after_stop=0;
  state.shapes.clear();state.reference=nullptr;
  if(state.stop_early&&(mode!="capture"||state.retention!="stream"||!state.label_filter||!c.contains("teacher_tokens")||!state.trace_input_positions.empty()||!state.stream_positions.empty()
     ||c.contains("patch")||c.contains("patch_sweep")||c.contains("watched_tokens")||c.contains("temperature")))
   throw std::runtime_error("stop_after_last_label reads a supplied answer under stream_labels and asks nothing else of the pass");
  // Watched tokens: their scores are saved at every decision, next to the top eight.
  const auto watched=c.value("watched_tokens",std::vector<llama_token>{});
  for(auto t:watched)if(t<0||t>=nv)throw std::runtime_error("watched token outside vocabulary");
  state.patch_on=state.sweeping=false;state.patches.clear();state.patched.clear();
  // Patch targets: {"label": recorded output, "row": row of the float32 vectors file}. One pass may carry several.
  const auto load_targets=[&](const json &targets,const std::string &vectors){
   state.patches.clear();state.patched.clear();
   for(const auto &target:targets){
    const std::string label=target.at("label").get<std::string>();
    if(!state.patches.emplace(label,patch_row(vectors,target.at("row").get<size_t>(),llama_model_n_embd(model))).second)throw std::runtime_error("patch target named twice: "+label);
   }
  };
  if(c.contains("patch")){
   if(state.retention!="stream")throw std::runtime_error("a patch requires stream retention");
   const auto &p=c.at("patch");state.patch_position=p.at("position").get<int>();
   if(state.patch_position<int(ids.size()))throw std::runtime_error("a patch position must be a single-token decode input");
   load_targets(p.at("targets"),p.at("vectors").get<std::string>());if(state.patches.empty())throw std::runtime_error("a patch needs at least one target");
   state.patch_on=true;
  }
  std::vector<llama_token> teacher;
  const bool forced=c.contains("teacher_tokens")||c.contains("teacher_text");
  const bool prefix=c.contains("teacher_prefix_tokens")||c.contains("teacher_prefix_text");
  if(forced&&prefix)throw std::runtime_error("full and prefix teacher modes are exclusive");
  if(c.contains("teacher_tokens"))teacher=c.at("teacher_tokens").get<std::vector<llama_token>>();
  else if(c.contains("teacher_text"))teacher=tokenize(vocab,c.at("teacher_text").get<std::string>());
  else if(c.contains("teacher_prefix_tokens"))teacher=c.at("teacher_prefix_tokens").get<std::vector<llama_token>>();
  else if(c.contains("teacher_prefix_text"))teacher=tokenize(vocab,c.at("teacher_prefix_text").get<std::string>());
  if(c.contains("teacher_prefix_tail")){
   if(!c.contains("teacher_prefix_tokens"))throw std::runtime_error("teacher tail requires exact prefix token IDs");
   const auto tail=tokenize(vocab,c.at("teacher_prefix_tail").get<std::string>());teacher.insert(teacher.end(),tail.begin(),tail.end());
  }
  for(auto t:teacher)if(t<0||t>=nv)throw std::runtime_error("teacher token outside vocabulary");
  if((forced||prefix)&&teacher.empty())throw std::runtime_error("empty teacher trajectory");
  if(prefix)for(auto t:teacher)if(llama_vocab_is_eog(vocab,t))throw std::runtime_error("teacher prefix contains end-of-generation token");
  const double temperature=c.value("temperature",0.0),top_p=c.value("top_p",1.0);
  if(!std::isfinite(temperature)||temperature<0||!std::isfinite(top_p)||top_p<=0||top_p>1)throw std::runtime_error("sampling settings");
  if((forced||prefix)&&temperature!=0)throw std::runtime_error("teacher forcing and stochastic sampling are separate modes");
  llama_sampler *sampler=nullptr;
  if(temperature>0){
   if(!c.contains("seed"))throw std::runtime_error("explicit sample seed required");
   const auto seed=c.at("seed").get<uint32_t>();if(seed==LLAMA_DEFAULT_SEED)throw std::runtime_error("random seed sentinel is not reproducible");
   sampler=llama_sampler_chain_init(llama_sampler_chain_default_params());
   // Nucleus is computed after temperature; no top-k, penalties, grammar or hidden defaults.
   llama_sampler_chain_add(sampler,llama_sampler_init_temp(float(temperature)));
   llama_sampler_chain_add(sampler,llama_sampler_init_top_p(float(top_p),1));
   llama_sampler_chain_add(sampler,llama_sampler_init_dist(seed));
  }
  const int max_tokens=forced?int(teacher.size()):c.value("max_tokens",256);if(max_tokens<1 || ids.size()+max_tokens>cp.n_ctx)throw std::runtime_error("context budget");
  if(prefix&&max_tokens<=int(teacher.size()))throw std::runtime_error("teacher prefix must leave an unforced continuation budget");
  llama_memory_clear(llama_get_memory(ctx),true);state.phase="prefill";state.position=ids.size()-1;
  auto prompt=ids;
  for(size_t begin=0;begin<prompt.size();begin+=cp.n_batch){
   const int count=std::min<size_t>(cp.n_batch,prompt.size()-begin);
   state.position=begin+count-1;state.micro=-1;state.pointers.clear();
   if(llama_decode(ctx,llama_batch_get_one(prompt.data()+begin,count)))throw std::runtime_error("prefill");
   if(begin+count<prompt.size())++state.call;
  }
  json steps=json::array();std::vector<int> generated;std::string content;bool ended=false;
  for(int step=0;step<max_tokens;step++){
   if(state.stop_early){
    // No vocabulary scores exist: the compute ended before the output head. The supplied token is fed as it is.
    llama_token next=teacher[step];
    steps.push_back({{"step",step},{"input_position",state.position},{"selected",next},{"piece",piece(vocab,next)},{"piece_bytes_hex",hex(piece(vocab,next))},{"selection","teacher"}});
    generated.push_back(next);if(llama_vocab_is_eog(vocab,next)){ended=true;break;}content+=piece(vocab,next);
    if(step+1<max_tokens){++state.call;state.phase="decode";state.position=ids.size()+step;state.micro=-1;state.pointers.clear();if(llama_decode(ctx,llama_batch_get_one(&next,1)))throw std::runtime_error("decode");}
    continue;
   }
   const float *logits=llama_get_logits_ith(ctx,-1);const llama_token greedy=std::max_element(logits,logits+nv)-logits;
   const bool step_forced=step<int(teacher.size());
   llama_token next=step_forced?teacher[step]:(sampler?llama_sampler_sample(sampler,ctx,-1):greedy);
   for(int k=0;k<nv;k++)if(!std::isfinite(logits[k]))throw std::runtime_error("nonfinite vocabulary score");
   unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(logits),nv*sizeof(float),digest);
   const auto score_hash=hex(std::string(reinterpret_cast<const char*>(digest),SHA256_DIGEST_LENGTH));
   std::vector<int>rank(nv);std::iota(rank.begin(),rank.end(),0);std::partial_sort(rank.begin(),rank.begin()+8,rank.end(),[&](int a,int b){return logits[a]>logits[b];});
   json top=json::array();for(int i=0;i<8;i++)top.push_back({{"id",rank[i]},{"piece",piece(vocab,rank[i])},{"piece_bytes_hex",hex(piece(vocab,rank[i]))},{"logit",logits[rank[i]]}});
   steps.push_back({{"step",step},{"input_position",state.position},{"selected",next},{"piece",piece(vocab,next)},{"piece_bytes_hex",hex(piece(vocab,next))},{"logits_sha256",score_hash},{"top",top}});
   steps.back()["greedy_token"]=greedy;steps.back()["selection"]=step_forced?"teacher":(sampler?"sample":"greedy");steps.back()["selected_logit"]=logits[next];
   if(!watched.empty()){json w=json::array();for(auto t:watched)w.push_back(logits[t]);steps.back()["watched_logits"]=w;}
   generated.push_back(next);if(llama_vocab_is_eog(vocab,next)){ended=true;break;}content+=piece(vocab,next);
   if(step+1<max_tokens){++state.call;state.phase="decode";state.position=ids.size()+step;state.micro=-1;state.pointers.clear();if(llama_decode(ctx,llama_batch_get_one(&next,1)))throw std::runtime_error("decode");}
  }
  json sweep=json::array();int sweep_position=-1;
  if(c.contains("patch")&&state.patched.size()!=state.patches.size())throw std::runtime_error("a patch target was not computed");
  if(c.contains("patch_sweep")){
   if(!forced||teacher.size()<2||generated.size()!=teacher.size()||state.retention!="stream"||state.patch_on)
    throw std::runtime_error("a patch sweep replays a supplied answer of at least two tokens under stream retention, without a single patch");
   const auto &ps=c.at("patch_sweep");const std::string vectors=ps.at("vectors").get<std::string>();
   llama_token input=teacher[teacher.size()-2];sweep_position=ids.size()+teacher.size()-2;
   state.sweeping=true;state.patch_position=sweep_position;size_t number=0;
   for(const auto &targets:ps.at("passes")){
    // The decision's own pass is removed from memory and run again; a pass without targets runs it unchanged.
    if(!llama_memory_seq_rm(llama_get_memory(ctx),0,sweep_position,-1))throw std::runtime_error("cannot remove the decision position from memory");
    load_targets(targets,vectors);state.patch_on=!state.patches.empty();
    ++state.call;state.phase="decode";state.position=sweep_position;state.micro=-1;state.pointers.clear();
    if(llama_decode(ctx,llama_batch_get_one(&input,1)))throw std::runtime_error("patched decode");
    if(state.patched.size()!=state.patches.size())throw std::runtime_error("a patch target was not computed in pass "+std::to_string(number));
    const float *logits=llama_get_logits_ith(ctx,-1);
    for(int k=0;k<nv;k++)if(!std::isfinite(logits[k]))throw std::runtime_error("nonfinite vocabulary score in a patched pass");
    unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(logits),nv*sizeof(float),digest);
    std::vector<int>rank(nv);std::iota(rank.begin(),rank.end(),0);std::partial_sort(rank.begin(),rank.begin()+8,rank.end(),[&](int a,int b){return logits[a]>logits[b];});
    json top=json::array();for(int i=0;i<8;i++)top.push_back({{"id",rank[i]},{"piece",piece(vocab,rank[i])},{"logit",logits[rank[i]]}});
    json w=json::array();for(auto t:watched)w.push_back(logits[t]);
    sweep.push_back({{"pass",number++},{"targets",targets.size()},{"logits_sha256",hex(std::string(reinterpret_cast<const char*>(digest),SHA256_DIGEST_LENGTH))},{"top",top},{"watched_logits",w}});
   }
   state.patch_on=state.sweeping=false;
  }

  json result={{"id",id},{"mode",mode},{"scope",state.scope},{"fusion_disabled",std::getenv("GGML_CUDA_DISABLE_FUSION")?std::getenv("GGML_CUDA_DISABLE_FUSION"):"0"},
   {"tokens",generated},{"content",content},{"content_bytes_hex",hex(content)},{"text_encoding","UTF-8 display replaces incomplete token-byte fragments; *_bytes_hex preserves exact original bytes"},{"ended",ended},{"stopped_limit",!ended&&int(generated.size())==max_tokens},{"prompt_tokens",ids.size()},{"vocabulary_size",nv},
   {"score_storage","SHA256 of all native float32 logits per decision, plus top eight; all logits checked finite"},{"retained_scope","All layer exits and final norm, input-scaled, attention/feed-forward addition operands/results and per-layer embedding at layers 12-16; callback boundaries remain full"},{"capture_events",state.events},{"tensor_bytes",state.bytes},{"counts",state.counts},{"steps",steps}};
  if(state.stop_early){
   state.end_compute();
   if(state.computes==0||state.stops!=state.computes-state.shapes.size())throw std::runtime_error("stop_after_last_label: a compute after the first of its shape did not stop");
   json shapes=json::array();std::set<size_t> wanted;
   for(const auto &[shape,labels]:state.shapes){shapes.push_back({{"tokens",shape.first},{"output",shape.second},{"wanted_outputs",labels.size()}});wanted.insert(labels.size());}
   result["stop_after_last_label"]={{"wanted_outputs_per_compute",wanted.size()==1?json(*wanted.begin()):json(nullptr)},{"computes",state.computes},{"whole_computes",state.shapes.size()},
    {"stopped",state.stops},{"callbacks_after_a_stop",state.after_stop},{"shapes",shapes}};
   result["score_storage"]="none: the first compute of each microbatch shape ran whole; every later one wrote the same wanted outputs in the same order and ended at the last of them, "
    "before the output head; recorded outputs checked finite, vocabulary scores and the layers after the stop unchecked; the supplied tokens were fed as they are";
  }
  result["selection"]=forced?"teacher":(prefix?"teacher-prefix-then-greedy":(sampler?"sample":"greedy"));result["teacher_completed"]=(forced||prefix)&&generated.size()>=teacher.size();
  if(state.retention=="connected")result["retained_scope"]="15 declared connected boundaries: layer13 exit, attention/FF/exit14 and15, attention input/raw projection/normalized contribution/FF/exit16, exits24/34 and final norm. Full callback boundaries and full recorded graph unchanged.";
  result["teacher_token_count"]=teacher.size();result["conditioned_continuation"]=prefix;
  if(!watched.empty()){json w=json::array();for(auto t:watched)w.push_back({{"id",t},{"piece",piece(vocab,t)}});result["watched_tokens"]=w;}
  if(c.contains("patch"))result["patch"]={{"position",state.patch_position},{"targets",state.patches.size()},{"applied",state.patched.size()}};
  if(!sweep.empty())result["patch_sweep"]={{"input_position",sweep_position},{"passes",sweep}};
  result["stopped_limit"]=!forced&&!ended&&int(generated.size())==max_tokens;
  result["sampling"]={{"temperature",temperature},{"top_p",top_p},{"seed",c.value("seed",json(nullptr))},{"chain","temperature, top_p, dist; no penalties"}};
  if(sampler)llama_sampler_free(sampler);
  if(state.retention=="stream"){
   if(state.persist){state.stream_file.close();state.stream_records.close();}
   result["stream"]={{"scope",state.stream_scope},{"events",state.stream_events},{"bytes",state.stream_bytes},{"counts",state.stream_counts},
    {"storage","stream.f32: logical x-fast float32 values of each recorded output; stream.jsonl: one record per output with layout, positions and byte offset"},
    {"boundary","Outputs only at every token: inp_scaled, attn_post_norm, attn_out, ffn_post_norm, pe_in, per_layer_embd_out, pe_out (per-layer merge), l_out, result_norm; projection scope adds attn_norm, ffn_norm, kqv_out, ffn_geglu, Qcur_pos, Kcur_pos, Vcur_normed. No input preimages, no per-operation device synchronization."}};
   result["retained_scope"]="Stream outputs at every token; complete signal_trace operations at declared trace_input_positions.";
   if(state.label_filter)result["stream_labels"]=c.at("stream_labels");
   if(!state.stream_positions.empty()){result["stream_positions"]=state.stream_positions;result["retained_scope"]="Stream outputs at the declared stream_positions only; every other token runs as ordinary inference.";}
   result["signal_trace"]=state.trace.finish();
   if(!state.trace_input_positions.empty())result["trace_input_positions"]=state.trace_input_positions;
   result["capture_events"]=state.stream_events+state.trace.count;result["tensor_bytes"]=state.stream_bytes+state.trace.bytes;
   state.events=state.stream_events+state.trace.count;state.bytes=state.stream_bytes+state.trace.bytes;
  }
  if(state.retention=="trace"){
   result["signal_trace"]=state.trace.finish();result["retained_scope"]="All materialized scheduler transformations, with native pre-operation inputs and post-operation outputs; static native weights bound to GGUF ranges, cache writes versioned.";
   state.events=state.trace.count;state.bytes=state.trace.bytes;
   result["capture_events"]=state.events;result["tensor_bytes"]=state.bytes;
   if(!state.trace_input_positions.empty()){
    result["trace_input_positions"]=state.trace_input_positions;
    result["retained_scope"]="Complete native operations at declared single-token input positions. Preceding history ran normally but was not recorded in this capture. Cache preimages are retained; predecessor writes outside the selected positions are not linked here.";
   }
  }else if(state.persist){state.tensor_file.close();state.record_file.close();save(dir/"graph.json",{{"identity","structural node IDs; execution identity additionally requires call and microbatch"},{"cache_dependency_boundary","KV runtime slot mappings and mutable cache contents are unrecorded. Structural cache-view edges do not establish links to previous captured token writes."},{"nodes",state.nodes}});}
  save(dir/"result.json",result);save(dir/"case.json",c);results.push_back({{"id",id},{"content",content},{"tokens",generated},{"ended",ended},{"tensor_bytes",state.bytes},{"capture_events",state.events}});
  std::cout<<results.back().dump(-1,' ',false,json::error_handler_t::replace)<<std::endl;
 }
 save(out/"results.json",results);save(out/"settings.json",{{"mode",mode},{"scope",state.scope},{"n_ctx",2048},{"n_batch",128},{"n_ubatch",32},{"flash_attention",true},{"selection","per-case; absent temperature/teacher fields means original greedy behavior"}});
 llama_free(ctx);llama_model_free(model);return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
