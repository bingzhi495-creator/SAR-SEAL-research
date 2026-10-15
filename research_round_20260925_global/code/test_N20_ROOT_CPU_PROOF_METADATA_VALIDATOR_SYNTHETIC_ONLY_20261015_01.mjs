// SYNTHETIC_ONLY: hand-built JSON metadata. No actual refs, tensors, data or IO.
import assert from 'node:assert/strict';
import { validateImageProof } from './N20_ROOT_CPU_PROOF_METADATA_VALIDATOR_SOURCE_ONLY_20261015_01.mjs';

const names = [24,8].flatMap(epoch => ['LOCAL_ERROR','CENTER_ERROR'].flatMap(arm =>
  ['atmost','exact'].map(mode => 'E'+epoch+'_'+arm+'_'+mode)));
const ref = name => ({path:'SYNTHETIC_ONLY/'+name,bytes:1,sha256:'a'.repeat(64)});
const row = {dataset_index:0,image_id:17,bundle:ref('bundle'),input_carrier:{bundle:ref('old')},
  member_locators:['SYNTHETIC_ONLY/input','SYNTHETIC_ONLY/profile'],
  query_profiles:Array.from({length:100},(_,q) => ({query_id:q,K:q===0?0:q===1?2:1,C_count:q===0?17:q===1?2049:1}))};
const prStats = points => Object.fromEntries(['original_head','independent_PR'].map(name => [name,{
  calls:Math.ceil(points/2048),points,strict_sign_exact:true,
  residual_max_absolute_error:1e-4,combined_max_absolute_error:1e-4}]));
const details = row.query_profiles.map(({query_id,K,C_count}) => {
  const active=K?C_count:0;
  const formula_stats=prStats(active);
  const profiles=Object.fromEntries(names.map(name => {
    const match=/^E(24|8)_(LOCAL_ERROR|CENTER_ERROR)_(atmost|exact)$/.exec(name);
    const epoch=Number(match[1]),arm=match[2],mode=match[3],selected=mode==='exact'?K:Math.min(K,1);
    return [name,{epoch,arm,mode,selected_points:selected,selected_indices_exact:true,
      full_mask_exact:true,outside_selected_native_exact:true,RLE_exact:true,PR_strict_sign_exact:true,
      CPU_scout_outputs_used_for_ranking:false,CPU_approximate_PR_used_for_ranking:false,
      main_result:epoch===24&&mode==='atmost',selected_PR_formula_stats:prStats(selected)}];
  }));
  const scout_numeric=Object.fromEntries([24,8].flatMap(epoch => ['LOCAL_ERROR','CENTER_ERROR'].map(arm =>
    ['E'+epoch+'_'+arm,{chunk_calls:Math.ceil(active/2048),points:active,
      actual_true_F9_or_center_inputs_byte_exact:true,factory_vs_actual_GPU_max_absolute_error:1e-4,
      independent_vs_actual_GPU_max_absolute_error:1e-4,CPU_two_formula_max_absolute_error:2e-4}])));
  return {query_id,K,C_points:C_count,profiles,scout_numeric,
    common_C_PR:{strict_sign_exact:true,formula_stats,
      formula_chunk_calls:Object.fromEntries(Object.entries(formula_stats).map(([name,stat])=>[name,stat.calls]))},
    all_selected_indices_exact:true,all_full_masks_and_RLE_exact:true,
    actual_true_F9_or_center_inputs_byte_exact:true,actual_saved_FP64_scores_canonical_exact:true,
    original_query_ids_classes_S0_export_order_require_external_binding:true,
    CPU_scout_outputs_used_for_ranking:false,AP_evaluated:false,ROOT_scientific_qualification_conferred:false,
    CPU_feature_rerender_equivalence_proven:false,actual_capture_provenance_proven_here:false,
    native_R_reference_reuse_proven_here:false,atol:1e-4,rtol:0};
});
const value={dataset_index:0,image_id:17,queries:100,bundle:row.bundle,input_carrier:row.input_carrier,
  consumption:{all_registered_members_consumed:true,members:2,queries:100,ROOT_scientific_qualification_conferred:false},details};
assert.deepEqual(validateImageProof(value,row,names),{members:2,queries:100,profiles:800,
  factory_vs_actual_GPU_max_absolute_error:1e-4,independent_vs_actual_GPU_max_absolute_error:1e-4});
const reject = mutate => {const bad=structuredClone(value);mutate(bad);assert.throws(()=>validateImageProof(bad,row,names));};
reject(v=>{[v.details[0],v.details[1]]=[v.details[1],v.details[0]];});
reject(v=>{delete v.details[1].profiles.E24_LOCAL_ERROR_exact;});
reject(v=>{v.details[1].common_C_PR.formula_stats.original_head.calls=1;});
reject(v=>{v.details[1].scout_numeric.E24_LOCAL_ERROR.factory_vs_actual_GPU_max_absolute_error=0.00010001;});
reject(v=>{v.details[1].scout_numeric.E24_LOCAL_ERROR.independent_vs_actual_GPU_max_absolute_error=NaN;});
reject(v=>{v.details[1].profiles.E24_LOCAL_ERROR_exact.selected_points=1;});
reject(v=>{v.details[1].ROOT_scientific_qualification_conferred=true;});
reject(v=>{v.details[1].K=1;});
reject(v=>{v.details[1].C_points=2048;});
reject(v=>{v.details[0].scout_numeric.E24_LOCAL_ERROR.chunk_calls=1;});
reject(v=>{v.consumption.members=true;});
console.log('SYNTHETIC_ONLY GREEN: hand fixture100queries/800profiles; K0,2048boundary,1e-4vsGPU and2e-4pair;11 rejection cases');
