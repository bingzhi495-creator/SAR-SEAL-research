// SOURCE_ONLY / UNBOUND: metadata-only mirror of ordinary finalfront02's PR/image validators.
// No IO/imports/Source gate/tensor arithmetic. ROOT supplies already-authenticated JSON/row.
// Successful validation does not confer scientific qualification or replay numeric proofs.
const ARMS=['LOCAL_ERROR','CENTER_ERROR'];
const EPOCHS=[24,8];
const MODES=['atmost','exact'];
const FORMULAS=['original_head','independent_PR'];
const ATOL=1e-4;
const object=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);
const integer=value=>Number.isSafeInteger(value)&&value>=0;
function need(ok,path,message){if(!ok)throw new TypeError('Image proof metadata invalid: '+path+': '+message);}
function keys(value,expected){return object(value)&&Object.keys(value).length===expected.length&&expected.every(key=>Object.hasOwn(value,key));}
function same(left,right){
  if(typeof left!==typeof right)return false;
  if(typeof left==='number')return Number.isFinite(left)&&Number.isFinite(right)&&left===right;
  if(left===null||right===null||typeof left!=='object')return left===right;
  if(Array.isArray(left)||Array.isArray(right))return Array.isArray(left)&&Array.isArray(right)&&left.length===right.length&&left.every((item,i)=>same(item,right[i]));
  return Object.keys(left).length===Object.keys(right).length&&Object.keys(left).every(key=>Object.hasOwn(right,key)&&same(left[key],right[key]));
}
function error(value,path){need(typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=ATOL,path,'finite absolute error in[0,1e-4]');}
function prStats(stats,points,path){
  need(keys(stats,FORMULAS),path,'exact two original PR formulas');
  for(const name of FORMULAS){
    const stat=stats[name],p=path+'.'+name;
    need(object(stat)&&integer(stat.calls)&&integer(stat.points)&&stat.calls===Math.ceil(points/2048)&&stat.points===points,p,'complete original2048 chunk/point counts');
    need(stat.strict_sign_exact===true,p+'.strict_sign_exact','must be true');
    for(const key of ['residual_max_absolute_error','combined_max_absolute_error'])error(stat[key],p+'.'+key);
  }
}

export function validateImageProof(value,row,names){
  const fixedNames=EPOCHS.flatMap(epoch=>ARMS.flatMap(arm=>MODES.map(mode=>'E'+epoch+'_'+arm+'_'+mode)));
  need(Array.isArray(names)&&names.length===8&&new Set(names).size===8&&fixedNames.every(name=>names.includes(name)),'names','all fixed eight profiles');
  need(object(row)&&Array.isArray(row.member_locators)&&Array.isArray(row.query_profiles)&&row.query_profiles.length===100,'row','original authenticated manifest row with100 queries');
  need(keys(value,['dataset_index','image_id','queries','bundle','input_carrier','consumption','details']),'image','exact seven fields');
  need(integer(value.dataset_index)&&integer(value.image_id)&&value.dataset_index===row.dataset_index&&value.image_id===row.image_id&&value.queries===100,'image','index/image/query count match original row');
  need(same(value.bundle,row.bundle)&&same(value.input_carrier,row.input_carrier),'image.binding','original actual bundle and input carrier exact');
  const members=row.member_locators.length,consumed=value.consumption;
  need(members>0&&object(consumed)&&consumed.all_registered_members_consumed===true&&integer(consumed.members)&&consumed.members===members&&consumed.queries===100&&consumed.ROOT_scientific_qualification_conferred===false,'consumption','all registered members; no self qualification');
  need(Array.isArray(value.details)&&value.details.length===100&&value.details.every((item,q)=>object(item)&&integer(item.query_id)&&item.query_id===q),'details','100 natural queries in order0..99');
  const scoutNames=EPOCHS.flatMap(epoch=>ARMS.map(arm=>'E'+epoch+'_'+arm));
  const trueFlags=['all_selected_indices_exact','all_full_masks_and_RLE_exact','actual_true_F9_or_center_inputs_byte_exact','actual_saved_FP64_scores_canonical_exact','original_query_ids_classes_S0_export_order_require_external_binding'];
  const falseFlags=['CPU_scout_outputs_used_for_ranking','AP_evaluated','ROOT_scientific_qualification_conferred','CPU_feature_rerender_equivalence_proven','actual_capture_provenance_proven_here','native_R_reference_reuse_proven_here'];
  let profiles=0,maxFactory=0,maxIndependent=0;
  for(let q=0;q<100;q++){
    const item=value.details[q],query=row.query_profiles[q],p='details['+q+']';
    need(object(query)&&integer(query.K)&&integer(query.C_count)&&query.K<=query.C_count,p+'.manifest','nonnegative physical K/C counts with K<=C');
    const K=query.K,C=query.C_count,active=K?C:0;
    need(integer(item.K)&&integer(item.C_points)&&item.K===K&&item.C_points===C,p+'.counts','original K/C exact');
    need(trueFlags.every(key=>item[key]===true)&&falseFlags.every(key=>item[key]===false),p+'.flags','numeric completion and explicit evidence limits');
    need(item.atol===ATOL&&item.rtol===0,p+'.tolerance','unchanged absolute1e-4/rtol0');
    need(keys(item.profiles,names)&&keys(item.scout_numeric,scoutNames),p+'.profiles','all eight profiles and four model stats');
    const common=item.common_C_PR;
    need(object(common)&&common.strict_sign_exact===true,p+'.common_C_PR','strict sign exact');
    prStats(common.formula_stats,active,p+'.common_C_PR.formula_stats');
    const calls=Object.fromEntries(FORMULAS.map(name=>[name,common.formula_stats[name].calls]));
    need(same(common.formula_chunk_calls,calls),p+'.common_C_PR.formula_chunk_calls','formula counts exact');
    for(const name of scoutNames){
      const stat=item.scout_numeric[name],s=p+'.scout_numeric.'+name;
      need(object(stat)&&integer(stat.chunk_calls)&&integer(stat.points)&&stat.chunk_calls===Math.ceil(active/2048)&&stat.points===active&&stat.actual_true_F9_or_center_inputs_byte_exact===true,s,'complete actual C2048 counts and actual-input check');
      error(stat.factory_vs_actual_GPU_max_absolute_error,s+'.factory_vs_actual_GPU_max_absolute_error');
      error(stat.independent_vs_actual_GPU_max_absolute_error,s+'.independent_vs_actual_GPU_max_absolute_error');
      maxFactory=Math.max(maxFactory,stat.factory_vs_actual_GPU_max_absolute_error);
      maxIndependent=Math.max(maxIndependent,stat.independent_vs_actual_GPU_max_absolute_error);
      // CPU_two_formula_max_absolute_error is diagnostic; no extra1e-4 threshold.
    }
    for(const name of names){
      const profile=item.profiles[name],s=p+'.profiles.'+name;
      need(object(profile)&&integer(profile.epoch)&&EPOCHS.includes(profile.epoch)&&ARMS.includes(profile.arm)&&MODES.includes(profile.mode)&&name==='E'+profile.epoch+'_'+profile.arm+'_'+profile.mode,s,'fixed epoch/arm/mode name');
      need(integer(profile.selected_points)&&profile.selected_points<=K&&(profile.mode!=='exact'||profile.selected_points===K),s+'.selected_points','legal atmost/exact K count');
      need(['selected_indices_exact','full_mask_exact','outside_selected_native_exact','RLE_exact','PR_strict_sign_exact'].every(key=>profile[key]===true)&&profile.CPU_scout_outputs_used_for_ranking===false&&profile.CPU_approximate_PR_used_for_ranking===false,s+'.flags','exact actual support/mask/PR/RLE; no approximate ranking');
      need(profile.main_result===(profile.epoch===24&&profile.mode==='atmost'),s+'.main_result','only E24-atmost main');
      prStats(profile.selected_PR_formula_stats,profile.selected_points,s+'.selected_PR_formula_stats');
      profiles++;
    }
  }
  return {members,queries:100,profiles,factory_vs_actual_GPU_max_absolute_error:maxFactory,
    independent_vs_actual_GPU_max_absolute_error:maxIndependent};
}
