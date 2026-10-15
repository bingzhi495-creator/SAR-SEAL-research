"""UNBOUND N20 full187 COCO final RUN tail; no RAW front, CLI or execution.

ROOT supplies source_gate/require/api/read/closed/bound/load/reference/write and
the current own CPU177/failed26/pins203 front. source_gate must join the four
original DEV GPU141..144 and 32 numeric CPU145..176 actual0 records before
returning; pending outputs alone cannot establish those original terminals.

Gate ABI: own closure/context/source/spec_ref/admission_ref/admission/common/
lineage/parent; legacy_root; numeric_CPU_evidence[32], each exact{descriptor,
parent,report,child}; child has parent_GPU_report/manifest plus scope_rows/
scope_expected_predictions/scope_capture_refs; full_rows[187], image_proofs[187],
registered_members(int), gpu_helpers.validate_legacy_COCO, N20_protocol,
training and trained_checkpoints. No model/numerical proof is loaded or rerun.

Spec: numeric_CPU_shards, completed_GPU_shards, numeric_CPU_shard_gate_source,
CPU_execution_partition_supplement, interpreter_source; frozen recovery/cache/
protocol and preserved failed24/25/26 refs. All must be SOURCE-closed by the front.
Final Q stays pending. Independent ROOT post-actual qualification is separate.
"""

import gc
import json
import math
import os
from pathlib import Path
import sys
import time


N20_FINAL_Q_SCHEMA = 'N20-fixedDEV-full187-CPU-COCO-qualification-v1'
N20_FINAL_STATUS = 'N20_ALL18700_FIXED_DEV_CPU_NUMERIC_AND_FULL_COCO_COMPLETE_PENDING_ROOT_ACTUAL0'
N20_FINAL_CPU_RANGES = ((0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 42), (42, 47), (47, 53), (53, 59), (59, 65), (65, 71), (71, 77), (77, 83), (83, 89), (89, 94), (94, 100), (100, 106), (106, 112), (112, 118), (118, 124), (124, 130), (130, 136), (136, 141), (141, 147), (147, 153), (153, 159), (159, 165), (165, 171), (171, 177), (177, 183), (183, 187))
N20_FINAL_GPU_OWNERS = (141,142,143,144)
N20_FINAL_CPU_OWNERS = (145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176)
N20_FINAL_THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
N20_FINAL_INTERPRETER_SHA = 'd5bcf29c97b465a3ad185437ae2c9f2c56a4bcbcced87da097b54ef9e5c04564'


def n20_final_policy_names():
    return ('native_official','strong_fixed_E24_Omega') + tuple(
        'E%d_%s_%s' % (epoch,arm,mode) for epoch in (24,8)
        for arm in ('LOCAL_ERROR','CENTER_ERROR') for mode in ('atmost','exact'))


def n20_final_AP_percent(value):
    require(type(value) in (int,float) and math.isfinite(value) and (value==-1 or 0<=value<=1),
            'Official COCO raw AP0..1 or legal undefined -1; no bool or hidden scaling')
    return -1 if value==-1 else value*100


def n20_validate_final_interpreter(interpreter, protocol, ref):
    expected_decision = {
        'Delta':'AP_LOCAL-max(AP_CENTER,strong_R)',
        'nonpositive':'STOP current recipe, no extra seeds/budget/fusion',
        'positive_below1':'at most seed202; permit203 only if two individually beat CENTER and meanDelta>=1',
        'at_least1':'replicate202 and203', 'secondary_not_rescue':['E8','exact']}
    require(ref['bytes']==3450 and ref['sha256']==N20_FINAL_INTERPRETER_SHA
        and interpreter.STRONG_AP==68.09938494745545
        and tuple(interpreter.SEEDS)==(202610150201,202610150202,202610150203)
        and interpreter.PROTOCOL_SCHEMA==protocol['schema']=='N20-true-feature-neighborhood-fixed24-protocol-v1'
        and interpreter.DECISION_LITERAL==protocol['decision']==expected_decision
        and protocol['strong_R_AP']==68.09938494745545
        and protocol['first_initialization_seed']==202610150201
        and protocol['repeat_initialization_seeds']==[202610150201,202610150202,202610150203]
        and protocol['primary_endpoint']=='E24 atmost whole18700 query COCO mask AP percent; old C/K, old score rule, positive support and original PR'
        and callable(getattr(interpreter,'interpret_fixed_main',None)),
        'Exact prelocked first201 interpreter/protocol decision; repeat permission is not publication acceptance')


def n20_final_main_preview(interpreter, results):
    return api(interpreter,'interpret_fixed_main')([dict(seed=202610150201,epoch=24,
        mode='atmost',AP_unit='percent',
        LOCAL_AP=n20_final_AP_percent(results['E24_LOCAL_ERROR_atmost']['stats'][0]),
        CENTER_AP=n20_final_AP_percent(results['E24_CENTER_ERROR_atmost']['stats'][0]),
        strong_AP=n20_final_AP_percent(results['strong_fixed_E24_Omega']['stats'][0]))])


def n20_merge_policy_predictions(pieces, image_ids, native, actual_GPU_predictions):
    require(type(pieces) is list and len(pieces)==32 and type(image_ids) is list
        and len(image_ids)==len(set(image_ids))==187
        and type(native) is list and type(actual_GPU_predictions) is list
        and len(native)==len(actual_GPU_predictions)==18700,
        '32 exact scopes and full original18700 population; never average local AP')
    combined=[]
    for values,(first,last) in zip(pieces,N20_FINAL_CPU_RANGES):
        expected=native[first*100:last*100]
        require(type(values) is list and len(values)==(last-first)*100
            and [value['image_id'] for value in values]==[image for image in image_ids[first:last] for _ in range(100)]
            and all(type(value) is dict and 'segmentation' in value
                and {key:item for key,item in value.items() if key!='segmentation'}==
                    {key:item for key,item in baseline.items() if key!='segmentation'}
                for value,baseline in zip(values,expected)),
            'Every child prediction retains original score/class/query/ordinal export order')
        combined.extend(values)
    require(combined==actual_GPU_predictions and len(combined)==18700,
            'Complete merged mask/RLE sequence equals original qualified actual GPU captures')
    return combined


def run(spec_path):
    started=time.monotonic()
    require(Path(spec_path).is_absolute(),'Own absolute N20 final CPU specification')
    path=Path(spec_path).resolve(strict=True);spec=read(path);gate=source_gate(path,spec)
    c=gate['closure'];context=gate['context'];gpu=gate['gpu_helpers']
    require(spec['schema']=='ROOT-N20-fixedDEV-full187-CPU-COCO-final-spec-v1'
        and spec['normal_RAW_count']==spec['expected_normal_RAW_count']==177
        and spec['total_failed_RAW_count']==26 and spec['total_terminal_identity_pin_count']==203
        and spec['initialization_seed']==202610150201 and spec['maximum_process_RSS_bytes']==16*1024**3
        and spec['deadline_seconds']==1800 and spec['automatic_retry'] is False,
        'Independent own final177/current26/pins203/seed201/resources, not historical recovery24')
    evidence=gate['numeric_CPU_evidence'];rows=gate['full_rows'];proof_refs=gate['image_proofs']
    require(type(evidence) is list and len(evidence)==32
        and all(type(item) is dict and set(item)=={'descriptor','parent','report','child'} for item in evidence)
        and [item['descriptor']['cpu_shard_owner_index'] for item in evidence]==list(N20_FINAL_CPU_OWNERS)
        and [item['shard_owner_index'] for item in spec['completed_GPU_shards']]==list(N20_FINAL_GPU_OWNERS)
        and len(rows)==len(proof_refs)==187 and [row['dataset_index'] for row in rows]==list(range(187))
        and len({row['image_id'] for row in rows})==187 and len({ref['path'] for ref in proof_refs})==187
        and type(gate['registered_members']) is int and gate['registered_members']>0,
        'Only complete four-GPU/32-CPU/187 image and registered-proof metadata from the front')
    output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir(),
            'Fresh admitted final CPU output with existing parent')
    for key in ('interpreter_source','numeric_CPU_shard_gate_source','CPU_execution_partition_supplement',
            'N20_root_training_qualified_result','N20_locked_protocol','legacy_qualified_result',
            'N20_failed_training_stderr','N20_failed_DEV_capture_stderr','N20_failed26_helper_source'):
        closed(c,spec[key])
    for key in ('reader_source','COCO_source','DEV'):
        closed(c,context[key])
    interpreter=load(spec['interpreter_source'],'_N20_locked_first201_main_only_interpreter')
    n20_validate_final_interpreter(interpreter,gate['N20_protocol'],spec['interpreter_source'])
    require(callable(getattr(gpu,'validate_legacy_COCO',None)), 'Original inline COCO12 semantic validator required')
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and all(os.environ.get(key)=='2' for key in N20_FINAL_THREAD_ENV)
        and psutil.virtual_memory().available>=16*1024**3 and psutil.disk_usage(str(output.parent)).free>=1024**3,
        'Actual independent CPU01/emptyCUDA/two threads/minRAM16GiB/free1GiB')
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0'
        and not torch.cuda.is_initialized(), 'Original closed cold CPU numerical libraries, no model load')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
    reader=load(context['reader_source'],'_N20_final_original_population')
    official=load(context['COCO_source'],'_N20_final_original_official_full_COCO')
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)
    cocoeval_module=sys.modules[COCOeval.__module__];closed(c,reference(cocoeval_module.__file__))
    outside=torch.get_rng_state().clone();peak_RSS=0;minimum_available_RAM=None
    def check():
        nonlocal peak_RSS,minimum_available_RAM
        rss=process.memory_info().rss;available=psutil.virtual_memory().available
        peak_RSS=max(peak_RSS,rss);minimum_available_RAM=available if minimum_available_RAM is None else min(minimum_available_RAM,available)
        require(time.monotonic()-started<1800 and rss<16*1024**3 and available>=16*1024**3
            and not torch.cuda.is_initialized() and process.cpu_affinity()==[0,1]
            and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            and all(os.environ.get(key)=='2' for key in N20_FINAL_THREAD_ENV)
            and torch.get_num_threads()==torch.get_num_interop_threads()==cv2.getNumThreads()==2,
            'Final CPU resource/cold-CUDA/deadline bounds, no retry or model/trajectory rerun')
    check();coco=COCO(str(closed(c,context['DEV'])));ids=list(coco.imgs)
    require(len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1}
        and ids==[item['id'] for item in coco.dataset['images']]
        and [row['image_id'] for row in rows]==ids,
        'Original full187/380GT/category1/canonical image order; no annotation normalization')
    original_results=gate['legacy_root']['control_results']
    native=read(closed(c,original_results['native_official']['predictions']))
    strong=read(closed(c,original_results['strong_fixed_E24_Omega']['predictions']))
    reader.population(coco,native,ids);reader.population(coco,strong,ids)
    for name in ('native_official','strong_fixed_E24_Omega'):
        gpu.validate_legacy_COCO(original_results[name])
    require(original_results['strong_fixed_E24_Omega']['stats'][0]*100==68.09938494745545,
        'Exact prelocked strong R; native all12 raw stats stay bound to the original qualified reference')
    output.mkdir();results={}
    for name in n20_final_policy_names():
        check()
        pieces=[read(closed(c,item['report']['control_results'][name]['predictions'])) for item in evidence]
        original=[value for item in evidence for value in item['child']['scope_expected_predictions'][name]]
        values=n20_merge_policy_predictions(pieces,ids,native,original)
        reader.population(coco,values,ids)
        if name=='native_official':require(values==native,'Complete original native prediction dictionaries exact')
        if name=='strong_fixed_E24_Omega':require(values==strong,'Complete original R prediction dictionaries exact')
        target=output/(name+'_PREDICTIONS.json');write(target,values)
        # COCO is a complete inline dict, never a strict physical ref.
        metrics=api(official,'coco_metrics')(coco,values,ids,COCOeval)
        result=dict(predictions=reference(target),COCO=metrics,stats=metrics['stats'])
        gpu.validate_legacy_COCO(result)
        if name in ('native_official','strong_fixed_E24_Omega'):
            require(result['stats']==original_results[name]['stats'] and result['COCO']==original_results[name]['COCO'],
                'Native/R all12 raw stats and original inline COCO GT/crowd/ignore/maxDets/no-filter semantics exact')
        results[name]=result;del values,pieces,original,metrics,result;gc.collect();check()
    preview=n20_final_main_preview(interpreter,results)
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source']
        and reference(spec['CPU_admission_file'])==gate['admission_ref']
        and torch.equal(outside,torch.get_rng_state())
        and gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)==libraries,
        'Final own complete Source/spec/admission/imports/CPU RNG exact')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        numeric_CPU_shards=spec['numeric_CPU_shards'],completed_GPU_shards=spec['completed_GPU_shards'],
        numeric_CPU_shard_gate_source=spec['numeric_CPU_shard_gate_source'],
        CPU_execution_partition_supplement=spec['CPU_execution_partition_supplement'],
        N20_root_training_qualified_result=spec['N20_root_training_qualified_result'],
        N20_locked_protocol=spec['N20_locked_protocol'],legacy_qualified_result=spec['legacy_qualified_result'],
        interpreter_source=spec['interpreter_source'],checkpoints=gate['trained_checkpoints'],
        N20_failed_training_record=spec['N20_failed_training_record'],
        N20_failed_training_stderr=spec['N20_failed_training_stderr'],
        N20_failed_DEV_capture_record=spec['N20_failed_DEV_capture_record'],
        N20_failed_DEV_capture_stderr=spec['N20_failed_DEV_capture_stderr'],
        N20_failed_CPU_numeric_record=spec['N20_failed_CPU_numeric_record'],
        N20_failed26_helper_source=spec['N20_failed26_helper_source'])
    manifests=list({item['child']['parent_GPU_report']['capture_manifest']['path']:
        item['child']['parent_GPU_report']['capture_manifest'] for item in evidence}.values())
    captures=[ref for item in evidence for ref in item['child']['scope_capture_refs']]
    require(len(manifests)==4 and len(captures)==len({ref['path'] for ref in captures})==187,
            'Exactly four original full manifests and187 own-scope actual physical bundles')
    details=dict(schema='N20-fixedDEV-full187-CPU-COCO-details-v1',**binding,
        control_results=results,image_proofs=proof_refs,capture_manifests=manifests,capture_refs=captures,
        registered_bundle_members_consumed=gate['registered_members'],actual_imported_files=libraries,
        numerical_forward_rerun=False,head_loaded=False,local_AP_averaged=False,
        original_training_GPU_actual0_verified=False,training_artifacts_recovery_qualified=True,failed26_partials_reused=False)
    write(output/'DETAILS.json',details)
    checks=dict(four_original_DEV_GPU0_thirty_two_CPU0_Source_spec_ad_phase_prefix_exact=True,
        all187_all100_registered_F9_FP64_score_numerical_imageproofs_complete=True,
        all_actual_GPU_logits_two_CPU_formulas_absolute1e4_and_exact_selected_PR_masks_RLE_preserved=True,
        all10_full18700_predictions_scores_classes_native_query_order_exact=True,
        all10_original_official_full_COCO12_semantics=True,native_R_all12stats_inline_COCO_and_full_predictions_exact=True,
        seed201_main_only_prelocked_interpreter_Source_exact=True,all26_failures_preserved_not_normal=True,
        original_training_GPU_exit1_and_independent_CPU_recovery_ancestry_preserved=True)
    write(output/'Q.json',dict(schema=N20_FINAL_Q_SCHEMA,status=N20_FINAL_STATUS,**binding,
        details=reference(output/'DETAILS.json'),images=187,GT=380,natural_queries=18700,
        policies=n20_final_policy_names(),control_results=results,initialization_seed=202610150201,
        normal_RAW_count=177,total_failed_RAW_count=26,total_terminal_identity_pin_count=203,
        primary_interpretation_preview_pending_ROOT_actual0=preview,checks=checks,
        qualified=False,ROOT_actual_CPU0_verified=False,current_owned_CPU_terminal_self_certified=False,
        numerical_forward_rerun=False,head_loaded=False,CUDA_initialized=False,local_AP_averaged=False,failed26_partials_reused=False,
        CPU_scout_forward_used_for_ranking=False,AP_results_interpreted_as_qualified=False,
        original_training_GPU_actual0_verified=False,training_artifacts_recovery_qualified=True,
        task_efficacy_established=False,innovation_established=False,TGRS_readiness_established=False,
        peak_observed_process_RSS_bytes=peak_RSS,minimum_observed_available_RAM_bytes=minimum_available_RAM,
        elapsed_seconds=time.monotonic()-started,
        evidence_scope='Internal fullDEV187 under frozen teacher/PR and source-qualified saved training recovery; first seed201/24 epochs only; pending original final CPU0 and separate ROOT post-actual checks, no TEST/generalization/innovation claim'))
    check()
    print(json.dumps(dict(status=N20_FINAL_STATUS,Q=reference(output/'Q.json'),qualified=False)),flush=True)
