# UNBOUND_DRAFT / UNEXECUTED: repeat202 thirty-two-scope DEV CPU numeric tail; no front/CLI.
"""ROOT's source_gate authenticates all four actual completed GPU shards first,
normally-qualified repeat202 training/cache/native/strong refs and the current own CPU.
Activation awaits all four new GPU shards and a new explicit CPU32 partition lock;
this draft neither reads nor certifies the currently active scientific GPU leaf.
Only ordinary source is prepared here. Importing this file performs no run.
"""
import gc
import json
import os
from pathlib import Path
import time

N20_CPU_RANGES = ((0,6),(6,12),(12,18),(18,24),(24,30),(30,36),(36,42),(42,47),
    (47,53),(53,59),(59,65),(65,71),(71,77),(77,83),(83,89),(89,94),
    (94,100),(100,106),(106,112),(112,118),(118,124),(124,130),(130,136),(136,141),
    (141,147),(147,153),(153,159),(159,165),(165,171),(171,177),(177,183),(183,187))
N20_CPU_PARENT_GPU_IDS = (0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3)
N20_CPU_GPU_RANGES = ((0,47),(47,94),(94,141),(141,187))
N20_CPU_THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
N20_CPU_REPORT_SCHEMA = 'N20-fixedDEV-feature-error-CPU-numeric-shard-report-v1'
N20_CPU_STATUS = 'N20_FIXEDDEV_FEATURE_ERROR_CPU_NUMERIC_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_FULL_COCO'


def n20_cpu_policy_names():
    return ('native_official','strong_fixed_E24_Omega') + tuple(
        'E%d_%s_%s' % (epoch,arm,mode) for epoch in (24,8)
        for arm in ('LOCAL_ERROR','CENTER_ERROR') for mode in ('atmost','exact'))


def n20_cpu_scope(spec, gate):
    """Verify front-provided immutable whole-parent and exact child metadata."""
    sid,pid=spec['cpu_shard_id'],spec['CPU_parent_GPU_id']
    require(type(sid) is int and 0<=sid<32 and type(pid) is int and pid==N20_CPU_PARENT_GPU_IDS[sid]
        and type(spec['normal_RAW_count']) is int and type(spec['qualification_owner_index']) is int
        and spec['qualification_owner_index']>=187
        and spec['normal_RAW_count']==spec['expected_normal_RAW_count']==spec['qualification_owner_index']
        and spec['cpu_shard_owner_index']==spec['normal_RAW_count']
        and spec['initialization_seed']==202610150202, 'Fixed CPU0..31 explicit repeat202 own index and parent mapping')
    first,last=N20_CPU_RANGES[sid];a,z=N20_CPU_GPU_RANGES[pid]
    require(spec['cpu_dataset_ordinal_range']==[first,last] and a<=first<last<=z,
            'Prelocked whole-image child range, no changing the parent partition')
    completed=gate['completed_GPU_shards']
    require(type(completed) is list and len(completed)==4 and all(
        type(item) is dict and type(item['shard_id']) is int and item['shard_id']==i
        and item['dataset_ordinal_range']==list(N20_CPU_GPU_RANGES[i])
        and type(item['shard_owner_index']) is int and 183<=item['shard_owner_index']<spec['qualification_owner_index']
        for i,item in enumerate(completed))
        and all(completed[i]['shard_owner_index']<completed[i+1]['shard_owner_index'] for i in range(3)),
        'All four explicit repeat202 GPU actual0 identities must precede own CPU and be joined by the front')
    require(spec['GPU_shard']==completed[pid], 'Keep original complete parent GPU descriptor verbatim')
    report,manifest=gate['parent_GPU_report'],gate['parent_GPU_manifest']
    require(report['schema']=='N20-fixedDEV-feature-error-GPU-shard-report-v1'
        and report['status']=='N20_FIXEDDEV_FEATURE_ERROR_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU'
        and report['shard_id']==pid and report['dataset_ordinal_range']==[a,z]
        and report['scope_dataset_indices']==list(range(a,z))
        and manifest['schema']=='N20-fixedDEV-feature-neighbors-GPU-shard-capture-manifest-v1'
        and manifest['dataset_ordinal_range']==[a,z] and manifest['shard_id']==pid
        and [row['dataset_index'] for row in manifest['rows']]==list(range(a,z))
        and report['capture_refs']==manifest['physical_capture_refs']==[row['bundle'] for row in manifest['rows']],
        'Complete actual original GPU parent report/manifest/capture population')
    rows=manifest['rows'][first-a:last-a]
    require(rows==gate['scope_rows'] and gate['scope_capture_refs']==[row['bundle'] for row in rows]
        and gate['scope_image_ids']==[row['image_id'] for row in rows]
        and len(set(gate['scope_image_ids']))==last-first and all(row['natural_queries']==100 for row in rows),
        'Own rows/physical refs/whole natural100 images are an exact immutable parent slice')
    require(tuple(report['controls'])==n20_cpu_policy_names()
        and set(gate['scope_expected_predictions'])==set(gate['scope_expected_prediction_groups'])==set(n20_cpu_policy_names()),
        'All ten fixed policies, no DEV winner substitution')
    for name in n20_cpu_policy_names():
        groups=gate['expected_prediction_groups'][name]
        expected=[value for row in rows for value in groups[row['image_id']]]
        require(expected==gate['scope_expected_predictions'][name] and len(expected)==(last-first)*100
            and gate['scope_expected_prediction_groups'][name]=={row['image_id']:groups[row['image_id']] for row in rows}
            and all([v['query_id'] for v in groups[row['image_id']]]==row['qids'] for row in rows),
            'Exact complete scoped original query/class/S0/prediction export order: '+name)
    return first,last


def n20_json_query_proof(value):
    return dict(value,profiles={name:{key:item for key,item in profile.items() if key!='reconstructed_mask'}
                               for name,profile in value['profiles'].items()})


def run(spec_path):
    started=time.monotonic()
    require(Path(spec_path).is_absolute(), 'Absolute independent N20 CPU specification')
    path=Path(spec_path).resolve(strict=True);spec=read(path);gate=source_gate(path,spec);c=gate['closure']
    first,last=n20_cpu_scope(spec,gate)
    output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir(),
            'Fresh owned CPU output with an existing parent')
    require(spec['deadline_seconds']==1800 and spec['automatic_retry'] is False
        and spec['maximum_process_RSS_bytes']==16*1024**3, 'Unchanged single1800s CPU leaf/no retry/RSS16GiB')
    source_keys=('GPU_outer_source','CPU_adapter_source','selector_cpu_proof_source','legacy_selector_proof_source',
                 'endpoint_body_source','PR_formula_source','labels_source','scout_module_source','N20_failed26_helper_source')
    for key in source_keys:closed(c,spec[key])
    closed(c,gate['prior_shared_validator_source'])
    for key in ('N20_repeat202_ROOT_training_qualification','N20_locked_protocol','legacy_qualified_result','N20_repeat202_CPU_execution_partition_lock'):
        closed(c,spec[key])
    import psutil
    process=psutil.Process()
    require(process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and all(os.environ.get(key)=='2' for key in N20_CPU_THREAD_ENV)
        and psutil.virtual_memory().available>=16*1024**3 and psutil.disk_usage(str(output.parent)).free>=1024**3,
        'Actual independent CPU01/emptyCUDA/two threads/minRAM16GiB/output free1GiB')
    import torch
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and cv2.__version__=='4.8.0'
        and not torch.cuda.is_initialized(), 'Original closed libraries/cold CUDA')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
    gpu=gate['gpu_helpers']
    require(Path(gpu.__file__).resolve(strict=True)==Path(spec['GPU_outer_source']['path']).resolve(strict=True)
        and callable(getattr(gpu,'n20_model_load_cpu',None)) and tuple(gpu.n20_policy_names())==n20_cpu_policy_names(),
        'Closed actual N20 GPU helper module; never call its GPU source_gate or run')
    adapter=load(spec['CPU_adapter_source'],'_N20_CPU_actual_F9_decoder')
    core=load(spec['selector_cpu_proof_source'],'_N20_CPU_actual_selected_numeric_core')
    endpoint=load(spec['endpoint_body_source'],'_N20_CPU_actual_independent_head_formula')
    legacy=load(spec['legacy_selector_proof_source'],'_N16_CPU_original_PR_numeric_kernel')
    manual=load(spec['PR_formula_source'],'_N20_CPU_original_PR_explicit_formula')
    helper=load(spec['labels_source'],'_N16_CPU_frozen_FP64_scoring')
    validator=load(gate['prior_shared_validator_source'],'_N17_CPU_actual_shared_validator')
    strong,models,frozen,model_proofs=gpu.n20_model_load_cpu(torch,spec,gate)
    require(core.ATOL==1e-4 and core.POINT_CHUNK==2048 and tuple(core.FORMULA_NAMES)==('original_head','independent_PR')
        and tuple(adapter.POLICIES)==n20_cpu_policy_names()[2:]
        and json.loads(json.dumps(model_proofs,allow_nan=False))==gate['parent_GPU_START']['model_proofs'],
        'New F9/five-tensor CPU core, fixed PR and original actual parent model proof bindings')
    states={key:frozen[key] for key in models};outside=torch.get_rng_state().clone()
    libraries=gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)
    peak_RSS=0;minimum_available_RAM=None
    def check():
        nonlocal peak_RSS,minimum_available_RAM
        rss=process.memory_info().rss;available=psutil.virtual_memory().available
        peak_RSS=max(peak_RSS,rss);minimum_available_RAM=available if minimum_available_RAM is None else min(minimum_available_RAM,available)
        require(time.monotonic()-started<1800 and rss<16*1024**3 and available>=16*1024**3
            and not torch.cuda.is_initialized() and process.cpu_affinity()==[0,1] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            and all(os.environ.get(key)=='2' for key in N20_CPU_THREAD_ENV)
            and torch.get_num_threads()==torch.get_num_interop_threads()==cv2.getNumThreads()==2,
            'Bounded independent CPU/coldCUDA/affinity/thread/deadline; no retry')
    def original_head(F,e,L):
        check();value=strong(F,e,L,L.new_zeros((len(L),49)),arm='PR');u=value['residual'];return u,L+u
    def independent_PR(F,e,L):
        check();return manual.independent_PR(torch,frozen['PR'],F,L)
    formulas=dict(original_head=original_head,independent_PR=independent_PR)
    def encode(mask):
        encoded=masks.encode(np.asfortranarray(mask.astype(np.uint8)))
        return dict(size=[int(v) for v in encoded['size']],counts=encoded['counts'].decode('ascii'))
    check();output.mkdir()
    all_predictions={name:[] for name in n20_cpu_policy_names()};image_proofs=[];used=set();member_count=0
    for row in gate['scope_rows']:
        check();index=row['dataset_index'];origin=gate['legacy_rows'][index];carrier=row['input_carrier']
        require(row['bundle']['path'] not in used,'Each new physical bundle is consumed once within this CPU scope')
        old_bundle=torch.load(str(closed(c,carrier['bundle'])),map_location='cpu')
        shared=validator.validate_actual_shared(torch,np,cv2,old_bundle,origin)
        bundle=torch.load(str(closed(c,row['bundle'])),map_location='cpu');used.add(row['bundle']['path'])
        native=gate['populations']['native_official'][row['image_id']]
        fixed=gate['populations']['strong_fixed_E24_Omega'][row['image_id']]
        require(native==gate['scope_expected_prediction_groups']['native_official'][row['image_id']]
            and fixed==gate['scope_expected_prediction_groups']['strong_fixed_E24_Omega'][row['image_id']],
            'Reuse only exact original qualified native/R predictions; no new baseline/GT/AP')
        decoder=adapter.decode_image(torch,np,shared,bundle,row,gate['parent_GPU_binding'],carrier,native);details=[]
        for q in range(100):
            check();kwargs=decoder.query(q)
            proof=core.proof_query(torch,np,cv2,helper,legacy,endpoint,heads=models,states=states,
                formulas=formulas,rle_encoder=encode,check_callback=check,**kwargs)
            require(proof['query_id']==q and all(proof[key] is True for key in ('all_selected_indices_exact','all_full_masks_and_RLE_exact',
                'actual_true_F9_or_center_inputs_byte_exact','actual_saved_FP64_scores_canonical_exact'))
                and proof['CPU_scout_outputs_used_for_ranking'] is False and proof['CPU_feature_rerender_equivalence_proven'] is False,
                'Actual natural-query F9/two-formula/saved-score/selection/PR/mask/RLE proof completed')
            details.append(n20_json_query_proof(proof));del proof,kwargs
        for name in n20_cpu_policy_names()[2:]:
            values=decoder.predictions(name)
            require(values==gate['scope_expected_prediction_groups'][name][row['image_id']],
                    'Exact actual GPU masks/RLE/classes/S0/query export order: '+name)
            all_predictions[name].extend(values)
        all_predictions['native_official'].extend(native);all_predictions['strong_fixed_E24_Omega'].extend(fixed)
        consumed=decoder.require_all_consumed();member_count+=consumed['members']
        target=output/('IMAGE_%03d_PROOF.json'%index)
        write(target,dict(dataset_index=index,image_id=row['image_id'],queries=100,bundle=row['bundle'],
            input_carrier=carrier,consumption=consumed,details=details));image_proofs.append(reference(target))
        del old_bundle,shared,bundle,decoder,details,native,fixed,consumed,values;gc.collect();check()
    require(used=={ref['path'] for ref in gate['scope_capture_refs']} and len(image_proofs)==last-first
        and all(len(values)==(last-first)*100 for values in all_predictions.values()),
        'Complete owned child scope/natural100/all8profiles/registered members consumed')
    outputs={}
    for name,values in all_predictions.items():
        require(values==gate['scope_expected_predictions'][name], 'Whole local original prediction sequence exact: '+name)
        target=output/(name+'_PREDICTIONS.json');write(target,values);outputs[name]=dict(predictions=reference(target))
    for key,head in [('PR',strong)]+list(models.items()):
        require(not head.training and all(not p.requires_grad and p.grad is None for p in head.parameters())
            and all(gpu.n20_same_bits(torch,value.detach().cpu(),frozen[key][name]) for name,value in head.state_dict().items()),
            'Original frozen PR and all four normally-qualified repeat202 five-tensor states unchanged')
    require(torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized(), 'External CPU RNG/CUDAcold unchanged')
    del head,strong,models,states,formulas;gc.collect()
    for ref in spec['sources']:check();bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source']
        and reference(spec['CPU_admission_file'])==gate['admission_ref']
        and gate['common'].imported_files(torch,np,cv2,masks,COCO,c,gate['admission'],ndimage)==libraries,
        'Final complete Source/currentCPU entry/spec/admission/actual libraries exact')
    gate['lineage'].final_endpoint_check(gate['parent']);check()
    binding=dict(source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        GPU_shard=spec['GPU_shard'],CPU_parent_GPU_id=spec['CPU_parent_GPU_id'],
        N20_failed_CPU_numeric_record=spec['N20_failed_CPU_numeric_record'],N20_failed26_helper_source=spec['N20_failed26_helper_source'],
        capture_manifest=gate['parent_GPU_report']['capture_manifest'],parent_GPU_binding=gate['parent_GPU_binding'],
        N20_repeat202_ROOT_training_qualification=spec['N20_repeat202_ROOT_training_qualification'],
        initialization_seed=202610150202,N20_locked_protocol=spec['N20_locked_protocol'],
        legacy_qualified_result=spec['legacy_qualified_result'],
        N20_repeat202_CPU_execution_partition_lock=spec['N20_repeat202_CPU_execution_partition_lock'],
        checkpoints=gate['trained_checkpoints'],CPU_adapter_source=spec['CPU_adapter_source'],
        selector_cpu_proof_source=spec['selector_cpu_proof_source'],legacy_selector_proof_source=spec['legacy_selector_proof_source'],
        endpoint_body_source=spec['endpoint_body_source'],PR_formula_source=spec['PR_formula_source'],labels_source=spec['labels_source'],
        scout_module_source=spec['scout_module_source'],GPU_outer_source=spec['GPU_outer_source'])
    write(output/'REPORT.json',dict(schema=N20_CPU_REPORT_SCHEMA,status=N20_CPU_STATUS,**binding,
        cpu_shard_id=spec['cpu_shard_id'],cpu_dataset_ordinal_range=[first,last],cpu_shard_owner_index=spec['normal_RAW_count'],
        normal_RAW_count=spec['normal_RAW_count'],total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
        scope_dataset_indices=list(range(first,last)),scope_image_ids=gate['scope_image_ids'],images=last-first,queries=(last-first)*100,
        global_images=187,global_queries=18700,global_GT=380,controls=n20_cpu_policy_names(),control_results=outputs,
        image_proofs=image_proofs,capture_refs=gate['scope_capture_refs'],actual_capture_files_consumed=len(used),
        registered_bundle_members_consumed=member_count,parent_original_DEV_GPU0_CUDA0_bound=True,
        all_actual_GPU_scout_logits_two_CPU_formulas_absolute1e4=True,all_actual_true_F9_or_center_inputs_byte_exact=True,
        all_actual_saved_FP64_scores_canonical_exact=True,all_selected_indices_masks_RLE_exact=True,
        all_PR_two_formulas_absolute1e4_strictsign=True,all_native_R_and_new_score_class_query_order_exact=True,
        independent_CPU_proof_complete=True,qualified=False,ROOT_actual_CPU0_verified=False,current_owned_CPU_terminal_self_certified=False,
        original_training_GPU_actual0_verified=True,training_saved_artifact_recovery=False,independent_endpoint_CPU_qualified=True,
        fixed_main_epoch=24,fixed_curve_epoch=8,DEV_winner_selected=False,CPU_scout_forward_used_for_ranking=False,
        CPU_F_rerendered=False,teacher_forward_calls=0,full_native_teacher_model_loaded=False,
        AP_evaluated=False,official_COCO_called=False,failed_partials_reused=False,failed26_partials_reused=False,CUDA_initialized=False,numeric_atol=1e-4,numeric_rtol=0.,
        actual_imported_files=libraries,peak_observed_process_RSS_bytes=peak_RSS,minimum_observed_available_RAM_bytes=minimum_available_RAM,
        elapsed_seconds=time.monotonic()-started,evidence_limits='One exact prelocked CPU child scope; combine all32 after originalactualCPU0 and Source checks for unified187COCO; no current CPUactual0 self-certification or CPU feature rerender equivalence'))
    print(json.dumps(dict(status=N20_CPU_STATUS,report=reference(output/'REPORT.json'),qualified=False,local_COCO_called=False)),flush=True)
