"""Unbound N10 fixed8 C/PR full-DEV evaluator SOURCE; ROOT alone admits execution.
Reuse independently qualified N07 F/e/M/class packets. Recompute full100 native
render/order/quality/S0 exactly before any residual; preserve verified scores.
No teacher forward, JPEG decode, GT-conditioned inference, training or TEST.
New GPU report is pending separate independent CPU evaluation qualification.
"""
import argparse
import ast
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

SCHEMA='ROOT-N10-C-PR-FL91-fixed8-fullDEV-GPU-spec-v1'
REPORT_SCHEMA='N10-C-PR-FL91-fixed8-fullDEV-GPU-report-v1'
STATUS='N10_C_PR_FL91_FIXED8_FULLDEV_GPU_COMPLETE_PENDING_CPU'
TRAIN_SCHEMA='ROOT-N10-C-PR-FL91-point-training-spec-v1'
TRAIN_REPORT_SCHEMA='N10-C-PR-FL91-point-training-report-v1'
TRAIN_STATUS='N10_C_PR_FL91_POINT_TRAINING_COMPLETE_PENDING_ROOT_QUALIFICATION'
TRAIN_CPU_SPEC_SCHEMA='ROOT-N10-C-PR-FL91-fixed8-training-CPU-spec-v1'
TRAIN_CPU_Q_SCHEMA='ROOT-N10-C-PR-FL91-fixed8-training-CPU-qualification-v1'
TRAIN_CPU_Q_STATUS='ROOT_N10_C_PR_FL91_FIXED8_TRAINING_CPU_QUALIFIED'
N07_REPORT_SCHEMA='N07-original-detail-fixed-endpoint-evaluation-report-v1'
N07_STATUS='N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_COMPLETE_PENDING_CPU'
N07_Q_SCHEMA='ROOT-N07-original-detail-fixed-endpoint-evaluation-CPU-qualification-v1'
N07_Q_STATUS='ROOT_N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
PACKET_SCHEMA='N07-original-detail-all100-native-fields-v1'
OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS=('receipt','phase','admission','phase_binding','bootstrap')
SEEDS=(202610120701,202610120702)
ARMS=('C','PR')
KEYS=tuple('%s_seed%d'%(arm,seed) for seed in SEEDS for arm in ARMS)
CONTROLS=('native_official',)+tuple(key+'_fixed_native_score' for key in KEYS)
REF_KEYS=('entry_source','head_training_entry_source','head_training_spec','head_training_report',
    'head_training_receipt','head_training_GPU_admission','head_training_CPU_qualification',
    'head_training_CPU_entry_source','head_training_CPU_spec','head_training_CPU_receipt','head_training_CPU_admission',
    'cache_entry_source','point_heads_source','PR_point_heads_source','prior_N07_report',
    'N07_cpu_qualification','native_predictions','DEV','reader_source','renderer_source','export_source','COCO_source')
REQUIRED_SPEC_KEYS=REF_KEYS+('schema','phase','sources','RAW_inputs','prior_failed_CPU_receipt',
    'additional_failed_RAW_inputs','GPU_admission_file','deadline_seconds','allocator_limit_MiB','whole_GPU_limit_MiB',
    'recipe','output_directory','head_training_phase','head_training_CPU_phase','head_endpoint_epoch','head_checkpoints')
RECIPE=dict(images=187,raw_GT=380,queries_per_image=100,predictions_per_control=18700,
    controls=list(CONTROLS),native_weights_epoch=24,head_endpoint_epoch=8,native_band_closed_abs_logit=2.,
    pixel_chunk=2048,feature_channel_chunk=16,render_dtype='float32',render_device='cuda:0',
    native_baseline_render_queries=100,native_baseline_quality_queries=100,
    native_baseline_render_order='original natural all100, then original query reorder',
    refinement_query_order='natural query IDs; e[0,q] and logits[q] stay aligned',
    patch_ABI='all-zero49; C internally zeros patch and PR never uses it',original_JPEG_decoded=False,
    fixed_native_scores=True,fixed_native_class=True,fixed_native_query_order=True,all100_queries_retained=True,
    no_NMS=True,no_score_cutoff=True,no_GT_query_selection=True,baseline_masks_and_scores_exact=True,
    baseline_tolerance=None,native_model_loaded=False,native_forward_calls=0,reuse_qualified_N07_packets=True,
    best_DEV_selection=False,no_TEST=True,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,CPU_affinity=[0,1],threads=2)

def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and
        len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),'Malformed absolute byte/SHA reference')


def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref); require(reference(ref['path'])==ref,'Actual pinned input differs: '+ref['path'])
    return Path(ref['path'])


def terminal(ref,phase,code=0):
    """Only original RAW JSON is read here: never hash/import a payload first."""
    ref_shape(ref); path=Path(ref['path'])
    require(path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,
        'Explicit original RAW/phase locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
        type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
        raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and
        raw['numeric_birth_absence']['original_birth_absent'] is True and raw.get('error') is None and
        raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
        'Original held/reaped terminal%d required before SHA/import'%code)
    if 'phase' in raw: require(raw['phase']==phase,'Original RAW phase differs')
    sha,peak=raw['admission_sha256'],raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and
        type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=8192 and
        type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB']==8192 and
        raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','Original RAW resource scope differs')
    return raw


def owner_join(row,raw,closure):
    """Join RAW/PB/bootstrap/owner without replaying an ancestor's payload tree."""
    owner,pb=read(bound(row['admission'])),read(bound(row['phase_binding']))
    bootstrap=bound(row['bootstrap']); runtime=Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path'])==runtime/'PHASE_BINDING.json' and bootstrap==runtime/'bootstrap.py' and
        owner['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['phase']==row['phase'] and
        owner['driver_sha256']==OWNER_SHA and owner['automatic_retry'] is False and
        owner['allowed_cpus']==[0,1] and owner['cuda_visible_devices'] in ('','0') and
        owner['whole_GPU_used_limit_MiB']==8192 and
        raw['admission_sha256']==row['admission']['sha256']==pb['admission_sha256'] and
        pb['bootstrap_sha256']==row['bootstrap']['sha256'] and pb['argv'][-1]==str(bootstrap) and
        pb['CPU_affinity']==owner['allowed_cpus'] and pb['CUDA_visible_devices']==owner['cuda_visible_devices'] and
        pb['automatic_retry'] is False,'Actual original RAW/owner/PB/bootstrap identity differs')
    entry,args=owner['entry'],owner['entry_args']
    require(entry in closure and owner['bound_source_sha256'].get(entry)==closure[entry]['sha256'],
        'Ancestor entry must remain actually sources-bound')
    tree=ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv=[ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(
        isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='sys' and t.attr=='argv'
        for t in node.targets)]
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and
        isinstance(node.func.value,ast.Name) and node.func.value.id=='runpy' and node.func.attr=='run_path']
    require(argv==[[entry]+args] and len(calls)==1 and ast.literal_eval(calls[0].args[0])==entry and
        {v.arg:ast.literal_eval(v.value) for v in calls[0].keywords}=={'run_name':'__main__'},'Actual ancestor bootstrap differs')
    if '--spec' in args:
        require(args.count('--spec')==1,'Ambiguous original SPEC argument')
        locator=args[args.index('--spec')+1]
        require(locator in closure and owner['bound_source_sha256'].get(locator)==closure[locator]['sha256'],
            'Actual ancestor SPEC must remain sources-bound')
    if 'scientific_spec' in owner:
        ref=owner['scientific_spec']
        require(ref in closure.values() and owner['bound_source_sha256'].get(ref['path'])==ref['sha256'],
            'Actual ancestor scientific SPEC differs')
    return owner


def load(ref,name):
    path=bound(ref); source=path.read_bytes()
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__); return module


def joined_phase(ref,source,spec_ref,admission,phase,cuda,rows,owners):
    require(all(v in ref.values() for v in (source,spec_ref,admission)),'Parent primary refs absent from actual closure')
    matches=[i for i,owner in enumerate(owners) if owner['entry']==source['path'] and
        owner['entry_args']==['--spec',spec_ref['path'],'--owner-execute']]
    require(len(matches)==1,'Exactly one actual successful parent execution required')
    i=matches[0]; row,owner=rows[i],owners[i]
    require(row['admission']==admission and row['phase']==phase and owner['scientific_spec']==spec_ref and
        owner['cuda_visible_devices']==cuda,'Parent report does not join its actual owned terminal')
    return i


def source_gate(path,spec):
    require(all(k in spec for k in REQUIRED_SPEC_KEYS) and spec['schema']==SCHEMA and
        type(spec['phase']) is str and spec['phase'] and type(spec['RAW_inputs']) is list and len(spec['RAW_inputs'])==31 and
        all(type(row) is dict and set(RAW_KEYS)<=set(row) for row in spec['RAW_inputs']),
        'Complete ROOT-authored fullDEV GPU SPEC required')
    rows=spec['RAW_inputs']; receipt_paths=[row['receipt']['path'] for row in rows]
    require(len(set(receipt_paths))==len(rows),'Normal ancestor RAW locators must be unique')
    raws=[terminal(row['receipt'],row['phase']) for row in rows]
    terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    failed_refs=[spec['prior_failed_CPU_receipt']]
    # ROOT may preserve other failed1 attempts explicitly, never in the normal list.
    require(type(spec['additional_failed_RAW_inputs']) is list and len(spec['additional_failed_RAW_inputs'])==2,'Both additional historical failures required')
    for row in spec['additional_failed_RAW_inputs']:
        require(type(row) is dict and set(row)=={'receipt','phase','returncode'} and
            type(row['returncode']) is int and row['returncode']==1,'Explicit historical failed1 locator required')
        terminal(row['receipt'],row['phase'],1); failed_refs.append(row['receipt'])
    require(len({r['path'] for r in failed_refs})==len(failed_refs) and
        not set(receipt_paths)&{r['path'] for r in failed_refs},'Failure cannot qualify a normal ancestor')
    # Every requested terminal gate is complete before the first byte hash/import.
    require(type(spec['sources']) is list and spec['sources'],'Complete byte/SHA source/input closure required')
    for ref in spec['sources']: ref_shape(ref)
    closure={ref['path']:ref for ref in spec['sources']}
    require(len(closure)==len(spec['sources']),'Unique source/input references required')
    for ref in spec['sources']: bound(ref)
    require(type(spec['head_checkpoints']) is dict and set(spec['head_checkpoints'])==set(KEYS),'Four fixed8 C/PR checkpoint references required')
    required=[spec[k] for k in REF_KEYS]+failed_refs+list(spec['head_checkpoints'].values())+[row[k] for row in rows for k in RAW_KEYS if k!='phase']
    require(all(ref in closure.values() for ref in required),'Actual primary inputs absent from sources')
    source,spec_ref=reference(__file__),reference(path)
    require(source==spec['entry_source'],'Actual N10 evaluation entry SOURCE differs')
    owners=[owner_join(row,raw,closure) for row,raw in zip(rows,raws)]
    admission_path=Path(spec['GPU_admission_file']); require(admission_path.is_absolute(),'Absolute actual own GPU admission required')
    admission_ref=reference(admission_path); admission=read(admission_path)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['driver_sha256']==OWNER_SHA and admission['phase']==spec['phase'] and
        admission['scientific_spec']==spec_ref and admission['entry']==source['path'] and
        admission['entry_args']==['--spec',str(path),'--owner-execute'] and admission['allowed_cpus']==[0,1] and
        admission['cuda_visible_devices']=='0' and admission['automatic_retry'] is False and
        admission['whole_GPU_used_limit_MiB']==8192 and type(spec['deadline_seconds']) is int and
        spec['deadline_seconds']>0 and admission['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and
        admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
        all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']),
        'Actual current ROOT owner/complete input closure differs')
    require(type(spec['allocator_limit_MiB']) is int and spec['allocator_limit_MiB']==4096 and
        type(spec['whole_GPU_limit_MiB']) is int and spec['whole_GPU_limit_MiB']==8192 and
        os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Actual fixed resource policy differs')
    require(spec['recipe']==RECIPE and spec['head_endpoint_epoch']==8,'Locked fixed8 evaluation recipe required')
    training=read(bound(spec['head_training_spec'])); tr=read(bound(spec['head_training_report']))
    train=load(spec['head_training_entry_source'],'_N10_fixed8_training_SOURCE')
    require(training['schema']==TRAIN_SCHEMA and training['entry_source']==spec['head_training_entry_source'] and
        training['RAW_inputs']==rows[:29] and training['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
        training['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'] and
        training['point_heads_source']==spec['point_heads_source'] and training['PR_point_heads_source']==spec['PR_point_heads_source'] and
        training['cache_entry_source']==spec['cache_entry_source'] and training['phase']==spec['head_training_phase'] and
        all(r in closure.values() for r in training['sources']) and train.SCHEMA==TRAIN_SCHEMA and tuple('%s_seed%d'%(arm,seed) for seed in train.SEEDS for arm in train.ARMS)==KEYS,
        'Actual fixed8 training SPEC and SOURCE closure differ')
    cache,cached,cache_report,cache_q,context,_,train_source,train_spec_ref,train_admission_ref=train.source_gate(Path(spec['head_training_spec']['path']),training)
    require(train_source==spec['head_training_entry_source'] and train_spec_ref==spec['head_training_spec'] and
        train_admission_ref==spec['head_training_GPU_admission'],'Actual qualified training provenance differs')
    require(tr['schema']==TRAIN_REPORT_SCHEMA and tr['status']==TRAIN_STATUS and tr['source']==spec['head_training_entry_source'] and
        tr['spec']==spec['head_training_spec'] and tr['GPU_admission']==spec['head_training_GPU_admission'] and
        tr['phase']==spec['head_training_phase'] and
        tr['point_heads_source']==spec['point_heads_source'] and tr['PR_point_heads_source']==spec['PR_point_heads_source'] and
        tr['head_CPU_qualification']==training['head_CPU_qualification'] and tr['recipe']==train.RECIPE and
        tr['completed_head_epochs']==8 and tr['continued_all4_to24'] is False and tr['all4_fixed_same_epochs'] is True and
        tr['paired_order_and_initialization'] is False and tr['paired_order'] is True and tr['paired_seed'] is True and
        tr['paired_cold_native_identity'] is True and tr['native_model_loaded'] is False and tr['native_forward_calls']==0 and
        tr['native_weights_updated'] is False and tr['cleanup']['allocated_after']==0 and tr['AP_computed'] is False and
        tr['no_DEV'] is True and tr['no_TEST'] is True and set(tr['checkpoints'])==set(KEYS) and
        tr['checkpoints']==tr['epoch8_checkpoints'],'Four independently qualified fixed8 endpoints required')
    for key in KEYS:
        row=tr['checkpoints'][key]
        require(row['checkpoint']==spec['head_checkpoints'][key] and row['completed_head_epochs']==8 and row['updates']==13288 and
            row['head_state_finite'] is True and row['no_pending_gradients'] is True and row['proof'] in closure.values(),
            'Fixed8 checkpoint/proof closure differs')
    require(all(tr[k] in closure.values() for k in ('start','trace','decision')),'Training records absent from closure')
    train_index=joined_phase(closure,tr['source'],tr['spec'],tr['GPU_admission'],tr['phase'],'0',rows,owners)
    require(train_index==29 and rows[29]['receipt']==spec['head_training_receipt'],'Actual held-terminal0 training GPU index differs')
    tq=read(bound(spec['head_training_CPU_qualification'])); ts=read(bound(spec['head_training_CPU_spec']))
    require(tq['schema']==TRAIN_CPU_Q_SCHEMA and tq['status']==TRAIN_CPU_Q_STATUS and
        tq['source']==spec['head_training_entry_source'] and tq['reader_source']==spec['head_training_CPU_entry_source'] and
        tq['spec']==spec['head_training_CPU_spec'] and tq['CPU_admission']==spec['head_training_CPU_admission'] and
        tq['training_spec']==spec['head_training_spec'] and tq['training_report']==spec['head_training_report'] and
        tq['training_GPU_RAW']==spec['head_training_receipt'] and tq['training_GPU_admission']==spec['head_training_GPU_admission'] and
        tq['point_heads_source']==spec['point_heads_source'] and tq['PR_point_heads_source']==spec['PR_point_heads_source'] and
        tq['head_CPU_qualification']==training['head_CPU_qualification'] and tq['checkpoints']==tr['checkpoints'] and tq['recipe']==train.RECIPE and
        tq['RAW_inputs']==rows[:30] and tq['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
        tq['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'] and tq['completed_head_epochs']==8 and
        tq['updates_per_model']==13288 and tq['C_parameters']==40257 and tq['PR_parameters']==40497 and
        tq['parameter_tensors_by_arm']==dict(C=6,PR=8) and tq['C_PR_identical_weight_initialization_claimed'] is False and
        tq['CPU_affinity']==[0,1] and tq['threads']==2 and tq['CUDA_initialized'] is False and
        all(tq[k] is True for k in ('per_arm_cold_initializer_proofs_CPU_checked',
            'private_image_order_and_endpoint_RNG_CPU_reconstructed','AdamW_endpoint_states_and_stepcounts_CPU_checked',
            'all16_epoch_records_CPU_checked','fixed8_no_extension_decision_CPU_checked','final_fullFIT_endpoint_BCE_CPU_recomputed')),
        'Actual independent fixed8 training CPU qualification required')
    require(ts['schema']==TRAIN_CPU_SPEC_SCHEMA and ts['entry_source']==spec['head_training_CPU_entry_source'] and
        ts['training_entry_source']==spec['head_training_entry_source'] and ts['training_spec']==spec['head_training_spec'] and
        ts['training_report']==spec['head_training_report'] and ts['training_GPU_RAW']==spec['head_training_receipt'] and
        ts['training_GPU_admission']==spec['head_training_GPU_admission'] and ts['RAW_inputs']==rows[:30] and
        ts['point_heads_source']==spec['point_heads_source'] and ts['PR_point_heads_source']==spec['PR_point_heads_source'] and
        ts['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and ts['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'] and
        ts['CPU_admission_file']==spec['head_training_CPU_admission']['path'] and all(r in closure.values() for r in ts['sources']),
        'Complete training CPU SPEC/source closure required')
    cpu_training_index=joined_phase(closure,tq['reader_source'],tq['spec'],tq['CPU_admission'],tq['phase'],'',rows,owners)
    require(cpu_training_index==30 and rows[30]['receipt']==spec['head_training_CPU_receipt'] and
        tq['phase']==ts['phase']==spec['head_training_CPU_phase'],'Actual held-terminal0 training CPU index differs')
    for key in KEYS:
        ep=tq['final_endpoints'][key]
        require(ep['instances']==1661 and ep['points']==1700864 and all(type(ep[k]) in (int,float) and math.isfinite(ep[k]) for k in
            ('CPU_mean_BCE','GPU_recorded_mean_BCE','absolute_difference')) and ep['absolute_difference']>=0 and
            ep['GPU_recorded_mean_BCE']==tr['checkpoints'][key]['mean_endpoint_BCE'], 'Qualified fullFIT endpoint BCE metadata differs')
    report,q=read(bound(spec['prior_N07_report'])),read(bound(spec['N07_cpu_qualification']))
    require(report['schema']==N07_REPORT_SCHEMA and report['status']==N07_STATUS and
        q['schema']==N07_Q_SCHEMA and q['status']==N07_Q_STATUS and
        q['evaluation_report']==spec['prior_N07_report'] and q['source']==report['source'] and
        q['evaluation_spec']==report['spec'] and q['evaluation_GPU_admission']==report['GPU_admission'] and
        q['reader_source']==spec['reader_source'] and report['DEV']==q['DEV']==spec['DEV'] and
        report['images']==q['images']==187 and report['raw_GT']==q['raw_GT']==380 and
        report['predictions_per_control']==q['predictions_per_control']==18700 and
        report['head_endpoint_epoch']==q['head_endpoint_epoch']==8 and
        report['native_forwards']==187 and report['native_weights_updated'] is False and
        report['heads_updated'] is False and report['matching_or_loss_executed'] is False and
        report['original187_all100_low_fields_saved'] is True and report['GPU_original_logits_saved'] is False and
        report['cleanup']['allocated_after']==0 and report['no_TEST'] is True and report['no_best_DEV_selection'] is True and
        q['complete_identity_fixed_scores_and_empty_query_population_CPU_checked'] is True and
        q['five_official_COCO12_CPU_recomputed'] is True and q['strong_E24_native_export_and_COCO_reference_CPU_checked'] is True and
        q['native_M200_original_render_CPU_replayed'] is True and q['no_NMS'] is True and
        q['no_score_cutoff'] is True and q['no_GT_query_selection'] is True and q['no_TEST'] is True,
        'Qualified original full187/380/all100 fixed8 N07 parents required')
    gpu_index=joined_phase(closure,report['source'],report['spec'],report['GPU_admission'],report['phase'],'0',rows,owners)
    cpu_index=joined_phase(closure,q['reader_source'],q['spec'],q['CPU_admission'],q['phase'],'',rows,owners)
    require(q['evaluation_GPU_RAW']==rows[gpu_index]['receipt'] and gpu_index<cpu_index<train_index and
        all(row in rows[:cpu_index] for row in q['RAW_inputs']) and
        q['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'],'Actual N07 GPU/CPU/training chronology differs')
    n07_spec=read(bound(report['spec']))
    reader_spec=read(bound(q['spec']))
    require(n07_spec['official_export_source']==spec['export_source'] and
        n07_spec['official_evaluation_CPU_reader']==spec['COCO_source'] and
        reader_spec['renderer_source']==spec['renderer_source'],'Actual original exporter/COCO/renderer SOURCE binding differs')
    packets=report['final_field_packets']
    require(type(packets) is list and len(packets)==187 and packets==q['packet_refs'] and
        len({ref['path'] for ref in packets})==187 and all(ref in closure.values() for ref in packets) and
        report['image_rows'] in closure.values() and report['trace'] in closure.values() and
        report['teacher_checkpoint']==tr['teacher_checkpoint'] and
        report['predictions']['native_official']==spec['native_predictions']==q['control_results']['native_official']['predictions'],
        'Actual archived packet/image-row/native-export inputs differ')
    return dict(closure=closure,source=source,spec_ref=spec_ref,admission_ref=admission_ref,admission=admission,
        train=train,training=training,training_report=tr,training_q=tq,cache=cache,cached=cached,context=context,
        report=report,q=q,packets=packets,owners=owners,failed_refs=failed_refs)


def original_export_kernels(ref):
    """Execute only audited original exporter expressions, without a fusion/model.

    The query-order and S0 expressions are copied as AST nodes from the actually
    bound N07 exporter, rather than inferred from a different postprocessor.
    Natural rendering and quality both use all100; every native value must be exact.
    """
    path=bound(ref); tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export_original']
    require(len(functions)==1,'Actual N07 official exporter function required')
    fragments=("probabilities = torch.nn.functional.softmax(classes[0], dim=-1)[:, :-1]",
        "top, indices = probabilities.flatten(0, 1).topk(100, sorted=False)",
        "qids = indices // fusion.num_classes", "binary = ordered > 0",
        "quality = (ordered.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)",
        "scores = top * quality")
    assignments=[]
    for text in fragments:
        expected=ast.parse(text).body[0]; signature=ast.dump(expected,include_attributes=False)
        matches=[n for n in functions[0].body if isinstance(n,ast.Assign) and ast.dump(n,include_attributes=False)==signature]
        require(len(matches)==1,'Original exporter expression ABI differs: '+text); assignments.append(copy.deepcopy(matches[0]))
    module=ast.parse('def order(torch, classes, fusion):\n    pass\ndef quality(torch, ordered, top):\n    pass\n')
    module.body[0].body=assignments[:3]+ast.parse('return top, qids').body
    module.body[1].body=assignments[3:]+ast.parse('return binary, quality, scores').body
    ast.fix_missing_locations(module); scope={}
    exec(compile(module,str(path),'exec'),scope)
    return types.SimpleNamespace(order=scope['order'],quality=scope['quality'])


def imported_files(torch,np,COCO,COCOeval,masks,gate):
    modules=dict(torch=torch,torch_binary=torch._C,numpy=np,numpy_binary=np.core._multiarray_umath,
        COCO=sys.modules[COCO.__module__],COCOeval=sys.modules[COCOeval.__module__],masks=masks,mask_binary=masks._mask)
    result={name:reference(module.__file__) for name,module in modules.items()}
    result['interpolation']=reference(inspect.getsourcefile(torch.nn.functional.interpolate))
    require(all(ref in gate['closure'].values() and
        gate['admission']['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in result.values()),
        'Actual imported library/binary/math SOURCE absent or unbound')
    return result


def segmentation(np,masks,binary):
    require(binary.dtype==np.uint8 and binary.ndim==2,'Original uint8 binary raster required')
    value=masks.encode(np.asfortranarray(binary)); counts=value['counts']
    return dict(size=list(value['size']),counts=counts.decode('ascii') if isinstance(counts,bytes) else counts)


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def stats_exact(actual,expected,message):
    require(type(actual) is list and type(expected) is list and len(actual)==len(expected)==12 and
        all(type(v) in (int,float) and math.isfinite(v) and (v==-1 or 0<=v<=1) for v in actual+expected) and
        actual==expected,message)


def head_endpoints(torch,factories,train,spec,report,closure):
    """CPU state/moment/proof and FIT endpoint consistency, not optimizer replay."""
    heads={}; states={}; cpu_rng=torch.random.get_rng_state().clone()
    start=read(bound(report['start'])); decision=read(bound(report['decision']))
    require(start['recipe']==train.RECIPE and start['source']==spec['head_training_entry_source'] and start['spec']==spec['head_training_spec'] and
            start['GPU_admission']==spec['head_training_GPU_admission'] and start['phase']==spec['head_training_phase'] and
            start['fresh_optimizer_state_entries_per_model']==dict.fromkeys(KEYS,0) and
            decision['rule']==train.RECIPE['extension_rule'] and decision['continue_all4']==report['continued_all4_to24'] and
            decision['fixed_final_epoch']==8 and decision['continue_all4'] is False and decision['DEV_accessed'] is False and
            decision['TEST_accessed'] is False and decision['best_epoch_selection'] is False and decision['automatic_extension'] is False and
            start['point_heads_source']==spec['point_heads_source'] and start['PR_point_heads_source']==spec['PR_point_heads_source'] and
            start['head_CPU_qualification']==report['head_CPU_qualification'] and start['paired_initialization'] is False and
            start['paired_seed'] is True and start['paired_cold_native_identity'] is True and start['paired_image_order'] is True,
            'Actual paired START/continuation decision differs')
    trace=[json.loads(line) for line in bound(report['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(trace)==2*spec['head_endpoint_epoch'] and len({(r['seed'],r['epoch']) for r in trace})==len(trace),'Complete paired epoch trace required')
    histories=report['endpoint_BCE_histories']; cp_by_key={}
    for key in KEYS:
        arm,seed_text=key.split('_seed',1); seed=int(seed_text); cp=torch.load(str(bound(spec['head_checkpoints'][key])),map_location='cpu'); meta=cp['meta']
        names,shapes,parameters=train.NAMES[arm],train.SHAPES[arm],train.PARAMETERS[arm]
        state=cp['state_dict']; proof=read(bound(report['checkpoints'][key]['proof']))
        require(meta['schema']==train.CP_SCHEMA and meta['source']==spec['head_training_entry_source'] and meta['spec']==spec['head_training_spec'] and
                meta['GPU_admission']==spec['head_training_GPU_admission'] and meta['phase']==spec['head_training_phase'] and meta['arm']==arm and meta['seed']==seed and
                meta['point_heads_source']==spec['point_heads_source'] and meta['PR_point_heads_source']==spec['PR_point_heads_source'] and
                meta['head_factory_source']==spec[train.FACTORY_SPEC_KEYS[arm]] and
                meta['head_CPU_qualification']==report['head_CPU_qualification'] and meta['cache_spec']==report['cache_spec'] and meta['cache_report']==report['cache_report'] and
                meta['cache_CPU_qualification']==report['cache_CPU_qualification'] and meta['teacher_checkpoint']==report['teacher_checkpoint'] and
                meta['native_weights_epoch']==24 and meta['native_weights_updated'] is False and meta['recipe']==train.RECIPE and
                meta['completed_head_epochs']==spec['head_endpoint_epoch'] and meta['updates']==meta['GT_visits']==spec['head_endpoint_epoch']*1661 and
                meta['image_visits']==spec['head_endpoint_epoch']*741 and meta['point_visits']==meta['updates']*1024 and
                meta['parameter_tensors']==len(names) and meta['parameter_scalars']==parameters and meta['no_pending_gradients'] is True,
                'Own per-arm fixed8 trained-head metadata differs')
        require(tuple(state)==names and all(isinstance(state[n],torch.Tensor) and state[n].device.type=='cpu' and state[n].dtype==torch.float32 and
                tuple(state[n].shape)==shape and bool(torch.isfinite(state[n]).all()) for n,shape in zip(names,shapes)), 'Own finite CPU head state differs')
        require(proof['checkpoint']==spec['head_checkpoints'][key] and proof['updates']==meta['updates'] and
                proof['parameter_tensors']==len(names) and proof['parameter_scalars']==parameters and proof['full_readback_exact'] is True and
                proof['state']=={n:train.digest(torch,v) for n,v in state.items()} and
                proof['private_order_state']==train.digest(torch,cp['private_order_state']),'Own saved state/order proof differs')
        opt=cp['optimizer_state_dict']; require(len(opt['param_groups'])==1 and len(opt['state'])==len(names),'Actual per-arm moment entries required')
        group=opt['param_groups'][0]
        require(group['lr']==3e-4 and group['weight_decay']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and
                group['amsgrad'] is False and len(group['params'])==len(names),'Own constant AdamW recipe differs')
        for name,shape,identifier in zip(names,shapes,group['params']):
            entry=opt['state'][identifier]
            require(set(entry)=={'step','exp_avg','exp_avg_sq'} and isinstance(entry['step'],torch.Tensor) and
                    entry['step'].device.type=='cpu' and entry['step'].dtype==torch.float32 and tuple(entry['step'].shape)==() and
                    float(entry['step'])==meta['updates'] and
                    all(entry[k].device.type=='cpu' and tuple(entry[k].shape)==shape and entry[k].dtype==torch.float32 and bool(torch.isfinite(entry[k]).all()) for k in ('exp_avg','exp_avg_sq')) and
                    proof['optimizer'][name]==dict({k:train.digest(torch,v) for k,v in entry.items()},updates=meta['updates']), 'Own AdamW moment proof differs')
        history=meta['endpoint_BCE_history']; require(history==histories[key] and len(history)==spec['head_endpoint_epoch'] and
                [row['epoch'] for row in history]==list(range(1,spec['head_endpoint_epoch']+1)) and
                all(type(row['BCE']) in (int,float) and math.isfinite(row['BCE']) and row['BCE']>=0 for row in history),'Complete FIT BCE history differs')
        for epoch in range(1,spec['head_endpoint_epoch']+1):
            row=[r for r in trace if r['seed']==seed and r['epoch']==epoch]
            require(len(row)==1 and row[0]['full_FIT_endpoint_BCE'][arm]==history[epoch-1]['BCE'] and row[0]['GT_instances']==1661 and
                    row[0]['images']==741 and row[0]['updates_per_model']==1661 and row[0]['point_visits_per_model']==1661*1024,'Full FIT endpoint trace differs')
        head=factories[arm].make_point_head(torch,seed)
        require({n:train.digest(torch,v) for n,v in head.state_dict().items()}==start['initial_head_state_proofs'][key],'Actual paired cold initializer differs')
        head.load_state_dict(state,strict=True); train.check_head(torch,head,arm); head.requires_grad_(False); head.eval()
        heads[key]=head; states[key]=state; cp_by_key[key]=cp
    for seed in SEEDS:
        left,right=cp_by_key['C_seed%d'%seed],cp_by_key['PR_seed%d'%seed]
        require(left['meta']['epoch_orders']==right['meta']['epoch_orders'] and torch.equal(left['private_order_state'],right['private_order_state']),
                'Paired complete image-order/RNG endpoint differs')
        rng=torch.Generator(device='cpu'); rng.manual_seed(seed)
        for epoch,row in enumerate(left['meta']['epoch_orders'],1):
            order=torch.randperm(741,generator=rng).tolist()
            require(row==dict(epoch=epoch,indices=order,sha256=hashlib.sha256(json.dumps(order,separators=(',',':')).encode('ascii')).hexdigest()),
                    'Independent private image-order reconstruction differs')
        require(len(left['meta']['epoch_orders'])==spec['head_endpoint_epoch'] and torch.equal(rng.get_state(),left['private_order_state']),'Own final order state differs')
    require(torch.equal(cpu_rng,torch.random.get_rng_state()) and not torch.cuda.is_initialized(),'CPU checkpoint loading changed external RNG/CUDA')
    return heads,states


def packet_check(torch,reader,coco,packet,row,index,ref,gate):
    report=gate['report']; iid=row['image_id']; identity=row['identity']; image=coco.imgs[iid]
    require(row['dataset_index']==index and row['packet']==ref and packet['schema']==PACKET_SCHEMA and
        packet['dataset_index']==index and packet['identity']==identity and packet['source']==report['source'] and
        packet['spec']==report['spec'] and packet['phase']==report['phase'] and
        packet['teacher_checkpoint']==report['teacher_checkpoint'] and packet['head_checkpoints']==report['head_checkpoints'] and
        packet['head_endpoint_epoch']==8 and packet['GPU_original_logits_saved'] is False and
        packet['prediction_query_ids']==row['prediction_query_ids'] and packet['field_digests']==row['final_fields'],
        'Actual archived all100 field packet provenance differs')
    H,W=image['height'],image['width']; h,w=identity['actual_img_shape']
    require(identity['image_id']==iid and identity['original_shape']==[H,W] and type(h) is int and type(w) is int and
        0<h<=800 and 0<w<=800 and identity['batch_input_shape']==[800,800] and
        identity['ann_ids']==row['ann_ids']==[a['id'] for a in coco.imgToAnns[iid]] and
        row['raw_GT']==len(identity['ann_ids']),'Canonical DEV original/crop/annotation metadata differs')
    qids=packet['prediction_query_ids']
    require(len(qids)==100 and all(type(q) is int for q in qids) and sorted(qids)==list(range(100)),
        'All100 exact archived query identities required')
    fields=packet['native_fields']
    require(set(fields)==set(packet['field_digests'])=={'F256','e0','M200','final_class_logits'},
        'Original four-field packet ABI differs')
    for name,shape in (('F256',(1,256,200,200)),('e0',(1,100,256)),('M200',(1,100,200,200)),('final_class_logits',(1,100,2))):
        value=fields[name]
        require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and
            tuple(value.shape)==shape and not value.requires_grad and bool(torch.isfinite(value).all()) and
            reader.digest(value)==packet['field_digests'][name],'Finite original CPU mask/class field differs: '+name)
    require(dict(reader.digest(fields['final_class_logits'][0]),values=fields['final_class_logits'][0].tolist())==row['native_class_packet'],
        'Archived original class values/digest differ')
    require(type(row['band_counts_all100']) is list and len(row['band_counts_all100'])==100 and
        all(type(v) is int and 0<=v<=H*W for v in row['band_counts_all100']),'Original all100 closed-band counts required')
    require(row['head_seed']==100000+iid and row['matching_or_loss_executed'] is False and
        all(row[k] is True for k in ('GT_removed_from_forward','loader_accesses_annotations','native610_SOURCE_unchanged',
            'native_all10_versions_unchanged','no_GT_query_selection','all100_queries_retained','no_NMS','no_score_cutoff',
            'fixed_native_score','new_empty_geometry_may_keep_positive_native_score','external_RNG_restored')),
        'Original producer inference scope differs')
    return fields,identity,qids


def original_feature(torch,renderer,F,identity):
    H,W=identity['original_shape']; original=F.new_empty((256,H,W))
    for first in range(0,256,16):
        values=renderer.render_original(torch,F[:,first:first+16],identity)
        require(tuple(values.shape)==(16,H,W) and values.dtype==torch.float32 and bool(torch.isfinite(values).all()),
            'Original coarse-feature projection must keep channel16 FP32 renderer recipe')
        original[first:first+16]=values
    return original


def refine(torch,head,arm,F_original,e,L,spec,started):
    refined=L.clone(); counts=[]; calls=0; feature_points=F_original.flatten(1)
    for q in range(100):
        indices=torch.nonzero((L[q].abs()<=2.).flatten(),as_tuple=False).flatten(); counts.append(int(indices.numel()))
        for first in range(0,indices.numel(),2048):
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked point refinement deadline exceeded')
            idx=indices[first:first+2048]; native=L[q].flatten()[idx]
            result=head(feature_points[:,idx].T,e[0,q].expand(idx.numel(),256),native,
                L.new_zeros((idx.numel(),49)),arm=arm)
            require(result['logits'].dtype==torch.float32 and result['logits'].device==L.device and
                tuple(result['logits'].shape)==tuple(native.shape) and bool(result['band'].all()) and
                bool(torch.isfinite(result['logits']).all()),'Finite native-band per-query point-head output required')
            refined[q].flatten()[idx]=result['logits']; calls+=1
    band=L.abs()<=2.
    require(torch.equal(refined[~band],L[~band]),'Point head changed logits outside original closed band')
    return refined,counts,calls

def fixed_score_rows(np,masks,native,logits):
    rows=copy.deepcopy(native)
    for row in rows:
        q=row['query_id']; binary=(logits[q]>0).cpu().numpy().astype(np.uint8)
        rle=masks.encode(np.asfortranarray(binary)); counts=rle['counts'].decode('ascii') if isinstance(rle['counts'],bytes) else rle['counts']
        row['segmentation']=dict(size=list(rle['size']),counts=counts)
    require(all({k:v for k,v in left.items() if k!='segmentation'}=={k:v for k,v in right.items() if k!='segmentation'} for left,right in zip(native,rows)),
            'Only geometry may change; no empty-mask score rezero/filter/reorder')
    return rows


def evaluate_image(torch,np,masks,renderer,kernels,reader,coco,packet,row,index,ref,native,heads,gate,spec,started):
    fields,identity,archived_qids=packet_check(torch,reader,coco,packet,row,index,ref,gate)
    require([r['query_id'] for r in native]==archived_qids,'Exact original native export order differs')
    F=e=M=C=top=qids=L=ordered=binary=quality=scores=F_original=refined=before=after=None
    baseline=[]; predictions={}; control_diagnostics={}; band_counts=None
    try:
        with torch.no_grad():
            F=fields['F256'].to('cuda:0'); e=fields['e0'].to('cuda:0')
            M=fields['M200'].to('cuda:0'); C=fields['final_class_logits'].to('cuda:0')
            require(torch.equal(torch.einsum('bqc,bchw->bqhw',e,F),M),'Qualified archived native M=eF projection must reproduce exactly')
            top,qids=kernels.order(torch,C,types.SimpleNamespace(num_classes=1))
            require(qids.cpu().tolist()==archived_qids,'Original CUDA topk100 sorted=False order changed')
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked native all100 baseline deadline exceeded')
            L=renderer.render_original(torch,M,identity)
            require(L.device.type=='cuda' and L.dtype==torch.float32 and tuple(L.shape)==(100,)+tuple(identity['original_shape']) and
                bool(torch.isfinite(L).all()),'Original natural all100 FP32 renderer differs')
            ordered=L[qids]; binary,quality,scores=kernels.quality(torch,ordered,top)
            native_scores=scores.cpu().tolist(); native_masks=binary.cpu().numpy().astype(np.uint8)
            for ordinal in range(100):
                recomputed=dict(image_id=identity['image_id'],query_id=archived_qids[ordinal],category_id=1,
                    score=float(native_scores[ordinal]),segmentation=segmentation(np,masks,native_masks[ordinal]))
                require(recomputed==native[ordinal],'Native query/mask/class/order/S0 must reproduce exactly; no tolerance/substitution')
                if not native_masks[ordinal].any(): require(recomputed['score']==0.,'Original empty-support S0 must stay zero')
                baseline.append(recomputed)
            require(baseline==native,'Complete all100 native baseline exact gate required before any residual')
            # Keep natural L: logits[q] and e[0,q] share query IDs. Ordered is baseline-only.
            M=C=top=qids=ordered=binary=quality=scores=None
            F_original=original_feature(torch,renderer,F,identity); F=None
            for key in KEYS:
                arm=key.split('_seed',1)[0]; control=key+'_fixed_native_score'
                refined,counts,calls=refine(torch,heads[key],arm,F_original,e,L,spec,started)
                require(counts==row['band_counts_all100'],'Original natural all100 closed-band population changed')
                if band_counts is None: band_counts=counts
                require(counts==band_counts,'C/PR original band populations must be identical')
                exported=fixed_score_rows(np,masks,baseline,refined); predictions[control]=exported; per_query=[]
                for ordinal,original in enumerate(baseline):
                    q=original['query_id']; before=native_masks[ordinal].astype(bool)
                    after=(refined[q]>0).cpu().numpy().astype(bool)
                    added=int(np.count_nonzero(~before & after)); removed=int(np.count_nonzero(before & ~after))
                    per_query.append(dict(query_id=q,native_export_ordinal=ordinal,fixed_native_score=original['score'],
                        band_pixels=counts[q],native_foreground_pixels=int(before.sum()),refined_foreground_pixels=int(after.sum()),
                        added_pixels=added,removed_pixels=removed,changed_pixels=added+removed,
                        native_empty=not bool(before.any()),refined_empty=not bool(after.any()),
                        native_empty_new_nonempty_keeps_zero_score=not bool(before.any()) and bool(after.any()),
                        new_empty_keeps_positive_native_score=bool(before.any()) and not bool(after.any()) and original['score']>0))
                control_diagnostics[control]=dict(arm=arm,seed=int(key.split('_seed',1)[1]),head_point_chunk_calls=calls,
                    query_diagnostics=per_query,**{name:sum(r[name] for r in per_query) for name in
                        ('added_pixels','removed_pixels','changed_pixels','native_empty_new_nonempty_keeps_zero_score',
                         'new_empty_keeps_positive_native_score')},
                    changed_queries=sum(r['changed_pixels']>0 for r in per_query),
                    expanded_queries=sum(r['added_pixels']>0 for r in per_query),shrunk_queries=sum(r['removed_pixels']>0 for r in per_query),
                    native_empty_queries=sum(r['native_empty'] for r in per_query),refined_empty_queries=sum(r['refined_empty'] for r in per_query))
                refined=before=after=None
        require(all(all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'}
            for a,b in zip(baseline,values)) for values in predictions.values()),'Only segmentation geometry may change')
        summary=dict(dataset_index=index,image_id=identity['image_id'],packet=ref,queries=100,native_query_order=archived_qids,
            native_M_equals_eF_exact=True,native_masks_scores_and_classes_exact=True,
            fixed_native_scores_and_class_and_order_exact=True,original_band_only=True,band_counts_natural_query_order=band_counts,
            controls=control_diagnostics,native_forward_calls=0,native_model_loaded=False,original_JPEG_decoded=False,
            GT_used_to_choose_queries_or_residual=False)
        return baseline,predictions,summary
    finally:
        F=e=M=C=top=qids=L=ordered=binary=quality=scores=F_original=refined=before=after=None


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute ROOT-authored SPEC required')
    path=path.resolve(strict=True); spec=read(path); gate=source_gate(path,spec)
    output=Path(spec['output_directory']); require(output.is_absolute() and not output.exists(),'Fresh absolute output directory required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 required before scientific imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Qualified Torch and cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    train=gate['train']; factories={arm:load(spec[train.FACTORY_SPEC_KEYS[arm]],'_N10_evaluation_'+arm+'_factory') for arm in ARMS}
    heads,head_states=head_endpoints(torch,factories,train,spec,gate['training_report'],gate['closure'])
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5','Qualified original NumPy required')
    libraries=imported_files(torch,np,COCO,COCOeval,masks,gate)
    reader=load(spec['reader_source'],'_N10_old_N07_pure_reader')
    renderer=load(spec['renderer_source'],'_N10_actual_original_renderer')
    official=load(spec['COCO_source'],'_N10_actual_original_COCO')
    require(reader.Q_SCHEMA==N07_Q_SCHEMA and reader.STATUS==N07_Q_STATUS,'Qualified old N07 pure-reader ABI differs')
    kernels=original_export_kernels(spec['export_source'])
    coco=COCO(str(bound(spec['DEV']))); ids=list(coco.imgs)
    require(len(ids)==len(set(ids))==187 and len(coco.anns)==380 and set(coco.cats)=={1},'Full canonical DEV187/380/category1 required')
    native=read(bound(spec['native_predictions'])); grouped=reader.population(coco,native,ids)
    image_rows=read(bound(gate['report']['image_rows']))
    old_trace=[json.loads(line) for line in bound(gate['report']['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(type(image_rows) is list and image_rows==old_trace and len(image_rows)==187 and [r['image_id'] for r in image_rows]==ids and
        sum(r['raw_GT'] for r in image_rows)==380,'Exact canonical archived DEV image rows/trace required')
    require(not torch.cuda.is_initialized(),'Input/checkpoint/library qualification must precede CUDA initialization')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    require(torch.cuda.memory_allocated(0)==0,'Clean GPU allocator required before archived-field evaluation')
    torch.cuda.reset_peak_memory_stats(0)
    for head in heads.values(): head.to('cuda:0')
    output.mkdir()
    state_proofs={key:{name:train.digest(torch,value) for name,value in state.items()} for key,state in head_states.items()}
    write(output/'START.json',dict(schema='N10-C-PR-FL91-fixed8-fullDEV-GPU-start-v1',phase=spec['phase'],
        source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],recipe=RECIPE,
        head_training_report=spec['head_training_report'],head_training_CPU_qualification=spec['head_training_CPU_qualification'],
        head_checkpoints=spec['head_checkpoints'],head_state_proofs=state_proofs,
        point_heads_source=spec['point_heads_source'],PR_point_heads_source=spec['PR_point_heads_source'],
        prior_N07_report=spec['prior_N07_report'],N07_cpu_qualification=spec['N07_cpu_qualification'],
        native_predictions=spec['native_predictions'],archived_packet_refs=gate['packets'],actual_imported_files=libraries,
        native_model_loaded=False,native_forward_calls=0,original_JPEG_decoded=False,
        paired_seed_and_image_order=True,C_PR_identical_weight_initialization_claimed=False,
        no_optimizer=True,no_weight_update=True,no_TEST=True,best_DEV_selection=False))
    predictions={control:[] for control in CONTROLS}; diagnostics=[]
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for index,(row,ref) in enumerate(zip(image_rows,gate['packets'])):
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked fullDEV GPU deadline exceeded')
            packet=torch.load(str(bound(ref)),map_location='cpu')
            base,refined,diagnostic=evaluate_image(torch,np,masks,renderer,kernels,reader,coco,packet,row,index,ref,
                grouped[row['image_id']],heads,gate,spec,started)
            predictions['native_official']+=base
            for control,values in refined.items(): predictions[control]+=values
            diagnostics.append(diagnostic); trace.write(json.dumps(diagnostic,allow_nan=False)+'\n'); trace.flush()
            del packet,base,refined,diagnostic
            print(json.dumps(dict(evaluated_DEV_images=index+1,native_exact_queries=100*(index+1),controls=len(CONTROLS))),flush=True)
    require(predictions['native_official']==native and all(len(values)==18700 for values in predictions.values()),
        'Full native exactness and all5 all100 population required')
    for control,values in predictions.items():
        reader.population(coco,values,ids)
        require(all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'}
            for a,b in zip(native,values)),'Full18700 fixed original identity/S0/class/order differs')
    require(all(torch.equal(value.detach().cpu(),head_states[key][name]) for key,head in heads.items() for name,value in head.state_dict().items()) and
        all(not p.requires_grad and p.grad is None for head in heads.values() for p in head.parameters()) and
        all(not module.training for head in heads.values() for module in head.modules()),'Frozen head state changed during evaluation')
    torch.cuda.synchronize(0); peak=torch.cuda.max_memory_allocated(0)
    for head in heads.values(): head.cpu()
    heads.clear(); del head,heads; gc.collect(); before=torch.cuda.memory_allocated(0)
    torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache(); torch.cuda.synchronize(0)
    cleanup=dict(allocated_before=before,allocated_after=torch.cuda.memory_allocated(0),reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after']==0 and peak<=4096*1024**2,'Clean allocator4096 GPU completion required')
    prediction_refs={}; control_results={}
    for control,values in predictions.items():
        target=output/(control.upper()+'_PREDICTIONS.json'); write(target,values); prediction_refs[control]=reference(target)
        require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked official COCO deadline exceeded')
        result=official.coco_metrics(coco,values,ids,COCOeval)
        stats_exact(result['stats'],result['stats'],'Official COCO12 must keep finite defined values and undefined -1')
        require(result['iouType']=='segm' and result['maxDets']==[1,10,100] and result['original_GT_preserved'] is True and
            result['score_filter'] is None and result['official_COCO_crowd_ignore_semantics'] is True,'Original official COCO protocol differs')
        control_results[control]=dict(predictions=prediction_refs[control],stats=result['stats'],COCO=result)
    baseline_COCO=control_results['native_official']['COCO']
    stats_exact(baseline_COCO['stats'],gate['report']['control_results']['native_official']['stats'],
        'Native0 COCO12 must equal original GPU reference exactly')
    stats_exact(baseline_COCO['stats'],gate['q']['control_results']['native_official']['COCO']['stats'],
        'Native0 COCO12 must equal qualified CPU reference exactly')
    stats_exact(baseline_COCO['stats'],gate['q']['expected_E24_COCO_stats'],
        'Native0 COCO12 must equal strong E24 reference exactly')
    write(output/'IMAGE_DIAGNOSTICS.json',diagnostics)
    total_names=('added_pixels','removed_pixels','changed_pixels','expanded_queries','shrunk_queries','changed_queries',
        'native_empty_queries','refined_empty_queries','native_empty_new_nonempty_keeps_zero_score','new_empty_keeps_positive_native_score')
    totals={control:{name:sum(row['controls'][control][name] for row in diagnostics) for name in total_names}
        for control in CONTROLS[1:]}
    require(all(value['changed_pixels']==value['added_pixels']+value['removed_pixels'] for value in totals.values()),
        'Per-control pixel change accounting differs')
    head_calls={control:sum(row['controls'][control]['head_point_chunk_calls'] for row in diagnostics) for control in CONTROLS[1:]}
    for ref in spec['sources']: bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['GPU_admission_file'])==gate['admission_ref'] and imported_files(torch,np,COCO,COCOeval,masks,gate)==libraries and
        psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2 and
        torch.cuda.memory_allocated(0)==0,'Final provenance/libraries/resources/clean GPU changed')
    result=dict(schema=REPORT_SCHEMA,status=STATUS,phase=spec['phase'],source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),
        image_diagnostics=reference(output/'IMAGE_DIAGNOSTICS.json'),predictions=prediction_refs,
        native0_predictions=prediction_refs['native_official'],control_results=control_results,native0_COCO=baseline_COCO,
        head_training_entry_source=spec['head_training_entry_source'],head_training_spec=spec['head_training_spec'],
        head_training_report=spec['head_training_report'],head_training_CPU_qualification=spec['head_training_CPU_qualification'],
        head_checkpoints=spec['head_checkpoints'],head_endpoint_epoch=8,head_state_proofs=state_proofs,
        point_heads_source=spec['point_heads_source'],PR_point_heads_source=spec['PR_point_heads_source'],
        teacher_checkpoint=gate['training_report']['teacher_checkpoint'],prior_N07_report=spec['prior_N07_report'],
        N07_cpu_qualification=spec['N07_cpu_qualification'],native_reference_predictions=spec['native_predictions'],
        DEV=spec['DEV'],packet_refs=gate['packets'],reader_source=spec['reader_source'],renderer_source=spec['renderer_source'],
        export_source=spec['export_source'],COCO_source=spec['COCO_source'],recipe=RECIPE,
        images=187,raw_GT=380,predictions_per_control=18700,change_totals=totals,head_point_chunk_calls=head_calls,
        native0_COCO_exact_reference=True,native0_export_exact_reference=True,native0_exact_queries=18700,
        native_M_equals_eF_exact=True,all100_identity_and_fixed_scores_exact=True,
        native_model_loaded=False,native_forward_calls=0,teacher_forwards=0,point_heads_loaded=True,
        original_JPEG_decoded=False,patch_input_ABI='zero49; independently qualified function independence',
        optimizer_created=False,backward_calls=0,weights_updated=False,training_or_matching_loss_executed=False,
        no_GT_query_selection=True,no_NMS=True,no_score_cutoff=True,no_TEST=True,best_DEV_selection=False,
        original_native_empty_score_rule_preserved=True,new_empty_geometry_may_keep_positive_native_score=True,
        fixed_native_query_GT_matched_pixel_diagnostics_deferred_to_CPU=True,independent_CPU_evaluation_replayed=False,
        independent_full_training_trajectory_replayed=False,actual_imported_files=libraries,
        torch_peak_allocated_bytes=peak,cleanup=cleanup,elapsed_seconds=time.monotonic()-started,
        evidence_limits=[
            'C and PR use the same qualified frozen carrier, cached supervision, seed/image orders and fixed8 endpoints; their weights and architectures differ.',
            'PR is a PointRend-style matched-carrier adaptation, not the complete author pipeline or a new primitive.',
            'Native reproduction is full natural100 rendering, original reorder and full100 quality/S0, exact against all18700 archived exports and strong E24 COCO12.',
            'Point heads use the same original natural-query logits, original closed band and F/e projection; baseline S0 is verified before residuals and never recomputed afterward.',
            'Archived packet reuse includes zero teacher forward and zero JPEG decode. This evaluation does not replay teacher training/forward; prior carrier compute is a separate inherited cost.',
            'Per-query expansion/shrinkage are descriptive counts; independent CPU and matched-GT mechanism checks remain separate.',
            'Two seeds and fixed8 DEV results do not establish convergence, blind TEST, superiority over external SOTA, causal feature credit, or publication readiness.',
            'Endpoint CPU qualification checks states/order/optimizer/BCE, not an independent full gradient trajectory replay.',
            'ROOT closes actual original RAW, process reaping and whole-GPU observations; this report does not self-certify terminal success.'])
    write(output/'REPORT.json',result)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'ROOT owned execution switch required'); run(args.spec)
