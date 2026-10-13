"""UNBOUND N13 v4 SOURCE: unchanged paired training from qualified complete-epoch parents.
Only the new qualified FIT cache is consumed. LOG_PR/GEO_PR share the exact
PR-FL91 factory, cold weights, image orders and fixed 48-epoch budget. A leaf
runs one paired seed in E09--16, E17--24, E25--32, E33--40 or E41--48.
Only successful v3/v4 endpoints or the explicitly CPU-adopted seed701 v2 E08
endpoint can initialize it. Optimizer, private RNG and history are preserved.
No native model, image/GT decoder, DEV, TEST or AP is used here.
ROOT must independently qualify every actual owned terminal after exit.
"""
import argparse
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

SCHEMA='ROOT-N13-sameK-region-point-training-spec-v4'
REPORT_SCHEMA='N13-sameK-region-point-training-report-v4'
STATUS='N13_SAMEK_REGION_POINT_TRAINING_LEAF_COMPLETE_PENDING_ROOT_QUALIFICATION'
CP_SCHEMA='N13-sameK-region-PR-FL91-head-checkpoint-v4'
START_SCHEMA='N13-sameK-region-point-training-start-v4'
LOADED_SCHEMA='N13-sameK-region-point-training-loaded-state-v4'
CPU_SPEC_SCHEMA='ROOT-N13-sameK-region-point-training-CPU-spec-v4'
CPU_Q_SCHEMA='ROOT-N13-sameK-region-point-training-CPU-qualification-v4'
CPU_Q_STATUS='ROOT_N13_SAMEK_REGION_POINT_TRAINING_CPU_QUALIFIED_V4'
OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
SEEDS=(202610120701,202610120702,202610120703)
ARMS=('LOG_PR','GEO_PR')
NAMES=('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias',
       'hidden3.weight','hidden3.bias','final.weight','final.bias')
SHAPES=((91,257),(91,),(91,92),(91,),(91,92),(91,),(1,92),(1,))
PARAMETERS=40497
RAW_KEYS=('receipt','phase','admission','phase_binding','bootstrap')
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
REF_KEYS=('entry_source','region_common_source','cache_contract_source','point_heads_source','original_PR_point_heads_source','PR_factory_CPU_qualification','cache_entry_source','cache_spec','cache_report','cache_GPU_admission','cache_GPU_RAW','cache_CPU_qualification','cache_CPU_source','cache_CPU_spec','cache_CPU_admission','cache_CPU_RAW','FIT','relationship_entry_source','relationship_spec','relationship_report','relationship_GPU_admission','relationship_GPU_RAW','relationship_ROOT_qualification',)
RECIPE=dict(arms=list(ARMS),seeds=list(SEEDS),fixed_final_epochs=48,
    updates_per_epoch_per_model=1661,images_per_epoch=741,points_per_GT=1024,
    final_updates_per_model=79728,instance_mean_BCE=True,one_update_per_GT_instance=True,
    no_accumulation=True,AdamW_lr=3e-4,AdamW_weight_decay=1e-4,AdamW_betas=[.9,.999],
    AdamW_eps=1e-8,constant_lr=True,gradient_clip=None,fixed_cached_points_all_epochs=True,
    no_augmentation=True,head_parameters=40497,head_parameter_tensors=8,
    paired_seed=True,paired_initialization=True,paired_image_order=True,
    raw_GT_order_within_image=True,external_region_gate=True,
    empty_region_updates='K0 has zero data residual gradient; ordinary AdamW step incl momentum/decay is retained and counted',
    logits='native_L + point_in_region * ungated_PR_residual',
    training_inference_regions_identical=True,GT_used_for_region_selection=False,
    supervision='768 draws in each arm region plus common256 global draws; fixed cache',
    treatment='support-domain and sampled-supervision joint comparison',
    endpoint_BCE='full1661 instance mean over1024 cached points after every epoch',
    editable_BCE='mean of within-region point means over instances with >=1 editable sampled point',
    convergence_windows=[list(range(37,43)),list(range(43,49))],
    editable_relative_drop_threshold=.01,convergence_rule='relative drop >1% flags not stable; does not extend48',
    curve_epochs=[8,24,48],best_epoch_selection=False,native_weights_updated=False,
    resume='only qualified own N13 complete-epoch checkpoints; preserve optimizer and private order RNG')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and
        len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),'Malformed absolute pinned reference')


def reference(path):
    path=Path(path).resolve(strict=True); h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def bound(ref):
    ref_shape(ref); require(reference(ref['path'])==ref,'Pinned input changed: '+ref['path']); return Path(ref['path'])


def terminal(ref,phase,code=0):
    # Reads only original RAW JSON; this function never hashes or imports science.
    ref_shape(ref); path=Path(ref['path'])
    require(path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,'Original RAW locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
        type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
        raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
        raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
        'Actual held exit/reap/original birth absence required before every SHA/import')
    if 'phase' in raw: require(raw['phase']==phase,'RAW phase differs')
    peak,sha=raw['whole_GPU_observed_peak_MiB'],raw['admission_sha256']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and
        type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=8192 and
        raw['whole_GPU_used_limit_MiB']==8192 and raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','RAW resources differ')
    return raw


def pre_gate(spec,minimum):
    rows=spec['RAW_inputs']; expected=spec['normal_RAW_count']
    require(type(expected) is int and expected>=minimum and type(rows) is list and len(rows)==expected and
        len({row['receipt']['path'] for row in rows})==expected,'All prelocked normal RAW ancestors required')
    normals=[terminal(row['receipt'],row['phase']) for row in rows]
    failed=terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    extra=spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra)==6 and all(type(row) is dict and set(row)=={'receipt','phase','returncode'} and
        type(row['returncode']) is int and row['returncode']==1 for row in extra),'Preserve exact three-key extra failed6 rows, seven failures total')
    failures=[failed]+[terminal(row['receipt'],row['phase'],1) for row in extra]
    failed_paths=[spec['prior_failed_CPU_receipt']['path']]+[row['receipt']['path'] for row in extra]
    require(len(set(failed_paths))==7 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
        'Normal ancestors and seven preserved failures must be distinct')
    all_rows=[dict(row,returncode=0) for row in rows]+[dict(receipt=spec['prior_failed_CPU_receipt'],phase=Path(spec['prior_failed_CPU_receipt']['path']).parent.name,returncode=1)]+extra
    pins=spec['terminal_identity_pins']; locators=[row['receipt']['path'] for row in all_rows]
    require(type(pins) is dict and set(pins)==set(locators),'Exact all-normal-plus-seven-failure identity pins required')
    for row,raw in zip(all_rows,normals+failures):
        pin=pins[row['receipt']['path']]
        require(type(pin) is dict and set(pin)=={'phase','returncode','original_identity','admission_sha256'} and
            pin['phase']==row['phase'] and type(pin['returncode']) is int and pin['returncode']==row['returncode'] and
            pin['original_identity']==raw['original_identity'] and pin['admission_sha256']==raw['admission_sha256'],
            'Actual original identity/admission must match its prelocked pin before every SHA/import')
    # FIRST HASH may occur only after ALL above actual terminal checks.
    closure={ref['path']:ref for ref in spec['sources']}
    require(len(closure)==len(spec['sources']),'Unique SOURCE closure required')
    for ref in spec['sources']: bound(ref)
    require(all(row[k] in closure.values() for row in rows for k in RAW_KEYS if k!='phase') and
        spec['prior_failed_CPU_receipt'] in closure.values() and all(row['receipt'] in closure.values() for row in extra),
        'RAW owner metadata and failed receipts must be sources-bound')
    return normals,closure


def load(ref,name):
    path=bound(ref); source=path.read_bytes(); require(len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__); return module


def owner_ref_join(spec,owners,receipt,admission,entry,spec_ref,cuda):
    positions=[i for i,row in enumerate(spec['RAW_inputs']) if row['receipt']==receipt]
    require(len(positions)==1,'Exact normal actual RAW receipt required')
    i=positions[0]; row,owner=spec['RAW_inputs'][i],owners[i]
    require(row['admission']==admission and owner['scientific_spec']==spec_ref and owner['entry']==entry['path'] and
        owner['entry_args']==['--spec',spec_ref['path'],'--owner-execute'] and owner['cuda_visible_devices']==cuda,
        'Exact ancestral entry/spec/admission/RAW owner join differs')
    return i



PARENT_ABIS={
    'ROOT-N13-sameK-region-point-training-spec-v3':dict(
        report='N13-sameK-region-point-training-report-v3',cp='N13-sameK-region-PR-FL91-head-checkpoint-v3',
        cpu='ROOT-N13-sameK-region-point-training-CPU-spec-v3',q='ROOT-N13-sameK-region-point-training-CPU-qualification-v3',
        status='ROOT_N13_SAMEK_REGION_POINT_TRAINING_CPU_QUALIFIED_V3'),
    'ROOT-N13-sameK-region-point-training-spec-v4':dict(
        report='N13-sameK-region-point-training-report-v4',cp='N13-sameK-region-PR-FL91-head-checkpoint-v4',
        cpu='ROOT-N13-sameK-region-point-training-CPU-spec-v4',q='ROOT-N13-sameK-region-point-training-CPU-qualification-v4',
        status='ROOT_N13_SAMEK_REGION_POINT_TRAINING_CPU_QUALIFIED_V4')}
RESUME_BASE_KEYS=('type','training_entry_source','training_spec','training_GPU_RAW','training_GPU_admission',
    'CPU_entry_source','CPU_spec','CPU_qualification','CPU_RAW','CPU_admission','checkpoints')
ADOPTION_SCHEMA='ROOT-N13-completed8-failedGPU-endpoint-adoption-v1'
ADOPTION_STATUS='ROOT_ACTUAL_CPU0_COMPLETE_E08_ADOPTED_ORIGINAL_GPU_EXIT1_PRESERVED'
ADOPTION_RECORD_SHA='a53834223f6a59b925a11d2be1e0aa63d57febbb4a9084220be30c86e1043b32'
RECOVERY_SPEC_SCHEMA='ROOT-N13-completed-epoch-artifact-recovery-CPU-spec-v3'
RECOVERY_REPORT_SCHEMA='N13-completed-epoch-artifact-recovery-CPU-report-v3'
RECOVERY_Q_SCHEMA='ROOT-N13-completed-epoch-artifact-recovery-CPU-qualification-v3'
RECOVERY_Q_STATUS='ROOT_N13_COMPLETED8_FAILED_GPU_ENDPOINT_CPU_CHECKED_PENDING_ROOT_TERMINAL'


def same_parent_controls(spec,parent,closure):
    keys=('cache_spec','cache_report','cache_CPU_qualification','point_heads_source',
        'region_common_source','cache_contract_source','validation_protocol')
    require(parent['recipe']==spec['recipe']==RECIPE and parent['seed']==spec['seed'] and
        all(parent[k]==spec[k] for k in keys) and all(ref in closure.values() for ref in parent['sources']),
        'Parent must use the exact same seed, recipe, qualified cache and factory')
    require(all(parent[k]==spec[k] for k in ('relationship_entry_source','relationship_spec','relationship_report',
        'relationship_GPU_admission','relationship_GPU_RAW','relationship_ROOT_qualification')),
        'Parent qualified M-to-L relationship must remain exact')


def parent_pin_prefix(spec,parent):
    require(parent['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
        all(spec['terminal_identity_pins'].get(k)==v for k,v in parent['terminal_identity_pins'].items()),
        'Exact parent original identity/admission pins required')


def successful_parent(spec,resume,closure,owners):
    parent=read(bound(resume['training_spec'])); report=read(bound(resume['training_report']))
    cpu=read(bound(resume['CPU_spec'])); q=read(bound(resume['CPU_qualification']))
    require(parent['schema'] in PARENT_ABIS,'Only explicit successful v3/v4 training parents are admitted')
    abi=PARENT_ABIS[parent['schema']]; same_parent_controls(spec,parent,closure)
    require(report['schema']==abi['report'] and report['status']==STATUS and report['recipe']==RECIPE and
        report['seed']==spec['seed'] and parent['last_epoch']==report['last_epoch']==spec['first_epoch']-1 and
        parent['entry_source']==report['source']==resume['training_entry_source'] and report['spec']==resume['training_spec'] and
        report['GPU_admission']==resume['training_GPU_admission'] and report['phase']==parent['phase'] and
        Path(resume['training_report']['path'])==Path(parent['output_directory'])/'REPORT.json',
        'Exact successful parent report/source/spec/admission/endpoint required')
    require(cpu['schema']==abi['cpu'] and q['schema']==abi['q'] and q['status']==abi['status'] and
        q['source']==resume['training_entry_source'] and q['reader_source']==resume['CPU_entry_source'] and
        q['spec']==resume['CPU_spec'] and q['CPU_admission']==resume['CPU_admission'] and
        q['training_spec']==resume['training_spec'] and q['training_report']==resume['training_report'] and
        q['training_GPU_RAW']==resume['training_GPU_RAW'] and q['training_GPU_admission']==resume['training_GPU_admission'] and
        q['recipe']==RECIPE and q['seed']==spec['seed'] and q['last_epoch']==spec['first_epoch']-1 and
        q['checkpoints']==report['checkpoints'] and Path(resume['CPU_qualification']['path'])==Path(cpu['Q_path']) and
        q['cold_paired_weights_CPU_checked'] is True and q['resume_loaded_state_CPU_checked'] is True and
        q['private_image_order_and_endpoint_RNG_CPU_reconstructed'] is True and
        q['optimizer_endpoint_structure_and_stepcounts_CPU_checked'] is True and q['full_update_visit_sequence_CPU_checked'] is True and
        q['final_fullFIT_endpoint_BCE_CPU_recomputed'] is True and q['final_editable_BCE_CPU_recomputed'] is True and
        q['per_step_gradient_or_AdamW_replayed'] is False and q['current_owned_CPU_terminal_self_certified'] is False and
        q['CPU_affinity']==[0,1] and q['threads']==2 and q['CUDA_initialized'] is False,
        'Separately actual-terminal-qualified successful v3/v4 CPU endpoint required')
    gi=owner_ref_join(spec,owners,resume['training_GPU_RAW'],resume['training_GPU_admission'],
        resume['training_entry_source'],resume['training_spec'],'0')
    ci=owner_ref_join(spec,owners,resume['CPU_RAW'],resume['CPU_admission'],resume['CPU_entry_source'],resume['CPU_spec'],'')
    require(ci==gi+1 and parent['normal_RAW_count']==gi and parent['RAW_inputs']==spec['RAW_inputs'][:gi] and
        cpu['normal_RAW_count']==ci and cpu['RAW_inputs']==spec['RAW_inputs'][:ci] and q['RAW_inputs']==cpu['RAW_inputs'] and
        cpu['entry_source']==resume['CPU_entry_source'] and cpu['training_entry_source']==resume['training_entry_source'] and
        cpu['training_spec']==resume['training_spec'] and cpu['training_report']==resume['training_report'] and
        cpu['training_GPU_RAW']==resume['training_GPU_RAW'] and cpu['training_GPU_admission']==resume['training_GPU_admission'] and
        q['terminal_identity_pins']==cpu['terminal_identity_pins'] and
        parent['additional_failed_RAW_inputs']==cpu['additional_failed_RAW_inputs']==q['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'] and
        cpu['prior_failed_CPU_receipt']==q['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
        all(ref in closure.values() for ref in cpu['sources']),
        'Locate adjacent parent GPU0/CPU0 by exact history refs; do not use the current tail')
    parent_pin_prefix(spec,parent); parent_pin_prefix(spec,cpu)
    artifacts=[report[k] for k in ('start','trace','update_trace','decision','initial_loaded_state')]
    artifacts += [report['checkpoints'][arm][k] for arm in ARMS for k in ('checkpoint','proof')]
    require(set(report['checkpoints'])==set(ARMS) and all(ref in closure.values() for ref in artifacts) and
        all(report['checkpoints'][arm]['checkpoint']==resume['checkpoints'][arm] and
            report['checkpoints'][arm]['completed_head_epochs']==spec['first_epoch']-1 and
            report['checkpoints'][arm]['updates']==1661*(spec['first_epoch']-1) for arm in ARMS),
        'All successful parent artifacts and exact paired complete-epoch checkpoints required')
    return dict(type=resume['type'],spec=parent,report=report,CPU_spec=cpu,Q=q,CP_schema=abi['cp'],
        gpu_index=gi,cpu_index=ci,checkpoints=report['checkpoints'])


def adopted_failed_parent(spec,resume,closure,owners):
    require(spec['seed']==202610120701 and spec['first_epoch']==9 and spec['last_epoch']==16 and
        resume['ROOT_adoption_record']['bytes']==11137 and resume['ROOT_adoption_record']['sha256']==ADOPTION_RECORD_SHA,
        'Adoption is restricted to the one ROOT-qualified original seed701 E08 record')
    parent=read(bound(resume['training_spec'])); cpu=read(bound(resume['CPU_spec']))
    q=read(bound(resume['CPU_qualification'])); report=read(bound(resume['recovery_report']))
    record=read(bound(resume['ROOT_adoption_record'])); same_parent_controls(spec,parent,closure)
    require(parent['schema']=='ROOT-N13-sameK-region-point-training-spec-v2' and parent['entry_source']==resume['training_entry_source'] and
        parent['first_epoch']==1 and parent['last_epoch']==8 and parent['resume'] is None and
        cpu['schema']==RECOVERY_SPEC_SCHEMA and q['schema']==RECOVERY_Q_SCHEMA and q['status']==RECOVERY_Q_STATUS and
        report['schema']==RECOVERY_REPORT_SCHEMA and report['status']==RECOVERY_Q_STATUS,
        'Original v2 failed endpoint requires the distinct v3 CPU recovery contract')
    ci=owner_ref_join(spec,owners,resume['CPU_RAW'],resume['CPU_admission'],resume['CPU_entry_source'],resume['CPU_spec'],'')
    require(ci==40 and cpu['normal_RAW_count']==40 and cpu['RAW_inputs']==spec['RAW_inputs'][:ci] and
        parent['normal_RAW_count']==40 and parent['RAW_inputs']==cpu['RAW_inputs'] and q['RAW_inputs']==cpu['RAW_inputs'] and
        cpu['entry_source']==q['source']==q['reader_source']==report['source']==resume['CPU_entry_source'] and
        q['spec']==report['spec']==resume['CPU_spec'] and q['CPU_admission']==report['CPU_admission']==resume['CPU_admission'] and
        q['recovery_report']==resume['recovery_report'] and Path(resume['CPU_qualification']['path'])==Path(cpu['Q_path']) and
        Path(resume['recovery_report']['path'])==Path(cpu['report_path']) and
        q['terminal_identity_pins']==cpu['terminal_identity_pins'] and report['terminal_identity_pins']==cpu['terminal_identity_pins'] and
        parent['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'][:5] and
        cpu['additional_failed_RAW_inputs']==q['additional_failed_RAW_inputs']==report['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'] and
        cpu['prior_failed_CPU_receipt']==q['prior_failed_CPU_receipt']==report['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
        all(ref in closure.values() for ref in cpu['sources']), 'Actual recovery CPU0 and exact historical failure prefixes required')
    parent_pin_prefix(spec,parent); parent_pin_prefix(spec,cpu)
    keys=('training_entry_source','training_spec','training_GPU_admission','training_GPU_RAW',
        'region_common_source','cache_contract_source','point_heads_source','cache_CPU_qualification')
    require(all(cpu[k]==q[k]==report[k]==resume[k] for k in keys[:4]) and
        all(cpu[k]==q[k]==report[k]==spec[k] for k in keys[4:]) and
        all(cpu[k]==q[k]==report[k]==spec[k] for k in ('relationship_entry_source','relationship_spec','relationship_report',
            'relationship_GPU_admission','relationship_GPU_RAW','relationship_ROOT_qualification')) and
        cpu['recipe']==q['recipe']==report['recipe']==RECIPE and q['seed']==report['seed']==202610120701 and
        q['first_epoch']==report['first_epoch']==1 and q['last_epoch']==report['last_epoch']==8 and
        q['checkpoints']==report['checkpoints'] and q['artifacts']==report['artifacts']==cpu['artifacts'],
        'Recovery source/spec/report/Q must bind the original failed GPU and exact cache/factory')
    require(q['original_GPU_actual_returncode']==report['original_GPU_actual_returncode']==1 and
        q['original_GPU_report_written'] is False and report['original_GPU_report_written'] is False and
        q['original_exit_is_not_relabelled'] is True and report['original_exit_is_not_relabelled'] is True and
        q['original_GPU_artifacts_unchanged'] is True and report['original_GPU_artifacts_unchanged'] is True and
        q['cold_paired_weights_CPU_checked'] is True and q['initial_loaded_state_CPU_checked'] is True and
        q['private_image_order_and_endpoint_RNG_CPU_reconstructed'] is True and
        q['optimizer_endpoint_structure_and_stepcounts_CPU_checked'] is True and q['full_update_visit_sequence_CPU_checked'] is True and
        q['final_fullFIT_endpoint_BCE_CPU_recomputed'] is True and q['final_editable_BCE_CPU_recomputed'] is True and
        q['training_rerun'] is False and q['optimizer_steps_executed']==0 and q['per_step_gradient_or_AdamW_replayed'] is False and
        q['current_endpoint_adoption_self_certified'] is False and q['current_owned_CPU_terminal_self_certified'] is False and
        q['resume_eligible_only_after_ROOT_actual_CPU0'] is True and q['CPU_affinity']==[0,1] and q['threads']==2 and q['CUDA_initialized'] is False,
        'Independent recovery checks do not reclassify the failed GPU or replay training')
    current_row=spec['RAW_inputs'][ci]
    record_refs=('CPU_spec','CPU_admission','CPU_RAW','CPU_phase_binding','CPU_bootstrap','CPU_report','CPU_qualification',
        'original_GPU_failure_record','original_GPU_RAW')
    require(all(record[k] in closure.values() for k in record_refs) and record['schema']==ADOPTION_SCHEMA and record['status']==ADOPTION_STATUS and
        record['CPU_spec']==resume['CPU_spec'] and record['CPU_admission']==resume['CPU_admission'] and record['CPU_RAW']==resume['CPU_RAW'] and
        record['CPU_phase_binding']==current_row['phase_binding'] and record['CPU_bootstrap']==current_row['bootstrap'] and
        record['CPU_report']==resume['recovery_report'] and record['CPU_qualification']==resume['CPU_qualification'] and
        record['original_GPU_failure_record']==q['training_failure_ROOT_record'] and record['original_GPU_RAW']==resume['training_GPU_RAW'] and
        record['original_GPU_actual_returncode']==1 and record['original_GPU_report_absent'] is True and
        record['normal_RAW_count']==ci+1==41 and record['preserved_failed_RAW_count']==7 and
        record['full_source_checked_count']==len(cpu['sources'])==8740 and record['original_execution_session']==56178 and
        record['seed']==202610120701 and record['completed_head_epochs']==8 and record['updates_per_model']==13288 and record['fixed_final_epoch']==48 and
        record['optimizer_updates_executed_by_recovery']==0 and record['training_rerun'] is False and record['per_step_AdamW_or_gradient_replayed'] is False and
        record['adopted_for_continuation_only'] is True and record['requires_separate_adoption_aware_resume_entry'] is True and
        record['method_efficacy'] is False and record['AP_evaluated'] is False and
        record['checkpoint_refs']==q['checkpoints'] and record['original_artifacts']==q['artifacts'] and
        record['final_endpoints']==q['final_endpoints']==report['final_endpoints'],
        'Pinned ROOT actual-CPU0 adoption record must join every recovery input and endpoint')
    failed=spec['additional_failed_RAW_inputs'][-1]
    require(failed==dict(receipt=resume['training_GPU_RAW'],phase=parent['phase'],returncode=1),
        'Original GPU remains the seventh failed RAW and never becomes a normal ancestor')
    raw=terminal(resume['training_GPU_RAW'],parent['phase'],1)
    row=dict(receipt=resume['training_GPU_RAW'],phase=parent['phase'],admission=resume['training_GPU_admission'],
        phase_binding=q['training_GPU_phase_binding'],bootstrap=q['training_GPU_bootstrap'])
    require(all(row[k] in closure.values() for k in RAW_KEYS if k!='phase'),'Original failed owner primary closure required')
    common=load(spec['region_common_source'],'_N13_adopted_parent_owner_common')
    owner=common.owner_join(row,raw,closure)
    require(owner['scientific_spec']==resume['training_spec'] and owner['entry']==resume['training_entry_source']['path'] and
        owner['entry_args']==['--spec',resume['training_spec']['path'],'--owner-execute'] and owner['cuda_visible_devices']=='0' and
        parent['GPU_admission_file']==resume['training_GPU_admission']['path'], 'Original failed GPU owner must retain its exact SOURCE/spec/admission')
    missing=Path(parent['output_directory'])/'REPORT.json'; require(not missing.exists() and not missing.is_symlink(),'No original GPU REPORT may be fabricated')
    view=report['artifact_derived_training_view']
    require(view['view_type']=='ARTIFACT_DERIVED_COMPATIBILITY_VIEW_NOT_ORIGINAL_GPU_REPORT' and
        view['source']==resume['training_entry_source'] and view['spec']==resume['training_spec'] and view['GPU_admission']==resume['training_GPU_admission'] and
        view['original_GPU_actual_returncode']==1 and view['original_GPU_report_written'] is False and
        view['checkpoints']==q['checkpoints'] and len(q['artifacts'])==9 and all(ref in closure.values() for ref in q['artifacts'].values()) and
        all(q['checkpoints'][arm]['checkpoint']==resume['checkpoints'][arm] and q['checkpoints'][arm]['completed_head_epochs']==8 and
            q['checkpoints'][arm]['updates']==13288 for arm in ARMS), 'Use the labelled recovery view and original paired v2 checkpoints only')
    return dict(type=resume['type'],spec=parent,report=view,CPU_spec=cpu,Q=q,CP_schema='N13-sameK-region-PR-FL91-head-checkpoint-v2',
        cpu_index=ci,checkpoints=q['checkpoints'],ROOT_record=record,missing_original_GPU_report=str(missing))


def parent_gate(spec,closure,owners):
    resume=spec['resume']; require(type(resume) is dict and resume.get('type') in
        ('QUALIFIED_SUCCESSFUL_GPU_ENDPOINT','CPU_ADOPTED_FAILED_GPU_ENDPOINT'),'Explicit qualified endpoint type required')
    extras=('training_report',) if resume['type']=='QUALIFIED_SUCCESSFUL_GPU_ENDPOINT' else ('recovery_report','ROOT_adoption_record')
    keys=RESUME_BASE_KEYS+extras
    require(set(resume)==set(keys) and type(resume['checkpoints']) is dict and set(resume['checkpoints'])==set(ARMS),
        'Exact typed parent interface and two same-seed checkpoints required')
    refs=[resume[k] for k in keys if k not in ('type','checkpoints')]+list(resume['checkpoints'].values())
    require(all(ref in closure.values() for ref in refs),'Complete typed parent provenance closure required')
    if resume['type']=='QUALIFIED_SUCCESSFUL_GPU_ENDPOINT': return successful_parent(spec,resume,closure,owners)
    return adopted_failed_parent(spec,resume,closure,owners)


def parent_final_check(gate):
    parent=gate['parent']
    if 'missing_original_GPU_report' in parent:
        path=Path(parent['missing_original_GPU_report'])
        require(not path.exists() and not path.is_symlink(),'Original failed GPU REPORT must remain absent after resumed computation')


def source_gate(path,spec,owner_environment=True):
    require(spec['schema']==SCHEMA,'Exact N13 training schema required')
    raws,closure=pre_gate(spec,45)
    keys=REF_KEYS
    require(all(spec[k] in closure.values() for k in keys),'Complete new training/cache/factory refs absent')
    original_common=load(spec['region_common_source'],'_N13_training_region_common')
    adapter=load(spec['cache_contract_source'],'_N13_v2_qualified_cache_contract')
    require(reference(adapter.__file__)==spec['cache_contract_source'],'Actually bound new adapter SOURCE differs')
    common=adapter.bind_common(original_common)
    require(spec['validation_protocol']==adapter.VALIDATION_PROTOCOL,'Exact new cache evidence protocol required')
    require(common.ARMS==ARMS,'Common region arm ABI differs')
    owners=[common.owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    cached,report,q=common.validate_cache_input(spec,closure,owners)
    require(report['images']==741 and report['raw_GT']==1661 and len(report['packet_refs'])==len(report['rows'])==len(report['validation_carrier_refs'])==741 and
        all(ref in closure.values() for ref in report['validation_carrier_refs']),
        'Complete qualified741/1661 new frozen FIT cache required')
    old_q=read(bound(spec['PR_factory_CPU_qualification'])); original=spec['original_PR_point_heads_source']; fresh=spec['point_heads_source']
    require(old_q['schema']=='ROOT-N10-C-PR-FL91-synthetic-head-CPU-qualification-v1' and
        old_q['status']=='ROOT_N10_C_PR_FL91_SYNTHETIC_HEAD_CPU_QUALIFIED' and old_q['PR_point_heads_source']==original and
        old_q['PR_parameters']==40497 and old_q['synthetic_cold_identity_and_band_checked'] is True and
        old_q['synthetic_backward_update_checked'] is True and old_q['synthetic_CPU_only'] is True and
        original['bytes']==fresh['bytes'] and original['sha256']==fresh['sha256'] and
        old_q['spec'] in closure.values() and old_q['source'] in closure.values() and old_q['CPU_admission'] in closure.values(),
        'New factory must be byte-identical to prior independently qualified PR factory')
    require(spec['RAW_inputs'][28]['admission']==old_q['CPU_admission'] and owners[28]['scientific_spec']==old_q['spec'] and
        owners[28]['entry']==old_q['source']['path'] and owners[28]['cuda_visible_devices']=='',
        'Original synthetic head qualification actual owner differs')
    require(spec['recipe']==RECIPE and type(spec['seed']) is int and spec['seed'] in SEEDS and
        type(spec['first_epoch']) is int and type(spec['last_epoch']) is int and
        spec['first_epoch'] in (9,17,25,33,41) and spec['last_epoch']==spec['first_epoch']+7 and
        not any(spec['first_epoch']<=curve<spec['last_epoch'] for curve in (8,24)) and
        type(spec['deadline_seconds']) is int and 0<spec['deadline_seconds']<=1800 and
        spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192,'Prelocked same48 budget/leaf/resources required')
    parent=parent_gate(spec,closure,owners)
    source,spec_ref=reference(__file__),reference(path)
    require(source==spec['entry_source'],'Actual N13 training SOURCE differs')
    admission_path=Path(spec['GPU_admission_file']); require(admission_path.is_absolute(),'Absolute current owner admission required')
    admission_ref=reference(admission_path); admission=read(admission_path)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['driver_sha256']==OWNER_SHA and
        admission['phase']==spec['phase'] and admission['scientific_spec']==spec_ref and admission['entry']==source['path'] and
        admission['entry_args']==['--spec',str(path),'--owner-execute'] and admission['allowed_cpus']==[0,1] and
        admission['cuda_visible_devices']=='0' and admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB']==8192 and
        admission['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and
        admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
        all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']),
        'Complete single owned training admission differs')
    if owner_environment:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='2' for k in THREAD_ENV),'GPU/thread environment differs')
    return dict(common=common,cached=cached,cache_report=report,cache_Q=q,parent=parent,closure=closure,owners=owners,
                source=source,spec_ref=spec_ref,admission_ref=admission_ref,admission=admission)


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def digest(torch,tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def cpu_tree(torch,value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(torch,v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(torch,v) for v in value)
    return value


def same_tree(torch,left,right):
    if isinstance(left,torch.Tensor): return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if type(left) is not type(right): return False
    if isinstance(left,dict): return left.keys()==right.keys() and all(same_tree(torch,left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)): return len(left)==len(right) and all(same_tree(torch,a,b) for a,b in zip(left,right))
    return left==right


def imported_files(psutil,torch,np,gate):
    modules=dict(psutil=psutil,psutil_platform=psutil._psplatform,psutil_binary=psutil._psplatform.cext,
        torch=torch,torch_binary=torch._C,torch_tensor=sys.modules['torch._tensor'],torch_autograd=sys.modules['torch.autograd'],
        torch_random=torch.random,numpy=np,numpy_binary=np.core._multiarray_umath)
    result={name:reference(module.__file__) for name,module in modules.items()}
    objects=dict(torch_Module=torch.nn.Module,torch_Linear=torch.nn.Linear,AdamW=torch.optim.AdamW,
        Optimizer=torch.optim.Optimizer,point_math=torch.nn.functional.binary_cross_entropy_with_logits,
        torch_serialization=torch.load)
    result.update({name:reference(inspect.getsourcefile(obj)) for name,obj in objects.items()})
    require(all(ref in gate['closure'].values() and gate['admission']['bound_source_sha256'].get(ref['path'])==ref['sha256']
        for ref in result.values()),'Actual imported library/source/binary is not fully pinned')
    return result


def check_state(torch,state):
    require(type(state) is dict and tuple(state)==NAMES,'Exact8 parameter names required')
    for name,shape in zip(NAMES,SHAPES):
        p=state[name]; require(isinstance(p,torch.Tensor) and p.dtype==torch.float32 and tuple(p.shape)==shape and
            not p.requires_grad and bool(torch.isfinite(p).all()),'Exact finite FP32 state required: '+name)
    require(sum(p.numel() for p in state.values())==PARAMETERS,'Exactly40497 scalars required')


def check_head(torch,head):
    require(tuple(n for n,p in head.named_parameters())==NAMES and tuple(tuple(p.shape) for p in head.parameters())==SHAPES and
        sum(p.numel() for p in head.parameters())==PARAMETERS and not tuple(head.buffers()) and
        all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in head.parameters()),'Exact finite40497 PR head required')


def cache_packet(torch,gate,index):
    report,q,common=gate['cache_report'],gate['cache_Q'],gate['common']
    require(report['packet_refs'][index]==q['packet_refs'][index],'Exact qualified cache packet order required')
    value=common.load_packet(torch,report,index)
    G=len(value['query_ids'])  # common.load_packet already performs exact packet_validate once
    require(G==len(value['query_ids']) and 1<=G<=100,'Complete qualified raw-GT cache instance count required')
    carrier=report['validation_carrier_refs'][index]
    require(carrier in gate['closure'].values() and value['validation_carrier']==carrier and
        report['rows'][index]['validation_carrier']==carrier,'Independent CPU carrier must remain bound')
    # Sidecar carriers are never loaded in training or in this endpoint loss.
    return value


def endpoint(torch,gate,heads,spec,started,device):
    losses={arm:[] for arm in ARMS}; editable={arm:[] for arm in ARMS}; point_counts={arm:0 for arm in ARMS}
    modes={arm:head.training for arm,head in heads.items()}
    try:
        for head in heads.values(): head.eval()
        with torch.no_grad():
            for index in range(741):
                require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked leaf deadline exceeded at fullFIT endpoint')
                value=cache_packet(torch,gate,index)
                for g in range(len(value['query_ids'])):
                    for arm in ARMS:
                        F,e,L,y,flag=gate['common'].instance_inputs(torch,value,g,arm,device)
                        logits=gate['common'].region_logits(torch,heads[arm],F,e,L,flag)['logits']
                        per=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                        require(bool(torch.isfinite(per).all()) and torch.equal(logits[~flag],L[~flag]),'Finite same-region endpoint required')
                        losses[arm].append(float(per.mean().cpu())); count=int(flag.sum().cpu()); point_counts[arm]+=count
                        if count: editable[arm].append(float(per[flag].mean().cpu()))
                del value
    finally:
        for arm,head in heads.items(): head.train(modes[arm])
    require(all(len(v)==1661 for v in losses.values()),'Full1661 endpoint population required')
    return {arm:dict(BCE=math.fsum(losses[arm])/1661,editable_BCE=(math.fsum(editable[arm])/len(editable[arm]) if editable[arm] else None),
        instances=1661,points=1700864,editable_instances=len(editable[arm]),
        empty_editable_instances=1661-len(editable[arm]),editable_points=point_counts[arm]) for arm in ARMS}


def optimizer_proof(torch,opt,head,updates):
    state=cpu_tree(torch,opt.state_dict()); require(len(state['param_groups'])==1,'Single AdamW group required')
    group=state['param_groups'][0]
    require(group['lr']==3e-4 and group['weight_decay']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and
        group['amsgrad'] is False and group['maximize'] is False and group['capturable'] is False and
        group['differentiable'] is False and group['foreach'] is None and len(group['params'])==len(state['state'])==8,'Exact AdamW recipe/moments required')
    proof={}
    for name,param,identifier in zip(NAMES,head.parameters(),group['params']):
        entry=state['state'][identifier]
        require(set(entry)=={'step','exp_avg','exp_avg_sq'} and float(entry['step'])==updates and
            all(entry[k].dtype==torch.float32 and tuple(entry[k].shape)==tuple(param.shape) and bool(torch.isfinite(entry[k]).all())
                for k in ('exp_avg','exp_avg_sq')),'Finite cumulative AdamW moments/step counts required')
        proof[name]={k:digest(torch,v) for k,v in entry.items()}; proof[name]['updates']=updates
    return state,proof


def stability(history,epoch):
    if epoch!=48: return dict(status='NOT_FINAL48',independent_convergence_proven=False)
    by_epoch={row['epoch']:row for row in history}; values=[by_epoch[e]['editable_BCE'] for e in range(37,49)]
    if any(v is None for v in values): return dict(status='NO_EDITABLE_POPULATION',independent_convergence_proven=False)
    before=math.fsum(by_epoch[e]['editable_BCE'] for e in range(37,43))/6
    after=math.fsum(by_epoch[e]['editable_BCE'] for e in range(43,49))/6
    drop=(before-after)/before if before>0 else 0.
    return dict(status=('NOT_STABLE_GT1PCT_DROP' if drop>.01 else 'NO_GT1PCT_DROP_FLAG'),window37_42=before,window43_48=after,
        relative_drop=drop,threshold=.01,independent_convergence_proven=False,automatic_extension=False)


def save_checkpoint(torch,pair,arm,gate,spec,output):
    head,opt=pair['heads'][arm],pair['optimizers'][arm]; epoch=spec['last_epoch']; updates=1661*epoch
    check_head(torch,head); state=cpu_tree(torch,head.state_dict()); opt_state,opt_proof=optimizer_proof(torch,opt,head,updates)
    meta=dict(schema=CP_SCHEMA,source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        arm=arm,seed=spec['seed'],head_factory_source=spec['point_heads_source'],cache_spec=spec['cache_spec'],cache_report=spec['cache_report'],
        cache_CPU_qualification=spec['cache_CPU_qualification'],recipe=RECIPE,first_epoch_this_leaf=spec['first_epoch'],completed_head_epochs=epoch,
        updates=updates,GT_visits=updates,point_visits=updates*1024,image_visits=epoch*741,native_weights_updated=False,
        training_resume_eligible=True,resume_only_qualified_N13=True,formal_native_initializer=False,
        parameter_tensors=8,parameter_scalars=40497,no_pending_gradients=all(p.grad is None for p in head.parameters()),
        endpoint_BCE_history=pair['history'][arm],epoch_orders=pair['orders'],gradient_activity=pair['activity'][arm],
        cold_state_proofs=pair['cold_proofs'],paired_initialization=True,paired_image_order=True,
        parent_CPU_qualification=spec['resume']['CPU_qualification'],parent_endpoint_type=spec['resume']['type'],
        parent_checkpoint_schema=gate['parent']['CP_schema'],stability=stability(pair['history'][arm],epoch))
    require(meta['no_pending_gradients'],'Clear gradients at complete epoch boundary')
    cp=dict(state_dict=state,optimizer_state_dict=opt_state,private_order_state=pair['order_rng'].get_state().clone(),meta=meta)
    target=output/('HEAD_%s_SEED%d_EPOCH%02d.pth'%(arm,spec['seed'],epoch))
    with target.open('xb') as stream: torch.save(cp,stream)
    require(same_tree(torch,cp,torch.load(str(target),map_location='cpu')),'Own full checkpoint readback differs')
    proof_path=target.with_suffix('.PROOF.json')
    write(proof_path,dict(checkpoint=reference(target),state={n:digest(torch,v) for n,v in state.items()},optimizer=opt_proof,
        private_order_state=digest(torch,cp['private_order_state']),updates=updates,full_readback_exact=True))
    return dict(checkpoint=reference(target),proof=reference(proof_path),completed_head_epochs=epoch,updates=updates,
        endpoint=pair['history'][arm][-1],stability=meta['stability'])


def make_pair(torch,factory,spec,gate):
    seed=spec['seed']; outside=torch.random.get_rng_state().clone()
    heads={arm:factory.make_point_head(torch,seed) for arm in ARMS}
    for head in heads.values():
        check_head(torch,head); require(not bool((head.final.weight!=0).any()) and not bool((head.final.bias!=0).any()),'Cold final layer must be zero')
    require(same_tree(torch,heads[ARMS[0]].state_dict(),heads[ARMS[1]].state_dict()) and
        torch.equal(outside,torch.random.get_rng_state()) and not torch.cuda.is_initialized(),'Exact cold pair/outside RNG/coldCUDA required')
    cold={arm:{name:digest(torch,value) for name,value in head.state_dict().items()} for arm,head in heads.items()}
    order_rng=torch.Generator(device='cpu'); order_rng.manual_seed(seed)
    pair=dict(heads=heads,optimizers={},order_rng=order_rng,cold_proofs=cold,
        history={arm:[] for arm in ARMS},orders=[],activity={arm:{n:dict(present=0,nonzero=0) for n in NAMES} for arm in ARMS})
    if spec['resume']:
        cps={arm:torch.load(str(bound(spec['resume']['checkpoints'][arm])),map_location='cpu') for arm in ARMS}
        for arm,cp in cps.items():
            check_state(torch,cp['state_dict']); meta=cp['meta']
            require(meta['schema']==gate['parent']['CP_schema'] and meta['recipe']==RECIPE and meta['seed']==seed and meta['arm']==arm and
                meta['source']==spec['resume']['training_entry_source'] and meta['spec']==spec['resume']['training_spec'] and
                meta['GPU_admission']==spec['resume']['training_GPU_admission'] and meta['completed_head_epochs']==spec['first_epoch']-1 and
                meta['head_factory_source']==spec['point_heads_source'] and meta['cache_CPU_qualification']==spec['cache_CPU_qualification'] and
                meta['training_resume_eligible'] is True and meta['resume_only_qualified_N13'] is True and meta['cold_state_proofs']==cold,
                'Actual own cumulative checkpoint identity differs')
            heads[arm].load_state_dict(cp['state_dict'],strict=True); pair['history'][arm]=meta['endpoint_BCE_history']; pair['activity'][arm]=meta['gradient_activity']
        require(cps[ARMS[0]]['meta']['epoch_orders']==cps[ARMS[1]]['meta']['epoch_orders'] and
            torch.equal(cps[ARMS[0]]['private_order_state'],cps[ARMS[1]]['private_order_state']),'Parent paired cumulative image RNG differs')
        pair['orders']=cps[ARMS[0]]['meta']['epoch_orders']; order_rng.set_state(cps[ARMS[0]]['private_order_state']); pair['parent_cps']=cps
    require(torch.equal(outside,torch.random.get_rng_state()) and not torch.cuda.is_initialized(),'Resume changed outside RNG or CUDA')
    return pair


def save_loaded(torch,pair,gate,spec,output):
    data=dict(schema=LOADED_SCHEMA,source=gate['source'],spec=gate['spec_ref'],seed=spec['seed'],first_epoch=spec['first_epoch'],
        cold_state_proofs=pair['cold_proofs'],parent_endpoint_type=spec['resume']['type'],parent_checkpoint_schema=gate['parent']['CP_schema'],
        arms={arm:dict(state_dict=cpu_tree(torch,pair['heads'][arm].state_dict()),
        optimizer_state_dict=cpu_tree(torch,pair['optimizers'][arm].state_dict()),history=pair['history'][arm],gradient_activity=pair['activity'][arm]) for arm in ARMS},
        private_order_state=pair['order_rng'].get_state().clone(),epoch_orders=pair['orders'])
    path=output/'INITIAL_LOADED_STATE.pth'
    with path.open('xb') as stream: torch.save(data,stream)
    require(same_tree(torch,data,torch.load(str(path),map_location='cpu')),'Pre-update loaded state readback differs')
    return reference(path)


def train_epochs(torch,pair,gate,spec,started,trace,updates_trace):
    for epoch in range(spec['first_epoch'],spec['last_epoch']+1):
        order=torch.randperm(741,generator=pair['order_rng']).tolist()
        sha=hashlib.sha256(json.dumps(order,separators=(',',':')).encode('ascii')).hexdigest()
        pair['orders'].append(dict(epoch=epoch,indices=order,sha256=sha)); online={arm:[] for arm in ARMS}; visits=0
        for index in order:
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked leaf training deadline exceeded')
            value=cache_packet(torch,gate,index)
            for g in range(len(value['query_ids'])):
                record=dict(seed=spec['seed'],epoch=epoch,image_index=index,GT_index=g,annotation_id=value['identity']['ann_ids'][g],
                    query_id=int(value['query_ids'][g]),global_step=1661*(epoch-1)+visits+1,BCE={},gradient_present={},gradient_nonzero={})
                for arm in ARMS:
                    head,opt=pair['heads'][arm],pair['optimizers'][arm]; opt.zero_grad(set_to_none=True)
                    F,e,L,y,flag=gate['common'].instance_inputs(torch,value,g,arm,'cuda:0')
                    result=gate['common'].region_logits(torch,head,F,e,L,flag); logits=result['logits']
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
                    require(bool(torch.isfinite(loss)) and loss.requires_grad and torch.equal(logits[~flag],L[~flag]),'Finite connected externally gated loss required')
                    loss.backward(); present=[]; nonzero=[]
                    for name,p in head.named_parameters():
                        require(p.grad is not None and bool(torch.isfinite(p.grad).all()),'Finite present head gradients required')
                        nz=int(bool((p.grad!=0).any())); pair['activity'][arm][name]['present']+=1; pair['activity'][arm][name]['nonzero']+=nz
                        present.append(1); nonzero.append(nz)
                    opt.step(); opt.zero_grad(set_to_none=True); check_head(torch,head)
                    measured=float(loss.detach().cpu()); online[arm].append(measured); record['BCE'][arm]=measured
                    record['gradient_present'][arm]=present; record['gradient_nonzero'][arm]=nonzero
                updates_trace.write(json.dumps(record,allow_nan=False)+'\n'); visits+=1
            del value,F,e,L,y,flag,result,logits,loss
        require(visits==1661 and all(len(v)==1661 for v in online.values()),'Complete paired1661 per-epoch updates required')
        measured=endpoint(torch,gate,pair['heads'],spec,started,'cuda:0')
        for arm in ARMS: pair['history'][arm].append(dict(epoch=epoch,**measured[arm]))
        row=dict(seed=spec['seed'],epoch=epoch,image_order_sha256=sha,images=741,GT_instances=1661,updates_per_model=1661,
            cumulative_updates_per_model=epoch*1661,point_visits_per_model=1700864,
            online_BCE={arm:math.fsum(v)/1661 for arm,v in online.items()},full_FIT_endpoint=measured,
            paired_initialization=True,paired_image_order=True,native_forward_calls=0,native_weights_updated=False)
        trace.write(json.dumps(row,allow_nan=False)+'\n'); trace.flush(); updates_trace.flush()
        print(json.dumps(dict(seed=spec['seed'],epoch=epoch,endpoint=measured)),flush=True)


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute ROOT SPEC required'); path=path.resolve(strict=True)
    spec=read(path); gate=source_gate(path,spec); output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists(),'Fresh absolute leaf output required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'CPU01 required before numerical imports')
    import torch
    import numpy as np
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and not torch.cuda.is_initialized(),'Pinned original Torch/NumPy/coldCUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    imported=imported_files(psutil,torch,np,gate); factory=load(spec['point_heads_source'],'_N13_identical_PR_factory')
    pair=make_pair(torch,factory,spec,gate)
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0); outside=[torch.random.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone()]
    for arm,head in pair['heads'].items():
        head.to('cuda:0'); head.train(); opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
        if spec['resume']: opt.load_state_dict(pair['parent_cps'][arm]['optimizer_state_dict'])
        else: require(len(opt.state)==0,'Actually fresh cold optimizer required')
        pair['optimizers'][arm]=opt
    require('mmdet' not in sys.modules,'No native model may be imported')
    output.mkdir(); loaded=save_loaded(torch,pair,gate,spec,output)
    write(output/'START.json',dict(schema=START_SCHEMA,source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],
        phase=spec['phase'],seed=spec['seed'],first_epoch=spec['first_epoch'],last_epoch=spec['last_epoch'],recipe=RECIPE,
        cache_CPU_qualification=spec['cache_CPU_qualification'],point_heads_source=spec['point_heads_source'],initial_loaded_state=loaded,
        cold_state_proofs=pair['cold_proofs'],resume_parent_CPU_qualification=spec['resume']['CPU_qualification'],
        parent_endpoint_type=spec['resume']['type'],parent_checkpoint_schema=gate['parent']['CP_schema'],
        optimizer_reset=False,private_order_RNG_reset=False,paired_initialization=True,native_model_loaded=False,no_DEV=True,no_TEST=True,no_AP=True))
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        with (output/'UPDATE_TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as update_trace:
            train_epochs(torch,pair,gate,spec,started,trace,update_trace)
    final={arm:save_checkpoint(torch,pair,arm,gate,spec,output) for arm in ARMS}
    require(all(len(pair['history'][arm])==len(pair['orders'])==spec['last_epoch'] for arm in ARMS),'Complete cumulative epoch records required')
    decision=dict(fixed_final_epoch=48,last_epoch_this_leaf=spec['last_epoch'],continue_to_prelocked48=spec['last_epoch']<48,
        stability={arm:stability(pair['history'][arm],spec['last_epoch']) for arm in ARMS},
        automatic_extension=False,best_epoch_selection=False,DEV_accessed=False,TEST_accessed=False)
    write(output/'DECISION.json',decision)
    histories=pair['history']; activity=pair['activity']; cold=pair['cold_proofs']
    require(torch.equal(outside[0],torch.random.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),'Outside RNG changed')
    peak=torch.cuda.max_memory_allocated(0)
    for head in pair['heads'].values(): head.cpu()
    pair['optimizers'].clear(); pair['heads'].clear(); pair.pop('parent_cps',None)
    del pair,head,opt; gc.collect(); before=torch.cuda.memory_allocated(0)
    torch.cuda.synchronize(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    require(torch.cuda.memory_allocated(0)==0,'Live GPU aliases remain after leaf cleanup')
    require(psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2,'Final resources differ')
    require(imported_files(psutil,torch,np,gate)==imported,'Actual imported source changed')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['GPU_admission_file'])==gate['admission_ref'],
        'Final source/spec/admission changed')
    parent_final_check(gate)
    result=dict(schema=REPORT_SCHEMA,status=STATUS,source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
        seed=spec['seed'],first_epoch=spec['first_epoch'],last_epoch=spec['last_epoch'],fixed_final_epoch=48,recipe=RECIPE,
        start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),update_trace=reference(output/'UPDATE_TRACE.jsonl'),
        decision=reference(output/'DECISION.json'),initial_loaded_state=loaded,checkpoints=final,endpoint_histories=histories,
        cache_spec=spec['cache_spec'],cache_report=spec['cache_report'],cache_CPU_qualification=spec['cache_CPU_qualification'],
        point_heads_source=spec['point_heads_source'],region_common_source=spec['region_common_source'],
        cache_contract_source=spec['cache_contract_source'],validation_protocol=spec['validation_protocol'],
        **{k:spec[k] for k in gate['common'].RELATION_REF_KEYS},
        normal_RAW_count=spec['normal_RAW_count'],RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],terminal_identity_pins=spec['terminal_identity_pins'],
        cold_state_proofs=cold,gradient_activity=activity,
        resume_parent_CPU_qualification=spec['resume']['CPU_qualification'],resume_parent_type=spec['resume']['type'],
        resume_parent_checkpoint_schema=gate['parent']['CP_schema'],
        leaf_updates_per_model=1661*(spec['last_epoch']-spec['first_epoch']+1),cumulative_updates_per_model=1661*spec['last_epoch'],
        paired_initialization=True,paired_image_order=True,optimizer_reset=False,private_order_RNG_reset=False,
        native_model_loaded=False,native_forward_calls=0,native_weights_updated=False,AP_computed=False,no_DEV=True,no_TEST=True,
        own_stage_cpu_qualification=False,current_owned_terminal_self_certified=False,independent_training_trajectory_replayed=False,
        imported_library_refs=imported,elapsed_seconds=time.monotonic()-started,torch_peak_allocated_bytes=peak,
        cleanup=dict(allocated_before=before,allocated_after=torch.cuda.memory_allocated(0)),
        evidence_limits=['This is a support-domain plus sampled-supervision joint comparison, not isolated support causality.',
            'Geometry selection and the reused PR head are known constructions; efficacy would not establish novelty.',
            'Training logs record the complete visit/update sequence, not an independent replay of each gradient or AdamW update.',
            '48 epochs is a fixed practical budget; a FIT loss plateau is not proof of generalization or full convergence.',
            'DEV region design history remains disclosed; no test-set or cross-dataset claim is made.',
            'New own checkpoints alone are resumable; old N10 endpoints are not new48-budget baselines.',
            'The current process cannot qualify its own held terminal or independent CPU endpoint.'])
    require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked whole training leaf deadline exceeded')
    write(output/'REPORT.json',result); print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Explicit single ROOT-owned execution required'); run(args.spec)
