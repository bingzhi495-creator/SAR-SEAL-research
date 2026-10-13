"""UNBOUND stdlib-only N13 full187 evaluation lineage. ROOT alone admits science.
Original v2 GPU1 is retained; actual recovery CPU v3 and ROOT v1 are distinct.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import types

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
    require(type(extra) is list and len(extra)==7 and all(type(row) is dict and set(row)=={'receipt','phase','returncode'} and
        type(row['returncode']) is int and row['returncode']==1 for row in extra),'Preserve exact three-key extra failed7 rows, eight failures total')
    failures=[failed]+[terminal(row['receipt'],row['phase'],1) for row in extra]
    failed_paths=[spec['prior_failed_CPU_receipt']['path']]+[row['receipt']['path'] for row in extra]
    require(len(set(failed_paths))==8 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
        'Normal ancestors and eight preserved failures must be distinct')
    all_rows=[dict(row,returncode=0) for row in rows]+[dict(receipt=spec['prior_failed_CPU_receipt'],phase=Path(spec['prior_failed_CPU_receipt']['path']).parent.name,returncode=1)]+extra
    pins=spec['terminal_identity_pins']; locators=[row['receipt']['path'] for row in all_rows]
    require(type(pins) is dict and set(pins)==set(locators),'Exact all-normal-plus-eight-failure identity pins required')
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
        parent['additional_failed_RAW_inputs']==cpu['additional_failed_RAW_inputs']==q['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'][:6] and
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
        cpu['additional_failed_RAW_inputs']==q['additional_failed_RAW_inputs']==report['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'][:6] and
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
    failed=spec['additional_failed_RAW_inputs'][5]
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


GPU_SCHEMA='ROOT-N13-sameK-fixed-fullDEV-GPU-spec-v1'
GPU_REPORT_SCHEMA='N13-sameK-fixed-fullDEV-GPU-report-v1'
GPU_STATUS='N13_SAMEK_FIXED_FULLDEV_GPU_COMPLETE_PENDING_CPU'
CPU_SCHEMA='ROOT-N13-sameK-fixed-fullDEV-CPU-spec-v1'
CPU_Q_SCHEMA='ROOT-N13-sameK-fixed-fullDEV-CPU-qualification-v1'
CPU_Q_STATUS='ROOT_N13_SAMEK_FIXED_FULLDEV_CPU_QUALIFIED_V1'
SELECTOR_SCHEMA='N13-sameK-native-all100-selector-manifest-v1'
SELECTOR_PACKET_SCHEMA='N13-sameK-native-all100-exact-GPU-L-v1'
N07_REPORT_SCHEMA='N07-original-detail-fixed-endpoint-evaluation-report-v1'
N07_STATUS='N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_COMPLETE_PENDING_CPU'
N07_Q_SCHEMA='ROOT-N07-original-detail-fixed-endpoint-evaluation-CPU-qualification-v1'
N07_Q_STATUS='ROOT_N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
EVAL_RECIPE=dict(images=187,raw_GT=380,queries_per_image=100,predictions_per_control=18700,
    arms=list(ARMS),native_weights_epoch=24,curve_epochs=[8,24],primary_epoch=48,best_DEV_selection=False,
    fixed_native_scores=True,fixed_native_class=True,fixed_native_query_order=True,all100_queries_retained=True,
    natural_GPU_render_batch=100,feature_channel_chunk=16,head_pixel_chunk=2048,render_dtype='float32',
    region_selector='exact saved GPU L only; sameK lex(Chebyshev opposite-class distance,absL,flat)',
    logits='where(exact arm ROI, native L + ungated PR residual, native L)',
    no_GT_query_selection=True,no_NMS=True,no_score_cutoff=True,no_TEST=True,
    native_model_loaded=False,native_forward_calls=0,original_JPEG_decoded=False,
    optimizer_created=False,weights_updated=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
    CPU_affinity=[0,1],threads=2,deadline_seconds=1800,automatic_retry=False)
NUMERIC=dict(class_rtol=1e-6,class_atol=1e-6,quality_rtol=1e-4,quality_atol=1e-4,
    refined_support_rtol=1e-4,refined_support_atol=1e-4,COCO_stats_atol=1e-12,
    feature_channel_chunk=16,head_pixel_chunk=2048,official_quality_eps=1e-6,
    exact_GPU_L_is_selector=True,region_boundary_allowance=False,certified_GPU_error_bound=False)
EVAL_REFS=('entry_source','endpoint_lineage_source','N10_evaluation_source','N10_evaluation_CPU_source',
    'region_common_source','point_heads_source','CPU_region_source','independent_head_source',
    'reader_source','renderer_source','export_source','COCO_source','prior_N07_report',
    'prior_N07_cpu_qualification','native_reference_predictions','DEV')
SELECTOR_PARENT_KEYS=('evaluation_entry_source','evaluation_spec','evaluation_report','evaluation_GPU_RAW',
    'evaluation_GPU_admission','CPU_entry_source','CPU_spec','CPU_qualification','CPU_RAW','CPU_admission')


def controls(seed,epoch):
    return ('native_official',)+tuple('%s_seed%d_E%02d_fixed_native_score'%(arm,seed,epoch) for arm in ARMS)


def check_recipe(spec):
    require(spec['recipe']==EVAL_RECIPE and spec['training_recipe']==RECIPE and spec['numeric_protocol']==NUMERIC and
        type(spec['seed']) is int and spec['seed'] in SEEDS and type(spec['head_endpoint_epoch']) is int and
        spec['head_endpoint_epoch'] in (8,24,48) and spec['controls']==list(controls(spec['seed'],spec['head_endpoint_epoch'])) and
        spec['deadline_seconds']==1800 and spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192,
        'Locked full187 paired N13 endpoint and resource recipe differs')


def own_gate(path,spec,entry_file,closure,raws,cpu=False):
    source,spec_ref=reference(entry_file),reference(path)
    require(source==spec['entry_source'] and all(spec[k] in closure.values() for k in EVAL_REFS),
        'Actual current entry and complete evaluation primary closure required')
    key='CPU_admission_file' if cpu else 'GPU_admission_file'
    adpath=Path(spec[key]); require(adpath.is_absolute(),'Absolute actual own admission required')
    adref=reference(adpath); ad=read(adpath)
    require(ad['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and ad['driver_sha256']==OWNER_SHA and
        ad['phase']==spec['phase'] and ad['scientific_spec']==spec_ref and ad['entry']==source['path'] and
        ad['entry_args']==['--spec',str(path),'--owner-execute'] and ad['allowed_cpus']==[0,1] and
        ad['cuda_visible_devices']==('' if cpu else '0') and ad['automatic_retry'] is False and
        ad['whole_GPU_used_limit_MiB']==8192 and ad['qualified_leaf_deadline_seconds']==1800 and
        ad['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
        all(ad['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']) and
        os.environ.get('CUDA_VISIBLE_DEVICES')==('' if cpu else '0') and all(os.environ.get(k)=='2' for k in THREAD_ENV),
        'Actual single ROOT owner/source/resource/environment differs')
    owners=[owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    return dict(source=source,spec_ref=spec_ref,admission_ref=adref,admission=ad,closure=closure,owners=owners)


def endpoint_gate(spec,closure,owners):
    endpoint=spec['endpoint']; epoch=spec['head_endpoint_epoch']
    require(type(endpoint) is dict and endpoint.get('type') in
        ('QUALIFIED_SUCCESSFUL_GPU_ENDPOINT','CPU_ADOPTED_FAILED_GPU_ENDPOINT'),'Explicit endpoint branch required')
    context=dict(spec,recipe=RECIPE,first_epoch=epoch+1,last_epoch=epoch+8,resume=endpoint)
    if endpoint['type']=='CPU_ADOPTED_FAILED_GPU_ENDPOINT':
        require(spec['seed']==202610120701 and epoch==8 and
            spec['recovered_endpoint_use']=='ROOT_EXPLICIT_FIXED_E08_CURVE_EVALUATION',
            'New ROOT admission must explicitly authorize only this fixed E08 curve')
    else:
        require(spec['recovered_endpoint_use'] is None,'Successful endpoint cannot carry adoption permission')
    parent=parent_gate(context,closure,owners)
    common=load(spec['region_common_source'],'_N13_eval_original_region_helpers')
    adapter=load(spec['cache_contract_source'],'_N13_eval_actual_cache_contract')
    facade=adapter.bind_common(common)
    require(spec['validation_protocol']==adapter.VALIDATION_PROTOCOL and common.ARMS==ARMS,
        'Exact qualified cache/region external gate ABI required')
    cached,cache_report,cache_q=facade.validate_cache_input(spec,closure,owners)
    factory_q=read(bound(spec['PR_factory_CPU_qualification']))
    original,fresh=spec['original_PR_point_heads_source'],spec['point_heads_source']
    require(factory_q['schema']=='ROOT-N10-C-PR-FL91-synthetic-head-CPU-qualification-v1' and
        factory_q['status']=='ROOT_N10_C_PR_FL91_SYNTHETIC_HEAD_CPU_QUALIFIED' and
        factory_q['PR_point_heads_source']==original and factory_q['PR_parameters']==40497 and
        factory_q['synthetic_cold_identity_and_band_checked'] is True and
        factory_q['synthetic_backward_update_checked'] is True and factory_q['synthetic_CPU_only'] is True and
        original['bytes']==fresh['bytes'] and original['sha256']==fresh['sha256'] and
        all(factory_q[k] in closure.values() for k in ('spec','source','CPU_admission')),
        'Evaluation PR factory must remain exactly the qualified byte-identical 8/40497 factory')
    synthetic=owner_ref_join(spec,owners,spec['RAW_inputs'][28]['receipt'],factory_q['CPU_admission'],
        factory_q['source'],factory_q['spec'],'')
    require(synthetic==28,'Original PR factory qualification identity differs')
    return dict(parent=parent,common=common,cache_report=cache_report,cache_q=cache_q,cached=cached)


def prior_gate(spec,closure,owners):
    prior=read(bound(spec['prior_N07_report'])); q=read(bound(spec['prior_N07_cpu_qualification']))
    require(prior['schema']==N07_REPORT_SCHEMA and prior['status']==N07_STATUS and
        q['schema']==N07_Q_SCHEMA and q['status']==N07_Q_STATUS and
        q['evaluation_report']==spec['prior_N07_report'] and q['source']==prior['source'] and
        q['evaluation_spec']==prior['spec'] and q['reader_source']==spec['reader_source'] and
        prior['DEV']==q['DEV']==spec['DEV'] and prior['images']==q['images']==187 and
        prior['raw_GT']==q['raw_GT']==380 and prior['predictions_per_control']==q['predictions_per_control']==18700 and
        prior['predictions']['native_official']==spec['native_reference_predictions']==q['control_results']['native_official']['predictions'] and
        q['complete_identity_fixed_scores_and_empty_query_population_CPU_checked'] is True and
        q['five_official_COCO12_CPU_recomputed'] is True and q['strong_E24_native_export_and_COCO_reference_CPU_checked'] is True and
        q['native_M200_original_render_CPU_replayed'] is True and
        q['packet_refs']==prior['final_field_packets'] and len(q['packet_refs'])==187,
        'Original full18700 native fields and independent CPU qualification differ')
    gi=owner_ref_join(spec,owners,q['evaluation_GPU_RAW'],prior['GPU_admission'],prior['source'],prior['spec'],'0')
    ci=owner_ref_join(spec,owners,spec['RAW_inputs'][next(i for i,o in enumerate(owners) if o['scientific_spec']==q['spec'])]['receipt'],
        q['CPU_admission'],q['reader_source'],q['spec'],'')
    require(gi<ci<31,'Original N07 GPU/CPU chronology differs')
    prior_spec=read(bound(prior['spec'])); cpu_spec=read(bound(q['spec']))
    require(prior_spec['official_export_source']==spec['export_source'] and
        prior_spec['official_evaluation_CPU_reader']==spec['COCO_source'] and cpu_spec['renderer_source']==spec['renderer_source'] and
        all(ref in closure.values() for payload in (prior_spec,cpu_spec) for ref in payload['sources']) and
        all(prior[k] in closure.values() for k in ('image_rows','trace','native_query_score_factors')) and
        all(ref in closure.values() for ref in q['packet_refs']),'Full original native source/packet closure differs')
    return prior,q


def selector_parent_gate(spec,closure,owners):
    mode=spec['selector_mode']; parent=spec['selector_parent']
    require(mode in ('produce','reuse'),'Explicit selector mode required')
    if mode=='produce':
        require(parent is None and spec['seed']==202610120701 and spec['head_endpoint_epoch']==8,
            'Only the prelocked first seed701 E08 pair produces the shared selector')
        return None
    require(type(parent) is dict and set(parent)==set(SELECTOR_PARENT_KEYS) and
        all(parent[k] in closure.values() for k in SELECTOR_PARENT_KEYS),'Complete immutable selector parent refs required')
    gpu=read(bound(parent['evaluation_spec'])); report=read(bound(parent['evaluation_report']))
    cpu=read(bound(parent['CPU_spec'])); q=read(bound(parent['CPU_qualification']))
    require(gpu['schema']==GPU_SCHEMA and report['schema']==GPU_REPORT_SCHEMA and report['status']==GPU_STATUS and
        gpu['selector_mode']==report['selector_mode']=='produce' and
        report['source']==parent['evaluation_entry_source'] and report['spec']==parent['evaluation_spec'] and
        report['GPU_admission']==parent['evaluation_GPU_admission'] and cpu['schema']==CPU_SCHEMA and
        q['schema']==CPU_Q_SCHEMA and q['status']==CPU_Q_STATUS and
        q['source']==parent['evaluation_entry_source'] and q['reader_source']==parent['CPU_entry_source'] and
        q['spec']==parent['CPU_spec'] and q['CPU_admission']==parent['CPU_admission'] and
        q['evaluation_spec']==parent['evaluation_spec'] and q['evaluation_report']==parent['evaluation_report'] and
        q['evaluation_GPU_RAW']==parent['evaluation_GPU_RAW'] and q['evaluation_GPU_admission']==parent['evaluation_GPU_admission'] and
        q['selector_manifest']==report['selector_manifest'] and q['exact_GPU_L_regions_all18700_CPU_checked'] is True and
        q['all18700_identity_fixed_score_and_order_CPU_checked'] is True and q['current_owned_CPU_terminal_self_certified'] is False and
        all(gpu[k]==spec[k] for k in ('DEV','native_reference_predictions','prior_N07_report','prior_N07_cpu_qualification','renderer_source')) and
        all(ref in closure.values() for payload in (gpu,cpu) for ref in payload['sources']),
        'Shared selector requires its actual producer GPU0 and independent CPU0 qualification')
    gi=owner_ref_join(spec,owners,parent['evaluation_GPU_RAW'],parent['evaluation_GPU_admission'],
        parent['evaluation_entry_source'],parent['evaluation_spec'],'0')
    ci=owner_ref_join(spec,owners,parent['CPU_RAW'],parent['CPU_admission'],parent['CPU_entry_source'],parent['CPU_spec'],'')
    require(gi<ci and gpu['RAW_inputs']==spec['RAW_inputs'][:gi] and cpu['RAW_inputs']==spec['RAW_inputs'][:ci] and
        q['RAW_inputs']==cpu['RAW_inputs'] and cpu['evaluation_entry_source']==parent['evaluation_entry_source'] and
        cpu['evaluation_spec']==parent['evaluation_spec'] and cpu['evaluation_report']==parent['evaluation_report'] and
        cpu['evaluation_GPU_RAW']==parent['evaluation_GPU_RAW'] and
        all(spec['terminal_identity_pins'].get(k)==v for payload in (gpu,cpu) for k,v in payload['terminal_identity_pins'].items()),
        'Shared selector producer owner/prefix/identity join differs')
    manifest=read(bound(report['selector_manifest']))
    require(manifest['schema']==SELECTOR_SCHEMA and manifest['source']==parent['evaluation_entry_source'] and
        manifest['spec']==parent['evaluation_spec'] and manifest['GPU_admission']==parent['evaluation_GPU_admission'] and
        manifest['images']==187 and manifest['queries']==18700 and len(manifest['rows'])==187 and
        all(row['carrier'] in closure.values() for row in manifest['rows']),
        'Full immutable shared-selector manifest/carrier closure differs')
    return dict(manifest=manifest,manifest_ref=report['selector_manifest'],report=report,parent=parent)


def final_endpoint_check(parent):
    if 'missing_original_GPU_report' in parent:
        p=Path(parent['missing_original_GPU_report'])
        require(not p.exists() and not p.is_symlink(),'Original failed GPU REPORT must remain absent')
