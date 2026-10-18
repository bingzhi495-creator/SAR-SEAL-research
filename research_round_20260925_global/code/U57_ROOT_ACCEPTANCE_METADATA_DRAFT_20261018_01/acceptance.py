"""UNUSED pure metadata assembler for a future actual Root U57 acceptance.

No file reads, hashes, ML imports, remote calls, ledger writes or execution.
Root must supply its independently observed terminal and actual byte postcheck.
"""
from copy import deepcopy
from pathlib import PurePosixPath


SPEC_SCHEMA='ROOT-U57-DGX-original-update-training-spec-v1'
OPERATION='GPU_U57_DGX_ORIGINAL_UPDATE_TRAINING'
POST_SCHEMA='U57-Root-training-postcheck-v1'
ARMS={'PLAIN','FULL'}
COMPONENTS={'shared_initialization','full_legacy','full_dsconv','full_head','full_feature',
            'plain_legacy','plain_feature','instance_adapter','image_geometry','u36_helpers',
            'u39_training_ops','portable_reader'}
FLAGS={'source_bytes_verified','CUDA_uninitialized_before','source_unchanged_after',
       'module_aliases_restored','caller_CPU_RNG_preserved','caller_NumPy_RNG_preserved',
       'caller_Python_RNG_preserved','dtype_grad_threads_preserved','backend_controls_restored',
       'child_resources_within_limits','caller_determinism_controls_restored'}
POLICY=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic_algorithms_enabled=False,
    deterministic_warn_only=False,cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False,
    cudnn_benchmark=False,cudnn_deterministic=True)


def need(ok,message):
    if not ok:raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and PurePosixPath(value['path']).is_absolute()
         and str(PurePosixPath(value['path']))==value['path'] and '..' not in PurePosixPath(value['path']).parts
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Exact remote path/bytes/SHA metadata, never a file read')
    return value


def cursor(step):
    return dict(global_updates=step,completed_epochs=step//741,within_epoch=step%741)


def budget_credit(mode,start,stop):
    need(mode in ('FORMAL','SMOKE') and type(start) is int and type(stop) is int
         and 0<=start<stop<=37050,'Explicit nonempty same two-arm span')
    if mode=='SMOKE':need((start,stop) in ((0,1),(1,2)),'Only the two prelocked disposable smoke segments')
    delta=stop-start
    return dict(updates_per_arm=delta,aggregate_optimizer_updates=2*delta,
                formal_credit_per_arm=delta if mode=='FORMAL' else 0,
                formal_credit_aggregate=2*delta if mode=='FORMAL' else 0)


def validate_terminal(value):
    expected=dict(root_observed_tool_exit_code=0,SSH_returncode=0,worker_returncode=0,
        reaped=True,owner_original_birth_absent=True,child_original_birth_absent=True,
        remaining_owned_group=[],all_owned_processes_absent=True)
    need(type(value) is dict and set(value)==set(expected)
         and all(type(value[k]) is type(wanted) and value[k]==wanted for k,wanted in expected.items()),
         'Actual Root/SSH/worker zero, real reap, both original births absent and empty owned group')


def validate_byte_evidence(value):
    keys={'source_and_logs_byte_verified','metadata_records_byte_verified','outputs_byte_verified'}
    need(type(value) is dict and keys<=set(value) and all(value[key] is True for key in keys),
         'All actual Root input/log/spec/report/checkpoint/trace/pair byte verifications are required')


def _refs(values):
    result={}
    for value in values:
        strict_ref(value)
        need(value['path'] not in result or result[value['path']]==value,'No conflicting reference for one path')
        result[value['path']]=value
    return [result[key] for key in sorted(result)]


def _spec(spec):
    keys={'schema','operation','phase','entry_source','body_source','owner_source','source_refs',
          'library_sources','context','package_refs','segment','prelock','output_directory','owner_output_directory','sources'}
    need(type(spec) is dict and set(spec)==keys and spec['schema']==SPEC_SCHEMA and spec['operation']==OPERATION,
         'The admitted U57 spec, never a U40/U56 substitution')
    pre=spec['prelock'];span=spec['segment'];context=spec['context']
    need(type(pre) is dict and pre['schema']=='U57-DGX-original-update-training-prelock-v1'
         and type(pre['run_id']) is str and pre['run_id'] and pre['backend_policy']==POLICY,
         'Explicit independent mode/run and nonstrict original-update backend policy')
    need(type(span) is dict and set(span)=={'start','stop'},'Exact paired segment bounds')
    credit=budget_credit(pre['mode'],span['start'],span['stop'])
    need(type(context) is dict and set(context)=={'manifest_ref','order_ref','order_generation_ref',
         'U54_validation_ref','U54_validation_sources','parent_pair'},'Explicit admitted source/model-parent context')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])==COMPONENTS
         and type(spec['library_sources']) is dict and set(spec['library_sources'])=={'psutil','torch','numpy'}
         and type(spec['package_refs']) is list and len(spec['package_refs'])==741
         and len({strict_ref(v)['path'] for v in spec['package_refs']})==741,'Twelve components and all741 package identities')
    refs=[spec[k] for k in ('entry_source','body_source','owner_source')]
    refs+=list(spec['source_refs'].values())+list(spec['library_sources'].values())+spec['package_refs']
    refs+=[context[k] for k in ('manifest_ref','order_ref','order_generation_ref','U54_validation_ref')]
    need(type(context['U54_validation_sources']) is dict
         and set(context['U54_validation_sources'])=={'worker_report_ref','owner_receipt_ref'},'Both original U54 evidence refs')
    refs+=list(context['U54_validation_sources'].values())
    parent=context['parent_pair']
    if span['start']==0:need(parent is None,'New independent zero cursor has no old/hidden state')
    else:
        need(type(parent) is dict and set(parent)=={'qualified_ref','pair_commit_ref','checkpoint_refs','trace_refs'}
             and set(parent['checkpoint_refs'])==set(parent['trace_refs'])==ARMS,'Explicit prior Root-accepted complete paired parent')
        refs+=[parent['qualified_ref'],parent['pair_commit_ref']]+list(parent['checkpoint_refs'].values())+list(parent['trace_refs'].values())
    need(spec['sources']==_refs(refs),'Exact full admitted source closure; no omitted or appended payloads')
    return pre,span,context,credit


def _zero_gpu(value):
    return type(value) is dict and set(value)=={'allocated','reserved'} and all(type(v) is int and v==0 for v in value.values())


def accept_training(spec,original_terminal,owner_receipt,worker_report,postcheck):
    """Return a candidate Root acceptance only from explicit actual evidence.

    This function does not observe anything, create files, advance a ledger,
    prevent duplicate accounting, or independently replay numerical training.
    """
    validate_terminal(original_terminal)
    validate_byte_evidence(postcheck)
    pre,span,context,credit=_spec(spec)
    need(type(postcheck) is dict and set(postcheck)=={'schema','spec_ref','worker_report_ref','owner_receipt_ref',
         'logs','owner_identity','child_identity','verified_source_refs','verified_output_refs',
         'source_and_logs_byte_verified','metadata_records_byte_verified','outputs_byte_verified','pair_commit'}
         and postcheck['schema']==POST_SCHEMA,'Actual Root postcheck with explicit byte-verified objects')
    for key in ('spec_ref','worker_report_ref','owner_receipt_ref'):strict_ref(postcheck[key])
    owner=owner_receipt;worker=worker_report
    need(type(owner) is dict and owner['schema']=='U57-Linux-owned-original-update-training-receipt-v1'
         and owner['status']=='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW'
         and type(owner['returncode']) is int and owner['returncode']==0
         and owner['reaped'] is owner['original_birth_absent'] is owner['final_group_scan_resolved'] is True
         and owner['all_owned_processes_absent'] is True and owner['child_alive'] is False
         and owner['remaining_owned_group']==[] and owner['violations']==[] and owner['qualified'] is False
         and owner['paired_segment_complete'] is True,'Actual successful owned child and cleanup, not a self-certified terminal')
    need(type(worker) is dict and worker['schema']=='U57-DGX-original-update-training-worker-report-v1'
         and worker['status']=='COMPLETE' and worker['operation']==spec['operation'] and worker['phase']==spec['phase']
         and owner['operation']==spec['operation'] and owner['phase']==spec['phase']
         and owner['spec_ref']==worker['spec_ref']==postcheck['spec_ref']
         and owner['worker_report_ref']==postcheck['worker_report_ref']
         and owner['owner_identity']==postcheck['owner_identity'] and owner['child_identity']==postcheck['child_identity'],
         'Same actual owner/worker/spec bytes, phase and original births')
    need(all(worker[k]==spec[k] for k in ('entry_source','body_source','owner_source','source_refs','library_sources',
         'context','package_refs','segment','prelock')) and postcheck['verified_source_refs']==spec['sources'],
         'Root verified every input/source used by the actual worker, including parent state')
    need(type(owner['logs']) is dict and set(owner['logs'])=={'stdout.log','stderr.log'}
         and owner['logs']==postcheck['logs'],'Both actual original owner-captured logs are byte-verified')
    for value in owner['logs'].values():strict_ref(value)
    need(type(worker['flags']) is dict and set(worker['flags'])==FLAGS
         and all(worker['flags'][k] is True for k in FLAGS) and worker['cleanup_errors']==[],
         'Complete source/backend/caller/RNG/resource evidence; missing flags cannot vacuously pass')
    need(worker['strict_backend_settings_before_body']==worker['strict_backend_settings_after_body']==POLICY
         and worker['CUBLAS_WORKSPACE_CONFIG_before_Torch_import']==':4096:8'
         and worker['outside_determinism_before']==worker['outside_determinism_after'],
         'The prelocked nonstrict U57 policy and exact caller restoration, not strict U56 certification')
    body=worker['body_report'];pair=postcheck['pair_commit'];delta=credit['updates_per_arm']
    need(body['schema']=='U57-DGX-original-update-training-segment-report-v1'
         and body['status']=='SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE'
         and body['mode']==worker['mode']==owner['mode']==pre['mode']
         and body['run_id']==worker['run_id']==owner['run_id']==pre['run_id']
         and body['span']==span and body['seed']==202610173101
         and body['paired_checkpoint_prefix']==worker['paired_checkpoint_prefix']==owner['paired_checkpoint_prefix']==cursor(span['stop']),
         'Same independent run/profile, complete span and per-arm global cursor')
    need(all(type(record[k]) is int and record[k]==credit['aggregate_optimizer_updates']
         for record in (owner,worker,body) for k in ('successful_optimizer_updates','optimizer_steps_entered'))
         and all(record['new_training'] is True for record in (owner,worker,body))
         and all(type(record['formal_training_budget_credit']) is int and record['formal_training_budget_credit']==0
                 for record in (owner,worker,body))
         and all(record['pilot_only'] is (pre['mode']=='SMOKE') for record in (owner,worker,body))
         and worker['main_training'] is owner['main_training'] is (pre['mode']=='FORMAL')
         and all(record['formal_main_training_started'] is (pre['mode']=='FORMAL') for record in (owner,worker,body)),
         'Actual two-arm returned/entered updates; smoke never receives formal credit')
    need(all(record[k] is False for record in (owner,worker,body) for k in ('qualified','AP','DEV','TEST','paired_prefix_advanced'))
         and owner['U40_frontier_modified'] is worker['U40_frontier_modified'] is False
         and _zero_gpu(body['final_GPU_cleanup']) and body['cleanup_errors']==[]
         and body['backend_settings_before']==body['backend_settings_after']==POLICY
         and all(body[k] is True for k in ('source_unchanged_after','caller_CPU_RNG_restored','caller_NumPy_RNG_restored',
              'caller_Python_RNG_restored','caller_postinitialization_CUDA_RNG_restored')),
         'Complete source/RNG/GPU cleanup, no evaluation or prior self-advancement')
    for key in ('checkpoint_refs','trace_refs'):
        need(type(body[key]) is dict and set(body[key])==ARMS and worker[key]==owner[key]==body[key],
             'Both complete checkpoint and trace references must agree')
    need(body['pair_commit_ref']==worker['pair_commit_ref']==owner['pair_commit_ref'],'Same pending pair output ref')
    outputs=list(body['checkpoint_refs'].values())+list(body['trace_refs'].values())+[body['pair_commit_ref']]
    need(len({strict_ref(v)['path'] for v in outputs})==5 and postcheck['verified_output_refs']==_refs(outputs),
         'Root byte-verified all two checkpoints/two traces/pair, no file omitted')
    need(type(pair) is dict and pair['schema']=='U57-DGX-paired-checkpoint-commit-v1'
         and pair['status']=='PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE'
         and pair['identity']==body['identity'] and pair['span']==span
         and pair['cursor']==cursor(span['stop']) and pair['order_ref']==context['order_ref']
         and pair['previous_pair']==(None if context['parent_pair'] is None else context['parent_pair']['pair_commit_ref'])
         and pair['checkpoint_refs']==body['checkpoint_refs'] and pair['trace_refs']==body['trace_refs']
         and pair['actual_trace_sha256']==body['actual_trace_sha256'],
         'Actual byte-verified pair JSON binds this identity/span/order/parent and outputs')
    need(set(body['arms'])==set(pair['checkpoints'])==set(pair['traces'])==ARMS,'Exactly both required arms')
    for arm in ('PLAIN','FULL'):
        record=body['arms'][arm];checkpoint=pair['checkpoints'][arm]
        need(record['status']=='CHECKPOINT_WRITTEN' and type(record['updates_completed']) is int and record['updates_completed']==delta
             and record['start_cursor']==cursor(span['start']) and record['end_cursor']==cursor(span['stop'])
             and record['restored_all_state_exact'] is (True if span['start'] else None)
             and record['checkpoint']==checkpoint and checkpoint['reference']==body['checkpoint_refs'][arm]
             and checkpoint['serialization_roundtrip_verified'] is True and _zero_gpu(record['GPU_cleanup'])
             and pair['traces'][arm]==record['trace_ref']==body['trace_refs'][arm]
             and record['actual_trace_sha256']==body['actual_trace_sha256'],
             'Each arm finished the actual shared trace, complete state roundtrip/restore and release')
    result=dict(pair_commit_ref=deepcopy(body['pair_commit_ref']),checkpoint_refs=deepcopy(body['checkpoint_refs']),
        trace_refs=deepcopy(body['trace_refs']),order_ref=deepcopy(context['order_ref']),paired_checkpoint_prefix=cursor(span['stop']))
    return dict(schema='Root-U57-DGX-original-update-training-accepted-v1',root_accepted=True,
        mode=pre['mode'],run_id=pre['run_id'],paired_prefix_advanced=True,result=result,
        accepted_span=deepcopy(span),budget=credit,formal_training_budget_credit=credit['formal_credit_aggregate'],
        formal_training_budget_credit_unit='aggregate_image_optimizer_updates_across_both_arms',
        cursor_unit='image_optimizer_updates_per_arm',original_terminal=deepcopy(original_terminal),
        evidence_refs={k:deepcopy(postcheck[k]) for k in ('spec_ref','worker_report_ref','owner_receipt_ref')},
        acceptance_scope='Root actual terminal and byte postcheck plus source-closed worker state/trace metadata; no neural replay',
        AP=False,DEV=False,TEST=False,ROOT_neural_replay=False,ledger_modified=False)
