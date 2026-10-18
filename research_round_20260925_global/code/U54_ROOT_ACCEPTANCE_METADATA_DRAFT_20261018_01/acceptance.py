"""SOURCE-ONLY, not used on an actual run. Pure Root metadata checks; no I/O.

The caller must supply genuine, independently observed terminal/postcheck data.
This module cannot establish that a process exited or that file bytes were read.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath

SPEC_SCHEMA='ROOT-U54-DGX-full-FIT-CPU-validation-spec-v1'
OPERATION='CPU_U54_DGX_FULL_FIT_PORTABLE_VALIDATION'
LIMITS=dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
    minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
ENVIRONMENT=dict(python='3.12.14',torch='2.10.0+cu130',numpy='1.26.4',system='Linux',machine='aarch64')
WORKER_FLAGS={'source_bytes_verified','CUDA_uninitialized_before','CUDA_uninitialized_after',
    'source_unchanged_after','module_aliases_restored','caller_CPU_RNG_preserved',
    'caller_NumPy_RNG_preserved','caller_Python_RNG_preserved','dtype_grad_threads_preserved',
    'child_resources_within_limits'}
BODY_FLAGS={'CUDA_uninitialized_before','CUDA_uninitialized_after','metadata_population_verified',
    'all_field_proofs_verified','all_returned_tensor_objects_released'}
FALSE_FIELDS=('qualified','new_training','pilot_only','main_training','AP','DEV','TEST',
    'U40_frontier_modified','innovation_established','speedup_established')
BODY_FALSE=('qualified','new_training','AP','DEV','TEST','main_training','model_constructed',
    'model_forward_executed','optimizer_created','GT_matching_recomputed')
EXIT_FIELDS=('root_observed_tool_exit_code','SSH_returncode','worker_returncode')
TERMINAL_FLAGS=('reaped','owner_original_birth_absent','child_original_birth_absent',
    'all_owned_processes_absent','source_and_logs_byte_verified')


def need(ok,message):
    if not ok:raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
        and type(value['path']) is str and PurePosixPath(value['path']).is_absolute()
        and str(PurePosixPath(value['path']))==value['path'] and '..' not in PurePosixPath(value['path']).parts
        and type(value['bytes']) is int and value['bytes']>=0 and type(value['sha256']) is str
        and len(value['sha256'])==64 and set(value['sha256'])<=set('0123456789abcdef'),
        'Explicit POSIX path/bytes/SHA metadata, never an opened file')
    return value


def canonical_refs(values):
    need(type(values) is list and values,'Explicit finite reference list')
    result={}
    for value in values:
        strict_ref(value);path=value['path']
        need(path not in result or result[path]==value,'Conflicting reference metadata')
        result[path]=value
    return [result[path] for path in sorted(result)]


def package_refs_sha256(values):
    """Hash only caller-supplied reference JSON, preserving original ordinal order."""
    need(type(values) is list and values,'Explicit ordered package reference metadata')
    for value in values:strict_ref(value)
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':'),
        ensure_ascii=True,allow_nan=False).encode('utf-8')).hexdigest()


def validate_original_terminal(value):
    need(type(value) is dict and set(value)==set(EXIT_FIELDS)|set(TERMINAL_FLAGS),
        'Original terminal must have exactly three external exit codes and five closure flags')
    need(all(type(value[key]) is int and value[key]==0 for key in EXIT_FIELDS)
        and all(value[key] is True for key in TERMINAL_FLAGS),
        'Genuine externally observed zero exits, wait/reap, both births absent and Root byte postcheck required')
    return deepcopy(value)


def _count(value,expected,label):
    need(type(value) is int and value==expected,label)


def _duration(value,label):
    need(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1800,label)


def _resources(elapsed,rss,available):
    _duration(elapsed,'Finite original CPU duration within1800s')
    need(type(rss) is int and 0<=rss<=16*1024**3 and type(available) is int and available>=16*1024**3,
        'Actual RSS/free RAM within original16GiB limits')


def _spec(spec):
    keys={'schema','operation','phase','entry_source','body_source','owner_source','source_refs',
        'library_sources','manifest_ref','package_refs','prelock','output_directory','owner_output_directory','sources'}
    need(type(spec) is dict and set(spec)==keys and spec['schema']==SPEC_SCHEMA
        and spec['operation']==OPERATION and type(spec['phase']) is str and spec['phase'],
        'Exact supplied U54 CPU validation spec')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])=={'portable_reader','image_geometry'}
        and type(spec['library_sources']) is dict and set(spec['library_sources'])=={'psutil','torch','numpy'},
        'Exact two modules and three genuine selected library roles')
    packages=spec['package_refs']
    need(type(packages) is list and len(packages)==741
        and len({strict_ref(value)['path'] for value in packages})==741,'One distinct original-ordinal package per741 images')
    values=[spec[key] for key in ('entry_source','body_source','owner_source','manifest_ref')]
    values+=list(spec['source_refs'].values())+list(spec['library_sources'].values())+packages
    expected=canonical_refs(values)
    need(len(expected)==750 and spec['sources']==expected,
        'Actual750 sources exactly derived from supplied spec, no guessed hidden inputs')
    pre=spec['prelock']
    need(pre==dict(schema='U54-full-FIT-CPU-validation-prelock-v1',environment=ENVIRONMENT,
        dataset_indices=list(range(741)),images=741,GT=1661,CUDA_visible_devices='',limits=LIMITS,
        no_new_training=True,no_forward=True,no_matching=True,no_AP=True,no_DEV=True,no_TEST=True),
        'Unchanged complete CPU validation prelock and limits')
    for key in ('output_directory','owner_output_directory'):
        value=spec[key]
        need(type(value) is str and value!='/' and PurePosixPath(value).is_absolute()
            and str(PurePosixPath(value))==value and '..' not in PurePosixPath(value).parts,'Explicit canonical output metadata')
    return expected


def _postcheck(spec,owner,worker,post,expected_sources):
    keys={'schema','spec_ref','worker_report_ref','owner_receipt_ref','private_audit_ref','logs',
        'verified_source_refs','verified_output_refs','owner_identity','child_identity','source_and_logs_byte_verified'}
    need(type(post) is dict and set(post)==keys and post['schema']=='Root-U54-explicit-byte-postcheck-v1'
        and post['source_and_logs_byte_verified'] is True,'Real Root byte postcheck, never a worker self-claim')
    refs=[strict_ref(post[key]) for key in ('spec_ref','worker_report_ref','owner_receipt_ref','private_audit_ref')]
    need(type(post['logs']) is dict and set(post['logs'])=={'stdout.log','stderr.log'},'Both original owned logs')
    outputs=canonical_refs(refs+list(post['logs'].values()))
    need(len(outputs)==6 and post['verified_output_refs']==outputs
        and post['verified_source_refs']==expected_sources,
        'Root actually byte-verified all750 Sources and exact spec/worker/owner/audit/two logs')
    need(post['spec_ref']==owner['spec_ref']==worker['spec_ref']
        and post['worker_report_ref']==owner['worker_report_ref']
        and post['private_audit_ref']==worker['private_audit_ref']==owner['private_audit_ref']
        and post['logs']==owner['logs'],'One exact original spec, worker, audit and log identity')
    worker_out=PurePosixPath(spec['output_directory']);owner_out=PurePosixPath(spec['owner_output_directory'])
    need(post['worker_report_ref']['path']==str(worker_out/'REPORT.json')
        and post['private_audit_ref']['path']==str(worker_out/'IMAGE_CHECKS.jsonl')
        and post['private_audit_ref']['bytes']>0
        and post['owner_receipt_ref']['path']==str(owner_out/'OWNER_RECEIPT.json')
        and all(post['logs'][key]['path']==str(owner_out/key) for key in post['logs']),
        'All verified outputs belong to this original owned attempt')
    need(post['owner_identity']==owner['owner_identity'] and post['child_identity']==owner['child_identity'],
        'Root external absence checks refer to the original owner and child births')
    for key in ('owner_identity','child_identity'):
        identity=post[key]
        need(type(identity) is dict and type(identity.get('pid')) is int and identity['pid']>0
            and type(identity.get('start_ticks')) is int and identity['start_ticks']>0,'Recorded actual Linux birth identity')
    child=post['child_identity']
    need(child['process_group']==child['session']==child['pid'] and owner['owner_pid']==post['owner_identity']['pid'],
        'Original owned child session/group and original owner PID')


def accept_validation(spec,original_terminal,owner_receipt,worker_report,postcheck):
    """Pure dictionary API. Invoke only after Root really collected every input.

    It neither reads/hashes payloads nor changes files, Ops, RAW counts or U40.
    Root must save/hash the returned JSON separately after genuine acceptance.
    """
    terminal=validate_original_terminal(original_terminal)
    need(type(postcheck) is dict and postcheck.get('source_and_logs_byte_verified') is True,
        'Missing actual Root byte postcheck cannot be replaced by COMPLETE worker metadata')
    expected_sources=_spec(spec);owner=owner_receipt;worker=worker_report
    need(type(owner) is dict and type(worker) is dict,'Parsed original owner/worker JSON metadata')
    need(owner['schema']=='U54-Linux-owned-full-FIT-CPU-validation-receipt-v1'
        and owner['status']=='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW'
        and owner['operation']==spec['operation'] and owner['phase']==spec['phase']
        and type(owner['returncode']) is int and owner['returncode']==0
        and owner['reaped'] is owner['original_birth_absent'] is owner['final_group_scan_resolved'] is True
        and owner['all_owned_processes_absent'] is owner['CPU_only'] is owner['CPU_validation_complete'] is True
        and owner['child_alive'] is owner['termination_requested'] is False
        and owner['remaining_owned_group']==owner['violations']==owner['signals']==[]
        and owner['CUDA_visible_devices']=='' and owner['GPU_allocator_contract_applicable'] is False,
        'Actual original CPU owner completed, waited/reaped child and resolved the entire absent owned group')
    need(worker['schema']=='U54-full-FIT-CPU-validation-worker-report-v1' and worker['status']=='COMPLETE'
        and worker['operation']==spec['operation'] and worker['phase']==spec['phase']
        and worker['actual_environment']==ENVIRONMENT and worker['cleanup_errors']==[],
        'Complete actual independent CPU worker, not a planned or failed input check')
    for key in ('entry_source','body_source','owner_source','source_refs','library_sources','manifest_ref','package_refs','prelock'):
        need(worker[key]==spec[key],'One admitted original source/input chain: '+key)
    need(worker['actual_library_paths']=={key:value['path'] for key,value in spec['library_sources'].items()},
        'Genuine selected imported library locations')
    need(set(worker['flags'])==WORKER_FLAGS and all(value is True for value in worker['flags'].values()),
        'All ten actual CPU/source/caller restoration flags')
    body=worker['body_report']
    need(body['schema']=='U54-full-FIT-CPU-validation-body-report-v1' and body['status']=='COMPLETE_PENDING_ROOT'
        and set(body['flags'])==BODY_FLAGS and all(value is True for value in body['flags'].values()),
        'All five actual metadata, field-proof, release and cold-CUDA checks')
    for key in ('attempted_images','validated_images','released_images','recorded_images'):_count(body[key],741,'Complete741 '+key)
    for obj in (worker,owner):
        _count(obj['validated_images'],741,'Actual741 complete images')
        _count(obj['matched_instances'],1661,'Actual1661 original matched instances')
        _count(obj['audit_records_written'],741,'Actual741 private scalar audit rows')
    _count(body['matched_instances'],1661,'Actual1661 loaded full-G targets')
    _count(body['reader_failed_loads'],0,'No failed image/field proof skipped')
    _count(body['remaining_returned_tensor_objects'],0,'All returned current-image tensors released')
    need(type(body['reader_visits']) is list and all(type(index) is int for index in body['reader_visits'])
        and body['reader_visits']==list(range(741)),'Every original index once, in original order')
    for obj in (worker,owner):
        need(all(obj[key] is False for key in FALSE_FIELDS) and type(obj['formal_training_budget_credit']) is int
            and obj['formal_training_budget_credit']==0,'Input validation never grants model/evaluation/training credit')
    need(all(body[key] is False for key in BODY_FALSE) and body['formal_training_budget_credit']==0,
        'CPU input proof is not forward, matching, optimizer work or self-qualification')
    need(body['FIT_GT_tensor_read_attempted'] is body['FIT_GT_tensors_read'] is True
        and body['original_annotation_records_parsed'] is False,'Actual inherited GT tensor access is explicitly disclosed')
    observations=worker['resource_observations']
    need(observations['schema']=='U54-Linux-child-control-observations-v1' and type(observations['samples']) is int
        and observations['samples']>0 and observations['violations']==[], 'Actual sampled child resource evidence')
    _resources(observations['elapsed_seconds'],observations['peak_RSS_bytes'],observations['minimum_available_RAM_bytes'])
    _resources(owner['elapsed_seconds'],owner['peak_group_RSS_bytes'],owner['minimum_available_RAM_bytes'])
    _duration(worker['cost']['elapsed_seconds'],'Worker remains within original1800s')
    _duration(body['cost']['elapsed_seconds'],'Body remains within original1800s')
    need(owner['child_resource_observations']==observations and owner['worker_cost']==worker['cost'],
        'Owner retains this actual worker resource/cost evidence')
    need(all('error' not in obj and 'cleanup_error' not in obj for obj in (owner,worker,body)),
        'No hidden worker/owner failure beneath a completion label')
    _postcheck(spec,owner,worker,postcheck,expected_sources)
    return dict(schema='Root-U54-DGX-full-FIT-accepted-v1',status='ROOT_ACCEPTED_ALL741_CPU_VALIDATION',
        root_accepted=True,images=741,GT=1661,failed_loads=0,dataset_indices=list(range(741)),
        manifest_ref=deepcopy(spec['manifest_ref']),package_refs_sha256=package_refs_sha256(spec['package_refs']),
        worker_report_ref=deepcopy(postcheck['worker_report_ref']),owner_receipt_ref=deepcopy(postcheck['owner_receipt_ref']),
        original_terminal=terminal,no_training=True,CUDA_uninitialized=True)
