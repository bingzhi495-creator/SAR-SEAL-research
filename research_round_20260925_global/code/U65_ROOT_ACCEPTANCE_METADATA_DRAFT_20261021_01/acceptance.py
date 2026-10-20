"""UNBOUND U65 Root acceptance checks, pure supplied JSON metadata; no I/O.

The caller must supply genuine, independently observed terminal/postcheck data.
This module cannot establish that a process exited or that file bytes were read.
It cannot turn a simulated tuple or completed byte transfer into a real U59 load.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath

SPEC_SCHEMA='ROOT-U65-DGX-full-DEV-CPU-reader-validation-spec-v1'
OPERATION='CPU_U65_DGX_DEV_FULL187_PORTABLE_READER_VALIDATION'
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
    'U40_frontier_modified','innovation_established','speedup_established',
    'model_constructed','model_forward_executed','optimizer_created','GT_matching_recomputed','GT_rasterization_executed')
BODY_FALSE=('qualified','new_training','AP','DEV','TEST','main_training','model_constructed',
    'model_forward_executed','optimizer_created','GT_matching_recomputed','GT_rasterization_executed')
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


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')


def _metadata_sha256(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


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
        'Exact supplied U65 CPU reader validation spec')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])=={'portable_reader','dev_reader','dev_rgb_reader','image_geometry'}
        and type(spec['library_sources']) is dict and set(spec['library_sources'])=={'psutil','torch','numpy'},
        'Exact four source modules and three genuine selected library roles')
    packages=spec['package_refs']
    need(type(packages) is list and len(packages)==187
        and len({strict_ref(value)['path'] for value in packages})==187,'One distinct original-ordinal package per187 images')
    values=[spec[key] for key in ('entry_source','body_source','owner_source','manifest_ref')]
    values+=list(spec['source_refs'].values())+list(spec['library_sources'].values())+packages
    expected=canonical_refs(values)
    need(len(expected)==len(values) and spec['sources']==expected,
        'Every direct role/input has distinct bytes metadata in the exact canonical Source union')
    pre=spec['prelock']
    need(type(pre) is dict and _canonical(pre)==_canonical(dict(schema='U65-DEV187-CPU-reader-validation-prelock-v1',environment=ENVIRONMENT,
        dataset_indices=list(range(187)),images=187,queries=18700,CUDA_visible_devices='',limits=LIMITS,
        no_new_training=True,no_forward=True,no_matching=True,no_AP=True,no_DEV=True,no_TEST=True,
        no_GT_rasterization=True,DEV_technical_input=True)),
        'Unchanged complete CPU validation prelock and limits')
    outputs=[]
    for key in ('output_directory','owner_output_directory'):
        value=spec[key]
        need(type(value) is str and value!='/' and PurePosixPath(value).is_absolute()
            and str(PurePosixPath(value))==value and '..' not in PurePosixPath(value).parts,'Explicit canonical output metadata')
        outputs.append(PurePosixPath(value))
    worker,owner=outputs
    need(worker!=owner and worker not in owner.parents and owner not in worker.parents,
        'Separate owner and worker output attempts')
    need(all(out!=PurePosixPath(ref['path']) and out not in PurePosixPath(ref['path']).parents
        for out in outputs for ref in expected),'Outputs cannot contain admitted Source files')
    return expected


def _postcheck(spec,owner,worker,post,expected_sources):
    keys={'schema','spec_ref','worker_report_ref','owner_receipt_ref','private_audit_ref','logs',
        'verified_source_refs','verified_output_refs','owner_identity','child_identity','source_and_logs_byte_verified',
        'manifest','private_audit_records'}
    need(type(post) is dict and set(post)==keys and post['schema']=='Root-U65-explicit-byte-postcheck-v1'
        and post['source_and_logs_byte_verified'] is True,'Real Root byte postcheck, never a worker self-claim')
    refs=[strict_ref(post[key]) for key in ('spec_ref','worker_report_ref','owner_receipt_ref','private_audit_ref')]
    need(type(post['logs']) is dict and set(post['logs'])=={'stdout.log','stderr.log'},'Both original owned logs')
    outputs=canonical_refs(refs+list(post['logs'].values()))
    need(len(outputs)==6 and post['verified_output_refs']==outputs
        and post['verified_source_refs']==expected_sources,
        'Root actually byte-verified the exact derived Sources and spec/worker/owner/audit/two logs')
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
    need(type(child.get('process_group')) is int and type(child.get('session')) is int
        and type(owner['owner_pid']) is int and type(child.get('parent_pid')) is int
        and child['process_group']==child['session']==child['pid']
        and child['parent_pid']==owner['owner_pid']==post['owner_identity']['pid']
        and child['pid']!=owner['owner_pid'] and child['start_ticks']>=post['owner_identity']['start_ticks'],
        'Original owned child session/group and original owner PID')


def _audits(spec,body,post):
    """Join only Root-supplied, genuinely byte-verified parsed metadata.

    Re-encoding is exact for the frozen Linux worker's compact ASCII JSONL.
    Native metadata hashes are retained opaque; no package/GT decode is done.
    """
    manifest=post['manifest'];audits=post['private_audit_records']
    need(type(manifest) is dict and type(manifest.get('entries')) is list
        and len(manifest['entries'])==187 and type(audits) is list and len(audits)==187,
        'Actual full manifest and187 actual private scalar audit records, never counts alone')
    need(body['manifest_sha256']==_metadata_sha256(manifest),
        'Body consumed this actual manifest metadata')
    keys={'dataset_index','image_id','queries','status','source_package','original_record_sha256',
        'native_predictions_metadata_sha256','query_ids_metadata_sha256','CUDA_initialized',
        'returned_tensor_objects_released'}
    records=[];image_ids=set()
    for index,(entry,audit,package) in enumerate(zip(manifest['entries'],audits,spec['package_refs'])):
        need(type(entry) is dict and set(entry)=={'dataset_index','image_id','query_ids','package_ref'}
            and type(entry['dataset_index']) is int and entry['dataset_index']==index
            and type(entry['image_id']) is int and entry['image_id'] not in image_ids
            and entry['package_ref']==package,'Exact distinct original187 manifest ordinal/image/package joins')
        image_ids.add(entry['image_id']);qids=entry['query_ids']
        need(type(qids) is list and len(qids)==100 and all(type(qid) is int for qid in qids)
            and set(qids)==set(range(100)),'Every original full100 export query permutation retained')
        record=dict(dataset_index=index,image_id=entry['image_id'],qids=qids);records.append(record)
        need(type(audit) is dict and set(audit)==keys
            and type(audit['dataset_index']) is int and audit['dataset_index']==index
            and type(audit['image_id']) is int and audit['image_id']==entry['image_id']
            and type(audit['queries']) is int and audit['queries']==100 and audit['status']=='VALIDATED'
            and audit['source_package']==package
            and audit['original_record_sha256']==_metadata_sha256(record)
            and audit['query_ids_metadata_sha256']==_metadata_sha256(qids)
            and audit['CUDA_initialized'] is False and audit['returned_tensor_objects_released'] is True,
            'Each real scalar row joins its original ordinal, query order and actual release')
        native=audit['native_predictions_metadata_sha256']
        need(type(native) is str and len(native)==64 and set(native)<=set('0123456789abcdef'),
            'Original native metadata digest retained without reconstructing or decoding predictions')
    need(body['record_metadata_sha256']==_metadata_sha256(records),
        'Actual reader records match the complete manifest order')
    raw=b''.join(_canonical(row)+b'\n' for row in audits)
    need(len(raw)==post['private_audit_ref']['bytes']
        and hashlib.sha256(raw).hexdigest()==post['private_audit_ref']['sha256'],
        'Parsed scalar rows must reproduce the actual Root-verified worker JSONL artifact')


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
    need(owner['schema']=='U65-Linux-owned-full-DEV-CPU-reader-validation-receipt-v1'
        and owner['status']=='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW'
        and owner['operation']==spec['operation'] and owner['phase']==spec['phase']
        and type(owner['returncode']) is int and owner['returncode']==0
        and owner['reaped'] is owner['original_birth_absent'] is owner['final_group_scan_resolved'] is True
        and owner['all_owned_processes_absent'] is owner['CPU_only'] is owner['CPU_validation_complete'] is True
        and owner['child_alive'] is owner['termination_requested'] is False
        and owner['remaining_owned_group']==owner['violations']==owner['signals']==[]
        and owner['CUDA_visible_devices']=='' and owner['GPU_allocator_contract_applicable'] is False
        and owner['child_launched'] is owner['controller_returning'] is True
        and owner['automatic_retry'] is False,
        'Actual original CPU owner completed, waited/reaped child and resolved the entire absent owned group')
    need(worker['schema']=='U65-DEV187-CPU-reader-validation-worker-report-v1' and worker['status']=='COMPLETE'
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
    need(all(key not in obj for obj in (owner,worker,body) for key in
        ('GT','G','targets','matched_instances','FIT_GT_tensor_read_attempted','FIT_GT_tensors_read')),
        'Legacy target-reading/matched-instance scopes cannot be accepted as U65 technical input')
    need(body['schema']=='U65-DEV187-CPU-reader-validation-body-report-v1' and body['status']=='COMPLETE_PENDING_ROOT'
        and set(body['flags'])==BODY_FLAGS and all(value is True for value in body['flags'].values()),
        'All five actual metadata, field-proof, release and cold-CUDA checks')
    for key in ('attempted_images','validated_images','released_images','recorded_images'):_count(body[key],187,'Complete187 '+key)
    for obj in (worker,owner):
        _count(obj['validated_images'],187,'Actual187 complete images')
        _count(obj['queries'],18700,'Actual18700 complete original query candidates')
        _count(obj['audit_records_written'],187,'Actual187 private scalar audit rows')
    _count(body['queries'],18700,'Actual18700 queries from real U59 loads')
    _count(body['reader_failed_loads'],0,'No failed image/field proof skipped')
    _count(body['remaining_returned_tensor_objects'],0,'All returned current-image tensors released')
    need(type(body['reader_visits']) is list and all(type(index) is int for index in body['reader_visits'])
        and body['reader_visits']==list(range(187)),'Every original index once, in original order')
    for obj in (worker,owner):
        need(all(obj[key] is False for key in FALSE_FIELDS) and type(obj['formal_training_budget_credit']) is int
            and obj['formal_training_budget_credit']==0 and obj['DEV_technical_input'] is True,
            'Input validation never grants model/evaluation/training credit')
    need(all(body[key] is False for key in BODY_FALSE) and type(body['formal_training_budget_credit']) is int
        and body['formal_training_budget_credit']==0 and body['DEV_technical_input'] is True,
        'CPU input proof is not forward, matching, optimizer work or self-qualification')
    need(type(body['original_metadata_population']) is dict
        and _canonical(body['original_metadata_population'])==_canonical(dict(images=187,queries=18700,tensor_population_revalidated=False))
        and body['original_annotation_records_parsed'] is False
        and body['raw_provenance_inputs_revalidated'] is False and body['private_scalar_audit_only'] is True,
        'Technical DEV reads do not replay historical raw provenance or annotation records')
    need(worker['full_binary_closure_claimed'] is False
        and worker['Linux_original_terminal_qualification_claimed'] is False,
        'Worker cannot self-certify complete binary closure or its owner actual OS exit')
    observations=worker['resource_observations']
    need(observations['schema']=='U65-Linux-child-control-observations-v1' and type(observations['samples']) is int
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
    _audits(spec,body,postcheck)
    return dict(schema='Root-U65-DGX-DEV187-CPU-reader-input-accepted-v1',status='ROOT_ACCEPTED_ALL187_CPU_READER_VALIDATION',
        root_accepted=True,images=187,queries=18700,failed_loads=0,dataset_indices=list(range(187)),
        manifest_ref=deepcopy(spec['manifest_ref']),package_refs_sha256=package_refs_sha256(spec['package_refs']),
        worker_report_ref=deepcopy(postcheck['worker_report_ref']),owner_receipt_ref=deepcopy(postcheck['owner_receipt_ref']),
        private_audit_ref=deepcopy(postcheck['private_audit_ref']),source_count=len(expected_sources),
        original_terminal=terminal,no_training=True,CUDA_uninitialized=True,DEV_technical_input=True,
        AP=False,DEV=False,TEST=False,model_constructed=False,model_forward_executed=False,
        optimizer_created=False,GT_matching_recomputed=False,GT_rasterization_executed=False,
        original_annotation_records_parsed=False,raw_provenance_inputs_revalidated=False,
        formal_training_budget_credit=0)
