"""UNBOUND Root shard receipt validation; stdlib only, never a launcher."""
from copy import deepcopy
import base64
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import types

SCHEMA='Root-U75-DGX-fixed50-shard-accepted-v1'
RANGES=((0,47),(47,94),(94,141),(141,187))
LIMITS=dict(allowed_cpus=[0,1],threads=2,allocator_bytes=4*1024**3,maximum_RSS_bytes=16*1024**3,
    minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
CODE_PINS={
    'entry':(25044,'26a0e5066371829c84610d421919957fc587bcbcaf14e25f6a0401488e61e066'),
    'owner':(26696,'5286468fcc4ab98a29be181251e1aee950d5d8becb59cb46c6c6d68c3ec6648e'),
    'worker':(21286,'af3860ae0df9de7efa338f725ad05f9d7f4c1c76ee8fba84bc7c6be00a8b907f'),
    'core':(18285,'71e10c6539f34105c9726c73e9a0953973a76334da07059cf7a5fc2b619606be')}
NUMERICAL_ROLES=('training_state','training_entry','u36_helpers','u39_training_ops','shared_initialization',
    'full_legacy','full_dsconv','full_head','full_feature','plain_legacy','plain_feature','instance_adapter',
    'restore_state','portable_reader','image_geometry','prediction_core','encoder','mask_api','population_contract','dev_reader','dev_rgb_reader')
RESTORE_ROLES=NUMERICAL_ROLES[:12]
PROOF_KEYS={'schema','status','root_accepted','qualified','AP','DEV','DEV_technical_input','TEST','new_training',
    'formal_training_budget_credit','spec_ref','phase','attempt_id','prediction_ref','endpoint','source_refs','outer_source_refs',
    'library_sources','original_terminal','evidence_refs','raw_JSON','closed_refs','shard_index','dataset_range','images','queries','fixed_epoch','updates_per_arm'}
EVIDENCE_KEYS={'spec','postcheck','manifest','owner_receipt','owner_run_state','entry_report','worker_report','prediction',
    'qualified','endpoint_spec','pair_commit','order'}


def need(ok,message):
    if not ok:raise ValueError(message)


def signature(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def strict_ref(value,*,allow_empty=False):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'},'Exact explicit file ref')
    path=value['path']
    need(type(path) is str and '\\' not in path and '\x00' not in path,'Canonical explicit path')
    pure=PureWindowsPath(path) if PureWindowsPath(path).drive else PurePosixPath(path)
    need(pure.is_absolute() and pure.as_posix()==path and '..' not in pure.parts,'Canonical absolute ref')
    need(type(value['bytes']) is int and value['bytes']>=(0 if allow_empty else 1)
        and type(value['sha256']) is str and len(value['sha256'])==64
        and set(value['sha256'])<=set('0123456789abcdef'),'Exact nonboolean bytes/SHA identity')
    return value


def integer(value,expected,message):
    need(type(value) is int and value==expected,message)


def validate_original_terminal(terminal,*,phase,spec_ref,owner_source,sys_executable):
    """Caller supplies actual original tool/exec/wait observations, never owner JSON."""
    need(type(terminal) is dict and set(terminal)=={'root_observed_tool_exit_code','root_observed_tool_chunk_id',
        'SSH_returncode','owner_process_exit_code','original_exec_wait_observed','command','stdout_ref','stderr_ref',
        'attempt_id','phase','spec_ref'},'Exact independent original terminal observations')
    for key in ('root_observed_tool_exit_code','SSH_returncode','owner_process_exit_code'):
        integer(terminal[key],0,'Actual original integer zero: '+key)
    need(type(terminal['root_observed_tool_chunk_id']) is str and 0<len(terminal['root_observed_tool_chunk_id'])<=200,
        'Original Root tool terminal chunk, not planned exit')
    need(type(terminal['attempt_id']) is str and 0<len(terminal['attempt_id'])<=160
        and '/' not in terminal['attempt_id'] and '\\' not in terminal['attempt_id'],'One explicit original attempt')
    strict_ref(spec_ref);strict_ref(owner_source)
    need(terminal['phase']==phase and terminal['spec_ref']==spec_ref
        and terminal['original_exec_wait_observed'] is True,'Actual original exec/wait tied to this phase/spec')
    need(type(sys_executable) is str and PurePosixPath(sys_executable).is_absolute()
        and terminal['command']==['exec',sys_executable,'-u',owner_source['path'],'--spec',spec_ref['path'],
            '--spec-sha',spec_ref['sha256']], 'Exact original SSH exec pinned owner/spec/SHA command')
    strict_ref(terminal['stdout_ref'],allow_empty=True);strict_ref(terminal['stderr_ref'],allow_empty=True)
    need(terminal['stdout_ref']['path']!=terminal['stderr_ref']['path'],'Separate original SSH stdout/stderr')
    return True


def identity(value):
    need(type(value) is dict and set(value)=={'pid','state','parent_pid','process_group','session','start_ticks'},
        'Original complete Linux process identity')
    for key in ('pid','parent_pid','process_group','session','start_ticks'):
        need(type(value[key]) is int and value[key]>=(0 if key=='parent_pid' else 1),'Integer process identity: '+key)
    need(type(value['state']) is str and len(value['state'])==1,'Original Linux process state')


def validate_postcheck(post,terminal,owner):
    """Independent external observation, with no process inspection performed here."""
    need(type(post) is dict and set(post)=={'root_observed_tool_exit_code','root_observed_tool_chunk_id',
        'spec_ref','phase','attempt_id','original_terminal','boot_id','owner_identity','child_identity',
        'owner_current_identity','child_current_identity','owner_original_birth_absent','child_original_birth_absent',
        'owned_group_scan_resolved','remaining_owned_group','all_owned_processes_absent','evidence_refs'},
        'Exact independent post-terminal observation')
    integer(post['root_observed_tool_exit_code'],0,'Actual independent post tool integer zero')
    need(type(post['root_observed_tool_chunk_id']) is str and 0<len(post['root_observed_tool_chunk_id'])<=200
        and post['root_observed_tool_chunk_id']!=terminal['root_observed_tool_chunk_id'],'Independent post tool chunk')
    need(all(post[k]==terminal[k] for k in ('spec_ref','phase','attempt_id'))
        and signature(post['original_terminal'])==signature(terminal),'Post must bind the exact original attempt/command/stdout/stderr')
    need(type(post['boot_id']) is str and len(post['boot_id'])==36 and post['boot_id']==owner['boot_id'],
        'Same observed original boot; reboot/unknown refuses')
    identity(owner['owner_identity']);identity(owner['child_identity'])
    integer(owner['owner_pid'],owner['owner_identity']['pid'],'Actual original owner PID')
    need(post['owner_identity']==owner['owner_identity'] and post['child_identity']==owner['child_identity'],
        'Same original owner and child births')
    child=owner['child_identity'];parent=owner['owner_identity']
    need(child['parent_pid']==parent['pid'] and child['process_group']==child['session']==child['pid']
        and child['start_ticks']>=parent['start_ticks'],'Original child parent/session/group ownership')
    for role in ('owner','child'):
        original=post[role+'_identity'];current=post[role+'_current_identity']
        if current is not None:
            identity(current)
            need(current['pid']==original['pid'] and current['start_ticks']!=original['start_ticks'],
                'Original birth must actually be absent, not just a returning report')
        need(post[role+'_original_birth_absent'] is True,'Independent original birth absence: '+role)
    need(post['owned_group_scan_resolved'] is True and post['remaining_owned_group']==[]
        and post['all_owned_processes_absent'] is True,'Resolved independent original owned-group absence')
    need(type(post['evidence_refs']) is dict and set(post['evidence_refs'])=={
        'owner_receipt','owner_run_state','entry_report','worker_report','prediction'},'Five explicit original output refs')
    for value in post['evidence_refs'].values():strict_ref(value)
    return True


def close_reference(value,closed,*,retain_bytes=False,allow_empty=False):
    """Hash actual callback-provided file bytes; never deserialize tensor payloads."""
    strict_ref(value,allow_empty=allow_empty);need(callable(closed),'Explicit genuine closed(ref) capability')
    path=closed(deepcopy(value))
    actual=path.as_posix() if isinstance(path,Path) else str(path)
    need(actual==value['path'],'Callback must return the same actual ref path')
    if isinstance(path,Path):
        need(path.is_absolute() and not path.is_symlink() and path.resolve(strict=True)==path,
            'Actual canonical file, no host/path translation or symlink')
    digest=hashlib.sha256();size=0;parts=[]
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            need(type(block) is bytes,'Actual binary file bytes')
            size+=len(block);digest.update(block)
            if retain_bytes:parts.append(block)
    need(size==value['bytes'] and digest.hexdigest()==value['sha256'],'Actual closed bytes/SHA must match')
    return b''.join(parts) if retain_bytes else deepcopy(value)


def definitions(value,raw):
    """Only the two fixed, ordinary stdlib definition modules; never their mains."""
    need((value['bytes'],value['sha256']) in (CODE_PINS['entry'],CODE_PINS['worker']),
        'Only pinned U74/U73 stdlib metadata definitions may execute')
    need(len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'],'Actual fixed definition bytes')
    module=types.ModuleType('_U75_METADATA_'+value['sha256'][:12]);module.__file__=value['path']
    exec(compile(raw,value['path'],'exec'),module.__dict__)
    return module


def canonical_refs(values):
    table={}
    for value in values:
        strict_ref(value,allow_empty=True)
        need(value['path'] not in table or table[value['path']]==value,'No conflicting paths in finite closure')
        table[value['path']]=deepcopy(value)
    return [table[path] for path in sorted(table)]


def pinned_codes(spec):
    request=spec['request'];values={'entry':spec['entry_source'],'owner':spec['owner_source'],
        'worker':request['worker_source'],'core':request['prediction_core_source']}
    for role,value in values.items():
        strict_ref(value)
        need((value['bytes'],value['sha256'])==CODE_PINS[role],'Frozen exact source: '+role)
    return values


def finite_sources(spec,entries):
    request=spec['request'];index=request['shard_index'];span=request['dataset_range']
    need(type(index) is int and 0<=index<4 and type(span) is list and all(type(i) is int for i in span)
        and tuple(span)==RANGES[index],'One original fixed shard')
    need(type(entries) is list and len(entries)==187,'Original full187 manifest metadata')
    ids=[]
    for ordinal,row in enumerate(entries):
        integer(row['dataset_index'],ordinal,'Original integer dataset ordinal')
        need(type(row['image_id']) is int and row['image_id'] not in ids,'Unique integer image ID')
        ids.append(row['image_id'])
        need(type(row['query_ids']) is list and len(row['query_ids'])==100
            and all(type(q) is int for q in row['query_ids']) and set(row['query_ids'])==set(range(100)),
            'Original full100 query permutation')
        strict_ref(row['package_ref'])
    need(set(request['source_refs'])==set(NUMERICAL_ROLES)|{'fit_portable_reader'},'Outer exact22 source roles')
    need(set(request['library_sources'])=={'torch','numpy','psutil','pycocotools','mask_extension'}
        and set(request['endpoint_refs'])=={'qualified','spec','pair_commit','order'}
        and set(request['checkpoint_refs'])=={'PLAIN','FULL'},'Original finite source/library/endpoint roles')
    refs=[spec['entry_source'],spec['owner_source']]+[request[k] for k in ('worker_source','prediction_core_source','manifest_ref')]
    for key in ('source_refs','library_sources','endpoint_refs','checkpoint_refs'):refs+=list(request[key].values())
    refs += [row['package_ref'] for row in entries[span[0]:span[1]]]
    for value in refs:strict_ref(value,allow_empty=value==request['library_sources']['pycocotools'])
    expected=canonical_refs(refs)
    need(signature(spec['sources'])==signature(expected),'Exact finite source union, no recursive history or extra payload')
    return expected


def resource_observations(value):
    need(value['schema']=='U74-Linux-child-control-observations-v1' and type(value['samples']) is int
        and value['samples']>0 and value['violations']==[],'Actual child resource samples')
    need(type(value['peak_RSS_bytes']) is int and 0<=value['peak_RSS_bytes']<=LIMITS['maximum_RSS_bytes']
        and type(value['minimum_available_RAM_bytes']) is int
        and value['minimum_available_RAM_bytes']>=LIMITS['minimum_available_RAM_bytes'],'RSS/free RAM limits')
    bounded_time(value['elapsed_seconds'])


def bounded_time(value):
    need(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1800,'Actual bounded elapsed time')


def owner_complete(owner,spec,spec_ref,entry,refs):
    need(owner['schema']=='U74-Linux-owned-fixed50-single-shard-prediction-receipt-v1'
        and owner['status']=='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW'
        and owner['phase']==spec['phase'] and owner['spec_ref']==spec_ref and owner['operation']==spec['operation'],
        'Actual original pending owner identity')
    integer(owner['returncode'],0,'Actual direct child wait integer zero')
    for key in ('child_launched','reaped','original_birth_absent','final_group_scan_resolved',
                'all_owned_processes_absent','single_shard_prediction_complete','source_union_verified_before_payload_hash','source_bytes_verified_before_launch'):
        need(owner[key] is True,'Owner closure check: '+key)
    for key in ('child_alive','termination_requested','qualified','ROOT_actual_terminal_self_certified','automatic_retry',
                'AP','TEST','new_training','full_binary_closure_claimed','whole_GPU_WDDM8GiB_contract_certified'):
        need(owner[key] is False,'No failure/self qualification: '+key)
    need(owner['signals']==owner['violations']==owner['remaining_owned_group']==[] and owner['owner_termination_signal'] is None
        and not any(k in owner for k in ('error','cleanup_error','group_error')),'No unresolved failure or surviving owned group')
    need(owner['DEV'] is owner['DEV_technical_input'] is True,'DEV prediction only')
    integer(owner['formal_training_budget_credit'],0,'No training credit')
    need(owner['CUDA_visible_devices']=='0' and owner['GPU_allocator_contract_applicable'] is True
        and owner['GPU_allocator_limit_bytes']==LIMITS['allocator_bytes'] and type(owner['GPU_allocator_limit_bytes']) is int
        and owner['whole_GPU_observed_bytes'] is None
        and owner['whole_GPU_observation_status']=='NOT_AVAILABLE_GB10_UNIFIED_MEMORY','Original allocator, whole-GPU N/A is not zero or8GiB certification')
    need(owner['python_version']=='3.12.14' and owner['psutil_version']=='6.1.1' and owner['child_CPU_affinity']==[0,1],
        'Original runtime/CPU01')
    need(type(owner['peak_group_RSS_bytes']) is int and 0<=owner['peak_group_RSS_bytes']<=LIMITS['maximum_RSS_bytes']
        and type(owner['minimum_available_RAM_bytes']) is int and owner['minimum_available_RAM_bytes']>=LIMITS['minimum_available_RAM_bytes'],
        'Original process-group RSS/free RAM')
    bounded_time(owner['elapsed_seconds'])
    need(type(owner['observations']) is list and bool(owner['observations']),'Actual sampled group observations')
    for sample in owner['observations']:
        bounded_time(sample['elapsed_seconds'])
        need(type(sample['group_RSS_bytes']) is int and 0<=sample['group_RSS_bytes']<=owner['peak_group_RSS_bytes']
            and type(sample['available_RAM_bytes']) is int and sample['available_RAM_bytes']>=owner['minimum_available_RAM_bytes'],
            'Actual group sample within observed envelope')
        for member in sample['member_identities']:
            identity({k:member[k] for k in ('pid','state','parent_pid','process_group','session','start_ticks')})
            need(member['process_group']==member['session']==owner['child_identity']['pid']
                and member['start_ticks']>=owner['child_identity']['start_ticks'] and member['CPU_affinity']==[0,1],
                'Actual sampled original owned group and CPU01')
    death=owner['parent_death_signal']
    need(death['signal']=='SIGKILL' and death['prctl']=='PR_SET_PDEATHSIG' and death['preexec_succeeded'] is True
        and type(death['expected_parent_pid']) is int and death['expected_parent_pid']==owner['owner_pid'],'Actual original parent-death setup')
    for key,role in (('entry_report_ref','entry_report'),('worker_report_ref','worker_report'),('prediction_ref','prediction')):
        need(owner[key]==refs[role],'Owner actual output ref: '+role)
    need(owner['entry_report_status']==owner['worker_report_status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE'
        and owner['child_resource_observations']==entry['resource_observations'],'Owner/entry completion/resource join')
    resource_observations(entry['resource_observations'])


def prediction_complete(prediction,request,entries):
    span=request['dataset_range'];n=span[1]-span[0]
    need(prediction['schema']=='U66-DGX-fixed50-DEV-shard-pending-v1'
        and prediction['status']=='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE','Complete original U66 body')
    for key,value in (('shard_index',request['shard_index']),('images',n),('queries',100*n),('fixed_epoch',50),
                      ('updates_per_arm',37050),('query_batch',4),('remaining_model_objects',0),('formal_training_budget_credit',0)):
        integer(prediction[key],value,'Original complete U66 counter: '+key)
    need(signature(prediction['dataset_range'])==signature(span),'Original range')
    for key in ('qualified','AP','TEST','new_training','whole_population_predicted','DEV_metrics_computed','GT_used_for_prediction','GT_rasterization_executed'):
        need(prediction[key] is False,'Prediction-only scope: '+key)
    for key in ('DEV','DEV_technical_input','model_BN_Torch_RNG_unchanged','original_full_S0_class_order_preserved','source_bytes_rechecked_after'):
        need(prediction[key] is True,'Original complete prediction check: '+key)
    need(prediction['source_refs']=={k:request['source_refs'][k] for k in NUMERICAL_ROLES}
        and set(prediction['arms'])=={'PLAIN','FULL'} and prediction['cleanup']=={'allocated_after':0,'reserved_after':0},
        'Original21 refs/two arms/final release')
    for key in ('allocated_after','reserved_after'):integer(prediction['cleanup'][key],0,'Original final integer release')
    expected=[(row['image_id'],q) for row in entries[span[0]:span[1]] for q in row['query_ids']]
    original=prediction['identity_predictions']
    need(type(original) is list and len(original)==100*n,'Complete original native identity list')
    for arm in ('PLAIN','FULL'):
        record=prediction['arms'][arm];values=record['predictions']
        need(record['status']=='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE' and type(values) is list and len(values)==100*n
            and record['identity_predictions']==original and prediction['predictions'][arm]==values
            and prediction['cost'][arm]==record['cost'],'Whole original body, no omitted arm or double-counted aliases')
        for key,count in (('encode_calls',n),('readout_calls',25*n),('queries',100*n)):
            integer(record['cost'][key],count,'Original recorded call count: '+key)
        need(len(record['proofs'])==n,'One original calibration proof per image')
        for ordinal,proof in zip(range(*span),record['proofs']):
            integer(proof['dataset_index'],ordinal,'Original proof ordinal');integer(proof['image_id'],entries[ordinal]['image_id'],'Original proof image')
            integer(proof['queries'],100,'All100 original candidates')
            need(all(proof[k] is True for k in ('zero_residual_native_L_exact','native_RLE_and_prediction_dictionaries_exact','original_full_S0_class_order_preserved')),
                'Original zero calibration/native dictionary proofs')
        for value,native,pair in zip(values,original,expected):
            need(type(value['image_id']) is int and type(value['query_id']) is int
                and (value['image_id'],value['query_id'])==pair==(native['image_id'],native['query_id']),
                'Original image/query export order')
            need(signature({k:v for k,v in value.items() if k!='segmentation'})==signature({k:v for k,v in native.items() if k!='segmentation'})
                and 'segmentation' in value and 'segmentation' in native,'Original S0/class/non-mask dictionaries retained')
        need(set(record['cleanup'])=={'allocated_after','reserved_after'},'Original per-arm release fields')
        for key in ('allocated_after','reserved_after'):integer(record['cleanup'][key],0,'Original per-arm integer release')
        resources=record['resources']
        need(type(resources['CUDA_peak_allocated_bytes']) is int and 0<=resources['CUDA_peak_allocated_bytes']<=LIMITS['allocator_bytes']
            and type(resources['sampled_RSS_peak_bytes']) is int and 0<=resources['sampled_RSS_peak_bytes']<=LIMITS['maximum_RSS_bytes']
            and resources['RSS_sampling_errors']==[],'Original recorded per-arm resource bounds')
    need(prediction['Omega_support']==prediction['arms']['PLAIN']['Omega_support']
        and len(prediction['Omega_support'])==100*n and prediction['arms']['FULL']['Omega_support']==[],
        'Only original PLAIN top Omega support')
    for ordinal,(support,pair) in enumerate(zip(prediction['Omega_support'],expected)):
        integer(support['dataset_index'],span[0]+ordinal//100,'Original support dataset ordinal')
        need(type(support['image_id']) is int and type(support['query_id']) is int
            and (support['image_id'],support['query_id'])==pair,'Original support image/query order')


def endpoint_join(endpoint,documents,request):
    cursor=dict(global_updates=37050,completed_epochs=50,within_epoch=0)
    need(endpoint['schema']=='U58-U57-DGX-fixed-endpoint-metadata-v1' and endpoint['mode']=='FORMAL'
        and endpoint['formal_endpoint'] is True and signature(endpoint['cursor'])==signature(cursor)
        and endpoint['endpoint_refs']==request['endpoint_refs'] and endpoint['checkpoint_refs']==request['checkpoint_refs']
        and endpoint['pair_commit_ref']==request['endpoint_refs']['pair_commit'] and endpoint['order_ref']==request['endpoint_refs']['order']
        and endpoint['training_qualified']==request['endpoint_refs']['qualified'],'Whole original FORMAL E50 endpoint/ref join')
    q,spec,pair=(documents[k] for k in ('qualified','endpoint_spec','pair_commit'))
    need(q['schema']=='Root-U57-DGX-original-update-training-accepted-v1' and q['root_accepted'] is True and q['mode']=='FORMAL'
        and spec['prelock']['mode']=='FORMAL' and q['evidence_refs']['spec_ref']==request['endpoint_refs']['spec'],
        'Original explicitly closed training acceptance/spec')
    integer(spec['segment']['stop'],37050,'Actual final training stop')
    need(q['accepted_span']==spec['segment']==pair['span'] and q['result']['checkpoint_refs']==pair['checkpoint_refs']==request['checkpoint_refs']
        and signature(q['result']['paired_checkpoint_prefix'])==signature(pair['cursor'])==signature(cursor)
        and q['result']['pair_commit_ref']==request['endpoint_refs']['pair_commit']
        and q['result']['order_ref']==pair['order_ref']==request['endpoint_refs']['order'],'One actual endpoint, CP pair and final segment')
    original=dict(training_state=spec['body_source'],training_entry=spec['entry_source'],**spec['source_refs'])
    need(set(original)==set(RESTORE_ROLES)|{'image_geometry','portable_reader'} and endpoint['restore_source_refs']==original,
        'Original fourteen restoration source refs')
    for role in RESTORE_ROLES:need(request['source_refs'][role]==original[role],'Original twelve entire restore refs: '+role)
    need(request['source_refs']['image_geometry']==original['image_geometry']
        and request['source_refs']['fit_portable_reader']==original['portable_reader'],'Original geometry and FIT entire refs')


def decoded_records(proof):
    refs=proof['evidence_refs'];raw=proof['raw_JSON']
    need(type(refs) is dict and set(refs)==EVIDENCE_KEYS,'Exact finite twelve metadata/output ref roles')
    need(set(raw)==set(refs)-{'prediction'},'Exact original JSON byte records')
    result={}
    for role,text in raw.items():
        need(type(text) is str,'Original JSON bytes encoded losslessly')
        data=base64.b64decode(text.encode('ascii'),validate=True);value=refs[role];strict_ref(value)
        need(len(data)==value['bytes'] and hashlib.sha256(data).hexdigest()==value['sha256'],'Raw original JSON/ref integrity: '+role)
        result[role]=json.loads(data)
    return result


def validate_accepted_proof(proof,prediction):
    """Pure proof/body consistency. Genuine Root origin is still an external prerequisite."""
    need(type(proof) is dict and set(proof)==PROOF_KEYS,'Exact scoped proof ABI, no unknown self certification')
    need(proof['schema']==SCHEMA and proof['status']=='ROOT_ACCEPTED_SINGLE_SHARD_PREDICTION' and proof['root_accepted'] is True,
        'Scoped Root-produced single-shard acceptance')
    need(all(proof[k] is False for k in ('qualified','AP','TEST','new_training'))
        and proof['DEV'] is proof['DEV_technical_input'] is True,'No academic/CPU/AP/training qualification')
    integer(proof['formal_training_budget_credit'],0,'No training credit')
    data=decoded_records(proof);spec=data['spec'];request=spec['request'];entry=data['entry_report'];owner=data['owner_receipt'];body=data['worker_report']
    pinned_codes(spec);sources=finite_sources(spec,data['manifest']['entries'])
    need(signature(spec['limits'])==signature(LIMITS),'Original six limits, no new resource contract')
    need(data['owner_run_state']==owner and entry['worker_report']==body,'Actual three persisted report joins')
    need(proof['spec_ref']==proof['evidence_refs']['spec'] and proof['phase']==spec['phase']
        and proof['attempt_id']==proof['original_terminal']['attempt_id'],'Proof original attempt identity')
    validate_original_terminal(proof['original_terminal'],phase=spec['phase'],spec_ref=proof['spec_ref'],
        owner_source=spec['owner_source'],sys_executable=owner['sys_executable'])
    validate_postcheck(data['postcheck'],proof['original_terminal'],owner)
    need(all(data['postcheck']['evidence_refs'][k]==proof['evidence_refs'][k] for k in data['postcheck']['evidence_refs']),
        'Independent post refs equal actual accepted records')
    refs=proof['evidence_refs'];out=PurePosixPath(spec['owner_output_directory']);work=PurePosixPath(request['output_directory'])
    paths={'owner_receipt':str(out/'OWNER_RECEIPT.json'),'owner_run_state':str(out/'RUN_STATE.json'),
        'entry_report':str(PurePosixPath(spec['output_directory'])/'REPORT.json'),'worker_report':str(work/'REPORT.json'),'prediction':str(work/'U66_SHARD.json')}
    need(all(refs[k]['path']==path for k,path in paths.items()) and refs['manifest']==request['manifest_ref'],'Actual original output locations')
    need(entry['status']==body['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE' and entry['spec_ref']==proof['spec_ref']
        and entry['worker_report_ref']==refs['worker_report'] and entry['prediction_ref']==body['prediction_ref']==refs['prediction'],
        'Actual completed entry/worker and full prediction ref')
    for key in ('entry_source','owner_source','request','limits','sources','output_directory','owner_output_directory'):
        need(signature(entry[key])==signature(spec[key]),'Original entry/spec identity: '+key)
    need(type(entry['flags']) is dict and bool(entry['flags']) and all(v is True for v in entry['flags'].values())
        and entry['cleanup_errors']==body['cleanup_errors']==[] and body['original_restore_source_closure_verified'] is True
        and body['source_refs']==request['source_refs'] and body['source_aliases_restored'] is True
        and body['outer_runtime_states_restored'] is True and body['source_and_metadata_bytes_rechecked'] is True,
        'Original full22/14 closure, aliases/state and byte rechecks')
    need(body['direct_input_refs']==[r for r in sources if r not in (spec['entry_source'],spec['owner_source'])],'Honest original complete22 finite input report')
    owner_complete(owner,spec,proof['spec_ref'],entry,refs)
    prediction_complete(prediction,request,data['manifest']['entries'])
    endpoint_join(prediction['endpoint'],data,request)
    need(proof['endpoint']==prediction['endpoint'] and proof['source_refs']==prediction['source_refs']
        and proof['outer_source_refs']==request['source_refs'] and proof['library_sources']==request['library_sources']
        and proof['prediction_ref']==refs['prediction'],'Whole original endpoint/source/library/body ref projection')
    for key in ('shard_index','dataset_range','images','queries','fixed_epoch','updates_per_arm'):
        need(signature(proof[key])==signature(prediction[key]),'Original prediction projection: '+key)
    for key,value in entry['prediction_metadata'].items():
        expected={a:{'status':prediction['arms'][a]['status']} for a in ('PLAIN','FULL')} if key=='arms' else prediction[key]
        need(signature(value)==signature(expected),'Full original U66 body matches U74 metadata: '+key)
    logs=owner['logs'];need(set(logs)=={'stdout.log','stderr.log'},'Both original child raw logs')
    for name,value in logs.items():
        strict_ref(value,allow_empty=True);need(value['path']==str(out/name),'Actual original child log path')
    required=canonical_refs(sources+list(refs.values())+list(logs.values())+[
        proof['original_terminal']['stdout_ref'],proof['original_terminal']['stderr_ref']])
    need(proof['closed_refs']==required,'Exact actually byte-closed source/log/output set')
    return True


def accept_shard(spec_ref,original_terminal,postcheck,*,closed):
    """Future external Root use only, after both genuine original and post tool terminals."""
    verified=[];raw_json={}
    def read(value):
        data=close_reference(value,closed,retain_bytes=True);verified.append(deepcopy(value))
        return data
    spec_raw=read(spec_ref);spec=json.loads(spec_raw);codes=pinned_codes(spec)
    entry=definitions(codes['entry'],read(codes['entry']));entry.validate_envelope(spec)
    post_raw=read(postcheck);post=json.loads(post_raw)
    refs=deepcopy(post['evidence_refs'])
    owner_out=PurePosixPath(spec['owner_output_directory']);work=PurePosixPath(spec['request']['output_directory'])
    expected_paths={'owner_receipt':str(owner_out/'OWNER_RECEIPT.json'),'owner_run_state':str(owner_out/'RUN_STATE.json'),
        'entry_report':str(PurePosixPath(spec['output_directory'])/'REPORT.json'),
        'worker_report':str(work/'REPORT.json'),'prediction':str(work/'U66_SHARD.json')}
    need(set(refs)==set(expected_paths) and all(refs[k]['path']==v for k,v in expected_paths.items()),
        'Only the exact original five output paths may be read')
    owner_raw=read(refs['owner_receipt']);owner=json.loads(owner_raw)
    validate_original_terminal(original_terminal,phase=spec['phase'],spec_ref=spec_ref,
        owner_source=spec['owner_source'],sys_executable=owner['sys_executable'])
    validate_postcheck(post,original_terminal,owner)
    raw_json.update(spec=spec_raw,postcheck=post_raw,owner_receipt=owner_raw)
    worker=definitions(codes['worker'],read(codes['worker']))
    manifest_raw=read(spec['request']['manifest_ref']);manifest=json.loads(manifest_raw)
    exact=worker.direct_references(spec['request'],manifest['entries'])+[spec['entry_source'],spec['owner_source']]
    need(canonical_refs(exact)==spec['sources'],'Original pinned U73 exact22 direct union')
    for value in spec['sources']:
        close_reference(value,closed,allow_empty=value==spec['request']['library_sources']['pycocotools']);verified.append(deepcopy(value))
    refs.update(spec=deepcopy(spec_ref),postcheck=deepcopy(postcheck),manifest=deepcopy(spec['request']['manifest_ref']))
    refs.update({('endpoint_spec' if role=='spec' else role):deepcopy(value) for role,value in spec['request']['endpoint_refs'].items()})
    raw_json.update(manifest=manifest_raw)
    values={}
    for role,value in refs.items():
        if role not in raw_json:raw_json[role]=read(value)
        values[role]=json.loads(raw_json[role])
    prediction=values.pop('prediction');prediction_raw=raw_json.pop('prediction')
    need(entry.completed_entry(values['entry_report'],spec,spec_ref)==values['worker_report'],'Original U74 completion against actual worker bytes')
    need(entry.prediction_metadata(prediction,spec['request'])==values['entry_report']['prediction_metadata'],
        'Actual complete U66 bytes, not only a projection or source reader success')
    worker.validate_endpoint_source_closure(prediction['endpoint'],spec['request']['source_refs'])
    owner=values['owner_receipt']
    for value in list(owner['logs'].values())+[original_terminal['stdout_ref'],original_terminal['stderr_ref']]:
        close_reference(value,closed,allow_empty=True);verified.append(deepcopy(value))
    proof=dict(schema=SCHEMA,status='ROOT_ACCEPTED_SINGLE_SHARD_PREDICTION',root_accepted=True,qualified=False,AP=False,
        DEV=True,DEV_technical_input=True,TEST=False,new_training=False,formal_training_budget_credit=0,
        spec_ref=deepcopy(spec_ref),phase=spec['phase'],attempt_id=original_terminal['attempt_id'],prediction_ref=deepcopy(refs['prediction']),
        endpoint=deepcopy(prediction['endpoint']),source_refs=deepcopy(prediction['source_refs']),outer_source_refs=deepcopy(spec['request']['source_refs']),
        library_sources=deepcopy(spec['request']['library_sources']),original_terminal=deepcopy(original_terminal),
        evidence_refs=refs,raw_JSON={role:base64.b64encode(data).decode('ascii') for role,data in raw_json.items()},closed_refs=canonical_refs(verified))
    for key in ('shard_index','dataset_range','images','queries','fixed_epoch','updates_per_arm'):proof[key]=deepcopy(prediction[key])
    validate_accepted_proof(proof,prediction)
    return proof
