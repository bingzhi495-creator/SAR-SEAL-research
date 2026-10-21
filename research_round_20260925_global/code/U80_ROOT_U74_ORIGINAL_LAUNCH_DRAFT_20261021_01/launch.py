"""UNBOUND fixed U74/U71 original SSH launch; import performs no I/O."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import sys
import traceback

TRAINING_KEY='U57_independent_DGX_original_update_training_20261018_01'
PREDICTION_KEY='U75_fixed50_shard_predictions_20261021_01'
PROFILE_KEY='U80_original_saved_spec_launches_20261021_01'
SCHEMA='Root-U80-original-saved-spec-launch-request-v1'
U75_PIN=(34428,'fd7ee03ded035ced8f3aea9c74afbf018b7e6ef73a25b5a463018d9c3b4f41c2')
PINS={
 'U74_GPU':dict(entry=(25044,'26a0e5066371829c84610d421919957fc587bcbcaf14e25f6a0401488e61e066'),
     owner=(26696,'5286468fcc4ab98a29be181251e1aee950d5d8becb59cb46c6c6d68c3ec6648e'),
     schema='ROOT-U74-DGX-fixed50-single-shard-supervision-spec-v1',operation='GPU_U74_DGX_FIXED50_SINGLE_SHARD_PREDICTION'),
 'U71_CPU':dict(entry=(23263,'01d13ff489ad6836d4e16bf969069bc966341671f53c5b1e44b5c1509d91acc5'),
     owner=(26829,'bf6e48fac5560d4f9320bb52e68bd84d1383f70154098ce62179b9f8c4a0e798'),
     schema='ROOT-U71-DGX-fixed50-population-metrics-supervision-spec-v1',operation='CPU_U71_DGX_FIXED50_POPULATION_METRICS')}


def need(value,message):
    if not value:raise ValueError(message)


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def check_authority(state,spec,mode,endpoint,final_ref):
    need(mode in PINS,'Only fixed GPU prediction or CPU population metrics')
    training=state[TRAINING_KEY]
    need(state['current_all_leaf_sessions_reaped'] is True and state['GPU_owner_or_worker_active'] is False
        and state['gpu_training_active'] is False and training['active_phase'] is None,
        'Root authority must actually be idle, with no unresolved numerical session')
    need(type(training['FORMAL_prefix']) is int and training['FORMAL_prefix']==37050,
        'Only current strictly accepted FORMAL37050')
    need(endpoint['schema']=='U58-U57-DGX-fixed-endpoint-metadata-v1' and endpoint['mode']=='FORMAL'
        and endpoint['formal_endpoint'] is True and encoded(endpoint['cursor'])==encoded(dict(
            global_updates=37050,completed_epochs=50,within_epoch=0))
        and endpoint['training_qualified']==final_ref==training['latest_accepted']['FORMAL'],
        'One whole current final training acceptance and fixed E50 endpoint')
    profile=state.get(PROFILE_KEY,dict(active_attempt=None,attempts=[]))
    need(type(profile) is dict and profile['active_attempt'] is None and type(profile['attempts']) is list,
        'No unresolved prior original-launch attempt')
    need(all(row['phase']!=spec['phase'] for row in profile['attempts']),'A phase cannot launch twice')
    if mode=='U74_GPU':
        need(spec['request']['endpoint_refs']['qualified']==final_ref,'GPU spec uses the actual current final acceptance')
    else:
        account=state[PREDICTION_KEY];rows=account['accepted_shards']
        need(type(account['accepted_count']) is int and account['accepted_count']==4 and type(rows) is list
            and len(rows)==4 and account['AP'] is False,'Four actual Root-committed prediction shards required')
        for i,row in enumerate(rows):
            need(type(row['shard_index']) is int and row['shard_index']==i
                and row['proof_ref']==spec['external_shard_admission_refs'][i]
                and row['prediction_ref']==spec['request']['shard_refs'][i],
                'CPU source spec must use the original four accepted proof/prediction refs in order')


def linux_resources():
    """Only explicitly invoked on the remote host, never during import."""
    import platform
    need(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.python_version()=='3.12.14',
        'Original Linux/aarch64/Python3.12.14 preflight')
    fields={line.split(':',1)[0]:line.split(':',1)[1].split() for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    need(fields['MemAvailable'][1:]==['kB'],'Actual Linux available-memory units')
    available=int(fields['MemAvailable'][0])*1024
    need(available>=16*1024**3 and {0,1}<=set(os.sched_getaffinity(0)),'Original free16GiB and available CPU0/1')
    return dict(available_RAM_bytes=available,CPU01_available=True)


def remote_preflight(payload):
    """Actual remote stdlib I/O, no scientific import or owner execution."""
    import types
    mode=payload['mode'];pins=PINS[mode];spec_ref=payload['spec_ref']
    def closed(reference):
        need(type(reference) is dict and set(reference)=={'path','bytes','sha256'}
            and type(reference['bytes']) is int and reference['bytes']>=0,'Explicit original file ref')
        path=Path(reference['path'])
        need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink() and path.is_file(),
            'Original canonical non-symlink remote file')
        before=path.stat();digest=hashlib.sha256();count=0
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block);count+=len(block)
        key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        need(key(before)==key(path.stat()) and count==reference['bytes'] and digest.hexdigest()==reference['sha256'],
            'Stable actual source/JSON/checkpoint bytes and SHA')
        return path
    def raw(reference):
        path=closed(reference);need(reference['bytes']<=64*1024**2,'Bounded metadata/source only')
        value=path.read_bytes()
        need(len(value)==reference['bytes'] and hashlib.sha256(value).hexdigest()==reference['sha256'],'Exact closed original raw bytes')
        return value
    def definitions(reference,pin,alias):
        need((reference['bytes'],reference['sha256'])==pin,'Exact fixed preflight definition bytes')
        module=types.ModuleType(alias);module.__file__=reference['path']
        exec(compile(raw(reference),reference['path'],'exec'),module.__dict__)
        return module
    resources=linux_resources()
    need(not any(k.split('.')[0] in ('torch','numpy','pycocotools','PIL') for k in sys.modules),'Stdlib-only preflight')
    spec=json.loads(raw(spec_ref))
    need(spec['schema']==pins['schema'] and spec['operation']==pins['operation'],'Only dedicated original owner profile')
    need((spec['owner_source']['bytes'],spec['owner_source']['sha256'])==pins['owner'],'Fixed original owner content')
    entry=definitions(spec['entry_source'],pins['entry'],'_U80_FIXED_ENTRY_METADATA')
    # Only original worker/manifest metadata is read before the exact union gate.
    admitted=entry.validate_envelope(spec)
    need(entry.close_source_union(spec,closed)==admitted,'Exact original entry source-union gate')
    for reference in spec['sources']:closed(reference)
    helper_ref=payload['acceptance_source_ref']
    need((helper_ref['bytes'],helper_ref['sha256'])==U75_PIN,'Fixed U75 extra preflight helper, outside science source union')
    if mode=='U74_GPU':
        helper=definitions(helper_ref,U75_PIN,'_U80_FIXED_U75_METADATA')
        docs={('endpoint_spec' if key=='spec' else key):json.loads(raw(value))
            for key,value in spec['request']['endpoint_refs'].items()}
        helper.endpoint_join(payload['expected_endpoint'],docs,spec['request'])
    else:
        closed(helper_ref)
        for proof_ref,prediction_ref in zip(spec['external_shard_admission_refs'],spec['request']['shard_refs']):
            proof=json.loads(raw(proof_ref))
            need(proof['root_accepted'] is True and proof['endpoint']==payload['expected_endpoint']
                and proof['prediction_ref']==prediction_ref,'Same already Root-committed four complete original shards')
    outputs=[spec['output_directory'],spec['owner_output_directory'],spec['request']['output_directory']]
    for directory in outputs:
        path=Path(directory)
        need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and not path.exists()
            and path.parent.is_dir(),'Three fresh canonical original outputs with existing parents')
    need(not any(k.split('.')[0] in ('torch','numpy','pycocotools','PIL') for k in sys.modules),'No scientific import during preflight')
    return dict(schema='Root-U80-remote-preflight-v1',status='PREFLIGHT_COMPLETE_NO_OWNER_LAUNCHED',mode=mode,
        spec_ref=spec_ref,phase=spec['phase'],expected_endpoint=payload['expected_endpoint'],source_refs=spec['sources'],
        output_directories=outputs,no_owner_launched=True,resource_observations=resources)


def file_reference(path,*,allow_empty=False):
    path=Path(path)
    need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink() and path.is_file(),
        'Actual canonical local artifact')
    before=path.stat();digest=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block);size+=len(block)
    key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    need(key(before)==key(path.stat()) and size==before.st_size and (allow_empty or size>0),'Stable actual artifact bytes')
    return dict(path=path.as_posix(),bytes=size,sha256=digest.hexdigest())


def read_verified(reference):
    need(type(reference) is dict and set(reference)=={'path','bytes','sha256'}
        and type(reference['bytes']) is int and reference['bytes']>0,'Exact nonempty local reference')
    need(file_reference(reference['path'])==reference and reference['bytes']<=64*1024**2,'Original local bytes/SHA and finite metadata')
    raw=Path(reference['path']).read_bytes()
    need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Exact local raw bytes')
    return raw


def write_new(path,value):
    raw=encoded(value)
    with Path(path).open('xb') as stream:
        need(stream.write(raw)==len(raw),'Complete exclusive artifact write');stream.flush();os.fsync(stream.fileno())
    return dict(path=Path(path).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def _flush_directory(path):
    if os.name=='posix':
        descriptor=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)


def _remote_ref(reference):
    need(type(reference) is dict and set(reference)=={'path','bytes','sha256'}
        and type(reference['path']) is str and '\\' not in reference['path']
        and PurePosixPath(reference['path']).anchor=='/' and str(PurePosixPath(reference['path']))==reference['path']
        and '..' not in PurePosixPath(reference['path']).parts and type(reference['bytes']) is int and reference['bytes']>0
        and type(reference['sha256']) is str and len(reference['sha256'])==64
        and set(reference['sha256'])<=set('0123456789abcdef'),'Explicit unchanged original remote ref')


def validate_request(request):
    need(type(request) is dict and set(request)=={'schema','mode','spec_ref','spec_copy_ref','transport_config_ref',
        'authority_ref','expected_endpoint','expected_final_acceptance_ref','acceptance_source_ref','persistence_directory'}
        and request['schema']==SCHEMA and request['mode'] in PINS,'Exact fixed-profile original-launch request')
    _remote_ref(request['spec_ref']);_remote_ref(request['expected_final_acceptance_ref']);_remote_ref(request['acceptance_source_ref'])
    need((request['acceptance_source_ref']['bytes'],request['acceptance_source_ref']['sha256'])==U75_PIN,'Frozen U75 preflight helper')
    spec_raw=read_verified(request['spec_copy_ref'])
    need(all(request['spec_copy_ref'][k]==request['spec_ref'][k] for k in ('bytes','sha256')),'Byte-exact local transport copy of original saved spec')
    spec=json.loads(spec_raw);pins=PINS[request['mode']]
    need(spec['schema']==pins['schema'] and spec['operation']==pins['operation'],'Only fixed original U74/U71 profile')
    for role in ('entry','owner'):
        reference=spec[role+'_source'];_remote_ref(reference)
        need((reference['bytes'],reference['sha256'])==pins[role],'Exact frozen '+role+' source')
    config=json.loads(read_verified(request['transport_config_ref']))
    need(type(config) is dict and set(config)=={'schema','ssh_executable','ssh_arguments','destination','remote_python','authority_path'}
        and config['schema']=='Root-U80-private-SSH-transport-v1' and config['authority_path']==request['authority_ref']['path'],
        'Root-selected private transport configuration and actual authority identity')
    need(type(config['ssh_executable']) is str and Path(config['ssh_executable']).is_absolute()
        and type(config['ssh_arguments']) is list and all(type(v) is str and '\x00' not in v and '\n' not in v for v in config['ssh_arguments'])
        and type(config['destination']) is str and config['destination'] and not config['destination'].startswith('-')
        and not any(v in config['destination'] for v in ('\x00','\n','\r'))
        and type(config['remote_python']) is str and str(PurePosixPath(config['remote_python']))==config['remote_python']
        and PurePosixPath(config['remote_python']).anchor=='/' and '..' not in PurePosixPath(config['remote_python']).parts,
        'Finite explicit private SSH arguments and original remote interpreter')
    authority_raw=read_verified(request['authority_ref']);state=json.loads(authority_raw)
    check_authority(state,spec,request['mode'],request['expected_endpoint'],request['expected_final_acceptance_ref'])
    out=Path(request['persistence_directory'])
    need(out.is_absolute() and out.resolve()==out and not out.is_symlink() and not out.exists() and out.parent.is_dir(),
        'Fresh independent local attempt persistence directory')
    for key in ('spec_copy_ref','transport_config_ref','authority_ref'):
        path=Path(request[key]['path']);need(out!=path and out not in path.parents,'Attempt cannot enclose original local inputs')
    return spec,config,authority_raw


def reserve_authority(request,spec,attempt,authority_raw):
    """Root-only cooperative reservation; never marks remote processes reaped."""
    path=Path(request['authority_ref']['path']);lock=Path(str(path)+'.u80.lock');locked=False
    try:
        write_new(lock,dict(attempt_id=attempt['attempt_id'],expected_sha256=request['authority_ref']['sha256']));locked=True
        need(path.read_bytes()==authority_raw,'Stale Root authority before original launch')
        state=json.loads(authority_raw)
        check_authority(state,spec,request['mode'],request['expected_endpoint'],request['expected_final_acceptance_ref'])
        result=deepcopy(state)
        profile=result.setdefault(PROFILE_KEY,dict(active_attempt=None,attempts=[]))
        profile['attempts'].append(deepcopy(attempt));profile['active_attempt']=deepcopy(attempt)
        result['current_all_leaf_sessions_reaped']=False
        if request['mode']=='U74_GPU':result['GPU_owner_or_worker_active']=True
        need(result[TRAINING_KEY]==state[TRAINING_KEY] and result['gpu_training_active'] is False,'No training state or budget change')
        allowed={PROFILE_KEY,'current_all_leaf_sessions_reaped','GPU_owner_or_worker_active'}
        need({k:v for k,v in result.items() if k not in allowed}=={k:v for k,v in state.items() if k not in allowed},
            'Every unrelated Root field and account preserved')
        next_path=Path(str(path)+'.u80.'+attempt['attempt_id']+'.next')
        write_new(next_path,result)
        need(path.read_bytes()==authority_raw,'Authority changed before atomic reservation')
        os.replace(str(next_path),str(path));_flush_directory(path.parent)
        need(path.read_bytes()==encoded(result),'Actual persisted reservation reread')
        return file_reference(path)
    finally:
        if locked:
            primary=sys.exc_info()[0] is not None
            try:lock.unlink()
            except OSError:
                if not primary:raise


def remote_script():
    import inspect
    prefix='from pathlib import Path\nimport os,sys,json,hashlib\nPINS='+repr(PINS)+'\nU75_PIN='+repr(U75_PIN)+'\n'
    definitions='\n'.join(inspect.getsource(f) for f in (need,linux_resources,remote_preflight))
    return prefix+definitions+'\nprint(json.dumps(remote_preflight(json.load(sys.stdin)),sort_keys=True,allow_nan=False),flush=True)\n'


class SSHTransport:
    """Actual SSH implementation; no credentials, hosts or deployment paths in source."""
    def __init__(self,config):self.config=deepcopy(config)
    def command(self,remote_command):
        import shlex
        return [self.config['ssh_executable']]+self.config['ssh_arguments']+[
            self.config['destination'],shlex.join(remote_command)]
    def preflight(self,payload,directory):
        import subprocess
        command=self.command(['exec',self.config['remote_python'],'-I','-c',remote_script()])
        stdout_path=directory/'PREFLIGHT.stdout';stderr_path=directory/'PREFLIGHT.stderr'
        with stdout_path.open('xb') as stdout,stderr_path.open('xb') as stderr:
            completed=subprocess.run(command,input=encoded(payload),stdout=stdout,stderr=stderr,timeout=1800,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),check=False)
            stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
        result=json.loads(stdout_path.read_bytes()) if type(completed.returncode) is int and completed.returncode==0 else None
        return dict(SSH_returncode=completed.returncode,result=result,command=command,
            stdout_ref=file_reference(stdout_path,allow_empty=True),stderr_ref=file_reference(stderr_path,allow_empty=True))
    def popen(self,remote_command,stdout,stderr):
        import subprocess
        return subprocess.Popen(self.command(remote_command),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)


def launch_saved_spec(request,*,transport=None):
    """Actually preflight/reserve/launch/wait once. CPU callers must be fixed U81."""
    import uuid
    request=deepcopy(request)
    spec,config,authority_raw=validate_request(request)
    out=Path(request['persistence_directory']);out.mkdir()
    result=dict(schema='Root-U80-original-launch-pending-report-v1',status='PREFLIGHT_STARTED',mode=request['mode'],
        request=request,spec_ref=request['spec_ref'],phase=spec['phase'],owner_process_exit_code=None,
        owner_OS_exit_independently_observed=False,SSH_returncode=None,local_SSH_pid=None,local_SSH_wait_returned=False,
        authority_reservation_attempted=False,authority_active_committed=False,root_accepted=False,qualified=False,
        AP=False if request['mode']=='U74_GPU' else None,TEST=False,new_training=False,formal_training_budget_credit=0,
        automatic_retry=False,stdout_ref=None,stderr_ref=None,cleanup_errors=[])
    primary=None
    try:
        write_new(out/'REQUEST.json',request)
        transport=SSHTransport(config) if transport is None else transport
        payload=dict(mode=request['mode'],spec_ref=request['spec_ref'],expected_endpoint=request['expected_endpoint'],
            acceptance_source_ref=request['acceptance_source_ref'])
        preflight=transport.preflight(payload,out);result['preflight']=preflight
        need(type(preflight['SSH_returncode']) is int and preflight['SSH_returncode']==0,'Actual independent preflight SSH zero required')
        observed=preflight['result']
        need(observed['schema']=='Root-U80-remote-preflight-v1' and observed['status']=='PREFLIGHT_COMPLETE_NO_OWNER_LAUNCHED'
            and observed['mode']==request['mode'] and observed['spec_ref']==request['spec_ref'] and observed['phase']==spec['phase']
            and observed['expected_endpoint']==request['expected_endpoint'] and observed['source_refs']==spec['sources']
            and observed['output_directories']==[spec['output_directory'],spec['owner_output_directory'],spec['request']['output_directory']]
            and observed['no_owner_launched'] is True,'Original preflight joined to exact spec/endpoint/source table/outputs')
        for key in ('stdout_ref','stderr_ref'):
            need(file_reference(preflight[key]['path'],allow_empty=True)==preflight[key],'Actual original preflight stream bytes')
        write_new(out/'PREFLIGHT.json',preflight)
        command=['exec',config['remote_python'],'-u',spec['owner_source']['path'],'--spec',request['spec_ref']['path'],
            '--spec-sha',request['spec_ref']['sha256']]
        attempt=dict(attempt_id=uuid.uuid4().hex,phase=spec['phase'],mode=request['mode'],spec_ref=request['spec_ref'],
            persistence_directory=out.as_posix(),command=command,status='RESERVED_REQUIRES_EXTERNAL_POST',
            transport_config_ref=request['transport_config_ref'])
        result.update(attempt_id=attempt['attempt_id'],command=command)
        result['attempt_ref']=write_new(out/'ATTEMPT.json',attempt)
        reservation=dict(attempt,attempt_ref=result['attempt_ref'])
        result['authority_reservation_attempted']=True
        result['active_authority_ref']=reserve_authority(request,spec,reservation,authority_raw)
        result['authority_active_committed']=True
        result['status']='ORIGINAL_SSH_OUTCOME_UNKNOWN'
        stdout_path=out/'ORIGINAL.stdout';stderr_path=out/'ORIGINAL.stderr'
        result['stdout_path']=stdout_path.as_posix();result['stderr_path']=stderr_path.as_posix()
        with stdout_path.open('xb') as stdout,stderr_path.open('xb') as stderr:
            child=transport.popen(command,stdout,stderr)
            need(type(child.pid) is int and child.pid>0,'Original actual local SSH PID')
            result['local_SSH_pid']=child.pid
            code=child.wait()
            need(type(code) is int,'Unknown/noninteger original SSH wait is not a terminal')
            result['SSH_returncode']=code;result['local_SSH_wait_returned']=True
            stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
        result['stdout_ref']=file_reference(stdout_path,allow_empty=True)
        result['stderr_ref']=file_reference(stderr_path,allow_empty=True)
        result['status']='ORIGINAL_SSH_TERMINAL_PENDING_ROOT_TOOL_AND_POST'
    except BaseException as error:
        primary=error
        result['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        if not result['authority_reservation_attempted']:result['status']='PREFLIGHT_FAILED_NO_OWNER_LAUNCH'
        elif not result['authority_active_committed']:result['status']='AUTHORITY_RESERVATION_OUTCOME_REQUIRES_ROOT_INSPECTION'
        else:result['status']='ORIGINAL_SSH_OUTCOME_UNKNOWN'
    try:
        report_ref=write_new(out/'REPORT.json',result);_flush_directory(out)
    except BaseException as error:
        if primary is None:raise
        result['cleanup_errors'].append(dict(type=type(error).__name__,message=str(error)))
    if primary is not None:raise primary
    return dict(result,report_ref=report_ref)


def launch_cpu_saved_spec(request,*,transport=None):
    """Production CPU call site is fixed U81's verified original saved binding."""
    need(type(request) is dict and request.get('mode')=='U71_CPU','Only fixed U81 CPU saved-spec call')
    return launch_saved_spec(request,transport=transport)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--request-file',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv)
    try:
        reference=file_reference(args.request_file)
        need(reference['sha256']==args.request_sha256,'Exact original CLI request SHA')
        request=json.loads(read_verified(reference))
        need(request.get('mode')=='U74_GPU','CPU production calls must use fixed U81 original-saved-binding entry')
        result=launch_saved_spec(request)
        need(type(result['SSH_returncode']) is int,'Unknown SSH outcome is not successful launch')
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
        return result['SSH_returncode']
    except BaseException as error:
        print(json.dumps(dict(schema='Root-U80-launch-error-v1',status='FAILED_OR_UNKNOWN',type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc()),sort_keys=True),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':sys.exit(main())
