"""UNBOUND stdlib external collection/envelope CLI; import observes no process."""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import types

ACCEPTANCE_PIN=(34428,'fd7ee03ded035ced8f3aea9c74afbf018b7e6ef73a25b5a463018d9c3b4f41c2')


def need(ok,message):
    if not ok:raise ValueError(message)


def _identity(value):
    need(type(value) is dict and set(value)=={'pid','state','parent_pid','process_group','session','start_ticks'},'Complete original process identity')
    for key in ('pid','parent_pid','process_group','session','start_ticks'):
        need(type(value[key]) is int and value[key]>=(0 if key=='parent_pid' else 1),'Integer process identity')
    need(type(value['state']) is str and len(value['state'])==1,'Linux process state')


def parse_proc_stat(text):
    closing=text.rfind(')')
    need(closing>=0 and '(' in text[:closing],'Linux /proc stat format')
    pid=int(text[:text.index('(')].strip());fields=text[closing+1:].split()
    need(len(fields)>=20,'Complete Linux /proc stat')
    result=dict(pid=pid,state=fields[0],parent_pid=int(fields[1]),process_group=int(fields[2]),session=int(fields[3]),start_ticks=int(fields[19]))
    _identity(result);return result


def _current(proc_root,pid):
    directory=proc_root/str(pid);path=directory/'stat'
    need(not directory.is_symlink() and not path.is_symlink(),'No substituted process path')
    try:text=path.read_text(encoding='utf-8')
    except FileNotFoundError:
        need(not directory.exists(),'UNKNOWN: process directory exists but stat is unavailable')
        return None
    value=parse_proc_stat(text);need(value['pid']==pid,'Process stat matches requested PID')
    return value


def observe_processes(proc_root,owner):
    """Explicit test fixture injection; production main only supplies real /proc."""
    proc_root=Path(proc_root)
    need(proc_root.is_absolute() and proc_root.resolve(strict=True)==proc_root and not proc_root.is_symlink(),
        'Explicit canonical proc namespace')
    parent,child=owner['owner_identity'],owner['child_identity'];_identity(parent);_identity(child)
    need(type(owner['owner_pid']) is int and owner['owner_pid']==parent['pid']
        and child['parent_pid']==parent['pid'] and child['process_group']==child['session']==child['pid']
        and child['start_ticks']>=parent['start_ticks'],'Original owner/child parent/session/group')
    boot_path=proc_root/'sys/kernel/random/boot_id';boot=boot_path.read_text(encoding='ascii').strip()
    need(len(boot)==36 and boot==owner['boot_id'],'UNKNOWN: different or unavailable original boot')
    current_owner=_current(proc_root,parent['pid']);current_child=_current(proc_root,child['pid'])
    need(current_owner is None and current_child is None,'Original PID live or reused; cannot certify original group absence')
    members=[]
    for directory in sorted(proc_root.iterdir(),key=lambda p:p.name):
        if not directory.name.isdigit():continue
        need(directory.is_dir() and not directory.is_symlink(),'UNKNOWN: incomplete numeric proc entry')
        value=_current(proc_root,int(directory.name))
        if value is None:continue
        if value['process_group']!=child['pid']:continue
        need(value['session']==child['pid'] and value['start_ticks']>=child['start_ticks'],'UNKNOWN: original owned-group identity')
        members.append(value)
    need(boot_path.read_text(encoding='ascii').strip()==boot,'UNKNOWN: boot changed during scan')
    need(not members,'Original owned process group still has members')
    return dict(boot_id=boot,owner_identity=deepcopy(parent),child_identity=deepcopy(child),
        owner_current_identity=None,child_current_identity=None,owner_original_birth_absent=True,child_original_birth_absent=True,
        owned_group_scan_resolved=True,remaining_owned_group=[],all_owned_processes_absent=True)


def production_proc_root():
    need(platform.system()=='Linux','Production collection requires genuine Linux /proc')
    return Path('/proc')


def _signature(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _file(path,*,expected=None,retain=False,allow_empty=False):
    path=Path(path)
    need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink(),'Actual canonical file on this host')
    if expected is not None:
        need(type(expected) is dict and set(expected)=={'path','bytes','sha256'}
            and expected['path']==path.as_posix() and type(expected['bytes']) is int
            and expected['bytes']>=(0 if allow_empty else 1) and type(expected['sha256']) is str
            and len(expected['sha256'])==64 and set(expected['sha256'])<=set('0123456789abcdef'),'Exact explicit readable ref')
    before=path.stat();key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    need(not retain or before.st_size<=64*1024**2,'Bounded original JSON/source bytes')
    digest=hashlib.sha256();size=0;parts=[]
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            size+=len(block);digest.update(block)
            if retain:parts.append(block)
    need(key(before)==key(path.stat()) and size==before.st_size and size>=(0 if allow_empty else 1),'Stable complete actual file bytes')
    reference=dict(path=path.as_posix(),bytes=size,sha256=digest.hexdigest())
    need(expected is None or reference==expected,'Actual bytes/SHA differ from original ref')
    return reference,b''.join(parts) if retain else None


def _read(reference):
    actual,raw=_file(reference['path'],expected=reference,retain=True)
    value=json.loads(raw);need(type(value) is dict,'Original JSON object')
    return value


def _load_acceptance(reference):
    need((reference['bytes'],reference['sha256'])==ACCEPTANCE_PIN,'Exact fixed U75 definitions')
    actual,raw=_file(reference['path'],expected=reference,retain=True)
    module=types.ModuleType('_U77_FIXED_U75');module.__file__=reference['path']
    exec(compile(raw,reference['path'],'exec'),module.__dict__)
    return module


def _write(path,value):
    raw=(_signature(value)+'\n').encode('utf-8')
    with path.open('xb') as stream:
        need(stream.write(raw)==len(raw),'Complete exclusive artifact write');stream.flush();os.fsync(stream.fileno())
    return dict(path=path.as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def _request(request,request_ref,operation):
    fields={'schema','operation','collector_source_ref','acceptance_source_ref','output_directory'}
    fields |= {'spec_ref','original_terminal'} if operation=='COLLECT' else {'collection_ref','collector_terminal'}
    schema='U77-external-'+('collect' if operation=='COLLECT' else 'envelope')+'-request-v1'
    need(type(request) is dict and set(request)==fields and request['schema']==schema and request['operation']==operation,'Exact finite U77 request')
    need(_signature(_read(request_ref))==_signature(request),'Actual request bytes equal caller request')
    own=request['collector_source_ref']
    need(own['path']==Path(__file__).resolve().as_posix(),'This actual collector source path')
    _file(own['path'],expected=own)
    return _load_acceptance(request['acceptance_source_ref'])


def _fresh_output(request,input_paths):
    out=Path(request['output_directory'])
    need(out.is_absolute() and out.resolve()==out and not out.is_symlink() and out.parent.is_dir(),'Fresh canonical output with existing parent')
    for value in input_paths:
        path=Path(value)
        need(out!=path and out not in path.parents and path not in out.parents,'Output cannot replace or contain an input/original output')
    out.mkdir()
    return out


def collect(request,request_ref):
    """Explicit operation only; production has no configurable procfs root."""
    acceptance=_request(request,request_ref,'COLLECT');proc_root=production_proc_root()
    spec=_read(request['spec_ref']);acceptance.pinned_codes(spec)
    need(spec['schema']=='ROOT-U74-DGX-fixed50-single-shard-supervision-spec-v1'
        and spec['operation']=='GPU_U74_DGX_FIXED50_SINGLE_SHARD_PREDICTION','Original pinned U74 spec identity')
    out=_fresh_output(request,[request_ref['path'],request['spec_ref']['path'],request['collector_source_ref']['path'],
        request['acceptance_source_ref']['path'],spec['output_directory'],spec['owner_output_directory'],spec['request']['output_directory']])
    report=dict(schema='U77-external-collection-v1',status='UNKNOWN_OR_FAILED',root_accepted=False,qualified=False,
        AP=False,DEV=False,TEST=False,new_training=False,formal_training_budget_credit=0,automatic_retry=False,
        U75_acceptance_called=False,waits_for_external_root_terminal=True,request_ref=deepcopy(request_ref),
        request=deepcopy(request),collector_source_ref=deepcopy(request['collector_source_ref']),
        acceptance_source_ref=deepcopy(request['acceptance_source_ref']),spec_ref=deepcopy(request['spec_ref']),phase=spec['phase'],
        attempt_id=request['original_terminal']['attempt_id'],original_terminal=deepcopy(request['original_terminal']),observations=None)
    primary=None
    try:
        owner_out=Path(spec['owner_output_directory']);entry_out=Path(spec['output_directory']);worker_out=Path(spec['request']['output_directory'])
        paths=dict(owner_receipt=owner_out/'OWNER_RECEIPT.json',owner_run_state=owner_out/'RUN_STATE.json',
            entry_report=entry_out/'REPORT.json',worker_report=worker_out/'REPORT.json',prediction=worker_out/'U66_SHARD.json')
        refs={};values={}
        for role,path in paths.items():
            reference,raw=_file(path,retain=role!='prediction');refs[role]=reference
            if role!='prediction':values[role]=json.loads(raw)
        owner=values['owner_receipt'];entry=values['entry_report'];body=values['worker_report']
        need(values['owner_run_state']==owner,'Actual terminal RUN_STATE and owner receipt agree')
        need(owner['schema']=='U74-Linux-owned-fixed50-single-shard-prediction-receipt-v1'
            and owner['status']=='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW'
            and owner['spec_ref']==request['spec_ref'] and owner['phase']==spec['phase'] and owner['operation']==spec['operation'],
            'Actual original successful pending owner receipt')
        need(type(owner['returncode']) is int and owner['returncode']==0 and owner['reaped'] is True
            and owner['original_birth_absent'] is True and owner['final_group_scan_resolved'] is True
            and owner['remaining_owned_group']==[] and owner['all_owned_processes_absent'] is True,'Original actual child wait/closure')
        acceptance.validate_original_terminal(request['original_terminal'],phase=spec['phase'],spec_ref=request['spec_ref'],
            owner_source=spec['owner_source'],sys_executable=owner['sys_executable'])
        need(entry['schema']=='U74-DGX-fixed50-single-shard-entry-report-v1' and entry['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE'
            and entry['spec_ref']==request['spec_ref'] and entry['phase']==spec['phase']
            and body['schema']=='U73-single-shard-worker-report-v1' and body['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE'
            and entry['worker_report']==body,'Actual entry and worker records join')
        need(owner['entry_report_ref']==refs['entry_report'] and owner['worker_report_ref']==entry['worker_report_ref']==refs['worker_report']
            and owner['prediction_ref']==entry['prediction_ref']==body['prediction_ref']==refs['prediction'],'Actual five output byte refs join')
        logs={};need(set(owner['logs'])=={'stdout.log','stderr.log'},'Both raw original owner logs')
        for name,role in (('stdout.log','owner_stdout'),('stderr.log','owner_stderr')):
            value=owner['logs'][name]
            need(value['path']==(owner_out/name).as_posix(),'Original raw child log path')
            logs[role]=_file(value['path'],expected=value,allow_empty=True)[0]
        for name,role in (('stdout_ref','original_stdout'),('stderr_ref','original_stderr')):
            value=request['original_terminal'][name];logs[role]=_file(value['path'],expected=value,allow_empty=True)[0]
        report.update(evidence_refs=refs,log_refs=logs)
        report['observations']=observe_processes(proc_root,owner)
        report['status']='COLLECTED_AWAITING_ROOT_TOOL_TERMINAL'
    except BaseException as error:
        primary=error;report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    try:reference=_write(out/'COLLECTION.json',report)
    except BaseException as write_error:
        if primary is not None:raise primary from write_error
        raise
    if primary is not None:raise primary
    return dict(status=report['status'],collection_ref=reference)


def envelope(request,request_ref):
    """A later explicit invocation, after Root has truly observed collector completion."""
    acceptance=_request(request,request_ref,'ENVELOPE')
    collection=_read(request['collection_ref'])
    need(collection['schema']=='U77-external-collection-v1' and collection['status']=='COLLECTED_AWAITING_ROOT_TOOL_TERMINAL'
        and collection['root_accepted'] is collection['qualified'] is collection['U75_acceptance_called'] is False
        and 'error' not in collection and 'root_observed_tool_exit_code' not in collection
        and 'root_observed_tool_chunk_id' not in collection,'Only pending original collection, never self-certified future terminal')
    need(collection['collector_source_ref']==request['collector_source_ref']
        and collection['acceptance_source_ref']==request['acceptance_source_ref'],'Same fixed collector and U75 source refs')
    original=_read(collection['request_ref'])
    need(original==collection['request'] and original['schema']=='U77-external-collect-request-v1'
        and original['operation']=='COLLECT' and original['collector_source_ref']==request['collector_source_ref']
        and original['acceptance_source_ref']==request['acceptance_source_ref'],'Original collect request raw identity')
    terminal=request['collector_terminal']
    need(type(terminal) is dict and set(terminal)=={'root_observed_tool_exit_code','root_observed_tool_chunk_id',
        'collection_ref','request_ref','collector_source_ref','spec_ref','phase','attempt_id'},'Exact eight externally observed collector-terminal fields')
    need(type(terminal['root_observed_tool_exit_code']) is int and terminal['root_observed_tool_exit_code']==0,
        'Root must observe actual collector tool integer zero before this later call')
    need(type(terminal['root_observed_tool_chunk_id']) is str and 0<len(terminal['root_observed_tool_chunk_id'])<=200
        and terminal['root_observed_tool_chunk_id']!=collection['original_terminal']['root_observed_tool_chunk_id'],
        'Independent actual collector tool chunk, never the original numerical terminal')
    expected=dict(collection_ref=request['collection_ref'],request_ref=collection['request_ref'],
        collector_source_ref=collection['collector_source_ref'],spec_ref=collection['spec_ref'],
        phase=collection['phase'],attempt_id=collection['attempt_id'])
    need(all(terminal[k]==value for k,value in expected.items()),'Collector terminal bound to this source/request/spec/phase/attempt/collection')
    spec=_read(collection['spec_ref']);acceptance.pinned_codes(spec)
    need(original['spec_ref']==collection['spec_ref'] and original['original_terminal']==collection['original_terminal']
        and spec['phase']==collection['phase'],'Original numerical spec and terminal identity retained')
    owner=_read(collection['evidence_refs']['owner_receipt'])
    acceptance.validate_original_terminal(collection['original_terminal'],phase=spec['phase'],spec_ref=collection['spec_ref'],
        owner_source=spec['owner_source'],sys_executable=owner['sys_executable'])
    observed=collection['observations']
    need(type(observed) is dict and set(observed)=={'boot_id','owner_identity','child_identity','owner_current_identity',
        'child_current_identity','owner_original_birth_absent','child_original_birth_absent','owned_group_scan_resolved',
        'remaining_owned_group','all_owned_processes_absent'},'Exact original independent process observations')
    post=dict(root_observed_tool_exit_code=terminal['root_observed_tool_exit_code'],
        root_observed_tool_chunk_id=terminal['root_observed_tool_chunk_id'],spec_ref=deepcopy(collection['spec_ref']),
        phase=collection['phase'],attempt_id=collection['attempt_id'],original_terminal=deepcopy(collection['original_terminal']),
        evidence_refs=deepcopy(collection['evidence_refs']),**deepcopy(observed))
    acceptance.validate_postcheck(post,collection['original_terminal'],owner)
    out=_fresh_output(request,[request_ref['path'],request['collection_ref']['path'],collection['request_ref']['path'],
        collection['spec_ref']['path'],request['collector_source_ref']['path'],request['acceptance_source_ref']['path'],
        original['output_directory'],spec['output_directory'],spec['owner_output_directory'],spec['request']['output_directory']])
    post_ref=_write(out/'POST.json',post)
    record=dict(schema='U77-external-envelope-v1',status='POST_WRITTEN_PENDING_ROOT_ACCEPTANCE',root_accepted=False,qualified=False,
        AP=False,DEV=False,TEST=False,new_training=False,formal_training_budget_credit=0,U75_acceptance_called=False,
        envelope_tool_terminal_self_observed=False,request_ref=deepcopy(request_ref),collection_ref=deepcopy(request['collection_ref']),
        collector_terminal=deepcopy(terminal),post_ref=post_ref,spec_ref=deepcopy(collection['spec_ref']),phase=collection['phase'],attempt_id=collection['attempt_id'])
    envelope_ref=_write(out/'ENVELOPE.json',record)
    return dict(status=record['status'],post_ref=post_ref,envelope_ref=envelope_ref)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-file',required=True)
    parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv)
    try:
        need(platform.system()=='Linux','Production CLI requires genuine Linux; no procfs override')
        reference,raw=_file(Path(args.request_file),retain=True)
        need(type(args.request_sha256) is str and len(args.request_sha256)==64
            and set(args.request_sha256)<=set('0123456789abcdef') and reference['sha256']==args.request_sha256,
            'Original CLI request bytes/SHA')
        request=json.loads(raw)
        need(type(request) is dict and request['operation'] in ('COLLECT','ENVELOPE'),'One explicit collect or later envelope operation')
        result=(collect if request['operation']=='COLLECT' else envelope)(request,reference)
        print(_signature(result))
        return 0
    except BaseException as error:
        print(_signature(dict(schema='U77-cli-error-v1',status='FAILED',type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc())),file=sys.stderr)
        return 1


if __name__=='__main__':sys.exit(main())
