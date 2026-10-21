"""Unbound real U75 acceptance caller and Root-only, single-writer persistence.

No SSH, launch, numerical computation, credentials or host configuration. Actual
tool terminal observations must be supplied later by the observing Root agent.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

ACCEPT_SCHEMA='Root-U79-accept-shard-request-v1'
PERSIST_SCHEMA='Root-U79-shard-persistence-v1'
ACCOUNT_KEY='U75_fixed50_shard_predictions_20261021_01'
TRAINING_KEY='U57_independent_DGX_original_update_training_20261018_01'
U75_PIN=dict(bytes=34428,sha256='fd7ee03ded035ced8f3aea9c74afbf018b7e6ef73a25b5a463018d9c3b4f41c2')

def need(value,message):
    if not value:raise ValueError(message)

def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')

def sha(raw):return hashlib.sha256(raw).hexdigest()

def reference(path,raw):return dict(path=Path(path).as_posix(),bytes=len(raw),sha256=sha(raw))

def read_verified(ref,read_raw):
    need(type(ref) is dict and set(ref)=={'path','bytes','sha256'}
         and type(ref['path']) is str and '\x00' not in ref['path']
         and type(ref['bytes']) is int and ref['bytes']>0
         and type(ref['sha256']) is str and len(ref['sha256'])==64
         and set(ref['sha256'])<=set('0123456789abcdef'),'Exact original nonempty reference')
    raw=read_raw(deepcopy(ref))
    need(type(raw) is bytes and len(raw)==ref['bytes'] and sha(raw)==ref['sha256'],'Actual original bytes/SHA')
    return raw

def load_source(ref,raw):
    need(all(ref.get(k)==v for k,v in U75_PIN.items()),'Exact frozen U75 source')
    need(len(raw)==ref['bytes'] and sha(raw)==ref['sha256'],'Closed U75 source bytes')
    module=ModuleType('_U79_FIXED_U75');module.__file__=ref['path']
    exec(compile(raw,ref['path'],'exec'),module.__dict__)
    return module

def fixed_acceptance(ref,read_raw):
    need(type(ref) is dict and all(ref.get(k)==v for k,v in U75_PIN.items()),'Exact frozen U75 source')
    return load_source(ref,read_verified(ref,read_raw))

def write_new(path,value):
    raw=encoded(value)
    with Path(path).open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return reference(path,raw)

def flush_directory(path):
    if os.name=='posix':
        descriptor=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)

def fresh_directory(path):
    out=Path(path)
    need(out.is_absolute() and out.resolve()==out and not out.is_symlink()
         and out.parent.is_dir(),'Canonical fresh persistence directory')
    if out.exists():raise FileExistsError(str(out))
    return out

def accept_and_persist(request,*,read_raw,closed):
    """Actually invoke fixed U75 on an actual U77 POST; persist without CAS.

    Run on the original artifact host. closed(ref) must open the original exact
    path, never an implicit mirror. U75 independently streams/rechecks all bytes.
    This function cannot know its outer tool exit/chunk and never invents them.
    """
    request=deepcopy(request)
    need(type(request) is dict and set(request)=={'schema','acceptance_source_ref','spec_ref',
         'postcheck_ref','original_terminal','persistence_directory'} and request['schema']==ACCEPT_SCHEMA,
         'Exact U79 accept request')
    module=fixed_acceptance(request['acceptance_source_ref'],read_raw)
    out=fresh_directory(request['persistence_directory'])
    proof=module.accept_shard(request['spec_ref'],request['original_terminal'],request['postcheck_ref'],closed=closed)
    for ref in proof['closed_refs']+[request['acceptance_source_ref']]:
        path=Path(ref['path'])
        need(path!=out and out not in path.parents,'Persistence cannot enclose any accepted input')
    out.mkdir()
    # On any persistence failure keep partial files and the original exception.
    proof_ref=write_new(out/'PROOF.json',proof)
    result=dict(schema=PERSIST_SCHEMA,status='PERSISTED_AWAITING_ROOT_ACCEPT_TOOL_TERMINAL',
        proof_ref=proof_ref,prediction_ref=deepcopy(proof['prediction_ref']),phase=proof['phase'],
        attempt_id=proof['attempt_id'],AP=False,CAS_performed=False,formal_training_budget_credit=0)
    report_ref=write_new(out/'PERSISTENCE.json',result);flush_directory(out)
    return dict(result,accept_report_ref=report_ref)

def validate_commit_inputs(source_ref,report_ref,proof_ref,prediction_ref,*,read_raw):
    module=fixed_acceptance(source_ref,read_raw)
    report=json.loads(read_verified(report_ref,read_raw))
    proof=json.loads(read_verified(proof_ref,read_raw))
    prediction=json.loads(read_verified(prediction_ref,read_raw))
    module.validate_accepted_proof(proof,prediction)
    need(type(report) is dict and set(report)=={'schema','status','proof_ref','prediction_ref',
        'phase','attempt_id','AP','CAS_performed','formal_training_budget_credit'}
        and report['schema']==PERSIST_SCHEMA and report['status']=='PERSISTED_AWAITING_ROOT_ACCEPT_TOOL_TERMINAL'
        and report['proof_ref']==proof_ref and report['prediction_ref']==prediction_ref==proof['prediction_ref']
        and report['phase']==proof['phase'] and report['attempt_id']==proof['attempt_id']
        and report['AP'] is report['CAS_performed'] is False
        and type(report['formal_training_budget_credit']) is int and report['formal_training_budget_credit']==0,
        'Actual acceptance persistence report joined to exact proof and full prediction')
    post=module.decoded_records(proof)['postcheck']
    return proof,report,post

def commit_shard(authority_path,expected_sha256,terminal,report_ref,proof_ref,prediction_ref,*,
                 read_raw,acceptance_source_ref=None):
    """Root-only commit after *actual* acceptance tool terminal, never a launcher.

    Cooperative exclusive lock + raw-byte compare before replacement. Root is
    the sole authority writer; this is not a database CAS against other writers
    that ignore this lock. Preserve every unrelated field and all training cost.
    """
    need(type(terminal) is dict and set(terminal)=={'root_observed_tool_exit_code',
        'root_observed_tool_chunk_id','accept_report_ref','proof_ref'},'Exact actual acceptance tool facts')
    need(type(terminal['root_observed_tool_exit_code']) is int and terminal['root_observed_tool_exit_code']==0
        and type(terminal['root_observed_tool_chunk_id']) is str
        and 0<len(terminal['root_observed_tool_chunk_id'])<=200,'Actual acceptance tool integer zero/chunk')
    need(terminal['accept_report_ref']==report_ref and terminal['proof_ref']==proof_ref
        and bool(report_ref) and bool(proof_ref),'Actual acceptance tool bound to exact persisted artifacts')
    proof,report,post=validate_commit_inputs(acceptance_source_ref,report_ref,proof_ref,prediction_ref,read_raw=read_raw)
    need(terminal['root_observed_tool_chunk_id'] not in (
        proof['original_terminal']['root_observed_tool_chunk_id'],post['root_observed_tool_chunk_id']),
        'Original, collector and acceptance terminals are separate observed events')
    authority=Path(authority_path)
    need(authority.is_absolute() and authority.resolve()==authority and not authority.is_symlink(),
        'Explicit canonical Root authority path')
    lock=Path(str(authority)+'.u79.lock');locked=False
    try:
        with lock.open('xb') as stream:
            locked=True;stream.write(encoded(dict(pid=os.getpid(),expected_sha256=expected_sha256)))
            stream.flush();os.fsync(stream.fileno())
        before=authority.read_bytes()
        need(sha(before)==expected_sha256,'Stale authority: no update')
        state=json.loads(before);training=state[TRAINING_KEY]
        need(state['current_all_leaf_sessions_reaped'] is True
            and state['GPU_owner_or_worker_active'] is state['gpu_training_active'] is False
            and training['active_phase'] is None and type(training['FORMAL_prefix']) is int
            and training['FORMAL_prefix']==37050,'Only idle genuine fixed E50 authority can accept predictions')
        need(proof['endpoint']['training_qualified']==training['latest_accepted']['FORMAL'],
             'Prediction proof bound to the current actual final training acceptance')
        result=deepcopy(state)
        profile=result.setdefault(ACCOUNT_KEY,dict(accepted_shards=[],accepted_count=0,AP=False,
            formal_training_budget_credit=0,automatic_CPU_launch=False))
        need(type(profile['accepted_shards']) is list and type(profile['accepted_count']) is int
            and profile['accepted_count']==len(profile['accepted_shards'])
            and profile['AP'] is False and profile['automatic_CPU_launch'] is False
            and type(profile['formal_training_budget_credit']) is int and profile['formal_training_budget_credit']==0,
            'Existing scoped prediction account remains intact')
        for row in profile['accepted_shards']:
            need(row['shard_index']!=proof['shard_index'] and row['attempt_id']!=proof['attempt_id']
                 and row['proof_ref']['path']!=proof_ref['path'],'No duplicate accepted shard/attempt/proof')
        profile['accepted_shards'].append(dict(shard_index=proof['shard_index'],phase=proof['phase'],
            attempt_id=proof['attempt_id'],proof_ref=deepcopy(proof_ref),prediction_ref=deepcopy(prediction_ref),
            accept_report_ref=deepcopy(report_ref),actual_accept_terminal=deepcopy(terminal)))
        profile['accepted_shards'].sort(key=lambda row:row['shard_index'])
        profile['accepted_count']=len(profile['accepted_shards'])
        need({k:v for k,v in result.items() if k!=ACCOUNT_KEY}==
             {k:v for k,v in state.items() if k!=ACCOUNT_KEY},'All other authority fields preserved')
        next_path=Path(str(authority)+'.u79.'+expected_sha256[:16]+'.next')
        write_new(next_path,result)
        need(authority.read_bytes()==before,'Authority changed before replace: no update')
        os.replace(str(next_path),str(authority));flush_directory(authority.parent)
        need(authority.read_bytes()==encoded(result),'Re-read persisted authority bytes')
        return dict(status='SCOPED_SHARD_PROOF_COMMITTED',accepted_count=profile['accepted_count'],
            authority_sha256=sha(encoded(result)),AP=False,new_training=False,formal_training_budget_credit=0)
    finally:
        if locked:
            primary=sys.exc_info()[0] is not None
            try:lock.unlink()
            except OSError:
                if not primary:raise

def actual_raw(ref):
    path=Path(ref['path'])
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and path.is_file(),
        'Original canonical non-symlink actual file')
    return path.read_bytes()

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--request-file',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv)
    raw=actual_raw(dict(path=args.request_file));need(sha(raw)==args.request_sha256,'Exact CLI request SHA')
    request=json.loads(raw)
    if request.get('schema')==ACCEPT_SCHEMA:
        result=accept_and_persist(request,read_raw=actual_raw,closed=lambda ref:Path(ref['path']))
    elif request.get('schema')=='Root-U79-commit-shard-request-v1':
        need(set(request)=={'schema','authority_path','expected_authority_sha256','terminal','accept_report_ref',
            'proof_ref','prediction_ref','acceptance_source_ref','transport_copies'},'Exact Root commit request')
        copies=request['transport_copies'];need(type(copies) is list,'Explicit raw artifact transport copies')
        by_path={}
        for row in copies:
            need(type(row) is dict and set(row)=={'original_ref','copy_ref'}
                and row['original_ref']['path'] not in by_path,'Unique explicit original/copy binding')
            need(all(row['original_ref'][k]==row['copy_ref'][k] for k in ('bytes','sha256')),
                'Transport must preserve exact original bytes/SHA')
            by_path[row['original_ref']['path']]=row
        def transported(ref):
            if ref['path'] in by_path:
                row=by_path[ref['path']];need(row['original_ref']==ref,'Whole original transported ref')
                return read_verified(row['copy_ref'],actual_raw)
            return actual_raw(ref)
        result=commit_shard(request['authority_path'],request['expected_authority_sha256'],request['terminal'],
            request['accept_report_ref'],request['proof_ref'],request['prediction_ref'],read_raw=transported,
            acceptance_source_ref=request['acceptance_source_ref'])
    else:raise ValueError('Only explicit acceptance or later Root commit operation')
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
