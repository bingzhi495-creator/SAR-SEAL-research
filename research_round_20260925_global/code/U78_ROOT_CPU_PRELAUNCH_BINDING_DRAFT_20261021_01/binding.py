"""UNBOUND Root U76-to-U71 prelaunch wiring. Import defines stdlib functions only."""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import traceback
from types import ModuleType

SCHEMA='Root-U78-CPU-prelaunch-binding-request-v1'
PINS={
    'admission_source_ref':dict(bytes=9071,sha256='9b7ae0dbd2605598d9f83d5f31f8805aad01ff2c37006e427138af0f68d2f75b'),
    'acceptance_source_ref':dict(bytes=34428,sha256='fd7ee03ded035ced8f3aea9c74afbf018b7e6ef73a25b5a463018d9c3b4f41c2'),
    'entry_source':dict(bytes=23263,sha256='01d13ff489ad6836d4e16bf969069bc966341671f53c5b1e44b5c1509d91acc5'),
    'owner_source':dict(bytes=26829,sha256='bf6e48fac5560d4f9320bb52e68bd84d1383f70154098ce62179b9f8c4a0e798'),
}
KEYS={'schema','phase','admission_source_ref','acceptance_source_ref','entry_source','owner_source',
      'proof_refs','prediction_refs','expected_endpoint','cpu_request','output_directory',
      'owner_output_directory','persistence_directory'}


def need(ok,message):
    if not ok:raise ValueError(message)


def signature(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)


def definitions(reference,raw):
    module=ModuleType('_U78_BYTE_CLOSED_METADATA')
    module.__file__=reference['path']
    exec(compile(raw,reference['path'],'exec'),module.__dict__)
    return module


def artifact_path(value):
    need(type(value) is str and value,'Explicit output/persistence path')
    path=PurePosixPath(value) if value.startswith('/') else PureWindowsPath(value)
    need(path.is_absolute() and '..' not in path.parts and str(path)!=path.anchor,
         'Absolute non-root output/persistence path')
    return path


class BytesPath:
    """Only the already-admitted immutable byte array; no filesystem access."""
    def __init__(self,path,raw):self.path,self.raw=path,raw
    def __str__(self):return self.path
    def read_bytes(self):return self.raw


def prepare_binding(request,*,read_raw):
    """Actually run fixed U76 then fixed U71 metadata checks, without any launch.

    read_raw(ref) is Root's explicit original-byte transport boundary. No nested
    historical ref, CPU data input, library, model or checkpoint is read here.
    """
    request=deepcopy(request);signature(request)
    need(type(request) is dict and set(request)==KEYS and request['schema']==SCHEMA,'Exact U78 binding request')
    for role,pin in PINS.items():
        need(type(request[role]) is dict and all(request[role].get(k)==v for k,v in pin.items()),
             'Frozen source pin: '+role)
    need(type(request['proof_refs']) is list and len(request['proof_refs'])==4
         and type(request['prediction_refs']) is list and len(request['prediction_refs'])==4,'Explicit four original proof/prediction refs')
    refs=[request[k] for k in PINS]+request['proof_refs']+request['prediction_refs']+[request['cpu_request']['worker_source']]
    allowed={}
    for value in refs:
        need(type(value) is dict and set(value)=={'path','bytes','sha256'}
             and type(value['path']) is str and type(value['bytes']) is int and value['bytes']>0
             and type(value['sha256']) is str and len(value['sha256'])==64
             and set(value['sha256'])<=set('0123456789abcdef'),'Explicit byte input reference')
        artifact_path(value['path'])
        need(value['path'] not in allowed or allowed[value['path']]==value,'No conflicting direct input refs')
        allowed[value['path']]=deepcopy(value)
    actual={}
    def closed(value):
        need(type(value) is dict and allowed.get(value.get('path'))==value,'Only explicit direct metadata/source bytes may be read')
        path=value['path']
        if path not in actual:
            raw=read_raw(deepcopy(value))
            need(type(raw) is bytes and len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'],
                 'Actual original source/artifact bytes/SHA')
            actual[path]=raw
        return BytesPath(path,actual[path])
    admission_module=definitions(request['admission_source_ref'],closed(request['admission_source_ref']).read_bytes())
    entry=definitions(request['entry_source'],closed(request['entry_source']).read_bytes())
    closed(request['owner_source'])  # Source identity only; never execute owner.
    inputs=[request['acceptance_source_ref']]+request['proof_refs']+request['prediction_refs']
    admission=admission_module.admit_population(sorted(inputs,key=lambda v:v['path']),
        {v['path']:closed(v).read_bytes() for v in inputs},proof_refs=request['proof_refs'],
        prediction_refs=request['prediction_refs'],endpoint=request['expected_endpoint'],
        cpu_request=request['cpu_request'],acceptance_source_ref=request['acceptance_source_ref'])
    need(admission['external_shard_admission_refs']==request['proof_refs']
         and admission['shard_refs']==request['cpu_request']['shard_refs'],'Same actual admission ordinal refs')
    spec=dict(schema=entry.SCHEMA,operation=entry.OPERATION,phase=request['phase'],
        entry_source=deepcopy(request['entry_source']),owner_source=deepcopy(request['owner_source']),
        request=deepcopy(request['cpu_request']),limits=deepcopy(entry.LIMITS),
        output_directory=request['output_directory'],owner_output_directory=request['owner_output_directory'],
        external_shard_admission_refs=deepcopy(admission['external_shard_admission_refs']))
    union={}
    for value in entry.base_references(spec):
        need(value['path'] not in union or union[value['path']]==value,'Consistent explicit U71 source union')
        union[value['path']]=deepcopy(value)
    spec['sources']=[union[path] for path in sorted(union)]
    outputs=[artifact_path(value) for value in (spec['output_directory'],spec['owner_output_directory'],
             spec['request']['output_directory'],request['persistence_directory'])]
    need(all(a!=b and a not in b.parents and b not in a.parents for i,a in enumerate(outputs) for b in outputs[i+1:]),
         'Separate non-nested output/persistence directories')
    for value in list(allowed.values())+spec['sources']:
        path=artifact_path(value['path'])
        need(all(out!=path and out not in path.parents for out in outputs),'No output/persistence may enclose a direct input')
    entry.validate_envelope(spec)
    entry.close_source_union(spec,closed)
    need(signature(spec['request'])==signature(request['cpu_request'])
         and hashlib.sha256(signature(spec['request']).encode()).hexdigest()==admission['cpu_request_metadata_sha256'],
         'Exactly the same CPU request after admission and U71 checks')
    record=dict(schema='Root-U78-CPU-prelaunch-binding-record-v1',status='PREPARED_U71_SPEC_NOT_LAUNCHED',
        binding_request_metadata_sha256=hashlib.sha256(signature(request).encode()).hexdigest(),
        cpu_request_metadata_sha256=admission['cpu_request_metadata_sha256'],
        direct_read_refs=[allowed[path] for path in sorted(actual)],
        source_refs={k:deepcopy(request[k]) for k in PINS},
        U76_admission_called=True,U71_envelope_checked=True,U71_exact_source_union_checked=True,
        qualified=False,AP=False,CPU_launched=False,GPU_launched=False,CAS_performed=False,
        persistence_performed=False,formal_training_budget_credit=0,actual_runtime_sources_hashed=False,
        real_remote_output_absence_verified=False,Root_original_observations_self_certified=False)
    return dict(spec=spec,admission=admission,record=record)


def write_json_exclusive(path,value):
    raw=(signature(value)+'\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def write_binding(request,*,read_raw):
    """Fresh Root metadata artifact only; no owner, accounting or launch call."""
    request=deepcopy(request)
    out=Path(request['persistence_directory'])
    need(out.is_absolute() and out.resolve()==out and not out.is_symlink()
         and out.parent.is_dir(),'Fresh canonical local persistence directory')
    if out.exists():raise FileExistsError(str(out))
    result=prepare_binding(request,read_raw=read_raw)
    out.mkdir()  # Exclusive; a concurrent creator cannot be reused.
    try:
        persisted={role:write_json_exclusive(out/name,result[role])
                   for role,name in (('spec','SPEC.json'),('admission','ADMISSION.json'))}
        result['record'].update(persistence_performed=True,spec_ref=persisted['spec'],admission_ref=persisted['admission'])
        persisted['record']=write_json_exclusive(out/'PREPARATION.json',result['record'])
        if os.name=='posix':
            descriptor=os.open(str(out),os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        result['persisted_refs']=persisted
        return result
    except BaseException as error:
        # Keep original partial files and original exception. A failed audit
        # write must not replace the original persistence failure.
        try:
            write_json_exclusive(out/'FAILED.json',dict(status='FAILED_NOT_LAUNCHED',qualified=False,AP=False,
                CPU_launched=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())))
        except BaseException:pass
        raise


def read_local(reference):
    """Same-host explicit file only. Cross-host opaque refs need Root's reader."""
    path=Path(reference['path'])
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and path.is_file(),
         'Canonical same-host original artifact; no implicit cross-host relocation')
    return path.read_bytes()


def main(argv=None):
    parser=argparse.ArgumentParser(description='Root metadata prepare only; never launch CPU evaluation')
    parser.add_argument('--request',required=True)
    parser.add_argument('--request-sha',required=True)
    args=parser.parse_args(argv)
    path=Path(args.request)
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and path.is_file(),
         'Explicit canonical original request file')
    raw=path.read_bytes()
    need(type(args.request_sha) is str and len(args.request_sha)==64
         and hashlib.sha256(raw).hexdigest()==args.request_sha,'Actual original request SHA')
    request=json.loads(raw);signature(request)
    result=write_binding(request,read_raw=read_local)
    print(signature(dict(status=result['record']['status'],spec_ref=result['persisted_refs']['spec'],
        admission_ref=result['persisted_refs']['admission'],preparation_ref=result['persisted_refs']['record'],
        CPU_launched=False,qualified=False,AP=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
