"""UNBOUND saved U78 artifacts -> fixed U80 CPU launch. No import-time I/O."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType

SCHEMA='Root-U81-U71-saved-binding-launch-request-v1'
BINDING_PIN=dict(bytes=10913,sha256='0b62b3b6ebb9e2fef0dc6240f876d66dae58b02cddce9f7f145f332429f82899')
LAUNCHER_PIN=dict(bytes=26813,sha256='11b0e17a28a9a6e4b9f9493894c24cc69119fc0ec6db647f3d3db8a92ba6eb81')
KEYS={'schema','binding_source_ref','binding_request_ref','persisted_refs','binding_terminal','launcher_source_ref','launch_request'}


def need(ok,message):
    if not ok:raise ValueError(message)


def signature(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'} and type(value['path']) is str
         and type(value['bytes']) is int and value['bytes']>0 and type(value['sha256']) is str
         and len(value['sha256'])==64 and set(value['sha256'])<=set('0123456789abcdef'),'Exact original artifact ref')
    path=PurePosixPath(value['path']) if value['path'].startswith('/') else PureWindowsPath(value['path'])
    need(path.is_absolute() and '..' not in path.parts,'Explicit absolute original artifact identity')
    return value


def definitions(reference,raw):
    value=ModuleType('_U81_CLOSED_METADATA');value.__file__=reference['path']
    exec(compile(raw,reference['path'],'exec'),value.__dict__)
    return value


class BytesPath:
    def __init__(self,path,raw):self.path,self.raw=path,raw
    def __str__(self):return self.path
    def read_bytes(self):return self.raw


def actual_raw(reference):
    path=Path(reference['path'])
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and path.is_file(),
         'Canonical same-host original file; no implicit relocation')
    return path.read_bytes()


def launch_saved_binding(request,*,read_raw=None,transport=None):
    """Consume original saved bytes, then actually call the shared CPU launcher.

    Root supplies genuinely observed preparation terminal facts. JSON alone is
    not their authentication; U80 independently gates the actual Root authority.
    """
    request=deepcopy(request);signature(request)
    need(type(request) is dict and set(request)==KEYS and request['schema']==SCHEMA,'Exact U81 saved-binding request')
    terminal=request['binding_terminal']
    need(type(terminal) is dict and set(terminal)=={'root_observed_tool_exit_code','root_observed_tool_chunk_id',
         'binding_source_ref','binding_request_ref','persisted_refs'}
         and type(terminal['root_observed_tool_exit_code']) is int and terminal['root_observed_tool_exit_code']==0
         and type(terminal['root_observed_tool_chunk_id']) is str and 0<len(terminal['root_observed_tool_chunk_id'])<=200,
         'Actual external U78 integer-zero terminal/chunk, never owner self-report')
    need(all(terminal[key]==request[key] for key in ('binding_source_ref','binding_request_ref','persisted_refs')),
         'Actual U78 tool terminal bound to the exact original request/source/returned artifacts')
    if read_raw is None:read_raw=actual_raw
    cache={}
    def read(value):
        strict_ref(value);key=value['path']
        if key not in cache:
            raw=read_raw(deepcopy(value))
            need(type(raw) is bytes and len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'],
                 'Actual original saved artifact/source bytes/SHA')
            cache[key]=(deepcopy(value),raw)
        need(cache[key][0]==value,'No conflicting original refs at one path')
        return cache[key][1]
    for role,pin in (('binding_source_ref',BINDING_PIN),('launcher_source_ref',LAUNCHER_PIN)):
        strict_ref(request[role])
        need(all(request[role][key]==value for key,value in pin.items()),'Fixed original source pin: '+role)
    binding=definitions(request['binding_source_ref'],read(request['binding_source_ref']))
    persisted=request['persisted_refs']
    need(type(persisted) is dict and set(persisted)=={'spec','admission','record'},'Exact U78 returned persisted refs')
    original=json.loads(read(request['binding_request_ref']))
    need(type(original) is dict and set(original)==binding.KEYS and original['schema']==binding.SCHEMA,'Original exact U78 request bytes')
    spec_raw=read(persisted['spec'])
    spec=json.loads(spec_raw);admission=json.loads(read(persisted['admission']));record=json.loads(read(persisted['record']))
    for value in (original,spec,admission,record):signature(value)
    need(record['schema']=='Root-U78-CPU-prelaunch-binding-record-v1' and record['status']=='PREPARED_U71_SPEC_NOT_LAUNCHED'
         and record['persistence_performed'] is True and record['spec_ref']==persisted['spec']
         and record['admission_ref']==persisted['admission'],'Actual U78 saved artifact identities')
    need(all(record[key] is True for key in ('U76_admission_called','U71_envelope_checked','U71_exact_source_union_checked'))
         and all(record[key] is False for key in ('qualified','AP','CPU_launched','GPU_launched','CAS_performed',
             'actual_runtime_sources_hashed','real_remote_output_absence_verified','Root_original_observations_self_certified'))
         and type(record['formal_training_budget_credit']) is int and record['formal_training_budget_credit']==0,
         'Original preparation scope is admitted metadata only, never runtime qualification')
    need(admission['schema']=='Root-U76-fixed50-population-admission-material-v1'
         and admission['status']=='FOUR_ROOT_ACCEPTED_SHARDS_BOUND_FOR_FUTURE_CPU'
         and admission['qualified'] is admission['AP'] is False
         and type(admission['images']) is int and admission['images']==187
         and type(admission['queries']) is int and admission['queries']==18700,
         'Original full-population U76 admission artifact')
    for role,filename in (('spec','SPEC.json'),('admission','ADMISSION.json'),('record','PREPARATION.json')):
        need(binding.artifact_path(persisted[role]['path'])==binding.artifact_path(original['persistence_directory'])/filename,
             'Original returned U78 artifact path, not an untracked copy')
    need(record['binding_request_metadata_sha256']==hashlib.sha256(signature(original).encode()).hexdigest(),
         'Same original U78 request metadata digest')
    cpu_digest=hashlib.sha256(signature(spec['request']).encode()).hexdigest()
    need(cpu_digest==record['cpu_request_metadata_sha256']==admission['cpu_request_metadata_sha256']
         and signature(spec['request'])==signature(original['cpu_request']),'Same saved CPU request and digest')
    need(spec['external_shard_admission_refs']==admission['external_shard_admission_refs']==original['proof_refs']
         and spec['request']['shard_refs']==admission['shard_refs']==original['prediction_refs'],
         'Same original ordinal four accepted proof/prediction refs')
    need(all(spec[key]==original[key] for key in ('phase','output_directory','owner_output_directory')),
         'Same saved preparation phase and output identities')
    for role,pin in binding.PINS.items():
        need(all(original[role][key]==value for key,value in pin.items())
             and record['source_refs'][role]==original[role],'Original fixed preparation source identities')
    need(spec['entry_source']==original['entry_source'] and spec['owner_source']==original['owner_source'],
         'Original saved U71 entry/owner identities')
    entry=definitions(spec['entry_source'],read(spec['entry_source']))
    read(spec['owner_source'])  # Byte identity only; never execute owner here.
    entry.validate_envelope(spec)
    def closed(value):
        need(value==spec['request']['worker_source'],'Only fixed U70 metadata bytes before launch')
        return BytesPath(value['path'],read(value))
    entry.close_source_union(spec,closed)
    launch_request=request['launch_request']
    need(launch_request['schema']=='Root-U80-original-saved-spec-launch-request-v1' and launch_request['mode']=='U71_CPU'
         and launch_request['spec_copy_ref']==persisted['spec'],'CPU launch consumes exactly this saved SPEC ref')
    endpoint=launch_request['expected_endpoint']
    need(signature(endpoint)==signature(original['expected_endpoint'])==signature(admission['endpoint'])
         and endpoint['schema']=='U58-U57-DGX-fixed-endpoint-metadata-v1'
         and endpoint['mode']=='FORMAL' and endpoint['formal_endpoint'] is True
         and signature(endpoint['cursor'])==signature(dict(global_updates=37050,completed_epochs=50,within_epoch=0))
         and launch_request['expected_final_acceptance_ref']==endpoint['training_qualified'],
         'Same whole accepted fixed-E50 endpoint and original final acceptance')
    strict_ref(launch_request['acceptance_source_ref'])
    need(all(launch_request['acceptance_source_ref'][key]==value for key,value in binding.PINS['acceptance_source_ref'].items()),
         'Original fixed U75 code at its explicit actual launch-host ref')
    strict_ref(launch_request['spec_ref'])
    need(all(launch_request['spec_ref'][key]==persisted['spec'][key] for key in ('bytes','sha256')),
         'Deployed SPEC is byte-exact, with its own actual ref')
    launcher=definitions(request['launcher_source_ref'],read(request['launcher_source_ref']))
    return launcher.launch_cpu_saved_spec(deepcopy(launch_request),transport=transport)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Root: consume this exact saved U78 binding, then launch original U71 CPU owner')
    parser.add_argument('--request-file',required=True)
    parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv);path=Path(args.request_file)
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and path.is_file(),
         'Explicit canonical original U81 request file')
    raw=path.read_bytes()
    need(type(args.request_sha256) is str and len(args.request_sha256)==64
         and hashlib.sha256(raw).hexdigest()==args.request_sha256,'Actual original U81 request SHA')
    request=json.loads(raw);signature(request)
    result=launch_saved_binding(request)
    need(type(result.get('SSH_returncode')) is int,'Actual shared SSH wait integer returncode')
    print(signature(result))
    return result['SSH_returncode']


if __name__=='__main__':raise SystemExit(main())
