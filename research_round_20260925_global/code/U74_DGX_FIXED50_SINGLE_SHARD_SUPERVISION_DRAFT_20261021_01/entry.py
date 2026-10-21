"""UNBOUND future U74 single-shard entry; stdlib-only definitions on import."""
from copy import deepcopy
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import traceback
import types

SCHEMA='ROOT-U74-DGX-fixed50-single-shard-supervision-spec-v1'
OPERATION='GPU_U74_DGX_FIXED50_SINGLE_SHARD_PREDICTION'
U73_SHA256='af3860ae0df9de7efa338f725ad05f9d7f4c1c76ee8fba84bc7c6be00a8b907f'
ENCODER_BYTES=61452
ENCODER_SHA256='49c6dbc37434be372d8bb5b7e841a519491db8122f1eacef314a9e87f29a94e7'
LIMITS=dict(allowed_cpus=[0,1],threads=2,allocator_bytes=4*1024**3,
    maximum_RSS_bytes=16*1024**3,minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
REQUEST_KEYS={'worker_source','prediction_core_source','source_refs','library_sources','endpoint_refs',
    'checkpoint_refs','manifest_ref','shard_index','dataset_range','output_directory'}
RANGES=((0,47),(47,94),(94,141),(141,187))
SUCCESS_FLAGS={'source_union_verified','source_bytes_verified_before_worker','no_scientific_imports_before_worker',
    'source_bytes_rechecked_after_worker','worker_report_bytes_verified','prediction_bytes_verified',
    'module_aliases_restored','child_resources_within_limits'}
FALSE_FIELDS=('qualified','AP','TEST','new_training','ROOT_actual_terminal_self_certified','automatic_retry',
    'encoder_original_provenance_self_qualified','whole_GPU_WDDM8GiB_contract_certified','full_binary_closure_claimed')


def need(ok,message):
    if not ok:raise ValueError(message)


def signature(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def strict_ref(value,*,allow_empty=False):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'} and type(value['path']) is str
        and '\\' not in value['path'] and PurePosixPath(value['path']).anchor=='/'
        and str(PurePosixPath(value['path']))==value['path'] and '..' not in PurePosixPath(value['path']).parts
        and type(value['bytes']) is int and value['bytes']>=(0 if allow_empty else 1)
        and type(value['sha256']) is str and len(value['sha256'])==64
        and set(value['sha256'])<=set('0123456789abcdef'),'Strict canonical explicit file ref')
    return value


def _base_references(spec):
    request=spec['request']
    refs=[spec['entry_source'],spec['owner_source']]
    refs += [request[k] for k in ('worker_source','prediction_core_source','manifest_ref')]
    for key in ('source_refs','library_sources','endpoint_refs','checkpoint_refs'):
        need(type(request[key]) is dict,'Explicit reference table: '+key)
        for role,value in request[key].items():
            strict_ref(value,allow_empty=key=='library_sources' and role=='pycocotools');refs.append(value)
    return refs


def validate_envelope(spec):
    """Pure metadata guard only; exact manifest-derived union is checked later."""
    need(type(spec) is dict and set(spec)=={'schema','operation','phase','entry_source','owner_source',
        'request','limits','sources','output_directory','owner_output_directory'}
        and spec['schema']==SCHEMA and spec['operation']==OPERATION,'Exact dedicated U74 envelope')
    need(type(spec['phase']) is str and 1<=len(spec['phase'])<=160 and '/' not in spec['phase']
        and '\\' not in spec['phase'],'Finite original phase identity')
    need(type(spec['limits']) is dict and signature(spec['limits'])==signature(LIMITS),'Unchanged six resource limits')
    request=spec['request']
    need(type(request) is dict and set(request)==REQUEST_KEYS,'Exact original U73 ten-key request')
    for value in (spec['entry_source'],spec['owner_source'],request['worker_source'],
                  request['prediction_core_source'],request['manifest_ref']):strict_ref(value)
    need(request['worker_source']['sha256']==U73_SHA256,'Exact frozen U73 worker')
    need(type(request['source_refs']) is dict and 'encoder' in request['source_refs'],'Explicit original encoder ref')
    encoder=strict_ref(request['source_refs']['encoder'])
    need(encoder['bytes']==ENCODER_BYTES and encoder['sha256']==ENCODER_SHA256,
        'Only Root-traced original DEV-context encoder bytes; runtime qualification remains separate')
    need(type(request['shard_index']) is int and 0<=request['shard_index']<4
        and type(request['dataset_range']) is list and all(type(i) is int for i in request['dataset_range'])
        and tuple(request['dataset_range'])==RANGES[request['shard_index']],'One unchanged DEV shard')
    need(type(request['library_sources']) is dict and set(request['library_sources'])=={
        'torch','numpy','psutil','pycocotools','mask_extension'},'Original five explicit library roles')
    for key,roles in (('endpoint_refs',{'qualified','spec','pair_commit','order'}),('checkpoint_refs',{'PLAIN','FULL'})):
        need(type(request[key]) is dict and set(request[key])==roles,'Exact existing endpoint roles')
    outputs=[]
    for value in (spec['output_directory'],spec['owner_output_directory'],request['output_directory']):
        need(type(value) is str and '\\' not in value and PurePosixPath(value).anchor=='/'
            and str(PurePosixPath(value))==value and value!='/' and '..' not in PurePosixPath(value).parts,
            'Canonical explicit output directory')
        outputs.append(PurePosixPath(value))
    need(all(left!=right and left not in right.parents and right not in left.parents
        for i,left in enumerate(outputs) for right in outputs[i+1:]),'Three separate non-nested ENTRY/OWNER/WORKER outputs')
    need(type(spec['sources']) is list and bool(spec['sources']),'Finite proposed source allowlist')
    admitted={}
    for value in spec['sources']:
        strict_ref(value,allow_empty=True)
        need(value['path'] not in admitted,'No duplicate source-table paths')
        admitted[value['path']]=deepcopy(value)
        path=PurePosixPath(value['path'])
        need(all(out!=path and out not in path.parents for out in outputs),'Outputs cannot contain inputs')
    need(spec['sources']==[admitted[k] for k in sorted(admitted)],'Canonical source table ordering')
    for value in _base_references(spec):
        need(admitted.get(value['path'])==value,'Every explicit code/library/endpoint input must be proposed')
    return admitted


def closed_bytes(close,value):
    path=close(value)
    need(str(path)==value['path'],'Actual exact closed path')
    raw=path.read_bytes()
    need(len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'],'Actual closed byte identity')
    return raw


def load_closed(close,value,alias):
    """Definitions only; no new sys.modules aliases or unclosed dependency path."""
    module=types.ModuleType(alias);module.__file__=value['path']
    exec(compile(closed_bytes(close,value),value['path'],'exec'),module.__dict__)
    need(module.__file__==value['path'],'Actual source-backed definition module')
    return module


def close_source_union(spec,closed):
    """Read only frozen U73 source and manifest; reject extras before payload I/O."""
    admitted=validate_envelope(spec)
    worker=load_closed(closed,spec['request']['worker_source'],'_U74_U73_METADATA')
    manifest=json.loads(closed_bytes(closed,spec['request']['manifest_ref']))
    need(type(manifest) is dict and 'entries' in manifest,'Original explicit manifest entries')
    refs=worker.direct_references(spec['request'],manifest['entries'])+[spec['entry_source'],spec['owner_source']]
    expected={}
    for value in refs:
        strict_ref(value,allow_empty=value==spec['request']['library_sources']['pycocotools'])
        need(value['path'] not in expected or expected[value['path']]==value,'No conflicting exact union refs')
        expected[value['path']]=value
    need(spec['sources']==[expected[k] for k in sorted(expected)] and expected==admitted,
        'Exact U73 manifest-derived direct union plus only entry/owner')
    return admitted


def no_scientific_imports():
    return not any(name.split('.')[0] in ('torch','numpy','pycocotools','PIL') for name in sys.modules)


def _write_json(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8')
    with path.open('xb') as stream:
        need(stream.write(raw)==len(raw),'Complete entry report write');stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def read_output(path):
    need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink(),'Canonical actual output')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    need(key(before)==key(after) and len(raw)==after.st_size,'Stable complete output bytes')
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),json.loads(raw)


def _resources(observed):
    need(type(observed) is dict and observed['schema']=='U74-Linux-child-control-observations-v1'
        and type(observed['samples']) is int and observed['samples']>0 and observed['violations']==[]
        and type(observed['peak_RSS_bytes']) is int and 0<=observed['peak_RSS_bytes']<=LIMITS['maximum_RSS_bytes']
        and type(observed['minimum_available_RAM_bytes']) is int
        and observed['minimum_available_RAM_bytes']>=LIMITS['minimum_available_RAM_bytes'],
        'Actual child resource samples within unchanged limits')
    elapsed=observed['elapsed_seconds']
    need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=1800,'Bounded actual child duration')


def prediction_metadata(value,request):
    """Only completion/identity projection; no masks, COCO or new math."""
    n=request['dataset_range'][1]-request['dataset_range'][0]
    need(value['schema']=='U66-DGX-fixed50-DEV-shard-pending-v1'
        and value['status']=='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
        and type(value['shard_index']) is int and value['shard_index']==request['shard_index']
        and signature(value['dataset_range'])==signature(request['dataset_range']), 'Actual complete original U66 shard identity')
    for key,expected in (('images',n),('queries',100*n),('fixed_epoch',50),('updates_per_arm',37050),('query_batch',4),
                         ('remaining_model_objects',0),('formal_training_budget_credit',0)):
        need(type(value[key]) is int and value[key]==expected,'Original prediction completion counter: '+key)
    false=('qualified','AP','TEST','new_training','whole_population_predicted','DEV_metrics_computed',
        'GT_used_for_prediction','GT_rasterization_executed')
    true=('DEV','DEV_technical_input','model_BN_Torch_RNG_unchanged','original_full_S0_class_order_preserved',
        'source_bytes_rechecked_after')
    need(all(value[k] is False for k in false) and all(value[k] is True for k in true),
        'Prediction scope only; no AP/training/Root qualification')
    endpoint=value['endpoint']
    need(endpoint['schema']=='U58-U57-DGX-fixed-endpoint-metadata-v1' and endpoint['mode']=='FORMAL'
        and endpoint['formal_endpoint'] is True and signature(endpoint['cursor'])==signature(dict(
            global_updates=37050,completed_epochs=50,within_epoch=0))
        and endpoint['checkpoint_refs']==request['checkpoint_refs'],'Original final E50 two-arm endpoint')
    numerical_refs={k:v for k,v in request['source_refs'].items() if k!='fit_portable_reader'}
    need(len(numerical_refs)==21 and value['source_refs']==numerical_refs and set(value['arms'])=={'PLAIN','FULL'}
        and all(value['arms'][a]['status']=='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE' for a in ('PLAIN','FULL'))
        and signature(value['cleanup'])==signature(dict(allocated_after=0,reserved_after=0)),
        'Actual two-arm completion, source identity and U66 final release')
    keys=('schema','status','shard_index','dataset_range','images','queries','fixed_epoch','updates_per_arm','query_batch',
        'remaining_model_objects','formal_training_budget_credit','endpoint','source_refs','cleanup')+false+true
    result={k:deepcopy(value[k]) for k in keys}
    result['arms']={a:dict(status=value['arms'][a]['status']) for a in ('PLAIN','FULL')}
    return result


def completed_entry(report,spec,spec_ref):
    """Parent's pure JSON completion check, not an observation or qualification."""
    validate_envelope(spec);strict_ref(spec_ref)
    need(report['schema']=='U74-DGX-fixed50-single-shard-entry-report-v1'
        and report['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE' and report['operation']==OPERATION
        and report['phase']==spec['phase'] and report['spec_ref']==spec_ref,'Exact pending entry identity')
    for key in ('entry_source','owner_source','request','limits','sources','output_directory','owner_output_directory'):
        need(signature(report[key])==signature(spec[key]),'Exact entry input: '+key)
    need(type(report['flags']) is dict and set(report['flags'])==SUCCESS_FLAGS
        and all(value is True for value in report['flags'].values()) and report['cleanup_errors']==[]
        and all(report[k] is False for k in FALSE_FIELDS) and report['DEV'] is report['DEV_technical_input'] is True
        and type(report['formal_training_budget_credit']) is int and report['formal_training_budget_credit']==0,
        'Complete entry checks, no self qualification or training credit')
    _resources(report['resource_observations'])
    elapsed=report['cost']['elapsed_seconds']
    need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=1800,'Bounded entry duration')
    request=spec['request'];worker=report['worker_report']
    need(worker['schema']=='U73-single-shard-worker-report-v1' and worker['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE'
        and all(worker[k] is False for k in ('qualified','AP','TEST','new_training','ROOT_actual_terminal_self_certified','automatic_retry'))
        and worker['DEV'] is worker['DEV_technical_input'] is True
        and type(worker['formal_training_budget_credit']) is int and worker['formal_training_budget_credit']==0
        and worker['cleanup_errors']==[] and worker['source_aliases_restored'] is True
        and worker['source_and_metadata_bytes_rechecked'] is True and worker['outer_runtime_states_restored'] is True,
        'Actual complete pending U73 worker; source tests/readers are insufficient')
    need(worker['worker_source']==request['worker_source'] and worker['prediction_core_source']==request['prediction_core_source']
        and worker['shard_index']==request['shard_index'] and worker['dataset_range']==request['dataset_range']
        and worker['direct_input_refs']==[r for r in spec['sources'] if r not in (spec['entry_source'],spec['owner_source'])],
        'Exact underlying U73 source/shard/input union')
    need(worker['original_restore_source_closure_verified'] is True
        and type(worker['source_refs']) is dict and len(worker['source_refs'])==22
        and signature(worker['source_refs'])==signature(request['source_refs']),
        'Original restore source closure and complete22 outer refs required')
    strict_ref(report['worker_report_ref']);strict_ref(report['prediction_ref'])
    need(report['worker_report_ref']['path']==str(PurePosixPath(request['output_directory'])/'REPORT.json')
        and report['prediction_ref']['path']==str(PurePosixPath(request['output_directory'])/'U66_SHARD.json')
        and report['prediction_ref']==worker['prediction_ref'],'Exact actual worker/report/prediction refs')
    need(prediction_metadata(report['prediction_metadata'],request)==report['prediction_metadata'],
        'Exact projection of the separately byte-verified U66 output')
    return worker


def execute_owned(spec,spec_ref,c,worker,*,report):
    """Actual future child path after main has authenticated owner/psutil/source."""
    need(type(report) is dict and not report,'Fresh entry report')
    started=time.monotonic();primary=None;created=False;out=None
    report.update(schema='U74-DGX-fixed50-single-shard-entry-report-v1',status='STARTED',operation=OPERATION,
        flags={k:False for k in SUCCESS_FLAGS},cleanup_errors=[],worker_report={},
        DEV=True,DEV_technical_input=True,formal_training_budget_credit=0,**{k:False for k in FALSE_FIELDS})
    try:
        validate_envelope(spec);strict_ref(spec_ref)
        report.update(phase=spec['phase'],spec_ref=deepcopy(spec_ref))
        for key in ('entry_source','owner_source','request','limits','sources','output_directory','owner_output_directory'):
            report[key]=deepcopy(spec[key])
        out=Path(spec['output_directory'])
        need(out.is_absolute() and out.resolve()==out and not out.is_symlink() and not out.exists() and out.parent.is_dir(),
            'Fresh original entry output')
        out.mkdir();created=True
        close_source_union(spec,c.closed);report['flags']['source_union_verified']=True
        for value in spec['sources']:c.closed(value)
        report['flags']['source_bytes_verified_before_worker']=True
        need(no_scientific_imports(),'No scientific imports before genuine U73 admission')
        report['flags']['no_scientific_imports_before_worker']=True
        need(type(worker) is types.ModuleType and worker.__file__==spec['request']['worker_source']['path'],
            'Actual source-loaded U73 worker')
        c.check();returned=worker.run_worker(c,spec['request'],report=report['worker_report'])
        need(returned is report['worker_report'],'Actual U73 caller-owned worker report')
        worker_path=Path(spec['request']['output_directory'])/'REPORT.json'
        worker_ref,persisted=read_output(worker_path)
        need(signature(persisted)==signature(returned),'Actual persisted U73 report equals returned report')
        report['worker_report_ref']=worker_ref;report['flags']['worker_report_bytes_verified']=True
        prediction_path=Path(spec['request']['output_directory'])/'U66_SHARD.json'
        actual_ref,prediction=read_output(prediction_path)
        need(actual_ref==returned['prediction_ref'],'Actual complete prediction output bytes')
        report['prediction_ref']=actual_ref;report['prediction_metadata']=prediction_metadata(prediction,spec['request'])
        report['flags']['prediction_bytes_verified']=True
        for value in spec['sources']:c.closed(value)
        report['flags']['source_bytes_rechecked_after_worker']=True
        need(returned['source_aliases_restored'] is True,'U73 original aliases restored')
        report['flags']['module_aliases_restored']=True
        report['status']='COMPLETE_PENDING_FINAL_RESOURCE_CHECK'
    except BaseException as error:
        primary=error;report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
    finally:
        try:
            c.check();report['resource_observations']=c.summary();_resources(report['resource_observations'])
            report['flags']['child_resources_within_limits']=True
        except BaseException as error:
            report['status']='FAILED';report['cleanup_errors'].append(dict(type=type(error).__name__,message=str(error)))
            try:report['resource_observations']=c.summary()
            except BaseException as summary_error:
                report['cleanup_errors'].append(dict(type=type(summary_error).__name__,message=str(summary_error)))
            if primary is None:primary=error
        report['cost']=dict(elapsed_seconds=time.monotonic()-started)
        if report['status']=='COMPLETE_PENDING_FINAL_RESOURCE_CHECK':
            report['status']='COMPLETE_PENDING_ROOT_ACCEPTANCE'
            try:completed_entry(report,spec,spec_ref)
            except BaseException as error:
                report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                if primary is None:primary=error
        if created:
            try:_write_json(out/'REPORT.json',report)
            except BaseException as error:
                report['status']='FAILED';report['report_write_error']=dict(type=type(error).__name__,message=str(error))
                if primary is None:primary=error
    if primary is not None:raise primary
    need(report['status']=='COMPLETE_PENDING_ROOT_ACCEPTANCE','Actual complete entry required')
    return report


def verify_file(value):
    """Stdlib bootstrap for explicitly selected small source/metadata refs only."""
    strict_ref(value)
    path=Path(value['path'])
    need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink(),'Canonical bootstrap file')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    key=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    need(key(before)==key(after) and len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'],
        'Stable pinned bootstrap bytes')
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True)
    args=parser.parse_args(argv)
    spec=spec_ref=None;entered_execution=False;spec_hash_verified=False
    try:
        need(platform.system()=='Linux' and platform.machine()=='aarch64' and platform.python_version()=='3.12.14',
            'Dedicated original DGX Python3.12.14 Linux/aarch64 child')
        need(no_scientific_imports(),'Fresh child before source admission')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
            and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
            'Original preimport CUDA0/cuBLAS/two-thread owner environment')
        path=Path(args.spec)
        need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink()
            and 0<path.stat().st_size<=2*1024**2,'Canonical bounded owner-passed spec')
        spec_ref,spec=read_output(path)
        need(spec_ref['sha256']==args.spec_sha,'Actual owner-passed spec SHA')
        spec_hash_verified=True
        validate_envelope(spec)
        need(spec['entry_source']['path']==str(Path(__file__).resolve(strict=True)),'This exact executed entry source')
        selected=[spec['entry_source'],spec['owner_source'],spec['request']['worker_source'],
            spec['request']['manifest_ref'],spec['request']['library_sources']['psutil']]
        bootstrap={value['path']:value for value in selected}
        def close(value):
            need(bootstrap.get(value['path'])==value,'Bootstrap admits only original selected source/metadata')
            return verify_file(value)
        close(spec['entry_source'])
        owner=load_closed(close,spec['owner_source'],'_U74_ORIGINAL_OWNER')
        need(owner.SCHEMA==SCHEMA and owner.OPERATION==OPERATION and signature(owner.LIMITS)==signature(LIMITS),
            'Actual dedicated original owner ABI and unchanged limits')
        close_source_union(spec,close)
        expected=spec['request']['library_sources']['psutil'];close(expected)
        psutil=importlib.import_module('psutil')
        need(type(psutil) is types.ModuleType and str(psutil.__version__)=='6.1.1'
            and str(Path(psutil.__file__).resolve(strict=True))==expected['path'],'Genuine selected psutil6.1.1 source')
        close(expected)
        control=owner.make_control(spec,psutil);control.check()
        worker=load_closed(control.closed,spec['request']['worker_source'],'_U74_ORIGINAL_U73')
        entered_execution=True
        return execute_owned(spec,spec_ref,control,worker,report={})
    except BaseException as error:
        # Do not overwrite/retry an execution report. The owner also retains
        # actual stdout/stderr and real returncode if bootstrap reporting fails.
        if spec_hash_verified and spec is not None and spec_ref is not None and not entered_execution:
            try:
                validate_envelope(spec);out=Path(spec['output_directory'])
                out.mkdir()
                report=dict(schema='U74-DGX-fixed50-single-shard-entry-report-v1',status='FAILED',operation=OPERATION,
                    phase=spec['phase'],spec_ref=spec_ref,flags={k:False for k in SUCCESS_FLAGS},
                    cleanup_errors=[],worker_report={},DEV=True,DEV_technical_input=True,formal_training_budget_credit=0,
                    error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),
                    **{k:False for k in FALSE_FIELDS})
                _write_json(out/'REPORT.json',report)
            except BaseException:
                pass  # Original error remains visible and produces a nonzero exit.
        raise


if __name__=='__main__':main()
