"""UNBOUND U54 Linux CPU worker; no scientific import or execution on import."""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time
import traceback
import types


SCHEMA='ROOT-U54-DGX-full-FIT-CPU-validation-spec-v1'
OPERATION='CPU_U54_DGX_FULL_FIT_PORTABLE_VALIDATION'
BODY_BYTES=7912
BODY_SHA256='30421a352bd9cfb70770e7057f0a6927583a25518c0cbeb627216f2d5d9a43f0'
SOURCE_PINS={
    'portable_reader':'b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557',
    'image_geometry':'65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
}
SUCCESS_FLAGS={'source_bytes_verified','CUDA_uninitialized_before','CUDA_uninitialized_after',
    'source_unchanged_after','module_aliases_restored','caller_CPU_RNG_preserved',
    'caller_NumPy_RNG_preserved','caller_Python_RNG_preserved','dtype_grad_threads_preserved',
    'child_resources_within_limits'}
FALSE_FIELDS=('qualified','new_training','pilot_only','main_training','AP','DEV','TEST',
              'U40_frontier_modified','innovation_established','speedup_established')


def need(ok,message):
    if not ok:raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and PurePosixPath(value['path']).is_absolute()
         and str(PurePosixPath(value['path']))==value['path'] and '..' not in PurePosixPath(value['path']).parts
         and type(value['bytes']) is int and value['bytes']>0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Strict DGX-local canonical POSIX file reference')
    return value


def canonical_sources(values):
    """Derived exact Source set, independent of JSON key order or a guessed count."""
    need(type(values) is list and values,'Explicit direct Source references')
    result={}
    for value in values:
        strict_ref(value);key=value['path']
        need(key not in result or result[key]==value,'No conflicting Source reference for the same path')
        result[key]=value
    return [result[key] for key in sorted(result)]


def source_table(spec):
    required={'schema','operation','phase','entry_source','body_source','owner_source','source_refs',
              'library_sources','manifest_ref','package_refs','prelock','output_directory','owner_output_directory','sources'}
    need(type(spec) is dict and set(spec)==required and spec['schema']==SCHEMA and spec['operation']==OPERATION
         and type(spec['phase']) is str and spec['phase'] and '/' not in spec['phase'],'Exact independent U54 CPU worker spec')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])==set(SOURCE_PINS)
         and type(spec['library_sources']) is dict and set(spec['library_sources'])=={'psutil','torch','numpy'}
         and type(spec['package_refs']) is list and len(spec['package_refs'])==741,
         'Two source modules, three selected genuine libraries and741 original-index package refs')
    for role,pin in SOURCE_PINS.items():
        need(strict_ref(spec['source_refs'][role])['sha256']==pin,'Unchanged actual reader/geometry bytes: '+role)
    need(strict_ref(spec['body_source'])['sha256']==BODY_SHA256 and spec['body_source']['bytes']==BODY_BYTES,
         'Exact new full-population CPU validation body')
    need(len({strict_ref(value)['path'] for value in spec['package_refs']})==741,'One distinct local file per original ordinal')
    values=[spec[key] for key in ('entry_source','body_source','owner_source','manifest_ref')]
    values+=list(spec['source_refs'].values())+list(spec['library_sources'].values())+spec['package_refs']
    expected=canonical_sources(values)
    need(type(spec['sources']) is list and spec['sources']==expected,'Exact derived sorted Source closure, no hidden data/model input')
    outputs=[]
    for value in (spec['output_directory'],spec['owner_output_directory']):
        need(type(value) is str and PurePosixPath(value).is_absolute() and value!='/'
             and str(PurePosixPath(value))==value and '..' not in PurePosixPath(value).parts,'Canonical explicit output directory')
        outputs.append(PurePosixPath(value))
    worker,owner=outputs
    need(worker!=owner and worker not in owner.parents and owner not in worker.parents,'Separate owner and worker fresh outputs')
    for value in expected:
        path=PurePosixPath(value['path'])
        need(all(out!=path and out not in path.parents for out in outputs),'Outputs cannot contain admitted inputs/sources')
    return {value['path']:value for value in expected}


def reference(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def load_closed(close,value,alias,originals):
    path=close(value);originals.setdefault(alias,(alias in sys.modules,sys.modules.get(alias)))
    module=types.ModuleType(alias);module.__file__=str(path);sys.modules[alias]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    need(Path(module.__file__).resolve()==path.resolve(),'Exact source-backed module object')
    return module


def main(argv=None):
    started=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True);args=parser.parse_args(argv)
    out=c=torch=np=psutil=owner=body_module=modules=manifest=audit_stream=None
    outside_cpu=outside_numpy=outside_python=outside_dtype=outside_grad=None
    aliases,cache={},{};original_error=None;created=False
    report=dict(schema='U54-full-FIT-CPU-validation-worker-report-v1',status='STARTED',operation=OPERATION,
        flags={},cleanup_errors=[],body_report={},formal_training_budget_credit=0,
        audit_records_written=0,private_audit_ref=None,validated_images=0,matched_instances=0,
        full_binary_closure_claimed=False,Linux_original_terminal_qualification_claimed=False,
        controls_scope='after owned CPU01/two-thread setup',**{name:False for name in FALSE_FIELDS})
    try:
        need(platform.system()=='Linux' and platform.machine()=='aarch64','Independent Linux/aarch64 CPU worker only')
        need(not any(name.split('.')[0] in ('torch','numpy') for name in sys.modules),'No ML before environment/Source closure')
        path=Path(args.spec);need(path.is_absolute() and not path.is_symlink(),'Absolute owner-pinned spec')
        raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==args.spec_sha,'Owner passes exact spec SHA')
        spec=json.loads(raw);admitted=source_table(spec)
        report.update(phase=spec['phase'],spec_ref=dict(path=str(path),bytes=len(raw),sha256=args.spec_sha),
            actual_library_paths={})
        for key in ('entry_source','body_source','owner_source','source_refs','library_sources','manifest_ref','package_refs','prelock'):
            report[key]=spec[key]
        out=Path(spec['output_directory'])
        need(not out.exists() and not out.is_symlink() and out.resolve()==out,'Fresh canonical worker output')
        out.mkdir(parents=True,exist_ok=False)
        created=True
        def close(value):
            strict_ref(value);file=Path(value['path'])
            need(admitted.get(value['path'])==value and file.resolve()==file and not file.is_symlink(),'Exact canonical admitted local file')
            def identity():
                st=file.stat();return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns
            before=identity()
            if cache.get(value['path'])!=(value,before):
                need(reference(file)==value and identity()==before,'Stable admitted bytes/SHA')
                cache[value['path']]=(value,before)
            return file
        for value in spec['sources']:close(value)
        need(Path(__file__).resolve()==close(spec['entry_source']),'This exact admitted worker source')
        report['flags']['source_bytes_verified']=True
        body_module=load_closed(close,spec['body_source'],'_U54_CPU_VALIDATION_BODY',aliases)
        body_module.validate_prelock(spec['prelock'])
        owner=load_closed(close,spec['owner_source'],'_U54_CPU_LINUX_OWNER',aliases)
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Actual CPU owner hides CUDA')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            need(os.environ.get(key)=='2','Actual owner two-thread environment: '+key)
        psutil=importlib.import_module('psutil')
        need(reference(Path(psutil.__file__).resolve())==spec['library_sources']['psutil'],'Actual selected psutil source')
        c=owner.make_control(spec,psutil);c.check()
        imported={'psutil':psutil}
        for name in ('torch','numpy'):
            imported[name]=importlib.import_module(name)
            need(reference(Path(imported[name].__file__).resolve())==spec['library_sources'][name],'Actual selected library source: '+name)
            c.check()
        torch,np=imported['torch'],imported['numpy']
        actual=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__),system=platform.system(),machine=platform.machine())
        need(actual==spec['prelock']['environment'] and not torch.cuda.is_initialized()
             and torch.get_default_dtype()==torch.float32,'Exact independent environment and cold CUDA/FP32 default')
        report['actual_environment']=actual
        report['actual_library_paths']={key:str(Path(module.__file__).resolve()) for key,module in imported.items()}
        report['flags']['CUDA_uninitialized_before']=True
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        need(torch.get_num_threads()==torch.get_num_interop_threads()==2,'Actual two Torch threads')
        outside_cpu=torch.get_rng_state().clone();outside_numpy=np.random.get_state();outside_python=random.getstate()
        outside_dtype,outside_grad=torch.get_default_dtype(),torch.is_grad_enabled()
        modules={name:load_closed(close,value,'_U54_CPU_'+name.upper(),aliases) for name,value in spec['source_refs'].items()}
        manifest=json.loads(c.closed(spec['manifest_ref']).read_bytes())
        need(type(manifest.get('rows')) is list and len(manifest['rows'])==741
             and [row['package_ref'] for row in manifest['rows']]==spec['package_refs'],
             'Manifest exact original-order package refs equal every declared direct input')
        audit_path=out/'IMAGE_CHECKS.jsonl'
        with audit_path.open('x',encoding='utf-8') as audit_stream:
            def record_image(value):
                need(value['dataset_index']==report['audit_records_written'],'Private audit exact original-index order')
                audit_stream.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
                audit_stream.flush();report['audit_records_written']+=1;c.check()
            result=body_module.run(torch,c,manifest,reader_module=modules['portable_reader'],
                image_geometry=modules['image_geometry'],source_refs=spec['source_refs'],prelock=spec['prelock'],
                report=report['body_report'],record_image=record_image)
        need(result is report['body_report'],'Preserve actual complete/partial body report object')
        body_module.completed_population(result)
        need(result['remaining_returned_tensor_objects']==0 and report['audit_records_written']==741,
             'All741 actual scalar audit records and no returned tensor retained')
        for value in spec['sources']:c.closed(value)
        report['flags']['source_unchanged_after']=True
        c.check();report['status']='COMPLETE'
    except BaseException as error:
        original_error=error
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
    finally:
        modules=manifest=body_module=owner=None;gc.collect()
        for alias,(existed,previous) in reversed(list(aliases.items())):
            if existed:sys.modules[alias]=previous
            else:sys.modules.pop(alias,None)
        report['flags']['module_aliases_restored']=all((alias in sys.modules)==existed
            and (not existed or sys.modules[alias] is previous) for alias,(existed,previous) in aliases.items())
        if torch is not None:
            report['flags']['CUDA_uninitialized_after']=not torch.cuda.is_initialized()
        if torch is not None and outside_cpu is not None:
            try:
                report['flags']['caller_CPU_RNG_preserved']=bool(torch.equal(outside_cpu,torch.get_rng_state()))
                current=np.random.get_state()
                report['flags']['caller_NumPy_RNG_preserved']=current[0]==outside_numpy[0] and bool(np.array_equal(current[1],outside_numpy[1])) and current[2:]==outside_numpy[2:]
                report['flags']['caller_Python_RNG_preserved']=random.getstate()==outside_python
                report['flags']['dtype_grad_threads_preserved']=torch.get_default_dtype()==outside_dtype and torch.is_grad_enabled()==outside_grad and torch.get_num_threads()==torch.get_num_interop_threads()==2
            except BaseException as error:report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        outside_cpu=outside_numpy=outside_python=None
        if c is not None:
            try:
                c.check();observed=c.summary();report['resource_observations']=observed
                report['flags']['child_resources_within_limits']=(observed['violations']==[] and observed['samples']>0
                    and 0<=observed['elapsed_seconds']<=1800 and 0<=observed['peak_RSS_bytes']<=16*1024**3
                    and observed['minimum_available_RAM_bytes']>=16*1024**3)
            except BaseException as error:
                try:report['resource_observations']=c.summary()
                except BaseException:pass
                report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        if created and (out/'IMAGE_CHECKS.jsonl').is_file():
            try:report['private_audit_ref']=reference(out/'IMAGE_CHECKS.jsonl')
            except BaseException as error:report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        body=report['body_report'];report.update(validated_images=body.get('validated_images',0),
            matched_instances=body.get('matched_instances',0),cost=dict(elapsed_seconds=time.monotonic()-started))
        if report['status']=='COMPLETE' and (set(report['flags'])!=SUCCESS_FLAGS
            or not all(value is True for value in report['flags'].values()) or report['cleanup_errors']
            or report['cost']['elapsed_seconds']>1800):
            report.update(status='FAILED',error=dict(type='RuntimeError',message='Final CPU validation/source/resource/control contract failed'))
        if created and out.is_dir():
            with (out/'REPORT.json').open('x',encoding='utf-8') as stream:
                json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        if report['status']!='COMPLETE' and original_error is None:original_error=RuntimeError('U54 CPU validation failed; partial audit/report retained')
    if original_error is not None:raise original_error
    print('U54 full741/1661 CPU input validation complete; pending actual owner/Root review. No training or AP.',flush=True)


if __name__=='__main__':main()
