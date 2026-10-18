"""UNBOUND owned CPU teacher-restoration worker. Not executed while authoring."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import weakref


SCHEMA='ROOT-U47-external-teacher-CPU-restore-spec-v1'
OPERATION='CPU_U47_EXTERNAL_TEACHER_RESTORE'
ROLES=('entry_source','restore_source')
VERSIONS=dict(torch='2.0.1+cu118',numpy='1.23.5')
LIMITS=dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
    minimum_available_RAM_bytes=16*1024**3,deadline_seconds=120,CUDA_visible_devices='')
RESTORE_PIN=(12320,'019e2741d739b38ec33f015a1511cf9370d7828b040cfd1b4b878fd9dcd1afaa')


def need(ok,message):
    if not ok:raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Strict source reference')
    return value


def reference(path,raw=None):
    path=Path(path)
    if raw is not None:return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def load(path,alias,aliases):
    aliases[alias]=(alias in sys.modules,sys.modules.get(alias))
    spec=importlib.util.spec_from_file_location(alias,str(path));module=importlib.util.module_from_spec(spec)
    sys.modules[alias]=module;spec.loader.exec_module(module)
    need(Path(module.__file__).resolve()==path.resolve(),'Actually loaded helper source')
    return module


def main(argv=None):
    started=time.monotonic();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True)
    args=parser.parse_args(argv);out=None;output_created=False;torch=None;sampler=None;stopped=threading.Event()
    process=None;sample=None;aliases={};model_ref=None;model=result=modules=registration=context=helper=None
    original_error=None;receipt={};source_map={};closed_cache={}
    observed=dict(peak_RSS_bytes=0,minimum_available_RAM_bytes=None,samples=0,violations=[],sampling_errors=[])
    flags=dict(CPU_only=False,source_bytes_verified=False,CUDA_uninitialized_before=False,
        CUDA_uninitialized_after=False,module_aliases_restored=False,teacher_model_released=False)
    report=dict(schema='U47-external-teacher-CPU-restore-report-v1',status='STARTED',operation=OPERATION,
        qualified=False,new_training=False,AP=False,teacher_load_receipt=receipt,teacher_state_proof=None,
        teacher_state_proof_canonical_sha256=None,
        flags=flags,actual_versions={},library_paths={},resource_observations=observed,full_binary_closure_claimed=False,
        dataset_constructed=False,image_decoded=False,annotation_records_parsed=False,model_forward_executed=False,
        optimizer_created=False,historical_lineage_source_hashing_may_include_data_bytes=True)
    try:
        need(not any(name.split('.')[0] in ('torch','numpy') for name in sys.modules),'No ML import before source/environment closure')
        spec_path=Path(args.spec);need(spec_path.is_absolute(),'Absolute original-driver-bound spec')
        raw=spec_path.read_bytes();spec=json.loads(raw)
        need(spec['schema']==SCHEMA and spec['operation']==OPERATION and type(spec['phase']) is str and spec['phase']
             and spec['expected_versions']==VERSIONS and spec['limits']==LIMITS,'Exact CPU-only restoration spec')
        report.update(phase=spec['phase'],spec_ref=reference(spec_path,raw),source_roles=spec['source_roles'],
            teacher_sources=spec['teacher_sources'],library_sources=spec['library_sources'],limits=spec['limits'])
        out=Path(spec['output_directory'])
        need(out.is_absolute() and out.drive.casefold()==out.resolve().drive.casefold()=='d:'
             and not out.exists() and not out.is_symlink(),'Fresh canonical D output directory')
        out.mkdir(parents=True,exist_ok=False);output_created=True
        need(set(spec['source_roles'])==set(ROLES) and set(spec['library_sources'])=={'psutil','torch','numpy'}
             and set(spec['teacher_sources'])=={'context','modules','registration'}
             and type(spec['sources']) is list and type(spec['input_sources']) is list,'Explicit source maps and closed input list')
        for value in spec['sources']:
            strict_ref(value);key=str(Path(value['path']).resolve()).lower()
            need(key not in source_map,'Unique actual admitted source path');source_map[key]=value

        def closed(value):
            strict_ref(value);path=Path(value['path']);key=str(path.resolve()).lower()
            need(source_map.get(key)==value,'Exact admitted input reference: '+value['path'])
            def identity():
                st=path.stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
            actual_identity=identity()
            if closed_cache.get(key)!=(value,actual_identity):
                need(reference(path)==value and identity()==actual_identity,'Exact stable admitted input bytes: '+value['path'])
                closed_cache[key]=(value,actual_identity)
            need(len(closed_cache)<=len(source_map),'Source-bounded per-leaf hash cache')
            return path

        # Stream hashes; never read whole weight containers into a byte buffer.
        for value in spec['sources']:closed(value)
        direct=list(spec['source_roles'].values())+list(spec['library_sources'].values())+spec['input_sources']
        direct += [spec['teacher_sources']['context']]+list(spec['teacher_sources']['modules'].values())+list(spec['teacher_sources']['registration'].values())
        for value in direct:need(source_map.get(str(Path(strict_ref(value)['path']).resolve()).lower())==value,'Every declared executing/input reference is admitted')
        need(Path(__file__).resolve()==Path(spec['source_roles']['entry_source']['path']).resolve(),'Actual entry self-source')
        restore_ref=spec['source_roles']['restore_source'];need((restore_ref['bytes'],restore_ref['sha256'])==RESTORE_PIN,'Exact reviewed teacher helper')
        flags['source_bytes_verified']=True
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Original driver must hide CUDA before ML import')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            need(os.environ.get(name)=='2','Original driver CPU thread environment: '+name)
        import psutil
        need(reference(Path(psutil.__file__).resolve())==spec['library_sources']['psutil'],'Actually imported canonical psutil source')
        process=psutil.Process();need(process.cpu_affinity()==[0,1],'Actual CPU0/1 affinity before ML import')
        report['library_paths']['psutil']=str(Path(psutil.__file__).resolve())

        def sample():
            try:
                rss,available=process.memory_info().rss,psutil.virtual_memory().available
                observed['peak_RSS_bytes']=max(observed['peak_RSS_bytes'],int(rss))
                old=observed['minimum_available_RAM_bytes'];observed['minimum_available_RAM_bytes']=int(available) if old is None else min(old,int(available))
                observed['samples']+=1
                if rss>LIMITS['maximum_RSS_bytes'] or available<LIMITS['minimum_available_RAM_bytes']:
                    observed['violations'].append('RSS/free-RAM boundary')
            except BaseException as error:observed['sampling_errors'].append(type(error).__name__+': '+str(error))

        def check():
            sample();need(not observed['violations'] and not observed['sampling_errors'],'CPU resource observation failed')
            need(time.monotonic()-started<=120,'Owned restoration deadline120 exceeded')

        def monitor():
            while not stopped.wait(.02):sample()

        class Control:
            allowed=source_map
            def closed(self,value):check();return closed(value)
            def check(self):check()

        check();sampler=threading.Thread(target=monitor,daemon=True);sampler.start()
        import torch as actual_torch
        import numpy as np
        torch=actual_torch
        for name,obj in (('torch',torch),('numpy',np)):
            need(reference(Path(obj.__file__).resolve())==spec['library_sources'][name],'Actually imported canonical '+name+' source')
            report['library_paths'][name]=str(Path(obj.__file__).resolve())
        report['actual_versions']=dict(torch=str(torch.__version__),numpy=str(np.__version__))
        need(report['actual_versions']==VERSIONS and not torch.cuda.is_initialized(),'Original numerical versions and cold CUDA')
        flags['CUDA_uninitialized_before']=True
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        need(torch.get_num_threads()==torch.get_num_interop_threads()==2,'Exactly two Torch threads')
        # These are exactly the registration imports used by original N26 setup.
        # Their six defining source files are preclosed above and rejoined by
        # teacher_restore to the actually injected objects before their use.
        from mmengine import Config
        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import init_default_scope
        import mmdet.registry as registry_module
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        from mmdet.datasets import CocoDataset
        registration=dict(Config=Config,revert_sync_batchnorm=revert_sync_batchnorm,init_default_scope=init_default_scope,
            registry_module=registry_module,MODELS=MODELS,register_all_modules=register_all_modules,CocoDataset=CocoDataset)
        check();helper=load(closed(restore_ref),'_U47_TEACHER_RESTORE',aliases)
        context=json.loads(closed(spec['teacher_sources']['context']).read_text(encoding='utf-8-sig'))
        modules={role:load(closed(value),'_U47_NUMERICAL_'+role,aliases) for role,value in spec['teacher_sources']['modules'].items()}
        result=helper.restore_teacher_cpu(torch,Control(),context,modules=modules,registration=registration,
            source_refs=spec['teacher_sources'],report=receipt)
        need(set(result)=={'model','receipt','teacher_state_proof'} and result['receipt'] is receipt
             and receipt['status']=='RESTORED_CPU_ONLY' and receipt['state_entries']==610 and receipt['parameter_tensors']==451
             and receipt['teacher_state_proof']==result['teacher_state_proof'],'Actual original teacher restoration receipt')
        model=result['model'];model_ref=weakref.ref(model);report['teacher_state_proof']=result['teacher_state_proof']
        report['teacher_state_proof_canonical_sha256']=hashlib.sha256(json.dumps(report['teacher_state_proof'],
            sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
        need(len(report['teacher_state_proof'])==610,'Full original610 state proof retained')
        result=None;check();need(not torch.cuda.is_initialized(),'Teacher restore stayed CPU-only')
        report['status']='COMPLETE'
    except BaseException as error:
        original_error=error;report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
        raise
    finally:
        # No model or independent anchor survives the owned worker. Receipt and
        # numerical state proof are JSON-only; caller/window use is a later task.
        model=result=modules=registration=context=helper=None;gc.collect()
        flags['teacher_model_released']=model_ref is not None and model_ref() is None
        if callable(sample):sample()
        stopped.set()
        if sampler is not None:
            try:
                sampler.join(timeout=1)
                if sampler.is_alive():observed['sampling_errors'].append('Sampler did not terminate')
            except BaseException as error:observed['sampling_errors'].append(type(error).__name__+': '+str(error))
        for alias,(existed,original) in aliases.items():
            if existed:sys.modules[alias]=original
            else:sys.modules.pop(alias,None)
        flags['module_aliases_restored']=all((alias in sys.modules)==existed and (not existed or sys.modules[alias] is original)
            for alias,(existed,original) in aliases.items())
        flags['CUDA_uninitialized_after']=torch is not None and not torch.cuda.is_initialized()
        flags['CPU_only']=flags['CUDA_uninitialized_before'] and flags['CUDA_uninitialized_after']
        report['cost']=dict(elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=observed['peak_RSS_bytes']);report['RSS_peak_is_sampled']=True
        report['source_hash_cache']=dict(scope='single admitted leaf; recheck file identity/size/mtime/ctime at each use',
            unique_paths_verified=len(closed_cache),maximum_paths=len(source_map),whole_environment_closure_claimed=False)
        bad=report['status']=='COMPLETE' and (not all(flags.values()) or bool(observed['violations'])
            or bool(observed['sampling_errors']) or report['cost']['elapsed_seconds']>120)
        if bad:report.update(status='FAILED',error=dict(type='RuntimeError',message='Final CPU teacher-release/source/resource contract failed'))
        if output_created and out.is_dir():
            try:
                with (out/'REPORT.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
            except BaseException:
                if original_error is None:raise
        if bad and original_error is None:raise RuntimeError('Final CPU teacher-release/source/resource contract failed')


if __name__=='__main__':main()
