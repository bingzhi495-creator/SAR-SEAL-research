"""UNBOUND owned one-image native GPU worker. Prepared only; not executed."""
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

SCHEMA='ROOT-U48-external-native-capture-spec-v1'
OPERATION='GPU_U48_EXTERNAL_NATIVE_CAPTURE'
ROLES={'entry_source','body_source','restore_source','capture_source'}
VERSIONS=dict(torch='2.0.1+cu118',numpy='1.23.5')
LIMITS=dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
    minimum_available_RAM_bytes=16*1024**3,deadline_seconds=180,CUDA_visible_devices='0')
ALLOCATOR=3*1024**3
PINS=dict(restore_source=(12320,'019e2741d739b38ec33f015a1511cf9370d7828b040cfd1b4b878fd9dcd1afaa'),
    capture_source=(16047,'7fe5faf26a189216980cb527d4f9f2b44804a27a0b1c157d5eb7b4411ef37c0b'),
    body_source=(15814,'c7ce35eed245f6c50c363cbb9fb5711816edf40d2dd6a484db8f35f4e6838c72'))
CAPTURE_MAP=dict(Compose='Compose',PackDetInputs='formatting',pseudo_collate='pseudo_collate',
    observer='observer',renderer='renderer',exporter='exporter',pref='pref',masks='masks')
CAPTURE_KEYS=set(CAPTURE_MAP.values())|{'data_preprocessor','base_preprocessor'}


def need(ok,message):
    if not ok:raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Strict original admitted reference')
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
    need(Path(module.__file__).resolve()==path.resolve(),'Actual loaded original module source')
    return module


def main(argv=None):
    started=time.monotonic();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True)
    args=parser.parse_args(argv);out=None;created=False;torch=None;process=None;sample=None;sampler=None
    stopped=threading.Event();aliases={};source_map={};cache={};original_error=None;packet_path=None
    result=packet=body=restore=capture=restore_modules=capture_modules=registration=context=outside_cpu=None
    outside_backends=read_backends=None
    body_report={};observed=dict(peak_RSS_bytes=0,minimum_available_RAM_bytes=None,samples=0,violations=[],sampling_errors=[])
    flags=dict(source_bytes_verified=False,CUDA_uninitialized_before=False,GPU_capture_executed=False,
        CPU_packet_only=False,module_aliases_restored=False,GPU_cleanup_zero=False,callerCPU_RNG_preserved=False,
        default_dtype_preserved=False,grad_mode_preserved=False,thread_controls_preserved=False,backend_controls_restored=False)
    report=dict(schema='U48-single-external-native-capture-report-v1',status='STARTED',operation=OPERATION,
        qualified=False,new_training=False,AP=False,DEV=False,TEST=False,body_report=body_report,flags=flags,
        packet_ref=None,packet_summary=None,teacher_state_proof=None,teacher_state_proof_canonical_sha256=None,
        actual_versions={},library_paths={},resource_observations=observed,full_binary_closure_claimed=False,
        dataset_constructed=False,new_image_annotation_records_parsed=False,optimizer_created=False,
        Source_byte_closure_may_open_annotation_files=True,
        annotation_flags_scope='image-only numerical path; Source hashing may open historical or version-chain annotation bytes',
        gpu_math_settings=None,gpu_math_settings_after_body=None,gpu_math_settings_set_before_first_CUDA=False,
        outside_gpu_math_settings_before=None,outside_gpu_math_settings_after=None,backend_restore_errors=[])
    try:
        need(not any(name.split('.')[0] in ('torch','numpy') for name in sys.modules),'No ML before original environment/Source closure')
        spec_path=Path(args.spec);need(spec_path.is_absolute(),'Absolute original-driver-bound spec')
        raw=spec_path.read_bytes();spec=json.loads(raw)
        need(spec['schema']==SCHEMA and spec['operation']==OPERATION and spec['limits']==LIMITS
             and spec['expected_versions']==VERSIONS and spec['allocator_limit_bytes']==ALLOCATOR
             and type(spec['phase']) is str and spec['phase'],'Exact original one-image GPU contract')
        for key in ('source_roles','teacher_sources','library_sources','body_sources','capture_source_refs',
                    'image_metadata','selection','selection_proof_ref','loading_receipt_ref','allocator_limit_bytes','limits'):
            report[key]=spec[key]
        report.update(phase=spec['phase'],spec_ref=reference(spec_path,raw))
        out=Path(spec['output_directory'])
        need(out.is_absolute() and out.drive.casefold()==out.resolve().drive.casefold()=='d:'
             and not out.exists() and not out.is_symlink(),'Fresh canonical D single-image output')
        out.mkdir(parents=True,exist_ok=False);created=True;packet_path=out/'NATIVE_PACKET.pt'
        need(set(spec['source_roles'])==ROLES and set(spec['library_sources'])=={'psutil','torch','numpy'}
             and set(spec['teacher_sources'])=={'context','modules','registration'}
             and set(spec['body_sources'])==set(CAPTURE_MAP) and set(spec['capture_source_refs'])==CAPTURE_KEYS
             and type(spec['sources']) is list and type(spec['input_sources']) is list,'Exact admitted object/source maps')
        for role,key in CAPTURE_MAP.items():need(spec['body_sources'][role]==spec['capture_source_refs'][key],'Exact capture source alias: '+role)
        need(spec['body_sources']['pref']==spec['teacher_sources']['modules']['pref'],'Same original pref for restoration/capture')
        need(spec['selection']==dict(dataset='HRSID',split='train',dataset_index=0)
             and set(spec['image_metadata'])=={'image_id','image_ref','original_shape'}
             and type(spec['image_metadata']['image_id']) is int and spec['image_metadata']['image_id']==0
             and spec['image_metadata']['original_shape']==[800,800],'Original preselected HRSID TRAIN0 image-only identity')
        for value in spec['sources']:
            strict_ref(value);key=str(Path(value['path']).resolve()).lower()
            need(key not in source_map,'Unique admitted Source path');source_map[key]=value

        def closed(value):
            strict_ref(value);path=Path(value['path']);key=str(path.resolve()).lower()
            need(source_map.get(key)==value,'Exact input Source membership')
            def identity():
                st=path.stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
            observed_identity=identity()
            if cache.get(key)!=(value,observed_identity):
                need(reference(path)==value and identity()==observed_identity,'Stable exact Source bytes: '+value['path'])
                cache[key]=(value,observed_identity)
            need(len(cache)<=len(source_map),'Explicit Source-bounded single-leaf hash cache');return path

        for value in spec['sources']:closed(value)
        direct=list(spec['source_roles'].values())+list(spec['library_sources'].values())+spec['input_sources']
        direct += [spec['teacher_sources']['context']]+list(spec['teacher_sources']['modules'].values())+list(spec['teacher_sources']['registration'].values())
        direct += list(spec['body_sources'].values())+list(spec['capture_source_refs'].values())
        direct += [spec['image_metadata']['image_ref'],spec['selection_proof_ref'],spec['loading_receipt_ref']]
        for value in direct:need(source_map.get(str(Path(strict_ref(value)['path']).resolve()).lower())==value,'Every declared dependency is admitted')
        for role,pin in PINS.items():
            value=spec['source_roles'][role];need((value['bytes'],value['sha256'])==pin,'Original reviewed helper pin: '+role)
        need(Path(__file__).resolve()==Path(spec['source_roles']['entry_source']['path']).resolve(),'Actual entry source')
        flags['source_bytes_verified']=True
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Original driver must expose only CUDA0')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            need(os.environ.get(name)=='2','Original CPU thread environment: '+name)
        import psutil
        need(reference(Path(psutil.__file__).resolve())==spec['library_sources']['psutil'],'Canonical actual psutil source')
        process=psutil.Process();need(process.cpu_affinity()==[0,1],'Actual CPU0/1 affinity')
        report['library_paths']['psutil']=str(Path(psutil.__file__).resolve())

        def sample():
            try:
                rss,available=process.memory_info().rss,psutil.virtual_memory().available
                observed['peak_RSS_bytes']=max(observed['peak_RSS_bytes'],int(rss));prior=observed['minimum_available_RAM_bytes']
                observed['minimum_available_RAM_bytes']=int(available) if prior is None else min(prior,int(available));observed['samples']+=1
                if rss>LIMITS['maximum_RSS_bytes'] or available<LIMITS['minimum_available_RAM_bytes']:
                    observed['violations'].append('RSS/free-RAM boundary')
            except BaseException as error:observed['sampling_errors'].append(type(error).__name__+': '+str(error))

        def check():
            sample();need(not observed['violations'] and not observed['sampling_errors'],'Actual RAM/RSS observations')
            need(time.monotonic()-started<=180,'Original180-second window exceeded')

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
            need(reference(Path(obj.__file__).resolve())==spec['library_sources'][name],'Canonical actual '+name+' source')
            report['library_paths'][name]=str(Path(obj.__file__).resolve())
        report['actual_versions']=dict(torch=str(torch.__version__),numpy=str(np.__version__))
        need(report['actual_versions']==VERSIONS and not torch.cuda.is_initialized(),'Original numerical versions and initially cold CUDA')
        flags['CUDA_uninitialized_before']=True
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        need(torch.get_num_threads()==torch.get_num_interop_threads()==2,'Exactly two Torch threads')

        def read_backends():
            return dict(cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic))

        outside_backends=read_backends();report['outside_gpu_math_settings_before']=outside_backends
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        report['gpu_math_settings']=read_backends()
        report['gpu_math_settings_set_before_first_CUDA']=not torch.cuda.is_initialized()
        need(report['gpu_math_settings']==dict(cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False,
             cudnn_benchmark=False,cudnn_deterministic=True) and report['gpu_math_settings_set_before_first_CUDA'],
             'Original N26 FP32/noTF32/deterministic backend policy before first CUDA')
        from mmengine import Config
        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import init_default_scope
        import mmdet.registry as registry_module
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        from mmdet.datasets import CocoDataset
        from mmengine.dataset import Compose,pseudo_collate
        from mmdet.datasets.transforms import PackDetInputs
        from pycocotools import mask as masks
        registration=dict(Config=Config,revert_sync_batchnorm=revert_sync_batchnorm,init_default_scope=init_default_scope,
            registry_module=registry_module,MODELS=MODELS,register_all_modules=register_all_modules,CocoDataset=CocoDataset)
        body=load(closed(spec['source_roles']['body_source']),'_U48_CAPTURE_BODY',aliases)
        restore=load(closed(spec['source_roles']['restore_source']),'_U48_TEACHER_RESTORE',aliases)
        capture=load(closed(spec['source_roles']['capture_source']),'_U48_NATIVE_CAPTURE',aliases)
        restore_modules={role:load(closed(value),'_U48_RESTORE_'+role,aliases) for role,value in spec['teacher_sources']['modules'].items()}
        capture_modules=dict(Compose=Compose,PackDetInputs=PackDetInputs,pseudo_collate=pseudo_collate,masks=masks,pref=restore_modules['pref'])
        for role in ('observer','renderer','exporter'):capture_modules[role]=load(closed(spec['body_sources'][role]),'_U48_CAPTURE_'+role,aliases)
        context=json.loads(closed(spec['teacher_sources']['context']).read_text(encoding='utf-8-sig'));check()
        outside_cpu=torch.get_rng_state().clone();before_dtype=torch.get_default_dtype();before_grad=torch.is_grad_enabled()
        before_threads=(torch.get_num_threads(),torch.get_num_interop_threads())
        result=body.capture_one(torch,np,Control(),context,spec['image_metadata'],selection=spec['selection'],
            allocator_limit_bytes=ALLOCATOR,restore_module=restore,restore_modules=restore_modules,registration=registration,
            restore_source_refs=spec['teacher_sources'],capture_module=capture,capture_modules=capture_modules,
            capture_source_refs=spec['capture_source_refs'],helper_source_refs=dict(restore=spec['source_roles']['restore_source'],
                capture=spec['source_roles']['capture_source']),loading_receipt_ref=spec['loading_receipt_ref'],report=body_report)
        flags['callerCPU_RNG_preserved']=bool(torch.equal(outside_cpu,torch.get_rng_state()))
        flags['default_dtype_preserved']=torch.get_default_dtype()==before_dtype
        flags['grad_mode_preserved']=torch.is_grad_enabled()==before_grad
        flags['thread_controls_preserved']=(torch.get_num_threads(),torch.get_num_interop_threads())==before_threads
        report['gpu_math_settings_after_body']=read_backends()
        need(report['gpu_math_settings_after_body']==report['gpu_math_settings'],'Original GPU math settings unchanged during capture')
        need(set(result)=={'packet','report'} and result['report'] is body_report
             and body_report['status']=='CAPTURED_CPU_PENDING_ROOT_ACTUAL_AND_INTAKE'
             and body_report['GPU_postinitialization_RNG_preserved'] is True,'Actual one-image capture and post-init GPU RNG')
        packet=result['packet'];result=None

        def cpu_only(value):
            if isinstance(value,torch.Tensor):return value.device.type=='cpu' and not value.requires_grad
            if isinstance(value,dict):return all(cpu_only(v) for v in value.values())
            if isinstance(value,(list,tuple)):return all(cpu_only(v) for v in value)
            return value is None or type(value) in (int,float,bool,str,bytes)

        flags['CPU_packet_only']=cpu_only(packet)
        qids=packet['export_query_ids'];predictions=packet['native_predictions'];receipt=packet['receipt']
        need(packet['natural_query_ids']==list(range(100)) and type(qids) is list and sorted(qids)==list(range(100))
             and len(predictions)==len(packet['S0_by_query'])==len(packet['score_factors_by_query'])==100
             and [row['query_id'] for row in predictions]==qids and packet['category_ids_by_query']==[1]*100
             and all(row['image_id']==0 and row['category_id']==1 and row['score']==packet['S0_by_query'][row['query_id']] for row in predictions),
             'Original natural/export100 bijection, all S0 and class/order preserved')
        shapes=dict(F=[1,256,200,200],e=[1,100,256],M=[1,100,200,200],C=[1,100,2],L=[100,800,800])
        need(all(isinstance(packet[k],torch.Tensor) and list(packet[k].shape)==shape and packet[k].dtype==torch.float32 for k,shape in shapes.items()),'Original complete native field shapes')
        need(receipt==body_report['native_capture_receipt'] and receipt['images']==1 and receipt['queries']==100
             and receipt['teacher_forward_calls']==body_report['original_teacher_forward_calls']==1
             and receipt['original_fusion_masks_scores_labels_exact'] is True
             and receipt['annotations_opened'] is False and receipt['GT_forwarded'] is False
             and receipt['loss_or_matching_called'] is False and receipt['optimizer_created'] is False,'Actual original native calibration and image-only call')
        flags['GPU_capture_executed']=True
        report['packet_summary']=dict(shapes=shapes,natural_query_ids=packet['natural_query_ids'],export_query_ids=qids,
            predictions_count=len(predictions),S0_count=len(packet['S0_by_query']),score_factors_count=len(packet['score_factors_by_query']),
            class_packet_entries=len(packet['class_packet']['values']),category_ids_by_query=packet['category_ids_by_query'],
            native_RLE_calibration=dict(original_fusion_masks_scores_labels_exact=True,RLE_from_original_exporter=True,independent_RLE_decode_executed=False))
        need(report['packet_summary']['class_packet_entries']==100 and flags['CPU_packet_only'],'Complete100 class packet and CPU-only serialization')
        report['teacher_state_proof']=body_report['teacher_load_receipt']['teacher_state_proof']
        need(len(report['teacher_state_proof'])==610,'Original610 teacher state proof')
        report['teacher_state_proof_canonical_sha256']=hashlib.sha256(json.dumps(report['teacher_state_proof'],sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
        need(all(flags[k] for k in ('callerCPU_RNG_preserved','default_dtype_preserved','grad_mode_preserved','thread_controls_preserved')),'Actual caller controls preserved')
        check()
        with packet_path.open('xb') as stream:torch.save(packet,stream)
        packet=None;gc.collect();report['packet_ref']=reference(packet_path);check();report['status']='COMPLETE'
    except BaseException as error:
        original_error=error;report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
        raise
    finally:
        result=packet=body=restore=capture=restore_modules=capture_modules=registration=context=outside_cpu=None;gc.collect()
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
        flags['module_aliases_restored']=all((alias in sys.modules)==existed and (not existed or sys.modules[alias] is original) for alias,(existed,original) in aliases.items())
        if outside_backends is not None:
            try:
                torch.backends.cuda.matmul.allow_tf32=outside_backends['cuda_matmul_allow_tf32']
                torch.backends.cudnn.allow_tf32=outside_backends['cudnn_allow_tf32']
                torch.backends.cudnn.benchmark=outside_backends['cudnn_benchmark']
                torch.backends.cudnn.deterministic=outside_backends['cudnn_deterministic']
                report['outside_gpu_math_settings_after']=read_backends()
                flags['backend_controls_restored']=report['outside_gpu_math_settings_after']==outside_backends
            except BaseException as error:report['backend_restore_errors'].append(type(error).__name__+': '+str(error))
        initialized=torch is not None and torch.cuda.is_initialized()
        report['cleanup']=dict(CUDA_initialized=initialized,allocated_bytes=int(torch.cuda.memory_allocated(0)) if initialized else None,
            reserved_bytes=int(torch.cuda.memory_reserved(0)) if initialized else None,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)) if initialized else None)
        flags['GPU_cleanup_zero']=initialized and report['cleanup']['allocated_bytes']==report['cleanup']['reserved_bytes']==0
        report['cost']=dict(elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=observed['peak_RSS_bytes']);report['RSS_peak_is_sampled']=True
        report['source_hash_cache']=dict(unique_paths_verified=len(cache),maximum_paths=len(source_map),scope='single admitted leaf; identity/size/mtime/ctime rechecked each use')
        if packet_path is not None and packet_path.exists() and report['packet_ref'] is None:
            report['partial_packet_file']=dict(path=str(packet_path),bytes=packet_path.stat().st_size,complete=False)
        bad=report['status']=='COMPLETE' and (not all(flags.values()) or bool(observed['violations']) or bool(observed['sampling_errors'])
            or bool(report['backend_restore_errors'])
            or report['cost']['elapsed_seconds']>180 or report['cleanup']['peak_allocated_bytes']>ALLOCATOR)
        if bad:report.update(status='FAILED',error=dict(type='RuntimeError',message='Final caller/source/resource/cleanup contract failed'))
        if created and out.is_dir():
            try:
                with (out/'REPORT.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,allow_nan=False,indent=2)
            except BaseException:
                if original_error is None:raise
        if bad and original_error is None:raise RuntimeError('Final caller/source/resource/cleanup contract failed')


if __name__=='__main__':main()
