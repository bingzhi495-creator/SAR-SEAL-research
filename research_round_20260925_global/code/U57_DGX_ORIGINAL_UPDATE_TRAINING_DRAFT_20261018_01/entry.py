"""UNBOUND Linux U57 worker; authoring/importing this module executes no ML."""
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


SCHEMA = 'ROOT-U57-DGX-original-update-training-spec-v1'
OPERATION = 'GPU_U57_DGX_ORIGINAL_UPDATE_TRAINING'
BODY_SHA256 = 'd80ab26ad7092431f4001535705e7afcc4fda7cbf66bacb4dc2cb66edd3d6724'
BODY_BYTES = 43423
SOURCE_PINS = {'shared_initialization': '4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
 'full_legacy': 'c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
 'full_dsconv': 'b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
 'full_head': '6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
 'full_feature': 'f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
 'plain_legacy': '902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
 'plain_feature': 'd93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
 'instance_adapter': '9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
 'image_geometry': '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
 'u36_helpers': '1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
 'u39_training_ops': '378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d',
 'portable_reader': 'b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557'}
LOAD_ORDER = (
    ('full_dsconv','dsconv'),('full_head','head'),('full_legacy','model_full'),
    ('full_feature','feature_backbone'),('plain_legacy','plain_model1'),
    ('plain_feature','plain_feature'),('instance_adapter','full_instance_adapter'),
    ('image_geometry','image_geometry'),('shared_initialization','shared_initialization'),
    ('u36_helpers','_U57_U36_HELPERS'),('u39_training_ops','_U57_U39_TRAINING_OPS'),
    ('portable_reader','_U57_PORTABLE_READER'),
)
GPU_MATH = dict(cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
    cudnn_benchmark=False, cudnn_deterministic=True)
SUCCESS_FLAGS = {'source_bytes_verified','CUDA_uninitialized_before','source_unchanged_after',
    'module_aliases_restored','caller_CPU_RNG_preserved','caller_NumPy_RNG_preserved',
    'caller_Python_RNG_preserved','dtype_grad_threads_preserved','backend_controls_restored',
    'child_resources_within_limits','caller_determinism_controls_restored'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path','bytes','sha256'}
         and type(value['path']) is str and PurePosixPath(value['path']).is_absolute()
         and str(PurePosixPath(value['path'])) == value['path'] and '..' not in PurePosixPath(value['path']).parts
         and type(value['bytes']) is int and value['bytes'] > 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict DGX-local absolute POSIX file reference')
    return value


def validate_cursor_bounds(start,stop,parent_pair):
    """Bounds and parent presence only; actual parent acceptance is checked separately."""
    need(type(start) is int and type(stop) is int and 0<=start<stop<=37050,
         'Exact nonempty independent global training span')
    need(parent_pair is None if start==0 else type(parent_pair) is dict,
         'Only a new zero cursor may omit its Root-accepted parent pair')
    return start,stop


def canonical_sources(values):
    expected={}
    for value in values:
        strict_ref(value)
        need(value['path'] not in expected or expected[value['path']]==value,
             'Repeated source path must retain the same exact bytes/SHA')
        expected[value['path']]=value
    return [expected[key] for key in sorted(expected)]


def parent_sources(parent):
    if parent is None:return []
    need(type(parent) is dict and set(parent)=={'qualified_ref','pair_commit_ref','checkpoint_refs','trace_refs'}
         and type(parent['checkpoint_refs']) is dict and set(parent['checkpoint_refs'])=={'PLAIN','FULL'}
         and type(parent['trace_refs']) is dict and set(parent['trace_refs'])=={'PLAIN','FULL'},
         'Explicit accepted parent proof/pair/two checkpoints/two traces')
    return [parent['qualified_ref'],parent['pair_commit_ref']]+list(parent['checkpoint_refs'].values())+list(parent['trace_refs'].values())


def source_table(spec):
    keys={'schema','operation','phase','entry_source','body_source','owner_source','source_refs',
          'library_sources','context','package_refs','segment','prelock','output_directory','owner_output_directory','sources'}
    need(type(spec) is dict and set(spec)==keys and spec['schema']==SCHEMA and spec['operation']==OPERATION
         and type(spec['phase']) is str and spec['phase'] and '/' not in spec['phase'],'Exact independent U57 spec')
    need(type(spec['source_refs']) is dict and set(spec['source_refs'])==set(SOURCE_PINS)
         and type(spec['library_sources']) is dict and set(spec['library_sources'])=={'psutil','torch','numpy'},
         'Twelve original numerical/reader components and three explicit selected libraries')
    for role,pin in SOURCE_PINS.items():
        need(strict_ref(spec['source_refs'][role])['sha256']==pin,'Exact original component: '+role)
    need(strict_ref(spec['body_source'])['sha256']==BODY_SHA256 and spec['body_source']['bytes']==BODY_BYTES,
         'Final reviewed independent U57 body pin; unfinished draft cannot run')
    context=spec['context'];segment=spec['segment']
    need(type(context) is dict and set(context)=={'manifest_ref','order_ref','order_generation_ref',
         'U54_validation_ref','U54_validation_sources','parent_pair'},'Exact explicit input and accepted-state context')
    need(type(segment) is dict and set(segment)=={'start','stop'},'Only the paired common global span')
    validate_cursor_bounds(segment['start'],segment['stop'],context['parent_pair'])
    validation=context['U54_validation_sources']
    need(type(validation) is dict and set(validation)=={'worker_report_ref','owner_receipt_ref'},
         'Explicit linked U54 worker and original owner evidence, not discovered inputs')
    packages=spec['package_refs']
    need(type(packages) is list and len(packages)==741,'All741 original-index package refs are pre-admitted')
    need(len({strict_ref(value)['path'] for value in packages})==741,'One distinct package per original ordinal')
    values=[spec[key] for key in ('entry_source','body_source','owner_source')]
    values+=list(spec['source_refs'].values())+list(spec['library_sources'].values())+packages
    values+=[context[key] for key in ('manifest_ref','order_ref','order_generation_ref','U54_validation_ref')]
    values+=list(validation.values())+parent_sources(context['parent_pair'])
    expected=canonical_sources(values)
    need(type(spec['sources']) is list and spec['sources']==expected,'Exact dynamic source closure, no old fixed file-count cap')
    outputs=[]
    for value in (spec['output_directory'],spec['owner_output_directory']):
        need(type(value) is str and PurePosixPath(value).is_absolute() and str(PurePosixPath(value))==value
             and value!='/' and '..' not in PurePosixPath(value).parts,'Canonical distinct POSIX output')
        outputs.append(PurePosixPath(value))
    worker,owner=outputs
    need(worker!=owner and worker not in owner.parents and owner not in worker.parents,'Separate fresh owned outputs')
    for value in expected:
        path=PurePosixPath(value['path'])
        need(all(out!=path and out not in path.parents for out in outputs),'Outputs cannot contain admitted sources/inputs')
    return {value['path']:value for value in expected}


def validate_U54_acceptance(c,spec):
    """Only Root creates this input after its actual complete CPU acceptance."""
    context=spec['context'];value=json.loads(c.closed(context['U54_validation_ref']).read_bytes())
    keys={'schema','status','root_accepted','images','GT','failed_loads','dataset_indices','manifest_ref',
          'package_refs_sha256','worker_report_ref','owner_receipt_ref','original_terminal','no_training','CUDA_uninitialized'}
    need(type(value) is dict and set(value)==keys and value['schema']=='Root-U54-DGX-full-FIT-accepted-v1'
         and value['status']=='ROOT_ACCEPTED_ALL741_CPU_VALIDATION' and value['root_accepted'] is True
         and type(value['images']) is int and value['images']==741
         and type(value['GT']) is int and value['GT']==1661
         and type(value['failed_loads']) is int and value['failed_loads']==0
         and value['dataset_indices']==list(range(741)) and value['manifest_ref']==context['manifest_ref']
         and value['no_training'] is True and value['CUDA_uninitialized'] is True,
         'Actual Root acceptance of all741/1661 CPU inputs is a prerequisite, not a draft boolean')
    package_sha=hashlib.sha256(json.dumps(spec['package_refs'],sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
    need(value['package_refs_sha256']==package_sha,'Same original-ordinal741 local packages as accepted U54')
    terminal={'root_observed_tool_exit_code':0,'SSH_returncode':0,'worker_returncode':0,'reaped':True,
              'owner_original_birth_absent':True,'child_original_birth_absent':True,
              'all_owned_processes_absent':True,'source_and_logs_byte_verified':True}
    need(type(value['original_terminal']) is dict and set(value['original_terminal'])==set(terminal)
         and all(type(value['original_terminal'][k]) is type(v) and value['original_terminal'][k]==v for k,v in terminal.items()),
         'Independent original U54 actual terminal, identity/reap and source/log acceptance')
    for key,reference_value in context['U54_validation_sources'].items():
        need(value[key]==reference_value,'Root acceptance linked original U54 evidence exact')
        c.closed(reference_value)
    manifest=json.loads(c.closed(context['manifest_ref']).read_bytes())
    need(type(manifest.get('rows')) is list and len(manifest['rows'])==741
         and [row['package_ref'] for row in manifest['rows']]==spec['package_refs'],
         'Manifest original package order is exactly the accepted declared population')
    return value


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load_closed(close, value, alias, originals):
    path = close(value)
    originals.setdefault(alias, (alias in sys.modules, sys.modules.get(alias)))
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    sys.modules[alias] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    need(Path(module.__file__).resolve() == path.resolve(), 'Actual loaded admitted module file: '+alias)
    return module


def math_settings(torch):
    return dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic)


def set_math(torch, values):
    torch.backends.cuda.matmul.allow_tf32 = values['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = values['cudnn_allow_tf32']
    torch.backends.cudnn.benchmark = values['cudnn_benchmark']
    torch.backends.cudnn.deterministic = values['cudnn_deterministic']


def determinism_settings(torch):
    return dict(deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def completed_strict_controls(report,policy):
    """Metadata completion guard; actual API/env observations come from this worker."""
    keys={'deterministic_algorithms_enabled','deterministic_warn_only'}
    need(report['CUBLAS_WORKSPACE_CONFIG_before_Torch_import']==':4096:8'
         and report['strict_backend_settings_before_body']==policy
         and report['strict_backend_settings_after_body']==policy
         and type(report['outside_determinism_before']) is dict
         and set(report['outside_determinism_before'])==keys
         and type(report['outside_determinism_after']) is dict
         and set(report['outside_determinism_after'])==keys
         and all(type(v) is bool for v in report['outside_determinism_before'].values())
         and all(type(v) is bool for v in report['outside_determinism_after'].values())
         and report['outside_determinism_before']==report['outside_determinism_after']
         and report['flags']['caller_determinism_controls_restored'] is True,
         'Actual strict workspace/algorithm controls throughout comparison and exact caller enabled/warn-only restoration')


def main(argv=None):
    started=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha',required=True);args=parser.parse_args(argv)
    out=c=torch=np=psutil=body_module=owner=modules=None;created=False
    outside_cpu=outside_numpy=outside_python=outside_math=outside_dtype=outside_grad=outside_determinism=None
    aliases,cache={},{};original_error=None
    report=dict(schema='U57-DGX-original-update-training-worker-report-v1',status='STARTED',operation=OPERATION,
        qualified=False,new_training=False,pilot_only=False,main_training=False,formal_main_training_started=False,
        formal_training_budget_credit=0,innovation_established=False,speedup_established=False,
        U40_frontier_modified=False,paired_prefix_advanced=False,AP=False,DEV=False,TEST=False,
        body_report={},flags={},cleanup_errors=[],whole_GPU_observed_peak_MiB=None,
        whole_GPU_WDDM8GiB_contract_certified=False,full_binary_closure_claimed=False,
        Linux_original_terminal_qualification_claimed=False)
    try:
        need(platform.system()=='Linux' and platform.machine()=='aarch64','Independent Linux/aarch64 training only')
        need(not any(name.split('.')[0] in ('torch','numpy') for name in sys.modules),'No ML before explicit source/environment closure')
        path=Path(args.spec);need(path.is_absolute() and not path.is_symlink(),'Absolute owner-pinned spec')
        raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==args.spec_sha,'Actual owner passes exact spec SHA')
        spec=json.loads(raw);admitted=source_table(spec)
        report.update(phase=spec['phase'],spec_ref=dict(path=str(path),bytes=len(raw),sha256=args.spec_sha),actual_library_paths={})
        for key in ('entry_source','body_source','owner_source','source_refs','library_sources','context','package_refs','segment','prelock'):
            report[key]=spec[key]
        out=Path(spec['output_directory'])
        need(not out.exists() and not out.is_symlink() and out.resolve()==out,'Fresh canonical worker output')
        out.mkdir(parents=True,exist_ok=False);created=True
        def close(value):
            strict_ref(value);file=Path(value['path'])
            need(admitted.get(value['path'])==value and file.resolve()==file and not file.is_symlink(),'Exact explicitly admitted local source')
            def identity():
                st=file.stat();return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns
            before=identity()
            if cache.get(value['path'])!=(value,before):
                need(reference(file)==value and identity()==before,'Stable exact admitted bytes/SHA')
                cache[value['path']]=(value,before)
            return file
        for value in spec['sources']:close(value)
        need(Path(__file__).resolve()==close(spec['entry_source']),'Actual admitted worker entry')
        report['flags']['source_bytes_verified']=True
        body_module=load_closed(close,spec['body_source'],'_U57_TRAINING_BODY',aliases)
        body_module.validate_prelock(spec['prelock'])
        owner=load_closed(close,spec['owner_source'],'_U57_LINUX_OWNER',aliases)
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','One explicitly visible GPU')
        report['CUBLAS_WORKSPACE_CONFIG_before_Torch_import']=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        need(report['CUBLAS_WORKSPACE_CONFIG_before_Torch_import']==':4096:8','Preimport independent cuBLAS policy')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            need(os.environ.get(key)=='2','Actual two-thread environment: '+key)
        psutil=importlib.import_module('psutil')
        need(reference(Path(psutil.__file__).resolve())==spec['library_sources']['psutil'],'Actual selected psutil source')
        c=owner.make_control(spec,psutil);c.check()
        report['U54_validation']=validate_U54_acceptance(c,spec)
        imported={'psutil':psutil}
        for name in ('torch','numpy'):
            imported[name]=importlib.import_module(name)
            need(reference(Path(imported[name].__file__).resolve())==spec['library_sources'][name],'Actual selected library: '+name)
            c.check()
        torch,np=imported['torch'],imported['numpy']
        report['actual_library_paths']={key:str(Path(module.__file__).resolve()) for key,module in imported.items()}
        actual=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__),system=platform.system(),machine=platform.machine())
        need(actual==spec['prelock']['environment'] and not torch.cuda.is_initialized()
             and torch.get_default_dtype()==torch.float32,'Exact independent runtime, cold CUDA/FP32')
        report['actual_environment']=actual;report['flags']['CUDA_uninitialized_before']=True
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        need(torch.get_num_threads()==torch.get_num_interop_threads()==2,'Actual Torch two-thread setup')
        outside_cpu=torch.get_rng_state().clone();outside_numpy=np.random.get_state();outside_python=random.getstate()
        outside_dtype,outside_grad=torch.get_default_dtype(),torch.is_grad_enabled()
        outside_math=math_settings(torch);outside_determinism=determinism_settings(torch)
        report['outside_determinism_before']=outside_determinism;report['outside_gpu_math_settings_before']=outside_math
        set_math(torch,GPU_MATH);torch.use_deterministic_algorithms(False,warn_only=False)
        report['strict_backend_settings_before_body']=body_module.backend_settings(torch)
        need(report['strict_backend_settings_before_body']==spec['prelock']['backend_policy']
             and not torch.cuda.is_initialized(),'Prelocked algorithms=False policy before first CUDA, not U56 strict mode')
        modules={}
        for role,alias in LOAD_ORDER:
            modules[role]=load_closed(c.closed,spec['source_refs'][role],alias,aliases)
            need(not torch.cuda.is_initialized(),'Original module definitions stay CPU-only');c.check()
        need(modules['full_legacy'].DSConv is modules['full_dsconv'].DSConv
             and modules['full_legacy'].CoordConv is modules['full_head'].CoordConv,'Exact original dependency objects')
        result=body_module.run_training(torch,np,c,modules=modules,source_refs=spec['source_refs'],
            context=spec['context'],segment=spec['segment'],prelock=spec['prelock'],
            body_source_ref=spec['body_source'],output_directory=Path(spec['output_directory'])/'SEGMENT',report=report['body_report'])
        need(result is report['body_report'],'Retain the same actual complete/partial report object')
        report['strict_backend_settings_after_body']=body_module.backend_settings(torch)
        need(report['strict_backend_settings_after_body']==spec['prelock']['backend_policy'],'Prelocked backend held for both arms')
        body_module.completed_training(result,spec['segment'],spec['prelock'])
        for value in spec['sources']:c.closed(value)
        report['flags']['source_unchanged_after']=True;c.check();report['status']='COMPLETE'
    except BaseException as error:
        original_error=error
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__);error.__traceback__=None
    finally:
        modules=body_module=owner=None;gc.collect()
        for alias,(existed,previous) in reversed(list(aliases.items())):
            if existed:sys.modules[alias]=previous
            else:sys.modules.pop(alias,None)
        report['flags']['module_aliases_restored']=all((alias in sys.modules)==existed
            and (not existed or sys.modules[alias] is previous) for alias,(existed,previous) in aliases.items())
        if torch is not None and outside_cpu is not None:
            try:
                report['flags']['caller_CPU_RNG_preserved']=bool(torch.equal(outside_cpu,torch.get_rng_state()))
                current=np.random.get_state()
                report['flags']['caller_NumPy_RNG_preserved']=current[0]==outside_numpy[0] and bool(np.array_equal(current[1],outside_numpy[1])) and current[2:]==outside_numpy[2:]
                report['flags']['caller_Python_RNG_preserved']=random.getstate()==outside_python
                report['flags']['dtype_grad_threads_preserved']=torch.get_default_dtype()==outside_dtype and torch.is_grad_enabled()==outside_grad and torch.get_num_threads()==torch.get_num_interop_threads()==2
            except BaseException as error:report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        if torch is not None and outside_math is not None:
            try:
                set_math(torch,outside_math);report['outside_gpu_math_settings_after']=math_settings(torch)
                report['flags']['backend_controls_restored']=report['outside_gpu_math_settings_after']==outside_math
            except BaseException as error:report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        if torch is not None and outside_determinism is not None:
            try:
                torch.use_deterministic_algorithms(outside_determinism['deterministic_algorithms_enabled'],warn_only=outside_determinism['deterministic_warn_only'])
                report['outside_determinism_after']=determinism_settings(torch)
                report['flags']['caller_determinism_controls_restored']=report['outside_determinism_after']==outside_determinism
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
        body=report['body_report']
        for key in ('successful_optimizer_updates','optimizer_steps_entered','new_training','pilot_only','main_training',
                    'formal_main_training_started','paired_checkpoint_prefix','pair_commit_ref','checkpoint_refs','trace_refs','mode','run_id'):
            if key in body:report[key]=body[key]
        report['main_training']=body.get('mode')=='FORMAL' and body.get('formal_main_training_started') is True
        report['cost']=dict(elapsed_seconds=time.monotonic()-started)
        if report['status']=='COMPLETE':
            try:completed_strict_controls(report,spec['prelock']['backend_policy'])
            except BaseException as error:report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error)))
        if report['status']=='COMPLETE' and (set(report['flags'])!=SUCCESS_FLAGS
             or not all(value is True for value in report['flags'].values()) or report['cleanup_errors']
             or report['cost']['elapsed_seconds']>1800):
            report.update(status='FAILED',error=dict(type='RuntimeError',message='Final source/resource/caller-control contract failed'))
        if created:
            with (out/'REPORT.json').open('x',encoding='utf-8') as stream:
                json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        if report['status']!='COMPLETE' and original_error is None:original_error=RuntimeError('U57 segment failed; original partial counters/state retained')
    if original_error is not None:raise original_error
    print('U57 paired original-update segment complete; pending actual owner/Root acceptance. No AP.',flush=True)


if __name__=='__main__':main()
