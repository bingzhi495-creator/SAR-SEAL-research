"""UNBOUND Linux U52 worker; authoring/importing this module executes no ML."""
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


SCHEMA = 'ROOT-U52-DGX-equivalence-spec-v1'
OPERATION = 'GPU_U52_DGX_SHARED_ENCODER_EQUIVALENCE'
BODY_SHA256 = '82ae34c912696bff1cabe0d88a89c8a50757a4f3e75322378cc7c0a91854339a'
BODY_BYTES = 32259
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
 'u50_pilot': '003641b7c81e198bb70502073aba22f623f9dac82554ca9ce90ba059ac20625b',
 'u50_loader': 'aceadae1454b6a99a1a191e9c71ad2b9b636914f4e9a85dc4907a92eeb09eee1',
 'candidate': '80e6b777ccd1b41cbdc5bd4ef34f4e4d3d429b827f039c830b78bc08a847eed9'}
GPU_MATH = dict(cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
    cudnn_benchmark=False, cudnn_deterministic=True)
SUCCESS_FLAGS = {'source_bytes_verified','CUDA_uninitialized_before','source_unchanged_after',
    'module_aliases_restored','caller_CPU_RNG_preserved','caller_NumPy_RNG_preserved',
    'caller_Python_RNG_preserved','dtype_grad_threads_preserved','backend_controls_restored',
    'child_resources_within_limits'}


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


def source_table(spec):
    """Pure explicit source/context closure, before loading any real libraries."""
    required = {'schema','operation','phase','entry_source','body_source','owner_source','source_refs',
        'library_sources','context','prelock','output_directory','owner_output_directory','sources'}
    need(type(spec) is dict and set(spec) == required and spec['schema'] == SCHEMA
         and spec['operation'] == OPERATION and type(spec['phase']) is str and spec['phase']
         and '/' not in spec['phase'], 'Exact independent U52 worker spec')
    need(type(spec['source_refs']) is dict and set(spec['source_refs']) == set(SOURCE_PINS)
         and type(spec['library_sources']) is dict and set(spec['library_sources']) == {'psutil','torch','numpy'},
         'Exactly fourteen source roles and three selected libraries')
    context=spec['context']
    need(type(context) is dict and set(context)=={'portable_refs','scratch_directory'}
         and type(context['portable_refs']) is dict and set(context['portable_refs'])=={'54','710'},
         'Original two private U49 portable refs and exclusive audit scratch')
    for role, sha in SOURCE_PINS.items():
        need(strict_ref(spec['source_refs'][role])['sha256'] == sha, 'Exact original/candidate source: '+role)
    need(strict_ref(spec['body_source'])['sha256'] == BODY_SHA256 and spec['body_source']['bytes'] == BODY_BYTES,
         'Exact reviewed U52 equivalence body')
    values = [spec[k] for k in ('entry_source','body_source','owner_source')]
    values += list(spec['source_refs'].values())+list(spec['library_sources'].values())+list(context['portable_refs'].values())
    expected = {}
    for value in values:
        strict_ref(value)
        need(value['path'] not in expected, 'Twenty-two distinct actual local files')
        expected[value['path']] = value
    need(len(expected)==22 and type(spec['sources']) is list
         and spec['sources'] == sorted(values,key=lambda value:value['path']),
         'Exact sorted22 Source closure, no hidden input or old Windows chain')
    for path in (spec['output_directory'],spec['owner_output_directory'],context['scratch_directory']):
        need(type(path) is str and PurePosixPath(path).is_absolute() and path != '/'
             and str(PurePosixPath(path)) == path and '..' not in PurePosixPath(path).parts,
             'Explicit canonical POSIX output/scratch path')
    worker,owner,scratch=(PurePosixPath(value) for value in
        (spec['output_directory'],spec['owner_output_directory'],context['scratch_directory']))
    need(worker!=owner and worker not in owner.parents and owner not in worker.parents
         and worker in scratch.parents,'Separate owner/worker outputs; private scratch within this worker attempt')
    for value in expected:
        need(worker!=PurePosixPath(value) and worker not in PurePosixPath(value).parents
             and owner!=PurePosixPath(value) and owner not in PurePosixPath(value).parents,
             'Output directories cannot contain admitted sources or inputs')
    return expected


def validate_spec(spec, prelock_validator):
    sources = source_table(spec)
    prelock_validator(spec['prelock'])  # Actual source-closed original pure contract, not a new recipe.
    return sources


def completed_equivalence(report):
    """Negative candidate comparison is valid completion; operational/control failure is not."""
    verdicts={'WITHIN_PRELOCKED_TOLERANCES','CANDIDATE_OUTSIDE_PRELOCKED_TOLERANCES'}
    need(report.get('status')=='COMPARISON_COMPLETED_PENDING_ROOT' and report.get('verdict') in verdicts
         and report['new_training'] is report['pilot_only'] is True and report['formal_training_budget_credit']==0
         and all(report[key] is False for key in ('main_training','U40_frontier_modified','qualified','AP','DEV','TEST',
              'innovation_established','speedup_established'))
         and report['successful_optimizer_updates']==report['optimizer_steps_entered']==14
         and all(report[key] is True for key in ('caller_CPU_RNG_restored','caller_NumPy_RNG_restored',
              'caller_Python_RNG_restored','caller_postinitialization_CUDA_RNG_restored','module_aliases_restored'))
         and report['cleanup_errors']==[] and report['final_CUDA_cleanup']==dict(allocated_bytes=0,reserved_bytes=0)
         and report['reader_visits']==[54,710,54,710,54,710,54]*2 and set(report['arms'])=={'PLAIN','FULL'},
         'Fourteen real disposable steps, complete controlled comparison and actual clean release/RNG')
    candidate=[]
    expected_trajectory=['WARMUP','BASE_A','BASE_A','BASE_B','BASE_B','CANDIDATE','CANDIDATE']
    for arm in ('PLAIN','FULL'):
        record=report['arms'][arm];stages=record['stages']
        need(record['status']=='COMPARISON_COMPLETED' and record['updates_completed']==7
             and record['baseline_control_exact'] is True and type(record['candidate_within']) is bool
             and record['warm_head_nonzero'] is True
             and record['final_CUDA_cleanup']==dict(allocated_bytes=0,reserved_bytes=0)
             and len(stages)==7 and [row['trajectory'] for row in stages]==expected_trajectory
             and [row['dataset_index'] for row in stages]==[54,710,54,710,54,710,54]
             and [row['G'] for row in stages]==[1,27,1,27,1,27,1]
             and [row['step'] for row in stages]==[1,2,3,2,3,2,3]
             and all(row['optimizer_step_returned'] is True for row in stages),
             'One warmup and three carried trajectories per arm; baseline exact is mandatory')
        for row in stages:
            calls=row['calls']
            need(calls['backbone']==calls['feature_exit']==1 and calls['legacy_forward']==0
                 and all(calls['final_super_0_'+str(i)]==0 for i in range(1,5)),
                 'Exactly one encode and no legacy semantic-head execution')
        runs=record['trajectories']
        need(set(runs)=={'BASE_A','BASE_B','CANDIDATE'}
             and all(run['steps']==[2,3] and run['after_release']==dict(allocated_bytes=0,reserved_bytes=0)
                     for run in runs.values())
             and runs['BASE_A']['passed'] is runs['BASE_B']['passed'] is True
             and runs['CANDIDATE']['passed'] is record['candidate_within'],
             'Actual complete baseline/candidate comparison and between-trajectory GPU release')
        candidate.append(record['candidate_within'])
    need(report['verdict']==('WITHIN_PRELOCKED_TOLERANCES' if all(candidate)
         else 'CANDIDATE_OUTSIDE_PRELOCKED_TOLERANCES'),'Recorded verdict follows both actual candidate comparisons')


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


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha', required=True)
    args = parser.parse_args(argv)
    out = c = torch = np = psutil = body_module = owner = None
    outside_cpu = outside_numpy = outside_python = outside_math = outside_dtype = outside_grad = None
    aliases, cache = {}, {}
    original_error = None
    report = dict(schema='U52-DGX-equivalence-worker-report-v1', status='STARTED', operation=OPERATION,
        qualified=False, pilot_only=True, main_training=False, formal_training_budget_credit=0,
        innovation_established=False, speedup_established=False, diagnostic_verdict=None, candidate_within_prelocked_tolerances=None,
        U40_frontier_modified=False, AP=False, DEV=False, TEST=False, body_report={}, flags={}, cleanup_errors=[],
        whole_GPU_observed_peak_MiB=None, whole_GPU_WDDM8GiB_contract_certified=False,
        full_binary_closure_claimed=False, Linux_original_terminal_qualification_claimed=False)
    try:
        need(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Linux aarch64 worker, never Windows terminal emulation')
        need(not any(name.split('.')[0] in ('torch','numpy') for name in sys.modules), 'No ML import before Source/environment closure')
        path = Path(args.spec)
        need(path.is_absolute() and not path.is_symlink(), 'Absolute actual spec file')
        raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Original owner passes exact spec SHA256')
        spec = json.loads(raw)
        admitted = source_table(spec)
        report.update(phase=spec['phase'], spec_ref=dict(path=str(path),bytes=len(raw),sha256=args.spec_sha),
            source_refs=spec['source_refs'], context=spec['context'], prelock=spec['prelock'],
            actual_library_paths={})
        for key in ('entry_source','body_source','owner_source','library_sources'):
            report[key] = spec[key]
        out = Path(spec['output_directory'])
        need(not out.exists() and not out.is_symlink() and out.resolve() == out, 'Fresh canonical worker output directory')
        out.mkdir(parents=True, exist_ok=False)

        def close(value):
            strict_ref(value)
            file = Path(value['path'])
            need(admitted.get(value['path']) == value and file.resolve() == file and not file.is_symlink(), 'Exact canonical admitted local file')
            def identity():
                stat = file.stat()
                return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns
            before = identity()
            if cache.get(value['path']) != (value,before):
                need(reference(file) == value and identity() == before, 'Stable admitted file bytes/SHA')
                cache[value['path']] = (value,before)
            return file

        for value in spec['sources']:
            close(value)
        need(Path(__file__).resolve() == close(spec['entry_source']), 'This actual admitted worker entry')
        report['flags']['source_bytes_verified'] = True
        body_module = load_closed(close, spec['body_source'], '_U52_EQUIVALENCE_BODY', aliases)
        validate_spec(spec, body_module.validate_prelock)
        body_module.validate_context_metadata(spec['context'])
        owner = load_closed(close, spec['owner_source'], '_U52_LINUX_OWNER', aliases)
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Original Linux owner exposes one GPU')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            need(os.environ.get(key) == '2', 'Original owner two-thread environment: '+key)
        psutil = importlib.import_module('psutil')
        need(reference(Path(psutil.__file__).resolve()) == spec['library_sources']['psutil'], 'Actual selected psutil file')
        c = owner.make_control(spec, psutil)
        c.check()
        imported = {'psutil': psutil}
        for name in ('torch','numpy'):
            imported[name] = importlib.import_module(name)
            need(reference(Path(imported[name].__file__).resolve()) == spec['library_sources'][name], 'Actual selected library file: '+name)
            c.check()
        torch, np = imported['torch'], imported['numpy']
        report['actual_library_paths'] = {key:str(Path(module.__file__).resolve()) for key,module in imported.items()}
        actual = dict(python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__),system=platform.system(),machine=platform.machine())
        need(actual == spec['prelock']['environment'] and not torch.cuda.is_initialized()
             and torch.get_default_dtype() == torch.float32, 'Exact independent runtime, cold CUDA and original FP32 default')
        report['actual_environment'] = actual
        report['flags']['CUDA_uninitialized_before'] = True
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        need(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Actual Torch two-thread setup')
        outside_cpu = torch.get_rng_state().clone()
        outside_numpy, outside_python = np.random.get_state(), random.getstate()
        outside_dtype, outside_grad = torch.get_default_dtype(), torch.is_grad_enabled()
        outside_math = math_settings(torch)
        report['outside_gpu_math_settings_before'] = outside_math
        set_math(torch, GPU_MATH)
        need(math_settings(torch) == GPU_MATH and not torch.cuda.is_initialized(), 'Explicit FP32 backend settings before first CUDA')
        report['gpu_math_settings'] = math_settings(torch)
        print('U52 independent DGX equivalence starts: original warmup, BASE_A/BASE_B and candidate carried G27/G1.', flush=True)
        result = body_module.run_equivalence(torch,np,c,context=spec['context'],source_refs=spec['source_refs'],
            prelock=spec['prelock'],body_source_ref=spec['body_source'],report=report['body_report'])
        need(result is report['body_report'], 'Preserve exact complete/partial equivalence report object')
        completed_equivalence(result)
        report['gpu_math_settings_after_body'] = math_settings(torch)
        need(report['gpu_math_settings_after_body'] == GPU_MATH, 'Numerical backend settings retained throughout equivalence diagnostic')
        for value in spec['sources']:
            c.closed(value)
        report['flags']['source_unchanged_after'] = True
        c.check()
        report['status'] = 'COMPLETE'
    except BaseException as error:
        original_error = error
        report.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    finally:
        body_module = owner = None
        for alias, (existed, previous) in reversed(list(aliases.items())):
            if existed:
                sys.modules[alias] = previous
            else:
                sys.modules.pop(alias, None)
        report['flags']['module_aliases_restored'] = all((alias in sys.modules) == existed
            and (not existed or sys.modules[alias] is previous) for alias,(existed,previous) in aliases.items())
        gc.collect()
        if torch is not None and outside_cpu is not None:
            try:
                report['flags']['caller_CPU_RNG_preserved'] = bool(torch.equal(outside_cpu,torch.get_rng_state()))
                current = np.random.get_state()
                report['flags']['caller_NumPy_RNG_preserved'] = current[0] == outside_numpy[0] and bool(np.array_equal(current[1],outside_numpy[1])) and current[2:] == outside_numpy[2:]
                report['flags']['caller_Python_RNG_preserved'] = random.getstate() == outside_python
                report['flags']['dtype_grad_threads_preserved'] = torch.get_default_dtype() == outside_dtype and torch.is_grad_enabled() == outside_grad and torch.get_num_threads() == torch.get_num_interop_threads() == 2
                set_math(torch,outside_math)
                report['outside_gpu_math_settings_after'] = math_settings(torch)
                report['flags']['backend_controls_restored'] = report['outside_gpu_math_settings_after'] == outside_math
            except BaseException as error:
                report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        outside_cpu = outside_numpy = outside_python = None
        if c is not None:
            try:
                c.check()
                report['resource_observations'] = c.summary()
                observed = report['resource_observations']
                report['flags']['child_resources_within_limits'] = (observed['violations'] == []
                    and 0 <= observed['elapsed_seconds'] <= 1800 and observed['samples'] > 0
                    and 0 <= observed['peak_RSS_bytes'] <= 16*1024**3
                    and observed['minimum_available_RAM_bytes'] >= 16*1024**3)
            except BaseException as error:
                # Summary remains useful after a real limit refusal; do not hide the violation.
                try:
                    report['resource_observations'] = c.summary()
                except BaseException:
                    pass
                report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        body = report['body_report']
        report.update(successful_optimizer_updates=body.get('successful_optimizer_updates',0),
            optimizer_steps_entered=body.get('optimizer_steps_entered',0),new_training=body.get('new_training',False),
            diagnostic_verdict=body.get('verdict'),
            baseline_control_exact=all(value.get('baseline_control_exact') is True for value in body.get('arms',{}).values()) if len(body.get('arms',{}))==2 else None,
            candidate_within_prelocked_tolerances=(body.get('verdict')=='WITHIN_PRELOCKED_TOLERANCES') if body.get('verdict') in ('WITHIN_PRELOCKED_TOLERANCES','CANDIDATE_OUTSIDE_PRELOCKED_TOLERANCES') else None,
            cost=dict(elapsed_seconds=time.monotonic()-started))
        if report['status'] == 'COMPLETE' and (set(report['flags']) != SUCCESS_FLAGS
                or not all(value is True for value in report['flags'].values()) or report['cleanup_errors']
                or report['cost']['elapsed_seconds'] > 1800):
            report.update(status='FAILED',error=dict(type='RuntimeError',message='Final worker control/resource cleanup contract failed'))
        if out is not None and out.is_dir():
            with (out/'REPORT.json').open('x',encoding='utf-8') as stream:
                json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
                stream.write('\n')
        if report['status'] != 'COMPLETE' and original_error is None:
            original_error = RuntimeError('U52 worker failed; original partial report and returned updates preserved')
    if original_error is not None:
        raise original_error
    print('U52 fourteen disposable updates and controlled comparison complete: '+report['diagnostic_verdict']+'; pending Linux-owner/Root review, no AP or main-budget credit.', flush=True)


if __name__ == '__main__':
    main()
