"""UNBOUND Linux U50 worker; authoring/importing this module executes no ML."""
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


SCHEMA = 'ROOT-U50-DGX-paired-pilot-spec-v1'
OPERATION = 'GPU_U50_DGX_PAIRED_DISPOSABLE_PILOT'
PILOT_SHA256 = '003641b7c81e198bb70502073aba22f623f9dac82554ca9ce90ba059ac20625b'
SOURCE_PINS = {
    'shared_initialization': '4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
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
}
LOAD_ORDER = (
    ('full_dsconv', 'dsconv'), ('full_head', 'head'), ('full_legacy', 'model_full'),
    ('full_feature', 'feature_backbone'), ('plain_legacy', 'plain_model1'),
    ('plain_feature', 'plain_feature'), ('instance_adapter', 'full_instance_adapter'),
    ('image_geometry', 'image_geometry'), ('shared_initialization', 'shared_initialization'),
    ('u36_helpers', '_U50_U36_HELPERS'), ('u39_training_ops', '_U50_U39_TRAINING_OPS'),
)
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
    """Pure metadata closure used before loading even the stdlib-only pilot/owner."""
    required = {'schema','operation','phase','entry_source','pilot_source','owner_source','source_refs',
        'library_sources','portable_refs','prelock','output_directory','owner_output_directory','sources'}
    need(type(spec) is dict and set(spec) == required
         and spec['schema'] == SCHEMA and spec['operation'] == OPERATION
         and type(spec['phase']) is str and spec['phase'] and '/' not in spec['phase'], 'Exact independent U50 worker spec')
    need(type(spec['source_refs']) is dict and set(spec['source_refs']) == set(SOURCE_PINS)
         and type(spec['library_sources']) is dict and set(spec['library_sources']) == {'psutil','torch','numpy'}
         and type(spec['portable_refs']) is dict and set(spec['portable_refs']) == {'54','710'},
         'Exactly eleven code roles, three selected libraries and two matched portable inputs')
    for role, sha in SOURCE_PINS.items():
        need(strict_ref(spec['source_refs'][role])['sha256'] == sha, 'Exact original numeric source: '+role)
    need(strict_ref(spec['pilot_source'])['sha256'] == PILOT_SHA256 and spec['pilot_source']['bytes'] == 20338,
         'Original unchanged U50 body')
    values = [spec[k] for k in ('entry_source','pilot_source','owner_source')]
    values += list(spec['source_refs'].values())+list(spec['library_sources'].values())+list(spec['portable_refs'].values())
    expected = {}
    for value in values:
        strict_ref(value)
        need(value['path'] not in expected, 'Nineteen distinct actual local files')
        expected[value['path']] = value
    need(type(spec['sources']) is list and spec['sources'] == sorted(values, key=lambda value:value['path']),
         'Exact sorted Source closure, with no additional Windows inputs or hidden data')
    for key in ('output_directory','owner_output_directory'):
        path = spec[key]
        need(type(path) is str and PurePosixPath(path).is_absolute() and path != '/'
             and str(PurePosixPath(path)) == path and '..' not in PurePosixPath(path).parts, 'Explicit absolute POSIX output path')
    worker, owner = (PurePosixPath(spec[key]) for key in ('output_directory','owner_output_directory'))
    need(worker != owner and worker not in owner.parents and owner not in worker.parents,
         'Separate worker and original Linux-owner output directories')
    for value in expected:
        need(worker != PurePosixPath(value) and worker not in PurePosixPath(value).parents,
             'Worker output cannot contain admitted code, library or input files')
    return expected


def validate_spec(spec, prelock_validator):
    sources = source_table(spec)
    prelock_validator(spec['prelock'])  # Actual source-closed original pure contract, not a new recipe.
    return sources


def completed_pilot(report):
    need(report.get('status') == 'COMPLETED_PENDING_ROOT_REVIEW'
         and report['new_training'] is report['pilot_only'] is True
         and report['formal_training_budget_credit'] == 0
         and all(report[key] is False for key in ('main_training','U40_frontier_modified','weights_written','qualified','AP','DEV','TEST'))
         and report['successful_optimizer_updates'] == report['optimizer_steps_entered'] == 6
         and report['distinct_images'] == 2 and report['distinct_matched_GT'] == 28
         and report['GT_visits_per_arm'] == 29 and report['total_GT_visits'] == 58
         and report['image_order'] == [54,710,54]
         and report['caller_CPU_RNG_preserved'] is report['postinitialization_CUDA_RNG_preserved'] is True
         and report['final_CUDA_cleanup'] == dict(allocated_bytes=0,reserved_bytes=0)
         and set(report['arms']) == {'PLAIN','FULL'}, 'Actual complete paired pilot, six updates and hard-zero GPU cleanup')
    for arm in ('PLAIN','FULL'):
        record = report['arms'][arm]
        need(record['status'] == 'PASS' and record['updates_completed'] == 3 and len(record['updates']) == 3
             and [row['dataset_index'] for row in record['updates']] == [54,710,54]
             and [row['G'] for row in record['updates']] == [1,27,1]
             and sum(row['all_matched_GT'] for row in record['updates']) == 29
             and all(row['status'] == 'PASS' and row['optimizer_step_returned'] is True for row in record['updates'])
             and record['after_arm_release'] == dict(CUDA_allocated_bytes=0,CUDA_reserved_bytes=0),
             'Both original arms completed all GT in order and released all GPU references')


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
    out = c = torch = np = psutil = pilot = owner = modules = None
    outside_cpu = outside_numpy = outside_python = outside_math = outside_dtype = outside_grad = None
    aliases, cache = {}, {}
    original_error = None
    report = dict(schema='U50-DGX-paired-pilot-worker-report-v1', status='STARTED', operation=OPERATION,
        qualified=False, pilot_only=True, main_training=False, formal_training_budget_credit=0,
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
            source_refs=spec['source_refs'], portable_refs=spec['portable_refs'], prelock=spec['prelock'],
            actual_library_paths={})
        for key in ('entry_source','pilot_source','owner_source','library_sources'):
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
        pilot = load_closed(close, spec['pilot_source'], '_U50_DGX_PILOT_BODY', aliases)
        validate_spec(spec, pilot.validate_prelock)
        owner = load_closed(close, spec['owner_source'], '_U50_LINUX_OWNER', aliases)
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
        modules = {}
        for role, alias in LOAD_ORDER:
            modules[role] = load_closed(c.closed, spec['source_refs'][role], alias, aliases)
            need(not torch.cuda.is_initialized(), 'All source-closed module definitions load without CUDA initialization')
            c.check()
        need(modules['full_legacy'].DSConv is modules['full_dsconv'].DSConv
             and modules['full_legacy'].CoordConv is modules['full_head'].CoordConv, 'Exact original import dependency objects')
        print('U50 independent DGX paired pilot starts: two images; three disposable updates per arm.', flush=True)
        result = pilot.run_pilot(torch,np,c,portable_refs={int(key):value for key,value in spec['portable_refs'].items()},
            modules=modules,source_refs=spec['source_refs'],prelock=spec['prelock'],pilot_source_ref=spec['pilot_source'],report=report['body_report'])
        need(result is report['body_report'], 'Preserve original complete/partial report object')
        completed_pilot(result)
        report['gpu_math_settings_after_body'] = math_settings(torch)
        need(report['gpu_math_settings_after_body'] == GPU_MATH, 'Numerical backend settings retained throughout pilot')
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
        modules = pilot = owner = None
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
            original_error = RuntimeError('U50 worker failed; original partial report and returned updates preserved')
    if original_error is not None:
        raise original_error
    print('U50 six disposable updates completed; pending original Linux-owner/Root terminal review; no AP.', flush=True)


if __name__ == '__main__':
    main()
