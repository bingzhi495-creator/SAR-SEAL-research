"""UNEXECUTED U60 CPU-only independent SMOKE2 restoration entry draft."""
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time
import traceback
import types

SCHEMA = 'ROOT-U60-DGX-SMOKE2-CPU-restore-spec-v1'
OPERATION = 'CPU_U60_DGX_INDEPENDENT_SMOKE2_RESTORE'
HELPER_SHA256 = '0943096b6d9787bdb370212c8c311427c37f5ef1eb4c32ab397c1ea35faf3d14'
ENVIRONMENT = dict(python='3.12.14', torch='2.10.0+cu130', numpy='1.26.4', system='Linux', machine='aarch64')
LIMITS = dict(allowed_cpus=[0, 1], threads=2, maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800)
RESTORE_PINS = dict(
    training_state='d80ab26ad7092431f4001535705e7afcc4fda7cbf66bacb4dc2cb66edd3d6724',
    training_entry='cd62a640e63210bcb63e58e09bb1e429f1a68ea66dcfec240831a5f1db31aca9',
    shared_initialization='4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    full_legacy='c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    full_dsconv='b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    full_head='6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    full_feature='f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    plain_legacy='902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    plain_feature='d93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    instance_adapter='9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    u36_helpers='1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    u39_training_ops='378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d')
AUXILIARY_PINS = dict(
    image_geometry='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    portable_reader='b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557')
LOAD_ORDER = (
    ('full_dsconv', 'dsconv'), ('full_head', 'head'), ('full_legacy', 'model_full'),
    ('full_feature', 'feature_backbone'), ('plain_legacy', 'plain_model1'),
    ('plain_feature', 'plain_feature'), ('instance_adapter', 'full_instance_adapter'),
    ('image_geometry', 'image_geometry'), ('shared_initialization', 'shared_initialization'),
    ('u36_helpers', '_U57_U36_HELPERS'), ('u39_training_ops', '_U57_U39_TRAINING_OPS'),
    ('portable_reader', '_U57_PORTABLE_READER'),
    ('training_state', '_U60_ORIGINAL_TRAINING_STATE'), ('training_entry', '_U60_ORIGINAL_TRAINING_ENTRY'))
FALSE_FIELDS = ('qualified', 'numerical_runtime_qualified', 'AP', 'DEV', 'TEST', 'new_training',
                'main_training', 'pilot_only', 'U40_frontier_modified', 'innovation_established', 'speedup_established')
SUCCESS_FLAGS = {'source_bytes_verified', 'CUDA_uninitialized_before', 'CUDA_uninitialized_after',
    'source_unchanged_after', 'module_aliases_restored', 'caller_CPU_RNG_restored',
    'caller_NumPy_RNG_restored', 'caller_Python_RNG_restored', 'dtype_grad_threads_preserved',
    'child_resources_within_limits', 'both_adapters_released'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def validate_environment(environment):
    need(environment.get('CUDA_VISIBLE_DEVICES') == '', 'Owner must hide CUDA')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        need(environment.get(name) == '2', 'Owner must set exactly two threads: ' + name)


def prelock():
    return dict(schema='U60-DGX-SMOKE2-CPU-restore-prelock-v1', environment=deepcopy(ENVIRONMENT),
        limits=deepcopy(LIMITS), CUDA_visible_devices='', mode='SMOKE', endpoint_updates=2,
        no_new_training=True, no_forward=True, no_FIT=True, no_DEV=True, no_TEST=True, no_AP=True)


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and '\\' not in value['path']
         and PurePosixPath(value['path']).is_absolute() and str(PurePosixPath(value['path'])) == value['path']
         and '..' not in PurePosixPath(value['path']).parts and type(value['bytes']) is int and value['bytes'] > 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Exact positive canonical POSIX file reference')
    return value


def source_table(spec):
    keys = {'schema', 'operation', 'phase', 'entry_source', 'owner_source', 'restore_helper_source',
            'restore_sources', 'auxiliary_sources', 'library_sources', 'endpoint_refs', 'checkpoint_refs',
            'prelock', 'output_directory', 'owner_output_directory', 'sources'}
    need(type(spec) is dict and set(spec) == keys and spec['schema'] == SCHEMA and spec['operation'] == OPERATION
         and type(spec['phase']) is str and 1 <= len(spec['phase']) <= 160
         and '/' not in spec['phase'] and '\\' not in spec['phase'], 'Exact dedicated U60 spec')
    need(type(spec['prelock']) is dict and json.dumps(spec['prelock'], sort_keys=True, allow_nan=False)
         == json.dumps(prelock(), sort_keys=True, allow_nan=False), 'Exact typed CPU SMOKE2 prelock and unchanged limits')
    values = [spec[k] for k in ('entry_source', 'owner_source', 'restore_helper_source')]
    need(strict_ref(spec['restore_helper_source'])['sha256'] == HELPER_SHA256, 'Exact reviewed U58 helper')
    for key, pins in (('restore_sources', RESTORE_PINS), ('auxiliary_sources', AUXILIARY_PINS)):
        need(type(spec[key]) is dict and set(spec[key]) == set(pins), 'Exact original code roles: ' + key)
        for role, pin in pins.items():
            need(strict_ref(spec[key][role])['sha256'] == pin, 'Exact original code pin: ' + role)
        values += list(spec[key].values())
    for key, roles in (('library_sources', {'psutil', 'torch', 'numpy'}),
                       ('endpoint_refs', {'qualified', 'spec', 'pair_commit', 'order'}),
                       ('checkpoint_refs', {'PLAIN', 'FULL'})):
        need(type(spec[key]) is dict and set(spec[key]) == roles, 'Complete explicit role set: ' + key)
        values += list(spec[key].values())
    admitted = {}
    for value in values:
        strict_ref(value)
        need(value['path'] not in admitted, 'Distinct direct input/source roles, no aliasing')
        admitted[value['path']] = value
    need(spec['sources'] == [admitted[k] for k in sorted(admitted)], 'Only exact direct restoration Source closure')
    outputs = []
    for key in ('output_directory', 'owner_output_directory'):
        value = spec[key]
        need(type(value) is str and '\\' not in value and PurePosixPath(value).is_absolute()
             and str(PurePosixPath(value)) == value and value != '/' and '..' not in PurePosixPath(value).parts,
             'Canonical explicit output directory')
        outputs.append(PurePosixPath(value))
    need(outputs[0] != outputs[1] and outputs[0] not in outputs[1].parents
         and outputs[1] not in outputs[0].parents, 'Separate owner and worker outputs')
    for value in admitted:
        path = PurePosixPath(value)
        need(all(out != path and out not in path.parents for out in outputs), 'Outputs cannot contain admitted sources')
    return admitted


def restore_pair(torch, np, control, helper, modules, source_refs, metadata, report):
    """Outer wiring, pending evidence only. CLI closes actual library/module bytes.

    Injectable solely for stdlib wiring tests; this function alone cannot prove
    actual model construction. No caller-supplied initialization receipt exists.
    """
    need(metadata['spec']['prelock']['mode'] == 'SMOKE', 'Independent SMOKE only; never any FORMAL cursor')
    need(type(report) is dict and not report, 'Fresh caller-owned partial report')
    report.update(schema='U60-SMOKE2-CPU-pair-restore-body-v1', status='STARTED',
                  qualified=False, numerical_runtime_qualified=False, AP=False, DEV=False, TEST=False,
                  new_training=False, main_training=False, formal_training_budget_credit=0,
                  arms={}, constructor_executed=False, model_forward_executed=False)
    adapters = receipt = None
    try:
        endpoint = helper.validate_smoke_endpoint(modules['training_state'], modules['training_entry'], metadata)
        need(endpoint['mode'] == 'SMOKE' and endpoint['cursor'] == dict(
            global_updates=2, completed_epochs=0, within_epoch=2), 'Accepted independent SMOKE2 endpoint')
        control.check()
        need(not torch.cuda.is_initialized(), 'Cold CUDA before original CPU constructor')
        initializer = modules['shared_initialization']
        adapters, receipt = initializer.build_pair(torch,
            {role: modules[role] for role in initializer.SOURCE_SHA256},
            {role: source_refs[role] for role in initializer.SOURCE_SHA256})
        need(type(adapters) is dict and set(adapters) == {'PLAIN', 'FULL'}
             and adapters['PLAIN'] is not adapters['FULL'], 'Original two distinct arm objects')
        need(not torch.cuda.is_initialized(), 'Constructor must leave CUDA cold')
        report.update(constructor_executed=True, constructor_role='shared_initialization.build_pair',
                      fresh_constructor_receipt=receipt, cursor=endpoint['cursor'])
        for arm in ('PLAIN', 'FULL'):
            control.check()
            partial = report['arms'][arm] = {}
            returned = helper.restore_smoke_arm(torch, np, control, adapters[arm], arm, metadata, receipt,
                                               modules=modules, source_refs=source_refs, report=partial)
            need(returned is partial and partial['status'] == 'RESTORED_CPU_ONLY'
                 and partial['mode'] == 'SMOKE' and partial['formal_endpoint'] is False
                 and partial['model_forward_executed'] is False, 'Exact successful SMOKE CPU restore: ' + arm)
        need(not torch.cuda.is_initialized(), 'Both actual restorations leave CUDA cold')
        control.check()
        report['status'] = 'COMPLETE_PENDING_ROOT'
    except BaseException:
        report['status'] = 'FAILED'
        raise
    finally:
        adapters = receipt = None
    return report


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
    need(Path(module.__file__).resolve() == path.resolve(), 'Actual source-backed module: ' + alias)
    return module


def completed_report(report, spec, spec_ref):
    """Bounded JSON-only owner check; never load CP or replay the restoration."""
    need(report['schema'] == 'U60-SMOKE2-CPU-restore-worker-report-v1'
         and report['status'] == 'COMPLETE' and report['operation'] == OPERATION
         and report['phase'] == spec['phase'] and report['spec_ref'] == spec_ref,
         'Exact complete worker identity')
    for key in ('entry_source', 'owner_source', 'restore_helper_source', 'restore_sources',
                'auxiliary_sources', 'library_sources', 'endpoint_refs', 'checkpoint_refs', 'prelock'):
        need(report[key] == spec[key], 'Exact original restoration report input: ' + key)
    need(type(report['flags']) is dict and set(report['flags']) == SUCCESS_FLAGS
         and all(report['flags'][key] is True for key in SUCCESS_FLAGS)
         and report['cleanup_errors'] == [] and all(report[k] is False for k in FALSE_FIELDS)
         and type(report['formal_training_budget_credit']) is int and report['formal_training_budget_credit'] == 0,
         'No failed cleanup, self-qualification, training credit or evaluation claim')
    body = report['body_report']
    need(body['schema'] == 'U60-SMOKE2-CPU-pair-restore-body-v1' and body['status'] == 'COMPLETE_PENDING_ROOT'
         and body['cursor'] == dict(global_updates=2, completed_epochs=0, within_epoch=2)
         and body['constructor_executed'] is True and body['constructor_role'] == 'shared_initialization.build_pair'
         and report['actual_build_pair_source_verified'] is True
         and body['constructor_source'] == spec['restore_sources']['shared_initialization']
         and body['constructor_component_sources'] == {role: spec['restore_sources'][role] for role in (
             'full_legacy', 'full_dsconv', 'full_head', 'full_feature', 'plain_legacy', 'plain_feature', 'instance_adapter')},
         'Actual closed original constructor invocation, never receipt alone')
    need(set(body['arms']) == {'PLAIN', 'FULL'} and body['new_training'] is False
         and body['qualified'] is False and body['numerical_runtime_qualified'] is False
         and all(body[key] is False for key in ('AP', 'DEV', 'TEST', 'main_training'))
         and body['formal_training_budget_credit'] == 0 and body['model_forward_executed'] is False,
         'Two technical smoke arms and no model forward or scientific result')
    for arm, result in body['arms'].items():
        need(result['status'] == 'RESTORED_CPU_ONLY' and result['arm'] == arm and result['mode'] == 'SMOKE'
             and result['formal_endpoint'] is False and result['checkpoint_ref'] == spec['checkpoint_refs'][arm]
             and result['cursor'] == body['cursor'] and result['restore_sources'] == spec['restore_sources']
             and result['endpoint_refs'] == spec['endpoint_refs'] and result['checkpoints_loaded'] == 1,
             'Complete exact independent SMOKE2 arm: ' + arm)
        for key in ('restored_model_state_exact', 'all_modules_eval', 'all_parameters_frozen',
                    'all_parameter_grads_none', 'external_CPU_RNG_unchanged', 'external_NumPy_RNG_unchanged',
                    'external_Python_RNG_unchanged', 'source_and_endpoint_bytes_rechecked',
                    'CPU_checkpoint_optimizer_payload_loaded', 'CPU_checkpoint_RNG_payload_loaded', 'CPU_checkpoint_payload_released'):
            need(result[key] is True, 'Actual complete restore/release flag: ' + key)
        for key in ('qualified', 'numerical_runtime_qualified', 'AP', 'DEV', 'TEST', 'new_training',
                    'optimizer_created', 'optimizer_restored', 'RNG_restored', 'CUDA_initialized', 'model_forward_executed',
                    'adapter_object_replaced', 'trained_head_reset', 'receipt_alone_proves_build_pair', 'population_qualification'):
            need(result[key] is False, 'No forbidden restoration claim or operation: ' + key)
    observed = report['resource_observations']
    need(observed['schema'] == 'U60-Linux-child-control-observations-v1'
         and type(observed['samples']) is int and observed['samples'] > 0 and observed['violations'] == []
         and type(observed['peak_RSS_bytes']) is int and 0 <= observed['peak_RSS_bytes'] <= LIMITS['maximum_RSS_bytes']
         and type(observed['minimum_available_RAM_bytes']) is int
         and observed['minimum_available_RAM_bytes'] >= LIMITS['minimum_available_RAM_bytes'], 'Actual bounded CPU child resources')
    for elapsed in (observed['elapsed_seconds'], report['cost']['elapsed_seconds']):
        need(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 1800, 'Actual bounded duration')
    need(report['actual_environment'] == ENVIRONMENT, 'Actual genuine DGX environment')
    return body


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha', required=True)
    args = parser.parse_args(argv)
    out = control = torch = np = modules = helper = owner = None
    outside_cpu = outside_numpy = outside_python = outside_dtype = outside_grad = None
    aliases = {}
    error_original = None
    created = False
    report = dict(schema='U60-SMOKE2-CPU-restore-worker-report-v1', status='STARTED', operation=OPERATION,
                  body_report={}, flags={}, cleanup_errors=[], formal_training_budget_credit=0,
                  actual_build_pair_source_verified=False, full_binary_closure_claimed=False,
                  Linux_original_terminal_qualification_claimed=False, **{k: False for k in FALSE_FIELDS})
    try:
        need(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Dedicated Linux/aarch64 CPU child only')
        need(not any(name.split('.')[0] in ('torch', 'numpy', 'PIL', 'pycocotools') for name in sys.modules),
             'No scientific imports before explicit admission')
        validate_environment(os.environ)
        path = Path(args.spec)
        need(path.is_absolute() and not path.is_symlink() and path.resolve(strict=True) == path
             and 0 < path.stat().st_size <= 2*1024**2, 'Small exact canonical Root-pinned spec')
        raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Actual owner-passed spec SHA')
        spec = json.loads(raw)
        source_table(spec)
        report.update(phase=spec['phase'], spec_ref=dict(path=str(path), bytes=len(raw), sha256=args.spec_sha))
        for key in ('entry_source', 'owner_source', 'restore_helper_source', 'restore_sources',
                    'auxiliary_sources', 'library_sources', 'endpoint_refs', 'checkpoint_refs', 'prelock'):
            report[key] = spec[key]
        # Bootstrap only the explicitly admitted stdlib owner, before scientific imports.
        owner_ref = spec['owner_source']
        owner_path = Path(owner_ref['path'])
        need(owner_path.resolve(strict=True) == owner_path and not owner_path.is_symlink()
             and reference(owner_path) == owner_ref, 'Actual admitted U60 owner bytes')
        owner = load_closed(lambda value: owner_path, owner_ref, '_U60_CPU_OWNER', aliases)
        # Close selected psutil bytes before its non-stdlib import as well.
        need(reference(Path(spec['library_sources']['psutil']['path'])) == spec['library_sources']['psutil'],
             'Selected psutil source bytes before import')
        psutil = importlib.import_module('psutil')
        need(str(psutil.__version__) == '6.1.1'
             and reference(Path(psutil.__file__).resolve()) == spec['library_sources']['psutil'], 'Genuine selected psutil6.1.1')
        control = owner.make_control(spec, psutil)
        control.check()
        for value in spec['sources']:
            control.closed(value)
        need(Path(__file__).resolve() == control.closed(spec['entry_source']), 'This exact admitted worker')
        report['flags']['source_bytes_verified'] = True
        out = Path(spec['output_directory'])
        need(out.parent.is_dir() and out.resolve() == out and not out.exists() and not out.is_symlink(), 'Fresh canonical worker output')
        out.mkdir()
        created = True
        imported = {'psutil': psutil}
        for name in ('torch', 'numpy'):
            control.closed(spec['library_sources'][name])
            imported[name] = importlib.import_module(name)
            need(type(imported[name]) is types.ModuleType and imported[name].__name__ == name
                 and reference(Path(imported[name].__file__).resolve()) == spec['library_sources'][name], 'Genuine admitted library: ' + name)
            control.check()
        torch, np = imported['torch'], imported['numpy']
        actual = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=str(np.__version__),
                      system=platform.system(), machine=platform.machine())
        need(actual == ENVIRONMENT and torch.get_default_dtype() == torch.float32
             and not torch.cuda.is_initialized(), 'Exact CPU FP32 runtime and CUDA cold')
        report['actual_environment'] = actual
        report['actual_library_paths'] = {key: str(Path(value.__file__).resolve()) for key, value in imported.items()}
        report['flags']['CUDA_uninitialized_before'] = True
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        need(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Actual two Torch threads')
        outside_cpu, outside_numpy, outside_python = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        outside_dtype, outside_grad = torch.get_default_dtype(), torch.is_grad_enabled()
        all_code = dict(spec['restore_sources'], **spec['auxiliary_sources'])
        loaded = {}
        for role, alias in LOAD_ORDER:
            loaded[role] = load_closed(control.closed, all_code[role], alias, aliases)
            need(not torch.cuda.is_initialized(), 'Original source definitions must leave CUDA cold')
            control.check()
        modules = {role: loaded[role] for role in RESTORE_PINS}
        need(modules['full_legacy'].DSConv is modules['full_dsconv'].DSConv
             and modules['full_legacy'].CoordConv is modules['full_head'].CoordConv, 'Original constructor dependency objects')
        helper = load_closed(control.closed, spec['restore_helper_source'], '_U60_U58_RESTORE', aliases)
        need(helper.RESTORE_PINS == RESTORE_PINS, 'Unchanged U58 exact restoration role pins')
        metadata = dict(refs=deepcopy(spec['endpoint_refs']))
        for key, value in spec['endpoint_refs'].items():
            metadata[key] = json.loads(control.closed(value).read_bytes())
        endpoint = helper.validate_smoke_endpoint(modules['training_state'], modules['training_entry'], metadata)
        helper.validate_restore_sources(endpoint, spec['restore_sources'])
        need(endpoint['checkpoint_refs'] == spec['checkpoint_refs']
             and endpoint['restore_source_refs'] == all_code, 'Actual accepted SMOKE2 two checkpoints and exact 14 code refs')
        initializer = modules['shared_initialization']
        constructor_roles = set(helper.CONSTRUCTOR_ROLES) - {'shared_initialization'}
        need(set(initializer.SOURCE_SHA256) == constructor_roles
             and initializer.SOURCE_SHA256 == {role: RESTORE_PINS[role] for role in constructor_roles},
             'Actual original constructor seven component pins')
        report['actual_build_pair_source_verified'] = True
        body = restore_pair(torch, np, control, helper, modules, spec['restore_sources'], metadata, report['body_report'])
        body['constructor_source'] = spec['restore_sources']['shared_initialization']
        body['constructor_component_sources'] = {role: spec['restore_sources'][role] for role in constructor_roles}
        for value in spec['sources']:
            control.closed(value)
        report['flags']['source_unchanged_after'] = True
        control.check()
        report['status'] = 'COMPLETE'
    except BaseException as error:
        error_original = error
        report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    finally:
        modules = helper = owner = None
        if 'loaded' in locals():
            loaded.clear()
        gc.collect()
        report['flags']['both_adapters_released'] = True
        for alias, (existed, previous) in reversed(list(aliases.items())):
            if existed:
                sys.modules[alias] = previous
            else:
                sys.modules.pop(alias, None)
        report['flags']['module_aliases_restored'] = all((alias in sys.modules) == existed
            and (not existed or sys.modules[alias] is previous) for alias, (existed, previous) in aliases.items())
        if torch is not None:
            report['flags']['CUDA_uninitialized_after'] = not torch.cuda.is_initialized()
        if torch is not None and outside_cpu is not None:
            try:
                torch.set_rng_state(outside_cpu)
                np.random.set_state(outside_numpy)
                random.setstate(outside_python)
                report['flags']['caller_CPU_RNG_restored'] = bool(torch.equal(outside_cpu, torch.get_rng_state()))
                current = np.random.get_state()
                report['flags']['caller_NumPy_RNG_restored'] = current[0] == outside_numpy[0] and bool(np.array_equal(current[1], outside_numpy[1])) and current[2:] == outside_numpy[2:]
                report['flags']['caller_Python_RNG_restored'] = random.getstate() == outside_python
                report['flags']['dtype_grad_threads_preserved'] = torch.get_default_dtype() == outside_dtype and torch.is_grad_enabled() == outside_grad and torch.get_num_threads() == torch.get_num_interop_threads() == 2
            except BaseException as error:
                report['cleanup_errors'].append(type(error).__name__ + ': ' + str(error))
        if control is not None:
            try:
                control.check()
                observed = control.summary()
                report['resource_observations'] = observed
                report['flags']['child_resources_within_limits'] = (observed['violations'] == [] and observed['samples'] > 0
                    and 0 <= observed['elapsed_seconds'] <= 1800 and observed['peak_RSS_bytes'] <= 16*1024**3
                    and observed['minimum_available_RAM_bytes'] >= 16*1024**3)
            except BaseException as error:
                report['resource_observations'] = control.summary()
                report['cleanup_errors'].append(type(error).__name__ + ': ' + str(error))
        report['cost'] = dict(elapsed_seconds=time.monotonic() - started)
        if report['status'] == 'COMPLETE':
            try:
                completed_report(report, spec, report['spec_ref'])
            except BaseException as error:
                report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error)))
        if created:
            with (out/'REPORT.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        if report['status'] != 'COMPLETE' and error_original is None:
            error_original = RuntimeError('U60 restore failed; preserve original partial report, never retry')
    if error_original is not None:
        raise error_original
    print('U60 independent SMOKE2 CPU restoration complete; pending actual owner/Root acceptance. No AP.', flush=True)


if __name__ == '__main__':
    main()
