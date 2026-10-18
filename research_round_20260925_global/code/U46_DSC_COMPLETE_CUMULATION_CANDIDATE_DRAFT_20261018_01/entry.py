"""Owned CPU entry for the U46 candidate coordinate probe; separate from U40 training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import warnings


SCHEMA = 'ROOT-U45-coordinate-diagnostic-spec-v1'
OPERATION = 'CPU_U45_DSC_COORDINATE_VJP_DIAGNOSTIC'
ROLES = ('entry_source', 'probe_source', 'dsconv_source')
VERSIONS = dict(torch='2.0.1+cu118', numpy='1.23.5')
LIMITS = dict(allowed_cpus=[0, 1], threads=2, maximum_RSS_bytes=16*1024**3,
    minimum_available_RAM_bytes=16*1024**3, deadline_seconds=120, CUDA_visible_devices='')
PINS = dict(probe_source=(8197, '1a18acb181b12dd67f955d0839775245a5cdc0501fe35fbca2871a907d715824'),
    dsconv_source=(15206, '321e9f23681d1da17ad2ecec008e4a58f64b66df4ad2bf78b68309a7fae820ea'))


def need(value, message):
    if not value:
        raise ValueError(message)


def reference(path, raw):
    return dict(path=str(Path(path)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def closed(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict admitted source reference required')
    path = Path(value['path'])
    need(reference(path, path.read_bytes()) == value, 'Admitted source bytes differ: '+str(path))
    return path


def load(path, alias):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    need(Path(module.__file__).resolve() == path.resolve(), 'Actual loaded source path differs')
    return module


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args(argv)
    out = None; output_created = False; sampler = None; stopped = threading.Event(); process = None
    aliases = {}; diagnostic = {}; torch = None; original_error = None
    observed = dict(peak_RSS_bytes=0, minimum_available_RAM_bytes=None, samples=0,
                    violations=[], sampling_errors=[])
    flags = dict(CPU_only=False, source_bytes_verified=False, CUDA_uninitialized_before=False,
                 CUDA_uninitialized_after=False, caller_controls_restored=False, module_aliases_restored=False)
    report = dict(schema='U45-coordinate-vjp-diagnostic-report-v1', status='STARTED',
        operation=OPERATION, qualified=False, diagnostic=diagnostic, flags=flags,
        actual_versions={}, library_paths={}, full_binary_closure_claimed=False,
        resource_observations=observed, new_training=False, AP=False)
    try:
        need(not any(name.split('.')[0] in ('torch', 'numpy') for name in sys.modules), 'No ML import before source/environment closure')
        spec_path = Path(args.spec)
        need(spec_path.is_absolute(), 'Absolute original-driver-bound spec required')
        raw = spec_path.read_bytes(); spec = json.loads(raw)
        need(spec['schema'] == SCHEMA and spec['operation'] == OPERATION
             and type(spec['phase']) is str and spec['phase']
             and spec['expected_versions'] == VERSIONS and spec['limits'] == LIMITS, 'Exact CPU-only diagnostic contract')
        report.update(phase=spec['phase'], spec_ref=reference(spec_path, raw),
                      source_roles=spec['source_roles'], limits=spec['limits'])
        out = Path(spec['output_directory'])
        need(out.is_absolute() and out.drive.casefold() == out.resolve().drive.casefold() == 'd:' and not out.exists()
             and not out.is_symlink(), 'Fresh absolute D diagnostic-output directory required')
        out.mkdir(parents=True, exist_ok=False)
        output_created = True
        need(type(spec['source_roles']) is dict and set(spec['source_roles']) == set(ROLES)
             and type(spec['sources']) is list, 'Exactly three explicit executable-source roles')
        source_map = {}
        for value in spec['sources']:
            path = closed(value); key = str(path.resolve()).casefold()
            need(key not in source_map, 'Unique admitted source paths required')
            source_map[key] = value
        for role in ROLES:
            value = spec['source_roles'][role]
            need(source_map.get(str(Path(value['path']).resolve()).casefold()) == value, 'Every executing source must be admitted')
            if role in PINS:
                need((value['bytes'], value['sha256']) == PINS[role], 'Exact U46 numerical source pin: '+role)
        need(Path(__file__).resolve() == Path(spec['source_roles']['entry_source']['path']).resolve(), 'Entry self-source binding differs')
        flags['source_bytes_verified'] = True
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Original driver must hide CUDA before any ML import')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            need(os.environ.get(name) == '2', 'Original driver must set '+name+'=2 before import')
        import psutil
        process = psutil.Process()
        need(process.cpu_affinity() == [0, 1], 'Original driver must pin CPU0/1 before import')
        report['library_paths']['psutil'] = str(Path(psutil.__file__).resolve())

        def sample():
            try:
                rss, available = process.memory_info().rss, psutil.virtual_memory().available
                observed['peak_RSS_bytes'] = max(observed['peak_RSS_bytes'], int(rss))
                previous = observed['minimum_available_RAM_bytes']
                observed['minimum_available_RAM_bytes'] = int(available) if previous is None else min(previous, int(available))
                observed['samples'] += 1
                if rss > LIMITS['maximum_RSS_bytes'] or available < LIMITS['minimum_available_RAM_bytes']:
                    observed['violations'].append('RSS/free-RAM resource boundary')
            except BaseException as error:
                observed['sampling_errors'].append(type(error).__name__+': '+str(error))

        def check():
            sample()
            need(not observed['violations'] and not observed['sampling_errors'], 'CPU resource observation failed')
            need(time.monotonic()-started <= LIMITS['deadline_seconds'], 'Diagnostic deadline exceeded')

        def monitor():
            while not stopped.wait(.02):
                sample()

        check(); sampler = threading.Thread(target=monitor, daemon=True); sampler.start()
        import torch as imported_torch
        import numpy as np
        torch = imported_torch
        report['actual_versions'] = dict(torch=str(torch.__version__), numpy=str(np.__version__))
        report['library_paths'].update(torch=str(Path(torch.__file__).resolve()), numpy=str(Path(np.__file__).resolve()))
        need(report['actual_versions'] == VERSIONS and not torch.cuda.is_initialized(), 'Exact installed versions and cold CUDA required')
        flags['CUDA_uninitialized_before'] = True
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        need(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Exactly two Torch threads')
        check()
        for alias in ('_U46_CANDIDATE_DSC', '_U46_COORDINATE_PROBE'):
            aliases[alias] = (alias in sys.modules, sys.modules.get(alias))
        with warnings.catch_warnings():
            dsc = load(Path(spec['source_roles']['dsconv_source']['path']), '_U46_CANDIDATE_DSC')
            probe = load(Path(spec['source_roles']['probe_source']['path']), '_U46_COORDINATE_PROBE')
            probe.run_diagnostic(torch, dsc.DSC, diagnostic)
        need(diagnostic['numerical_execution_completed'] is True and diagnostic['autograd_calls'] == 2
             and diagnostic['coordinate_map_calls'] == 74 and set(diagnostic['cases']) == {'0', '1'}
             and all(len(row['channels']) == 18 for row in diagnostic['cases'].values()), 'Complete candidate coordinate diagnostic required')
        flags['caller_controls_restored'] = diagnostic['caller_controls_restored'] is True
        need(flags['caller_controls_restored'] and not torch.cuda.is_initialized(), 'Probe controls restored and CUDA remained cold')
        check(); report['status'] = 'COMPLETE'
    except BaseException as error:
        original_error = error
        report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        stopped.set()
        if sampler is not None:
            try:
                sampler.join(timeout=1)
                if sampler.is_alive():
                    observed['sampling_errors'].append('Sampler did not terminate')
            except BaseException as cleanup_error:
                observed['sampling_errors'].append(type(cleanup_error).__name__+': '+str(cleanup_error))
        for alias, (existed, original) in aliases.items():
            if existed:
                sys.modules[alias] = original
            else:
                sys.modules.pop(alias, None)
        flags['module_aliases_restored'] = all((alias in sys.modules) == existed
            and (not existed or sys.modules[alias] is original) for alias, (existed, original) in aliases.items())
        flags['CUDA_uninitialized_after'] = torch is not None and not torch.cuda.is_initialized()
        flags['CPU_only'] = flags['CUDA_uninitialized_before'] and flags['CUDA_uninitialized_after']
        report['cost'] = dict(elapsed_seconds=time.monotonic()-started, peak_RSS_bytes=observed['peak_RSS_bytes'])
        report['RSS_peak_is_sampled'] = True
        final_bad = report['status'] == 'COMPLETE' and (not flags['CPU_only'] or not flags['module_aliases_restored']
            or bool(observed['sampling_errors']) or bool(observed['violations']))
        if final_bad:
            report.update(status='FAILED', error=dict(type='RuntimeError', message='Final cold-CUDA/source/resource contract failed'))
        try:
            if output_created and out.is_dir():
                with (out/'REPORT.json').open('x', encoding='utf-8') as stream:
                    json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        except BaseException:
            if original_error is None:
                raise
        if final_bad and original_error is None:
            raise RuntimeError('Final cold-CUDA/source/resource contract failed')


if __name__ == '__main__':
    main()
