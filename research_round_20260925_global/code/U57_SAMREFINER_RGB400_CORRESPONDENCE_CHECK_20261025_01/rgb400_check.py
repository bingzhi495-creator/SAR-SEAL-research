"""Unexecuted CPU-only first-image RGB400 correspondence check."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main(argv=None):
    import resource
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha', required=True)
    args = parser.parse_args(argv)
    raw = Path(args.spec).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.spec_sha
    spec = json.loads(raw)
    if sys.flags.optimize:
        raise RuntimeError('Root must run with assertions enabled')
    assert spec['operation'] == 'CPU_FIRST_ORIGINAL_RGB400_CACHE_CORRESPONDENCE'
    assert spec['ordinal'] == 0 and spec['image_id'] == 4
    assert spec['external_wall_timeout_seconds'] == 90 and spec['single_attempt'] is True
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not any(name.split('.')[0] in ('torch', 'numpy') for name in sys.modules)
    os.sched_setaffinity(0, {0, 1})
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
    out = Path(spec['output_directory'])
    assert str(out.resolve()) == str(out)
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    report = dict(schema='SAMRefiner-first-image-RGB400-correspondence-report-v1',
                  status='STARTED', comparison_performed=False, ordinal=0, image_id=4,
                  AP=False, GT_accessed=False, SAM_model_loaded=False, training_updates=0,
                  full187=False, original_full_resolution_historical_pixel_equality_claimed=False,
                  external_wall_timeout_seconds=90, external_group_acceptance_required=True,
                  continuous_RAM_guarantee=False, stages=[])
    stat = Path('/proc/self/stat').read_text()
    report.update(pid=os.getpid(), pgid=os.getpgrp(),
                  start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    c = None
    item = reader = rebuilt = cached = batch = rgb = d = None
    aliases = {}
    code = 1

    def stage(name):
        report['stage'] = name
        report['stages'].append(dict(name=name, elapsed_seconds=time.monotonic()-started))
        temporary = out/'REPORT.next.json'
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, out/'REPORT.json')

    try:
        stage('AUTHENTICATING_REUSED_HELPERS')
        ref = spec['first_input_worker_source']
        helper_path = Path(ref['path'])
        helper_bytes = helper_path.read_bytes()
        assert str(helper_path.resolve()) == ref['path'] and len(helper_bytes) == ref['bytes']
        assert hashlib.sha256(helper_bytes).hexdigest() == ref['sha256']
        helpers = runpy.run_path(str(helper_path), run_name='_RGB400_FIRST_HELPERS')
        owner = helpers['load_source'](spec['owner_source'], '_RGB400_CPU_CONTROL')
        import psutil
        c = owner.make_control(spec, psutil)
        assert owner.LIMITS == dict(allowed_cpus=[0, 1], threads=2,
            maximum_RSS_bytes=16*1024**3, minimum_available_RAM_bytes=16*1024**3,
            deadline_seconds=1800)
        refs = spec['source_refs']
        assert set(refs) == {'u65_entry', 'u65_body', 'portable_reader',
                             'dev_reader', 'dev_rgb_reader', 'image_geometry'}
        assert set(spec['library_sources']) == {'psutil', 'torch', 'numpy', 'PIL'}
        values = [spec[key] for key in ('worker_source', 'owner_source', 'first_input_worker_source',
                  'manifest_ref', 'package0_ref', 'png_ref', 'first_report_ref')]
        values += list(refs.values()) + list(spec['library_sources'].values())
        closure = {}
        for value in values:
            assert value['path'] not in closure or closure[value['path']] == value
            closure[value['path']] = value
        assert spec['sources'] == [closure[key] for key in sorted(closure)]
        for value in spec['sources']:
            c.closed(value)
        assert Path(__file__).resolve() == c.closed(spec['worker_source'])
        stage('ADMITTED_SOURCE_AND_SINGLE_PACKAGE_BYTES_CLOSED')
        import torch
        import numpy as np
        import PIL
        from PIL import Image
        for name, module in (('psutil', psutil), ('torch', torch), ('numpy', np), ('PIL', PIL)):
            assert helpers['reference'](Path(module.__file__).resolve()) == spec['library_sources'][name]
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 2
        assert not torch.cuda.is_initialized()
        entry = helpers['load_source'](refs['u65_entry'], '_RGB400_U65_ENTRY')
        def source(role, alias):
            return entry.load_closed(c.closed, refs[role], alias, aliases)
        body = source('u65_body', '_RGB400_U65_BODY')
        reader_module = source('portable_reader', '_RGB400_READER')
        geometry = source('image_geometry', '_RGB400_GEOMETRY')
        manifest = json.loads(c.closed(spec['manifest_ref']).read_bytes())
        assert manifest['entries'][0]['package_ref'] == spec['package0_ref']
        reader = reader_module.prepare(torch, c, manifest['entries'], image_geometry=geometry,
            source_refs={key: refs[key] for key in ('dev_reader', 'dev_rgb_reader', 'image_geometry')})
        item, row, originals = reader.load(0)
        body.validate_loaded_metadata(0, item, row, originals, manifest['entries'][0])
        assert row['dataset_index'] == 0 and row['image_id'] == 4
        first = json.loads(c.closed(spec['first_report_ref']).read_bytes())
        assert first['status'] == 'FIRST_ORIGINAL_RGB_AND_NATIVE100_INPUT_CHECK_PASSED'
        assert first['ordinal'] == 0 and first['image_id'] == 4 and first['PNG_ref'] == spec['png_ref']
        shape = item['identity']['original_shape']
        assert shape == first['original_shape'] == [317, 420]
        stage('U59_ORDINAL0_AND_FIRST_PNG_IDENTITY_JOINED')
        with Image.open(c.closed(spec['png_ref'])) as image:
            assert image.format == 'PNG' and image.size == (shape[1], shape[0])
            with image.convert('RGB') as converted:
                rgb = np.array(converted, dtype=np.uint8, copy=True)
        assert rgb.dtype == np.uint8 and rgb.shape == (*shape, 3)
        rgb_digest = hashlib.sha256(rgb.tobytes(order='C')).hexdigest()
        assert rgb_digest == first['original_RGB_SHA256']
        c.check()
        stage('CURRENT_ORIGINAL_RGB_BYTES_VERIFIED')
        with torch.inference_mode():
            batch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
            rebuilt = item['geometry'].prepare_rgb(batch)
            cached = item['rgb400']
            assert all(value.device.type == 'cpu' and value.dtype == torch.float32
                       and tuple(value.shape) == (1, 3, 400, 400)
                       and bool(torch.isfinite(value).all()) for value in (rebuilt, cached))
            rebuilt_bytes = rebuilt.detach().contiguous().numpy().tobytes(order='C')
            cached_bytes = cached.detach().contiguous().numpy().tobytes(order='C')
            d = rebuilt.to(dtype=torch.float64) - cached.to(dtype=torch.float64)
            exact = bool(torch.equal(rebuilt, cached))
            close = bool(torch.allclose(rebuilt, cached, atol=1e-7, rtol=0))
            report.update(comparison_performed=True, torch_equal=exact,
                raw_FP32_bytes_equal=rebuilt_bytes == cached_bytes,
                numerical_mismatch=not exact, byte_mismatch=rebuilt_bytes != cached_bytes,
                allclose_atol_1e_7_rtol_0=close, tolerance_is_descriptive_not_quality_gate=True,
                signed_min=float(d.min()), signed_max=float(d.max()), signed_mean=float(d.mean()),
                max_abs=float(d.abs().max()), mean_abs=float(d.abs().mean()),
                RMSE=float(d.square().mean().sqrt()), nonzero_count=int(torch.count_nonzero(d)),
                rebuilt_raw_FP32_SHA256=hashlib.sha256(rebuilt_bytes).hexdigest(),
                cached_raw_FP32_SHA256=hashlib.sha256(cached_bytes).hexdigest(),
                raw_array_layout='contiguous C-order [1,3,400,400] float32, no .pt/.npy headers',
                statistics_accumulation='cast each input separately to float64 before subtraction',
                original_RGB_SHA256=rgb_digest, original_shape=shape,
                torch_version=str(torch.__version__), numpy_version=np.__version__, Pillow_version=PIL.__version__)
        c.check()
        assert not torch.cuda.is_initialized()
        report['status'] = 'COMPARISON_COMPLETED_PENDING_ROOT_REVIEW'
        report['CUDA_initialized'] = False
        stage('COMPARISON_COMPLETED_EQUAL_OR_MISMATCH_REPORTED')
        code = 0
    except BaseException as error:
        report.update(status='FAILED_INPUT_ABI_OR_RESOURCE_PRESERVED',
                      error_type=type(error).__name__, error=str(error))
    finally:
        item = reader = rebuilt = cached = batch = rgb = d = None
        gc.collect()
        if c is not None:
            try:
                c.check()
            except BaseException as error:
                report.update(status='FINAL_RESOURCE_FAILURE_PRESERVED', final_resource_error=str(error))
                code = 1
            report['sampled_control'] = c.summary()
        for alias, (existed, previous) in aliases.items():
            if existed: sys.modules[alias] = previous
            else: sys.modules.pop(alias, None)
        report.update(elapsed_seconds=time.monotonic()-started, worker_exit_code=code,
                      peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        stage('WORKER_RETURNING')
    return code


if __name__ == '__main__':
    raise SystemExit(main())

