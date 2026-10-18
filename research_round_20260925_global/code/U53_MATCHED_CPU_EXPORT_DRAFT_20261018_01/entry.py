"""UNBOUND owned CPU export worker; authoring/testing never calls main."""
import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import sys
import threading
import time
import traceback
import weakref


SCHEMA = 'U53-matched-CPU-export-spec-v1'
OPERATION = 'CPU_U53_MATCHED_TRAINING_INPUT_EXPORT'
ROLES = {'entry_source', 'export_source', 'rgb_reader_source', 'matched_reader_source', 'geometry_source'}
LIBRARIES = {'psutil', 'torch', 'numpy', 'PIL', 'PIL.Image', 'PIL.ImageFile', 'pycocotools',
             'pycocotools.coco', 'pycocotools.mask', 'pycocotools._mask'}
VERSIONS = dict(torch='2.0.1+cu118', numpy='1.23.5')
LIMITS = dict(allowed_cpus=[0, 1], threads=2, maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800, CUDA_visible_devices='')
SOURCE_PINS = {
    'export_source': 'c4b9ede7b942a57f81a9fb3cf029100ac734aa5c0a8b409d48c3423a1fb371e2',
    'rgb_reader_source': '678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025',
    'matched_reader_source': '82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870',
    'geometry_source': '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
}

EXPORT_SOURCE_BYTES = 13430  # Exact independently authored U53 helper STOP.


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str
         and (PureWindowsPath(value['path']).is_absolute() or PurePosixPath(value['path']).is_absolute())
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict absolute reference')
    return value


def _key(value):
    return strict_ref(value)['path'].replace('\\', '/').casefold()


def validate_spec(spec):
    """Pure metadata admission contract, without opening referenced files."""
    need(type(spec) is dict and spec.get('schema') == SCHEMA and spec.get('operation') == OPERATION
         and spec.get('expected_versions') == VERSIONS and spec.get('limits') == LIMITS
         and type(spec.get('dataset_indices')) is list and 0 < len(spec['dataset_indices']) <= 741
         and all(type(x) is int and 0 <= x < 741 for x in spec['dataset_indices'])
         and len(set(spec['dataset_indices'])) == len(spec['dataset_indices'])
         and type(spec.get('phase')) is str and spec['phase'], 'Fixed original CPU/versions and prelocked nonempty batch of at most741 unique original ordinals')
    need(set(spec['source_roles']) == ROLES and set(spec['library_sources']) == LIBRARIES
         and type(spec['input_sources']) is list and spec['input_sources']
         and type(spec['sources']) is list, 'Five code roles, ten selected genuine libraries and explicit inputs')
    admitted = {}
    for ref in spec['sources']:
        key = _key(ref)
        need(key not in admitted, 'Unique Source references')
        admitted[key] = ref
    direct = list(spec['source_roles'].values())+list(spec['library_sources'].values())
    direct += [spec['context_ref']]+spec['input_sources']
    for ref in direct:
        need(admitted.get(_key(ref)) == ref, 'Every declared direct dependency must be admitted')
    for role, pin in SOURCE_PINS.items():
        need(spec['source_roles'][role]['sha256'] == pin, 'Exact reviewed source: '+role)
    need(type(EXPORT_SOURCE_BYTES) is int and spec['source_roles']['export_source']['bytes'] == EXPORT_SOURCE_BYTES, 'Exact U53 STOP export helper bytes')


def summarize_export(package, receipt):
    """Whitelist proof scalars; do not copy identity/GT ids/queries/paths."""
    fields = {}
    for key in ('rgb400', 'e', 'M', 'L', 'targets'):
        value = package['equivalence'][key]
        fields[key] = {name: value[name] for name in ('shape', 'dtype', 'logical_bytes',
            'source_sha256', 'sha256', 'bitexact', 'independent_storage', 'compact_storage')}
    return dict(images=1, matched_instances=package['metadata']['G'], fields=fields,
                package_bytes=receipt['package_ref']['bytes'], package_sha256=receipt['package_ref']['sha256'],
                weights_only_CPU_roundtrip_verified=receipt['weights_only_CPU_roundtrip_verified'])


def reference(path, raw=None):
    path = Path(path)
    if raw is not None:
        return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def load(path, alias, aliases):
    aliases[alias] = (alias in sys.modules, sys.modules.get(alias))
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    need(Path(module.__file__).resolve() == path.resolve(), 'Actual admitted helper module')
    return module


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args(argv)
    out = None
    created = False
    original_error = None
    torch = np = psutil = process = sample = sampler = None
    reader = item = package = record = receipt = tensor = helper = modules = context = fit = dense = None
    outside_cpu = outside_numpy = outside_python = outside_dtype = outside_grad = outside_threads = None
    aliases, sources, cache, tensor_refs = {}, {}, {}, []
    stopped = threading.Event()
    observed = dict(peak_RSS_bytes=0, minimum_available_RAM_bytes=None, samples=0, violations=[], sampling_errors=[])
    flags = dict(source_bytes_verified=False, source_unchanged_after=False, CUDA_uninitialized_before=False,
        CUDA_uninitialized_after=False, CPU_only=False, callerCPU_RNG_preserved=False,
        callerNumPy_RNG_preserved=False, callerPython_RNG_preserved=False, default_dtype_preserved=False,
        grad_mode_preserved=False, thread_controls_preserved=False, module_aliases_restored=False,
        data_objects_released=False)
    report = dict(schema='U53-matched-CPU-export-report-v1', status='STARTED', operation=OPERATION,
        qualified=False, new_training=False, AP=False, DEV=False, TEST=False, whole_population_exported=False,
        actual_versions={}, library_paths={}, flags=flags, images=[], package_refs={}, resource_observations=observed,
        requested_batch_images=None, actual_completed_images=0, batch_export_complete=False,
        full_binary_closure_claimed=False, model_constructed=False, optimizer_created=False,
        model_forward_executed=False, loss_computed=False, matching_recomputed=False,
        FIT_annotation_records_parsed=None, COCO_dataset_object_constructed=None,
        FIT_annotation_records_parsing_attempted=False, selected_GT_rasterization_attempted=False,
        original_metadata_population=None, cleanup_errors=[], controls_scope='after owned CPU01/two-thread setup')
    try:
        need(not any(name.split('.')[0] in ('torch', 'numpy', 'PIL', 'pycocotools') for name in sys.modules),
             'No numerical/image library before source/environment closure')
        spec_path = Path(args.spec)
        need(spec_path.is_absolute(), 'Absolute original-driver-bound spec')
        raw = spec_path.read_bytes()
        spec = json.loads(raw)
        report.update(phase=spec.get('phase'), spec_ref=reference(spec_path, raw))
        out = Path(spec['output_directory'])
        need(out.is_absolute() and out.drive.casefold() == out.resolve().drive.casefold() == 'd:'
             and not out.exists() and not out.is_symlink(), 'Fresh canonical D output directory')
        out.mkdir(parents=True, exist_ok=False)
        created = True
        validate_spec(spec)
        report['requested_batch_images'] = len(spec['dataset_indices'])
        for key in ('source_roles', 'library_sources', 'context_ref', 'input_sources', 'limits', 'dataset_indices'):
            report[key] = spec[key]
        for ref in spec['sources']:
            key = str(Path(ref['path']).resolve()).casefold()
            need(key not in sources, 'Unique actual canonical Source path')
            sources[key] = ref

        def closed(ref):
            strict_ref(ref)
            path = Path(ref['path'])
            key = str(path.resolve()).casefold()
            need(sources.get(key) == ref, 'Exact admitted source/input membership')
            def identity():
                stat = path.stat()
                return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
            before = identity()
            if cache.get(key) != (ref, before):
                need(reference(path) == ref and identity() == before, 'Stable exact admitted source bytes')
                cache[key] = (ref, before)
            need(len(cache) <= len(sources), 'Source-bounded single-leaf hash cache')
            return path

        # All explicit bytes are closed before any numerical/image import.
        for ref in spec['sources']:
            closed(ref)
        need(Path(__file__).resolve() == closed(spec['source_roles']['entry_source']).resolve(), 'Entry self-source')
        flags['source_bytes_verified'] = True
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA hidden by original CPU admission')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            need(os.environ.get(key) == '2', 'Original two-thread environment: '+key)
        psutil = importlib.import_module('psutil')
        need(reference(Path(psutil.__file__).resolve()) == spec['library_sources']['psutil'], 'Canonical actual psutil source')
        report['library_paths']['psutil'] = str(Path(psutil.__file__).resolve())
        process = psutil.Process()
        need(process.cpu_affinity() == [0, 1], 'Actual admitted CPU0/1 before numerical import')

        def sample():
            try:
                rss, available = int(process.memory_info().rss), int(psutil.virtual_memory().available)
                observed['peak_RSS_bytes'] = max(observed['peak_RSS_bytes'], rss)
                prior = observed['minimum_available_RAM_bytes']
                observed['minimum_available_RAM_bytes'] = available if prior is None else min(prior, available)
                observed['samples'] += 1
                if rss > LIMITS['maximum_RSS_bytes'] or available < LIMITS['minimum_available_RAM_bytes']:
                    observed['violations'].append('RSS/free-RAM boundary')
            except BaseException as error:
                observed['sampling_errors'].append(type(error).__name__+': '+str(error))

        def check():
            sample()
            need(not observed['violations'] and not observed['sampling_errors'], 'Actual CPU resource observations')
            need(time.monotonic()-started <= LIMITS['deadline_seconds'], 'Owned export deadline1800 exceeded')

        def monitor():
            while not stopped.wait(.02):
                sample()

        class Control:
            allowed = sources
            def closed(self, ref):
                check()
                return closed(ref)
            def check(self):
                check()

        check()
        sampler = threading.Thread(target=monitor, daemon=True)
        sampler.start()
        imported = {}
        for name in ('torch', 'numpy', 'PIL', 'PIL.Image', 'PIL.ImageFile', 'pycocotools',
                     'pycocotools.coco', 'pycocotools.mask', 'pycocotools._mask'):
            obj = importlib.import_module(name)
            need(reference(Path(obj.__file__).resolve()) == spec['library_sources'][name], 'Canonical actual library: '+name)
            report['library_paths'][name] = str(Path(obj.__file__).resolve())
            imported[name] = obj
            check()
        torch, np = imported['torch'], imported['numpy']
        report['actual_versions'] = dict(torch=str(torch.__version__), numpy=str(np.__version__))
        need(report['actual_versions'] == VERSIONS and not torch.cuda.is_initialized()
             and torch.get_default_dtype() == torch.float32, 'Original versions, FP32 default and cold CUDA')
        flags['CUDA_uninitialized_before'] = True
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        need(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Actual two Torch threads')
        outside_cpu, outside_numpy, outside_python = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        outside_dtype, outside_grad = torch.get_default_dtype(), torch.is_grad_enabled()
        outside_threads = torch.get_num_threads(), torch.get_num_interop_threads()
        helper = load(closed(spec['source_roles']['export_source']), '_U53_EXPORT_HELPER', aliases)
        need(helper.validate_batch(spec['dataset_indices']) == spec['dataset_indices'], 'Exact unchanged helper-validated batch order')
        modules = {
            'matched_reader': load(closed(spec['source_roles']['matched_reader_source']), '_U53_MATCHED_READER', aliases),
            'image_geometry': load(closed(spec['source_roles']['geometry_source']), '_U53_IMAGE_GEOMETRY', aliases),
            'rgb_matched_reader': load(closed(spec['source_roles']['rgb_reader_source']), '_U53_RGB_READER', aliases),
        }
        context = json.loads(closed(spec['context_ref']).read_text(encoding='utf-8-sig'))
        need(context['schema'] == 'ROOT-U38-full-method-CPU-context-v1'
             and all(key in context for key in ('FIT_context', 'N13_qualified', 'N13_report')), 'Original U38 three-ref context')
        context_refs = {key: context[key] for key in ('FIT_context', 'N13_qualified', 'N13_report')}
        for ref in context_refs.values():
            closed(ref)
        fit = json.loads(closed(context_refs['FIT_context']).read_bytes())
        dense = json.loads(closed(context_refs['N13_report']).read_bytes())
        expected_inputs = list(context_refs.values())+[fit['FIT']]
        for index in spec['dataset_indices']:
            expected_inputs.extend([dense['validation_carrier_refs'][index], fit['existing_packet_refs'][index],
                                    fit['existing_native_L_refs'][index], fit['identities'][index]['image']])
        wanted = {_key(ref): ref for ref in expected_inputs}
        actual = {_key(ref): ref for ref in spec['input_sources']}
        need(actual == wanted and len(actual) == len(spec['input_sources']), 'Exactly original metadata and explicitly selected batch image/tensor refs, deduplicated by actual path')
        for ref in expected_inputs:
            closed(ref)
        source_refs = {key: spec['source_roles'][role] for key, role in
                      (('rgb_matched_reader', 'rgb_reader_source'), ('matched_reader', 'matched_reader_source'),
                       ('image_geometry', 'geometry_source'))}
        provenance = dict(context=context_refs, source_refs=source_refs, input_refs=spec['input_sources'])
        report['FIT_annotation_records_parsing_attempted'] = True
        reader = modules['rgb_matched_reader'].prepare(torch, np, Control(), context,
            matched_reader=modules['matched_reader'], image_geometry=modules['image_geometry'],
            source_refs={key: source_refs[key] for key in ('matched_reader', 'image_geometry')})
        report['COCO_dataset_object_constructed'] = report['FIT_annotation_records_parsed'] = True
        need(len(reader.records) == 741 and sum(row['G'] for row in reader.records) == 1661, 'Original complete metadata population')
        report['original_metadata_population'] = dict(images=741, GT=1661, tensor_population_revalidated=False)
        for index in spec['dataset_indices']:
            check()
            # Previous image passed its release check; retain weakrefs for this image only.
            tensor_refs = []
            stage = dict(dataset_index=index, status='LOADING', reader_load_returned=False, package_written=False)
            report['images'].append(stage)
            record = reader.records[index]
            need(record['dataset_index'] == index, 'Keep original dataset ordinal, not compact index0/1')
            report['selected_GT_rasterization_attempted'] = True
            item = reader.load(index)
            stage['reader_load_returned'] = True
            for tensor in item.values():
                if isinstance(tensor, torch.Tensor):
                    tensor_refs.append(weakref.ref(tensor))
            tensor = None
            stage['status'] = 'COPYING_AND_VERIFYING'
            package = helper.export_item(torch, index, item, record, provenance=provenance, check=check,
                batch_indices=spec['dataset_indices'])
            tensor_refs.extend(weakref.ref(value) for value in package['tensors'].values())
            stage['status'] = 'SAVING_AND_ROUNDTRIPPING'
            receipt = helper.save_export(torch, package, out/('MATCHED_FIT_'+str(index)+'.pt'), check=check)
            stage.update(status='COMPLETE', package_written=True, summary=summarize_export(package, receipt))
            report['package_refs'][str(index)] = receipt['package_ref']
            report['actual_completed_images'] += 1  # Preserve actual returned save/roundtrip count even if later cleanup fails.
            item = package = record = receipt = None
            gc.collect()
            need(all(ref() is None for ref in tensor_refs), 'Per-image source/export/roundtrip tensors released before next image')
            check()
        for ref in spec['sources']:
            closed(ref)
        flags['source_unchanged_after'] = True
        need(not torch.cuda.is_initialized(), 'No CUDA initialization during reader/export')
        check()
        report['status'] = 'COMPLETE'
    except BaseException as error:
        original_error = error
        report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
        raise
    finally:
        reader = item = package = record = receipt = tensor = helper = modules = context = fit = dense = None
        gc.collect()
        flags['data_objects_released'] = bool(tensor_refs) and all(ref() is None for ref in tensor_refs)
        if torch is not None and outside_cpu is not None:
            try:
                flags['callerCPU_RNG_preserved'] = bool(torch.equal(outside_cpu, torch.get_rng_state()))
                current = np.random.get_state()
                flags['callerNumPy_RNG_preserved'] = current[0] == outside_numpy[0] and bool(np.array_equal(current[1], outside_numpy[1])) and current[2:] == outside_numpy[2:]
                flags['callerPython_RNG_preserved'] = random.getstate() == outside_python
                flags['default_dtype_preserved'] = torch.get_default_dtype() == outside_dtype
                flags['grad_mode_preserved'] = torch.is_grad_enabled() == outside_grad
                flags['thread_controls_preserved'] = (torch.get_num_threads(), torch.get_num_interop_threads()) == outside_threads
            except BaseException as error:
                report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        outside_cpu = outside_numpy = outside_python = None
        stopped.set()
        if sampler is not None:
            try:
                sampler.join(timeout=1)
                if sampler.is_alive():
                    observed['sampling_errors'].append('Resource sampler did not terminate')
            except BaseException as error:
                observed['sampling_errors'].append(type(error).__name__+': '+str(error))
        if callable(sample):
            sample()
        for alias, (existed, original) in reversed(list(aliases.items())):
            if existed:
                sys.modules[alias] = original
            else:
                sys.modules.pop(alias, None)
        flags['module_aliases_restored'] = all((alias in sys.modules) == existed and
            (not existed or sys.modules[alias] is original) for alias, (existed, original) in aliases.items())
        flags['CUDA_uninitialized_after'] = torch is not None and not torch.cuda.is_initialized()
        flags['CPU_only'] = flags['CUDA_uninitialized_before'] and flags['CUDA_uninitialized_after']
        report['cost'] = dict(elapsed_seconds=time.monotonic()-started, peak_RSS_bytes=observed['peak_RSS_bytes'])
        report['RSS_peak_is_sampled'] = True
        report['source_hash_cache'] = dict(unique_paths_verified=len(cache), maximum_paths=len(sources),
            scope='one owned leaf; file identity/size/mtime/ctime checked per use', whole_environment_closure_claimed=False)
        bad = report['status'] == 'COMPLETE' and (not all(flags.values()) or bool(observed['violations'])
            or bool(observed['sampling_errors']) or bool(report['cleanup_errors'])
            or report['cost']['elapsed_seconds'] > LIMITS['deadline_seconds']
            or report['actual_completed_images'] != report['requested_batch_images']
            or [row['dataset_index'] for row in report['images']] != spec['dataset_indices'])
        if bad:
            report.update(status='FAILED', error=dict(type='RuntimeError', message='Final CPU export/source/resource/cleanup contract failed'))
        report['batch_export_complete'] = report['status'] == 'COMPLETE'
        if created and out.is_dir():
            try:
                with (out/'REPORT.json').open('x', encoding='utf-8') as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            except BaseException:
                if original_error is None:
                    raise
        if bad and original_error is None:
            raise RuntimeError('Final CPU export/source/resource/cleanup contract failed')


if __name__ == '__main__':
    main()
