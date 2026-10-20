"""UNBOUND Windows CPU remaining186/full100 DEV export worker; main is not tested.

Root alone binds inputs, admission, supervision, actual termination and results.
No numerical/image imports occur at import time or in metadata-only functions.
"""
import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import sys
import threading
import time
import traceback
import types
import weakref


SCHEMA = 'U62-DEV-remaining186-full100-CPU-export-spec-v1'
OPERATION = 'CPU_U62_DEV_REMAINING186_FULL100_EXPORT'
ROLES = {'entry_source', 'export_source', 'dev_reader_source', 'dev_rgb_reader_source', 'geometry_source'}
LIBRARIES = {'psutil', 'torch', 'numpy', 'PIL', 'PIL.Image', 'PIL.ImageFile', 'pycocotools',
             'pycocotools.coco', 'pycocotools.mask', 'pycocotools._mask'}
VERSIONS = dict(torch='2.0.1+cu118', numpy='1.23.5')
LIMITS = dict(allowed_cpus=[0, 1], threads=2, maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800, CUDA_visible_devices='')
SOURCE_PINS = dict(
    export_source='8fabc800e4d36e2da3988e42acf88058c80790f9c8b3d664a1a4fa9232233473',
    dev_reader_source='daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    dev_rgb_reader_source='b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204',
    geometry_source='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7')
EXPORT_SOURCE_BYTES = 20627
SELECTED_INDICES = tuple(range(1, 187))
MINIMUM_OUTPUT_FREE_BYTES = 20*1024**3
SPEC_KEYS = {'schema', 'operation', 'phase', 'expected_versions', 'limits', 'dataset_indices',
             'source_roles', 'library_sources', 'context_ref', 'original_image_refs',
             'input_sources', 'sources', 'output_directory', 'minimum_output_free_bytes'}
CONTEXT_REF_KEYS = ('legacy_context', 'native_predictions', 'R_predictions', 'encoder_source', 'legacy_ROOT_result')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str
         and (PureWindowsPath(value['path']).is_absolute() or PurePosixPath(value['path']).is_absolute())
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict absolute path/bytes/SHA256 ref')
    return value


def _key(value):
    return strict_ref(value)['path'].replace('\\', '/').casefold()


def _ref_map(values, *, reject_duplicates=False):
    need(type(values) is list, 'Explicit reference list required')
    result = {}
    for reference in values:
        key = _key(reference)
        need(key not in result or (not reject_duplicates and result[key] == reference),
             'Duplicate or conflicting admitted path reference')
        result[key] = reference
    return result


def _image_references(values):
    need(type(values) is list and len(values) == 187, 'Original complete187 ordinal-ordered JPEG refs')
    _ref_map(values, reject_duplicates=True)
    return values


def validate_spec(spec):
    """Pure metadata only. No owner, parent/accounting, input bytes or evidence."""
    need(type(spec) is dict and set(spec) == SPEC_KEYS and spec['schema'] == SCHEMA
         and spec['operation'] == OPERATION and type(spec['phase']) is str and spec['phase'], 'Exact dedicated U62 spec ABI')
    need(spec['expected_versions'] == VERSIONS and spec['limits'] == LIMITS
         and type(spec['dataset_indices']) is list and spec['dataset_indices'] == list(SELECTED_INDICES)
         and all(type(index) is int for index in spec['dataset_indices']),
         'Only original ordinals1..186 in order; unchanged CPU resource/version contract')
    need(type(spec['minimum_output_free_bytes']) is int and spec['minimum_output_free_bytes'] == MINIMUM_OUTPUT_FREE_BYTES,
         'Fixed20GiB D output reserve, not an adjustable resource floor')
    need(type(spec['source_roles']) is dict and set(spec['source_roles']) == ROLES
         and type(spec['library_sources']) is dict and set(spec['library_sources']) == LIBRARIES,
         'Five explicit original/helper roles and ten genuine selected library refs')
    for role, pin in SOURCE_PINS.items():
        need(strict_ref(spec['source_roles'][role])['sha256'] == pin, 'Exact reviewed source: '+role)
    need(spec['source_roles']['export_source']['bytes'] == EXPORT_SOURCE_BYTES, 'Exact unchanged U59 helper bytes')
    _image_references(spec['original_image_refs'])
    need(type(spec['input_sources']) is list and spec['input_sources'], 'Explicit selected DEV input refs')
    _ref_map(spec['input_sources'], reject_duplicates=True)
    admitted = _ref_map(spec['sources'], reject_duplicates=True)
    direct = list(spec['source_roles'].values())+list(spec['library_sources'].values())
    direct += [spec['context_ref']]+spec['input_sources']
    need(admitted == _ref_map(direct), 'Source set is exactly declared code/libraries/context/selected inputs')
    need(all(admitted.get(_key(spec['original_image_refs'][index])) == spec['original_image_refs'][index]
             for index in SELECTED_INDICES), 'Every selected original JPEG1..186 must be admitted')
    need(type(spec['output_directory']) is str and PureWindowsPath(spec['output_directory']).is_absolute()
         and PureWindowsPath(spec['output_directory']).drive.casefold() == 'd:', 'Absolute new Windows D output')


def _records(context):
    need(type(context) is dict and context.get('schema') == 'ROOT-N28-fixed24-dense-native-DEV-context-v1',
         'Original full DEV context schema')
    ids, records = context['image_ids'], context['rows']
    need(type(ids) is list and len(ids) == 187 and all(type(value) is int and value >= 0 for value in ids)
         and len(set(ids)) == 187 and type(records) is list and len(records) == 187, 'Full187 unique-image metadata')
    for index, row in enumerate(records):
        need(type(row) is dict and type(row.get('dataset_index')) is int and row['dataset_index'] == index
             and type(row.get('image_id')) is int and row['image_id'] == ids[index], 'Original ordinal/image metadata joins')
        qids = row['qids']
        need(type(qids) is list and len(qids) == 100 and all(type(qid) is int for qid in qids)
             and set(qids) == set(range(100)), 'Original all100 query export permutation; never sort/filter')
    return records


def expected_input_references(context, original_image_refs):
    """Only metadata refs plus original selected1..186 manifests/bundles/packets/JPEGs.

    Unselected image refs remain metadata. Teacher refs inside the context are
    identity comparisons, never weight inputs or automatic recursive closure.
    """
    records = _records(context)
    _image_references(original_image_refs)
    refs = [context[key] for key in CONTEXT_REF_KEYS]
    for index in SELECTED_INDICES:
        refs += [records[index][key] for key in ('original_capture_manifest', 'bundle', 'dense_packet')]
        refs += [original_image_refs[index]]
    return list(_ref_map(refs).values())


def validate_context_inputs(spec, context):
    expected = expected_input_references(context, spec['original_image_refs'])
    need(_ref_map(spec['input_sources'], reject_duplicates=True) == _ref_map(expected),
         'Exactly original context refs and selected1..186 inputs; no extra row0 image/tensor/teacher/annotation')
    return expected


def item_input_references(spec, context, index):
    """Original U59 five-ref provenance for this actual, non-renumbered image."""
    need(type(index) is int and index in SELECTED_INDICES, 'Only original selected ordinal1..186')
    row = context['rows'][index]
    need(type(row['dataset_index']) is int and row['dataset_index'] == index, 'Original provenance ordinal join')
    values = dict(dev_context=spec['context_ref'], native_predictions=context['native_predictions'],
        original_capture_manifest=row['original_capture_manifest'], bundle=row['bundle'], dense_packet=row['dense_packet'])
    for value in values.values():
        strict_ref(value)
    return values


def observe_resources(observed, *, rss_bytes, available_ram_bytes, output_free_bytes):
    """Pure sampled-scalar aggregation; tests do not query any process or disk."""
    need(all(type(value) is int and value >= 0 for value in (rss_bytes, available_ram_bytes, output_free_bytes)),
         'Actual nonnegative integer resource observations')
    observed['peak_RSS_bytes'] = max(observed['peak_RSS_bytes'], rss_bytes)
    prior = observed['minimum_available_RAM_bytes']
    observed['minimum_available_RAM_bytes'] = available_ram_bytes if prior is None else min(prior, available_ram_bytes)
    prior = observed['minimum_output_disk_free_bytes']
    observed['minimum_output_disk_free_bytes'] = output_free_bytes if prior is None else min(prior, output_free_bytes)
    observed['samples'] += 1
    observed['output_disk_samples'] += 1
    if rss_bytes > LIMITS['maximum_RSS_bytes'] or available_ram_bytes < LIMITS['minimum_available_RAM_bytes']:
        observed['violations'].append('RSS/free-RAM boundary')
    if output_free_bytes < MINIMUM_OUTPUT_FREE_BYTES:
        observed['violations'].append('D output free-space boundary')


def validate_observations(observed):
    """Same pure resource-failure gate called by every production Control check."""
    need(not observed['violations'] and not observed['sampling_errors'], 'Actual CPU/D-output resource observations')


def validate_native_population(records, native):
    """Full187/18700 metadata join, without rewriting even unknown fields.

    The selected original100 dictionaries subsequently face U59's stricter ABI;
    this check must not strip an unknown field to make that export succeed.
    """
    need(type(records) is list and len(records) == 187 and type(native) is list and len(native) == 18700,
         'Complete original18700 native predictions')
    ids = set()
    for index, record in enumerate(records):
        need(type(record) is dict and type(record.get('dataset_index')) is int and record['dataset_index'] == index
             and type(record.get('image_id')) is int and record['image_id'] not in ids, 'Canonical unique native image order')
        ids.add(record['image_id'])
        qids = record['qids']
        need(type(qids) is list and len(qids) == 100 and all(type(qid) is int for qid in qids)
             and set(qids) == set(range(100)), 'Original all100 export permutation')
        for qid, row in zip(qids, native[index*100:(index+1)*100]):
            need(type(row) is dict and type(row.get('image_id')) is int and row['image_id'] == record['image_id']
                 and type(row.get('query_id')) is int and row['query_id'] == qid
                 and type(row.get('category_id')) is int and row['category_id'] == 1
                 and type(row.get('score')) in (int, float) and math.isfinite(row['score']) and 0 <= row['score'] <= 1
                 and type(row.get('segmentation')) is dict, 'Original native image/query/class/score/order, including empty masks')
    return dict(images=187, queries=18700, tensors_revalidated=False)


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
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    sys.modules[alias] = module
    # Avoid __pycache__ writes into the immutable admitted helper directories.
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    need(Path(module.__file__).resolve() == path.resolve(), 'Actual admitted helper module')
    return module


def summarize_export(package, receipt, checks, native_sha256):
    """Whitelist post-execution proof scalars, never whole identity/row data."""
    names = ('rgb400', 'e', 'M', 'L')
    need(type(checks) is dict and set(checks) == set(names)
         and all(set(checks[name]) == {'source_bitexact', 'independent_storage', 'compact_storage'}
                 and all(value is True for value in checks[name].values()) for name in names),
         'All four genuine source/export byte and storage checks must pass')
    need(type(native_sha256) is str and len(native_sha256) == 64
         and set(native_sha256) <= set('0123456789abcdef'), 'Explicit original100 dictionary metadata digest')
    fields = {name: dict({key: package['proof']['fields'][name][key] for key in ('shape', 'dtype', 'sha256')},
                         **checks[name]) for name in names}
    return dict(images=1, queries=100, fields=fields,
        metadata_sha256=package['proof']['metadata_sha256'], field_proof_schema=package['proof']['schema'],
        native_predictions_metadata_sha256=native_sha256, original_dictionaries_exact=True,
        query_order_exact=True, natural_query_axes_retained=True, weights_only_CPU_roundtrip_verified=True,
        package_bytes=receipt['package_ref']['bytes'], package_sha256=receipt['package_ref']['sha256'])


def main(argv=None):
    """Future Root-owned CPU leaf only. Metadata tests never call this path."""
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args(argv)
    out = None
    created = False
    original_error = None
    torch = np = psutil = process = sample = sampler = None
    reader = item = package = record = receipt = tensor = helper = modules = context = native = originals = None
    source_tensor = copied_tensor = None
    outside_cpu = outside_numpy = outside_python = outside_dtype = outside_grad = outside_threads = None
    aliases, sources, cache, tensor_refs = {}, {}, {}, []
    stopped = threading.Event()
    observed = dict(peak_RSS_bytes=0, minimum_available_RAM_bytes=None, samples=0, violations=[], sampling_errors=[],
        minimum_output_disk_free_bytes=None, output_disk_samples=0, output_disk_sample_path=None)
    flags = dict(source_bytes_verified=False, source_unchanged_after=False, CUDA_uninitialized_before=False,
        CUDA_uninitialized_after=False, CPU_only=False, callerCPU_RNG_preserved=False,
        callerNumPy_RNG_preserved=False, callerPython_RNG_preserved=False, default_dtype_preserved=False,
        grad_mode_preserved=False, thread_controls_preserved=False, module_aliases_restored=False,
        data_objects_released=False)
    report = dict(schema='U62-DEV-remaining186-full100-CPU-export-report-v1', status='STARTED', operation=OPERATION,
        qualified=False, new_training=False, AP=False, DEV=True, TEST=False, whole_population_exported=False,
        actual_versions={}, library_paths={}, flags=flags, images=[], package_refs={}, export_entries=[],
        resource_observations=observed, requested_batch_images=186, actual_completed_images=0,
        minimum_output_free_bytes=MINIMUM_OUTPUT_FREE_BYTES,
        actual_completed_queries=0, batch_export_complete=False, full_binary_closure_claimed=False,
        model_constructed=False, teacher_constructed=False, teacher_weights_loaded=False,
        optimizer_created=False, model_forward_executed=False, loss_computed=False,
        matching_recomputed=False, new_GT_rasterization_executed=False,
        COCO_dataset_object_constructed=False, DEV_annotation_opened=False,
        historical_GT_container_may_have_been_loaded=True,
        input_scope='Original full187 metadata; selected original ordinals1..186/full100 only; no ordinal0 re-export',
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
        observed['output_disk_sample_path'] = str(out.parent.resolve())
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

        # Reject overbroad input bindings before hashing any other image file.
        # Only this known context JSON is read to derive the exact selected union.
        context = json.loads(closed(spec['context_ref']).read_bytes())
        validate_context_inputs(spec, context)
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
                observe_resources(observed, rss_bytes=int(process.memory_info().rss),
                    available_ram_bytes=int(psutil.virtual_memory().available),
                    output_free_bytes=int(psutil.disk_usage(str(out.parent)).free))
            except BaseException as error:
                observed['sampling_errors'].append(type(error).__name__+': '+str(error))

        def check():
            sample()
            validate_observations(observed)
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
        helper = load(closed(spec['source_roles']['export_source']), '_U62_EXPORT_HELPER', aliases)
        modules = {
            'dev_reader': load(closed(spec['source_roles']['dev_reader_source']), '_U62_DEV_READER', aliases),
            'image_geometry': load(closed(spec['source_roles']['geometry_source']), '_U62_IMAGE_GEOMETRY', aliases),
            'dev_rgb_reader': load(closed(spec['source_roles']['dev_rgb_reader_source']), '_U62_DEV_RGB_READER', aliases),
        }
        source_refs = {key: spec['source_roles'][role] for key, role in
            (('dev_reader', 'dev_reader_source'), ('dev_rgb_reader', 'dev_rgb_reader_source'),
             ('image_geometry', 'geometry_source'))}
        control = Control()
        reader = modules['dev_rgb_reader'].prepare(torch, np, control, context, spec['original_image_refs'],
            dev_reader=modules['dev_reader'], image_geometry=modules['image_geometry'],
            source_refs={key: source_refs[key] for key in ('dev_reader', 'image_geometry')})
        need(len(reader.records) == 187 and reader.records == context['rows'], 'Original unchanged complete187 records')
        native = json.loads(closed(context['native_predictions']).read_bytes())
        report['original_metadata_population'] = validate_native_population(reader.records, native)
        report['original_image_references_metadata_sha256'] = helper.metadata_sha256(spec['original_image_refs'])
        for index in spec['dataset_indices']:
            check()
            tensor_refs = []
            stage = dict(dataset_index=index, status='LOADING', reader_load_returned=False, package_written=False, queries=100)
            report['images'].append(stage)
            record = reader.records[index]
            need(type(record['dataset_index']) is int and record['dataset_index'] == index and index in SELECTED_INDICES,
                 'Original selected ordinal1..186; never compact-renumber or repeat0')
            input_refs = item_input_references(spec, context, index)
            stage['image_id'] = record['image_id']
            item = reader.load(index)
            stage['reader_load_returned'] = True
            for tensor in item.values():
                if isinstance(tensor, torch.Tensor):
                    tensor_refs.append(weakref.ref(tensor))
            tensor = None
            stage['status'] = 'COPYING_AND_VERIFYING'
            originals = native[index*100:(index+1)*100]
            package = helper.export_item(torch, control, index, item, originals,
                image_geometry=modules['image_geometry'], source_refs=source_refs, input_refs=input_refs)
            tensor_refs.extend(weakref.ref(value) for value in package['tensors'].values())
            checks = {}
            for name in ('rgb400', 'e', 'M', 'L'):
                check()
                source_tensor, copied_tensor = item[name], package['tensors'][name]
                source_digest = hashlib.sha256(source_tensor.detach().contiguous().numpy().tobytes()).hexdigest()
                bitexact = bool(torch.equal(source_tensor, copied_tensor)) and source_digest == package['proof']['fields'][name]['sha256']
                independent = source_tensor.data_ptr() != copied_tensor.data_ptr()
                compact = (copied_tensor.is_contiguous() and copied_tensor.storage_offset() == 0
                    and copied_tensor.untyped_storage().nbytes() == copied_tensor.numel()*copied_tensor.element_size())
                need(bitexact and independent and compact, 'Original full100 bytes and independent compact storage: '+name)
                checks[name] = dict(source_bitexact=True, independent_storage=True, compact_storage=True)
            source_tensor = copied_tensor = None
            need(helper.canonical_json(package['metadata']['native_predictions']) == helper.canonical_json(originals)
                 and package['metadata']['query_ids'] == record['qids'] == item['query_ids']
                 and package['metadata']['natural_query_ids'] == list(range(100)),
                 'Original100 whole dictionaries and separate natural/export query axes unchanged')
            stage['status'] = 'SAVING_AND_ROUNDTRIPPING'
            receipt = helper.save_export(torch, control, package, out/('DEV_FULL100_'+str(index)+'.pt'),
                image_geometry=modules['image_geometry'], source_refs=source_refs)
            report['actual_completed_images'] += 1  # Returned real save/roundtrip count, even if cleanup later fails.
            report['actual_completed_queries'] += 100
            stage['package_written'] = True
            need(set(receipt) == {'dataset_index', 'image_id', 'query_ids', 'package_ref'}
                 and receipt['dataset_index'] == index and receipt['image_id'] == record['image_id']
                 and receipt['query_ids'] == record['qids'], 'Exact original single-image portable entry')
            stage.update(status='COMPLETE', summary=summarize_export(package, receipt, checks, helper.metadata_sha256(originals)))
            report['package_refs'][str(index)] = receipt['package_ref']
            report['export_entries'].append(receipt)
            item = package = record = receipt = originals = None
            gc.collect()
            need(all(ref() is None for ref in tensor_refs), 'Per-image source/export/roundtrip tensors released before next image')
            stage['data_objects_released'] = True
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
        reader = item = package = record = receipt = tensor = helper = modules = context = native = originals = None
        source_tensor = copied_tensor = None
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
            or report['actual_completed_queries'] != 18600
            or observed['minimum_output_disk_free_bytes'] is None
            or observed['minimum_output_disk_free_bytes'] < MINIMUM_OUTPUT_FREE_BYTES
            or len(report['export_entries']) != 186
            or [row['dataset_index'] for row in report['images']] != spec['dataset_indices'])
        if bad:
            report.update(status='FAILED', error=dict(type='RuntimeError', message='Final CPU export/source/resource/cleanup contract failed'))
        report['batch_export_complete'] = report['status'] == 'COMPLETE'
        if created and out.is_dir():
            # Final small failure audit is preserved even when the reserve gate
            # forbids further scientific writes. Sample again before this write.
            if callable(sample):
                sample()
                report['cost'] = dict(elapsed_seconds=time.monotonic()-started, peak_RSS_bytes=observed['peak_RSS_bytes'])
                if report['status'] == 'COMPLETE' and (observed['violations'] or observed['sampling_errors']
                    or report['cost']['elapsed_seconds'] > LIMITS['deadline_seconds']):
                    report.update(status='FAILED', error=dict(type='RuntimeError', message='Final D output/resource audit failed'))
                    report['batch_export_complete'] = False
                    bad = True
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
