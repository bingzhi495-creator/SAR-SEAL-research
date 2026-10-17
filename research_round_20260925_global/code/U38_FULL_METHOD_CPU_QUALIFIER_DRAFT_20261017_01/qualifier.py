"""UNBOUND CPU-only reader/common-initialization qualifier; no scientific imports.

The Root caller supplies Torch/NumPy, admitted controls and source-closed modules.
No model forward, optimizer, training, GPU resource qualification or file output.
"""

from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import weakref


TOP_SOURCE_SHA256 = {
    'shared_initialization': '4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    'rgb_matched_reader': '678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025',
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def metadata_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def resource_candidates(rows):
    """Pure metadata inventory, not a GPU admission or data-dependent AP choice."""
    need(type(rows) is list and bool(rows), 'At least one validated metadata row required')
    indices = set()
    for row in rows:
        need(type(row) is dict and type(row.get('dataset_index')) is int
             and row['dataset_index'] >= 0 and row['dataset_index'] not in indices,
             'Distinct nonnegative dataset indices required')
        indices.add(row['dataset_index'])
        need(type(row.get('G')) is int and 1 <= row['G'] <= 100,
             'Positive original matched-query count required')
        shape = row.get('original_shape')
        need(type(shape) in (list, tuple) and len(shape) == 2
             and all(type(v) is int and v > 0 for v in shape), 'Original H/W metadata required')

    def area(row):
        return row['original_shape'][0] * row['original_shape'][1]

    chosen = {
        'small': min(rows, key=lambda r: (r['G'], area(r), r['dataset_index'])),
        'max_area': min(rows, key=lambda r: (-area(r), r['dataset_index'])),
        'max_G': min(rows, key=lambda r: (-r['G'], r['dataset_index'])),
        'max_G_area': min(rows, key=lambda r: (-r['G']*area(r), r['dataset_index'])),
    }
    roles = {role: dict(dataset_index=row['dataset_index'], G=row['G'],
                        original_shape=list(row['original_shape']), area=area(row),
                        matched_pixel_burden=row['G']*area(row))
             for role, row in chosen.items()}
    inventory = []
    for role in ('small', 'max_area', 'max_G', 'max_G_area'):
        index = roles[role]['dataset_index']
        if index not in inventory:
            inventory.append(index)
    small = roles['small']['dataset_index']
    primary = [small, roles['max_G_area']['dataset_index'], small]
    by_index = {row['dataset_index']: row for row in rows}
    coverage = dict(unique_images=len(set(primary)),
                    distinct_original_shapes=len({tuple(by_index[i]['original_shape']) for i in primary}),
                    distinct_G=len({by_index[i]['G'] for i in primary}))
    return dict(roles=roles,
                candidate_inventory_with_return_small=inventory+[small],
                primary_three_image_proposal=primary, primary_coverage=coverage,
                selection_rule='small=min(G,area,index); each maximum ties by minimum original index',
                deduplication='Deduplicate first four inventory roles only; deliberate final small repeat',
                primary_rule='small -> max(G*original_H*original_W) -> same small',
                GPU_executed=False, GPU_gate_passed=False,
                all_extremes_mandatory=False, repeated_sequence_stability_verified=False)


def _closed_module(c, role, module, reference, digest):
    need(type(reference) is dict and set(reference) == {'path', 'bytes', 'sha256'}
         and type(reference['path']) is str and Path(reference['path']).is_absolute()
         and type(reference['bytes']) is int and reference['bytes'] >= 0
         and reference['sha256'] == digest, 'Exact reviewed module reference: '+role)
    path = Path(c.closed(reference)).resolve(strict=True)
    module_file = getattr(module, '__file__', None)
    need(type(module_file) is str and Path(module_file).resolve(strict=True) == path,
         'Loaded module must match admitted source: '+role)
    raw = path.read_bytes()
    need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == digest,
         'Reviewed source bytes differ: '+role)


def _cold(torch):
    need(not torch.cuda.is_initialized(), 'CPU qualifier requires CUDA to remain uninitialized')
    need(torch.get_default_dtype() == torch.float32, 'Explicit FP32 default required')


def _elapsed(start):
    elapsed = float(time.perf_counter()-start)
    need(math.isfinite(elapsed) and elapsed >= 0, 'Finite nonnegative elapsed time required')
    return elapsed


def _sample(c, report, phase, dataset_index=None):
    c.check()
    rss = int(c.psutil.Process().memory_info().rss)
    available = int(c.psutil.virtual_memory().available)
    row = dict(phase=phase, dataset_index=dataset_index, RSS_bytes=rss,
               system_available_RAM_bytes=available)
    report['resource_samples'].append(row)
    report['max_boundary_sampled_RSS_bytes'] = max(report['max_boundary_sampled_RSS_bytes'], rss)
    previous = report['min_boundary_sampled_available_RAM_bytes']
    report['min_boundary_sampled_available_RAM_bytes'] = available if previous is None else min(previous, available)


def _pair_metadata(torch, adapters, receipt, initializer):
    need(set(adapters) == {'PLAIN', 'FULL'}
         and receipt['schema'] == 'U34-common-operation-initialization-v1'
         and receipt['status'] == 'INITIALIZED_CPU_ONLY'
         and receipt['seed'] == receipt['adapter_seed'] == 202610173101
         and receipt['common_parameter_tensors'] == 68 and receipt['common_BN_buffers'] == 36,
         'Exact existing common-initialization receipt required')
    expected = initializer.static_state_contract()
    need(len(receipt['common_state']) == 104
         and {row['name'] for row in receipt['common_state']} == set(expected),
         'All 104 reviewed shared states checked by U34, without partial matching')
    need(all(receipt[key] is False for key in ('CUDA_initialized', 'forward_executed',
         'backward_executed', 'optimizer_created', 'training_started', 'old_UNet_weights_loaded')),
         'Initializer must remain CPU construction/copy only')
    counts = {}
    for arm, adapter in adapters.items():
        parameters = dict(adapter.named_parameters())
        inactive = {'backbone.'+name for name in adapter.backbone.inactive_parameter_names()}
        need(inactive <= set(parameters), 'Every original inactive name must remain allocated')
        need(all(p.device.type == 'cpu' and p.dtype == torch.float32 and p.grad is None
                 for p in parameters.values()), 'Fresh CPU FP32 parameters without gradients')
        need(all(value.device.type == 'cpu' for value in adapter.buffers()), 'CPU-only model buffers')
        allocated = sum(p.numel() for p in parameters.values())
        inactive_count = sum(parameters[name].numel() for name in inactive)
        counts[arm] = dict(allocated_parameters=int(allocated),
                          declared_inactive_parameters=int(inactive_count),
                          declared_effective_parameters=int(allocated-inactive_count),
                          allocated_parameter_tensors=len(parameters),
                          declared_inactive_parameter_tensors=len(inactive),
                          requires_grad_flags_changed=False)
    return counts


def qualify(torch, np, c, context, modules, source_refs, image_control=None, report=None):
    """Future CPU-only qualification; returns JSON metadata, never models/tensors.

    modules/source_refs contain shared_initialization, rgb_matched_reader, their
    seven U34 components and the two U37 dependencies (eleven roles in total).
    Root must admit inputs/libraries before this call and enforce runtime limits.
    Pass an empty report dict to preserve partial progress if this function raises.
    """
    if report is None:
        report = {}
    need(type(report) is dict and not report, 'Caller report must be a fresh empty dict')
    started = time.perf_counter()
    report.update(schema='U38-full-method-CPU-qualification-draft-v1', status='STARTED',
                  expected_images=741, expected_GT=1661, images_completed=0, GT_completed=0,
                  source_refs=deepcopy(source_refs), rows=[], resource_samples=[],
                  max_boundary_sampled_RSS_bytes=0,
                  min_boundary_sampled_available_RAM_bytes=None,
                  resource_scope='Boundary samples only, not continuous RSS peak or a leak diagnosis',
                  CUDA_initialized=False, model_forward_executed=False, backward_executed=False,
                  optimizer_created=False, new_training=False, GPU_resource_gate_passed=False,
                  AP=False, DEV=False, TEST=False, weights_loaded=False, weights_written=False)
    reader = item = adapters = initialization = None
    references = []
    try:
        _cold(torch)
        c.check()
        need(type(modules) is dict and type(source_refs) is dict, 'Injected module/ref dictionaries')
        for role, digest in TOP_SOURCE_SHA256.items():
            _closed_module(c, role, modules[role], source_refs[role], digest)
        initializer = modules['shared_initialization']
        rgb_reader = modules['rgb_matched_reader']
        child_hashes = dict(initializer.SOURCE_SHA256)
        need(not set(child_hashes).intersection(rgb_reader.SOURCE_SHA256), 'Distinct dependency roles')
        child_hashes.update(rgb_reader.SOURCE_SHA256)
        need(set(modules) == set(source_refs) == set(TOP_SOURCE_SHA256) | set(child_hashes),
             'Exactly eleven reviewed source-closed module roles')
        for role, digest in child_hashes.items():
            _closed_module(c, role, modules[role], source_refs[role], digest)
        _sample(c, report, 'before_reader')
        reader_started = time.perf_counter()
        reader = rgb_reader.prepare(torch, np, c, context,
            matched_reader=modules['matched_reader'], image_geometry=modules['image_geometry'],
            source_refs={key: source_refs[key] for key in rgb_reader.SOURCE_SHA256},
            image_control=image_control)
        need(len(reader.records) == 741, 'Original complete 741-image population required')
        report['input_reference_count'] = len(reader.input_references)
        report['input_references_metadata_sha256'] = metadata_sha256(reader.input_references)
        report['image_reference_count'] = len(reader.image_references)
        seen_images, seen_annotations = set(), set()
        for index in range(741):
            _cold(torch)
            c.check()
            row = reader.records[index]
            item = reader.load(index)  # Original reader performs all tensor/GT/source checks.
            identity = item['identity']
            need(row['dataset_index'] == index and identity == row['identity']
                 and item['G'] == row['G'] and item['query_ids'] == row['query_ids'],
                 'Original image identity/G/query sequence must remain exact')
            image_id, annotations = identity['image_id'], identity['ann_ids']
            G = item['G']
            need(type(image_id) is int and image_id not in seen_images
                 and type(G) is int and G == len(annotations) and 1 <= G <= 100
                 and len(set(annotations)) == G and not seen_annotations.intersection(annotations),
                 'Each original image and annotation identity must occur exactly once')
            need(len(item['query_ids']) == G and len(set(item['query_ids'])) == G,
                 'All original GT-matched query ids retained')
            seen_images.add(image_id)
            seen_annotations.update(annotations)
            # These receipt fields come from the exact reviewed U37 call, which
            # checks JPEG/EXIF, geometry and finite CPU RGB400. Do not reimplement
            # N28 carrier parsing, matcher, GT reconstruction or tensor digests.
            rgb = item['rgb_receipt']
            need(rgb['image'] == identity['image'] and rgb['original_shape'] == list(identity['original_shape'])
                 and rgb['actual_img_shape'] == list(identity['actual_img_shape'])
                 and rgb['canvas_size'] == 800 and rgb['branch_size'] == 400
                 and rgb['decoded_mode'] == 'RGB' and rgb['auto_orientation'] is False
                 and rgb['teacher_photometric_reconstruction'] is False,
                 'Same admitted image and new-branch geometry receipt')
            _cold(torch)
            metadata = dict(dataset_index=index, image_id=image_id, G=G,
                            original_shape=list(identity['original_shape']),
                            actual_img_shape=list(identity['actual_img_shape']),
                            identity_sha256=metadata_sha256(identity),
                            query_order_sha256=metadata_sha256(item['query_ids']),
                            image_sha256=rgb['image']['sha256'])
            report['rows'].append(metadata)
            report['images_completed'] += 1
            report['GT_completed'] += G
            _sample(c, report, 'loaded_image', index)
            item = None  # No retained F/e/M/L/GT/RGB tensor or per-image model state.
        need(report['images_completed'] == len(seen_images) == 741
             and report['GT_completed'] == len(seen_annotations) == 1661,
             'Full original 741/1661 population must complete without skipping')
        report['reader_elapsed_seconds'] = _elapsed(reader_started)
        report['resource_candidates'] = resource_candidates(report['rows'])
        report['population_summary'] = dict(
            distinct_original_shapes=len({tuple(row['original_shape']) for row in report['rows']}),
            distinct_content_shapes=len({tuple(row['actual_img_shape']) for row in report['rows']}),
            minimum_G=min(row['G'] for row in report['rows']),
            maximum_G=max(row['G'] for row in report['rows']),
            minimum_original_area=min(math.prod(row['original_shape']) for row in report['rows']),
            maximum_original_area=max(math.prod(row['original_shape']) for row in report['rows']))
        report['validation_scope'] = dict(
            original_F_e_M_L_GT_and_matcher='Unmodified exact N28 reader.load validations for every image',
            native_L_baseline='N16 native_L carrier, selected by original GT-ordered query ids; not PR-corrected L',
            RGB='Admitted original JPEG, orientation unchanged, reviewed common RGB400 geometry',
            native100_M_or_S0_required=False, geometric_render_forward_executed=False)
        reader = None
        gc.collect()
        _sample(c, report, 'reader_released_before_initialization')
        _cold(torch)
        init_started = time.perf_counter()
        adapters, initialization = initializer.build_pair(torch,
            {key: modules[key] for key in initializer.SOURCE_SHA256},
            {key: source_refs[key] for key in initializer.SOURCE_SHA256})
        report['parameter_counts'] = _pair_metadata(torch, adapters, initialization, initializer)
        report['initialization'] = initialization  # JSON hashes/metadata, never state tensors.
        references = [weakref.ref(value) for arm in ('PLAIN', 'FULL')
                      for value in (adapters[arm], adapters[arm].backbone, adapters[arm].backbone.legacy)]
        report['initialization_elapsed_seconds'] = _elapsed(init_started)
        _sample(c, report, 'CPU_pair_constructed_and_checked')
        adapters = initialization = None
        gc.collect()
        need(all(reference() is None for reference in references), 'No CPU model objects may escape qualification')
        report['constructed_model_objects_released'] = True
        _sample(c, report, 'CPU_pair_released')
        _cold(torch)
        c.check()
        if image_control is not None and image_control is not c:
            image_control.check()
        report['status'] = 'CPU_READER_AND_COMMON_INITIALIZATION_CHECKS_COMPLETED'
        report['elapsed_seconds'] = _elapsed(started)
        json.dumps(report, allow_nan=False)  # Reject any tensor or non-JSON result escaping.
        return report
    except BaseException as error:
        report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error)),
                      elapsed_seconds=_elapsed(started))
        raise
    finally:
        item = reader = adapters = initialization = None
        references.clear()
        gc.collect()
