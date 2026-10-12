"""Unbound N12 geometry-budget native-region coverage SOURCE; ROOT alone admits it.

Diagnostic only: no teacher, heads, optimizer, training, JPEG decode or TEST.
Original all100 natural rendering/reordering/full100 quality/S0 must reproduce
all18700 archived prediction dictionaries exactly before post-inference GT use.
No oracle masks are exported as predictions. A separate CPU reader qualifies
pixels conditional on saved exact GPU band RLEs and original fixed assignments.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import types


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def pre_gate(spec, count):
    """Only original RAW and stdlib first; every terminal gate precedes SHA/import."""
    keys = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
    rows = spec['RAW_inputs']
    require(type(rows) is list and len(rows) == count and
        all(type(row) is dict and set(keys) <= set(row) for row in rows),
        'Complete original normal RAW prefix required')
    failed = [dict(receipt=spec['prior_failed_CPU_receipt'],
        phase=Path(spec['prior_failed_CPU_receipt']['path']).parent.name, returncode=1)]
    require(type(spec['additional_failed_RAW_inputs']) is list and
        len(spec['additional_failed_RAW_inputs']) == 3, 'Four original failures required')
    for row in spec['additional_failed_RAW_inputs']:
        require(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
            type(row['returncode']) is int and row['returncode'] == 1, 'Explicit failed1 required')
        failed.append(row)
    all_rows = [dict(row, returncode=0) for row in rows] + failed
    locators = [row['receipt']['path'] for row in all_rows]
    pins = spec['terminal_identity_pins']
    require(len(set(locators)) == len(locators) and type(pins) is dict and
        set(pins) == set(locators), 'Unique complete original identity/admission pins required')
    raws = []
    for row in all_rows:
        ref, phase, code = row['receipt'], row['phase'], row['returncode']
        path = Path(ref['path'])
        require(path.is_absolute() and path.name == 'PARENT_RECEIPT.json' and
            path.parent.name == phase, 'Explicit original RAW/phase locator required')
        raw = read(path)
        pin = pins[str(path)]
        require(set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and pin['returncode'] == code and
            type(pin['original_identity']) is dict and pin['original_identity'] and
            raw['original_identity'] == pin['original_identity'] and
            raw['admission_sha256'] == pin['admission_sha256'] and
            raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and
            raw['returncode'] == code and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and
            raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual original held/reaped identity-specific terminal required before SHA/import')
        if 'phase' in raw:
            require(raw['phase'] == phase, 'Original RAW phase differs')
        peak = raw['whole_GPU_observed_peak_MiB']
        require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Original RAW resource scope differs')
        raws.append(raw)
    refs = spec['sources']
    require(type(refs) is list and refs, 'Complete source/input closure required')
    closure = {}
    for ref in refs:
        require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef') and
            ref['path'] not in closure, 'Unique absolute byte/SHA references required')
        closure[ref['path']] = ref
    require(all(row['receipt'] in closure.values() for row in all_rows), 'Original RAW refs absent from closure')
    for ref in refs:
        digest = hashlib.sha256()
        with Path(ref['path']).open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                digest.update(block)
        require(Path(ref['path']).stat().st_size == ref['bytes'] and digest.hexdigest() == ref['sha256'],
            'Bound source/input bytes changed: ' + ref['path'])
    common_ref = spec['common_source']
    require(common_ref in closure.values(), 'Actually bound common SOURCE required')
    module = types.ModuleType('_N12_provenance_common')
    module.__file__ = common_ref['path']
    exec(compile(Path(common_ref['path']).read_bytes(), common_ref['path'], 'exec'), module.__dict__)
    return module, raws[:count], closure


def original_export_kernels(common, ref):
    path = common.bound(ref)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export_original']
    require(len(functions) == 1, 'Original official exporter function required')
    fragments = ('probabilities = torch.nn.functional.softmax(classes[0], dim=-1)[:, :-1]',
        'top, indices = probabilities.flatten(0, 1).topk(100, sorted=False)',
        'qids = indices // fusion.num_classes', 'binary = ordered > 0',
        'quality = (ordered.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)',
        'scores = top * quality')
    assignments = []
    for expression in fragments:
        expected = ast.dump(ast.parse(expression).body[0], include_attributes=False)
        matches = [n for n in functions[0].body if isinstance(n, ast.Assign) and
            ast.dump(n, include_attributes=False) == expected]
        require(len(matches) == 1, 'Original exporter expression ABI differs')
        assignments.append(copy.deepcopy(matches[0]))
    module = ast.parse('def order(torch, classes, fusion):\n    pass\ndef quality(torch, ordered, top):\n    pass\n')
    module.body[0].body = assignments[:3] + ast.parse('return top, qids').body
    module.body[1].body = assignments[3:] + ast.parse('return binary, quality, scores').body
    ast.fix_missing_locations(module)
    scope = {}
    exec(compile(module, str(path), 'exec'), scope)
    return types.SimpleNamespace(order=scope['order'], quality=scope['quality'])


def rle(np, masks, binary):
    encoded = masks.encode(np.asfortranarray(binary.astype(np.uint8)))
    return dict(size=list(encoded['size']), counts=encoded['counts'].decode('ascii'))


def gpu_pair_pixels(np, masks, reader, coco, native, band, pair):
    G = masks.decode(reader.rle_bytes(coco.annToRLE(coco.anns[pair['gt_id']]))).astype(bool)
    P = masks.decode(reader.rle_bytes(native['segmentation'])).astype(bool)
    require(G.shape == P.shape == band.shape and bool(G.any()), 'Nonempty same-raster GT required')
    counts = {}
    for suffix, region in (('in', band), ('out', ~band)):
        counts['TP_' + suffix] = int(np.count_nonzero(G & P & region))
        counts['FP_' + suffix] = int(np.count_nonzero(~G & P & region))
        counts['FN_' + suffix] = int(np.count_nonzero(G & ~P & region))
    A = int(G.sum())
    tp, fp, fn = (counts[k + '_in'] + counts[k + '_out'] for k in ('TP', 'FP', 'FN'))
    require((tp, fp, fn) == (pair['pixel_TP'], pair['pixel_FP'], pair['pixel_FN']) and
        A == tp + fn and int(P.sum()) == tp + fp and
        tp / (tp + fp + fn) == pair['IoU'], 'Fixed original native matched pixel totals changed')
    oracle = (P & ~band) | (G & band)
    numerator, denominator = A - counts['FN_out'], A + counts['FP_out']
    actual_tp = int(np.count_nonzero(G & oracle))
    actual_union = int(np.count_nonzero(G | oracle))
    require(numerator == actual_tp and denominator == actual_union and denominator > 0 and
        not bool(np.any((oracle != P) & ~band)) and
        not bool(np.any((oracle != G) & band)), 'Direct oracle raster/formula identities differ')
    value = numerator / denominator
    require(pair['IoU'] <= value <= 1 and
        (value == 1) == (counts['FN_out'] == counts['FP_out'] == 0), 'Fixed-pair oracle inequalities fail')
    return dict(GT_pixels=A, native_prediction_pixels=int(P.sum()), band_pixels=int(band.sum()),
        **counts, native_IoU=pair['IoU'], oracle_IoU=value,
        oracle_numerator=numerator, oracle_denominator=denominator,
        oracle_ge_0_90_exact=10 * counts['FN_out'] + 9 * counts['FP_out'] <= A,
        oracle_ge_0_95_exact=20 * counts['FN_out'] + 19 * counts['FP_out'] <= A)


def author_boundary_function(common, torch, ref):
    """Execute only the pinned original author's function, with its exact order."""
    path = common.bound(ref)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_block_target']
    require(len(functions) == 1 and not functions[0].decorator_list, 'Pinned author boundary function required')
    module = ast.Module(body=[copy.deepcopy(functions[0])], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(torch=torch, F=torch.nn.functional)
    exec(compile(module, str(path), 'exec'), scope)
    return scope['generate_block_target']


def GPU_leaf_CPU_regions(np, ndimage, torch, author, P, z):
    """No GT input. Exact saved/native GPU logits; selected fixed queries only."""
    require(P.dtype == bool and z.dtype == np.float32 and P.shape == z.shape and
        bool(np.isfinite(z).all()) and np.array_equal(P, z > 0), 'Exact original P/logit input required')
    flat = np.arange(z.size, dtype=np.int64)
    absolute = np.abs(z.reshape(-1))
    B = np.abs(z) <= 2.0
    K = int(B.sum())
    uncertainty_order = np.lexsort((flat, absolute))
    uncertainty = np.zeros(z.size, dtype=bool)
    uncertainty[uncertainty_order[:K]] = True
    require(np.array_equal(uncertainty.reshape(z.shape), B), 'Old B and sameK uncertainty must be the exact same set')
    no_opposite = not bool(P.any()) or bool(P.all())
    if no_opposite:
        D = np.full(P.shape, np.inf, dtype=np.float64)
        order = uncertainty_order
    else:
        D = ndimage.distance_transform_cdt(P, metric='chessboard') + \
            ndimage.distance_transform_cdt(~P, metric='chessboard')
        require(D.shape == P.shape and bool((D >= 1).all()) and
            bool(np.isfinite(D).all()) and bool((D == np.floor(D)).all()),
            'Finite positive full-domain chessboard distances required')
        order = np.lexsort((flat, absolute, D.reshape(-1)))
    chosen = np.sort(order[:K])
    geometry = np.zeros(z.size, dtype=bool)
    geometry[chosen] = True
    geometry = geometry.reshape(P.shape)
    block = author(torch.from_numpy(P.astype(np.float32))[None], boundary_width=2)
    require(block.device.type == 'cpu' and tuple(block.shape) == (1,) + P.shape and
        bool(((block >= 0) & (block <= 2)).all()), 'Original author block-target ABI differs')
    width2 = block[0].numpy() == 1
    require(int(geometry.sum()) == K and np.array_equal(np.flatnonzero(geometry), chosen),
        'Exact per-query K and ascending-flat geometry indices required')
    return dict(old_B=B, geometry_same_K=geometry,
        RefineMask_width2_original_image=width2), D, K, chosen, no_opposite


def error_distance_histogram(np, masks, reader, coco, native, B, D, pair):
    G = masks.decode(reader.rle_bytes(coco.annToRLE(coco.anns[pair['gt_id']]))).astype(bool)
    P = masks.decode(reader.rle_bytes(native['segmentation'])).astype(bool)
    bins = [D <= 1, (D > 1) & (D <= 2), (D > 2) & (D <= 4),
        (D > 4) & (D <= 8), (D > 8) & (D <= 16), (D > 16) & np.isfinite(D), ~np.isfinite(D)]
    require(bool((np.sum(np.stack(bins), axis=0) == 1).all()), 'Distance histogram must partition domain')
    result = {}
    for name, error in (('native_FP', ~G & P), ('native_FN', G & ~P)):
        result[name] = {side: [int(np.count_nonzero(error & region & bin_mask)) for bin_mask in bins]
            for side, region in (('inside', B), ('outside', ~B))}
        expected = pair['pixel_FP' if name == 'native_FP' else 'pixel_FN']
        require(sum(result[name]['inside']) + sum(result[name]['outside']) == expected,
            'Full native FP/FN distance histogram conservation failed')
    return result


def summarize_regions(common, images):
    result, image_totals = [], []
    keys = ('GT_pixels', 'native_prediction_pixels', 'band_pixels', 'TP_in', 'FP_in', 'FN_in', 'TP_out', 'FP_out', 'FN_out')
    for position, (index, label, matched, fn) in enumerate(common.SLOTS):
        rows = [r for im in images for r in im['threshold_slots'][position]['GT']]
        require(len(rows) == 380 and sum(r['pair'] is not None for r in rows) == matched and
            sum(r['FN'] for r in rows) == fn, 'Threshold-specific original cohort conservation differs')
        regions = {}
        for name in common.REGIONS:
            pixels = [r['regions'][name] for r in rows if r['pair']]
            totals = {k: sum(p[k] for p in pixels) for k in keys}
            editable = totals['FP_in'] + totals['FN_in']
            errors = editable + totals['FP_out'] + totals['FN_out']
            regions[name] = dict(matched_pair_pixel_totals=totals,
                editable_error_fraction=editable / errors if errors else None,
                mean_native_IoU=math.fsum(p['native_IoU'] for p in pixels) / matched,
                mean_oracle_IoU=math.fsum(p['oracle_IoU'] for p in pixels) / matched,
                oracle_ge_0_90_fixed_pairs=sum(p['oracle_ge_0_90_exact'] for p in pixels),
                oracle_ge_0_95_fixed_pairs=sum(p['oracle_ge_0_95_exact'] for p in pixels))
        hist = {error: {side: [sum(r['error_distance_histogram'][error][side][bi] for r in rows if r['pair'])
            for bi in range(7)] for side in ('inside', 'outside')} for error in ('native_FP', 'native_FN')}
        result.append(dict(index=index, label=label, threshold=images[0]['threshold_slots'][position]['threshold'],
            raw_GT=380, native_matched_pairs=matched, native_FN=fn, ignored_GT=sum(r['ignored'] for r in rows),
            regions=regions, error_distance_histogram=hist, no_cross_slot_pooling=True,
            fixed_native_FN_have_no_selected_query=True))
        for image in images:
            selected = [r for r in image['threshold_slots'][position]['GT'] if r['pair']]
            image_totals.append(dict(image_id=image['image_id'], slot_index=index,
                matched_pairs=len(selected), regions={name: {k: sum(r['regions'][name][k] for r in selected)
                    for k in keys} for name in common.REGIONS}))
    canonical = result[0]
    old, geo = canonical['regions']['old_B'], canonical['regions']['geometry_same_K']
    gate = dict(canonical_assignment_index=0, denominator_original_fixed_pairs=374,
        native_FN_without_selected_query=6,
        geometry_minus_old_mean_oracle_IoU=geo['mean_oracle_IoU'] - old['mean_oracle_IoU'],
        geometry_minus_old_editable_error_fraction=geo['editable_error_fraction'] - old['editable_error_fraction'],
        geometry_mean_oracle_strictly_better=geo['mean_oracle_IoU'] > old['mean_oracle_IoU'],
        geometry_editable_error_fraction_strictly_better=geo['editable_error_fraction'] > old['editable_error_fraction'],
        H_region_diagnostic_supported=geo['mean_oracle_IoU'] > old['mean_oracle_IoU'] and
            geo['editable_error_fraction'] > old['editable_error_fraction'],
        model_efficacy_or_innovation_established=False, all380_GT_or_AP_ceiling=False)
    return result, image_totals, gate


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    common, raws, closure = pre_gate(spec, 35)
    gate = common.source_gate(path, spec, __file__, raws, closure)
    common.deadline(started)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists(), 'Fresh absolute output directory required')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1], 'CPU01 required before numerical imports')
    import torch
    import numpy as np
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5' and
        scipy.__version__ == '1.10.1' and not torch.cuda.is_initialized(), 'Qualified versions/cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    libraries = common.imported_files(psutil, torch, np, COCO, COCOeval, masks, gate, scipy=scipy)
    reader = common.load(spec['reader_source'], '_N12_qualified_native_helpers')
    renderer = common.load(spec['renderer_source'], '_N12_original_renderer')
    author = author_boundary_function(common, torch, spec['RefineMask_author_source'])
    kernels = original_export_kernels(common, spec['export_source'])
    coco = COCO(str(common.bound(spec['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and ids == [r['id'] for r in coco.dataset['images']] and
        len(coco.anns) == 380 and set(coco.cats) == {1}, 'Canonical DEV187/380 required')
    native = read(common.bound(spec['native_predictions']))
    grouped = reader.population(coco, native, ids)
    anchors = common.validate_anchors(coco, ids, gate['anchors'], gate['q'])
    rows = read(common.bound(gate['prior']['image_rows']))
    old_coverage = read(common.bound(gate['n11']['coverage']))['images']
    require(len(rows) == len(old_coverage) == 187 and [r['image_id'] for r in rows] == ids,
        'Canonical original image/qualified old-band population differs')
    require(not torch.cuda.is_initialized(), 'All source/library gates precede CUDA')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024**2 / torch.cuda.get_device_properties(0).total_memory), 0)
    require(torch.cuda.memory_allocated(0) == 0, 'Clean GPU allocator required')
    torch.cuda.reset_peak_memory_stats(0)
    output.mkdir()
    common.write(output / 'START.json', dict(schema='N12-native-regions-start-v1', phase=spec['phase'],
        source=gate['source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'], recipe=common.RECIPE,
        pair_anchors=spec['pair_anchors'], prior_N11_CPU_qualification=spec['prior_N11_CPU_qualification'],
        actual_imported_files=libraries, RefineMask_author_source=spec['RefineMask_author_source']))
    replay, images, logits_refs, baseline_seconds, selection_seconds = [], [], [], 0., 0.
    with torch.no_grad():
        for index, (row, ref, anchor, old_image) in enumerate(zip(rows, gate['packets'], anchors, old_coverage)):
            common.deadline(started)
            start_baseline = time.monotonic()
            packet = torch.load(str(common.bound(ref)), map_location='cpu')
            fields, identity, archived_qids = common.packet_check(torch, reader, coco, packet, row, index, ref, gate)
            M, C = fields['M200'].to('cuda:0'), fields['final_class_logits'].to('cuda:0')
            top, qids = kernels.order(torch, C, types.SimpleNamespace(num_classes=1))
            require(qids.cpu().tolist() == archived_qids == [r['query_id'] for r in grouped[row['image_id']]], 'Exact query order required')
            L = renderer.render_original(torch, M, identity)
            require(L.dtype == torch.float32 and L.device.type == 'cuda' and
                tuple(L.shape) == (100,) + tuple(identity['original_shape']) and bool(torch.isfinite(L).all()), 'Natural100 render ABI differs')
            ordered = L[qids]
            binary, quality, scores = kernels.quality(torch, ordered, top)
            masks_cpu, scores_cpu = binary.cpu().numpy(), scores.cpu().tolist()
            baseline = [dict(image_id=row['image_id'], query_id=q, category_id=1, score=float(scores_cpu[o]),
                segmentation=rle(np, masks, masks_cpu[o])) for o, q in enumerate(archived_qids)]
            require(baseline == grouped[row['image_id']], 'Original100 complete dictionaries must reproduce exactly')
            counts = (L.abs() <= 2.0).flatten(1).sum(1).cpu().tolist()
            require(counts == row['band_counts_all100'], 'Original all100 band counts must reproduce exactly')
            replay.extend(baseline)
            elapsed_baseline = time.monotonic() - start_baseline
            baseline_seconds += elapsed_baseline
            selected = sorted({r['pair']['query_id'] for s in anchor['threshold_slots'] for r in s['GT'] if r['pair']})
            saved = L[selected].detach().cpu().contiguous().clone()
            require(saved.dtype == torch.float32 and not saved.requires_grad and bool(torch.isfinite(saved).all()), 'Exact selected FP32 logits required')
            logit_path = output / ('NATIVE_L_%03d.pt' % index)
            torch.save(dict(schema='N12-selected-original-GPU-logits-v1', dataset_index=index, identity=identity,
                packet=ref, query_ids=selected, logits=saved), str(logit_path))
            logit_ref = common.reference(logit_path)
            logits_refs.append(logit_ref)
            by_query = {r['query_id']: r for r in baseline}
            region_arrays, distances, query_diagnostics = {}, {}, {}
            for local, q in enumerate(selected):
                started_selection = time.monotonic()
                P = masks.decode(reader.rle_bytes(by_query[q]['segmentation'])).astype(bool)
                z = saved[local].numpy()
                regions, D, K, chosen, fallback = GPU_leaf_CPU_regions(np, ndimage, torch, author, P, z)
                encoded = {name: rle(np, masks, region) for name, region in regions.items()}
                require(K == counts[q] and encoded['old_B'] == old_image['band_rles'][str(q)], 'Qualified exact old-band identity changed')
                elapsed_selection = time.monotonic() - started_selection
                selection_seconds += elapsed_selection
                region_arrays[q], distances[q] = regions, D
                query_diagnostics[str(q)] = dict(query_id=q, K=K, native_P=rle(np, masks, P), region_rles=encoded,
                    region_pixels={name: int(region.sum()) for name, region in regions.items()},
                    geometry_selected_flat_indices=chosen.tolist(), no_opposite_fallback=fallback,
                    uncertainty_same_K_exactly_old_B=True, geometry_distance_not_truncated=True,
                    CPU_selection_seconds=elapsed_selection, saved_logit_bytes=int(z.nbytes))
            slots = []
            for slot in anchor['threshold_slots']:
                records = []
                for fixed in slot['GT']:
                    record = copy.deepcopy(fixed)
                    pair = fixed['pair']
                    if pair:
                        q = pair['query_id']
                        record['regions'] = {name: gpu_pair_pixels(np, masks, reader, coco, by_query[q], region_arrays[q][name], pair)
                            for name in common.REGIONS}
                        record['error_distance_histogram'] = error_distance_histogram(np, masks, reader, coco, by_query[q],
                            region_arrays[q]['old_B'], distances[q], pair)
                    else:
                        record['regions'] = record['error_distance_histogram'] = None
                    records.append(record)
                slots.append(dict(index=slot['index'], label=slot['label'], threshold=slot['threshold'], GT=records))
            images.append(dict(dataset_index=index, image_id=row['image_id'], identity=identity, packet=ref,
                native_query_order=archived_qids, natural_band_counts=counts, selected_query_ids=selected,
                selected_GPU_logits=logit_ref, query_regions=query_diagnostics, threshold_slots=slots,
                native100_dictionaries_exact=True, baseline_seconds=elapsed_baseline,
                saved_selected_logit_bytes=int(saved.numel() * saved.element_size())))
            M = C = top = qids = L = ordered = binary = quality = scores = None
            del packet, fields, saved, region_arrays, distances
            common.deadline(started)
    require(replay == native and len(replay) == 18700, 'Complete original18700 exact replay required')
    summaries, image_totals, hypothesis_gate = summarize_regions(common, images)
    common.write(output / 'NATIVE_EXPORT_REPLAY.json', replay)
    common.write(output / 'COVERAGE.json', dict(schema='N12-native-regions-details-v1',
        pair_anchors=spec['pair_anchors'], images=images, image_slot_region_totals=image_totals))
    gc.collect()
    torch.cuda.synchronize()
    cleanup = dict(allocated_after=int(torch.cuda.memory_allocated(0)), allocator_peak_bytes=int(torch.cuda.max_memory_allocated(0)))
    require(cleanup['allocated_after'] == 0 and cleanup['allocator_peak_bytes'] <= 4096 * 1024**2, 'GPU cleanup/budget differs')
    common.final_closure(spec, path, gate, __file__, libraries,
        lambda: common.imported_files(psutil, torch, np, COCO, COCOeval, masks, gate, scipy=scipy))
    common.deadline(started)
    report = dict(schema=common.REPORT_SCHEMA, status=common.REPORT_STATUS, phase=spec['phase'],
        source=gate['source'], common_source=spec['common_source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'],
        recipe=common.RECIPE, regions_recipe=common.REGIONS_RECIPE, numeric_protocol=common.NUMERIC,
        RAW_inputs=spec['RAW_inputs'], prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'], terminal_identity_pins=spec['terminal_identity_pins'],
        prior_N11_CPU_qualification=spec['prior_N11_CPU_qualification'], pair_anchors=spec['pair_anchors'],
        DEV=spec['DEV'], native_predictions=spec['native_predictions'], packet_refs=gate['packets'],
        start=common.reference(output / 'START.json'), native_export_replay=common.reference(output / 'NATIVE_EXPORT_REPLAY.json'),
        coverage=common.reference(output / 'COVERAGE.json'), selected_GPU_logits=logits_refs,
        summaries=summaries, H_region_gate=hypothesis_gate, cleanup=cleanup, actual_imported_files=libraries,
        original18700_full_dictionaries_exact=True, sameK_geometry_and_uncertainty_B_exact=True,
        all380_GT_FN_retained_per_slot=True, fixed_native_assignment_only=True,
        prior_official_matching_reused_without_new_COCO_or_assignment=True,
        GPU_selected_native_L_saved_float32_exact=True, approximate_CPU_L_substituted=False,
        RefineMask_author_source=spec['RefineMask_author_source'], RefineMask_sameK_claimed=False,
        baseline_seconds=baseline_seconds, selected_query_CPU_selection_seconds=selection_seconds,
        saved_selected_native_L_bytes=sum(i['saved_selected_logit_bytes'] for i in images),
        native_model_loaded=False, native_forward_calls=0, original_JPEG_decoded=False,
        heads_loaded=False, head_forward_calls=0, optimizer_created=False, backward_calls=0, weights_updated=False,
        no_GT_directed_inference=True, no_TEST=True, actual_efficacy_established=False, innovation_established=False,
        current_owned_GPU_terminal_self_certified=False, whole_leaf_VRAM_certified=False,
        CPU_affinity=[0, 1], threads=2, elapsed_seconds=time.monotonic() - started, deadline_seconds=1800, automatic_retry=False)
    common.write(output / 'REPORT.json', report)
    print(json.dumps(dict(status=common.REPORT_STATUS, report=common.reference(output / 'REPORT.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Explicit ROOT-owned single execution required')
    run(args.spec)


