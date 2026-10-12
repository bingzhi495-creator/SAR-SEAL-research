"""Unbound N12 independent CPU geometry-budget diagnostic qualifier.

Original actual held/reaped RAW prefix comes first. Unchanged native exported
masks and saved exact GPU bands supply every pixel metric; CPU reconstructed M
is only a practical sign/near2 replay. It never replaces the GPU band or masks.
Independent official matching replay checks every compact fixed anchor and FN.
ROOT must separately qualify this reader's original held/reaped terminal.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
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
        len(spec['additional_failed_RAW_inputs']) == 2, 'Three original failures required')
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


def independent_pair_pixels(np, masks, reader, coco, native, B, pair):
    G = masks.decode(reader.rle_bytes(coco.annToRLE(coco.anns[pair['gt_id']]))).astype(bool)
    P = masks.decode(reader.rle_bytes(native['segmentation'])).astype(bool)
    require(G.shape == P.shape == B.shape and bool(G.any()), 'Independent nonempty GT/raster check failed')
    gi, pi, go, po = G[B], P[B], G[~B], P[~B]
    tp_i, fp_i, fn_i = int((gi & pi).sum()), int((~gi & pi).sum()), int((gi & ~pi).sum())
    tp_o, fp_o, fn_o = int((go & po).sum()), int((~go & po).sum()), int((go & ~po).sum())
    area = int(G.sum())
    tp, fp, fn = tp_i + tp_o, fp_i + fp_o, fn_i + fn_o
    require(area == tp + fn and int(P.sum()) == tp + fp and
        (tp, fp, fn) == (pair['pixel_TP'], pair['pixel_FP'], pair['pixel_FN']) and
        tp / (tp + fp + fn) == pair['IoU'], 'Independent native area/TP/FP/FN/IoU differs')
    # Separate formula: outside true positives + all GT pixels inside band.
    numerator = tp_o + int(gi.sum())
    denominator = area + fp_o
    ideal = np.where(B, G, P)
    direct_intersection = int((G & ideal).sum())
    direct_union = int((G | ideal).sum())
    require(numerator == area - fn_o == direct_intersection and
        denominator == direct_union > 0 and
        not bool(((ideal != P) & ~B).any()) and not bool(((ideal != G) & B).any()),
        'Independent direct oracle and formula disagree')
    value = numerator / denominator
    require(pair['IoU'] <= value <= 1 and (value == 1) == (fp_o == fn_o == 0),
        'Independent oracle upper-bound inequalities fail')
    return dict(GT_pixels=area, native_prediction_pixels=int(P.sum()), band_pixels=int(B.sum()),
        TP_in=tp_i, FP_in=fp_i, FN_in=fn_i, TP_out=tp_o, FP_out=fp_o, FN_out=fn_o,
        native_IoU=pair['IoU'], oracle_IoU=value,
        oracle_numerator=numerator, oracle_denominator=denominator,
        oracle_ge_0_90_exact=10 * numerator >= 9 * denominator,
        oracle_ge_0_95_exact=20 * numerator >= 19 * denominator)


def CPU_summary(common, images):
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


def CPU_regions(np, cv2, P, z):
    """Separate distance, structured lexicographic sort and integral-window BAR."""
    require(P.dtype == bool and z.dtype == np.float32 and P.shape == z.shape and
        bool(np.isfinite(z).all()) and np.array_equal(P, z > 0), 'Independent exact native P/logits required')
    absolute = np.abs(z).reshape(-1)
    flat = np.arange(z.size, dtype=np.int64)
    old = np.abs(z) <= 2.0
    K = int(np.count_nonzero(old))
    mixed = bool(P.any()) and not bool(P.all())
    if mixed:
        D = cv2.distanceTransform(P.astype(np.uint8), cv2.DIST_C, 3) + \
            cv2.distanceTransform((~P).astype(np.uint8), cv2.DIST_C, 3)
        require(bool(np.isfinite(D).all()) and bool((D >= 1).all()) and
            bool((D == np.floor(D)).all()), 'Independent finite positive Chebyshev distance required')
    else:
        D = np.full(P.shape, np.inf, dtype=np.float64)
    keys = np.empty(z.size, dtype=[('distance', np.float64), ('uncertainty', np.float32), ('flat', np.int64)])
    keys['distance'], keys['uncertainty'], keys['flat'] = D.reshape(-1), absolute, flat
    uncertainty_order = np.argsort(keys, order=['uncertainty', 'flat', 'distance'], kind='mergesort')
    check_uncertainty = np.zeros(z.size, dtype=bool)
    check_uncertainty[uncertainty_order[:K]] = True
    require(np.array_equal(check_uncertainty.reshape(P.shape), old), 'Independent uncertainty order must reproduce oldB exactly')
    order = np.argsort(keys, order=['distance', 'uncertainty', 'flat'], kind='mergesort') if mixed else uncertainty_order
    indices = np.sort(order[:K])
    geometry = np.zeros(z.size, dtype=bool)
    geometry[indices] = True
    geometry = geometry.reshape(P.shape)
    # Author width2: explicitly pad P with zero, and convolve complement of the
    # padded P (NOT a separately zero-padded complement). For binary P and k=5,
    # response is integer /25; threshold >.1 is exactly integer response >=3.
    pad = np.pad(P.astype(np.int64), ((2, 2), (2, 2)), mode='constant', constant_values=0)
    integral = np.pad(pad.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode='constant')
    window = integral[5:, 5:] - integral[:-5, 5:] - integral[5:, :-5] + integral[:-5, :-5]
    pos = (25 * P.astype(np.int64) - window) >= 3
    neg = (window - 25 * P.astype(np.int64)) >= 3
    block = np.zeros(P.shape, dtype=np.int64)
    block[pos | neg] = 1
    block[(P.astype(np.int64) - pos.astype(np.int64)) > 0] = 2
    width2 = block == 1
    require(int(geometry.sum()) == K and np.array_equal(np.flatnonzero(geometry), indices), 'Independent exactK/flat order failed')
    return dict(old_B=old, geometry_same_K=geometry, RefineMask_width2_original_image=width2), D, K, indices, not mixed


def CPU_histogram(np, masks, reader, coco, native, B, D, pair):
    G = masks.decode(reader.rle_bytes(coco.annToRLE(coco.anns[pair['gt_id']]))).astype(bool)
    P = masks.decode(reader.rle_bytes(native['segmentation'])).astype(bool)
    category = np.full(D.shape, 6, dtype=np.int64)
    finite = np.isfinite(D)
    category[finite] = 5
    for limit, code in ((16, 4), (8, 3), (4, 2), (2, 1), (1, 0)):
        category[finite & (D <= limit)] = code
    result = {}
    for error, selected in (('native_FP', P & ~G), ('native_FN', ~P & G)):
        result[error] = {side: np.bincount(category[selected & region], minlength=7).astype(np.int64).tolist()
            for side, region in (('inside', B), ('outside', ~B))}
        require(sum(result[error]['inside']) + sum(result[error]['outside']) ==
            pair['pixel_FP' if error == 'native_FP' else 'pixel_FN'], 'Independent full error distance conservation failed')
    return result


def qualify(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT CPU SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    common, raws, closure = pre_gate(spec, 36)
    gate = common.source_gate(path, spec, __file__, raws, closure, cpu=True)
    common.deadline(started)
    q_path = Path(spec['Q_path'])
    require(q_path.is_absolute() and not q_path.exists() and q_path.parent.exists(), 'Fresh absolute Q path required')
    report = read(common.bound(spec['GPU_report']))
    gpu_spec = read(common.bound(spec['GPU_spec']))
    require(gpu_spec['schema'] == common.GPU_SCHEMA and gpu_spec['RAW_inputs'] == spec['RAW_inputs'][:35] and
        all(gpu_spec[k] == spec[k] for k in common.REF_KEYS if k != 'entry_source') and
        gpu_spec['recipe'] == common.RECIPE and gpu_spec['regions_recipe'] == common.REGIONS_RECIPE and
        report['schema'] == common.REPORT_SCHEMA and report['status'] == common.REPORT_STATUS and
        report['source'] == spec['GPU_entry_source'] == gpu_spec['entry_source'] and
        report['spec'] == spec['GPU_spec'] and report['GPU_admission'] == spec['GPU_admission'] and
        report['phase'] == gpu_spec['phase'] and report['RAW_inputs'] == spec['RAW_inputs'][:35] and
        report['recipe'] == common.RECIPE and report['regions_recipe'] == common.REGIONS_RECIPE and
        report['numeric_protocol'] == common.NUMERIC and report['prior_N11_CPU_qualification'] == spec['prior_N11_CPU_qualification'] and
        report['pair_anchors'] == spec['pair_anchors'] and report['DEV'] == spec['DEV'] and
        report['native_predictions'] == spec['native_predictions'] and report['packet_refs'] == gate['packets'] and
        report['RefineMask_author_source'] == spec['RefineMask_author_source'] and
        report['cleanup']['allocated_after'] == 0 and
        all(report[k] is True for k in ('original18700_full_dictionaries_exact', 'sameK_geometry_and_uncertainty_B_exact',
            'all380_GT_FN_retained_per_slot', 'fixed_native_assignment_only', 'GPU_selected_native_L_saved_float32_exact')) and
        report['approximate_CPU_L_substituted'] is False and report['RefineMask_sameK_claimed'] is False,
        'Locked complete N12 GPU parent required')
    i = common.joined(report['source'], report['spec'], report['GPU_admission'], report['phase'], '0', spec['RAW_inputs'], gate['owners'])
    require(i == 35 and spec['RAW_inputs'][35]['receipt'] == spec['GPU_RAW'] and
        all(ref in closure.values() for ref in [report['start'], report['native_export_replay'], report['coverage']] + report['selected_GPU_logits']),
        'Actual held0 N12 GPU final ancestor/output closure required')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1], 'CPU01 required before numerical imports')
    import torch
    import numpy as np
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5' and
        cv2.__version__ == '4.8.0' and not torch.cuda.is_initialized(), 'Qualified CPU versions/cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    libraries = common.imported_files(psutil, torch, np, COCO, COCOeval, masks, gate, cv2=cv2)
    rng = torch.random.get_rng_state().clone()
    reader = common.load(spec['reader_source'], '_N12_original_population_helpers')
    renderer = common.load(spec['renderer_source'], '_N12_original_CPU_renderer')
    coco = COCO(str(common.bound(spec['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and ids == [r['id'] for r in coco.dataset['images']] and
        len(coco.anns) == 380 and set(coco.cats) == {1}, 'Canonical DEV187/380 required')
    native = read(common.bound(spec['native_predictions']))
    replay = read(common.bound(report['native_export_replay']))
    require(replay == native and len(replay) == 18700, 'Original18700 all100 dictionaries exact prerequisite differs')
    grouped = reader.population(coco, replay, ids)
    anchors = common.validate_anchors(coco, ids, gate['anchors'], gate['q'])
    coverage = read(common.bound(report['coverage']))
    require(coverage['schema'] == 'N12-native-regions-details-v1' and coverage['pair_anchors'] == spec['pair_anchors'], 'Coverage reference identity differs')
    images = coverage['images']
    rows = read(common.bound(gate['prior']['image_rows']))
    old_images = read(common.bound(gate['n11']['coverage']))['images']
    require(len(images) == len(rows) == len(old_images) == len(report['selected_GPU_logits']) == 187 and
        [r['image_id'] for r in images] == ids, 'Full original image population differs')
    factors, factor_proof = reader.factor_check(np, read(common.bound(gate['prior']['native_query_score_factors'])), replay, ids)
    checks = []
    with torch.no_grad():
        for index, (row, ref, image, anchor, old_image) in enumerate(zip(rows, gate['packets'], images, anchors, old_images)):
            common.deadline(started)
            packet = torch.load(str(common.bound(ref)), map_location='cpu')
            fields, identity, qids = common.packet_check(torch, reader, coco, packet, row, index, ref, gate)
            require(image['dataset_index'] == index and image['identity'] == identity and image['packet'] == ref and
                image['native_query_order'] == qids and image['natural_band_counts'] == row['band_counts_all100'] and
                image['native100_dictionaries_exact'] is True, 'Per-image original native identity differs')
            selected = sorted({r['pair']['query_id'] for s in anchor['threshold_slots'] for r in s['GT'] if r['pair']})
            require(image['selected_query_ids'] == selected and set(image['query_regions']) == {str(q) for q in selected} and
                image['selected_GPU_logits'] == report['selected_GPU_logits'][index], 'Exact diagnostic selected queries/logit closure differs')
            saved_packet = torch.load(str(common.bound(image['selected_GPU_logits'])), map_location='cpu')
            saved = saved_packet['logits']
            require(saved_packet['schema'] == 'N12-selected-original-GPU-logits-v1' and
                saved_packet['dataset_index'] == index and saved_packet['identity'] == identity and
                saved_packet['packet'] == ref and saved_packet['query_ids'] == selected and
                saved.device.type == 'cpu' and saved.dtype == torch.float32 and
                tuple(saved.shape) == (len(selected),) + tuple(identity['original_shape']) and
                saved.is_contiguous() and not saved.requires_grad and bool(torch.isfinite(saved).all()), 'Exact saved GPU L ABI differs')
            CPU_L = renderer.render_original(torch, fields['M200'], identity)
            require(CPU_L.dtype == torch.float32 and CPU_L.device.type == 'cpu' and
                tuple(CPU_L.shape) == (100,) + tuple(identity['original_shape']) and bool(torch.isfinite(CPU_L).all()), 'CPU natural100 renderer differs')
            probabilities = fields['final_class_logits'][0].softmax(-1)[:, 0]
            by_query = {r['query_id']: r for r in grouped[row['image_id']]}
            query_checks, independent_regions, independent_D = [], {}, {}
            for ordinal, q in enumerate(qids):
                z = CPU_L[q]
                P = masks.decode(reader.rle_bytes(by_query[q]['segmentation'])).astype(bool)
                P_tensor = torch.from_numpy(np.ascontiguousarray(P))
                tau = 1e-4 + 1e-4 * z.abs()
                sign_flips = P_tensor != (z > 0)
                near_zero, near_band = z.abs() <= tau, (z.abs() - 2).abs() <= tau
                require(not bool((sign_flips & ~near_zero).any()) and
                    abs(int((z.abs() <= 2).sum()) - row['band_counts_all100'][q]) <= int(near_band.sum()),
                    'Practical CPU native support/band allowance exceeded')
                factor = factors[row['image_id'], q]
                probability = float(probabilities[q])
                quality = float((z.sigmoid() * P_tensor).sum() / (P_tensor.sum() + 1e-6))
                require(math.isclose(probability, factor['class_probability'], rel_tol=1e-6, abs_tol=1e-6) and
                    math.isclose(quality, factor['mask_quality'], rel_tol=1e-4, abs_tol=1e-4), 'Practical CPU class/quality differs')
                selected_band_flips = None
                if q in selected:
                    local = selected.index(q)
                    original_z = saved[local].numpy()
                    regions, D, K, chosen, fallback = CPU_regions(np, cv2, P, original_z)
                    actual = image['query_regions'][str(q)]
                    require(actual['query_id'] == q and actual['K'] == K == row['band_counts_all100'][q] and
                        actual['native_P'] == by_query[q]['segmentation'] and
                        actual['geometry_selected_flat_indices'] == chosen.tolist() and
                        actual['no_opposite_fallback'] == fallback and actual['uncertainty_same_K_exactly_old_B'] is True and
                        set(actual['region_rles']) == set(actual['region_pixels']) == set(common.REGIONS), 'Independent selection/order/K/P differs')
                    for name in common.REGIONS:
                        exported = masks.decode(reader.rle_bytes(actual['region_rles'][name])).astype(bool)
                        require(np.array_equal(exported, regions[name]) and int(exported.sum()) == actual['region_pixels'][name],
                            'Independent full region raster/count differs: ' + name)
                    require(actual['region_rles']['old_B'] == old_image['band_rles'][str(q)], 'Qualified old exactGPU band changed')
                    diff = torch.from_numpy(np.ascontiguousarray(regions['old_B'])) != (z.abs() <= 2)
                    require(not bool((diff & ~near_band).any()), 'Exact savedL band differs outside practical replay allowance')
                    selected_band_flips = int(diff.sum())
                    independent_regions[q], independent_D[q] = regions, D
                query_checks.append(dict(query_id=q, native_export_ordinal=ordinal, CPU_sign_flips=int(sign_flips.sum()),
                    practical_near_zero=int(near_zero.sum()), practical_near_band=int(near_band.sum()),
                    selected_exact_GPU_band_CPU_replay_flips=selected_band_flips))
            require([s['index'] for s in image['threshold_slots']] == [0, 5, 8], 'Original fixed slots differ')
            for observed_slot, fixed_slot in zip(image['threshold_slots'], anchor['threshold_slots']):
                require(all(observed_slot[k] == fixed_slot[k] for k in ('index', 'label', 'threshold')) and
                    len(observed_slot['GT']) == len(fixed_slot['GT']), 'Original threshold/GT population differs')
                for actual, fixed in zip(observed_slot['GT'], fixed_slot['GT']):
                    require(set(actual) == set(fixed) | {'regions', 'error_distance_histogram'} and
                        {k: actual[k] for k in fixed} == fixed, 'Original every GT/FN/ignore/query pair changed')
                    if not fixed['pair']:
                        require(actual['regions'] is None and actual['error_distance_histogram'] is None, 'Native FN must not acquire query/oracle')
                    else:
                        pair = fixed['pair']
                        q = pair['query_id']
                        for name in common.REGIONS:
                            require(independent_pair_pixels(np, masks, reader, coco, by_query[q], independent_regions[q][name], pair) ==
                                actual['regions'][name], 'Independent six pixel counts/direct oracle differs')
                        require(CPU_histogram(np, masks, reader, coco, by_query[q], independent_regions[q]['old_B'], independent_D[q], pair) ==
                            actual['error_distance_histogram'], 'Independent full error-distance histogram differs')
            checks.append(dict(dataset_index=index, image_id=row['image_id'], selected_unique_queries=len(selected),
                exact_saved_GPU_L_used_for_selection=True, independent_OpenCV_distance_and_integral_BAR=True, natural100_checks=query_checks))
            del packet, fields, saved_packet, saved, CPU_L, independent_regions, independent_D
            common.deadline(started)
    summaries, image_totals, hypothesis_gate = CPU_summary(common, images)
    require(summaries == report['summaries'] and image_totals == coverage['image_slot_region_totals'] and
        hypothesis_gate == report['H_region_gate'], 'Independent every image/slot aggregate and hypothesis rule differs')
    require(torch.equal(rng, torch.random.get_rng_state()) and not torch.cuda.is_initialized() and
        'mmdet' not in sys.modules and psutil.Process().cpu_affinity() == [0, 1], 'CPU changed RNG/resources or loaded teacher/CUDA')
    common.final_closure(spec, path, gate, __file__, libraries,
        lambda: common.imported_files(psutil, torch, np, COCO, COCOeval, masks, gate, cv2=cv2))
    common.deadline(started)
    q = dict(schema=common.Q_SCHEMA, status=common.Q_STATUS, phase=spec['phase'], source=spec['GPU_entry_source'],
        reader_source=gate['source'], common_source=spec['common_source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'],
        GPU_spec=spec['GPU_spec'], GPU_report=spec['GPU_report'], GPU_RAW=spec['GPU_RAW'], GPU_admission=spec['GPU_admission'],
        RAW_inputs=spec['RAW_inputs'], prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'], terminal_identity_pins=spec['terminal_identity_pins'],
        prior_N11_CPU_qualification=spec['prior_N11_CPU_qualification'], pair_anchors=spec['pair_anchors'],
        DEV=spec['DEV'], native_predictions=spec['native_predictions'], packet_refs=gate['packets'],
        selected_GPU_logits=report['selected_GPU_logits'], coverage=report['coverage'], native_export_replay=report['native_export_replay'],
        recipe=common.RECIPE, regions_recipe=common.REGIONS_RECIPE, numeric_protocol=common.NUMERIC,
        images=187, raw_GT=380, predictions=18700, matched_pairs_per_slot=[374, 348, 107], native_FN_per_slot=[6, 32, 273],
        summaries=summaries, H_region_gate=hypothesis_gate, CPU_field_and_region_checks=checks, native_factor_check=factor_proof,
        actual_imported_files=libraries, RefineMask_author_source=spec['RefineMask_author_source'],
        original18700_export_unchanged=True, qualified_original_matching_chain_reused=True,
        COCO12_or_matching_rerun=False, independently_selected_region_rasters_counts_and_histograms_exact=True,
        independent_direct_oracle_pixel_and_image_slot_aggregate_checks=True,
        exact_saved_GPU_L_used=True, practical_CPU_native_renderer_replayed=True,
        approximate_CPU_L_or_P_B_K_substituted=False, certified_GPU_error_bound=False,
        CPU_GPU_bitwise_render_equality_claimed=False, actual_efficacy_established=False, innovation_established=False,
        native_model_loaded=False, native_forward_calls=0, original_JPEG_decoded=False,
        heads_loaded=False, head_forward_calls=0, optimizer_created=False, backward_calls=0, weights_updated=False,
        no_GT_directed_inference=True, no_TEST=True, current_owned_CPU_terminal_self_certified=False,
        whole_leaf_VRAM_certified=False, CPU_affinity=[0, 1], threads=2, CUDA_initialized=False,
        elapsed_seconds=time.monotonic() - started, deadline_seconds=1800, automatic_retry=False)
    gc.collect()
    common.write(q_path, q)
    print(json.dumps(dict(status=common.Q_STATUS, Q=common.reference(q_path))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Explicit ROOT-owned single execution required')
    qualify(args.spec)


