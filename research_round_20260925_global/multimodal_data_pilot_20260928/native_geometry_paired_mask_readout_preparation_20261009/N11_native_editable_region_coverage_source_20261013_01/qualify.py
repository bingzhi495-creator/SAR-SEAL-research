"""Unbound N11 independent CPU native-band diagnostic qualifier.

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
    module = types.ModuleType('_N11_provenance_common')
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


def independent_summary(common, images):
    result = []
    for position, (index, label, tp, fn) in enumerate(common.SLOTS):
        records = []
        for image in images:
            records.extend(image['threshold_slots'][position]['GT'])
        eligible = [r['pixels'] for r in records if r['pair'] is not None]
        require(len(records) == 380 and len(eligible) == tp and sum(r['FN'] for r in records) == fn,
            'Independent full GT/FN conditional cohort conservation failed')
        totals = {}
        for key in ('GT_pixels', 'native_prediction_pixels', 'band_pixels',
                    'TP_in', 'FP_in', 'FN_in', 'TP_out', 'FP_out', 'FN_out'):
            totals[key] = sum(r[key] for r in eligible)
        editable = totals['FP_in'] + totals['FN_in']
        error = editable + totals['FP_out'] + totals['FN_out']
        result.append(dict(index=index, label=label,
            threshold=images[0]['threshold_slots'][position]['threshold'], raw_GT=380,
            native_matched_pairs=tp, native_FN=fn, ignored_GT=sum(r['ignored'] for r in records),
            matched_pair_pixel_totals=totals,
            matched_pair_editable_error_fraction=editable / error if error else None,
            matched_pair_mean_native_IoU=math.fsum(r['native_IoU'] for r in eligible) / len(eligible),
            matched_pair_mean_oracle_IoU=math.fsum(r['oracle_IoU'] for r in eligible) / len(eligible),
            oracle_ge_0_90_fixed_pairs=sum(r['oracle_ge_0_90_exact'] for r in eligible),
            oracle_ge_0_95_fixed_pairs=sum(r['oracle_ge_0_95_exact'] for r in eligible),
            counts_are_conditional_on_this_original_fixed_assignment=True,
            native_FN_have_no_selected_query_or_oracle=True))
    canonical = dict(assignment_index=0, assignment_label='0.50', denominator_fixed_pairs=374,
        original_unmatched_GT_without_selected_query=6,
        oracle_ge_0_90=result[0]['oracle_ge_0_90_fixed_pairs'],
        oracle_ge_0_95=result[0]['oracle_ge_0_95_fixed_pairs'],
        not_all380_GT_ceiling=True, not_AP=True, not_jointly_realizable_model_prediction=True)
    return result, canonical


def compare_official_anchors(np, masks, reader, evaluator, ids, anchors, expected_q):
    matching = reader.matching_images(np, masks, evaluator, ids)
    summary = reader.matching_summary(matching)
    require(summary == expected_q['control_results']['native_official']['one_to_one'],
        'Independent native official one-to-one counts/pixel totals differ from qualification')
    for actual, compact in zip(matching, anchors):
        require(actual['image_id'] == compact['image_id'] and
            actual['actual_COCO_arrays']['gtIds'] == compact['original_GT_ids'],
            'Independent original COCO GT order differs')
        for slot in compact['threshold_slots']:
            cell = actual['thresholds'][slot['index']]
            require(cell['threshold'] == slot['threshold'], 'Independent actual fixed-index threshold differs')
            extracted = []
            for row in cell['GT']:
                original = dict(row)
                original['pair'] = actual['TP_pairs'][row['TP_pair_key']] if row['TP_pair_key'] else None
                extracted.append(original)
            require(extracted == slot['GT'], 'Independent every fixed native pair/FN/ignore/query/DT differs')
    return summary


def qualify(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT CPU SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    common, raws, closure = pre_gate(spec, 34)
    gate = common.source_gate(path, spec, __file__, raws, closure, cpu=True)
    common.deadline(started)
    q_path = Path(spec['Q_path'])
    require(q_path.is_absolute() and not q_path.exists() and q_path.parent.exists(), 'Fresh absolute CPU Q path required')
    report = read(common.bound(spec['GPU_report']))
    gpu_spec = read(common.bound(spec['GPU_spec']))
    require(gpu_spec['schema'] == common.GPU_SCHEMA and gpu_spec['RAW_inputs'] == spec['RAW_inputs'][:33] and
        gpu_spec['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        gpu_spec['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
        gpu_spec['recipe'] == common.RECIPE and
        all(gpu_spec[k] == spec[k] for k in common.REF_KEYS if k != 'entry_source') and
        report['schema'] == common.REPORT_SCHEMA and report['status'] == common.REPORT_STATUS and
        report['source'] == spec['GPU_entry_source'] == gpu_spec['entry_source'] and
        report['spec'] == spec['GPU_spec'] and report['GPU_admission'] == spec['GPU_admission'] and
        report['phase'] == gpu_spec['phase'] and report['RAW_inputs'] == spec['RAW_inputs'][:33] and
        report['recipe'] == common.RECIPE and report['numeric_protocol'] == common.NUMERIC and
        report['qualified_native_Q'] == spec['qualified_native_Q'] and
        report['pair_anchors'] == spec['pair_anchors'] and
        report['anchor_derivation_record'] == spec['anchor_derivation_record'] and
        report['DEV'] == spec['DEV'] and report['native_predictions'] == spec['native_predictions'] and
        report['packet_refs'] == gate['packets'] and report['cleanup']['allocated_after'] == 0 and
        all(report[k] is True for k in ('full18700_original_prediction_dictionaries_exact',
            'native_natural100_render_order_full100_quality_exact', 'exact_GPU_band_rles_saved',
            'fixed_native_assignment_only', 'all380_GT_FN_ignore_retained_per_slot', 'no_GT_directed_inference', 'no_TEST')) and
        report['oracle_AP_computed'] is False and report['native_forward_calls'] == report['head_forward_calls'] ==
            report['backward_calls'] == 0 and report['weights_updated'] is False,
        'Actually complete locked N11 GPU parent required')
    gpu_i = common.joined(report['source'], report['spec'], report['GPU_admission'], report['phase'],
        '0', spec['RAW_inputs'], gate['owners'])
    require(gpu_i == 33 and spec['RAW_inputs'][33]['receipt'] == spec['GPU_RAW'] and
        all(ref in closure.values() for ref in (report['start'], report['native_export_replay'], report['coverage'])),
        'N11 GPU must join actual held0 final ancestor and output closure')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1], 'Actual CPU01 required before numerical imports')
    import torch
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5' and
        not torch.cuda.is_initialized(), 'Qualified versions/cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    libraries = common.imported_files(psutil, torch, np, COCO, COCOeval, masks, gate)
    rng = torch.random.get_rng_state().clone()
    reader = common.load(spec['reader_source'], '_N11_original_official_matching_helpers')
    renderer = common.load(spec['renderer_source'], '_N11_same_original_CPU_renderer')
    require(reader.Q_SCHEMA == common.N07_Q_SCHEMA and reader.STATUS == common.N07_Q_STATUS,
        'Original qualified matching helper ABI differs')
    coco = COCO(str(common.bound(spec['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and ids == [r['id'] for r in coco.dataset['images']] and
        len(coco.anns) == 380 and set(coco.cats) == {1}, 'Canonical DEV187/380/category1 required')
    native = read(common.bound(spec['native_predictions']))
    replay = read(common.bound(report['native_export_replay']))
    require(replay == native and len(replay) == 18700, 'Exact complete original native export prerequisite differs')
    grouped = reader.population(coco, replay, ids)
    anchors = common.validate_anchors(coco, ids, gate['anchors'], gate['q'])
    evaluator, COCO_metrics = reader.official_metrics(coco, replay, ids, COCOeval)
    expected_stats = gate['q']['control_results']['native_official']['COCO']['stats']
    require(len(COCO_metrics['stats']) == len(expected_stats) == 12 and
        all(abs(a - b) <= 1e-12 for a, b in zip(COCO_metrics['stats'], expected_stats)),
        'Independent unchanged native official COCO12 differs')
    matching_summary = compare_official_anchors(np, masks, reader, evaluator, ids, anchors, gate['q'])
    del evaluator
    factors, factor_proof = reader.factor_check(np, read(common.bound(gate['prior']['native_query_score_factors'])), replay, ids)
    coverage = read(common.bound(report['coverage']))
    require(coverage['schema'] == 'N11-native-editable-coverage-GPU-details-v1' and
        coverage['pair_anchors'] == spec['pair_anchors'], 'Actual GPU coverage input identity differs')
    diagnostics = coverage['images']
    rows = read(common.bound(gate['prior']['image_rows']))
    require(len(rows) == len(diagnostics) == len(anchors) == 187 and
        [d['image_id'] for d in diagnostics] == ids, 'Full coverage image population differs')
    checks, checked_pairs = [], 0
    with torch.no_grad():
        for index, (row, ref, diagnostic, anchor) in enumerate(zip(rows, gate['packets'], diagnostics, anchors)):
            common.deadline(started)
            packet = torch.load(str(common.bound(ref)), map_location='cpu')
            fields, identity, qids = common.packet_check(torch, reader, coco, packet, row, index, ref, gate)
            require(diagnostic['dataset_index'] == index and diagnostic['image_id'] == row['image_id'] and
                diagnostic['packet'] == ref and diagnostic['identity'] == identity and
                diagnostic['native_query_order'] == qids and
                diagnostic['natural_band_counts'] == row['band_counts_all100'] and
                diagnostic['full100_native_export_exact'] is True and
                diagnostic['GT_did_not_select_inference_queries'] is True,
                'Exact original per-image GPU identity/band population differs')
            L = renderer.render_original(torch, fields['M200'], identity)
            require(L.device.type == 'cpu' and L.dtype == torch.float32 and
                tuple(L.shape) == (100,) + tuple(identity['original_shape']) and bool(torch.isfinite(L).all()),
                'Same original CPU natural all100 M renderer differs')
            probabilities = fields['final_class_logits'][0].softmax(-1)[:, 0]
            native_by_query = {r['query_id']: r for r in grouped[row['image_id']]}
            query_checks = []
            selected = sorted({r['pair']['query_id'] for s in anchor['threshold_slots'] for r in s['GT'] if r['pair']})
            require(diagnostic['selected_original_matched_query_ids'] == selected and
                set(diagnostic['band_rles']) == {str(q) for q in selected}, 'No extra/missing diagnostic query allowed')
            bands = {}
            for ordinal, q in enumerate(qids):
                z = L[q]
                P = masks.decode(reader.rle_bytes(native_by_query[q]['segmentation'])).astype(bool)
                native_support = torch.from_numpy(np.ascontiguousarray(P))
                tau = 1e-4 + 1e-4 * z.abs()
                sign_flips = native_support != (z > 0)
                near_zero = z.abs() <= tau
                require(not bool((sign_flips & ~near_zero).any()), 'CPU/native sign discrepancy outside practical zero allowance')
                near_band = (z.abs() - 2.0).abs() <= tau
                CPU_band = z.abs() <= 2.0
                require(abs(int(CPU_band.sum()) - row['band_counts_all100'][q]) <= int(near_band.sum()),
                    'All100 CPU band count discrepancy outside practical near2 allowance')
                factor = factors[row['image_id'], q]
                probability = float(probabilities[q])
                quality = float((z.sigmoid() * native_support).sum() / (native_support.sum() + 1e-6))
                require(math.isclose(probability, factor['class_probability'], rel_tol=1e-6, abs_tol=1e-6) and
                    math.isclose(quality, factor['mask_quality'], rel_tol=1e-4, abs_tol=1e-4),
                    'CPU class/full100 native-support quality replay differs')
                if not P.any():
                    require(native_by_query[q]['score'] == factor['mask_quality'] == 0,
                        'Exact original empty query/S0 scope differs')
                band_flips = None
                if q in selected:
                    band_rle = diagnostic['band_rles'][str(q)]
                    require(set(band_rle) == {'size', 'counts'} and
                        band_rle['size'] == identity['original_shape'] and type(band_rle['counts']) is str,
                        'Exact original-size compressed GPU band RLE required')
                    B = masks.decode(reader.rle_bytes(band_rle)).astype(bool)
                    require(B.shape == P.shape and int(B.sum()) == row['band_counts_all100'][q],
                        'Archived exact GPU band area/raster differs')
                    difference = torch.from_numpy(np.ascontiguousarray(B)) != CPU_band
                    require(not bool((difference & ~near_band).any()),
                        'Saved GPU band differs outside practical CPU near2 allowance')
                    band_flips = int(difference.sum())
                    bands[q] = B
                query_checks.append(dict(query_id=q, native_export_ordinal=ordinal,
                    native_support_flips=int(sign_flips.sum()), practical_near_zero_pixels=int(near_zero.sum()),
                    CPU_band_pixels=int(CPU_band.sum()), GPU_band_pixels=row['band_counts_all100'][q],
                    practical_near_band_pixels=int(near_band.sum()), selected_GPU_CPU_band_flips=band_flips,
                    absolute_class_probability_difference=abs(probability - factor['class_probability']),
                    absolute_quality_difference=abs(quality - factor['mask_quality'])))
            require([s['index'] for s in diagnostic['threshold_slots']] == [0, 5, 8], 'Fixed coverage slots differ')
            for actual_slot, fixed_slot in zip(diagnostic['threshold_slots'], anchor['threshold_slots']):
                require(all(actual_slot[k] == fixed_slot[k] for k in ('index', 'label', 'threshold')) and
                    len(actual_slot['GT']) == len(fixed_slot['GT']), 'Fixed original threshold/GT population differs')
                for actual, fixed in zip(actual_slot['GT'], fixed_slot['GT']):
                    require(set(actual) == set(fixed) | {'band_query_key', 'pixels'} and
                        {k: actual[k] for k in fixed} == fixed, 'Every original GT/FN/ignore/pair identity differs')
                    pair = fixed['pair']
                    if pair is None:
                        require(actual['pixels'] is None and actual['band_query_key'] is None,
                            'Original FN/ignore must retain null query/oracle')
                    else:
                        q = pair['query_id']
                        require(actual['band_query_key'] == str(q), 'Original fixed pair/band query alignment differs')
                        independent = independent_pair_pixels(np, masks, reader, coco, native_by_query[q], bands[q], pair)
                        require(independent == actual['pixels'], 'Every independent native/error/oracle pixel field differs')
                        checked_pairs += 1
            checks.append(dict(dataset_index=index, image_id=row['image_id'], packet=ref,
                natural_all100_CPU_replayed=True, fixed_GPU_bands_never_substituted=True, queries=query_checks))
            del packet, fields, L, bands
            common.deadline(started)
    require(checked_pairs == 374 + 348 + 107, 'Every threshold-specific fixed native pair must be checked')
    summary, canonical = independent_summary(common, diagnostics)
    require(summary == report['summaries'] and canonical == report['canonical_0_50_fixed_pair_ceiling'],
        'Independent full conditional summary/canonical374 ceiling differs')
    require(torch.equal(rng, torch.random.get_rng_state()) and not torch.cuda.is_initialized() and
        'mmdet' not in sys.modules and psutil.Process().cpu_affinity() == [0, 1],
        'CPU diagnostic changed RNG/resources, initialized CUDA or loaded teacher')
    common.final_closure(spec, path, gate, __file__, libraries,
        lambda: common.imported_files(psutil, torch, np, COCO, COCOeval, masks, gate))
    common.deadline(started)
    q = dict(schema=common.Q_SCHEMA, status=common.Q_STATUS, phase=spec['phase'],
        source=spec['GPU_entry_source'], reader_source=gate['source'], common_source=spec['common_source'],
        spec=gate['spec_ref'], CPU_admission=gate['admission_ref'],
        GPU_spec=spec['GPU_spec'], GPU_report=spec['GPU_report'], GPU_RAW=spec['GPU_RAW'],
        GPU_admission=spec['GPU_admission'], RAW_inputs=spec['RAW_inputs'],
        prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],
        terminal_identity_pins=spec['terminal_identity_pins'], qualified_native_Q=spec['qualified_native_Q'],
        pair_anchors=spec['pair_anchors'], anchor_derivation_record=spec['anchor_derivation_record'],
        DEV=spec['DEV'], native_predictions=spec['native_predictions'], packet_refs=gate['packets'],
        coverage=report['coverage'], native_export_replay=report['native_export_replay'],
        recipe=common.RECIPE, numeric_protocol=common.NUMERIC, images=187, raw_GT=380,
        native_predictions_count=18700, full_GT_rows_per_slot=380, checked_matched_pairs_per_slot=[374, 348, 107],
        native_FN_per_slot=[6, 32, 273], checked_pair_occurrences=checked_pairs,
        native_COCO=COCO_metrics, original_official_matching=matching_summary,
        native_factor_check=factor_proof, summaries=summary, canonical_0_50_fixed_pair_ceiling=canonical,
        CPU_native_field_and_selected_band_checks=checks, actual_imported_files=libraries,
        original18700_export_masks_scores_classes_order_unchanged=True,
        original_official_matching_every_compact_pair_FN_ignore_CPU_checked=True,
        exact_GPU_band_conditioned_pixel_decomposition_CPU_checked=True,
        fixed_pair_oracle_formula_and_direct_raster_CPU_checked=True,
        native_M200_natural_all100_CPU_render_replayed=True,
        practical_CPU_native_sign_and_selected_band_consistency_checked=True,
        GPU_band_RLE_or_exported_mask_substitution=False,
        CPU_GPU_bitwise_equality_claimed=False, certified_GPU_error_bound=False,
        native_teacher_CPU_replayed=False, F_e_projection_CPU_replayed=False, original_JPEG_decode_replayed=False,
        head_forward_calls=0, optimizer_created=False, backward_calls=0, weights_updated=False,
        no_GT_directed_inference=True, no_TEST=True, oracle_AP_computed=False,
        actual_efficacy_established=False, novelty_established=False,
        CPU_affinity=[0, 1], threads=2, CUDA_initialized=False,
        elapsed_seconds=time.monotonic() - started, deadline_seconds=1800, automatic_retry=False,
        current_owned_CPU_terminal_self_certified=False, whole_leaf_VRAM_certified=False,
        evidence_limits=[
            'Exact GPU band RLEs and unchanged native exports supply all pixel metrics; CPU near2 replay is only practical consistency, not GPU bitwise/error-bound certification.',
            'The .50/.75/.90 cohorts use fixed original official assignment indices 0/5/8; their selected queries may differ, and native FN/ignore rows have null oracle.',
            'Oracle support is unrestricted binary GT inside the fixed selected query band; it is not a deployable model, AP, all380-GT ceiling, or established efficacy.',
            'Canonical .50 ceiling counts condition on 374 original matched pairs; the six original unmatched GT have no selected query or inferred bound.',
            'No teacher, JPEG, F/e projection, head, optimization, training, TEST or full trajectory is replayed.',
            'ROOT must separately close and qualify this CPU process using its original actual held/reaped receipt; this Q cannot certify its own terminal.'
        ])
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
