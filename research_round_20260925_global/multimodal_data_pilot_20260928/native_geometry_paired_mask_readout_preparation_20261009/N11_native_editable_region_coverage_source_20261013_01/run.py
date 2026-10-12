"""Unbound N11 exact GPU native-band coverage SOURCE; ROOT alone admits it.

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


def summarize(common, images):
    summaries = []
    for position, (index, label, expected_tp, expected_fn) in enumerate(common.SLOTS):
        rows = [r for im in images for r in im['threshold_slots'][position]['GT']]
        pairs = [r['pixels'] for r in rows if r['pixels'] is not None]
        require(len(rows) == 380 and len(pairs) == expected_tp and
            sum(r['FN'] for r in rows) == expected_fn, 'Full threshold cohort conservation differs')
        totals = {k: sum(r[k] for r in pairs) for k in ('GT_pixels', 'native_prediction_pixels',
            'band_pixels', 'TP_in', 'FP_in', 'FN_in', 'TP_out', 'FP_out', 'FN_out')}
        editable_error = totals['FP_in'] + totals['FN_in']
        total_error = editable_error + totals['FP_out'] + totals['FN_out']
        summaries.append(dict(index=index, label=label,
            threshold=images[0]['threshold_slots'][position]['threshold'], raw_GT=380,
            native_matched_pairs=expected_tp, native_FN=expected_fn,
            ignored_GT=sum(r['ignored'] for r in rows), matched_pair_pixel_totals=totals,
            matched_pair_editable_error_fraction=editable_error / total_error if total_error else None,
            matched_pair_mean_native_IoU=math.fsum(p['native_IoU'] for p in pairs) / len(pairs),
            matched_pair_mean_oracle_IoU=math.fsum(p['oracle_IoU'] for p in pairs) / len(pairs),
            oracle_ge_0_90_fixed_pairs=sum(p['oracle_ge_0_90_exact'] for p in pairs),
            oracle_ge_0_95_fixed_pairs=sum(p['oracle_ge_0_95_exact'] for p in pairs),
            counts_are_conditional_on_this_original_fixed_assignment=True,
            native_FN_have_no_selected_query_or_oracle=True))
    canonical = dict(assignment_index=0, assignment_label='0.50', denominator_fixed_pairs=374,
        original_unmatched_GT_without_selected_query=6,
        oracle_ge_0_90=summaries[0]['oracle_ge_0_90_fixed_pairs'],
        oracle_ge_0_95=summaries[0]['oracle_ge_0_95_fixed_pairs'],
        not_all380_GT_ceiling=True, not_AP=True, not_jointly_realizable_model_prediction=True)
    return summaries, canonical


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    common, raws, closure = pre_gate(spec, 33)
    gate = common.source_gate(path, spec, __file__, raws, closure)
    common.deadline(started)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists(), 'Fresh absolute output directory required')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1], 'CPU01 required before numerical imports')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Qualified Torch/cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__ == '1.23.5', 'Qualified NumPy required')
    libraries = common.imported_files(psutil, torch, np, COCO, COCOeval, masks, gate)
    reader = common.load(spec['reader_source'], '_N11_old_qualified_native_reader')
    renderer = common.load(spec['renderer_source'], '_N11_original_renderer')
    require(reader.Q_SCHEMA == common.N07_Q_SCHEMA and reader.STATUS == common.N07_Q_STATUS,
        'Old qualified reader ABI differs')
    kernels = original_export_kernels(common, spec['export_source'])
    coco = COCO(str(common.bound(spec['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and ids == [r['id'] for r in coco.dataset['images']] and
        len(coco.anns) == 380 and set(coco.cats) == {1}, 'Canonical full DEV187/380/category1 required')
    native = read(common.bound(spec['native_predictions']))
    grouped = reader.population(coco, native, ids)
    anchors = common.validate_anchors(coco, ids, gate['anchors'], gate['q'])
    rows = read(common.bound(gate['prior']['image_rows']))
    trace = [read_line for read_line in (json.loads(line) for line in
        common.bound(gate['prior']['trace']).read_text(encoding='utf-8').splitlines() if line.strip())]
    require(rows == trace and len(rows) == 187 and [r['image_id'] for r in rows] == ids,
        'Original image rows/trace identity differs')
    require(not torch.cuda.is_initialized(), 'All input/module gates must precede CUDA')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024**2 /
        torch.cuda.get_device_properties(0).total_memory), 0)
    require(torch.cuda.memory_allocated(0) == 0, 'Clean GPU allocator required')
    torch.cuda.reset_peak_memory_stats(0)
    output.mkdir()
    common.write(output / 'START.json', dict(schema='N11-native-editable-coverage-GPU-start-v1',
        phase=spec['phase'], source=gate['source'], spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'], recipe=common.RECIPE,
        qualified_native_Q=spec['qualified_native_Q'], pair_anchors=spec['pair_anchors'],
        anchor_derivation_record=spec['anchor_derivation_record'], actual_imported_files=libraries,
        no_GT_directed_inference=True, no_training=True, no_TEST=True))
    replay, diagnostics = [], []
    with torch.no_grad():
        for index, (row, ref, anchor) in enumerate(zip(rows, gate['packets'], anchors)):
            common.deadline(started)
            packet = torch.load(str(common.bound(ref)), map_location='cpu')
            fields, identity, qids_archived = common.packet_check(torch, reader, coco, packet, row, index, ref, gate)
            M = fields['M200'].to('cuda:0')
            C = fields['final_class_logits'].to('cuda:0')
            top, qids = kernels.order(torch, C, types.SimpleNamespace(num_classes=1))
            require(qids.cpu().tolist() == qids_archived == [r['query_id'] for r in grouped[row['image_id']]],
                'Original query order must reproduce exactly')
            L = renderer.render_original(torch, M, identity)
            require(L.dtype == torch.float32 and L.device.type == 'cuda' and
                tuple(L.shape) == (100,) + tuple(identity['original_shape']) and bool(torch.isfinite(L).all()),
                'Original natural all100 renderer differs')
            ordered = L[qids]
            binary, quality, scores = kernels.quality(torch, ordered, top)
            masks_cpu, scores_cpu = binary.cpu().numpy(), scores.cpu().tolist()
            baseline = []
            for ordinal, q in enumerate(qids_archived):
                reconstructed = dict(image_id=row['image_id'], query_id=q, category_id=1,
                    score=float(scores_cpu[ordinal]), segmentation=rle(np, masks, masks_cpu[ordinal]))
                require(reconstructed == grouped[row['image_id']][ordinal],
                    'Original full100 prediction dictionary must reproduce exactly without tolerance')
                if not masks_cpu[ordinal].any():
                    require(reconstructed['score'] == 0, 'Original empty-support S0 must remain zero')
                baseline.append(reconstructed)
            replay.extend(baseline)
            all_band = L.abs() <= 2.0
            counts = all_band.flatten(1).sum(1).cpu().tolist()
            require(counts == row['band_counts_all100'], 'Exact original natural all100 band counts changed')
            # GT is first used below, after the exact unchanged inference prerequisite.
            selected = sorted({r['pair']['query_id'] for s in anchor['threshold_slots']
                for r in s['GT'] if r['pair'] is not None})
            band_rles = {str(q): rle(np, masks, all_band[q].cpu().numpy()) for q in selected}
            native_by_query = {r['query_id']: r for r in baseline}
            slots = []
            for slot in anchor['threshold_slots']:
                records = []
                for original in slot['GT']:
                    record = copy.deepcopy(original)
                    pair = original['pair']
                    record['band_query_key'] = str(pair['query_id']) if pair else None
                    record['pixels'] = gpu_pair_pixels(np, masks, reader, coco,
                        native_by_query[pair['query_id']], all_band[pair['query_id']].cpu().numpy(), pair) if pair else None
                    records.append(record)
                slots.append(dict(index=slot['index'], label=slot['label'], threshold=slot['threshold'], GT=records))
            diagnostics.append(dict(dataset_index=index, image_id=row['image_id'], packet=ref,
                identity=identity, native_query_order=qids_archived, natural_band_counts=counts,
                selected_original_matched_query_ids=selected, band_rles=band_rles, threshold_slots=slots,
                full100_native_export_exact=True, GT_did_not_select_inference_queries=True))
            M = C = top = qids = L = ordered = binary = quality = scores = all_band = None
            del packet, fields
            common.deadline(started)
    require(replay == native and len(replay) == 18700, 'Complete18700 exact native replay required')
    summary, canonical = summarize(common, diagnostics)
    common.write(output / 'NATIVE_EXPORT_REPLAY.json', replay)
    common.write(output / 'COVERAGE.json', dict(schema='N11-native-editable-coverage-GPU-details-v1',
        pair_anchors=spec['pair_anchors'], images=diagnostics))
    gc.collect()
    torch.cuda.synchronize()
    cleanup = dict(allocated_after=int(torch.cuda.memory_allocated(0)),
        allocator_peak_bytes=int(torch.cuda.max_memory_allocated(0)))
    require(cleanup['allocated_after'] == 0 and cleanup['allocator_peak_bytes'] <= 4096 * 1024**2,
        'GPU cleanup/allocation budget differs')
    common.final_closure(spec, path, gate, __file__, libraries,
        lambda: common.imported_files(psutil, torch, np, COCO, COCOeval, masks, gate))
    common.deadline(started)
    report = dict(schema=common.REPORT_SCHEMA, status=common.REPORT_STATUS,
        phase=spec['phase'], source=gate['source'], common_source=spec['common_source'],
        spec=gate['spec_ref'], GPU_admission=gate['admission_ref'], recipe=common.RECIPE,
        numeric_protocol=common.NUMERIC, RAW_inputs=spec['RAW_inputs'],
        prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],
        terminal_identity_pins=spec['terminal_identity_pins'], qualified_native_Q=spec['qualified_native_Q'],
        pair_anchors=spec['pair_anchors'], anchor_derivation_record=spec['anchor_derivation_record'],
        prior_N07_report=spec['prior_N07_report'], prior_N07_CPU_qualification=spec['prior_N07_CPU_qualification'],
        DEV=spec['DEV'], native_predictions=spec['native_predictions'], packet_refs=gate['packets'],
        start=common.reference(output / 'START.json'),
        native_export_replay=common.reference(output / 'NATIVE_EXPORT_REPLAY.json'),
        coverage=common.reference(output / 'COVERAGE.json'), summaries=summary,
        canonical_0_50_fixed_pair_ceiling=canonical, cleanup=cleanup, actual_imported_files=libraries,
        full18700_original_prediction_dictionaries_exact=True,
        native_natural100_render_order_full100_quality_exact=True,
        exact_GPU_band_rles_saved=True, fixed_native_assignment_only=True,
        all380_GT_FN_ignore_retained_per_slot=True, native_forward_calls=0,
        native_model_loaded=False, F_e_projection_replayed=False, original_JPEG_decoded=False,
        head_forward_calls=0, optimizer_created=False, backward_calls=0, weights_updated=False,
        no_GT_directed_inference=True, no_TEST=True, oracle_AP_computed=False,
        actual_efficacy_established=False, novelty_established=False,
        current_owned_GPU_terminal_self_certified=False, whole_leaf_VRAM_certified=False,
        CPU_affinity=[0, 1], threads=2, elapsed_seconds=time.monotonic() - started,
        deadline_seconds=1800, automatic_retry=False)
    common.write(output / 'REPORT.json', report)
    print(json.dumps(dict(status=common.REPORT_STATUS, report=common.reference(output / 'REPORT.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Explicit ROOT-owned single execution required')
    run(args.spec)
