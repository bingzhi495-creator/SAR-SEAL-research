"""Fixed native E24 COCO one-to-one error audit; no model, Torch or training.

FP labels are descriptive candidates, not causes. All100 and fixed rank1/10/100
are retained; best-IoU coverage is auxiliary geometry, never one-to-one recall.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

SCHEMA = 'ROOT-N06-fixed-E24-one-to-one-error-audit-spec-v1'
STATUS = 'ROOT_N06_FIXED_E24_ONE_TO_ONE_ERROR_AUDIT_CPU_COMPLETE'
E24_SCHEMA = 'ROOT-fixed-E24-native0-export-COCO-reference-recompute-CPU-qualification-v1'
E24_STATUS = 'ROOT_FIXED_E24_NATIVE0_EXPORT_COCO_REFERENCE_RECOMPUTED_CPU'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
FP_TYPES = ('duplicate_or_assignment_conflict', 'geometry_candidate', 'background_candidate')
RANKS = (1, 10, 100)
COUNT_KEYS = ('TP', 'FN', 'FP', 'ignored_DT', 'ignored_GT') + FP_TYPES


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] >= 0 and
            type(ref['sha256']) is str and len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed ref')


def bound(ref):
    ref_shape(ref); require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def terminal(ref, phase=None, exit_code=0):
    path = Path(ref['path'])
    require(path.is_absolute() and path.name == 'PARENT_RECEIPT.json' and
            (phase is None or type(phase) is str and phase and path.parent.name == phase), 'Explicit original RAW path/phase required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == exit_code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=exit_code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual held/reaped terminal%d required' % exit_code)
    if 'phase' in raw:
        require(type(raw['phase']) is str and raw['phase'] == path.parent.name, 'RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and type(peak) in (int, float) and
            math.isfinite(peak) and 0 <= peak <= 8192 and type(raw['whole_GPU_used_limit_MiB']) is int and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation', 'RAW resource/owner scope differs')
    return raw


def owner_join(row, raw, closure):
    owner, pb = read(bound(row['admission'])), read(bound(row['phase_binding']))
    boot = bound(row['bootstrap']); runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime/'PHASE_BINDING.json' and boot == runtime/'bootstrap.py' and
            owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['phase'] == row['phase'] and
            owner['driver_sha256'] == OWNER_SHA and owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
            owner['cuda_visible_devices'] in ('', '0') and owner['whole_GPU_used_limit_MiB'] == 8192 and
            raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
            pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and pb['argv'][-1] == str(boot) and
            pb['CPU_affinity'] == owner['allowed_cpus'] and pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and
            pb['automatic_retry'] is False, 'Actual original admission/PB/bootstrap differs')
    entry, args = owner['entry'], owner['entry_args']
    require(entry in closure and owner['bound_source_sha256'].get(entry) == closure[entry]['sha256'], 'Original entry absent/unbound')
    tree = ast.parse(boot.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'sys' and t.attr == 'argv' for t in n.targets)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
             isinstance(n.func.value, ast.Name) and n.func.value.id == 'runpy' and n.func.attr == 'run_path']
    require(argv == [[entry]+args] and len(calls) == 1 and ast.literal_eval(calls[0].args[0]) == entry and
            {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords} == {'run_name': '__main__'}, 'Original bootstrap argv differs')
    if '--spec' in args:
        require(args.count('--spec') == 1, 'Ambiguous original SPEC')
        original = args[args.index('--spec')+1]
        require(original in closure and owner['bound_source_sha256'].get(original) == closure[original]['sha256'], 'Original SPEC absent/unbound')
    if 'scientific_spec' in owner:
        original = owner['scientific_spec']
        require(original in closure.values() and owner['bound_source_sha256'].get(original['path']) == original['sha256'], 'Scientific SPEC absent/unbound')


def source_gate(path, spec):
    require(spec['schema'] == SCHEMA and type(spec['phase']) is str and spec['phase'] and
            len(spec['RAW_inputs']) == 16 and all(set(RAW_KEYS) <= set(r) for r in spec['RAW_inputs']) and
            len({r['receipt']['path'] for r in spec['RAW_inputs']}) == 16, 'Exactly sixteen normal RAWs required')
    raws = [terminal(r['receipt'], r['phase']) for r in spec['RAW_inputs']]
    failed = terminal(spec['prior_failed_CPU_receipt'], exit_code=1)  # Seventeen states before the first SHA/import.
    require(spec['prior_failed_CPU_receipt']['path'] not in {r['receipt']['path'] for r in spec['RAW_inputs']}, 'Failure cannot qualify an ancestor')
    for ref in spec['sources']: ref_shape(ref)
    closure = {r['path']: r for r in spec['sources']}; require(len(closure) == len(spec['sources']), 'Unique input closure required')
    refs = [spec[k] for k in ('prior_failed_CPU_receipt', 'predictions', 'DEV', 'E24_reference_Q', 'entry_source', 'protocol')]
    refs += [r[k] for r in spec['RAW_inputs'] for k in RAW_KEYS if k != 'phase']
    require(all(r in closure.values() for r in refs), 'All input refs must be sources-bound')
    for ref in spec['sources']: bound(ref)
    source, spec_ref = reference(__file__), reference(path)
    require(source == spec['entry_source'], 'Actual N06 SOURCE differs')
    for row, raw in zip(spec['RAW_inputs'], raws): owner_join(row, raw, closure)
    admission_path = Path(spec['CPU_admission_file']); require(admission_path.is_absolute(), 'Absolute CPU admission required')
    admit_ref = reference(admission_path); admit = read(admission_path)
    require(admit['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admit['phase'] == spec['phase'] and
            admit['driver_sha256'] == OWNER_SHA and admit['scientific_spec'] == spec_ref and admit['entry'] == source['path'] and
            admit['entry_args'] == ['--spec', str(path), '--owner-execute'] and admit['allowed_cpus'] == [0, 1] and
            admit['cuda_visible_devices'] == '' and admit['automatic_retry'] is False and admit['whole_GPU_used_limit_MiB'] == 8192 and
            type(admit['qualified_leaf_deadline_seconds']) is int and admit['qualified_leaf_deadline_seconds'] > 0 and
            admit['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
            all(admit['bound_source_sha256'].get(r['path']) == r['sha256'] for r in spec['sources']), 'Actual own CPU admission differs')
    q = read(bound(spec['E24_reference_Q']))
    require(q['schema'] == E24_SCHEMA and q['status'] == E24_STATUS and q['predictions'] == spec['predictions'] and
            q['DEV'] == spec['DEV'] and q['images'] == 187 and q['raw_GT'] == 380 and q['predictions_count'] == 18700,
            'Qualified original native0 E24 population differs')
    require(q['spec'] in closure.values(), 'Original E24 evaluation SPEC must be explicitly sources-bound')
    bound(q['spec'])
    require(type(spec['expected_COCO_stats']) is list and len(spec['expected_COCO_stats']) == 12 and
            len(q['COCO']['stats']) == 12, 'Twelve locked metrics required')
    for a, b in zip(spec['expected_COCO_stats'], q['COCO']['stats']):
        close(a, b, 'Locked E24 qualification stats differ')
    return source, spec_ref, admit_ref, failed


def close(a, b, message):
    require(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and
            -1 <= a <= 1 and -1 <= b <= 1 and abs(a-b) <= 1e-12, message)


def fp_kind(max_iou, threshold):
    require(type(max_iou) in (int, float) and type(threshold) in (int, float) and math.isfinite(max_iou) and
            math.isfinite(threshold) and 0 <= max_iou <= 1 and .5 <= threshold <= .95, 'Finite IoU/threshold required')
    if max_iou >= threshold: return FP_TYPES[0]
    return FP_TYPES[1] if max_iou >= .1 else FP_TYPES[2]


def area_group(area):
    require(type(area) in (int, float) and math.isfinite(area) and area >= 0, 'Nonnegative descriptive area required')
    return 'small' if area <= 1024 else ('medium' if area <= 9216 else 'large')


def density_group(count):
    require(type(count) is int and count >= 0, 'Nonnegative GT count required')
    return '0' if count == 0 else ('1' if count == 1 else ('2-4' if count <= 4 else '>=5'))


def empty_counts(): return {k: 0 for k in COUNT_KEYS}


def tally(dt_rows, eligible_gt_ids, ignored_gt_count):
    result = empty_counts(); matched = set()
    for row in dt_rows:
        if row['ignored']: result['ignored_DT'] += 1
        elif row['matched_GT']:
            require(row['matched_GT'] in eligible_gt_ids and row['matched_GT'] not in matched, 'Regular GT matched more than once')
            matched.add(row['matched_GT']); result['TP'] += 1
        else:
            require(row['FP_type'] in FP_TYPES, 'Every unignored unmatched DT needs one FP label')
            result['FP'] += 1; result[row['FP_type']] += 1
    result['FN'] = len(eligible_gt_ids)-len(matched); result['ignored_GT'] = ignored_gt_count
    require(result['TP']+result['FP']+result['ignored_DT'] == len(dt_rows) and
            result['TP']+result['FN'] == len(eligible_gt_ids) and result['FP'] == sum(result[k] for k in FP_TYPES), 'Count conservation differs')
    return result


def add_counts(target, value):
    for key in COUNT_KEYS: target[key] += value[key]


def rle_bytes(rle):
    return dict(size=list(rle['size']), counts=rle['counts'].encode('ascii') if isinstance(rle['counts'], str) else rle['counts'])


def population(np, coco, predictions):
    ids = [v['id'] for v in coco.dataset['images']]
    require(len(ids) == len(set(ids)) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1} and
            type(predictions) is list and len(predictions) == 18700, 'Complete DEV187/380/all100 population required')
    grouped = {i: [] for i in ids}
    for index, row in enumerate(predictions):
        require(type(row) is dict and set(row) == {'image_id', 'category_id', 'query_id', 'score', 'segmentation'} and
                row['image_id'] == ids[index//100] and type(row['category_id']) is int and row['category_id'] == 1 and
                type(row['query_id']) is int and 0 <= row['query_id'] < 100 and type(row['score']) in (int, float) and
                math.isfinite(row['score']) and 0 <= row['score'] <= 1, 'Native export identity/order/score differs')
        image = coco.imgs[row['image_id']]; rle = row['segmentation']
        require(type(rle) is dict and set(rle) == {'size', 'counts'} and rle['size'] == [image['height'], image['width']] and
                type(rle['counts']) is str, 'Original-size compressed RLE required')
        grouped[row['image_id']].append(row)
    require(all(len(v) == 100 and {r['query_id'] for r in v} == set(range(100)) for v in grouped.values()), 'All native query identities required')
    return ids


def pair_pixels(np, masks, coco_gt, coco_dt, gt_id, dt_id, expected_iou):
    gt = masks.decode(rle_bytes(coco_gt.annToRLE(coco_gt.anns[gt_id]))).astype(bool)
    dt = masks.decode(rle_bytes(coco_dt.annToRLE(coco_dt.anns[dt_id]))).astype(bool)
    require(gt.shape == dt.shape, 'Matched original raster shapes differ')
    tp = int(np.count_nonzero(gt & dt)); fp = int(np.count_nonzero(~gt & dt)); fn = int(np.count_nonzero(gt & ~dt))
    union, total = tp+fp+fn, 2*tp+fp+fn
    iou = tp/union if union else 0.; dice = 2*tp/total if total else 0.
    require(abs(iou-expected_iou) <= 1e-12, 'Matched pixel IoU differs from actual cached ordinary IoU')
    return dict(gt_id=gt_id, dt_id=dt_id, query_id=coco_dt.anns[dt_id]['query_id'], IoU=iou, Dice=dice,
                pixel_TP=tp, pixel_FP=fp, pixel_FN=fn, empty_union_IoU_convention=0)


def audit_image(np, masks, evaluator, image_id):
    p = evaluator.params
    cells = [v for v in evaluator.evalImgs if v is not None and v['image_id'] == image_id and v['category_id'] == 1 and
             v['aRng'] == p.areaRng[0] and v['maxDet'] == 100]
    require(len(cells) == 1, 'Exactly one actual area-all/maxDet100 evalImg required')
    cell = cells[0]; gt_ids, dt_ids = cell['gtIds'], cell['dtIds']
    require(len(dt_ids) == 100 and cell['dtMatches'].shape == cell['dtIgnore'].shape == (10, 100) and
            cell['gtMatches'].shape == (10, len(gt_ids)), 'COCO matching array population differs')
    original_gts = evaluator._gts[image_id, 1]; original_dts = evaluator._dts[image_id, 1]
    original_gt_ids = [v['id'] for v in original_gts]
    dt_order = np.argsort([-v['score'] for v in original_dts], kind='mergesort')[:100]
    require(dt_ids == [original_dts[int(i)]['id'] for i in dt_order] and set(gt_ids) == set(original_gt_ids), 'Actual cached matrix ID mapping differs')
    cached = evaluator.ious[image_id, 1]
    ious = np.asarray(cached, dtype=np.float64) if len(original_gt_ids) else np.zeros((100, 0), dtype=np.float64)
    require(ious.shape == (100, len(original_gt_ids)) and bool(np.isfinite(ious).all()) and bool(((ious >= 0) & (ious <= 1)).all()), 'Finite cached IoU matrix required')
    columns = {g: i for i, g in enumerate(original_gt_ids)}; gt_indices = {g: i for i, g in enumerate(gt_ids)}
    eligible = {g for g, ignore in zip(gt_ids, cell['gtIgnore']) if not bool(ignore)}
    require(all(not evaluator.cocoGt.anns[g].get('iscrowd', 0) for g in eligible), 'Eligible all-area GT unexpectedly crowd')
    max_dt = {d: max((float(ious[i, columns[g]]) for g in eligible), default=0.) for i, d in enumerate(dt_ids)}
    best_gt = {g: max((float(v) for v in ious[:, columns[g]]), default=0.) for g in gt_ids}
    gt_meta = [dict(gt_id=g, ignored=bool(cell['gtIgnore'][gt_indices[g]]),
                    iscrowd=bool(evaluator.cocoGt.anns[g].get('iscrowd', 0)),
                    area=float(evaluator.cocoGt.anns[g]['area']), area_group=area_group(float(evaluator.cocoGt.anns[g]['area'])),
                    all100_best_IoU_auxiliary=best_gt[g]) for g in gt_ids]
    image_density = density_group(len(gt_ids)); pairs, thresholds = {}, []
    for ti, threshold in enumerate(p.iouThrs):
        threshold = float(threshold); rows = []
        for di, dt_id in enumerate(dt_ids):
            gt_id = int(cell['dtMatches'][ti, di]); ignored = bool(cell['dtIgnore'][ti, di]); ann = evaluator.cocoDt.anns[dt_id]
            row = dict(dt_id=dt_id, query_id=ann['query_id'], score=float(ann['score']), rank=di+1, matched_GT=gt_id,
                       ignored=ignored, max_IoU_to_nonignore_noncrowd_GT=max_dt[dt_id],
                       area=float(ann['area']), area_group=area_group(float(ann['area'])), FP_type=None)
            if not ignored and gt_id:
                require(gt_id in eligible and int(cell['gtMatches'][ti, gt_indices[gt_id]]) == dt_id, 'TP reciprocity differs')
                key = '%d:%d' % (dt_id, gt_id)
                if key not in pairs:
                    pairs[key] = pair_pixels(np, masks, evaluator.cocoGt, evaluator.cocoDt, gt_id, dt_id, float(ious[di, columns[gt_id]]))
                row['TP_pair_key'] = key
            elif not ignored:
                row['FP_type'] = fp_kind(max_dt[dt_id], threshold)
            rows.append(row)
        ranks = {str(k): tally(rows[:k], eligible, len(gt_ids)-len(eligible)) for k in RANKS}
        gt_rows = [dict(gt_id=g, matched_DT=int(cell['gtMatches'][ti, gi]), ignored=bool(cell['gtIgnore'][gi]),
                        FN=not bool(cell['gtIgnore'][gi]) and int(cell['gtMatches'][ti, gi]) == 0) for gi, g in enumerate(gt_ids)]
        require(ranks['100']['FN'] == sum(v['FN'] for v in gt_rows) and
                ranks['100']['TP'] == sum(not v['ignored'] and v['matched_DT'] != 0 for v in gt_rows) and
                ranks['100']['ignored_DT'] == int(cell['dtIgnore'][ti].sum()), 'Full100 counts differ from COCO arrays')
        areas = {name: dict(GT=0, TP=0, FN=0, FP=0, ignored_DT=0, ignored_GT=0) for name in ('small', 'medium', 'large')}
        for meta, observed in zip(gt_meta, gt_rows):
            group = areas[meta['area_group']]
            if observed['ignored']: group['ignored_GT'] += 1
            else:
                group['GT'] += 1; group['FN' if observed['FN'] else 'TP'] += 1
        for row in rows:
            group = areas[row['area_group']]
            if row['ignored']: group['ignored_DT'] += 1
            elif not row['matched_GT']: group['FP'] += 1
        for key in ('TP', 'FN', 'FP', 'ignored_DT', 'ignored_GT'):
            require(sum(v[key] for v in areas.values()) == ranks['100'][key], 'Mutually exclusive area counts differ')
        thresholds.append(dict(threshold=threshold, DT=rows, GT=gt_rows, rank_counts=ranks, descriptive_area_counts=areas))
    t50 = next(i for i, t in enumerate(p.iouThrs) if abs(float(t)-.5) < 1e-12)
    t75 = next(i for i, t in enumerate(p.iouThrs) if abs(float(t)-.75) < 1e-12)
    missed75 = [g for g in eligible if not cell['gtMatches'][t75, gt_indices[g]]]
    matched50_missed75 = [g for g in missed75 if cell['gtMatches'][t50, gt_indices[g]]]
    return dict(image_id=image_id, original_GT_count=len(gt_ids), instance_density_group=image_density, GT=gt_meta,
                actual_COCO_arrays=dict(gtIds=gt_ids, dtIds=dt_ids, gtIgnore=cell['gtIgnore'].astype(bool).tolist(),
                    dtIgnore=cell['dtIgnore'].astype(bool).tolist(), gtMatches=cell['gtMatches'].astype(np.int64).tolist(),
                    dtMatches=cell['dtMatches'].astype(np.int64).tolist(), dtScores=[float(v) for v in cell['dtScores']],
                    area_range=list(cell['aRng']), maxDet=100), thresholds=thresholds, TP_pairs=list(pairs.values()),
                missed_at_075_GT_ids=missed75, missed075_but_matched050_GT_ids=matched50_missed75)


def summarize(images, thresholds):
    output = []
    for ti, threshold in enumerate(thresholds):
        ranks = {str(k): empty_counts() for k in RANKS}
        density = {name: {str(k): empty_counts() for k in RANKS} for name in ('0', '1', '2-4', '>=5')}
        areas = {name: dict(GT=0, TP=0, FN=0, FP=0, ignored_DT=0, ignored_GT=0) for name in ('small', 'medium', 'large')}
        for image in images:
            cell = image['thresholds'][ti]
            for k in RANKS:
                add_counts(ranks[str(k)], cell['rank_counts'][str(k)])
                add_counts(density[image['instance_density_group']][str(k)], cell['rank_counts'][str(k)])
            for name in areas:
                for key in areas[name]: areas[name][key] += cell['descriptive_area_counts'][name][key]
        for k in RANKS:
            require(ranks[str(k)]['TP']+ranks[str(k)]['FP']+ranks[str(k)]['ignored_DT'] == len(images)*k and
                    ranks[str(k)]['FP'] == sum(ranks[str(k)][name] for name in FP_TYPES), 'Aggregate DT/FP conservation differs')
            for key in COUNT_KEYS:
                require(ranks[str(k)][key] == sum(density[d][str(k)][key] for d in density), 'Density aggregate differs')
        output.append(dict(threshold=float(threshold), rank_counts=ranks, instance_density_counts=density,
                           descriptive_area_counts=areas))
    coverage = [g['all100_best_IoU_auxiliary'] for image in images for g in image['GT'] if not g['ignored']]
    return dict(thresholds=output, all100_nonignore_GT_best_IoU_auxiliary=dict(GT=len(coverage),
                mean=sum(coverage)/len(coverage) if coverage else None,
                at_050=sum(v >= .5 for v in coverage), at_075=sum(v >= .75 for v in coverage), not_one_to_one_recall=True),
                missed_at_075=sum(len(v['missed_at_075_GT_ids']) for v in images),
                missed075_but_matched050=sum(len(v['missed075_but_matched050_GT_ids']) for v in images))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def imported_files(np, masks, COCO, COCOeval, spec, admit_ref):
    closure = {r['path']: r for r in spec['sources']}; owner = read(bound(admit_ref))
    modules = dict(COCO=sys.modules[COCO.__module__], COCOeval=sys.modules[COCOeval.__module__],
                   masks=masks, mask_binary=masks._mask, numpy=np, numpy_binary=np.core._multiarray_umath)
    refs = {}
    for name, module in modules.items():
        ref = reference(module.__file__)
        require(closure.get(ref['path']) == ref and owner['bound_source_sha256'].get(ref['path']) == ref['sha256'],
                'Actual imported SOURCE/binary not input/owner-bound: ' + name)
        refs[name] = ref
    require('torch' not in sys.modules, 'Torch must not be imported')
    return refs


def run(spec_file):
    path = Path(spec_file); require(path.is_absolute(), 'Absolute SPEC required'); path = path.resolve(strict=True)
    spec = read(path); output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists(), 'Fresh absolute output directory required')
    source, spec_ref, admit_ref, failed = source_gate(path, spec)
    require('torch' not in sys.modules, 'Torch forbidden in this CPU audit')
    require(spec['numpy_version'] == '1.23.5' and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
            'Locked CPU-only numerical environment required')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1], 'CPU01 affinity required before scientific imports')
    import numpy as np
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    require(np.__version__ == spec['numpy_version'], 'NumPy version differs')
    actual_library_files = imported_files(np, masks, COCO, COCOeval, spec, admit_ref)
    coco = COCO(str(bound(spec['DEV']))); predictions = read(bound(spec['predictions'])); ids = population(np, coco, predictions)
    dataset = copy.deepcopy(coco); dataset.dataset.setdefault('info', {})
    evaluator = COCOeval(dataset, dataset.loadRes(copy.deepcopy(predictions)), 'segm')
    evaluator.params.imgIds = list(ids); evaluator.params.catIds = [1]; evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    stats = evaluator.stats.tolist(); require(len(stats) == 12 and len(evaluator.params.iouThrs) == 10, 'Official COCO12/10 thresholds required')
    for a, b in zip(stats, spec['expected_COCO_stats']): close(a, b, 'Recomputed official COCO12 differs')
    images = []
    for image_id in ids:
        images.append(audit_image(np, masks, evaluator, image_id))
        print(json.dumps(dict(N06_images_checked=len(images))), flush=True)
    totals = summarize(images, evaluator.params.iouThrs)
    require(sum(len(v['GT']) for v in images) == 380 and
            len({g['gt_id'] for v in images for g in v['GT']}) == 380, 'All original GT including ignored/FN must remain')
    for ref in spec['sources']: bound(ref)
    require(reference(path) == spec_ref and reference(__file__) == source and reference(spec['CPU_admission_file']) == admit_ref and
            psutil.Process().cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            actual_library_files == imported_files(np, masks, COCO, COCOeval, spec, admit_ref), 'Final inputs/resources changed')
    output.mkdir()
    write(output/'DETAILS.json', dict(schema='N06-fixed-E24-one-to-one-error-details-v1', images=images))
    report = dict(schema='N06-fixed-E24-one-to-one-error-report-v1', status=STATUS, phase=spec['phase'], source=source,
                  spec=spec_ref, CPU_admission=admit_ref, RAW_inputs=spec['RAW_inputs'],
                  prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'], prior_failure_is_not_qualification=True,
                  predictions=spec['predictions'], DEV=spec['DEV'], E24_reference_Q=spec['E24_reference_Q'], protocol=spec['protocol'],
                  details=reference(output/'DETAILS.json'), images=187, raw_GT=380, predictions_count=18700,
                  official_COCO=dict(stats=stats, iouType='segm', maxDets=[1, 10, 100], original_GT_preserved=True,
                                     score_filter=None, official_COCO_crowd_ignore_semantics=True), summary=totals,
                  primary_thresholds=[.5, .75], all_thresholds=[float(v) for v in evaluator.params.iouThrs], fixed_ranks=list(RANKS),
                  area_group_rule='Mutually exclusive descriptive bins: small<=1024; medium 1024<area<=9216; large>9216.',
                  COCO_area_boundary_rule='Official COCO uses overlapping inclusive [0,1024],[1024,9216],[9216,1e10]; descriptive bins are not APS/APM/APL.',
                  FP_rule='Unignored unmatched DT: max ordinary IoU to all area-all nonignore/noncrowd GT >=t duplicate_or_assignment_conflict; [0.1,t) geometry_candidate; <0.1 background_candidate.',
                  rank_rule='Stable score prefix K=1/10/100 of full100 greedy matching; regular GT counts as matched only when its DT is in the prefix.',
                  CPU_execution=dict(affinity=[0, 1], thread_environment=2, CUDA_VISIBLE_DEVICES='', numpy=np.__version__, Torch_imported=False),
                  actual_imported_source_and_binary_refs=actual_library_files,
                  no_training=True, no_model=True, no_score_change=True, no_NMS=True, no_cutoff=True,
                  DEV_historically_exposed=True, blind_TEST_claimed=False, causal_attribution_claimed=False,
                  evidence_limits=['Candidate FP labels do not identify causes, coastline, sensor, noise or a training mechanism.',
                      'All100 low-score tail counts alone do not establish the performance bottleneck; fixed rank1/10/100 are reported.',
                      'Best-per-GT IoU is auxiliary many-to-one coverage, not actual matching recall.',
                      'The input is one fixed qualified E24 DEV endpoint; no threshold tuning, selection or model change occurs.',
                      'ROOT must separately close this CPU job actual admission/RAW terminal after completion.'])
    write(output/'REPORT.json', report)
    print(json.dumps(dict(status=STATUS, report=reference(output/'REPORT.json'), details=report['details'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args(); require(args.owner_execute, 'Explicit ROOT-owned execution required'); run(args.spec)
