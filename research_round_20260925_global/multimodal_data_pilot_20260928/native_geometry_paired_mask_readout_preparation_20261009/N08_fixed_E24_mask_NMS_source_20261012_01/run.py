"""Fixed E24 mask greedy NMS at the sole prelocked IoU threshold 0.80.

No training, mask/score modification, GT-based suppression, or cutoff occurs.
This ordinary postprocessing comparator is not claimed as an innovation.
Only ROOT may execute this future owned CPU consumer; SOURCE is not a result.
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

SCHEMA = 'ROOT-N08-fixed-E24-mask-NMS-spec-v1'
STATUS = 'ROOT_N08_FIXED_E24_MASK_NMS_CPU_COMPLETE'
E24_SCHEMA = 'ROOT-fixed-E24-native0-export-COCO-reference-recompute-CPU-qualification-v1'
E24_STATUS = 'ROOT_FIXED_E24_NATIVE0_EXPORT_COCO_REFERENCE_RECOMPUTED_CPU'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
NMS_THRESHOLD = 0.80


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
            len(spec['RAW_inputs']) == 17 and all(set(RAW_KEYS) <= set(r) for r in spec['RAW_inputs']) and
            len({r['receipt']['path'] for r in spec['RAW_inputs']}) == 17, 'Exactly seventeen normal RAWs required')
    require(type(spec['NMS_threshold']) is float and spec['NMS_threshold'] == NMS_THRESHOLD, 'Sole prelocked NMS threshold must be 0.80')
    raws = [terminal(r['receipt'], r['phase']) for r in spec['RAW_inputs']]
    failed = terminal(spec['prior_failed_CPU_receipt'], exit_code=1)  # Eighteen states before the first SHA/import.
    require(spec['prior_failed_CPU_receipt']['path'] not in {r['receipt']['path'] for r in spec['RAW_inputs']}, 'Failure cannot qualify an ancestor')
    for ref in spec['sources']: ref_shape(ref)
    closure = {r['path']: r for r in spec['sources']}; require(len(closure) == len(spec['sources']), 'Unique input closure required')
    refs = [spec[k] for k in ('prior_failed_CPU_receipt', 'predictions', 'DEV', 'E24_reference_Q', 'entry_source', 'protocol')]
    refs += [r[k] for r in spec['RAW_inputs'] for k in RAW_KEYS if k != 'phase']
    require(all(r in closure.values() for r in refs), 'All input refs must be sources-bound')
    for ref in spec['sources']: bound(ref)
    source, spec_ref = reference(__file__), reference(path)
    require(source == spec['entry_source'], 'Actual N08 SOURCE differs')
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
    require(type(spec['expected_base_COCO_stats12']) is list and len(spec['expected_base_COCO_stats12']) == 12 and
            len(q['COCO']['stats']) == 12, 'Twelve locked metrics required')
    for a, b in zip(spec['expected_base_COCO_stats12'], q['COCO']['stats']):
        close(a, b, 'Locked E24 qualification stats differ')
    require(q['COCO']['stats'] == spec['expected_base_COCO_stats12'], 'Locked E24 COCO12 must be exactly equal')
    return source, spec_ref, admit_ref, failed


def close(a, b, message):
    require(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and
            -1 <= a <= 1 and -1 <= b <= 1 and abs(a-b) <= 1e-12, message)

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


def greedy_keep(scores, ious, threshold=0.80):
    """Pure boundary-testable greedy NMS; return local indices, not rows.

    Stable descending score ties preserve original candidate order. Only a
    kept candidate suppresses later candidates whose ordinary IoU is >0.80.
    Kept indices are returned in original input order. Empty masks have IoU0
    in the caller's ordinary noncrowd matrix; this function does not use GT.
    """
    require(type(threshold) is float and threshold == NMS_THRESHOLD, 'Threshold tuning forbidden')
    require(type(scores) is list and 1 <= len(scores) <= 100 and type(ious) is list and len(ious) == len(scores), 'Small or all100 matrix required')
    count = len(scores)
    require(all(type(s) in (int, float) and math.isfinite(s) and 0 <= s <= 1 for s in scores), 'Finite original scores required')
    require(all(type(row) is list and len(row) == count and all(type(v) in (int, float) and
                math.isfinite(v) and 0 <= v <= 1 for v in row) for row in ious), 'Finite square ordinary IoU matrix required')
    order = sorted(range(count), key=lambda i: -scores[i])
    suppressed, kept = set(), []
    for position, index in enumerate(order):
        if index in suppressed:
            continue
        kept.append(index)
        for other in order[position+1:]:
            if ious[index][other] > threshold:
                suppressed.add(other)
    require(not set(kept) & suppressed and set(kept) | suppressed == set(range(count)), 'Suppression conservation differs')
    return dict(keep_indices=sorted(kept), suppressed_indices=sorted(suppressed),
                kept_score_order=kept, stable_score_order=order)


def coco12(COCOeval, coco, predictions, image_ids):
    dataset = copy.deepcopy(coco); dataset.dataset.setdefault('info', {})
    evaluator = COCOeval(dataset, dataset.loadRes(copy.deepcopy(predictions)), 'segm')
    evaluator.params.imgIds = list(image_ids); evaluator.params.catIds = [1]; evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    stats = evaluator.stats.tolist()
    require(len(stats) == 12 and len(evaluator.params.iouThrs) == 10, 'Official COCO12/10 thresholds required')
    for value in stats: close(value, value, 'Finite official metric or genuine -1 required')
    return stats


def run(spec_file):
    path = Path(spec_file); require(path.is_absolute(), 'Absolute SPEC required'); path = path.resolve(strict=True)
    spec = read(path); output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists(), 'Fresh absolute output directory required')
    source, spec_ref, admit_ref, failed = source_gate(path, spec)
    require('torch' not in sys.modules, 'Torch forbidden in this CPU comparator')
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
    baseline = coco12(COCOeval, coco, predictions, ids)
    require(baseline == spec['expected_base_COCO_stats12'], 'Recomputed complete baseline COCO12 differs')
    keep_flags, images = [False] * len(predictions), []
    for image_index, image_id in enumerate(ids):
        rows = predictions[image_index*100:(image_index+1)*100]
        rles = [rle_bytes(row['segmentation']) for row in rows]
        ious = np.asarray(masks.iou(rles, rles, [0] * 100), dtype=np.float64)
        require(ious.shape == (100, 100) and bool(np.isfinite(ious).all()) and
                bool(((ious >= 0) & (ious <= 1)).all()), 'Finite ordinary noncrowd all100 pair IoU required')
        empty = np.asarray(masks.area(rles)) == 0
        require(bool((ious[empty, :] == 0).all()) and bool((ious[:, empty] == 0).all()), 'Empty mask IoU must remain zero')
        scores = [row['score'] for row in rows]; selection = greedy_keep(scores, ious.tolist())
        require(selection['stable_score_order'] == np.argsort([-s for s in scores], kind='mergesort').tolist(), 'Stable score ties differ')
        for index in selection['keep_indices']: keep_flags[image_index*100+index] = True
        images.append(dict(image_id=image_id, candidates=100, kept=len(selection['keep_indices']),
                           suppressed=len(selection['suppressed_indices']),
                           kept_query_ids=[rows[i]['query_id'] for i in selection['keep_indices']],
                           suppressed_query_ids=[rows[i]['query_id'] for i in selection['suppressed_indices']],
                           stable_score_query_ids=[rows[i]['query_id'] for i in selection['stable_score_order']],
                           empty_masks=int(empty.sum())))
        print(json.dumps(dict(N08_images_checked=len(images))), flush=True)
    kept_predictions = [row for row, keep in zip(predictions, keep_flags) if keep]
    require(len(kept_predictions) == sum(row['kept'] for row in images) and
            len(kept_predictions) + sum(row['suppressed'] for row in images) == 18700, 'Complete keep/suppression conservation differs')
    nms_stats = coco12(COCOeval, coco, kept_predictions, ids)
    for ref in spec['sources']: bound(ref)
    require(reference(path) == spec_ref and reference(__file__) == source and reference(spec['CPU_admission_file']) == admit_ref and
            psutil.Process().cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            actual_library_files == imported_files(np, masks, COCO, COCOeval, spec, admit_ref), 'Final inputs/resources changed')
    output.mkdir()
    write(output/'PREDICTIONS.json', kept_predictions)
    report = dict(schema='N08-fixed-E24-mask-NMS-report-v1', status=STATUS, phase=spec['phase'], source=source,
                  spec=spec_ref, CPU_admission=admit_ref, RAW_inputs=spec['RAW_inputs'],
                  prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'], prior_failure_is_not_qualification=True,
                  original_predictions=spec['predictions'], kept_predictions=reference(output/'PREDICTIONS.json'),
                  DEV=spec['DEV'], E24_reference_Q=spec['E24_reference_Q'], protocol=spec['protocol'],
                  images=187, raw_GT=380, original_candidates=18700, kept_candidates=len(kept_predictions),
                  suppressed_candidates=18700-len(kept_predictions), image_suppression=images,
                  baseline_COCO=dict(stats=baseline, iouType='segm', maxDets=[1, 10, 100]),
                  NMS_COCO=dict(stats=nms_stats, iouType='segm', maxDets=[1, 10, 100]),
                  COCO_delta=[a-b for a, b in zip(nms_stats, baseline)],
                  undefined_metric_indices=[i for i, (a, b) in enumerate(zip(nms_stats, baseline)) if a == -1 or b == -1],
                  NMS_threshold=NMS_THRESHOLD, suppression_rule='Stable greedy mask NMS, ordinary iscrowd=0 IoU >0.80 only.',
                  output_order='Original prediction input order after suppression.', score_geometry_category_query_fields_unchanged=True,
                  empty_mask_IoU=0, GT_used_for_suppression=False, threshold_tuned=False, no_training=True, no_model=True,
                  actual_imported_source_and_binary_refs=actual_library_files,
                  CPU_execution=dict(affinity=[0, 1], thread_environment=2, CUDA_VISIBLE_DEVICES='', numpy=np.__version__, Torch_imported=False),
                  DEV_historically_exposed=True, blind_TEST_claimed=False, innovation_claimed=False, causal_attribution_claimed=False,
                  evidence_limits=['Fixed ordinary postprocessing comparator at one prelocked threshold; no threshold sweep or selection.',
                      'Original masks and scores are preserved; AP differences do not isolate a learned error mechanism.',
                      'COCO_delta is undefined as a performance change when either recorded metric is genuine -1.',
                      'ROOT must separately close this CPU owner actual terminal after completion.'])
    write(output/'REPORT.json', report)
    print(json.dumps(dict(status=STATUS, report=reference(output/'REPORT.json'), kept_predictions=report['kept_predictions'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args(); require(args.owner_execute, 'Explicit ROOT-owned execution required'); run(args.spec)
