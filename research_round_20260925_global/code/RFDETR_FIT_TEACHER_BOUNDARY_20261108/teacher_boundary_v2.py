"""FIT-only boundary transfer feasibility for frozen E24 and native warm A."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import ModuleType


ROOT = Path(__file__).resolve().parent.parent
A_EVAL = ROOT / 'RFDETR_E24_WARM_NATIVE_A_EVAL_CANDIDATE_20261108_01/evaluate_a.py'
A_EVAL_SHA = '91c5bf954d19550eefc24714d649b82aea23d8b8af670c0106bcc65317406891'
GROUPS, REPEATS, QUERIES, SIDE, CHUNK = 24, 2, 100, 576, 32
MIB = 1024**2
EDGE_FIELDS = (
    'pairs', 'band_pixels', 'teacher_right_student_wrong', 'student_right_teacher_wrong',
    'teacher_FN', 'teacher_FP', 'student_FN', 'student_FP',
    'teacher_true_probability_sum', 'teacher_predicted_probability_sum',
    'teacher_true_probability_sum_on_gain', 'teacher_true_probability_sum_on_reverse',
)
VIEW_COUNT_FIELDS = (
    'targets_before', 'targets_after', 'zero_area_targets', 'teacher_reliable',
    'student_reliable', 'teacher_reliable_student_below', 'actionable',
    'unmatched_below_half', 'raw_competition',
)
COHORTS = ('all_pairs', 'teacher_reliable_student_below_pairs', 'actionable_pairs')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def assign(intersection, union):
    """Independent mask-only injection with cardinality at exact IoU 9/10 first."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    need(intersection.shape == union.shape and intersection.ndim == 2
         and intersection.shape[0] <= intersection.shape[1] <= QUERIES
         and np.issubdtype(intersection.dtype, np.integer) and np.issubdtype(union.dtype, np.integer)
         and bool(((0 <= intersection) & (intersection <= union) & (union <= SIDE**2)).all()),
         'Expected valid integer mask intersections and unions')
    targets = intersection.shape[0]
    reliable = (union > 0) & (10 * intersection >= 9 * union)
    iou = np.divide(intersection, union, out=np.zeros(intersection.shape), where=union > 0)
    rows, queries = linear_sum_assignment(iou + (targets + 1) * reliable, maximize=True)
    need(np.array_equal(rows, np.arange(targets)) and len(np.unique(queries)) == targets,
         'Expected one distinct assigned query per target in target order')
    return queries, reliable[rows, queries]


def assignment_flags(intersection, union, queries):
    import numpy as np
    rows = np.arange(len(queries))
    paired_i, paired_u = intersection[rows, queries], union[rows, queries]
    half = (paired_u > 0) & (2 * paired_i >= paired_u)
    raw_reliable = ((union > 0) & (10 * intersection >= 9 * union)).any(axis=1)
    return half, raw_reliable


def edge_stats(gt, teacher, student):
    """Measure direction of disagreement on one shared GT contour support."""
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion
    from scipy.special import expit

    need(gt.ndim == 2 and gt.dtype == np.bool_ and gt.shape == teacher.shape == student.shape,
         'Expected aligned binary GT and mask logits')
    axis = np.arange(-2, 3)
    disk = axis[:, None]**2 + axis[None, :]**2 <= 4
    band = binary_dilation(gt, structure=disk) & ~binary_erosion(gt, structure=disk)
    y, t, s = gt[band], teacher[band], student[band]
    need(bool(np.isfinite(t).all() and np.isfinite(s).all()), 'Expected finite band logits')
    t_mask, s_mask = t > 0, s > 0
    gain = (t_mask == y) & (s_mask != y)
    reverse = (s_mask == y) & (t_mask != y)
    true_prob = expit(np.where(y, t, -t).astype(np.float64))
    return dict(
        pairs=1, band_pixels=len(y), teacher_right_student_wrong=int(gain.sum()),
        student_right_teacher_wrong=int(reverse.sum()),
        teacher_FN=int((y & ~t_mask).sum()), teacher_FP=int((~y & t_mask).sum()),
        student_FN=int((y & ~s_mask).sum()), student_FP=int((~y & s_mask).sum()),
        teacher_true_probability_sum=float(true_prob.sum()),
        teacher_predicted_probability_sum=float(expit(np.abs(t).astype(np.float64)).sum()),
        teacher_true_probability_sum_on_gain=float(true_prob[gain].sum()),
        teacher_true_probability_sum_on_reverse=float(true_prob[reverse].sum()),
    )


def sum_edges(edges):
    return {field: (math.fsum(row[field] for row in edges) if 'probability_sum' in field
                    else sum(row[field] for row in edges)) for field in EDGE_FIELDS}


def summarize(views):
    need([(v['group'], v['repeat']) for v in views] ==
         [(group, repeat) for group in range(GROUPS) for repeat in range(REPEATS)],
         'Expected all fixed group/view scalar rows in order')
    need(all(v['all_pairs']['pairs'] == v['targets_after']
             and v['teacher_reliable_student_below_pairs']['pairs'] == v['teacher_reliable_student_below']
             and v['actionable_pairs']['pairs'] == v['actionable']
             and v['actionable'] + v['unmatched_below_half'] + v['raw_competition'] == v['teacher_reliable_student_below']
             and 0 <= v['teacher_reliable_student_below'] <= v['teacher_reliable'] <= v['targets_after']
             and 0 <= v['student_reliable'] <= v['targets_after'] for v in views),
         'Per-view target and reliability counts do not close')

    def totals(rows):
        result = {field: sum(row[field] for row in rows) for field in VIEW_COUNT_FIELDS}
        result['empty_views'] = sum(row['targets_after'] == 0 for row in rows)
        for cohort in COHORTS:
            result[cohort] = sum_edges([row[cohort] for row in rows])
        return result

    pairs = [views[i:i + REPEATS] for i in range(0, len(views), REPEATS)]
    group_stats = [totals(pair) for pair in pairs]
    group_coverage = dict(
        with_teacher_reliable=sum(any(view['teacher_reliable'] for view in pair) for pair in pairs),
        with_teacher_reliable_repeat0=sum(bool(pair[0]['teacher_reliable']) for pair in pairs),
        with_teacher_reliable_repeat1=sum(bool(pair[1]['teacher_reliable']) for pair in pairs),
        with_teacher_reliable_in_both_views=sum(all(view['teacher_reliable'] for view in pair) for pair in pairs),
        with_teacher_reliable_student_below=sum(bool(row['teacher_reliable_student_below']) for row in group_stats),
    )
    direction = {}
    for cohort in COHORTS:
        direction[cohort] = {}
        for unit, rows in (('views', views), ('groups', group_stats)):
            gains = [row[cohort]['teacher_right_student_wrong'] -
                     row[cohort]['student_right_teacher_wrong'] for row in rows]
            direction[cohort][unit] = dict(
                gain_exceeds_reverse=sum(n > 0 for n in gains), reverse_exceeds_gain=sum(n < 0 for n in gains),
                equal_with_nonzero_band=sum(n == 0 and row[cohort]['band_pixels'] > 0 for n, row in zip(gains, rows)),
                zero_band=sum(row[cohort]['band_pixels'] == 0 for row in rows),
            )
    return dict(total=totals(views), by_repeat=[totals(views[r::REPEATS]) for r in range(REPEATS)],
                group_coverage=group_coverage, direction_by_unit=direction)


def evaluator_helpers():
    data = A_EVAL.read_bytes()
    need(sha256(data).hexdigest() == A_EVAL_SHA, 'Frozen A evaluator differs')
    a = ModuleType('_fit_teacher_boundary_a')
    a.__file__ = str(A_EVAL)
    sys.modules[a.__name__] = a
    exec(compile(data, str(A_EVAL), 'exec'), a.__dict__)
    return a, a.load_base()


def mask_assignments(model, samples, gt, truth, torch, np, device):
    """Query masks use slot0 identity without class-score filtering or class cost."""
    raw = model(samples)
    logits, boxes, masks = (raw[key] for key in ('pred_logits', 'pred_boxes', 'pred_masks'))
    need(not model.training and model.num_queries == QUERIES and model.group_detr == 13
         and tuple(logits.shape) == (1, QUERIES, 2) and tuple(boxes.shape) == (1, QUERIES, 4)
         and tuple(masks.shape) == (1, QUERIES, 144, 144) and raw.get('pred_keypoints') is None
         and all(t.device == device and t.dtype == torch.float32 and bool(torch.isfinite(t).all())
                 for t in (logits, boxes, masks)), 'Expected native finite FP32 eval output and all 100 masks')
    if not len(gt):
        empty = np.empty(0, dtype=bool)
        return np.empty((0, SIDE, SIDE), dtype=np.float32), empty, empty, empty
    need(len(gt) <= QUERIES, 'Fixed non-padding target count exceeds available injective queries')

    def resize(planes):
        return torch.nn.functional.interpolate(planes.unsqueeze(1), (SIDE, SIDE), mode='bilinear',
                                                align_corners=False)[:, 0]

    # 0/1 products and all nonnegative partial sums on this grid are exact in FP32.
    truth_area = truth.flatten(1).sum(dim=1)
    truth_matrix = truth.flatten(1).float()
    intersections, unions = [], []
    for chunk in masks[0].split(CHUNK):
        values = resize(chunk)
        need(bool(torch.isfinite(values).all()), 'Finite resized query logits required')
        binary = (values > 0).flatten(1)
        projected = truth_matrix @ binary.float().T
        counts = projected.to(torch.int64)
        need(torch.equal(projected, counts.float()), 'Mask intersections must be integer-exact')
        intersections.append(counts)
        unions.append(truth_area[:, None] + binary.sum(dim=1)[None, :] - counts)
    intersection = torch.cat(intersections, dim=1).cpu().numpy()
    union = torch.cat(unions, dim=1).cpu().numpy()
    queries, hits = assign(intersection, union)

    assigned = masks[0].index_select(0, torch.as_tensor(queries, device=device))
    selected = np.concatenate([resize(chunk).cpu().numpy() for chunk in assigned.split(CHUNK)])
    binary = selected > 0
    rows = np.arange(len(gt))
    need(np.array_equal((binary & gt).sum(axis=(1, 2)), intersection[rows, queries])
         and np.array_equal((binary | gt).sum(axis=(1, 2)), union[rows, queries]),
         'Independent binary accounting differs for a selected query')
    half, raw = assignment_flags(intersection, union, queries)
    return selected, hits, half, raw


def run(args):
    started = time.monotonic()
    a, base = evaluator_helpers()
    need(sys.platform == 'win32' and Path(sys.executable).resolve(strict=True) == base.HOST.resolve(strict=True)
         and os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1',
         'Expected the admitted Windows runtime with offline model configuration')

    def offline(event, _args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('The FIT diagnostic is offline')

    sys.addaudithook(offline)
    acceptance = base.pinned_ref(args.acceptance.resolve(strict=True), args.acceptance_sha256)
    receipt = json.loads(base.read_bytes(acceptance, MIB))
    arm_dir = Path(receipt['arm']['checkpoint_ref']['path']).parent
    a_ref = a.validate_acceptance(base, arm_dir, args.protocol, receipt)
    out = args.output_dir.resolve()
    need(ROOT in out.parents and out.parent.is_dir() and not out.exists()
         and arm_dir not in out.parents, 'Expected fresh diagnostic output away from original checkpoints')
    base.read_bytes(base.pinned_ref(base.SETUP_PRIOR, base.SETUP_PRIOR_SHA), MIB)
    p68 = base.source_module(base.SETUP, base.SETUP_SHA, '_fit_teacher_boundary_p68')
    native = p68.read_pinned(p68.NATIVE / 'SPEC.json', p68.NATIVE_SHA)
    material = p68.read_pinned(p68.MATERIAL / 'SPEC.json', p68.MATERIAL_SHA)
    accepted = p68.read_pinned(p68.MATERIAL / 'ROOT_INDEPENDENT_MATERIALIZATION_ACCEPTED_PRIVATE.json',
                              p68.ACCEPTED_SHA)
    need(native['archive_ref']['sha256'] == base.E24_SHA
         and native['fit_image_manifest'] == material['image_manifest']
         and material['DEV'] is material['TEST'] is False
         and accepted['status'] == 'ACTUAL_FULL_FIT741_FIXED_MATCH_FEATURES_ACCEPTED',
         'Expected exact closed E24 and FIT-only input provenance')
    need(all(path.resolve() not in out.parents for path in (p68.NATIVE, p68.MATERIAL)),
         'Diagnostic output must not be in an immutable FIT source directory')
    out.mkdir()

    import numpy as np
    import torch
    from rfdetr.datasets.coco import ConvertCoco, make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.utilities.tensors import make_collate_fn

    need(torch.cuda.is_available() and torch.cuda.device_count() == 1
         and not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu'),
         'Expected the single native Windows CUDA route in eager FP32')
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    need(free >= 6 * 1024**3, 'FIT diagnostic CUDA free memory floor')
    torch.cuda.set_per_process_memory_fraction(min(.35, 6 * 1024**3 / total), device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)

    teacher, criterion = p68.setup(torch, device, native)
    del criterion
    teacher.eval()
    need(len(teacher.state_dict()) == 545, 'Exact teacher model state')
    student, criterion = p68.setup(torch, device, native)
    del criterion
    need(base.load_warm(student, 'A', a_ref, args.protocol, torch, device) == 545, 'Exact native A EMA state')
    convert = ConvertCoco(include_masks=True, include_keypoints=False, cat2label={1: 0},
                          num_keypoints_per_class=[])
    augment = make_coco_transforms_square_div_64(
        'train', SIDE, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=12, num_windows=1, aug_config=None, scale_jitter=True, gpu_postprocess=False,
        keypoint_flip_pairs=None,
    )
    collate = make_collate_fn(block_size=12, pack=True, pad_targets_to=None)
    fit = p68.doc(material['image_manifest'])
    dataset = p68.original_gt(material, fit['items'])
    annotations = defaultdict(list)
    for annotation in dataset['annotations']:
        annotations[annotation['image_id']].append(annotation)
    items = p68.sample_groups(fit['items'])[:GROUPS]
    need(len(items) == GROUPS and len({item['image_sha256'] for item in items}) == GROUPS
         and all(item['partition'] == 'head_fit' for item in items), 'Expected first 24 distinct head-fit groups')

    views = []
    with torch.inference_mode():
        for group, item in enumerate(items):
            data = p68.exact_bytes(item['image_ref'], 16 * MIB)
            image_id = item['image_id']
            for repeat in range(REPEATS):
                seed = 5301000 + 2 * item['dataset_index'] + repeat
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                image, scale = decode_pil_image_bytes(data, draft_size=None)
                h, w = item['original_hw']
                need(image.mode == 'RGB' and image.size == (w, h) and scale == (1.0, 1.0),
                     'Exact FIT image decode differs')
                image, source = convert(image, dict(image_id=image_id, annotations=annotations[image_id]))
                before = len(source['labels'])
                image, target = augment(image, source)
                gt = target['masks'].numpy()
                need(gt.dtype == np.bool_ and gt.shape == (len(target['labels']), SIDE, SIDE)
                     and bool((target['labels'] == 0).all()), 'Expected native transformed contiguous ship masks')
                samples, _ = collate([(image, target)])
                samples = samples.to(device, non_blocking=True)
                need(tuple(samples.tensors.shape) == (1, 3, SIDE, SIDE)
                     and samples.tensors.dtype == torch.float32 and not bool(samples.mask.any()),
                     'Expected one unpadded native FIT tensor')
                truth = torch.as_tensor(gt, device=device)
                t_logits, t_hits, _, _ = mask_assignments(teacher, samples, gt, truth, torch, np, device)
                s_logits, s_hits, s_half, s_raw = mask_assignments(student, samples, gt, truth, torch, np, device)
                eligible = t_hits & ~s_hits
                actionable = eligible & s_half & ~s_raw
                all_pairs = [edge_stats(y, t, s) for y, t, s in zip(gt, t_logits, s_logits, strict=True)]
                subset = [row for row, keep in zip(all_pairs, eligible, strict=True) if keep]
                action_pairs = [row for row, keep in zip(all_pairs, actionable, strict=True) if keep]
                views.append(dict(
                    group=group, repeat=repeat, targets_before=before, targets_after=len(gt),
                    zero_area_targets=int((~gt.any(axis=(1, 2))).sum()), teacher_reliable=int(t_hits.sum()),
                    student_reliable=int(s_hits.sum()), teacher_reliable_student_below=int(eligible.sum()),
                    actionable=int(actionable.sum()), unmatched_below_half=int((eligible & ~s_half & ~s_raw).sum()),
                    raw_competition=int((eligible & s_raw).sum()), all_pairs=sum_edges(all_pairs),
                    teacher_reliable_student_below_pairs=sum_edges(subset), actionable_pairs=sum_edges(action_pairs),
                ))
            if (group + 1) % 6 == 0:
                print(json.dumps(dict(event='FIT_TEACHER_BOUNDARY_PREFIX', groups=group + 1,
                                      views=len(views))), flush=True)

    torch.cuda.synchronize(device)
    report = summarize(views)
    view_ref = base.json_file(out / 'VIEW_SCALARS.json', dict(
        schema='RFDETR_FIT_TEACHER_BOUNDARY_VIEW_SCALARS_V1', rows=views,
    ), max_bytes=MIB)
    return base.json_file(out / 'RESULT.json', dict(
        schema='RFDETR_FIT_TEACHER_BOUNDARY_RESULT_V1',
        status='FIT_TEACHER_BOUNDARY_COMPLETE_PENDING_ROOT_ACCEPTANCE',
        population='FIT head_fit, first 24 deterministic unique image groups',
        source_groups=GROUPS, transformed_views=GROUPS * REPEATS, repeats_per_group=REPEATS,
        seed_rule='5301000 + 2 * dataset_index + repeat', native_train_augmentation=True,
        matching='Independent maximum cardinality at mask IoU >= 0.90 then maximum total IoU',
        actionable_rule='Teacher >=0.90, paired A in [0.50,0.90) and all raw A queries below 0.90',
        comparison='Same transformed GT index; both-view groups need not contain the same reliable instance',
        query_rule='All 100 native slot0 query masks, logits > 0 after bilinear resize, no class scores',
        boundary_rule='Shared GT Euclidean radius-2 disk dilation minus erosion, outside grid false',
        probability_rule='Teacher mask sigmoid on true sign and predicted sign; values are sums',
        precision='CUDA eager FP32, high matmul, cuDNN TF32', teacher_E24_sha256=base.E24_SHA,
        native_A_ema_sha256=a_ref['sha256'], native_A_protocol=args.protocol,
        fit_image_manifest_sha256=material['image_manifest']['sha256'], P68_source_sha256=base.SETUP_SHA,
        view_scalars_sha256=view_ref['sha256'], AP=False, DEV=False, TEST=False, new_training=False,
        terminal_self_certified=False, peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
        elapsed_seconds=time.monotonic() - started, **report,
    ), max_bytes=MIB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--acceptance-sha256', required=True)
    parser.add_argument('--protocol', required=True, help='Closed native A training protocol commit')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args)), flush=True)


if __name__ == '__main__':
    main()
