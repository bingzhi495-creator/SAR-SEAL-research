"""Measure post-hoc native proposal and exact geometry ceilings without changing score."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec


PROTOCOL_SHA = '418d9701f2b87fc85b6086429de03c1fb34ec5029fa0c18938558e47291e1feb'
SCORER_SHA = 'd4d16cdfab0903480c7e6f3686de973886b726d54f91b7e235686c1502d44735'
SCORE_SHA = 'fa2508ebde848235dada628406f095a330955f103fa611256964a3f714ea9972'
SCORE_TERMINAL_SHA = 'd09229a37a456a1f267303e03237fb54f7c813cd546e9cfaae833ae06e9cc430'
EDIT_SHA = 'b40dc74dc8414b30e115f04911e4da0b16b1a18ffb875636d5a9f05be6a9fe06'
EDIT_TERMINAL_SHA = 'd7591b15c9bb9a79a7a2bdd5075fadc3b16a6eb1a5c8c4d1e9247e3c06cd51c5'
EDITOR_SHA = 'd753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb'
HULL_THRESHOLDS = (0.50, 0.75, 0.90)
ATTEMPTED = dict(primary='distance_paired', attraction='dynamic_attraction')
LADDER = ('native_matched', 'other_native_mask_reaches', 'endpoint_cap_reaches',
          'box_fixed_area_reaches', 'box_free_area_reaches', 'no_box_local_method_reaches')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, 'File pin mismatch: ' + path.name)
    return raw


def module(path, expected):
    result = ModuleType(path.stem)
    result.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), 'exec'), result.__dict__)
    return result


def intersection_gain(fg, bg, k):
    return min(k, fg) - max(0, k - bg)


def exact_gains(gt, coordinates, k):
    require(0 <= k <= len(coordinates), 'Attained count within available support')
    add = gt.flat[coordinates[:, 0]]
    drop = gt.flat[coordinates[:, 1]]
    gains = add.astype(np.int8) - drop
    return int(np.sort(gains)[::-1][:k].sum()), intersection_gain(int(add.sum()), int((~drop).sum()), k)


def capped_gains(gt, coordinates, cap):
    require(0 <= cap <= len(coordinates), 'Cap within available support')
    add = gt.flat[coordinates[:, 0]]
    drop = gt.flat[coordinates[:, 1]]
    return min(cap, int((add & ~drop).sum())), min(cap, int(add.sum()), int((~drop).sum()))


def iou(hit, area, truth_area):
    return hit / (area + truth_area - hit) if hit else 0.


def box_iou(box, xywh):
    x0, y0, x1, y1 = box
    x, y, w, h = xywh
    hit = max(0., min(x1, x+w)-max(x0, x)) * max(0., min(y1, y+h)-max(y0, y))
    return iou(hit, (x1-x0)*(y1-y0), w*h)


def box_limits(stock, truth, crop, area, gt_area, hit):
    x0, y0, x1, y1 = crop
    native_crop = stock[y0:y1, x0:x1]
    truth_crop = truth[y0:y1, x0:x1]
    inside_area = int(native_crop.sum())
    truth_inside = int(truth_crop.sum())
    hit_outside = hit - int((native_crop & truth_crop).sum())
    unconstrained_hit = hit_outside + truth_inside
    return (iou(hit_outside + min(inside_area, truth_inside), area, gt_area),
            iou(unconstrained_hit, area - inside_area + truth_inside, gt_area), truth_inside / gt_area)


def fallback_assign(truth, matches, boxes):
    used = {position for _, position in matches.values()}
    unassigned = {row['id']: row for row in truth if row['id'] not in matches}
    fallback = {}
    for position, box in enumerate(boxes, 1):
        if position in used or not unassigned:
            continue
        ranked = [(box_iou(box, hull['bbox']), -aid) for aid, hull in unassigned.items()]
        overlap, reverse_id = max(ranked)
        if overlap >= 0.50:
            aid = -reverse_id
            del unassigned[aid]
            fallback[aid] = position
    return fallback


def supports(image_audit, candidates, base, methods, capture):
    planes = base.load_array(capture, image_audit['logits_ref']) if image_audit['logit_ranks'] else ()
    ranks = dict(zip(image_audit['logit_ranks'], planes, strict=True))
    overrides = {o['native_rank']: o for o in image_audit['posterior_overrides']}
    for candidate, audit in zip(candidates, image_audit['candidates'], strict=True):
        rank = audit['native_rank']
        mask = codec.decode(candidate['native']).astype(bool)
        area = audit['mask_area']
        require(int(mask.sum()) == area and (rank in ranks) == (32 <= area < 1024), 'Native raster area and eligibility')
        box = np.asarray(audit['box_xyxy'], dtype=np.float32)
        require(box.shape == (4,) and np.isfinite(box).all() and np.all(box[:2] <= box[2:])
                and np.all((box >= 0) & (box <= (*mask.shape[::-1], *mask.shape[::-1]))), 'Admitted native box')
        coords = np.empty((0, 2), dtype=np.int32)
        if rank in ranks:
            override = overrides.get(rank)
            prob = base.native_probability(ranks[rank], mask, None if override is None else base.load_array(capture, override))
            endpoints = base.expected_support(methods.rank, mask, box, prob)
            if len(endpoints[0]):
                coords = methods.pair.pair_assignment(*endpoints, mask.shape[1])
            base.check_support(methods, mask, box, prob, coords)
        require(not mask.flat[coords[:, 0]].any() and mask.flat[coords[:, 1]].all()
                and np.unique(coords).size == coords.size and len(coords) <= 32, 'Exact signed unique original endpoints')
        yield candidate, mask, area, tuple(map(float, box)), coords


def internal_edits(candidate, stock, coords, names, editor):
    ledger = {key: dict(swaps=candidate['counts'][name], rle=candidate['masks'][name])
              for key, name in names.items() if name in candidate['counts']}
    editor.validate_edits(stock, coords, ledger)
    if 'fixed' in ledger:
        fixed = codec.decode(ledger['fixed']['rle']).astype(bool)
        added = np.isin(coords[:, 0], np.flatnonzero(fixed & ~stock))
        removed = np.isin(coords[:, 1], np.flatnonzero(stock & ~fixed))
        require(np.array_equal(added, removed), 'Attained distance arm uses the exact original pair identities')


def read_archive(path, scorer, names):
    with path.open('rb') as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        require(digest.hexdigest() == scorer.ARCHIVE_SHA, 'Original whole author archive identity')
        with ZipFile(stream) as package:
            return scorer.exact_ground_truth(package, names)


def candidate_geometry(stock, area, box, coords, candidate, hull, truth):
    gt_area = hull['area']
    hit = int((stock & truth).sum())
    cap = min(8, area // 10, len(coords))
    cap_paired, cap_free = capped_gains(truth, coords, cap)
    crop = np.ceil(box).astype(int).tolist()
    fixed_box, free_box, crop_only = box_limits(stock, truth, crop, area, gt_area, hit)
    native = iou(hit, area, gt_area)
    result = dict(native=native, box_iou=box_iou(box, hull['bbox']),
                  capped_distance=iou(hit + cap_paired, area, gt_area),
                  capped_independent=iou(hit + cap_free, area, gt_area),
                  box_local_fixed_area=fixed_box, box_local_free_area=free_box, crop_only=crop_only)
    for family, arm in ATTEMPTED.items():
        k = candidate['counts'].get(arm, 0)
        require(0 <= k <= cap, 'Observed original count within reconstructed cap')
        paired, free = exact_gains(truth, coords, k)
        result[family + '_exact_k_distance'] = iou(hit + paired, area, gt_area)
        result[family + '_exact_k_independent'] = iou(hit + free, area, gt_area)
    require(native <= result['capped_distance'] <= result['capped_independent'] <= fixed_box <= free_box,
            'Nested exact geometry bounds')
    return result, cap


def summarize(rows, groups):
    if not rows:
        return dict(hulls=0, ship_bearing_groups=0, metrics={})
    names = list(rows[0][1])
    data = np.asarray([[metric[name] for name in names] for _, metric in rows])
    parent = np.array([groups[image] for image, _ in rows])
    group_means = np.asarray([data[parent == g].mean(axis=0) for g in sorted(set(parent))])
    return dict(hulls=len(rows), ship_bearing_groups=len(group_means), metrics={
        name: dict(hull_mean_iou_points=100*float(data[:, j].mean()),
                   parent_mean_iou_points=100*float(group_means[:, j].mean()),
                   coverage={format(t, '.2f'): int((data[:, j] >= t).sum()) for t in HULL_THRESHOLDS})
        for j, name in enumerate(names)})


def classify(matched, possible):
    if matched:
        return LADDER[0]
    for label, name in zip(LADDER[1:-1], ('native', 'capped_independent', 'box_local_fixed_area', 'box_local_free_area'), strict=True):
        if possible[name] >= .50:
            return label
    return LADDER[-1]


def scan(truth, all_matches, records, audits, groups, base, methods, capture, scorer, editor):
    truth_images = defaultdict(list)
    for hull in truth['annotations']:
        truth_images[hull['image_id']].append(hull)
    result = defaultdict(list)
    ladder, exposure = Counter(), Counter()
    for audit in audits:
        iid = audit['image_id']
        hulls = truth_images[iid]
        if not hulls:
            continue
        anchors = {hull['id']: all_matches[hull['id']] for hull in hulls if hull['id'] in all_matches}
        boxes = [v['box_xyxy'] for v in audit['candidates']]
        fallback = fallback_assign(hulls, anchors, boxes)
        max_values = defaultdict(Counter)
        assigned = {}
        truth_masks = {hull['id']: codec.decode(hull['segmentation']).astype(bool) for hull in hulls}
        for position, (candidate, stock, area, box, coords) in enumerate(
                supports(audit, records[iid], base, methods, capture), 1):
            internal_edits(candidate, stock, coords, scorer.ARMS, editor)
            for hull in hulls:
                aid = hull['id']
                values, cap = candidate_geometry(stock, area, box, coords, candidate, hull, truth_masks[aid])
                for key, value in values.items():
                    max_values[aid][key] = max(max_values[aid][key], value)
                if anchors.get(aid) != (iid, position) and fallback.get(aid) != position:
                    continue
                values.update({arm: scorer.hull_measures(candidate['masks'].get(arm, candidate['native']),
                                                         hull['segmentation'], hull['area'])[0]/100
                               for arm in scorer.ARMS.values()})
                require(values['distance_paired'] <= values['primary_exact_k_distance'] + 1e-12
                        and values['free_matched_count'] <= values['primary_exact_k_independent'] + 1e-12
                        and all(values[arm] <= values['attraction_exact_k_independent'] + 1e-12
                                for arm in ('dynamic_attraction', 'static_at_dynamic_count', 'learned_at_attraction_count')),
                        'Original observed geometry below its exact equal-count oracle')
                assigned[aid] = values
                scope = 'native' if aid in anchors else 'box_fallback'
                for key, check in dict(endpoint_cap=cap, primary_k=candidate['counts'].get('distance_paired', 0),
                                       attraction_k=candidate['counts'].get('dynamic_attraction', 0)).items():
                    exposure[scope + '_' + key + '_positive_hulls'] += bool(check)
                    exposure[scope + '_' + key + '_total'] += check
        for hull in hulls:
            aid = hull['id']
            possible = max_values[aid]
            require(bool(possible), 'At least one mapped candidate on each ship-bearing chip')
            result['any_candidate_all_hulls'].append((iid, possible))
            value = assigned.get(aid)
            if aid in anchors:
                result['any_candidate_native_matched'].append((iid, possible))
                result['native_fixed_matched_only'].append((iid, value))
                result['native_fixed_all_hulls'].append((iid, value))
            else:
                result['any_candidate_native_unmatched'].append((iid, possible))
                if aid in fallback:
                    result['residual_box_fallback_matched_only'].append((iid, value))
                result['native_fixed_all_hulls'].append((iid, None))
                result['residual_box_fallback_unmatched_denominator'].append((iid, value))
            ladder[classify(aid in anchors, possible)] += 1
    fixed_keys = result['native_fixed_matched_only'][0][1].keys()
    for name in ('native_fixed_all_hulls', 'residual_box_fallback_unmatched_denominator'):
        result[name] = [(iid, values if values is not None else dict.fromkeys(fixed_keys, 0.)) for iid, values in result[name]]
    require([len(result[k]) for k in ('any_candidate_all_hulls', 'any_candidate_native_matched',
                                     'any_candidate_native_unmatched')] == [139, 92, 47]
            and sum(ladder.values()) == 139 and ladder['native_matched'] == 92, 'Complete fixed truth cohorts and disjoint counts')
    return dict(geometry={key: summarize(rows, groups) for key, rows in result.items()},
                disjoint_iou50_ladder={name: ladder[name] for name in LADDER}, exposure=dict(exposure))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('protocol', 'score-run', 'capture-run', 'code', 'archive', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.parent.is_dir(), 'Fresh diagnosis output only')
    started = perf_counter()
    pinned(args.protocol, PROTOCOL_SHA)
    scorer = module(args.score_run / 'score_sarssp.py', SCORER_SHA)
    score = json.loads(pinned(args.score_run / 'score/RESULT_SAFE.json', SCORE_SHA))
    terminal = json.loads(pinned(args.score_run / 'observed_score/TERMINAL.json', SCORE_TERMINAL_SHA))
    require(terminal['result_sha256'] == SCORE_SHA and terminal['actual_exit'] == 0
            and terminal['output_footer_verified'] is True and terminal['worker_sha256'] == SCORER_SHA,
            'Author standalone score result closed and verified')
    args.edit_receipt_sha, args.edit_terminal_sha = EDIT_SHA, EDIT_TERMINAL_SHA
    records, names, _, capture_receipt, _ = scorer.qualified_records(args)
    capture = args.capture_run / 'capture'
    audits = scorer.read_stream(capture / 'CANDIDATE_AUDIT_PRIVATE.jsonl', capture_receipt['audit_sha256'])
    editor = module(args.code / 'UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py', EDITOR_SHA)
    base = module(args.code / editor.BASE / 'replay.py', editor.REPLAY_SHA)
    base.torch.set_num_threads(1)
    base.torch.set_num_interop_threads(1)
    base.torch.set_float32_matmul_precision('highest')
    methods = base.methods(args.code)
    evaluate = scorer.load_evaluator(args.code / scorer.EVALUATOR_PATH)
    truth, groups, _ = read_archive(args.archive, scorer, names)
    native_ap = evaluate(truth, scorer.coco_rows(records, 'native'))['AP']
    require(100 * native_ap == score['coco_mask_metric_points']['native']['AP'], 'Do not change original native score')
    pairs = scorer.native_matches(truth, records, native_ap)
    require(len(pairs) == 92 and not base.torch.cuda.is_initialized(), 'Original .50 mask matches and CPU diagnostics')
    print(json.dumps(dict(stage='input_parity_and_original_matching_verified', hulls=139, matched=len(pairs))), flush=True)
    result = scan(truth, pairs, records, audits, groups, base, methods, capture, scorer, editor)
    original_iou = score['fixed_native_match_all_truth']['arms']['native']['iou_points']
    diagnosis_iou = result['geometry']['native_fixed_all_hulls']['metrics']['native']['parent_mean_iou_points']
    require(abs(original_iou - diagnosis_iou) < 1e-12, 'Original all-hull fixed geometry parity')
    result.update(status='SAR_SSP_VAL232_B5_PROPOSAL_AND_EDIT_ORACLE_V1', phase='score',
                  protocol_sha256=PROTOCOL_SHA, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_score_sha256=SCORE_SHA, edit_receipt_sha256=EDIT_SHA, archive_sha256=scorer.ARCHIVE_SHA,
                  original_images=len(records), original_candidates=sum(map(len, records.values())),
                  source_groups=len(set(groups.values())), native_iou50_matches=len(pairs),
                  native_iou50_unmatched=len(truth['annotations']) - len(pairs), oracle_uses_author_masks=True,
                  per_hull_maxima_allow_candidate_reuse=True, original_assignment_changed=False,
                  any_oracle_ap_computed=False, optimizer_updates=0, detector_forwards=0,
                  seconds=perf_counter()-started)
    data = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    args.output.mkdir()
    with (args.output / 'RESULT_SAFE.json').open('xb') as out:
        out.write(data)
    print(json.dumps(dict(status=result['status'], result_sha256=sha256(data).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
