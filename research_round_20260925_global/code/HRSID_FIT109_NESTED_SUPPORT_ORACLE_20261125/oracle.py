"""Compute HRSID FIT109 exact same-budget support ceilings without training or AP."""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.ndimage import binary_dilation, binary_erosion


PROTOCOL = '584344c3e0533fcac919a55f5da253085a4985b9c38ed7f4cc62f3ef779df096'
CAPTURE = '7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f'
BASIS = 'fc759d9cf1982f3970f3b75627f8eccbca0b474c97ecb141ae487ba7dacc3e94'
HEAD_RECEIPT = '476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a'
HEAD_RUN = '114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64'
FORMAL = 'RFDETR_MATCHED_POINT_FORMAL_20261120'
SPOT = 'RFDETR_CORE36_MATCHED_POINT_SPOT_20261120'
PAIR = 'DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24'
COLUMNS = ('native', 'actual_distance', 'b5_equal', 'r2_equal', 'r4_equal', 'b5_reallocated', 'r4_reallocated')
CONTRASTS = {
    'primary_support_r4_minus_b5': (('r4_equal', 1), ('b5_equal', -1)),
    'uncertainty_gate_r2_minus_b5': (('r2_equal', 1), ('b5_equal', -1)),
    'extra_rings_r4_minus_r2': (('r4_equal', 1), ('r2_equal', -1)),
    'allocation_on_b5': (('b5_reallocated', 1), ('b5_equal', -1)),
    'allocation_on_r4': (('r4_reallocated', 1), ('r4_equal', -1)),
    'support_after_allocation': (('r4_reallocated', 1), ('b5_reallocated', -1)),
    'support_allocation_interaction': (('r4_reallocated', 1), ('r4_equal', -1), ('b5_reallocated', -1), ('b5_equal', 1)),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, 'Exact digest of ' + path.name)
    return raw


def module(path, digest):
    loaded = ModuleType('_nested_' + path.stem)
    loaded.__file__ = str(path)
    sys.modules[loaded.__name__] = loaded
    exec(compile(pinned(path, digest), str(path), 'exec'), loaded.__dict__)
    return loaded


def label_free(args):
    code = args.code
    cohort = module(code / FORMAL / 'run_fit109.py', '9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e')
    spot = module(code / SPOT / 'run_spot.py', '75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee')
    plan = module(code / FORMAL / 'fit_plan.py', '1fd0fe8f815b5ad4e01d0d8eeca0aa0e1480433292fc48f94fee17e5837e5c35')
    step = module(code / SPOT / 'fit_step.py', '94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1')
    rank = module(code / PAIR / 'fit109_probability_rank/rank_math.py', spot.RANK_SHA)
    pair = module(code / PAIR / 'fit109_paired_point/pair_math.py', spot.PAIR_SHA)
    torch = step.torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    need(sys.flags.isolated and sys.dont_write_bytecode and not torch.cuda.is_initialized(), 'Isolated CPU replay')
    args.acceptance_sha256 = CAPTURE
    roster, capture = cohort.read_population(args, spot)
    readback = json.loads(pinned(args.basis / 'INDEPENDENT_READBACK_SAFE.json', BASIS))
    need(readback['independent_readback_accepted'] and readback['old_features_reproduced']
         and readback['private_file_count'] == len(roster) == 109 and readback['accepted_capture_sha256'] == CAPTURE
         and readback['original_os_exit'] == 0 and readback['GT_opened'] is False, 'Original prelabel basis verified')
    head_receipt = json.loads(pinned(args.heads / 'INDEPENDENT_SAFE.json', HEAD_RECEIPT))
    fit = json.loads(pinned(args.heads / 'RESULT_SAFE.json', HEAD_RUN))
    need(head_receipt['fit_result_sha256'] == HEAD_RUN and head_receipt['run_natural_exit'] == 0
         and head_receipt['checkpoint_bundles_verified'] == 18 and fit['detector_checkpoint_sha256'] == cohort.CHECKPOINT_SHA,
         'Closed matching point-head checkpoints')
    trainer = step.load_trainer(code / PAIR / 'fit109_paired_point/train_point.py')
    folds = plan.source_folds([r['group'] for r, _, _ in roster])
    models = {}
    head_digests = []
    for entry in fit['all_initializations']:
        fold = entry['fold']
        if entry['seed'] == 5217 and fold is not None:
            expected_name = f'5217_fold_{fold}.npz'
            need(entry['relative_weights'] == expected_name, 'Exact original held-fold file')
            heads, center, scale = plan.restore_bundle(args.heads / 'POINT_HEADS_PRIVATE' / expected_name,
                                                       entry['weights_sha256'], trainer, step)
            models[fold] = heads['S'], center, scale
            head_digests.append(entry['weights_sha256'])
    need(set(models) == set(range(5)), 'All five frozen held-fold point heads')
    stocks = [json.loads(line) for line in pinned(args.capture / 'STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl',
                                                 capture['stock_predictions_sha256']).splitlines()]
    originals, infos, files = {}, {}, []
    support_count, point_count, positive_count = 0, 0, 0
    for (identity, ordinal, block), stock in zip(roster, stocks, strict=True):
        iid, group = identity['image_id'], identity['group']
        rows = spot.native_candidate_rows(block)
        expected = [dict(image_id=iid, category_id=1, score=c['score'], segmentation=c['rle']) for c in rows]
        need(stock['image_id'] == iid == block['image_id'] and expected == stock['coco']
             and stock['native_ranks'] == [p['rank'] for p in rows], 'Exact lookup to stock candidate rows')
        boxes = spot.author_boxes(block['raw_boxes_cxcywh'], torch)
        raw = (args.basis / f'NATIVE_ENDPOINT_BASIS_{ordinal:02d}_PRIVATE.npz').read_bytes()
        files.append(dict(ordinal=ordinal, sha256=sha256(raw).hexdigest(), bytes=len(raw)))
        with np.load(BytesIO(raw), allow_pickle=False) as z:
            payload = {name: z[name] for name in ('image_id', 'ranks', 'query_ids', 'offsets', 'coords', 'old14')}
        ids = list(zip(payload['ranks'].tolist(), payload['query_ids'].tolist(), strict=True))
        offsets, coords, features = payload['offsets'], payload['coords'], payload['old14']
        need(payload['image_id'] == iid and len(set(ids)) == len(ids) and len(offsets) == len(ids)+1
             and offsets[0] == 0 and offsets[-1] == len(coords) == len(features)
             and coords.shape == (len(coords), 2) and features.shape == (len(coords), 2, 14)
             and features.dtype == np.float32 and np.isfinite(features).all(), 'Exact accepted basis schema')
        support = {key: slice(a, b) for key, a, b in zip(ids, offsets[:-1], offsets[1:], strict=True)}
        native = {(c['rank'], c['query_id']): c for c in rows}
        need(set(support) <= native.keys(), 'Captured points use original candidate IDs')
        actions = {}
        for key, segment in support.items():
            p = coords[segment]
            n = len(p)
            area = int(codec.area(native[key]['rle']))
            need(1 <= n <= 32 and 32 <= area < 1024 and np.unique(p).size == p.size
                 and np.all((p >= 0) & (p < 800*800)), 'Original assigned signed endpoint support and area')
            q = step.predict(models[folds[group]][0], features[segment], *models[folds[group]][1:])
            selected = pair.choose_pairs(q[:, 0] - q[:, 1], min(8, area // 10, n))
            actions[key] = (p, selected)
            positive_count += bool(selected)
        support_count += len(support)
        point_count += len(coords)
        originals[iid] = expected
        infos[iid] = (identity, rows, boxes, actions)
    need(sha256(spot.encode(files)).hexdigest() == readback['private_output_sha256']
         and support_count == readback['candidate_sets'] and point_count == readback['point_pairs']
         and sum(map(len, originals.values())) == capture['native_mapped_rows'] and positive_count > 0
         and not torch.cuda.is_initialized(), 'Exact full independently accepted point archives')
    return originals, infos, rank, head_digests, positive_count


def fixed_coco(gt, originals):
    flat = [p for iid in sorted(originals) for p in originals[iid]]
    id_to_position = dict(enumerate(((iid, p) for iid in sorted(originals) for p in range(len(originals[iid]))), 1))
    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(gt)
        ref.createIndex()
        detections = ref.loadRes(deepcopy(flat))
        ev = COCOeval(ref, detections, 'segm')
        ev.params.catIds = [1]
        ev.params.imgIds = sorted(originals)
        ev.params.iouThrs = np.asarray([0.50])
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
    result = {}
    for entry in ev.evalImgs:
        if entry is None or entry['aRng'] != ev.params.areaRng[0] or entry['maxDet'] != 100:
            continue
        for aid, did, ignored in zip(entry['gtIds'], entry['gtMatches'][0], entry['gtIgnore'], strict=True):
            if did and not ignored:
                iid, index = id_to_position[int(did)]
                need(iid == entry['image_id'] and aid not in result, 'Exact score-ordered COCO native match')
                result[aid] = (iid, index)
    return ref, result


def signed_support(stock, box, rank):
    pools, _ = rank.editable_pools(stock, box)
    r2 = tuple(np.concatenate([pools[sign + str(depth)][0] for depth in (1,2)]) for sign in ('add', 'drop'))
    y, x = np.nonzero(stock)
    h, w = stock.shape
    y0, y1 = max(0, int(y.min())-4), min(h, int(y.max())+5)
    x0, x1 = max(0, int(x.min())-4), min(w, int(x.max())+5)
    fragment = stock[y0:y1, x0:x1]
    x, y = np.arange(x0, x1), np.arange(y0, y1)
    left, top, right, bottom = box
    inside = ((left <= x) & (x < right))[None, :] & ((top <= y) & (y < bottom))[:, None]
    structure = np.ones((3, 3), dtype=bool)
    outside_ring = binary_dilation(fragment, structure=structure, iterations=4) & ~fragment & inside
    inside_ring = fragment & ~binary_erosion(fragment, structure=structure, iterations=4) & inside
    r4 = []
    for ring in (outside_ring, inside_ring):
        yy, xx = np.nonzero(ring)
        r4.append((yy+y0)*w + xx+x0)
    need(all(np.isin(a, b).all() for a, b in zip(r2, r4, strict=True)), 'Depth-2 support nests in depth-4 support')
    return r2, r4


def iou(hit, area, truth_area):
    return hit / (area + truth_area - hit)


def oracle_pair(truth, hit, area, truth_area, adds, drops, k):
    need(k <= min(len(adds), len(drops)), 'Exact same signed budget is feasible')
    good_add, good_drop = int(truth.flat[adds].sum()), int((~truth.flat[drops]).sum())

    def for_split(add, drop):
        overlap = hit + min(add, good_add) - max(0, drop - good_drop)
        return iou(overlap, area + add - drop, truth_area)

    equal = for_split(k, k)
    variable = max(for_split(a, 2*k-a) for a in range(max(0, 2*k-len(drops)), min(2*k, len(adds))+1))
    return equal, variable


def geometry(annotation, iid, index, info, coco, rank):
    identity, rows, boxes, actions = info
    row = rows[index]
    stock = codec.decode(row['rle']).astype(bool)
    truth = codec.decode(coco.annToRLE(annotation)).astype(bool)
    area, gt_area, hit = int(stock.sum()), int(truth.sum()), int((stock & truth).sum())
    native = iou(hit, area, gt_area)
    entry = actions.get((row['rank'], row['query_id']))
    if entry is None or not len(entry[1]):
        return [native] * len(COLUMNS), 0
    p, selected = entry
    k = len(selected)
    adds, drops = p[:, 0], p[:, 1]
    need(not stock.flat[adds].any() and stock.flat[drops].all(), 'Verified original signed endpoints')
    subsets = signed_support(stock, boxes[row['query_id']], rank)
    original = (adds, drops)
    need(all(np.isin(a, s).all() for a, s in zip(original, subsets[0], strict=True)), 'Sampled B5 support nests exactly inside full R2')
    b5, b5_variable = oracle_pair(truth, hit, area, gt_area, *original, k)
    r2, _ = oracle_pair(truth, hit, area, gt_area, *subsets[0], k)
    r4, r4_variable = oracle_pair(truth, hit, area, gt_area, *subsets[1], k)
    realized = iou(hit + int(truth.flat[adds[selected]].sum()) - int(truth.flat[drops[selected]].sum()), area, gt_area)
    need(realized <= b5 <= r2 <= r4 <= r4_variable and b5 <= b5_variable <= r4_variable,
         'Observed value and exact nested feasible sets')
    return [native, realized, b5, r2, r4, b5_variable, r4_variable], k


def aggregate(values, groups):
    data = np.asarray(values)
    parent = np.asarray(groups)
    pool = sorted(set(groups))
    means = np.asarray([data[parent == g].mean(axis=0) for g in pool])
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(pool), size=(4096, len(pool)))
    metrics = {name: dict(parent_mean_points=100*float(means[:, i].mean()), hull_mean_points=100*float(data[:, i].mean()),
                          reach_iou={f'{cut:.2f}': int((data[:, i] >= cut).sum()) for cut in (0.50,0.75,0.90)})
               for i, name in enumerate(COLUMNS)}
    effects = {}
    for name, terms in CONTRASTS.items():
        col = lambda table: sum(table[:, COLUMNS.index(arm)] * sign for arm, sign in terms)
        d, raw = col(means), col(data)
        # A tolerance suppresses floating cancellation in the four-term interaction only.
        positive = raw > 1e-12
        effects[name] = dict(parent_delta_points=100*float(d.mean()),
                             percentile95_points=(100*np.percentile(d[draws].mean(axis=1), [2.5,97.5])).tolist(),
                             meaningful_positive_hulls=int(positive.sum()), meaningful_positive_prefixes=len(set(parent[positive])),
                             negative_hulls=int((raw < -1e-12).sum()))
    primary = effects['primary_support_r4_minus_b5']
    passes = (primary['parent_delta_points'] >= .30 and primary['percentile95_points'][0] > 0
              and primary['meaningful_positive_hulls'] >= 20 and primary['meaningful_positive_prefixes'] >= 8)
    return dict(metrics=metrics, effects=effects, passes_feasibility_gate=passes, truth_prefix_proxies=len(pool))


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for arg in ('code', 'protocol', 'capture', 'acceptance', 'source-cohort', 'basis', 'heads', 'fit-gt', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    args = p.parse_args()
    need(not args.output.exists() and args.output.parent.is_dir(), 'Fresh private output')
    t0 = perf_counter()
    pinned(args.protocol, PROTOCOL)
    originals, infos, rank, head_digests, positive_count = label_free(args)
    print(json.dumps(dict(stage='native_and_held_fold_positive_margin_verified', images=len(infos),
                          positive_budget_candidates=positive_count)), flush=True)
    helper = module(args.code / 'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                    '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b')
    gt = helper.annotations(args, originals)
    coco, native = fixed_coco(gt, originals)
    values, groups, exposure = [], [], Counter()
    for ann in gt['annotations']:
        iid = ann['image_id']
        info = infos[iid]
        groups.append(info[0]['group'])
        match = native.get(ann['id'])
        if match is None:
            values.append([0.] * len(COLUMNS))
            continue
        need(match[0] == iid, 'Native match stays on exact source chip')
        measures, k = geometry(ann, iid, match[1], info, coco, rank)
        need(measures[0] >= .50 - 1e-12, 'Native exact raster agrees with .50 matching')
        values.append(measures)
        exposure['native_matched_positive_k'] += bool(k)
        exposure['native_matched_signed_k'] += k
    need(len(values) == 496, 'All source hulls including unchanged and unmatched')
    result = aggregate(values, groups)
    result.update(status='HRSID_FIT109_EXACT_NESTED_SUPPORT_ORACLE_V1', phase='score', protocol_sha256=PROTOCOL,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), original_capture_sha256=CAPTURE,
                  original_basis_readback_sha256=BASIS, original_fit_head_sha256=HEAD_RUN, held_seed5217_heads_sha256=head_digests,
                  source_detector_fit_exposed=True, source_point_head_uses_original_held_fold=True,
                  source_images=len(infos), hulls=len(values), native_iou50_matches=len(native),
                  native_iou50_unmatched=len(values)-len(native), prefix_groups_are_acquisition_proxies=True,
                  exposure=dict(exposure), detector_forwards=0, optimizer_updates=0, computed_ap=False,
                  external_data_opened=False, seconds=perf_counter()-t0)
    payload = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    args.output.mkdir()
    with (args.output / 'RESULT_SAFE.json').open('xb') as out:
        out.write(payload)
    print(json.dumps(dict(status=result['status'], result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
