"""Screen aligned sibling-query boundary votes at exact frozen B5 signed counts."""

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
from scipy.optimize import brentq
from scipy.special import expit, logit


PROTOCOL = 'e5d54223e2f143d401de5260d042fe860b2a74ffc86f3325a42b7d4a3bb8bb93'
SOURCE_WORKER = '10b1b0fde04d207bcbd220526b627fa7924158675707fdbfb7423e94b53ea8b2'
SOURCE_TERMINAL = 'ee66b96176ae2d0dfadfbeca46621d8b194c80da9a5e91536422cc1bb41c00cf'
SOURCE_RESULT = 'f105b46804c981913294caeb5c9bfe465513dc9aa8e8a379364968c63cbdcf70'
ARMS = ('native', 'distance', 'independent', 'duplicate_vote', 'matched_sham', 'native_posterior',
        'static_attraction', 'aligned')
CONTROLS = ('independent', 'duplicate_vote', 'matched_sham', 'native_posterior', 'static_attraction')
REGULARIZATION = .01
METRICS = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, digest):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == digest, 'Exact file pin: ' + path.name)
    return data


def module(path, digest, name):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(pin(path, digest), str(path), 'exec'), obj.__dict__)
    return obj


def fit_coefficient(offset, cue, truth, weight):
    if not len(cue) or not np.any(cue):
        return 0.

    def derivative(beta):
        error = expit(offset + beta * cue) - truth
        return np.dot(weight, cue * error) + REGULARIZATION * beta

    return float(brentq(derivative, -200., 200., xtol=1e-11))


def ranked_writes(scores, coordinates, k):
    n = np.arange(len(coordinates))
    add = np.lexsort((n, -scores[:, 0]))[:k]
    drop = np.lexsort((n, scores[:, 1]))[:k]
    return np.r_[coordinates[add, 0], coordinates[drop, 1]]


def scramble(cue, depth, q, id_key):
    result = cue.copy()
    for side in (0, 1):
        for level in (1, 2):
            ix = np.flatnonzero(depth[:, side] == level)
            ix = ix[np.lexsort((ix, q[ix, side]))]
            for start in range(0, len(ix), 4):
                block = ix[start:start+4]
                seed_text = '|'.join(map(str, ('HRSID-SIBLING-SHAM-V1', *id_key, side, level, start//4)))
                seed = int.from_bytes(sha256(seed_text.encode()).digest()[:16], 'little')
                permutation = np.random.Generator(np.random.PCG64(seed)).permutation(block)
                result[block, side] = cue[permutation, side]
    return result


def sample_rle(rle, cells, sparse):
    height, width = rle['size']
    y, x = np.divmod(cells, width)
    runs = np.cumsum(sparse.decode_runs(rle['counts']), dtype=np.int64)
    need(runs[-1] == height*width, 'RLE runs cover original canvas')
    return (np.searchsorted(runs, x * height + y, side='right') % 2).astype(bool)


def reference_sources(args):
    terminal = json.loads(pin(args.source_run / 'observed_diagnosis/TERMINAL.json', SOURCE_TERMINAL))
    source_result = json.loads(pin(args.source_run / 'diagnosis/RESULT_SAFE.json', SOURCE_RESULT))
    need(terminal['actual_exit'] == 0 and terminal['output_footer_verified'] is True
         and terminal['worker_sha256'] == SOURCE_WORKER and terminal['result_sha256'] == SOURCE_RESULT
         and source_result['native_iou50_matches'] == 453 and source_result['hulls'] == 496, 'Prior source-only run closed')
    source = module(args.source_run / 'oracle.py', SOURCE_WORKER, '_sibling_source_reuse')
    sparse = module(args.code / 'RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py',
                    'cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5', '_sibling_sparse')
    static = module(args.code / 'FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py',
                    '8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6', '_sibling_static')
    originals, infos, _, head_digests, positive_count = source.label_free(args)
    step = sys.modules['_nested_fit_step']
    plan = sys.modules['_nested_fit_plan']
    pair = sys.modules['_nested_pair_math']
    trainer = step.load_trainer(args.code / source.PAIR / 'fit109_paired_point/train_point.py')
    folds = plan.source_folds([info[0]['group'] for info in infos.values()])
    fit = json.loads(pin(args.heads / 'RESULT_SAFE.json', source.HEAD_RUN))
    entries = [v for v in fit['all_initializations'] if v['seed'] == 5217 and v['fold'] is not None]
    need([v['weights_sha256'] for v in entries] == head_digests, 'Same five verified held-fold checkpoint digests')
    models = {}
    for entry in entries:
        heads, center, scale = plan.restore_bundle(args.heads / 'POINT_HEADS_PRIVATE' / entry['relative_weights'],
                                                  entry['weights_sha256'], trainer, step)
        models[entry['fold']] = heads['S'], center, scale
    return source, sparse, static, originals, infos, models, folds, pair, step, positive_count


def prepare(args, source, sparse, originals, infos, models, folds, pair, step):
    readback = json.loads(pin(args.basis / 'INDEPENDENT_READBACK_SAFE.json', source.BASIS))
    files, packets, exposure = [], {}, Counter()
    for ordinal, (iid, (identity, rows, _, old_actions)) in enumerate(infos.items(), 1):
        raw = (args.basis / f'NATIVE_ENDPOINT_BASIS_{ordinal:02d}_PRIVATE.npz').read_bytes()
        files.append(dict(ordinal=ordinal, sha256=sha256(raw).hexdigest(), bytes=len(raw)))
        with np.load(BytesIO(raw), allow_pickle=False) as z:
            image_id, ranks, ids, offsets, coords, depths, old14 = (
                z[key] for key in ('image_id', 'ranks', 'query_ids', 'offsets', 'coords', 'layers', 'old14'))
        need(image_id == iid and len(ranks) == len(ids) and len(offsets) == len(ranks)+1
             and depths.shape == coords.shape and np.isin(depths, [1, 2]).all(), 'Exact original depth and image keys')
        by_row = {(int(r), int(q)): slice(a, b)
                  for r, q, a, b in zip(ranks, ids, offsets[:-1], offsets[1:], strict=True)}
        need(by_row.keys() == old_actions.keys(), 'Original candidate support retained')
        rles = [native['segmentation'] for native in originals[iid]]
        overlaps = codec.iou(rles, rles, [0]*len(rles))
        np.fill_diagonal(overlaps, 0.)
        siblings = np.argmax(overlaps, axis=1)
        image_packets = {}
        g = identity['group']
        outer = folds[g]
        for pos, row in enumerate(rows):
            key = (row['rank'], row['query_id'])
            if key not in by_row:
                continue
            s = by_row[key]
            p, layers, feats = coords[s], depths[s], old14[s]
            original_points, selected = old_actions[key]
            need(np.array_equal(p, original_points) and len(p) > 0, 'Original endpoint row order and exact coordinates')
            probs = {fold: step.predict(m[0], feats, *m[1:]) for fold, m in models.items()}
            held = probs[outer]
            area = int(codec.area(rles[pos]))
            require_selected = pair.choose_pairs(held[:,0] - held[:,1], min(8, area//10, len(p)))
            need(require_selected == selected, 'Identical held-fold 5217 positive-margin budget')
            closest = int(siblings[pos])
            has_sibling = overlaps[pos, closest] >= .50
            occupied = sample_rle(rles[closest], p, sparse).astype(np.float64) if has_sibling else None
            score = rows[closest]['score'] if has_sibling else 0.
            centered = score * (occupied - occupied.mean(axis=0)) if has_sibling else np.zeros(p.shape)
            k = len(selected)
            exposure['supported'] += 1
            exposure['positive_k'] += bool(k)
            exposure['sibling'] += int(has_sibling)
            exposure['sibling_rank_signal'] += bool(np.any(centered))
            exposure['sibling_rank_signal_positive_k'] += bool(np.any(centered) and k)
            image_packets[pos] = dict(group=g, outer=outer, coords=p, depth=layers, q=probs, k=k, pair=selected,
                                      is_sibling=bool(has_sibling), cue=centered, occupied=occupied, sibling_score=score,
                                      stock_score=row['score'], raw_p=feats[:, :, 0], identity=(iid, *key), area=area)
        packets[iid] = image_packets
    file_hash = sha256((json.dumps(files, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()).hexdigest()
    need(file_hash == readback['private_output_sha256'] and exposure['supported'] == readback['candidate_sets']
         and exposure['sibling'] == 5928, 'Independent full original basis and eligibility agreement')
    return packets, exposure


def fixed_truth_records(gt, matches, coco, packets, sparse):
    rows = []
    for annotation in gt['annotations']:
        if annotation['id'] not in matches:
            continue
        iid, index = matches[annotation['id']]
        candidate = packets[iid].get(index)
        if candidate is None or not np.any(candidate['cue']):
            continue
        rle = coco.annToRLE(annotation)
        exact = sample_rle(rle, candidate['coords'], sparse).astype(np.float64)
        rows.append((candidate, exact))
    need(bool(rows), 'Original nonconstant sibling cues on fixed source matches')
    return rows


def logits(q):
    return logit(np.clip(q.astype(np.float64), 1e-6, 1-1e-6))


def fit_folds(rows):
    out, info = {}, {}
    for fold in range(5):
        train = [(row, truth) for row, truth in rows if row['outer'] != fold]
        sizes = Counter()
        for row, truth in train:
            sizes[row['group']] += len(truth)
        offsets, true, decoy, labels, weights = [], [], [], [], []
        for row, truth in train:
            q = row['q'][fold]
            offsets.append(logits(q))
            true.append(row['cue'])
            decoy.append(scramble(row['cue'], row['depth'], q, row['identity']))
            labels.append(truth)
            weights.extend([1/(len(sizes)*sizes[row['group']])] * len(truth))
        offset, y, w = np.concatenate(offsets), np.concatenate(labels), np.asarray(weights)
        need(abs(w.sum()-1) < 1e-12 and sizes, 'Source prefix-balanced calibration rows')
        cue, sham = np.concatenate(true), np.concatenate(decoy)
        betas = {key: [fit_coefficient(offset[:, s], value[:, s], y[:, s], w) for s in range(2)]
                 for key, value in [('aligned', cue), ('matched_sham', sham)]}
        out[fold] = betas
        info[fold] = dict(prefix_groups=len(sizes), fixed_matched_training_candidates=len(train), endpoints_each_side=len(w),
                          changed_sham_endpoint_values=int(np.count_nonzero(cue != sham)), coefficients=betas)
    return out, info


def arms_write(packet, native, betas, static, sparse):
    p, k, original_pair = packet['coords'], packet['k'], packet['pair']
    q = packet['q'][packet['outer']]
    need(0 <= k <= min(8, packet['area']//10, len(p)), 'Original exact signed budget')
    base = ranked_writes(q, p, k)
    if not k:
        return {}
    held = betas[packet['outer']]
    sham = scramble(packet['cue'], packet['depth'], q, packet['identity'])
    distance = np.r_[p[original_pair, 0], p[original_pair, 1]]
    if packet['is_sibling']:
        s, r = packet['stock_score'], packet['sibling_score']
        vote = (s*q + r*packet['occupied'])/(s+r) if s+r else q
        aligned_rank = logits(q) + packet['cue']*held['aligned']
        sham_rank = logits(q) + sham*held['matched_sham']
        # A constant cue leaves every original rank intact even if logits clip.
        aligned_rank = np.where(np.any(packet['cue'], axis=0)[None, :], aligned_rank, q)
        sham_rank = np.where(np.any(sham, axis=0)[None, :], sham_rank, q)
        aligned_pick = ranked_writes(aligned_rank, p, k)
        sham_pick = ranked_writes(sham_rank, p, k)
        duplicate_vote = ranked_writes(vote, p, k)
    else:
        aligned_pick = sham_pick = duplicate_vote = base
    stock = codec.decode(native).astype(bool)
    y, x = np.divmod(p, stock.shape[1])
    field = static._initial(stock, y, x)
    writes = dict(distance=distance, independent=base, duplicate_vote=duplicate_vote, matched_sham=sham_pick,
                  native_posterior=ranked_writes(packet['raw_p'], p, k), static_attraction=ranked_writes(field, p, k),
                  aligned=aligned_pick)
    result = {}
    for arm, cells in writes.items():
        need(len(cells) == 2*k == len(np.unique(cells)) and np.isin(cells[:k], p[:, 0]).all()
             and np.isin(cells[k:], p[:, 1]).all() and not stock.flat[cells[:k]].any() and stock.flat[cells[k:]].all(),
             'Exact original sided unique write support')
        mask = sparse.xor_rle(native, cells)
        pixel_states = sample_rle(mask, cells, sparse)
        area = int(codec.area(mask))
        retained = int(codec.area(codec.merge([native, mask], intersect=True)))
        need(mask['size'] == native['size'] and np.all(pixel_states[:k]) and not np.any(pixel_states[k:])
             and packet['area'] == area and area-retained == k, 'Actual exact signed XOR counts and area retained')
        result[arm] = mask
    return result


def build_worlds(originals, packets, betas, static, sparse):
    worlds = {name: {iid: [dict(row) for row in rows] for iid, rows in originals.items()} for name in ARMS}
    counts = {arm: Counter() for arm in ARMS if arm != 'native'}
    for iid, candidates in packets.items():
        for pos, packet in candidates.items():
            origin = originals[iid][pos]['segmentation']
            replacements = arms_write(packet, origin, betas, static, sparse)
            for arm, mask in replacements.items():
                worlds[arm][iid][pos]['segmentation'] = mask
                counts[arm]['edited_candidates'] += 1
                counts[arm]['added_pixels'] += packet['k']
                counts[arm]['deleted_pixels'] += packet['k']
    reference = counts['independent']
    need(all(row == reference for row in counts.values()), 'Whole-pool exact signed budget parity')
    return worlds, counts


def full_metrics(gt, predictions, originals):
    rows = []
    for iid in sorted(originals):
        for row, source in zip(predictions[iid], originals[iid], strict=True):
            need(all(row[k] == source[k] for k in ('image_id', 'score', 'category_id')), 'Unchanged candidate confidence and identity')
            rows.append(row)
    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(gt)
        ref.createIndex()
        det = ref.loadRes(deepcopy(rows))
        ev = COCOeval(ref, det, 'segm')
        ev.params.imgIds = sorted(originals)
        ev.params.catIds = [1]
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    metrics = {name: (100*float(val) if val >= 0 else None) for name, val in zip(METRICS, ev.stats, strict=True)}
    t, = np.flatnonzero(np.isclose(ev.params.iouThrs, .90))
    p = ev.eval['precision'][t, :, :, 0, -1]
    metrics['AP90'] = 100*float(p[p >= 0].mean())
    return metrics


def geometry_and_gate(gt, matches, coco, infos, worlds, packets, metrics):
    groups = defaultdict(list)
    contrasts = (*CONTROLS, 'native', 'distance')
    changed = {arm: set() for arm in contrasts}
    changed_prefix = {arm: set() for arm in contrasts}
    witness = Counter()
    for ann in gt['annotations']:
        aid, iid = ann['id'], ann['image_id']
        group = infos[iid][0]['group']
        if aid not in matches:
            groups[group].append([0.] * len(ARMS))
            continue
        found_iid, pos = matches[aid]
        need(iid == found_iid, 'Frozen assigned position remains in its own image')
        rles = [worlds[arm][iid][pos]['segmentation'] for arm in ARMS]
        values = codec.iou(rles, [coco.annToRLE(ann)], [0]).reshape(-1)
        need(values.size == len(ARMS) and values[0] >= .50 - 1e-12, 'Original fixed native raster anchor')
        groups[group].append(values.tolist())
        packet = packets[iid].get(pos)
        if packet is not None:
            witness['native_matched_supported'] += 1
            witness['native_matched_positive_k'] += bool(packet['k'])
            witness['native_matched_sibling'] += packet['is_sibling']
            witness['native_matched_sibling_signal_positive_k'] += bool(np.any(packet['cue']) and packet['k'])
            q = packet['q'][packet['outer']]
            witness['heldout_matched_sham_changed_endpoint_values'] += int(np.count_nonzero(
                packet['cue'] != scramble(packet['cue'], packet['depth'], q, packet['identity'])))
        for arm in contrasts:
            other = rles[ARMS.index(arm)]
            if rles[-1] != other:
                changed[arm].add(aid)
                changed_prefix[arm].add(group)
    parent_means = np.asarray([np.asarray(groups[g]).mean(axis=0) for g in sorted(groups)])
    hull_array = np.concatenate(list(groups.values()))
    require_counts = len(matches) == 453 and len(groups) == 109 and len(hull_array) == 496
    need(require_counts, 'Complete unchanged source scope and native anchor census')
    resamples = np.random.Generator(np.random.PCG64(5217)).integers(len(groups), size=(4096, len(groups)))
    readout = {arm: dict(parent_discrete_iou_points=100*float(parent_means[:, i].mean()),
                         raw_hull_mean_discrete_iou_points=100*float(hull_array[:, i].mean()),
                         reach={f'{t:.2f}': int((hull_array[:, i] >= t).sum()) for t in (.50,.75,.90)})
               for i, arm in enumerate(ARMS)}
    differences = {}
    for arm in contrasts:
        d = 100*(parent_means[:, -1] - parent_means[:, ARMS.index(arm)])
        interval = np.percentile(d[resamples].mean(axis=1), [2.5, 97.5]).tolist()
        differences[arm] = dict(ap_delta_points=metrics['aligned']['AP']-metrics[arm]['AP'],
                                geometry_delta_points=float(d.mean()), geometry_percentile95_points=interval,
                                changed_fixed_hulls=len(changed[arm]), changed_prefixes=len(changed_prefix[arm]))
    witness.update(changed_hulls_vs_independent=len(changed['independent']),
                   changed_prefixes_vs_independent=len(changed_prefix['independent']))
    active = witness['changed_hulls_vs_independent'] >= 20 and witness['changed_prefixes_vs_independent'] >= 8
    meets = {arm: differences[arm]['ap_delta_points'] >= .30 and differences[arm]['geometry_delta_points'] >= .20
                  and differences[arm]['geometry_percentile95_points'][0] > 0 for arm in CONTROLS}
    advance = active and all(meets.values())
    return readout, differences, dict(witness), dict(mechanism_coverage_pass=active, controls_pass=meets, advance=advance)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for arg in ('code', 'protocol', 'capture', 'acceptance', 'source-cohort', 'basis', 'heads', 'fit-gt', 'source-run', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    args = p.parse_args()
    need(not args.output.exists() and args.output.parent.is_dir(), 'Fresh private output')
    t0 = perf_counter()
    pin(args.protocol, PROTOCOL)
    source, sparse, static, originals, infos, models, folds, pair, step, positive_count = reference_sources(args)
    after_reference = perf_counter()
    packets, candidate_exposure = prepare(args, source, sparse, originals, infos, models, folds, pair, step)
    need(candidate_exposure['positive_k'] == positive_count, 'Preserved original held-fold positive edit count')
    t_feature = perf_counter()
    print(json.dumps(dict(stage='label_free_candidate_identity_and_cue_verified', **candidate_exposure)), flush=True)
    helper = module(args.code / 'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                    '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b', '_sibling_annotation_join')
    gt = helper.annotations(args, originals)
    coco, matches = source.fixed_coco(gt, originals)
    records = fixed_truth_records(gt, matches, coco, packets, sparse)
    before_fit = perf_counter()
    betas, fit_summary = fit_folds(records)
    t_fit = perf_counter()
    worlds, counts = build_worlds(originals, packets, betas, static, sparse)
    t_write = perf_counter()
    metrics = {arm: full_metrics(gt, worlds[arm], originals) for arm in ARMS}
    t_metric = perf_counter()
    geometry, comparisons, match_exposure, decision = geometry_and_gate(gt, matches, coco, infos, worlds, packets, metrics)
    need(not step.torch.cuda.is_initialized(), 'Source CPU diagnostic had no GPU initialization')
    timings = dict(reference_verification=after_reference-t0, q_replay_and_cue_preparation=t_feature-after_reference,
                   labels_and_fixed_matches=before_fit-t_feature, deterministic_coefficients=t_fit-before_fit,
                   sparse_edits_and_checks=t_write-t_fit, full_coco=t_metric-t_write, seconds=perf_counter()-t0)
    result = dict(status='HRSID_FIT109_SIBLING_ENDPOINT_RANK_COMPLETE_V1', phase='score', protocol_sha256=PROTOCOL,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), source_reference_run_sha256=SOURCE_RESULT,
                  source_capture_sha256=source.CAPTURE, source_basis_readback_sha256=source.BASIS,
                  source_head_fit_sha256=source.HEAD_RUN, source_gt_sha256=helper.ORIGINAL_GT, small_head_seed=5217,
                  detector_previously_exposed_to_source_images=True, outer_fold_holds_out_point_stage_only=True,
                  detector_forwards=0, new_point_head_optimizer_updates=0, supervised_coefficients_per_arm_per_fold=2,
                  total_new_scalar_coefficients=20, all_ranking_arms_exact_signed_counts=True,
                  external_data_opened=False, source_images=len(infos), candidates=sum(map(len, originals.values())),
                  hulls=len(gt['annotations']), native_matches=len(matches), native_unmatched=len(gt['annotations'])-len(matches),
                  source_prefix_proxies=len(folds), full_coco_points=metrics, fixed_all_hull_geometry=geometry,
                  paired_comparisons_aligned_minus_control=comparisons, decision=decision,
                  candidate_exposure=dict(candidate_exposure), fixed_match_exposure=match_exposure,
                  signed_endpoint_count={arm: dict(value) for arm, value in counts.items()},
                  fitted_fold_safe=fit_summary, timing_seconds=timings)
    encoded = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    args.output.mkdir()
    with (args.output/'RESULT_SAFE.json').open('xb') as out:
        out.write(encoded)
    print(json.dumps(dict(status=result['status'], result_sha256=sha256(encoded).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
