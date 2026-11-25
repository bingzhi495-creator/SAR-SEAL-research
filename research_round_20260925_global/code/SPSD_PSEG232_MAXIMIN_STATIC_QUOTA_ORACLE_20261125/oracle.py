"""Label-conditioned fixed-static-budget dual-reference geometry ceiling."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

LOCK = "2427641486dffb8d468aabd80eea4d912923b029c6bf99f27d838e3f0e6242fa"
CLASSICAL = "633da1df76185bb32177422eba0c3b51f2c0b1a01931ce432b993641c10aa6c7"
ATTRACTION = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
REFERENCE = "9baf865d510b3af95510ca55017943e5147a9c9cc68384665f14cb40b07434a9"
PRIOR = "dd152bd587d64453b874a9a34e6531adacbd7bf163de4f9b22ffb25c4dea52fc"
PAIRED_MECHANISM = "2522ec539f610fb1f01a474d35a2a0c35521c7dd8e3357522f884a63fedee1b1"
PREPARE = "SPSD_MAXIMIN_STATIC_FULL_POOL_PRELABEL_COMPLETE"
SCORED = "SPSD_MAXIMIN_STATIC_DUAL_REFERENCE_ORACLE_COMPLETE"
STATIC = "attraction_static_at_dynamic_count"
DYNAMIC = "attraction_dynamic"
LEDGER = "FULL_SAMPLE_POOL_PRIVATE.jsonl"
TOL = 1e-12


def need(test, why):
    if not test:
        raise ValueError(why)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Pinned file identity: " + path.name)
    return raw


def module(path, expected):
    source = ModuleType(path.stem)
    source.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), source.__dict__)
    return source


def safely_write(path, payload):
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def worker_sha():
    return digest(Path(__file__).read_bytes())


def closed_inputs(a):
    ref = module(a.code / "SPSD_PSEG232_ANNOTATION_SENSITIVITY_20261125/score_reference.py", REFERENCE)
    scorer = ref.module(a.code / "PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py", ref.SCORER)
    roster, blocks, rgb_keys, _, identity, swaps, candidates = ref.originals(a, scorer)
    prior_done = json.loads((a.prior_run / "observed_score_corrected/TERMINAL.json").read_bytes())
    need(prior_done["phase"] == "score" and prior_done["actual_exit"] == 0
         and prior_done["output_footer_verified"] is True and prior_done["worker_sha256"] == REFERENCE
         and prior_done["result_sha256"] == PRIOR, "Observed dual-reference terminal")
    prior = json.loads(pin(a.prior_run / "RESULT_SAFE.json", PRIOR))
    need(prior["status"] == ref.STATUS and identity == prior["identity"]
         and swaps == prior["realized_swaps"] and candidates == prior["edited_candidates"],
         "Exact original roster, candidate, arm and signed-budget ledger")
    return ref, scorer, roster, blocks, rgb_keys, identity, swaps, candidates, prior


def prepare(a, closed):
    _, _, roster, blocks, _, identity, swaps, candidates, _ = closed
    compare = module(a.code / "PSEG_TEST232_CLASSICAL_ATTRACTION_20261122/compare.py", CLASSICAL)
    attraction = module(a.code / "FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py", ATTRACTION)
    a.original_readback = a.old_readback
    a.phase_root = a.classical_root
    replay, m, capture_roster, receipt, _ = compare.inputs(a)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    frozen = m.old.restore(a)
    pair, rank = frozen[:2]
    pair.raw = m.legacy.pin(a.pair_math, m.native.PAIR)
    need(capture_roster == roster, "Exact closed author roster order")
    sums = Counter()
    sha = sha256()
    with (a.output.parent / LEDGER).open("xb") as stream:
        rows = m.legacy.blocks(a, m.author, roster, receipt, True)
        for offset, (item, coco, audit, planes) in enumerate(rows, 1):
            key = item["image_ref"]["path"]
            native, edits, _ = blocks[key]
            need(len(native) == len(coco) == len(audit) == len(planes) == 100
                 and native == coco, "Exact whole original native candidate sequence")
            image = m.legacy.rgb(a, item)
            adapted, probability = m.native.make_native(pair, rank, *item["original_hw"])
            features = pair.image_features(image)
            pools = {}
            for ordinal, (base, detail, plane) in enumerate(zip(coco, audit, planes, strict=True), 1):
                key_rank = str(ordinal)
                area = detail["mask_area"]
                expected = edits.get(key_rank, {})
                expected_k = expected.get(STATIC, {}).get("swaps", 0)
                need(expected_k == expected.get(DYNAMIC, {}).get("swaps", 0), "Closed arm count parity")
                sums["native_candidates"] += 1
                coordinates = None
                if 32 <= area < 1024:
                    sums["area_eligible"] += 1
                    raw_mask = base["segmentation"]
                    stock = mask.decode(raw_mask).astype(bool, order="C")
                    value = adapted.paired_points(stock, detail["box_xyxy"], area,
                                                  base["score"], plane, features, probability)
                    if value is not None:
                        coordinates = value[0]
                        sums["supported"] += 1
                if coordinates is None:
                    need(expected_k == 0, "No original static writes outside exact support")
                    continue
                cap = min(8, area // 10, len(coordinates))
                dynamic, static = attraction.select(stock, coordinates, cap)
                need(len(static) == len(dynamic) == expected_k, "Exact per-candidate prior budget")
                if not expected_k:
                    continue
                for name, choice in ((STATIC, static), (DYNAMIC, dynamic)):
                    actual = m.old.edit(stock, coordinates, choice)
                    m.old.validate_edit(raw_mask, actual, area, expected_k)
                    need(actual == expected[name]["rle"], "Exact original " + name + " RLE")
                pools[key_rank] = [expected_k, coordinates.tolist()]
                sums["matched_rles"] += 2
                sums["prior_swaps"] += expected_k
                sums["edited_candidates"] += 1
                sums["sampled_add_sites"] += len(coordinates)
                sums["sampled_drop_sites"] += len(coordinates)
            row = dict(image_id=item["image_id"], pools=pools)
            raw = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
            stream.write(raw)
            sha.update(raw)
            if offset % 50 == 0 or offset == 232:
                print(json.dumps(dict(progress_images=offset, verified_candidates=sums["native_candidates"])), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    need(sums["native_candidates"] == 23200 and sums["prior_swaps"] == swaps[STATIC]
         and sums["edited_candidates"] == candidates[STATIC] and sums["matched_rles"] == 2 * candidates[STATIC],
         "Whole accepted static budget and exact treatments")
    return dict(status=PREPARE, identity=identity, counts=dict(sums), support_ledger_sha256=sha.hexdigest(),
                labels_opened=False, no_new_AP=True)


def load_prepared(a, identity, roster, blocks, swaps, candidates):
    result_path = a.prepare_root / "PRELABEL_SAFE.json"
    terminal = json.loads((a.prepare_root / "observed_prepare/TERMINAL.json").read_bytes())
    need(terminal["phase"] == "prepare-edits" and terminal["actual_exit"] == 0
         and terminal["output_footer_verified"] is True and terminal["worker_sha256"] == worker_sha(), "Closed prepare observer")
    prepared = json.loads(pin(result_path, terminal["result_sha256"]))
    need(prepared["status"] == PREPARE and prepared["labels_opened"] is False
         and prepared["protocol_sha256"] == LOCK and prepared["worker_sha256"] == worker_sha()
         and prepared["identity"] == identity, "Prepared label-free receipt")
    rows = [json.loads(line) for line in pin(a.prepare_root / LEDGER,
                                            prepared["support_ledger_sha256"]).splitlines()]
    need(len(rows) == len(roster) == 232, "Complete prelabel ledger")
    count = Counter()
    final = {}
    for item, row in zip(roster, rows, strict=True):
        need(item["image_id"] == row["image_id"], "Prepared image order")
        native, edits, _ = blocks[item["image_ref"]["path"]]
        expected = {rank: arms[STATIC]["swaps"] for rank, arms in edits.items() if STATIC in arms}
        need({rank: value[0] for rank, value in row["pools"].items()} == expected, "Every original positive count retained")
        for rank, (k, loc) in row["pools"].items():
            stock_rle = native[int(rank)-1]["segmentation"]
            stock = mask.decode(stock_rle).astype(bool)
            written = mask.decode(edits[rank][STATIC]["rle"]).astype(bool)
            loc = np.asarray(loc, dtype=np.int64)
            need(loc.ndim == 2 and loc.shape[1] == 2 and 1 <= k <= min(8, len(loc))
                 and np.all((loc >= 0) & (loc < stock.size)) and len(np.unique(loc)) == 2*len(loc),
                 "Whole distinct signed endpoint pool")
            need(not np.any(stock.ravel()[loc[:,0]]) and np.all(stock.ravel()[loc[:,1]])
                 and set(np.flatnonzero(written & ~stock)) <= set(loc[:,0])
                 and set(np.flatnonzero(stock & ~written)) <= set(loc[:,1])
                 and np.count_nonzero(written & ~stock) == np.count_nonzero(stock & ~written) == k,
                 "Original static edits realize exactly the original signed budget and support")
            count["static_swaps"] += k
            count["verified_static_candidates"] += 1
            row["pools"][rank] = (k, loc)
        final[row["image_id"]] = row["pools"]
    need(count["static_swaps"] == swaps[STATIC] and count["verified_static_candidates"] == candidates[STATIC],
         "Closed all-candidate static census")
    return final, terminal["result_sha256"], dict(count)


def as_rle(ann, shape):
    seg = ann["segmentation"]
    return mask.merge(mask.frPyObjects(seg, *shape)) if isinstance(seg, list) else seg


def coco_match_only(gt, blocks):
    detections, map_rank = [], {}
    for image in gt["images"]:
        for rank, row in enumerate(blocks[image["file_name"]][0], 1):
            detections.append(dict(image_id=image["id"],category_id=0,score=row["score"],
                                   segmentation=deepcopy(row["segmentation"])))
            map_rank[len(detections)] = image["id"], rank
    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(gt)
        ref.createIndex()
        det = ref.loadRes(detections)
        ev = COCOeval(ref, det, "segm")
        ev.params.catIds = [0]
        ev.params.imgIds = sorted(i["id"] for i in gt["images"])
        ev.params.iouThrs = np.array([.50])
        ev.evaluate()
    chosen = [e for e in ev.evalImgs if e and e["aRng"] == ev.params.areaRng[0] and e["maxDet"] == 100]
    need(len(chosen) == 232, "All-area original stock match rows")
    matches = {}
    for image in chosen:
        for aid, did, ignored in zip(image["gtIds"], image["gtMatches"][0], image["gtIgnore"], strict=True):
            if did and not ignored:
                iid, rank = map_rank[int(did)]
                need(iid == image["image_id"] and aid not in matches, "Frozen .50 candidate identity")
                matches[aid] = iid, rank
    return matches


def count_vectors(available, k):
    """Enumerate all feasible counts in the four discrete endpoint membership classes."""
    return np.asarray([(j, l, m, k-j-l-m)
                       for j in range(min(k, available[0])+1)
                       for l in range(min(k-j, available[1])+1)
                       for m in range(min(k-j-l, available[2])+1)
                       if k-j-l-m <= available[3]], dtype=np.int64).reshape(-1, 4)


MEMBERSHIP = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)


def iou(intersection, total):
    return 100.0 * intersection / (total - intersection)


def maximin(p_codes, d_codes, k, raw_intersection, total, static_intersection, denominators):
    adds = count_vectors(np.bincount(p_codes, minlength=4), k)
    drops = count_vectors(np.bincount(d_codes, minlength=4), k)
    stock_static = iou(static_intersection, total)
    intersections = raw_intersection + (adds @ MEMBERSHIP)[:, None, :] - (drops @ MEMBERSHIP)[None, :, :]
    need(np.all((intersections >= 0) & (total - intersections > 0))
         and np.any(np.all(intersections == static_intersection, axis=2)), "Static is feasible with complete exact-count support")
    deltas = iou(intersections, total) - stock_static
    values = np.min(deltas / denominators, axis=2)
    i, j = np.unravel_index(int(np.argmax(values)), values.shape)
    lower = float(values[i, j])
    need(lower >= -TOL, "Static-feasible maximin lower bound")
    if lower <= TOL:
        return None, None, np.zeros(2), 0.0
    return adds[i], drops[j], deltas[i,j], lower


def selected_sites(codes, counts, eligible):
    return np.asarray([cell for kind, n in enumerate(counts)
                       for cell in eligible[codes == kind][:n]], dtype=np.int64)


def describe(vector, draws):
    pooled = vector[draws].mean(axis=1)
    return dict(points=float(vector.mean()), percentile95_points=np.percentile(pooled,[2.5,97.5]).tolist())


def score(a, closed):
    ref, scorer, roster, blocks, rgb_keys, identity, swaps, candidates, prior = closed
    pools, preparation, checked = load_prepared(a, identity, roster, blocks, swaps, candidates)
    mechanism = json.loads(pin(a.mechanism_result, PAIRED_MECHANISM))
    need(mechanism["natural_common_disagreement_advances_source_design"] is True
         and mechanism["prior_result_sha256"] == PRIOR, "Verified boundary-disagreement headroom precondition")
    pseg, spsd, pan, discrepancies = ref.read_references(a, scorer, roster)
    need(pan == prior["spsd_annotation_json_sha256"] and discrepancies == prior["spsd_metadata_disagreement"],
         "Identical admitted author labels")
    authors = (pseg, spsd)
    assigned = [coco_match_only(gt, blocks) for gt in authors]
    need([len(x) for x in assigned] == [402, 403], "Original each-author native assignments")
    images = sorted(pseg["images"], key=lambda r: rgb_keys[r["file_name"]])
    ann = [defaultdict(list), defaultdict(list)]
    for grouped, gt in zip(ann, authors, strict=True):
        for row in gt["annotations"]:
            grouped[row["image_id"]].append(row)
    deltas = np.zeros((232,2), dtype=float)
    lower = np.zeros(232, dtype=float)
    events = Counter()
    changed_images = 0
    for index, image in enumerate(images):
        iid = image["id"]
        native, edits, _ = blocks[image["file_name"]]
        roster = [d[iid] for d in ann]
        denom = np.asarray([len(v) for v in roster])
        shape = image["height"], image["width"]
        rr = [[as_rle(label, shape) for label in rows] for rows in roster]
        overlaps = mask.iou(rr[0], rr[1], [0]*len(rr[1]))
        pairs = zip(*linear_sum_assignment(-overlaps), strict=True)
        changed = 0
        for i, j in pairs:
            events["label_pairs"] += 1
            if overlaps[i, j] < .5:
                continue
            events["label_pairs_ge_50"] += 1
            p_id, s_id = roster[0][i]["id"], roster[1][j]["id"]
            pr, sr = assigned[0].get(p_id), assigned[1].get(s_id)
            if pr is None or pr != sr:
                continue
            events["same_native_rank"] += 1
            key = str(pr[1])
            if key not in pools[iid]:
                continue
            events["same_native_rank_positive_static_count"] += 1
            k, loc = pools[iid][key]
            raw = mask.decode(native[pr[1]-1]["segmentation"]).astype(bool).ravel()
            static = mask.decode(edits[key][STATIC]["rle"]).astype(bool).ravel()
            truth = np.stack((mask.decode(rr[0][i]).astype(bool).ravel(),
                              mask.decode(rr[1][j]).astype(bool).ravel()))
            labels = (truth[0].astype(np.uint8) + 2 * truth[1].astype(np.uint8))
            total = truth.sum(axis=1) + raw.sum()
            raw_int = truth[:, raw].sum(axis=1)
            static_int = truth[:, static].sum(axis=1)
            add_codes, drop_codes = labels[loc].T
            counts_a, counts_d, effect, conservative = maximin(add_codes, drop_codes, k, raw_int,
                                                                total, static_int, denom)
            if counts_a is None:
                continue
            result = raw.copy()
            result[selected_sites(add_codes, counts_a, loc[:,0])] = True
            result[selected_sites(drop_codes, counts_d, loc[:,1])] = False
            need(np.count_nonzero(result & ~raw) == np.count_nonzero(raw & ~result) == k
                 and not np.array_equal(result, static)
                 and np.allclose(iou(truth[:,result].sum(axis=1),total) - iou(static_int,total),effect,atol=1e-12,rtol=0),
                 "Winning integer intervention and exact geometry")
            deltas[index] += effect / denom
            lower[index] += conservative
            changed += 1
        events["changed_common_hulls"] += changed
        changed_images += int(changed > 0)
    need(events["label_pairs"] == 544 and events["label_pairs_ge_50"] == 493
         and np.all(lower <= np.min(deltas, axis=1)+TOL), "Frozen association and conservative aggregate bound")
    draws = np.random.Generator(np.random.PCG64(5297)).integers(232,size=(4096,232))
    interval = describe(lower, draws)
    joint = deltas[draws].mean(axis=1).min(axis=1)
    gate = bool(interval["points"] >= .20 and interval["percentile95_points"][0] > 0
                and events["changed_common_hulls"] >= 20 and changed_images >= 8)
    return dict(status=SCORED, identity=identity, prelabel_sha256=preparation, checked=checked,
                input_mechanism_sha256=PAIRED_MECHANISM, references={"PSeg_SSDD": describe(deltas[:,0],draws),
                "SPSD": describe(deltas[:,1],draws)}, per_image_conservative_worst_bound=interval,
                minimum_of_two_group_means=dict(points=float(deltas.mean(axis=0).min()),
                  percentile95_points=np.percentile(joint,[2.5,97.5]).tolist()),
                bootstrap_seed=5297, bootstrap_replicates=4096, image_groups=232, new_acquisition_groups=0,
                labels_opened=True, no_new_AP=True, oracle_based_on_target_references=True,
                events=dict(events), changed_image_groups=changed_images, advances_source_training=gate)


def main():
    p = ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=("prepare","score"))
    for option in ("code","owner","capture","classical-root","old-readback","original-replay",
                   "prior-run","protocol","output"):
        p.add_argument("--"+option, type=Path, required=True)
    for option in ("refit","official","slgsea","prepare-root","mechanism-result"):
        p.add_argument("--"+option, type=Path)
    a = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ["CUDA_VISIBLE_DEVICES"] == "-1",
         "Pinned isolated CPU-only execution")
    pin(a.protocol, LOCK)
    if a.phase == "prepare":
        need(a.refit and not any((a.official,a.slgsea,a.prepare_root,a.mechanism_result)), "Prepare has no target label paths")
    else:
        need(all((a.official,a.slgsea,a.prepare_root,a.mechanism_result)) and a.refit is None, "Score reads only closed inputs")
    start = perf_counter()
    inputs = closed_inputs(a)
    result = prepare(a, inputs) if a.phase == "prepare" else score(a, inputs)
    result.update(protocol_sha256=LOCK, worker_sha256=worker_sha(), detector_forwards=0,
                  optimizer_updates=0, elapsed_seconds=perf_counter()-start)
    payload = (json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    safely_write(a.output, payload)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(payload))),flush=True)


if __name__ == "__main__":
    main()
