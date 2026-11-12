"""Fixed edit-location headroom under exact P250 stock binary truth and counts."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROTOCOL_SHA = "3eca9965da9dc8d8a84fdc537fa737737443f99bba1df4b0d35df8454b349098"
WORKER_SHA = "40bc85ef6fc6612e23674d2cba82cd69dce35865320c40175df5c187fcc2afcc"
RESULT_SHA = "8d8c11965433551ade41501adcd0a2af77cc7c5f7de81e731f6da1f2d787497a"
PRELABEL_SHA = "1c8f06e33d74a4c97d9861ddf21da0342f8cc510d6fe81be2b7075dcc47ef49f"
READBACK_SHA = "44a4c65137d5ff9e89a50e0f11f9158d9d7c3ef1f0e8fcf6f483b526781a0d8a"
SCORE_TERMINAL_SHA = "b8c57c145f3ce891204b46147c5d9bdb5cb115e913fcc29787f0399df356b1ba"
MODES = ("raw", "vote", "majority_oracle", "full_band_oracle")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, digest):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Canonical closed input")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Exact digest " + path.name)
    return raw


def oracle_mask(anchor, truth, domain, additions, deletions):
    """Fixed-area IoU is maximized by independently selecting helpful adds and drops."""
    def best(options, beneficial, count):
        require(0 <= count <= int(options.sum()), "Infeasible exact edit count")
        return np.concatenate((np.flatnonzero(options & beneficial),
                               np.flatnonzero(options & ~beneficial)))[:count]

    take_add = best(domain & ~anchor, truth, additions)
    take_drop = best(domain & anchor, ~truth, deletions)
    after = anchor.copy(order="C")
    after.ravel()[take_add] = True
    after.ravel()[take_drop] = False
    return after


def fixed_matches(data, rows, ids):
    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = deepcopy(data)
        gt.createIndex()
        dt = gt.loadRes(deepcopy(rows))
        evaluator = COCOeval(gt, dt, "segm")
        evaluator.params.imgIds = list(ids)
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.params.iouThrs = np.array([.50])
        evaluator.params.areaRng = [[0, 1e10]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.evaluate()
    matches = {}
    for block in evaluator.evalImgs:
        if block is None:
            continue
        for dt_id, gt_id, ignore in zip(block["dtIds"], block["dtMatches"][0], block["dtIgnore"][0], strict=True):
            if gt_id and not ignore:
                matches[int(dt_id) - 1] = int(gt_id)
    return gt, matches


def mask_iou(mask, truth):
    return float(np.count_nonzero(mask & truth) / np.count_nonzero(mask | truth))


def precision(anchor, truth, mask):
    adds, drops = mask & ~anchor, anchor & ~mask
    return dict(additions=int(adds.sum()), deletions=int(drops.sum()),
                helpful_additions=int((adds & truth).sum()), helpful_deletions=int((drops & ~truth).sum()))


def support_counts(domain, anchor, truth, adds, drops):
    a = int((domain & ~anchor & truth).sum())
    d = int((domain & anchor & ~truth).sum())
    return dict(domain_pixels=int(domain.sum()), helpful_add_positions=a, helpful_drop_positions=d,
                anchors_with_helpful_add=int(adds > 0 and a > 0), anchors_with_helpful_drop=int(drops > 0 and d > 0),
                anchors_with_useful_opportunity=int((adds > 0 and a > 0) or (drops > 0 and d > 0)),
                anchors_all_edits_can_help=int(a >= adds and d >= drops))


def measure(worker, gt, matches, rows, locations, arms, variants, items):
    records = []
    edit_counts = defaultdict(Counter)
    support = defaultdict(Counter)
    edited = 0
    for i, ann_id in matches.items():
        row = rows[i]
        area = int(codec.area(row["segmentation"]))
        if not 32 <= area < 1024:
            continue
        ordinal, rank = locations[i]
        item = items[ordinal]
        block = variants[ordinal]
        require(item["image_id"] == block["image_id"] == row["image_id"], "Exact frozen annotation image")
        anchor = codec.decode(row["segmentation"]).astype(bool)
        truth = codec.decode(gt.annToRLE(gt.anns[ann_id])).astype(bool)
        require(anchor.shape == truth.shape and truth.any(), "Exact nonempty native binary truth")
        change = block["variants"].get(str(rank))
        masks = {mode: anchor for mode in MODES}
        if change is not None:
            edited += 1
            match, = [triplet for triplet in block["triplets"] if triplet[0] == rank]
            donors = []
            for seed, donor_rank in zip((5201, 5203), match[1:], strict=True):
                donor_block = arms[seed][ordinal]
                j = donor_block["native_ranks"].index(donor_rank)
                donors.append(codec.decode(donor_block["coco"][j]["segmentation"]).astype(bool))
            a, d = change["additions"], change["deletions"]
            band, _ = worker.layers(anchor)
            majority = band & (donors[0] == donors[1]) & (donors[0] != anchor)
            voted = codec.decode(change["vote"]).astype(bool)
            exact = worker.choose_pixels(anchor, *donors)
            require(exact is not None and np.array_equal(exact[0], voted) and exact[2:] == (a, d)
                    and not np.any((anchor ^ voted) & ~majority), "Exact original vote and support witness")
            masks.update(vote=voted, majority_oracle=oracle_mask(anchor, truth, majority, a, d),
                         full_band_oracle=oracle_mask(anchor, truth, band, a, d))
            for m in MODES[1:]:
                detail = precision(anchor, truth, masks[m])
                require((detail["additions"], detail["deletions"]) == (a, d) and
                        int(masks[m].sum()) == area+a-d, "Exact same-budget edit counts")
                edit_counts[m].update(detail)
            support["majority_oracle"].update(support_counts(majority, anchor, truth, a, d))
            support["full_band_oracle"].update(support_counts(band, anchor, truth, a, d))
        scores = [mask_iou(masks[m], truth) for m in MODES]
        require(scores[3] >= scores[2] >= scores[1], "Nested oracle domains and actual vote")
        records.append((item["image_ref"]["sha256"], scores))
    require((len(records), edited) == (273, 265), "Exact original P250 matched and edited fixed scope")
    totals = {m: dict(edit_counts[m]) for m in MODES[1:]}
    for t in totals.values():
        t.update(add_precision=t["helpful_additions"]/t["additions"],
                 drop_precision=t["helpful_deletions"]/t["deletions"],
                 combined_precision=(t["helpful_additions"]+t["helpful_deletions"])/(t["additions"]+t["deletions"]))
    return records, totals, {k: dict(v) for k, v in support.items()}, edited


def summarize(records):
    by_group = defaultdict(list)
    for group, values in records:
        by_group[group].append(values)
    paired = np.asarray([v for _, v in records], dtype=np.float64)
    groups = np.asarray([np.mean(by_group[key], axis=0) for key in sorted(by_group)], dtype=np.float64)
    draws = np.random.default_rng(5202).integers(len(groups), size=(10000, len(groups)))
    effects = {}
    comparisons = ((1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2))
    for a, b in comparisons:
        points = 100*(groups[:, a]-groups[:, b])
        low, high = np.quantile(points[draws].mean(axis=1), [.025, .975])
        effects[f"{MODES[a]}_minus_{MODES[b]}"] = dict(group_mean_points=float(points.mean()),
                                                       percentile95_points=[float(low), float(high)],
                                                       improved=int((paired[:, a] > paired[:, b]).sum()),
                                                       worsened=int((paired[:, a] < paired[:, b]).sum()),
                                                       unchanged=int((paired[:, a] == paired[:, b]).sum()))
    threshold = {}
    for value in (.75, .90):
        attained = paired >= value
        threshold[str(value)] = {mode: dict(total=int(attained[:, i].sum()),
                                            gained_vs_raw=int((attained[:, i] & ~attained[:, 0]).sum()),
                                            lost_vs_raw=int((~attained[:, i] & attained[:, 0]).sum()),
                                            gained_vs_vote=int((attained[:, i] & ~attained[:, 1]).sum()))
                                  for i, mode in enumerate(MODES)}
    return dict(jpeg_groups=len(groups), group_equal_iou_points=dict(zip(MODES, (100*groups.mean(axis=0)).tolist())),
                match_equal_iou_points=dict(zip(MODES, (100*paired.mean(axis=0)).tolist())), effects=effects,
                fixed_thresholds=threshold, paired_bootstrap_draws=10000, paired_bootstrap_seed=5202)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("worker", "prelabel", "readback", "score", "score-terminal", "plan", "independent", "code",
                  "protocol", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.environ.get("OMP_NUM_THREADS") == "1",
            "CPU-only single thread scope")
    pinned(args.protocol, PROTOCOL_SHA)
    actual = json.loads(pinned(args.score, RESULT_SHA))
    terminal = json.loads(pinned(args.score_terminal, SCORE_TERMINAL_SHA))
    require(actual["advance"] is False and terminal["original_os_exit"] == 0 and terminal["child_closed"] and
            not terminal["timed_out"] and terminal["output_sha256"] == RESULT_SHA, "Closed original score witness")
    worker = ModuleType("_closed_vote")
    worker.__file__ = str(args.worker)
    exec(compile(pinned(args.worker, WORKER_SHA), str(args.worker), "exec"), worker.__dict__)
    args.prelabel_sha = PRELABEL_SHA
    receipt, variants = worker.read_variants(args)
    readback = json.loads(pinned(args.readback, READBACK_SHA))
    require(receipt["GT_opened"] is readback["GT_opened"] is False and readback["all_edit_invariants"] and
            readback["variants_sha256"] == receipt["variants_sha256"], "Independent label-free original variants")
    original = worker.load_score(args.code)
    plan, _, ids, items, arms, scorer, _ = worker.read_inputs(args, original, worker.CHECKPOINTS)
    rows, loc = [], []
    for i, (image, anchor) in enumerate(zip(ids, arms[5202], strict=True)):
        require(anchor["image_id"] == image == variants[i]["image_id"], "Exact ordered raw image identity")
        rows.extend(anchor["coco"])
        loc.extend((i, rank) for rank in anchor["native_ranks"])
    require(len(rows) == len(loc) == 18684, "Exact raw count and stable indexing")
    data = original.document(original.read_ref(plan["integer_gt"], original.MIB))
    scorer.validate_gt(data, ids, items)
    gt, matches = fixed_matches(data, rows, ids)
    records, precision, support, edited = measure(worker, gt, matches, rows, loc, arms, variants, items)
    summary = summarize(records)
    require(all(abs(summary["match_equal_iou_points"][m] -
                    actual["fixed_binary_geometry"]["iou_points"][m]) < 1e-10 for m in ("raw", "vote")),
            "Exact independent fixed raw/vote endpoint reproduction")
    result = dict(status="HRSID_TO_SSDD_DEV187_RLE_ORACLE_HEADROOM_ACTUAL", DEV=True, TEST=False, AP=False,
                  model_forwards=0, optimizer_updates=0, ground_truth_assisted=True, prior_advance=False,
                  prior_score_sha256=RESULT_SHA, protocol_sha256=PROTOCOL_SHA, original_matches=len(records),
                  edited_matches=edited, untouched_matches=len(records)-edited, precision=precision,
                  editable_support=support, **summary)
    digest = worker.write_new(args.output, worker.stable_json(result))
    print(json.dumps(dict(status=result["status"], output_sha256=digest, original_matches=len(records),
                          group_equal_iou_points=result["group_equal_iou_points"]), sort_keys=True))


if __name__ == "__main__":
    main()