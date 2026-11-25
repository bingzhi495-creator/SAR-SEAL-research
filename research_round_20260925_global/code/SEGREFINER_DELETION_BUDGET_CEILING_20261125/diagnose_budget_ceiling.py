"""Compute an exact, label-dependent deletion ceiling from closed stock masks."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import shapely


RSDD_SIGN = "cdd7b3f7e1bbbeb528fb5100da76daee54b2a4f61f88e2381ef2b709e48a080e"
RSDD_RESULT = "02f2a09fd50d4cc3112ba43cc9b3ed76badff6720f386fb45143665d5dfb7fc5"
PSEG_SIGN = "a658416573b591d657e49e7deb5db1aaeb754aeb471d567c4077fcad2188d16c"
PSEG_RESULT = "a2570a140ace0104a81c91c148a1ce6f2f88d912a62b5906cbacbf162de6689f"
STATUS = "SEGREFINER_DELETION_ORACLE_COMPLETE"
FIELDS = ("stock_minus_add_iou", "oracle_minus_stock_iou", "oracle_minus_add_iou",
          "stock_true_removed", "oracle_true_removed", "excess_true_removed", "stock_true_added", "quota_minus_fixed_iou", "quota_minus_stock_iou",
          "fraction_of_deletions_withheld")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Unexpected signed scorer")
    module = ModuleType("original_signed_worker")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def cell_coverage(mask, polygon):
    y, x = np.nonzero(mask)
    return shapely.area(shapely.intersection(shapely.box(x, y, x + 1, y + 1), polygon))


def comparison(native, stock, polygon):
    added = stock & ~native
    removed = native & ~stock
    total = int(native.sum())
    area_add, area_rem = int(added.sum()), int(removed.sum())
    native_cover = cell_coverage(native, polygon)
    removed_cover = cell_coverage(removed, polygon)
    added_cover = cell_coverage(added, polygon)
    begin = float(native_cover.sum() + added_cover.sum())
    drop = float(removed_cover.sum())
    lowest = np.sort(native_cover)[:area_rem]
    minimum = float(lowest.sum())
    truth = polygon.area
    require(-1e-8 <= minimum <= drop + 1e-8, "Original deletion must be feasible")
    size_add = total + area_add
    size_stock = size_add - area_rem
    iou_add = 100 * begin / (size_add + truth - begin)
    iou_stock = 100 * (begin - drop) / (size_stock + truth - begin + drop)
    iou_best = 100 * (begin - minimum) / (size_stock + truth - begin + minimum)
    q_curve = np.concatenate(([0.0], np.cumsum(lowest)))
    k_curve = np.arange(area_rem + 1)
    iou_curve = 100 * (begin - q_curve) / (size_add - k_curve + truth - begin + q_curve)
    chosen = int(np.argmax(iou_curve))
    best_count_iou = float(iou_curve[chosen])
    require(best_count_iou >= max(iou_best, iou_add) - 1e-8, "Bounded-count ceiling must include both endpoints")
    return np.array((iou_stock - iou_add, iou_best - iou_stock, iou_best - iou_add,
                     100 * drop / truth, 100 * minimum / truth, 100 * (drop - minimum) / truth,
                     100 * added_cover.sum() / truth, best_count_iou - iou_best,
                     best_count_iou - iou_stock, (area_rem - chosen) / area_rem if area_rem else 0.0))


def selfcheck():
    native = np.ones((2, 3), dtype=bool)
    polygon = shapely.box(0.25, 0, 2.25, 2)
    cover = cell_coverage(native, polygon)
    for quota in range(1, native.size + 1):
        gold = min(sum(cover[list(indices)]) for indices in combinations(range(native.size), quota))
        stock = native.copy()
        stock.flat[:quota] = False
        data = comparison(native, stock, polygon)
        require(abs(data[4] - 100 * gold / polygon.area) < 1e-10 and data[1] >= -1e-10,
                "Oracle must agree with exhaustive fractional-cell selection")


def summarize(groups, bounds, expected, prior):
    means = np.stack([np.mean(groups[parent], axis=0) for parent in sorted(groups)])
    matrix = np.stack([v for rows in groups.values() for v in rows])
    center = means.mean(axis=0)
    require(len(matrix) == expected and abs(center[0] - prior) < 1e-8,
            "Original deletion contrast or eligibility has changed")
    require(np.allclose(matrix[:, 1] + matrix[:, 7], matrix[:, 8], rtol=0, atol=1e-8),
            "The fixed-location and bounded-count gains must sum exactly")
    ci = bounds(means)
    return dict(hulls=len(matrix), groups=len(groups),
                stock_harmed_hulls=int((matrix[:, 0] < -1e-8).sum()),
                ideal_still_harmed_hulls=int((matrix[:, 2] < -1e-8).sum()),
                hulls_with_mandatory_truth_removal=int((matrix[:, 4] > 1e-8).sum()),
                hulls_with_recoverable_iou=int((matrix[:, 1] > 1e-8).sum()),
                hulls_with_additional_quota_headroom=int((matrix[:, 7] > 1e-8).sum()),
                fraction_of_stock_truth_loss_recoverable=float(center[5] / center[3]),
                measures={name: dict(value=float(center[j]), interval_95=list(ci[j]))
                          for j, name in enumerate(FIELDS)})


def rsdd(args):
    args.cohort, args.support_root = 74, None
    signed = load(args.code / "SEGREFINER_SIGN_20261123/score_rsdd.py", RSDD_SIGN)
    stock = signed.pinned_module(args.code / "SEGREFINER_STOCK_20261123/score_rsdd.py", signed.STOCK_SOURCE)
    exposure = stock.exposure_source()
    _, deps = exposure.sources(args)
    _, api, cfa, history, reserve, old, evaluator, adapter, core = deps
    reference = json.loads(api.pin(args.code.parent / "results/RSDD74_SEGREFINER_SIGN_SCORE_SAFE_20261123.json", RSDD_RESULT))
    report, score_hash, score_terminal = api.observed(
        args.run / "score", args.run / "observed_score/TERMINAL.json", "score", stock.STATUS,
        signed.STOCK_SCORE, worker=signed.STOCK_SOURCE)
    infer, infer_hash, infer_terminal = api.observed(
        args.run / "inference", args.run / "observed_inference/TERMINAL.json", "inference", stock.INFER_STATUS,
        report["inference_result_sha256"], report["inference_terminal_sha256"], report["inference_worker_sha256"])
    require(reference["stock_score_sha256"] == score_hash and reference["inference_result_sha256"] == infer_hash
            and reference["stock_score_terminal_sha256"] == score_terminal
            and reference["inference_terminal_sha256"] == infer_terminal, "Signed lineage differs")
    loader = cfa.source(args.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = loader.load_population(args)
    require(lineage == reference["input_lineage"] == report["input_lineage"] == infer["input_lineage"], "Native inputs differ")
    blocks, areas, positions = exposure.native_blocks(rows, adapter, core)
    stream = api.pin(args.run / "inference" / stock.LEDGER, report["refined_masks_sha256"])
    counts = stock.attach_verified_masks(rows, blocks, areas, positions, stream, infer["counts"])
    require(counts == report["counts"], "Stock valid writes differ")
    print(json.dumps(dict(phase="qualified_original_writes", native_candidates=counts["native_candidates"])), flush=True)
    gt, _, polygons, sets, dates, label_sha = history.views_and_labels(args, old, reserve, blocks)
    require(label_sha == reference["author_labels_sha256"], "Original author polygons required")

    results = {}
    for view, expected in (("full", 39), ("date_separated", 26)):
        ids = set(sets[view])
        subset = {i: blocks[i] for i in ids}
        part = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                    annotations=[a for a in gt["annotations"] if a["image_id"] in ids])
        _, anchors = evaluator.score_one(part, subset, "raw", anchors=True)
        groups = defaultdict(list)
        for aid, (image, slot, det) in anchors.items():
            polygon = polygons[aid]
            if not stock.GATE <= areas[image][slot] < 1024 or not polygon.is_valid or polygon.area <= 0:
                continue
            native = codec.decode(det["segmentation"]).astype(bool)
            output = codec.decode(blocks[image]["changes"][slot]["stock_lr"]["rle"]).astype(bool)
            groups[blocks[image]["group"]].append(comparison(native, output, polygon))
        paired = reference["views"][view]["gate_exposed_geometry"]["paired"]["stock_lr_minus_add_only"]
        prior = paired["continuous_iou_points"]["delta"]
        keys = [dates[p] for p in sorted(groups)]
        results[view] = summarize(groups, lambda a: reserve.bootstrap(a, keys), expected, prior)
        results[view]["resampling_groups"] = len(set(keys))
    return results, RSDD_RESULT


def pseg(args):
    signed = load(args.code / "SEGREFINER_SIGN_20261123/score_pseg.py", PSEG_SIGN)
    stock = signed.load_stock(args.stock_source)
    exposure = stock.exposure_module()
    source = exposure.load_source(args.factorial_source)
    reference, _, _ = source.observed(args.signed_score, args.signed_score_terminal, "score",
                                       signed.STATUS, PSEG_SIGN)
    require(source.digest((args.signed_score / "RESULT_SAFE.json").read_bytes()) == PSEG_RESULT, "Signed output differs")
    prior, score_hash, score_terminal = source.observed(
        args.stock_score, args.stock_score_terminal, "score", stock.STATUS, signed.STOCK_WORKER)
    _, scorer, geom, _, capture, roster, audits, original, blocks = source.historical(args)
    records_by_name = exposure.native_records(source, blocks, roster, audits, original)
    infer, result, terminal = stock.verify_receipt(source, args, capture, exposure.PROTOCOL_SHA256)
    require(reference["stock_score_sha256"] == score_hash and reference["stock_score_terminal_sha256"] == score_terminal
            and reference["inference_result_sha256"] == result and reference["inference_terminal_sha256"] == terminal
            and reference["edits_sha256"] == infer["edits_sha256"], "Original signed lineage differs")
    stream = source.pin(args.inference / "EDITS_PRIVATE.jsonl", infer["edits_sha256"])
    counts, _ = stock.qualify_edits(source, records_by_name, roster, blocks, map(json.loads, stream.splitlines()))
    require(counts == prior["counts"] == infer["counts"] == reference["counts"], "Closed stock census differs")
    print(json.dumps(dict(phase="qualified_original_writes", native_candidates=counts["native_candidates"])), flush=True)
    gt = json.loads(source.pin(args.author_labels, scorer.GT_SHA))
    require(reference["author_labels_sha256"] == scorer.GT_SHA, "Same author polygons required")
    records = {im["id"]: records_by_name[im["file_name"]] for im in gt["images"]}
    continuous = geom.load(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    modules = continuous.geometry_modules(args.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    require(len(truth) == 546 and not neutral, "Original floating references required")

    def bootstrap(means):
        rng = np.random.Generator(np.random.PCG64(5217))
        indices = rng.integers(len(means), size=(4096, len(means)))
        return np.percentile(means[indices].mean(axis=1), [2.5, 97.5], axis=0).T.tolist()

    results = {}
    for view, expected in (("official", 67), ("exclude_exact_fit_overlap", 63)):
        ids = sorted(i for i, row in records.items() if view == "official" or row["name"] not in scorer.EXCLUDED)
        prior_score = prior["views"][view]["metrics_fraction"][stock.RAW]
        anchors, _ = geom.evaluate_raw(gt, records, ids, prior_score)
        groups = defaultdict(list)
        for aid, (iid, rank) in anchors.items():
            rec = records[iid]
            if not stock.GATE <= rec["areas"][rank - 1] < 1024:
                continue
            native = source.decoded(stock.mask_at(rec, rank, stock.RAW), rec["size"])
            output = source.decoded(stock.mask_at(rec, rank, stock.STOCK), rec["size"])
            groups[rec["rgb"]].append(comparison(native, output, truth[aid][0]))
        paired = reference["views"][view]["primary_gate_exposed_deletion"]["contexts"]
        value = paired["segrefiner_stock_minus_segrefiner_add_only"]["continuous_iou_points"]["delta"]
        results[view] = summarize(groups, bootstrap, expected, value)
        results[view]["resampling_groups"] = len(groups)
    return results, PSEG_RESULT


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="dataset", required=True)
    common = ("code", "author-labels", "protocol", "output")
    for name, options in (("rsdd", ("run", "stock-root", "point-root")),
                          ("pseg", ("signed-score", "signed-score-terminal", "stock-source",
                                    "stock-score", "stock-score-terminal", "factorial-source", "owner",
                                    "capture", "classical-root", "inference", "inference-terminal"))):
        child = sub.add_parser(name, allow_abbrev=False)
        for key in common + options:
            child.add_argument("--" + key, type=Path, required=True)
        child.add_argument("--protocol-sha256", required=True)
        if name == "pseg":
            child.add_argument("--original-readback", type=Path)
            for key in ("checkpoint-sha256", "runtime-sha256", "inference-worker-sha256"):
                child.add_argument("--" + key, required=True)
    args = parser.parse_args()
    require(sha256(args.protocol.read_bytes()).hexdigest() == args.protocol_sha256 and not args.output.exists(),
            "Frozen diagnostic and fresh output required")
    require((shapely.__version__, shapely.geos_version_string) == ("2.0.2", "3.11.2"), "Use original geometry runtime")
    selfcheck()
    views, lineage = (rsdd if args.dataset == "rsdd" else pseg)(args)
    result = dict(status=STATUS, phase="score", dataset=args.dataset, views=views, signed_parent_sha256=lineage,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), protocol_sha256=args.protocol_sha256,
                  historical_label_dependent_ceiling=True, detector_forwards=0, optimizer_updates=0, bootstrap_draws=4096)
    raw = (json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.mkdir()
    (args.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
