"""Score frozen three-seed contour fusion with exact shared edit counts."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

import numpy as np

PROTOCOL = "6a661116dd15c46c2f646372b6efd3ca0a3c2114bf1555e1cc3732ab8d5b0f43"
CAPTURE = "be37834f7ec657c938e15326ba04a9ac4a6dedaA4bf7a16c173490e93eb49cba".lower()
WITNESS = "53e0c9e200d92c8cef212e0d0d1c2b5b6f01d3b0d624434959cf8301af3c9880"
VERIFY = "f3fef91010fac2886970e94e267148def651a8c142ae107a5abbcd6d3620c863"
AREA = "448f37510bc87af1ce0bf2c483745b5966e243618783ebd8adc3b2d4815b52d1"
MATCH = "12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075"
MODES = ("raw", "anchor_count", "vote", "soft")
SEEDS = (5202, 5201, 5203)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def action_indices(hard, majority, anchor, mean3, cap):
    band_ids = np.arange(len(hard))
    disagreement = band_ids[majority != hard]
    order = np.lexsort((disagreement, -np.abs(mean3[disagreement] - .5)))
    vote = np.sort(disagreement[order[:cap]])
    adds = int((~hard[vote]).sum())
    drops = len(vote) - adds

    def same_counts(prob):
        off, on = band_ids[~hard], band_ids[hard]
        add = off[np.lexsort((off, -prob[off]))[:adds]]
        drop = on[np.lexsort((on, prob[on]))[:drops]]
        return np.sort(np.concatenate((add, drop)))

    return dict(vote=vote, soft=same_counts(mean3), anchor_count=same_counts(anchor))


def original_values(plane, mask, yy, xx, verify):
    values = verify.bilinear(plane, yy, xx, 800)
    if np.array_equal(values > 0, mask[yy, xx]):
        return values, False
    import torch
    import torch.nn.functional as nnf
    torch.set_num_threads(1)
    native = nnf.interpolate(torch.from_numpy(plane.copy())[None, None], size=(800, 800),
                             mode="bilinear", align_corners=False)[0, 0].numpy()
    values = native[yy, xx]
    require(np.array_equal(values > 0, mask[yy, xx]), "Exact native sign replay")
    return values, True


def band_edits(arms, planes, signal, verify, codec, expit):
    records = [arms[seed][rank] for seed, rank in
               ((5202, signal["rank"]), (5201, signal["donor_ranks"]["5201"]),
                (5203, signal["donor_ranks"]["5203"]))]
    hard = [codec.decode(record["rle"]).astype(bool) for record in records]
    require(all(mask.shape == (800, 800) for mask in hard), "Original native grid")
    yy, xx, inside = verify.band_points(np.flatnonzero(hard[0].ravel(order="F")))
    flags = [mask[yy, xx] for mask in hard]
    require(flags[0].sum() == inside, "Anchor band population")
    raw = [original_values(planes[seed][record["logit_row"]], mask, yy, xx, verify)
           for seed, record, mask in zip(SEEDS, records, hard, strict=True)]
    probs = [expit(values) for values, _ in raw]
    majority = np.sum(flags, axis=0) >= 2
    indices = action_indices(flags[0], majority, probs[0], np.mean(probs, axis=0), signal["cap"])
    counts = tuple(int(np.sum(flags[0][indices["vote"]] == positive)) for positive in (False, True))
    edits = {}
    for mode, selected in indices.items():
        if not len(selected):
            continue
        require(len(selected) == sum(counts) <= signal["cap"] and len(np.unique(selected)) == len(selected)
                and int(flags[0][selected].sum()) == counts[1], "Same counts and bounded raster support")
        edited = hard[0].copy()
        edited[yy[selected], xx[selected]] ^= True
        encoded = codec.encode(np.asfortranarray(edited, dtype=np.uint8))
        encoded["counts"] = encoded["counts"].decode("ascii")
        require(int(codec.area(encoded)) == records[0]["area"] + counts[0] - counts[1],
                "Exact after-edit area")
        edits[mode] = encoded
    return edits, counts, len(yy), sum(fallback for _, fallback in raw)


def prepare(args, area, verify, codec, expit):
    capture_info = area.json_value(area.pinned(args.capture / "RESULT_SAFE.json", CAPTURE))
    params = argparse.Namespace(capture=args.capture, capture_sha=CAPTURE, witness=args.witness, witness_sha=WITNESS)
    cases, counts, anchor_sha, signal_sha = area.read_admitted_capture(params)
    require(counts["mapped_native_candidates"] == 2620 and
            counts["covered_eligible_native_candidates"] == 468 and counts["covered_source_groups"] == 36,
            "Exact accepted candidate counts")
    arms = {x["seed"]: area.stream(args.capture / str(x["seed"]) / "CANDIDATES_PRIVATE.jsonl",
                                  x["prediction_sha256"]) for x in capture_info["arms"]}
    private = []
    visits = additions = drops = sampled = fallbacks = 0
    for image_i, case in enumerate(cases):
        blocks = {s: arms[s][image_i] for s in SEEDS}
        require(len({tuple(b[k] for k in ("ordinal", "group", "basename", "rgb_sha256"))
                     for b in blocks.values()}) == 1, "Exact across-seed image identity")
        candidates = {s: {r["rank"]: r for r in blocks[s]["candidates"]} for s in SEEDS}
        planes = {s: verify.array_from(args.capture / str(s), blocks[s]) for s in SEEDS}
        variants = {}
        for rank, signal in case["signals"].items():
            edited, (up, down), band, fallback = band_edits(candidates, planes, signal, verify, codec, expit)
            sampled += band
            fallbacks += fallback
            if edited:
                variants[rank] = edited
                visits += 1
                additions += up
                drops += down
        private.append(dict(basename=case["basename"], variants=variants))
    with (args.output / "MASK_EDITS_PRIVATE.jsonl").open("xb") as out:
        digest = sha256()
        for block in private:
            line = (json.dumps(block, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            out.write(line)
            digest.update(line)
    receipt = dict(status="HRSID_FIT_CONSENSUS_CONTOUR_PRELABEL", FIT=True, DEV=False, TEST=False,
                   GT_opened=False, original_candidates=2620, associated_candidates=468, edited_candidates=visits,
                   add_per_arm=additions, drop_per_arm=drops, native_band_positions=sampled,
                   author_torch_fallback_planes=fallbacks, variants_sha256=digest.hexdigest(),
                   anchor_sha256=anchor_sha, association_sha256=signal_sha, protocol_sha256=PROTOCOL,
                   capture_sha256=CAPTURE, witness_sha256=WITNESS)
    area.write_new(args.output / "PRELABEL_SAFE.json", receipt)
    return cases, private, receipt


def ap_score(gt, images, results):
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(StringIO()):
        dt = gt.loadRes(results)
        evaluator = COCOeval(gt, dt, "segm")
        evaluator.params.imgIds = images
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    ix = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .90)).item())
    strict_precision = evaluator.eval["precision"][ix, :, 0, 0, 2]
    strict_precision = strict_precision[strict_precision >= 0]
    require(len(strict_precision) > 0, "Exact COCO strict curve")
    return dict(mask_ap=100 * float(evaluator.stats[0]), ap75=100 * float(evaluator.stats[2]),
                ap90=100 * float(strict_precision.mean()), ar100=100 * float(evaluator.stats[8]))


def geometric_metrics(rle, truth, geom, codec):
    rectangles = geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]), *rle["size"])
    inter = geom.geometry.overlap_area(rectangles, truth)
    area = float(codec.area(rle))
    cap = min(area, truth.area)
    require(-1e-8 <= inter <= cap + 1e-8, "Exact native cell intersection")
    inter = min(cap, max(0.0, inter))
    return (100 * inter / (area + truth.area - inter),
            100 * (area - inter) / truth.area, 100 * (truth.area - inter) / truth.area)


def run_score(args, area, stock, geom, codec, cases, edits):
    from pycocotools.coco import COCO
    data = area.json_value(area.pinned(args.fit_index, area.LABEL_SHA, 64 * 2**20))
    require(len(data["images"]) == 2937 and len(data["annotations"]) == 9260, "Exact FIT label roster")
    names = {row["basename"] for row in cases}
    image_by_name = {im["file_name"]: im for im in data["images"] if im["file_name"] in names}
    require(set(image_by_name) == names and len(image_by_name) == 36, "Exact original image label join")
    ids = {im["id"] for im in image_by_name.values()}
    anns = defaultdict(list)
    for row in data["annotations"]:
        if row["image_id"] in ids:
            anns[row["image_id"]].append(row)
    subset = dict(images=list(image_by_name.values()), categories=data["categories"],
                  annotations=[a for aa in anns.values() for a in aa])
    targets, neutral, _ = geom.geometry.polygon_support(subset, geom.vector)
    gt = COCO()
    gt.dataset = subset
    with redirect_stdout(StringIO()):
        gt.createIndex()

    detections = {m: [] for m in MODES}
    geometry = defaultdict(list)
    counts = dict(edited_matches=0, edited_unmatched=0, edited_neutral=0, stock_matches=0)
    for row, changed in zip(cases, edits, strict=True):
        require(changed["basename"] == row["basename"], "Exact prelabel native mask join")
        image = image_by_name[row["basename"]]
        pool = row["candidates"]
        with redirect_stdout(StringIO()):
            matches, _, _ = stock.freeze_coco(image, anns[image["id"]], pool)
        counts["stock_matches"] += len(matches)
        by_rank = {rank: identity for identity, rank in matches.items()}
        for candidate in pool:
            rank = candidate["rank"]
            raw = candidate["rle"]
            variant = changed["variants"].get(rank, {})
            identity = by_rank.get(rank)
            valid = identity is not None and identity not in neutral and identity in targets
            if variant:
                counts["edited_matches" if valid else
                       "edited_unmatched" if identity is None else "edited_neutral"] += 1
            for mode in MODES:
                detections[mode].append(dict(image_id=image["id"], category_id=1, score=candidate["score"],
                                             segmentation=variant.get(mode, raw)))
            if valid and 32 <= candidate["area"] < 1024:
                truth = targets[identity][0]
                metrics = {mode: geometric_metrics(variant.get(mode, raw), truth, geom, codec) for mode in MODES}
                geometry[(row["group"], row["ordinal"] % 3)].append(metrics)
    require(all(len(items) == 2620 for items in detections.values()), "No candidate or confidence changes")
    require(sum(len(rr) for rr in geometry.values()) == 69 and len(geometry) == 31,
            "Original exact eligible fixed geometry denominator")

    ap = {m: ap_score(gt, sorted(ids), items) for m, items in detections.items()}
    per_group = {key: {m: tuple(mean(row[m][i] for row in rows) for i in range(3)) for m in MODES}
                 for key, rows in geometry.items()}
    metrics = {m: dict(
        **{field: mean(v[m][i] for v in per_group.values())
           for i, field in enumerate(("iou_points", "fp_per_truth_points", "fn_per_truth_points"))},
        strict_hits=sum(row[m][0] >= 90 for rows in geometry.values() for row in rows)) for m in MODES}
    fold_delta = {ctrl: [mean(v["soft"][0] - v[ctrl][0] for (_, f), v in per_group.items() if f == fold)
                         for fold in range(3)] for ctrl in ("raw", "anchor_count", "vote")}
    checks = dict(ap_vs_raw=ap["soft"]["mask_ap"] - ap["raw"]["mask_ap"] >= .50,
                  ap_vs_anchor_control=ap["soft"]["mask_ap"] - ap["anchor_count"]["mask_ap"] >= .30,
                  ap75_retained=ap["soft"]["ap75"] >= ap["raw"]["ap75"] - 1e-12,
                  fixed_geometry_fold_signs=all(sum(v > 0 for v in fold_delta[ctrl]) >= 2
                                                for ctrl in ("raw", "anchor_count")))
    return dict(AP=True, mask_metrics=ap, fixed_geometry=metrics, fixed_geometry_pairs=69,
                fixed_geometry_groups=31, soft_fixed_fold_delta=fold_delta, edited_candidate_scope=counts,
                gates=checks, advance=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("protocol", "capture", "witness", "fit-index", "code", "shapely-site", "output"):
        parser.add_argument("--" + item, required=True, type=Path)
    args = parser.parse_args()
    started = perf_counter()
    require(sha256(args.protocol.read_bytes()).hexdigest() == PROTOCOL, "Exact locked protocol")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    from scipy.special import expit
    import shapely
    from pycocotools import mask as codec
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry")
    from types import ModuleType
    path = args.code / "HRSID_FIT_SEED_AREA_CONFIRM_20261112/score/score_seed_area_confirm.py"
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == AREA, "Exact published carrier adapter")
    area = ModuleType("_consensus_area")
    area.__file__ = str(path)
    sys.modules[area.__name__] = area
    exec(compile(raw, str(path), "exec"), area.__dict__)
    verify = area.module("_consensus_verify", args.code /
                         "HRSID_FIT_SEED_AREA_CONFIRM_20261112/verify_area_signals.py", VERIFY)
    stock = area.module("_consensus_match", args.code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", MATCH)
    endpoint = area.module("_consensus_geometry", args.code /
                           "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", area.GEOMETRY_SHA)
    geom = endpoint.geometry_modules(args.code)
    args.output.mkdir()
    cases, edits, receipt = prepare(args, area, verify, codec, expit)
    result = run_score(args, area, stock, geom, codec, cases, edits)
    require(result["edited_candidate_scope"]["edited_matches"] +
            result["edited_candidate_scope"]["edited_unmatched"] +
            result["edited_candidate_scope"]["edited_neutral"] == receipt["edited_candidates"],
            "Exact edit scope accounting")
    result = dict(receipt, **result, status="HRSID_FIT_CONSENSUS_CONTOUR_ACTUAL", GT_opened=True,
                  model_forwards=0, optimizer_updates=0, source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  worker_seconds=perf_counter() - started)
    area.write_new(args.output / "RESULT_SAFE.json", result)
    print(json.dumps({key: result[key] for key in ("status", "edited_candidates", "gates", "advance")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only entry")
    # Torch's Windows import queries the local OS version before the offline hook.
    import torch
    torch.set_num_threads(1)
    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    main()
