"""Exact fixed-budget raster oracle bounds on a closed HRSID vote carrier."""
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
from types import ModuleType

import numpy as np

PROTOCOL = "6fca35ef4045b080293fe02eb79946ec92f741d774f1644aaadd78647da629ea"
BRIDGE_CODE = "2f3adfeabef3a4f3b39a5086f5223c5d709c52e6c3ec32de48326eb702830819"
BRIDGE_REPORT = "f331abcc0a3b2aa1ccac19be41d07b4e47dbf610a180f8469a66d2757aaff3f6"
BRIDGE_PRELABEL = "1f547c630e5b26a66ba51b558bdd76946c762313be7dcc6485b12c19df0fec12"
MODES = ("raw", "vote", "oracle_disagreement", "oracle_full_band")
CONTRASTS = (("vote", "raw"), ("oracle_disagreement", "vote"),
             ("oracle_full_band", "oracle_disagreement"), ("oracle_full_band", "raw"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source(path, digest, name):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, "Exact source " + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def oracle_counts(anchor, truth, support, adds, drops):
    add_sites = support & ~anchor
    drop_sites = support & anchor
    require(0 <= adds <= int(add_sites.sum()) and 0 <= drops <= int(drop_sites.sum()),
            "Exact edit counts must be feasible on both nested supports")
    return min(adds, int((add_sites & truth).sum())), min(drops, int((drop_sites & ~truth).sum()))


def confusion_iou(area, truth_area, overlap):
    union = area + truth_area - overlap
    require(0 <= overlap <= min(area, truth_area) and union > 0, "Exact nonempty mask support")
    return overlap / union


def geometry(area, truth_area, intersection, good_add=0, good_drop=0):
    union = area + truth_area - intersection
    return dict(iou=confusion_iou(area, truth_area, intersection),
                at75=4 * intersection >= 3 * union, at90=10 * intersection >= 9 * union,
                fp_per_truth=(area - intersection) / truth_area,
                fn_per_truth=(truth_area - intersection) / truth_area, good_add=good_add, good_drop=good_drop)


def score_mask(anchor, truth, vote, donors, bridge):
    area, truth_area, original = int(anchor.sum()), int(truth.sum()), int((anchor & truth).sum())
    require(truth_area > 0 and anchor.shape == truth.shape == vote.shape, "Exact same native grid")
    raw = geometry(area, truth_area, original)
    if donors is None:
        require(np.array_equal(anchor, vote), "Unsupported vote must stay raw")
        return dict(modes={mode: raw for mode in MODES}, adds=0, drops=0, supported=False,
                    disputed_pixels=0, full_band_pixels=0, disputed_helpful_add_sites=0,
                    disputed_helpful_drop_sites=0, full_helpful_add_sites=0, full_helpful_drop_sites=0,
                    cap_bound=False, coreless=False)
    donor1, donor2 = donors
    require(donor1.shape == donor2.shape == anchor.shape, "Exact donor grid")
    yy, xx = np.nonzero(anchor)
    h, w = anchor.shape
    region = np.s_[max(0, yy.min()-2):min(h, yy.max()+3), max(0, xx.min()-2):min(w, xx.max()+3)]
    a, y, v, d1, d2 = [mask[region] for mask in (anchor, truth, vote, donor1, donor2)]
    band, _, core = bridge.two_layers(a)
    disagreement = band & (d1 != a) & (d2 != a)
    cap = 2 * min(8, area // 10)
    expected = np.flatnonzero(disagreement)[:cap]
    actual_edits = a ^ v
    require(np.array_equal(expected, np.flatnonzero(actual_edits))
            and len(expected) == int((anchor ^ vote).sum()), "Exact closed RLE vote and in-frame edit support")
    add_mask, drop_mask = actual_edits & ~a, actual_edits & a
    adds, drops = int(add_mask.sum()), int(drop_mask.sum())
    revised_area = area + adds - drops

    def changed(good_add, good_drop):
        return geometry(revised_area, truth_area, original + good_add - drops + good_drop, good_add, good_drop)

    modes = dict(raw=raw, vote=changed(int((add_mask & y).sum()), int((drop_mask & ~y).sum())),
                 oracle_disagreement=changed(*oracle_counts(a, y, disagreement, adds, drops)),
                 oracle_full_band=changed(*oracle_counts(a, y, band, adds, drops)))
    require(modes["vote"]["iou"] <= modes["oracle_disagreement"]["iou"] <= modes["oracle_full_band"]["iou"],
            "Vote and oracle nested ordering")
    return dict(modes=modes, adds=adds, drops=drops, supported=True, disputed_pixels=int(disagreement.sum()),
                full_band_pixels=int(band.sum()), disputed_helpful_add_sites=int((disagreement & ~a & y).sum()),
                disputed_helpful_drop_sites=int((disagreement & a & ~y).sum()),
                full_helpful_add_sites=int((band & ~a & y).sum()),
                full_helpful_drop_sites=int((band & a & ~y).sum()),
                cap_bound=int(disagreement.sum()) > cap, coreless=not core.any())


def aggregate(records):
    groups = defaultdict(list)
    for source_group, row in records:
        groups[source_group].append(row)
    total = [row for _, row in records]
    edited = [row for row in total if row["adds"] + row["drops"] > 0]
    add_count, drop_count = (sum(row[k] for row in total) for k in ("adds", "drops"))
    by_mode = {}
    for mode in MODES:
        key = lambda row: row["modes"][mode]
        data = [key(row) for row in total]
        by_mode[mode] = dict(
            group_equal_iou_points=100 * mean(mean(key(row)["iou"] for row in rows) for rows in groups.values()),
            pooled_iou_points=100 * mean(r["iou"] for r in data),
            edited_only_pooled_iou_points=100 * mean(key(row)["iou"] for row in edited) if edited else None,
            mean_fp_per_truth_points=100 * mean(mean(key(row)["fp_per_truth"] for row in rows)
                                                for rows in groups.values()),
            mean_fn_per_truth_points=100 * mean(mean(key(row)["fn_per_truth"] for row in rows)
                                                for rows in groups.values()),
            helpful_adds=sum(r["good_add"] for r in data), helpful_drops=sum(r["good_drop"] for r in data),
        )
        for threshold in ("at75", "at90"):
            by_mode[mode][threshold] = dict(
                hits=sum(r[threshold] for r in data),
                gains=sum(key(row)[threshold] and not row["modes"]["raw"][threshold] for row in total),
                losses=sum(not key(row)[threshold] and row["modes"]["raw"][threshold] for row in total))
        if mode != "raw":
            correct_a, correct_d = by_mode[mode]["helpful_adds"], by_mode[mode]["helpful_drops"]
            by_mode[mode]["add_precision"] = correct_a / add_count if add_count else None
            by_mode[mode]["drop_precision"] = correct_d / drop_count if drop_count else None
            by_mode[mode]["edit_precision"] = (correct_a + correct_d) / (add_count + drop_count) if edited else None
    group_names = sorted(groups)
    draws = np.random.default_rng(5202).integers(len(group_names), size=(10000, len(group_names)))
    contrasts = {}
    for left, right in CONTRASTS:
        delta = lambda row: 100 * (row["modes"][left]["iou"] - row["modes"][right]["iou"])
        values = np.asarray([mean(delta(row) for row in groups[g]) for g in group_names])
        contrasts[left + "_minus_" + right] = dict(
            group_equal_delta_points=float(values.mean()),
            ci95_points=np.percentile(values[draws].mean(axis=1), [2.5, 97.5]).tolist(),
            positive_rows=sum(delta(row) > 0 for row in total),
            negative_rows=sum(delta(row) < 0 for row in total))
    bounds = ("disputed_pixels", "full_band_pixels", "disputed_helpful_add_sites",
              "disputed_helpful_drop_sites", "full_helpful_add_sites", "full_helpful_drop_sites")
    return dict(matches=len(total), groups=len(groups), edited_matches=len(edited),
                supported_matches=sum(row["supported"] for row in total),
                edited_groups=len({group for group, row in records if row["adds"]+row["drops"] > 0}),
                adds=add_count, drops=drop_count, cap_bound_matches=sum(row["cap_bound"] for row in total),
                coreless_supported_matches=sum(row["coreless"] for row in total),
                support_totals={k: sum(row[k] for row in total) for k in bounds},
                modes=by_mode, contrasts=contrasts, bootstrap_draws=10000, bootstrap_seed=5202)


def prepare(args, prior, area):
    receipt = area.json_value(area.pinned(args.bridge / "PRELABEL_SAFE.json", BRIDGE_PRELABEL))
    summary = area.json_value(area.pinned(args.bridge / "RESULT_SAFE.json", BRIDGE_REPORT))
    require(area.fit_only(receipt) and receipt["logit_planes_opened"] is False and summary["AP"]
            and summary["source_sha256"] == BRIDGE_CODE and summary["edited_candidates"] == 464
            and all(receipt[key] == summary[key] for key in receipt if key not in ("status", "GT_opened")),
            "Exact closed source voting evidence")
    args2 = argparse.Namespace(capture=args.capture, capture_sha=prior.CAPTURE, witness=args.witness,
                               witness_sha=prior.WITNESS)
    cases, counts, anchor_sha, signal_sha = area.read_admitted_capture(args2)
    require(counts["covered_eligible_native_candidates"] == 468 and anchor_sha == receipt["anchor_sha256"]
            and signal_sha == receipt["associations_sha256"], "Exact frozen source cohort and associations")
    cap = area.json_value(area.pinned(args.capture / "RESULT_SAFE.json", prior.CAPTURE))
    donors = {item["seed"]: area.stream(args.capture / str(item["seed"]) / "CANDIDATES_PRIVATE.jsonl",
                                        item["prediction_sha256"]) for item in cap["arms"] if item["seed"] != 5202}
    edits = area.stream(args.bridge / "MASK_EDITS_PRIVATE.jsonl", receipt["variants_sha256"])
    require(len(edits) == 36 and sum(len(row["variants"]) for row in edits) == 464, "Exact archived RLE edits")
    bundle = dict(status="HRSID_FIT_RLE_VOTE_HEADROOM_PRELABEL", FIT=True, DEV=False, TEST=False,
                  GT_opened=False, bridge_result_sha256=BRIDGE_REPORT, bridge_prelab_sha256=BRIDGE_PRELABEL,
                  protocol_sha256=PROTOCOL, candidate_rows=2620, candidate_associations=468,
                  images=36, variants_sha256=receipt["variants_sha256"], logit_planes_opened=False)
    area.write_new(args.output / "PRELABEL_SAFE.json", bundle)
    return cases, donors, edits, bundle


def run(args):
    started = perf_counter()
    require(sha256(args.protocol.read_bytes()).hexdigest() == PROTOCOL, "Exact prospective scoring protocol")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask as codec
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry version")
    code = args.code
    bridge = source(code / "HRSID_FIT_RLE_VOTE_SOURCE_BRIDGE_20261112/score_binary_bridge.py",
                    BRIDGE_CODE, "_headroom_bridge")
    prior = source(code / "HRSID_FIT_CONSENSUS_CONTOUR_20261112/score_consensus.py", bridge.PRIOR,
                   "_headroom_prior")
    area = source(code / "HRSID_FIT_SEED_AREA_CONFIRM_20261112/score/score_seed_area_confirm.py", prior.AREA,
                  "_headroom_area")
    stock = area.module("_headroom_match", code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", prior.MATCH)
    endpoint = area.module("_headroom_geometry", code /
                           "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", area.GEOMETRY_SHA)
    geom = endpoint.geometry_modules(code)
    args.output.mkdir()
    cases, donors, edits, receipt = prepare(args, prior, area)
    dataset = area.json_value(area.pinned(args.fit_index, area.LABEL_SHA, 64 * 2**20))
    names = {row["basename"] for row in cases}
    images = {i["file_name"]: i for i in dataset["images"] if i["file_name"] in names}
    ids = {i["id"] for i in images.values()}
    require(set(images) == names and len(ids) == 36 and len(dataset["images"]) == 2937
            and len(dataset["annotations"]) == 9260, "Exact FIT roster")
    anns = defaultdict(list)
    for row in dataset["annotations"]:
        if row["image_id"] in ids:
            anns[row["image_id"]].append(row)
    subset = dict(images=list(images.values()), categories=dataset["categories"],
                  annotations=[a for group in anns.values() for a in group])
    targets, neutral, _ = geom.geometry.polygon_support(subset, geom.vector)
    rows = []
    decode = lambda rle: codec.decode(rle).astype(bool)
    for index, (case, edit) in enumerate(zip(cases, edits, strict=True)):
        image = images[case["basename"]]
        pool = case["candidates"]
        with redirect_stdout(StringIO()):
            fixed, rles, _ = stock.freeze_coco(image, anns[image["id"]], pool)
        ranks = {row["rank"]: row for row in pool}
        require(edit["basename"] == case["basename"] and all(donors[s][index]["basename"] == case["basename"]
                and donors[s][index]["group"] == case["group"]
                and donors[s][index]["ordinal"] == case["ordinal"] for s in (5201, 5203)),
                "Exact three seed image keys and RLE vote join")
        donor_by_rank = {s: {r["rank"]: r for r in donors[s][index]["candidates"]} for s in (5201, 5203)}
        for identity, rank in fixed.items():
            original = ranks[rank]
            if identity in neutral or identity not in targets or not 32 <= original["area"] < 1024:
                continue
            original_rle = original["rle"]
            voted = edit["variants"].get(str(rank), {}).get("vote", original_rle)
            signal = case["signals"].get(rank)
            pair = ([decode(donor_by_rank[s][signal["donor_ranks"][str(s)]]["rle"]) for s in (5201, 5203)]
                    if signal else None)
            rows.append((case["group"], score_mask(decode(original_rle), decode(rles[identity]),
                                                   decode(voted), pair, bridge)))
    result = aggregate(rows)
    require(result["matches"] == 69 and result["groups"] == 31 and result["edited_matches"] == 68,
            "Exact prior fixed source match scope")
    report = dict(receipt, **result, status="HRSID_FIT_RLE_VOTE_HEADROOM_ACTUAL", GT_opened=True,
                  AP=False, model_forwards=0, optimizer_updates=0, worker_seconds=perf_counter()-started,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    area.write_new(args.output / "RESULT_SAFE.json", report)
    print(json.dumps({key: report[key] for key in ("status", "matches", "groups", "edited_matches", "contrasts")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "bridge", "capture", "witness", "code", "fit-index", "shapely-site", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    run(parser.parse_args())
