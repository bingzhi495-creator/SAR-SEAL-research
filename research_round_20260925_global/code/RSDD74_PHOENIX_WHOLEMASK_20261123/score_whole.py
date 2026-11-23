"""Score frozen full-mask Phoenix writers against archived native and matched-count arms."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from PIL import Image
from pycocotools import mask as codec
import shapely


LOCK_NAME = "RSDD74_PHOENIX_WHOLEMASK_PRELOCK_20261123.md"
LOCK = "57cf64a2881325446cd61f548933a9924fc611a3633d871d17ddb641845be40e"
HISTORIC = "a4f729f0154a56b17a75af06c3c1e880ca25f0fe029f7b9eccab175fc597e1f3"
REQUEST_WORKER = "f96cea8f68ed6182c4620d0773bb71d215d01ec85e5e7eb630b18927b8d8c0ad"
REQUEST_RESULT = "4245299e901531796a019978e039c8f2b3814128f4880aa4d2c6016a5116d55a"
REQUEST_TERMINAL = "0ac00ddd6894e6b5284afacb9cf1656d55e92b4195cb6dea6290b4fd0b4c5063"
INFER_WORKER = "1a7b423bd681edf3c83ab1c5a71125e3fd20c9bee9b7d3f344a2eae39ae088cd"
ROSTER = "91f95a50e19c9809b9a84f9e5825d2b6a31476d970dae727f144e852b60d4d20"
POLICIES = ("lvis", "coco")
OCCUPANCY = tuple(f"{s}/learned" for s in (5217, 5218, 5219))
MATCHED = ("raw", "phoenix", *OCCUPANCY)
ARMS = (*MATCHED, *POLICIES)
STATUS = "PHOENIX_RSDD74_WHOLE_SCORE_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def native_area(rle):
    """Validate the complete compressed COCO run stream before handing it to the C codec."""
    require(isinstance(rle, dict) and set(rle) == {"size", "counts"} and rle["size"] == [512, 512]
            and isinstance(rle["counts"], str) and rle["counts"], "Expected native 512-square compressed RLE")
    counts = []
    value = bits = 0
    for char in rle["counts"]:
        digit = ord(char) - 48
        require(0 <= digit < 64 and bits <= 20, "Invalid compressed RLE digit")
        value |= (digit & 31) << bits
        bits += 5
        if digit & 32:
            continue
        if digit & 16:
            value -= 1 << bits
        if len(counts) > 2:
            value += counts[-2]
        require(0 <= value <= 512**2, "Invalid compressed RLE run length")
        counts.append(value)
        value = bits = 0
    require(bits == 0 and sum(counts) == 512**2, "Mask run lengths must fill the complete image")
    area = sum(counts[1::2])
    require(area == int(codec.area(rle)), "Mask area disagrees with COCO codec")
    return area


def attach_image(ordinal, native, candidates, output, changes):
    require(set(output) == {"ordinal", *POLICIES} and type(output["ordinal"]) is int and output["ordinal"] == ordinal,
            "Whole mask output image order differs")
    areas = []
    for pos, (stock, cand) in enumerate(zip(native, candidates, strict=True), 1):
        require(set(cand) == {"position", "segmentation"} and type(cand["position"]) is int
                and cand["position"] == pos and cand["segmentation"] == stock, "Native request position or RLE differs")
        areas.append(int(codec.area(stock)))
    counters = {}
    for policy in POLICIES:
        expected = [i for i, area in enumerate(areas, 1) if (area > 0 if policy == "lvis" else 100 * area > 512**2)]
        require(isinstance(output[policy], list) and len(output[policy]) == len(expected),
                "Whole writer must return all and only eligible positions")
        totals = Counter(decoded_candidates=len(expected), pass_through=len(native)-len(expected),
                         decoder_batches=(len(expected)+1)//2, empty_outputs=0, unchanged_outputs=0, adds=0, drops=0)
        for pos, entry in zip(expected, output[policy], strict=True):
            require(set(entry) == {"position", "segmentation"} and type(entry["position"]) is int
                    and entry["position"] == pos, "Whole writer duplicated, displaced or reordered an eligible position")
            mask = entry["segmentation"]
            area = native_area(mask)
            overlap = int(codec.area(codec.merge([native[pos-1], mask], intersect=True)))
            adds, drops = area - overlap, areas[pos-1] - overlap
            require(min(adds, drops) >= 0, "Negative native-relative area")
            totals.update(empty_outputs=int(area == 0), unchanged_outputs=int(adds+drops == 0), adds=adds, drops=drops)
            changes.setdefault(pos, {})[policy] = {"rle":mask}
        counters[policy] = totals
    return counters


def geometry_vector(iou, area, truth_area):
    overlap = iou * (area + truth_area) / (1 + iou)
    require(truth_area > 0 and 0 <= overlap <= min(area, truth_area) + 1e-7, "Invalid fixed polygon intersection")
    return [100*iou, 100*(area-overlap)/truth_area, 100*(truth_area-overlap)/truth_area,
            100*overlap/truth_area, overlap]


def decision(ap, ci, matches, date_blocks):
    exposed = matches >= 20 and date_blocks >= 8
    return dict(eligible_matches=matches, sensor_date_blocks=date_blocks, exposure_supported=exposed,
                supported_advantage=bool(exposed and ap is not None and ap >= .30 and ci and ci[0] > 0),
                supported_loss=bool(exposed and ap is not None and ap <= -.30 and ci and ci[1] < 0))


def frozen_sources(a):
    path = a.code / "RSDD74_PHOENIX_SIGNED_SIDES_20261123/score.py"
    source = path.read_bytes()
    require(digest(source) == HISTORIC, "Expected the archived signed-side dependency loader")
    sides = ModuleType("historic_sources")
    sides.__file__ = str(path)
    exec(compile(source, str(path), "exec"), sides.__dict__)
    return sides, sides.sources(a)


def verify_inputs(a, sides, dependencies):
    phx, api, cfa, _, _, _, _, adapter, core = dependencies
    prior, prior_sha, prior_term = api.observed(a.phx_score, a.phx_score_terminal, "score", phx.SCORE,
                                                sides.PHX_SCORE, sides.PHX_TERM, sides.PHX_WORKER)
    old, old_sha, old_term = api.observed(a.phx_edits, a.phx_edits_terminal, "edits", phx.EDIT,
                                          prior["edits_result_sha256"], prior["edits_terminal_sha256"], sides.PHX_WORKER)
    require(prior["historical_arms_exact"] is True and prior["edits_sha256"] == old["edits_sha256"]
            and old["target_labels_opened"] is False, "Archived equal-count references required")
    req, rsha, rtsha = api.observed(a.request, a.request_terminal, "request", "PHOENIX_RSDD74_WHOLE_REQUEST_COMPLETE",
                                    REQUEST_RESULT, REQUEST_TERMINAL, REQUEST_WORKER)
    infer, isha, itsha = api.observed(a.inference, a.inference_terminal, "inference",
                                      "PHOENIX_RSDD74_WHOLE_INFERENCE_COMPLETE", worker=INFER_WORKER)
    require(infer["phase"] == "inference" and req["phase"] == "request" and
            infer["request_result_sha256"] == rsha and infer["request_terminal_sha256"] == rtsha
            and infer["request_sha256"] == req["request_sha256"]
            and req["protocol_sha256"] == infer["protocol_sha256"] == LOCK
            and req["target_labels_opened"] is infer["target_labels_opened"] is False
            and infer["native_detector_forwards"] == infer["optimizer_updates"] == 0
            and infer["baseline_worker_sha256"] == phx.INFER_SHA and infer["author_commit"] == phx.AUTHOR
            and infer["weight_sha256"] == phx.WEIGHT_SHA
            and tuple(infer[k] for k in ("workers","threads","batch_size","refine_iters","smooth_kernel","seed"))
                == (4, 2, 2, 5, 1, 5202)
            and req["ordered_native_population_sha256"] == infer["native_projection_sha256"] == ROSTER
            and all(infer[k] is True for k in ("binary_writer","multimask_output","binarize_input","policy_decodes_separate"))
            and infer["runtime"] == {"FastGeodis":"1.0.5", "numpy":"1.26.4", "torch":"2.10.0+cu130",
                                     "Pillow":"12.3.0","pycocotools":"2.0.11"},
            "Frozen image-only request, CPU writer and observed results required")
    payload = json.loads(api.pin(a.request / "REQUEST_PRIVATE.json", req["request_sha256"]))
    require(set(payload) == {"status", "protocol_sha256", "images"}
            and payload["status"] == "PHOENIX_RSDD74_WHOLE_REQUEST" and payload["protocol_sha256"] == LOCK
            and len(payload["images"]) == 74, "Expected complete image-only request")
    stream = api.pin(a.whole_masks or a.inference / "WHOLE_MASKS_PRIVATE.jsonl", infer["masks_sha256"])
    inputs = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = inputs.load_population(a)
    require(lineage == prior["input_lineage"] == old["input_lineage"]
            and req["stock_sha256"] == lineage["predictions_sha256"], "Historical native inputs differ")
    archive = api.pin(a.phx_edits / "EDITS_PRIVATE.jsonl", old["edits_sha256"])
    cfa.EDITED = (*cfa.EDITED, "phoenix")
    blocks, quota, _ = cfa.checked_blocks(rows, map(json.loads, archive.splitlines()), adapter, core)
    require(quota == dict(mapped_candidates=7346, eligible_candidates=4137, positive_budget_candidates=4107,
                          swaps_per_arm=29900), "Expected complete original signed-count controls")
    counts = {arm: Counter() for arm in POLICIES}
    total_native = total_empty = encodes = images = 0
    for j, (row, image, out) in enumerate(zip(rows, payload["images"], map(json.loads, stream.splitlines()), strict=True), 1):
        item, audit, pred = row[:3]
        image_id = item["image_id"]
        require(set(image) == {"image_file", "encoded_sha256", "decoded_rgb_sha256", "candidates"}
                and image["image_file"] == f"image_{j:03d}.png"
                and image["decoded_rgb_sha256"] == item["rgb_sha256"] == audit["rgb_sha256"],
                "Whole request native image identity differs")
        with Image.open(BytesIO(api.pin(a.request / image["image_file"], image["encoded_sha256"]))) as png:
            require(png.format == "PNG" and png.size == (512,512) and png.mode == "RGB"
                    and digest(png.tobytes()) == item["rgb_sha256"], "Decoded native image differs")
        native = [entry["segmentation"] for entry in pred["coco"]]
        census = attach_image(j, native, image["candidates"], out, blocks[image_id]["changes"])
        for arm in POLICIES:
            counts[arm].update(census[arm])
        images += 1
        total_native += len(native)
        total_empty += census["lvis"]["pass_through"]
        encodes += int(census["lvis"]["decoded_candidates"] > 0)
    verified = dict(images=images, native_candidates=total_native, native_empty=total_empty, image_encodes=encodes,
                    **{arm:dict(c) for arm,c in counts.items()})
    require((images,total_native,total_empty,encodes) == (74,7346,1267,74)
            and counts["lvis"]["decoded_candidates"] == 6079 and counts["coco"]["decoded_candidates"] == 172
            and all(infer["counts"][key] == value if not isinstance(value,dict) else
                    all(infer["counts"][key][k] == v for k,v in value.items()) for key,value in verified.items()),
            "Independent whole-mask policy census disagrees with inference")
    require(req["counts"] == dict(images=images, native_candidates=total_native, native_empty=total_empty,
                lvis_candidates=counts["lvis"]["decoded_candidates"], coco_candidates=counts["coco"]["decoded_candidates"]),
            "Area policies disagree with independent image-only request")
    for arm in POLICIES:
        c = verified[arm]
        c.update(changed_candidates=c["decoded_candidates"]-c["unchanged_outputs"],
                 signed_area_change=c["adds"]-c["drops"], total_unchanged=c["unchanged_outputs"]+c["pass_through"])
        actual = infer["counts"][arm]
        require((actual["changed_outputs"], actual["all_unchanged_outputs"], actual["signed_area_delta"],
                 actual["candidate_refinement_iterations"]) == (c["changed_candidates"], c["total_unchanged"],
                 c["signed_area_change"], 5*c["decoded_candidates"]), "Derived inference counts differ")
    lineage_info = dict(input_lineage=lineage, historical_score_sha256=prior_sha, historical_score_terminal_sha256=prior_term,
                        historical_edits_result_sha256=old_sha, historical_edits_terminal_sha256=old_term,
                        historical_edits_sha256=old["edits_sha256"], request_sha256=req["request_sha256"],
                        request_worker_sha256=REQUEST_WORKER, request_result_sha256=rsha, request_terminal_sha256=rtsha,
                        inference_worker_sha256=INFER_WORKER, inference_result_sha256=isha, inference_terminal_sha256=itsha,
                        masks_sha256=infer["masks_sha256"], author_commit=phx.AUTHOR, weight_sha256=phx.WEIGHT_SHA)
    return blocks, prior, verified, lineage_info, infer


PAIRS = tuple((arm, base) for arm in POLICIES for base in MATCHED) + (("lvis","coco"),)


def fixed_geometry(anchors, blocks, polygons, annotations, dates, cfa, core, reserve):
    groups = defaultdict(list)
    kept = Counter({arm:0 for arm in ARMS})
    counts = Counter(native_matches=len(anchors), eligible_matches=0, invalid_polygon_matches=0, area_excluded_matches=0)
    for gt_id, (image, pos, det) in anchors.items():
        line = annotations[gt_id]
        raster = codec.merge(codec.frPyObjects(line["segmentation"], 512,512))
        changes = blocks[image]["changes"].get(pos, {})
        stock = det["segmentation"]
        masks = [changes.get(arm, {}).get("rle",stock) for arm in ARMS]
        overlaps = codec.iou(masks, [raster], [False])[:,0]
        require(overlaps[0] >= .50, "Reconstructed original COCO .50 assignment differs")
        kept.update({arm:int(val >= .50) for arm,val in zip(ARMS, overlaps, strict=True)})
        poly = polygons[gt_id]
        if not poly.is_valid or poly.area <= 0:
            counts["invalid_polygon_matches"] += 1
            continue
        if not 32 <= int(codec.area(stock)) < 1024:
            counts["area_excluded_matches"] += 1
            continue
        counts["eligible_matches"] += 1
        memo = {}
        for mask in masks:
            key = mask["counts"]
            if key not in memo:
                iou = core.geometric_iou(mask, poly, shapely)
                memo[key] = geometry_vector(iou, int(codec.area(mask)), poly.area)
        groups[blocks[image]["group"]].append([memo[m["counts"]] for m in masks])
    parents = sorted(groups)
    keys = [dates[p] for p in parents]
    result = dict(counts=dict(counts), represented_groups=len(parents), sensor_date_blocks=len(set(keys)),
                  native_anchor_retained_50=dict(kept), group_equal_arms={}, paired={})
    if parents:
        means = np.stack([np.mean(groups[p], axis=0) for p in parents])
        result["group_equal_arms"] = {arm:dict(zip(cfa.FIELDS, means[:,i].mean(axis=0).tolist(), strict=True))
                                      for i,arm in enumerate(ARMS)}
        for arm,base in PAIRS:
            diff = means[:,ARMS.index(arm)]-means[:,ARMS.index(base)]
            result["paired"][arm+"_minus_"+base] = cfa.interval(diff, keys, reserve)
    return result


def score_view(ids, gt, all_blocks, polygons, dates, reference, dependencies):
    phx, _, cfa, history, reserve, _, evaluator, _, core = dependencies
    blocks = {key:value for key,value in all_blocks.items() if key in ids}
    part = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids])
    annotations = {r["id"]:r for r in part["annotations"]}
    require(len(annotations) == len(part["annotations"]) and all(not r.get("iscrowd") and not r.get("ignore")
            for r in annotations.values()), "All original hulls must be countable under the raster coverage denominator")
    hull_counts = Counter(blocks[r["image_id"]]["group"] for r in annotations.values())
    metrics, coverage = {}, {}
    for arm in ARMS:
        output, matched = evaluator.score_one(part, blocks, arm, anchors=True)
        require(len(output) == 13, "Incomplete historical COCO metric vector")
        metrics[arm] = {field:history.available(value) for field,value in output.items()}
        matched_groups = Counter(blocks[annotations[g]["image_id"]]["group"] for g in matched)
        positives = 0
        for block in blocks.values():
            for pos, entry in zip(block["ranks"], block["stock"], strict=True):
                mask = block["changes"].get(pos,{}).get(arm,{}).get("rle", entry["segmentation"])
                positives += int(int(codec.area(mask)) > 0)
        coverage[arm] = dict(nonempty_candidates=positives, matched_hulls_50=len(matched),
                             all_hulls=len(annotations), raster_recall_50=len(matched)/len(annotations),
                             parents_with_match_50=len(matched_groups), parents_with_hulls=len(hull_counts),
                             equal_parent_raster_recall_50=float(np.mean([matched_groups[g]/n for g,n in hull_counts.items()])))
        if arm == "raw":
            anchors = matched
    geometry = fixed_geometry(anchors, blocks, polygons, annotations, dates, cfa, core, reserve)
    for arm in ARMS:
        coverage[arm].update(original_matched_hulls=len(anchors), original_matched_retained_50=geometry["native_anchor_retained_50"][arm])
    census = dict(images=len(blocks), annotations=len(annotations), source_groups=len({b["group"] for b in blocks.values()}))
    result = dict(**census, metrics_fraction=metrics, geometry=geometry, coverage=coverage)
    phx.parity(result, reference, MATCHED)
    require(geometry["counts"] == reference["geometry"]["counts"] and
            all(geometry[k] == reference["geometry"][k] for k in ("represented_groups","sensor_date_blocks")),
            "Historical fixed-match population or date grouping differs")
    contrasts = {}
    for arm,base in PAIRS:
        name = arm+"_minus_"+base
        effects = history.difference(metrics[arm], metrics[base])
        paired = geometry["paired"].get(name)
        criteria = decision(effects["AP"], paired["continuous_iou_points"]["interval_95"] if paired else None,
                            geometry["counts"]["eligible_matches"], geometry["sensor_date_blocks"])
        contrasts[name] = dict(COCO_points=effects, criteria=criteria,
                               matched_hull_count_delta=coverage[arm]["matched_hulls_50"]-coverage[base]["matched_hulls_50"],
                               fixed_retained_count_delta=coverage[arm]["original_matched_retained_50"]-
                                                          coverage[base]["original_matched_retained_50"])
    result["contrasts"] = contrasts
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code","stock-root","point-root","phx-edits","phx-edits-terminal","phx-score","phx-score-terminal",
                 "request","request-terminal","inference","inference-terminal","author-labels","protocol","output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--whole-masks", type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    for key,v in vars(a).items():
        if isinstance(v, Path):
            setattr(a, key, v.resolve(strict=key != "output"))
    require(a.protocol.name == LOCK_NAME and digest(a.protocol.read_bytes()) == a.protocol_sha256 == LOCK
            and not a.output.exists(), "Published whole-mask protocol and fresh output required")
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Historical geometry runtime")
    a.cohort, a.support_root = 74, None
    start = perf_counter()
    sides, deps = frozen_sources(a)
    phx, api, _, history, reserve, old, _, _, _ = deps
    blocks, prior, counts, lineage, infer = verify_inputs(a, sides, deps)
    print(json.dumps(dict(phase="closed_masks_verified", images=counts["images"])), flush=True)
    # All ordered stock/edited masks, observed receipts, images and source byte identities have passed.
    gt, rekeyed, polygons, sets, dates, labels_sha = history.views_and_labels(a, old, reserve, blocks)
    cache, views = {}, {}
    for name, ids in sets.items():
        key = frozenset(ids)
        if key not in cache:
            cache[key] = score_view(ids, gt, blocks, polygons, dates, prior["views"][name], deps)
        views[name] = cache[key]
        phx.parity(views[name], prior["views"][name], MATCHED)
        print(json.dumps(dict(phase="scored_view", view=name, images=len(ids))), flush=True)
    require(len(views) == 5 and views["full"]["geometry"]["counts"]["eligible_matches"] == 47
            and views["date_separated"]["geometry"]["counts"]["eligible_matches"] == 32,
            "Predeclared views and frozen primary/secondary geometry denominators required")
    key = "lvis_minus_phoenix"
    primary = views["full"]["contrasts"][key]
    primary_iou = views["full"]["geometry"]["paired"][key]["continuous_iou_points"]
    result = dict(phase="score", status=STATUS, worker_sha256=digest(Path(__file__).read_bytes()),
                  protocol_sha256=LOCK, counts=counts, **lineage, historical_references_exact=True,
                  author_labels_sha256=labels_sha, author_ids_rekeyed=rekeyed, target_labels_opened=True,
                  non_simple_polygons_retained_in_AP=sum(not p.is_valid for p in polygons.values()),
                  score_detector_forwards=0, score_optimizer_updates=0,
                  inference_timing={k:v for k,v in infer.items() if "seconds" in k or "timing" in k},
                  score_seconds=perf_counter()-start, bootstrap_draws=4096, bootstrap_seed=5217,
                  cluster_unit="sensor/date; equal parent weight", primary_view="full", primary_contrast=key,
                  primary=dict(**primary, paired_continuous_iou_points=primary_iou), views=views,
                  views_below_exposure_floor=sum(not view["contrasts"][key]["criteria"]["exposure_supported"] for view in views.values()))
    a.output.mkdir()
    receipt = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=receipt)), flush=True)


if __name__ == "__main__":
    main()
