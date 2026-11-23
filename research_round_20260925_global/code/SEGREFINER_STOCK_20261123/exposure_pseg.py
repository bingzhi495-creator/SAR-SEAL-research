"""Audit the stock SegRefiner-LR area rule on previously qualified PSeg masks."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType


SOURCE_SHA256 = "5a638df3638a8647dc49b81a1f63ee9d72bee2d41dc1e94e276b5a0d2016fdc1"
PROTOCOL_NAME = "SEGREFINER_STOCK_PRELOCK_20261123.md"
PROTOCOL_SHA256 = "30007c832d47c6e086445d474d4f319642854de20da7844706ebac8e40a7df9c"
STATUS = "PSEG232_SEGREFINER_STOCK_EXPOSURE_COMPLETE"
BINS = (("eq_0", 0, 1), ("ge_1_lt_32", 1, 32), ("ge_32_lt_512", 32, 512),
        ("ge_512_lt_1024", 512, 1024), ("ge_1024", 1024, None))
VIEWS = {"official": (232, 546, 402, 293, 143),
         "exclude_exact_fit_overlap": (222, 502, 381, 279, 138)}


def load_source(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Exact historical PSeg scorer required")
    source = ModuleType(path.stem)
    source.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), source.__dict__)
    return source


def area_bin(area):
    if type(area) is not int or area < 0:
        raise ValueError("Native segmentation area must be a nonnegative integer")
    return next(name for name, lo, hi in BINS if lo <= area and (hi is None or area < hi))


def area_census(rows):
    counts, groups = Counter(), defaultdict(set)
    for area, group in rows:
        band = area_bin(area)
        counts[band] += 1
        groups[band].add(group)
    all_groups = set().union(*groups.values())
    eligible = ("ge_32_lt_512", "ge_512_lt_1024")
    stock = ("ge_512_lt_1024", "ge_1024")
    return dict(total=sum(counts.values()), decoded_groups=len(all_groups),
                area_counts={name: counts[name] for name, _, _ in BINS},
                decoded_groups_by_area={name: len(groups[name]) for name, _, _ in BINS},
                research_area_eligible=sum(counts[name] for name in eligible),
                research_area_eligible_decoded_groups=len(set().union(*(groups[name] for name in eligible))),
                stock_area_admitted=sum(counts[name] for name in stock),
                stock_area_admitted_decoded_groups=len(set().union(*(groups[name] for name in stock))),
                stock_admitted_within_research_area=counts["ge_512_lt_1024"],
                stock_admitted_within_research_area_decoded_groups=len(groups["ge_512_lt_1024"]))


def native_records(source, blocks, roster, audits, original):
    records = {}
    for item, audit, edit in zip(roster, audits, original, strict=True):
        name = item["image_ref"]["path"]
        native, _, size = blocks[name]
        group = edit["rgb_sha256"]
        source.require(item["image_id"] == audit["image_id"] == edit["image_id"]
                       and len(native) == len(audit["candidates"][:100]) == 100
                       and len(group) == 64, "Original image, 100 native slots and decoded-source identity required")
        areas = []
        for ordinal, (prediction, detail) in enumerate(zip(native, audit["candidates"][:100], strict=True), 1):
            area = int(source.codec.area(prediction["segmentation"]))
            source.require(detail["native_rank"] == ordinal and prediction["score"] == detail["score"]
                           and prediction["segmentation"]["size"] == size and area == detail["mask_area"]
                           and 0 <= area <= size[0] * size[1], "Historical native mask-audit area, rank or score differs")
            areas.append(area)
        records[name] = dict(name=name, rgb=group, native=native, areas=areas, size=size)
    return records


def evaluate(args):
    start = perf_counter()
    source = load_source(args.factorial_source)
    protocol_sha = source.digest(args.protocol.read_bytes())
    source.require(args.protocol.name == PROTOCOL_NAME and protocol_sha == args.protocol_sha256 == PROTOCOL_SHA256
                   and not args.output.exists(), "Exact released protocol and fresh output required")
    comp, scorer, geom, prior, capture, roster, audits, original, blocks = source.historical(args)
    excluded = set(scorer.EXCLUDED)
    source.require(len(excluded) == 10 and source.digest(("\n".join(scorer.EXCLUDED) + "\n").encode("ascii")) == scorer.EXCLUSION_SHA,
                   "Exact prior near-overlap exclusion roster required")
    records_by_name = native_records(source, blocks, roster, audits, original)
    source.require(len(records_by_name) == 232 and excluded <= set(records_by_name), "Original official and exclusion image rosters required")
    subsets = {"official": set(records_by_name), "exclude_exact_fit_overlap": set(records_by_name) - excluded}
    native_by_view = {view: area_census((area, records_by_name[name]["rgb"])
                                        for name in names for area in records_by_name[name]["areas"])
                      for view, names in subsets.items()}
    source.require(all(native_by_view[view]["total"] == expected[0] * 100 for view, expected in VIEWS.items())
                   and native_by_view["official"]["research_area_eligible"] == 11209,
                   "Native candidate and [32,1024) census must reproduce")
    print(json.dumps(dict(phase="verified_native_areas_before_labels", native_counts=native_by_view)), flush=True)

    gt = json.loads(source.pin(args.author_labels, scorer.GT_SHA))
    source.require(len(gt["images"]) == 232 and len(gt["annotations"]) == 546
                   and {im["file_name"] for im in gt["images"]} == set(records_by_name),
                   "Exact historical author labels and image roster required")
    records = {im["id"]: records_by_name[im["file_name"]] for im in gt["images"]}
    source.require(len(records) == len(gt["images"])
                   and all(records[im["id"]]["size"] == [im["height"], im["width"]] for im in gt["images"]),
                   "Unique author image IDs and native mask canvas required")
    continuous = geom.load(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    modules = continuous.geometry_modules(args.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    source.require(len(truth) == 546 and not neutral, "Original floating author polygon support required")
    reference_area = {aid: float(value[0].area) for aid, value in truth.items()}
    reference_area_eligible = {aid for aid, area in reference_area.items() if 32 <= area < 1024}
    views, all_original = {}, None
    for view, names in subsets.items():
        ids = sorted(i for i, rec in records.items() if rec["name"] in names)
        matches, info = geom.evaluate_raw(gt, records, ids, prior["views"][view]["metrics_fraction"]["raw"])
        if all_original is None:
            all_original = matches
        else:
            kept = set(ids)
            source.require(matches == {hull: pair for hull, pair in all_original.items() if pair[0] in kept},
                           "Sensitivity view must preserve the same retained native .50 matches")
        fixed = area_census((records[i]["areas"][rank-1], records[i]["rgb"]) for i, rank in matches.values())
        reference_fixed = area_census((records[i]["areas"][rank-1], records[i]["rgb"])
                                      for aid, (i, rank) in matches.items() if aid in reference_area_eligible)
        old_geometry = prior["geometry"][view]
        observed = (len(ids), info["gt_instances"], fixed["total"], fixed["research_area_eligible"],
                    fixed["research_area_eligible_decoded_groups"])
        source.require(observed == VIEWS[view] and observed[2:] == (old_geometry["original_native_matches"],
                       old_geometry["eligible_pairs"], old_geometry["decoded_image_groups_with_pairs"]),
                       "Original and area-eligible native matches or source groups differ from prior score")
        views[view] = dict(images=len(ids), annotations=info["gt_instances"], original_candidates=native_by_view[view],
                           native_fixed_matches_iou50=fixed,
                           native_areas_on_reference_polygon_area_eligible_matches_iou50=reference_fixed,
                           historical_raw_mask_AP_points=info["raw_mask_AP_points"],
                           historical_raw_AR100_points=info["raw_AR100_points"])
        print(json.dumps(dict(phase="counted_view", view=view, images=len(ids), original_matches=fixed["total"])), flush=True)
    return dict(status=STATUS, phase="score", worker_sha256=source.digest(Path(__file__).read_bytes()),
                protocol_sha256=protocol_sha, historical_source_sha256=SOURCE_SHA256, historical_score_sha256=source.PHASES["score"][0],
                historical_score_terminal_sha256=source.PHASES["score"][1], native_readback_sha256=comp.READBACK_SHA,
                manifest_sha256=capture["manifest_sha256"], native_predictions_sha256=capture["input_predictions_sha256"],
                native_audit_sha256=capture["candidate_audit_sha256"], author_labels_sha256=scorer.GT_SHA,
                historical_raw_COCO_and_fixed_match_parity=True, native_areas_verified_before_labels=True,
                stock_min_original_mask_area_pixels=512, research_area_half_open=[32, 1024],
                research_area_eligibility_basis="original native mask area, not the author polygon area",
                reference_polygon_area_census_basis="original IoU-.50 associations whose floating author polygon area is in [32,1024); the five bins still measure native mask area",
                native_score_rule="original top-100 ranked candidates in every official image",
                fixed_match_rule="original author COCO segmentation matching at .50 before the area split",
                decoded_group_rule="original decoded RGB content hash; groups can appear in multiple bins",
                stock_AP_computed=False, detector_forwards=0, segrefiner_forwards=0, optimizer_updates=0,
                primary_view="official", sensitivity_view="exclude_exact_fit_overlap", views=views,
                score_seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("factorial-source", "code", "owner", "capture", "classical-root", "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--original-readback", type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    result = evaluate(args)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("Incomplete safe score")
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
