"""Score matched-count relocation of observed SegRefiner deletions on Instance-RSDD."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import shapely


LOCK_NAME = "SEGREFINER_DEPTH_PRELOCK_20261123.md"
LOCK = "2c7a106bd8c70de20902a76b11d98ae0136b8708441cd150cf1829600d2163af"
SIGN_SOURCE = "cdd7b3f7e1bbbeb528fb5100da76daee54b2a4f61f88e2381ef2b709e48a080e"
SIGN_RESULT = "02f2a09fd50d4cc3112ba43cc9b3ed76badff6720f386fb45143665d5dfb7fc5"
ARMS = ("raw", "stock_lr", "periphery", "relocation")
PAIRS = (("periphery", "stock_lr"), ("periphery", "relocation"))
STATUS = "RSDD74_SEGREFINER_DEPTH_SCORE_COMPLETE"
VIEWS = ("full", "date_separated")


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned_module(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Frozen source identity differs for " + path.name)
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), obj.__dict__)
    return obj


def verify_locations(native, stock, periphery, relocation):
    require(native.shape == stock.shape and native.dtype == stock.dtype == np.bool_
            and all(isinstance(mask, np.ndarray) and mask.shape == native.shape and mask.dtype == np.bool_
                    for mask in (periphery, relocation)), "All alternatives must remain boolean native masks")
    original_adds = stock & ~native
    original_drops = native & ~stock
    periphery_drops = native & ~periphery
    relocation_drops = native & ~relocation
    d = int(original_drops.sum())
    require(np.array_equal(periphery & ~native, original_adds) and np.array_equal(relocation & ~native, original_adds)
            and int(periphery_drops.sum()) == int(relocation_drops.sum()) == d,
            "Matched relocation changed the observed addition set or deletion count")
    common = original_drops & periphery_drops
    require(np.array_equal(common, original_drops & relocation_drops),
            "Relocation comparator must retain exactly the same stock deletions")
    return dict(
        added_pixels=int(original_adds.sum()), deleted_pixels=d, stock_deletions_preserved=int(common.sum()),
        relocated_deletions=d-int(common.sum()), changed_periphery_vs_stock=int(not np.array_equal(periphery, stock)),
        changed_relocation_vs_stock=int(not np.array_equal(relocation, stock)),
        changed_periphery_vs_relocation=int(not np.array_equal(periphery, relocation)),
    )


def attach_locations(blocks, areas, gate, signed, depth_fn):
    totals, diagnostics = Counter(), Counter()
    for image, block in blocks.items():
        for position, det in zip(block["ranks"], block["stock"], strict=True):
            totals["native_candidates"] += 1
            options = block["changes"].get(position, {})
            if "stock_lr" not in options:
                require(areas[image][position] < gate, "Gate-positive native candidate cannot be bypassed")
                totals["native_bypasses"] += 1
                continue
            require(areas[image][position] >= gate, "Original gate excluded this observed stock output")
            native = codec.decode(det["segmentation"]).astype(bool)
            stock = codec.decode(options["stock_lr"]["rle"]).astype(bool)
            totals["gate_candidates"] += 1
            periphery, relocation, stats = depth_fn(native.copy(), stock.copy())
            verified = verify_locations(native, stock, periphery, relocation)
            totals.update(verified)
            witness = dict(native_pixels=int(native.sum()), stock_pixels=int(stock.sum()),
                           **{key: verified[key] for key in ("added_pixels", "deleted_pixels", "relocated_deletions")})
            require(stats == witness, "Depth helper diagnostics disagree with native mask audit")
            diagnostics.update(witness)
            for arm, mask in (("periphery", periphery), ("relocation", relocation)):
                encoded = signed.encode_mask(mask)
                require(np.array_equal(codec.decode(encoded), mask), "Counterfactual COCO mask round trip differs")
                options[arm] = {"rle": encoded}
    return dict(totals), dict(diagnostics)


def exposure(anchors, blocks, dates):
    counts = {name: Counter() for name in ("periphery_minus_stock_lr", "periphery_minus_relocation")}
    for image, position, det in anchors.values():
        block = blocks[image]
        options = block["changes"].get(position, {})
        raw = det["segmentation"]
        periphery = options.get("periphery", {"rle": raw})["rle"]
        for a, b in PAIRS:
            baseline = options.get(b, {"rle": raw})["rle"]
            if periphery != baseline:
                counts[a+"_minus_"+b][block["group"]] += 1
    return {name: dict(matches=sum(groups.values()), parents=len(groups),
                       sensor_date_blocks=len({dates[parent] for parent in groups}))
            for name, groups in counts.items()}


def score_view(ids, gt, blocks, areas, polygons, dates, reference, whole, deps, gate):
    _, _, cfa, _, reserve, _, evaluator, _, core = deps
    view = whole.score_view(ids, gt, blocks, polygons, dates, reference, deps)
    require(all(view["coverage"][arm] == reference["coverage"][arm] for arm in ("raw", "stock_lr")),
            "Native and observed stock rematched coverage differs")
    subset = {image: block for image, block in blocks.items() if image in ids}
    partial = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                   annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids])
    _, anchors = evaluator.score_one(partial, subset, "raw", anchors=True)
    valid = {gid: row for gid, row in anchors.items() if 32 <= areas[row[0]][row[1]] < 1024
             and polygons[gid].is_valid and polygons[gid].area > 0}
    eligible = {gid: row for gid, row in valid.items() if areas[row[0]][row[1]] >= gate}
    require(len(valid) == view["geometry"]["counts"]["eligible_matches"], "Fixed original anchor census differs")
    geometry = whole.fixed_geometry(eligible, subset, polygons, {ann["id"]: ann for ann in partial["annotations"]},
                                   dates, cfa, core, reserve)
    previous_geometry = reference["gate_exposed_geometry"]
    require(all(geometry[k] == previous_geometry[k] for k in ("counts", "represented_groups", "sensor_date_blocks"))
            and all(abs(value-previous_geometry["group_equal_arms"][arm][field]) < 1e-8
                    for arm in ("raw", "stock_lr") for field, value in geometry["group_equal_arms"][arm].items()),
            "Previously observed stock and native gate-exposed geometry differs")
    all_matches = exposure(anchors, subset, dates)
    all_fixed = exposure(valid, subset, dates)
    gate_fixed = exposure(eligible, subset, dates)
    for key, contrast in view["contrasts"].items():
        ap = contrast["COCO_points"]["AP"]
        criteria = {}
        for name, population, census in (("all_fixed", view["geometry"], all_fixed), ("gate_exposed", geometry, gate_fixed)):
            delta = population["paired"][key]["continuous_iou_points"]
            changed = census[key]
            criteria[name] = {**changed, **whole.decision(
                ap, delta["interval_95"], changed["matches"], changed["sensor_date_blocks"])}
        contrast["criteria"] = criteria
        contrast["changed_original_50_matches"] = all_matches[key]
    view["gate_exposed_geometry"] = geometry
    view["both_primary_supported"] = all(c["criteria"]["gate_exposed"]["supported_advantage"]
                                         for c in view["contrasts"].values())
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "run", "stock-root", "point-root", "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("score-terminal", "inference-terminal"):
        parser.add_argument("--" + name, type=Path)
    for name in ("protocol-sha256", "depth-sha256"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    for key, path in vars(a).items():
        if isinstance(path, Path):
            setattr(a, key, path.resolve(strict=key not in {"author_labels", "output"}))
    require(a.protocol.name == LOCK_NAME and a.protocol_sha256 == digest(a.protocol.read_bytes()) == LOCK
            and not a.output.exists(), "Published deletion-location prelock and fresh output required")
    require((np.__version__, version("pycocotools"), shapely.__version__, shapely.geos_version_string, version("scipy"))
            == ("1.26.4", "2.0.11", "2.0.2", "3.11.2", "1.15.3"), "Frozen numerical runtime required")
    a.score_terminal = a.score_terminal or a.run / "observed_score" / "TERMINAL.json"
    a.inference_terminal = a.inference_terminal or a.run / "observed_inference" / "TERMINAL.json"
    a.cohort, a.support_root = 74, None
    timer = perf_counter()
    depth = pinned_module(Path(__file__).with_name("depth_priority.py"), a.depth_sha256)
    signed = pinned_module(a.code / "SEGREFINER_SIGN_20261123/score_rsdd.py", SIGN_SOURCE)
    previous = signed.pinned_module(a.code / "SEGREFINER_STOCK_20261123/score_rsdd.py", signed.STOCK_SOURCE)
    exposure_previous = previous.exposure_source()
    whole, deps = exposure_previous.sources(a)
    _, api, cfa, history, reserve, old, _, adapter, core = deps
    sign_report = json.loads(api.pin(a.code.parent / "results" / "RSDD74_SEGREFINER_SIGN_SCORE_SAFE_20261123.json", SIGN_RESULT))
    report, score_sha, score_tsha = api.observed(a.run / "score", a.score_terminal, "score", previous.STATUS,
                                                signed.STOCK_SCORE, worker=signed.STOCK_SOURCE)
    infer, infer_sha, infer_tsha = api.observed(a.run / "inference", a.inference_terminal, "inference",
                                               previous.INFER_STATUS, report["inference_result_sha256"],
                                               report["inference_terminal_sha256"], report["inference_worker_sha256"])
    require(report["protocol_sha256"] == previous.LOCK and report["ledgers_verified_before_labels"] is True
            and report["original_coco_and_local_geometry_exact"] is True and infer["target_labels_opened"] is False
            and infer["scientific_replay"] is True and infer["exact_original_bypass"] is True
            and infer["candidate_ordinals_preserved"] is True and infer["refined_masks_sha256"] == report["refined_masks_sha256"]
            and sign_report["stock_score_sha256"] == score_sha and sign_report["inference_result_sha256"] == infer_sha
            and sign_report["inference_terminal_sha256"] == infer_tsha,
            "Exact closed historical stock mask and comparison qualification required")
    loader = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = loader.load_population(a)
    require(lineage == report["input_lineage"] == infer["input_lineage"], "Original frozen candidate lineage differs")
    blocks, areas, positions = exposure_previous.native_blocks(rows, adapter, core)
    stream = api.pin(a.run / "inference" / previous.LEDGER, report["refined_masks_sha256"])
    stock_counts = previous.attach_verified_masks(rows, blocks, areas, positions, stream, infer["counts"])
    require(stock_counts == report["counts"], "Original observed stock candidate or mask census differs")
    counts, diagnostics = attach_locations(blocks, areas, previous.GATE, signed, depth.deletion_location_masks)
    require(all(counts[key] == sign_report["counts"][key] for key in
                ("native_candidates", "gate_candidates", "native_bypasses", "added_pixels", "deleted_pixels"))
            and counts["relocated_deletions"] + counts["stock_deletions_preserved"] == counts["deleted_pixels"],
            "Matched-count mask census must recover the exact original signed action totals")
    print(json.dumps(dict(phase="matched_locations_verified", images=len(blocks),
                          gate_candidates=counts["gate_candidates"], target_labels_opened=False)), flush=True)

    a.author_labels = a.author_labels.resolve(strict=True)
    gt, rekeyed, polygons, sets, dates, label_sha = history.views_and_labels(a, old, reserve, blocks)
    require(label_sha == report["author_labels_sha256"] == sign_report["author_labels_sha256"],
            "Original author label identity differs")
    whole.ARMS, whole.MATCHED, whole.PAIRS = ARMS, ("raw", "stock_lr"), PAIRS
    views = {}
    for name, size in (("full", 74), ("date_separated", 42)):
        require(len(sets[name]) == size, "Historical original view differs")
        views[name] = score_view(sets[name], gt, blocks, areas, polygons, dates, report["views"][name], whole, deps, previous.GATE)
        print(json.dumps(dict(phase="scored_view", view=name, images=size)), flush=True)

    result = dict(
        status=STATUS, phase="score", worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=a.protocol_sha256,
        depth_worker_sha256=a.depth_sha256, stock_score_sha256=score_sha, stock_score_terminal_sha256=score_tsha,
        stock_signed_score_sha256=SIGN_RESULT, inference_result_sha256=infer_sha, inference_terminal_sha256=infer_tsha,
        refined_masks_sha256=infer["refined_masks_sha256"], input_lineage=lineage, author_labels_sha256=label_sha,
        author_ids_rekeyed=rekeyed, depth_diagnostics=diagnostics, counts=counts, arms=list(ARMS),
        exact_original_addition_sets_and_deletion_counts=True, same_stock_deletion_subset_in_both_controls=True,
        masks_verified_before_labels=True, target_labels_opened=True, native_and_stock_metrics_exact=True,
        detector_forwards=0, segrefiner_forwards=0, optimizer_updates=0, primary_view="full",
        sensitivity_view="date_separated", primary_geometry_scope="gate_exposed",
        contrasts=[f"{a}_minus_{b}" for a, b in PAIRS], bootstrap_draws=4096, bootstrap_seed=5217,
        cluster_unit="sensor/date; equal parent weight", views=views, score_seconds=perf_counter()-timer,
    )
    a.output.mkdir()
    stamp = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=stamp)), flush=True)


if __name__ == "__main__":
    main()
