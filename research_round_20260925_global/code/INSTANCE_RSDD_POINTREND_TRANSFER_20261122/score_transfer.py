"""Score the preregistered five-arm frozen PointRend transfer comparison."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


PROTOCOL = "b4bfdeab8d78c0b7b6c68e3615c3f2c217bb568237749bf43508ea4bf74fa4ad"
PRELOCK_COMMIT = "11da49604bcc8a7bb871329821754297c159905b"
PRIOR_SOURCE = "7039d7d0336f6a0cc3b64bfbe4b0c446b38b073cc90764f446a4dc6d09f90d85"
READBACK_STATUS = "INSTANCE_RSDD_POINTREND_REPLAY_ACCEPTED_V1"
STATUS = "INSTANCE_RSDD_POINTREND_TRANSFER_SCORE_COMPLETE_V1"
ARMS = ("raw", "coarse_paste", "point_paste", "point_equal", "learned_common")
MATCHED = ARMS[3:]
PASTE = ARMS[1:3]
PAIRS = ((ARMS[4], ARMS[3]), (ARMS[2], ARMS[1]), (ARMS[2], ARMS[0]),
         (ARMS[4], ARMS[0]), (ARMS[4], ARMS[2]), (ARMS[3], ARMS[0]))
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, f"Exact digest required for {path.name}")
    return raw


def predecessor(code):
    path = code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py"
    pinned(path, PRIOR_SOURCE)
    spec = spec_from_file_location("prior_rsdd_score", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_changes(codec, mask_check, audit, stock_rows, history, changes):
    """Check first-generation masks and equal realized edits before opening annotations."""
    require(isinstance(changes, dict), "Missing edit dictionary")
    require(all(x.isdigit() and str(int(x)) == x and 1 <= int(x) <= len(stock_rows)
                for x in changes), "Invalid compact candidate index")
    new, totals = {}, Counter()
    for rank, (row, candidate) in enumerate(zip(stock_rows, audit["candidates"], strict=True), 1):
        edits = changes.get(str(rank), {})
        require(set(edits) <= set(ARMS[1:]), "Unknown policy")
        stock = row["segmentation"]
        area = int(codec.area(stock))
        x0, y0, x1, y1 = candidate["box_xyxy"]
        valid = 32 <= area < 1024 and x0 < x1 and y0 < y1
        totals["area_eligible"] += 32 <= area < 1024
        totals["valid"] += valid
        if not valid:
            require(not edits, "Any excluded candidate must be an original-mask fallback")
            continue
        historic = history.get(rank, {}).get("free_natural", {})
        expected_k = historic.get("swaps", 0)
        if expected_k:
            require(all(arm in edits for arm in MATCHED)
                    and edits["learned_common"]["rle"] == historic["rle"],
                    "Archived learned output must be preserved on common support")
        for arm, edit in edits.items():
            rle = mask_check(edit["rle"], 512, 512)
            after = int(codec.area(rle))
            inter = int(codec.area(codec.merge([stock, rle], intersect=True)))
            add, drop = after-inter, area-inter
            if arm in MATCHED:
                require(type(edit["swaps"]) is int and 0 <= expected_k <= 8
                        and add == drop == edit["swaps"] == expected_k, "Exact balanced matched count")
            else:
                require(type(edit["adds"]) is type(edit["drops"]) is int
                        and add == edit["adds"] and drop == edit["drops"], "Exact paste XOR")
            totals[arm + "_edited_candidates"] += (add + drop > 0)
            totals[arm + "_adds"] += add
            totals[arm + "_drops"] += drop
            totals[arm + "_area_changed_candidates"] += (add != drop)
            totals[arm + "_emptied_candidates"] += (after == 0)
            new.setdefault(rank, {})[arm] = dict(edit)
    return new, totals


def load_blocks(a, old, adapter, core):
    old.terminal(a.new_terminal, a.new_terminal_sha, a.new_receipt_sha, "verify-edits")
    receipt = json.loads(pinned(a.new_receipt, a.new_receipt_sha))
    require(receipt["status"] == READBACK_STATUS and receipt["protocol_sha256"] == PROTOCOL
            and receipt["GT_opened"] is False and receipt["exact_replayed_policy_choices"] is True
            and receipt["exact_support_xor_area"] is True and receipt["images"] == 740
            and receipt["fit_weight_sha256"] == a.fit_weight_sha,
            "Independent closed label-free readback must match the frozen PointRend fit")
    cap = json.loads(pinned(a.cap_receipt, a.cap_receipt_sha))
    require(receipt["predictions_sha256"] == cap["predictions_sha256"]
            and receipt["candidate_audit_sha256"] == cap["audit_sha256"],
            "Identical frozen candidate confidence and identity")
    blocks = old.admitted_inputs(a, adapter, core)
    new = old.records(a.new_edits, receipt["edits_sha256"])
    audit = old.records(a.audit, cap["audit_sha256"])
    require(len(new) == len(audit) == len(blocks) == 740, "Exact author test population")
    stats = Counter()
    for (i, block), edit, item in zip(blocks.items(), new, audit, strict=True):
        require(i == edit["image_id"] == item["image_id"], "Original author order")
        block["changes"], counts = output_changes(
            core.codec, core.checked_rle, item, block["stock"], block["changes"], edit["changes"])
        stats.update(counts)
    require(stats["area_eligible"] == 43252
            and stats["point_equal_adds"] == stats["learned_common_adds"]
            and stats["point_equal_edited_candidates"] == stats["learned_common_edited_candidates"],
            "Whole-cohort eligibility and matched-count parity")
    return blocks, dict(stats)


def fixed_geometry(core, shapely, associations, polygons, blocks):
    groups, counts = defaultdict(list), Counter()
    exposed = set()
    marks = {str(t): {arm: Counter(gained=0, lost=0) for arm in ARMS} for t in (.5, .75, .9)}
    for gid, (image_id, rank, stock) in associations.items():
        polygon = polygons[gid]
        if not polygon.is_valid:
            counts["invalid_matched_continuous_excluded"] += 1
            continue
        original = stock["segmentation"]
        if not 32 <= core.codec.area(original) < 1024:
            counts["out_of_area_matched"] += 1
            continue
        block = blocks[image_id]
        edits = block["changes"].get(rank, {})
        if edits.get("learned_common", {}).get("swaps", 0) > 0:
            counts["positive_anchors"] += 1
            exposed.add(block["group"])
        values, cache = [], {}
        for arm in ARMS:
            rle = edits.get(arm, {}).get("rle", original)
            key = rle["counts"]
            if key not in cache:
                area = int(core.codec.area(rle))
                iou = core.geometric_iou(rle, polygon, shapely)
                inter = iou * (polygon.area + area) / (1 + iou)
                cache[key] = [100*iou, 100*(area-inter)/polygon.area, 100*(polygon.area-inter)/polygon.area]
            values.append(cache[key])
        groups[block["group"]].append(values)
        for t, per_arm in marks.items():
            before = values[0][0] >= 100*float(t)
            for arm, row in zip(ARMS, values, strict=True):
                after = row[0] >= 100*float(t)
                per_arm[arm]["gained"] += not before and after
                per_arm[arm]["lost"] += before and not after
    group_values = np.array([np.mean(groups[g], axis=0) for g in sorted(groups)], dtype=np.float64)
    chosen = np.random.Generator(np.random.PCG64(5217)).integers(len(groups), size=(4096, len(groups)))
    means = group_values.mean(axis=0)
    contrasts = {}
    for a, b in PAIRS:
        change = group_values[:, ARMS.index(a)] - group_values[:, ARMS.index(b)]
        intervals = np.percentile(change[chosen].mean(axis=1), [2.5, 97.5], axis=0)
        contrasts[a + " minus " + b] = {
            m: dict(delta=float(change[:, j].mean()), interval_95=intervals[:, j].tolist())
            for j, m in enumerate(FIELDS)
        }
    return dict(original_anchors=len(associations), eligible_anchors=sum(map(len, groups.values())),
                counts=dict(counts), groups=len(groups), positive_exposure_groups=len(exposed), transitions=marks,
                group_equal_arms={arm: dict(zip(FIELDS, means[j].tolist(), strict=True))
                                  for j, arm in enumerate(ARMS)}, paired_contrasts_points=contrasts)


def score_view(gt, keep, blocks, evaluator, core, polygons, shapely):
    subset = dict(gt, images=[r for r in gt["images"] if r["id"] in keep],
                  annotations=[r for r in gt["annotations"] if r["image_id"] in keep])
    scoped = {k: v for k, v in blocks.items() if k in keep}
    metrics = {}
    for arm in ARMS:
        metrics[arm], matched = evaluator.score_one(subset, scoped, arm, anchors=arm == "raw")
        if arm == "raw":
            associations = matched
    deltas = {a + " minus " + b: {k: 100*(v-metrics[b][k]) if min(v, metrics[b][k]) >= 0 else None
                                   for k, v in metrics[a].items()} for a, b in PAIRS}
    return dict(images=len(keep), annotations=len(subset["annotations"]),
                groups=len({row["group"] for row in scoped.values()}), metrics_fraction=metrics,
                paired_contrasts_points=deltas,
                geometry=fixed_geometry(core, shapely, associations, polygons, scoped))


def run(a):
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
            "Qualified GEOS geometry runtime")
    old = predecessor(a.code)
    evaluator = old.module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks, edit_stats = load_blocks(a, old, adapter, core)
    # Only the scoring stage may read the original target geometry.
    gt = {"info": {}, **json.loads(pinned(a.author_test, old.AUTHOR_TEST))}
    require(len(gt["images"]) == 740 and len(gt["annotations"]) == 872
            and {r["category_id"] for r in gt["annotations"]} == {1}
            and {r["id"] for r in gt["images"]} == set(blocks)
            and all(r["width"] == r["height"] == 512 and r["file_name"] == blocks[r["id"]]["file"]
                    for r in gt["images"]) and all(len(r["segmentation"]) == 1 for r in gt["annotations"]),
            "Author release and exact image identity")
    gt, rekeyed = old.coco_annotation_ids(gt)
    polygons = {r["id"]: shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1, 2))
                for r in gt["annotations"]}
    require(sum(not p.is_valid for p in polygons.values()) == 1, "Retain one non-simple polygon in COCO")
    full = set(blocks)
    gf3 = {i for i, r in blocks.items() if r["sensor"] == "GF-3"}
    subsets = dict(gf3=gf3, full=full, terrasar_x=full-gf3,
                   gf3_deduplicated=old.representatives(blocks, gf3),
                   full_deduplicated=old.representatives(blocks, full))
    results, cache = {}, {}
    for name, selection in subsets.items():
        key = frozenset(selection)
        if key not in cache:
            cache[key] = score_view(gt, selection, blocks, evaluator, core, polygons, shapely)
        results[name] = cache[key]
    primary = "learned_common minus point_equal"
    main = results["gf3"]
    improved = main["paired_contrasts_points"][primary]["AP"] > 0
    lower = main["geometry"]["paired_contrasts_points"][primary][FIELDS[0]]["interval_95"][0]
    return dict(status=STATUS, protocol_sha256=PROTOCOL, prelock_commit=PRELOCK_COMMIT,
                author_test_sha256=old.AUTHOR_TEST, evaluator_annotation_ids_rekeyed=rekeyed,
                detector_forwards=0, optimizer_updates=0, non_simple_ap_retained=1,
                independent_readback_sha256=a.new_receipt_sha, fit_weight_sha256=a.fit_weight_sha,
                realized_edits=edit_stats, views=results,
                within_recipe_ranking_criterion=bool(improved and lower > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(pinned(args.authority, args.authority_sha))
    paths = {"protocol", "code", "author_test", "edit_terminal", "edit_receipt", "cap_terminal", "cap_receipt",
             "predictions", "audit", "edits", "shapely_site", "new_terminal", "new_receipt", "new_edits"}
    hashes = {"edit_terminal_sha", "edit_receipt_sha", "cap_terminal_sha", "cap_receipt_sha",
              "new_terminal_sha", "new_receipt_sha", "fit_weight_sha"}
    require(set(values) == paths | hashes and not args.output.exists() and sys.flags.isolated
            and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11", "Qualified fresh CPU score")
    inputs = SimpleNamespace(**{k: Path(v) if k in paths else v for k, v in values.items()})
    pinned(inputs.protocol, PROTOCOL)
    sys.path.append(str(inputs.shapely_site))
    started = perf_counter()
    result = run(inputs)
    result.update(worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  authority_sha256=args.authority_sha, elapsed_seconds=perf_counter()-started)
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    args.output.mkdir()
    (args.output/"RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
