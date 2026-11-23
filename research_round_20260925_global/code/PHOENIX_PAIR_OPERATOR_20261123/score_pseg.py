"""Compare fixed geometric pairing and independent writes under a single archived Phoenix probability field."""
import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from scipy.optimize import linear_sum_assignment
import shapely


BALANCED_WORKER = "a74d9944e523c280efa848431633ec6b933006f51ceecec6ec9784543d3bd5c1"
BALANCED_RESULT = "7c8b7b32c1d49f8e15db2167cb940fdae9bf5aefb84fd76f29846fd0b9226962"
BALANCED_TERMINAL = "96747233686ac27960e01bda44ce4d19fdb084904eb1bffa98324d706001f110"
PROTOCOL_NAME = "PHOENIX_PAIR_OPERATOR_PRELOCK_20261123.md"
PROTOCOL_SHA256 = "a688a92f88d47307a29760fd84aba57180c79bb8cb6dde21c0ecdd92b1825d92"
INDEPENDENT = "phoenix_balanced"
PAIRED = "phoenix_fixed_pair"
STATUS = "PSEG232_PHOENIX_FIXED_VS_INDEPENDENT_SCORE_COMPLETE"


def load_balanced(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != BALANCED_WORKER:
        raise ValueError("The prior closed balanced worker is required")
    adapter = ModuleType(path.stem)
    adapter.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), adapter.__dict__)
    return adapter


def fixed_pairs(coords, width):
    """Rebuild the one-to-one squared-distance assignment from row-major sides."""
    add, drop = np.sort(coords[:, 0]), np.sort(coords[:, 1])
    ay, ax = np.divmod(add, width)
    dy, dx = np.divmod(drop, width)
    cost = (ay[:, None] - dy[None, :]) ** 2 + (ax[:, None] - dx[None, :]) ** 2
    i, j = linear_sum_assignment(cost)
    return np.column_stack((add[i], drop[j]))


def cutoff_tied(order, scores, k):
    return k < len(order) and scores[order[k-1]] == scores[order[k]]


def select_dyadic(probabilities, canonical_rows, k):
    """Use exact archived binary32 values for arithmetic and original row ordinal for ties."""
    exact = [[Fraction.from_float(float(q)) for q in column] for column in probabilities.T]
    gains = [exact[0][a] - exact[1][d] for a, d in canonical_rows]
    paired = sorted(range(len(gains)), key=lambda j: (-gains[j], j))
    add = sorted(range(len(gains)), key=lambda j: (-exact[0][j], j))
    drop = sorted(range(len(gains)), key=lambda j: (exact[1][j], j))
    ties = dict(candidates_with_paired_gain_tie=int(len(set(gains)) < len(gains)),
                candidates_with_paired_cutoff_tie=int(cutoff_tied(paired, gains, k)),
                candidates_with_add_cutoff_tie=int(cutoff_tied(add, exact[0], k)),
                candidates_with_drop_cutoff_tie=int(cutoff_tied(drop, exact[1], k)))
    paired, add, drop = paired[:k], add[:k], drop[:k]
    advantage = sum(exact[0][j] for j in add) - sum(exact[1][j] for j in drop) - sum(gains[j] for j in paired)
    return paired, add, drop, advantage, ties


def writeback(source, stock, hw, additions, removals):
    out = stock.copy(order="C")
    out.flat[additions], out.flat[removals] = True, False
    rle = source.encode(out)
    check = source.decoded(rle, hw)
    source.require(np.array_equal(np.flatnonzero(check & ~stock), np.sort(additions))
                   and np.array_equal(np.flatnonzero(stock & ~check), np.sort(removals))
                   and check.sum() == stock.sum(), "Serialized writeback differs from frozen signed edits")
    return rle


def construct_pairs(source, request, posteriors, blocks, roster):
    counts = Counter(candidates=0, additions=0, removals=0, different_candidates=0,
                     exact_posterior_advantage_positive=0, exact_posterior_advantage_zero=0)
    ledger = sha256()
    for image, tape, item in zip(request["images"], posteriors, roster, strict=True):
        source.require(image["image_id"] == tape["image_id"] == item["image_id"], "Matched image streams required")
        _, edits, hw = blocks[item["image_ref"]["path"]]
        for payload, posterior in zip(image["candidates"], tape["candidates"], strict=True):
            pos, k = payload["position"], payload["k"]
            source.require(pos == posterior["position"], "Captured sparse posterior is in a different slot")
            coords, q = np.asarray(payload["coords"], dtype=np.int64), np.asarray(posterior["probabilities"])
            canonical = fixed_pairs(coords, hw[1])
            add_rows = {coord: row for row, coord in enumerate(coords[:, 0])}
            drop_rows = {coord: row for row, coord in enumerate(coords[:, 1])}
            canonical_rows = [(add_rows[a], drop_rows[d]) for a, d in canonical]
            paired, add, drop, advantage, ties = select_dyadic(q, canonical_rows, k)
            source.require(advantage >= 0, "Independent edits failed the exact estimated-score ordering check")
            stock = source.decoded(payload["segmentation"], hw)
            old = edits[str(pos)][INDEPENDENT]["rle"]
            source.require(writeback(source, stock, hw, coords[add, 0], coords[drop, 1]) == old,
                           "Prior independent mask differs from exact separate-order replay")
            rle = writeback(source, stock, hw, canonical[paired, 0], canonical[paired, 1])
            edits[str(pos)][PAIRED] = {"rle": rle}
            ledger.update((json.dumps([image["image_id"], pos, rle], separators=(",", ":"), sort_keys=True) + "\n")
                          .encode("ascii"))
            counts.update(candidates=1, additions=k, removals=k, different_candidates=int(rle != old),
                          endpoint_probabilities=q.size, canonical_assignment_matches_archive=int(np.array_equal(canonical, coords)),
                          different_candidates_with_equal_gain=int(rle != old and advantage == 0),
                          exact_posterior_advantage_positive=int(advantage > 0),
                          exact_posterior_advantage_zero=int(advantage == 0), **ties)
    source.require(tuple(counts[k] for k in ("candidates", "additions", "removals", "endpoint_probabilities"))
                   == (11054, 73770, 73770, 665634), "Frozen signed quotas or candidate and probability census differs")
    return dict(counts), ledger.hexdigest()


def new_coverage(source, geom, gt, ids, records, anchors, metrics):
    annotations = {ann["id"]: ann for ann in gt["annotations"]}
    adjusted = {}
    for iid in ids:
        row = records[iid]
        adjusted[iid] = dict(native=[
            dict(detection, segmentation=row["edit"].get(str(rank), {}).get(PAIRED, {}).get("rle", detection["segmentation"]))
            for rank, detection in enumerate(row["native"], 1)])
    kept = 0
    for aid, (iid, rank) in anchors.items():
        predicted = adjusted[iid]["native"][rank-1]["segmentation"]
        ref = source.codec.merge(source.codec.frPyObjects(annotations[aid]["segmentation"], *predicted["size"]))
        kept += int(source.codec.iou([predicted], [ref], [False])[0, 0] >= .50)
    matched, details = geom.evaluate_raw(gt, adjusted, ids, metrics[PAIRED])
    return dict(original_pairs_retained_50=kept, original_pairs=len(anchors),
                rematched_hulls_50=len(matched), ground_truth_hulls=details["gt_instances"])


def gate(ap, geometry, exposure):
    lower, upper = geometry["continuous_iou_points"]["percentile95"]
    exposed = exposure["different_hulls"] >= 20 and exposure["different_decoded_groups"] >= 8
    return dict(exposure_supported=exposed, absolute_AP_threshold=.30,
                supported_advantage=bool(exposed and ap >= .30 and lower > 0),
                supported_loss=bool(exposed and ap <= -.30 and upper < 0))


def evaluate(args):
    start = perf_counter()
    adapter = load_balanced(args.balanced_source)
    source = adapter.load_bound(args.factorial_source)
    lock = source.digest(args.protocol.read_bytes())
    source.require(args.protocol.name == PROTOCOL_NAME and args.protocol_sha256 == lock == PROTOCOL_SHA256
                   and not args.output.exists() and shapely.__version__ == "2.0.2"
                   and shapely.geos_version_string == "3.11.2", "Expected the exact prelock and verified geometry runtime")
    parent, parent_sha, parent_terminal = source.observed(
        args.parent_score, args.parent_terminal, "score", source.SCORE_STATUS, adapter.SOURCE, (adapter.PARENT, adapter.PARENT_EXIT))
    old, old_sha, old_terminal = source.observed(
        args.balanced_score, args.balanced_terminal, "score", adapter.STATUS, BALANCED_WORKER,
        (BALANCED_RESULT, BALANCED_TERMINAL))
    source.require(old["parent_score_sha256"] == parent_sha and old["parent_terminal_sha256"] == parent_terminal
                   and old["protocol_sha256"] == adapter.LOCK, "Closed balanced score must bind the same factorial parent")
    history = source.historical(args)
    comp, scorer, geom, reference, _, roster, audits, original, blocks = history
    _, lineage, inference = source.build_inputs(args, history)
    source.require(all(parent[key] == value for key, value in lineage.items())
                   and inference["threshold_projection_exact"] and parent["protocol_sha256"] == source.LOCK,
                   "Closed sparse posterior, candidate archive or factorial source differs")
    request = json.loads(source.pin(args.request / "REQUEST_PRIVATE.json", lineage["request_sha256"]))
    tape = source.pin(args.inference / "NARROW_POSTERIORS_PRIVATE.jsonl", lineage["narrow_posteriors_sha256"])
    source.require(adapter.balanced(source, request, map(json.loads, tape.splitlines()), blocks, roster) == old["counts"],
                   "Prior independent replay counts differ")
    counts, masks_sha = construct_pairs(source, request, map(json.loads, tape.splitlines()), blocks, roster)
    print(json.dumps(dict(phase="verified_writebacks_before_labels", paired_rle_ledger_sha256=masks_sha, **counts)), flush=True)

    gt = json.loads(source.pin(args.author_labels, scorer.GT_SHA))
    source.require(len(gt["images"]) == 232 and len(gt["annotations"]) == 546
                   and source.digest(("\n".join(scorer.EXCLUDED) + "\n").encode("ascii")) == scorer.EXCLUSION_SHA,
                   "Original author polygons and exclusion roster required")
    scorer.ARMS += (INDEPENDENT, PAIRED)
    scored = scorer.score_populations(gt, blocks, scorer.EXCLUDED)
    base = geom.load(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", geom.OLD)
    continuous = geom.load(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    modules = continuous.geometry_modules(args.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    source.require(len(truth) == 546 and not neutral, "Unchanged continuous author reference support required")
    ids = {image["file_name"]: image["id"] for image in gt["images"]}
    records = {}
    for row, audit, edit in zip(roster, audits, original, strict=True):
        name = row["image_ref"]["path"]
        stock, changes, _ = blocks[name]
        records[ids[name]] = dict(name=name, rgb=edit["rgb_sha256"], native=stock,
                                  audit=audit["candidates"][:100], edit=changes)

    source.PAIRS = ((INDEPENDENT, PAIRED),)
    reference_arms = ("raw", *comp.NEW_ARMS, INDEPENDENT, PAIRED)
    views, cache, anchors = {}, {}, None
    for view, result in scored.items():
        expected = old["views"][view]
        metrics = result["metrics_fraction"]
        source.require(all(result[k] == expected[k] for k in ("images", "annotations", "candidates"))
                       and all(metrics[k] == values for k, values in expected["metrics_fraction"].items()),
                       "Previous balanced and factorial COCO vectors differ")
        earlier = reference["geometry"][view]
        expected_geo = dict(earlier, arms={**earlier["arms"], INDEPENDENT: expected["geometry"]["arms"][INDEPENDENT], PAIRED: {}})
        subset = sorted(i for i, row in records.items() if view == "official" or row["name"] not in scorer.EXCLUDED)
        geometry, exposures, _, coverage, current = source.fixed_view(
            gt, subset, records, metrics, expected_geo, geom, base, modules, truth, cache, reference_arms, anchors)
        if anchors is None:
            anchors = current
        keys = ("eligible_pairs", "decoded_image_groups_with_pairs", "original_native_matches")
        source.require(all(geometry[k] == expected["geometry"][k] for k in keys)
                       and all(geometry["arms"][arm] == data for arm, data in expected["geometry"]["arms"].items()),
                       "Previous fixed-match population or continuous score differs")
        coverage[INDEPENDENT] = expected["coverage"][INDEPENDENT]
        coverage[PAIRED] = new_coverage(source, geom, gt, subset, records, current, metrics)
        ap = source.difference(metrics[INDEPENDENT], metrics[PAIRED])
        pairwise = geometry["pairwise"][INDEPENDENT + " minus " + PAIRED]
        exposure = exposures[INDEPENDENT + "_minus_" + PAIRED]
        views[view] = dict(**result, geometry=geometry, coverage=coverage,
                           independent_minus_paired=dict(COCO_points=ap, exposure=exposure, paired_geometry=pairwise,
                                                         criteria=gate(ap["AP"], pairwise, exposure)))
        print(json.dumps(dict(phase="scored_view", view=view, images=len(subset))), flush=True)
    return dict(status=STATUS, phase="score", worker_sha256=source.digest(Path(__file__).read_bytes()), protocol_sha256=lock,
                parent_score_sha256=parent_sha, parent_terminal_sha256=parent_terminal,
                balanced_score_sha256=old_sha, balanced_terminal_sha256=old_terminal, **lineage,
                paired_rle_ledger_sha256=masks_sha, same_probability_field=True, exact_geometric_assignment=True,
                historical_COCO_and_geometry_parity=True, detector_forwards=0, phoenix_forwards=0, optimizer_updates=0,
                author_labels_open_after_verified_writebacks=True, bootstrap_draws=4096, bootstrap_seed=5217,
                counts=counts, views=views, score_seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("factorial-source", "balanced-source", "code", "owner", "capture", "classical-root",
                 "request", "request-terminal", "inference", "inference-terminal", "author-labels",
                 "parent-score", "parent-terminal", "balanced-score", "balanced-terminal", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    args.original_readback = args.masks = args.posteriors = None
    result = evaluate(args)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("Incomplete safe score")
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
