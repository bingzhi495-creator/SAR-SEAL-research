"""Replay exact local cardinality-k Phoenix matching on two frozen PSeg views."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


PRIOR_WORKER = "b939a9f85c23867913a297ab9cfcc39d9243b545e52867fca57b88b1b8b6a114"
PRIOR_RESULT = "e877a2ae9f074944f555efb5c3c4296778871e5e5aaa5e219b983378b71381bc"
PROTOCOL_NAME = "PHOENIX_BOUNDED_MATCH_PRELOCK_20261123.md"
PROTOCOL_SHA256 = "176ff6bd2bfb99f7474a4ffb89e2c634d9e955ee7af7d3b8a921d2485e7019f9"
INDEPENDENT = "phoenix_balanced"
BOUNDED = "phoenix_bounded_match"
FIXED = "phoenix_fixed_pair"
STATUS = "PSEG232_PHOENIX_BOUNDED_MATCH_SCORE_COMPLETE"


def load(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source digest mismatch for {path.name}")
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def selected_rows(mask, coords):
    return np.flatnonzero(mask.flat[coords[:, 0]]), np.flatnonzero(~mask.flat[coords[:, 1]])


def extend_candidates(prior, source, matching, request, tape, blocks, roster):
    counts, detail, ledger = Counter(), Counter(), sha256()
    for image, posterior, item in zip(request["images"], tape, roster, strict=True):
        source.require(image["image_id"] == posterior["image_id"] == item["image_id"], "Mismatched image streams")
        _, edits, hw = blocks[item["image_ref"]["path"]]
        for payload, score in zip(image["candidates"], posterior["candidates"], strict=True):
            pos, k = payload["position"], payload["k"]
            source.require(pos == score["position"], "Mismatched candidate score stream")
            coords, q = np.asarray(payload["coords"], dtype=np.int64), np.asarray(score["probabilities"])
            stock = source.decoded(payload["segmentation"], hw)
            arm = edits[str(pos)]
            fixed_rle, independent_rle = arm[FIXED]["rle"], arm[INDEPENDENT]["rle"]
            fa, fd = selected_rows(source.decoded(fixed_rle, hw), coords)
            ia, idrop = selected_rows(source.decoded(independent_rle, hw), coords)
            source.require(len(fa) == len(fd) == len(ia) == len(idrop) == k and np.array_equal(fa, fd),
                           "Expected paired archived rows and two exactly matched signed quotas")
            ba, bd, candidate_stats = matching.bounded_endpoints(coords, q, k, hw[1], fa, fd, ia, idrop)
            ba, bd = np.asarray(ba), np.asarray(bd)
            source.require(len(ba) == len(bd) == k and ba.dtype.kind in "iu" and bd.dtype.kind in "iu"
                           and np.all((0 <= ba) & (ba < len(coords))) and np.all((0 <= bd) & (bd < len(coords)))
                           and np.unique(ba).size == np.unique(bd).size == k,
                           "Bounded match must select unique archived rows of each sign")
            rle = prior.writeback(source, stock, hw, coords[ba, 0], coords[bd, 1])
            arm[BOUNDED] = {"rle": rle}
            ledger.update((json.dumps([image["image_id"], pos, rle], separators=(",", ":"), sort_keys=True) + "\n").encode("ascii"))
            counts.update(candidates=1, additions=k, removals=k, endpoint_probabilities=q.size,
                          different_from_fixed=int(rle != fixed_rle), different_from_independent=int(rle != independent_rle))
            detail.update(candidate_stats)
    source.require(tuple(counts[k] for k in ("candidates", "additions", "removals", "endpoint_probabilities"))
                   == (11054, 73770, 73770, 665634), "Archived candidate, quota or probability counters differ")
    return dict(counts), dict(detail), ledger.hexdigest()


def assess_recovery(local_over_fixed, independent_over_local):
    margin = 0.15
    interval = independent_over_local["paired_geometry"]["continuous_iou_points"]["percentile95"]
    ap = independent_over_local["COCO_points"]["AP"]
    within_ap = abs(ap) <= margin
    within_geometry = -margin <= interval[0] and interval[1] <= margin
    return dict(margin_points=margin, local_over_fixed_supported=local_over_fixed["criteria"]["supported_advantage"],
                within_AP_margin=within_ap, entire_geometry_interval_within_margin=within_geometry,
                recovered=bool(local_over_fixed["criteria"]["supported_advantage"] and within_ap and within_geometry))


def evaluate(args):
    start = perf_counter()
    prior = load(args.pair_operator_source, PRIOR_WORKER)
    matching = load(Path(__file__).with_name("matching.py"), args.matching_sha256)
    adapter = prior.load_balanced(args.balanced_source)
    source = adapter.load_bound(args.factorial_source)
    protocol_sha = source.digest(args.protocol.read_bytes())
    source.require(args.protocol.name == PROTOCOL_NAME and args.protocol_sha256 == protocol_sha == PROTOCOL_SHA256
                   and not args.output.exists() and shapely.__version__ == "2.0.2"
                   and shapely.geos_version_string == "3.11.2", "Expected the preregistered protocol and original geometry runtime")
    result = json.loads(source.pin(args.prior_score, PRIOR_RESULT))
    source.require(result["status"] == prior.STATUS and result["worker_sha256"] == PRIOR_WORKER
                   and result["protocol_sha256"] == prior.PROTOCOL_SHA256 and result["historical_COCO_and_geometry_parity"]
                   and result["author_labels_open_after_verified_writebacks"], "Previous published score must be qualified")
    parent, parent_sha, parent_terminal = source.observed(
        args.parent_score, args.parent_terminal, "score", source.SCORE_STATUS, adapter.SOURCE,
        (adapter.PARENT, adapter.PARENT_EXIT))
    old, old_sha, old_terminal = source.observed(
        args.balanced_score, args.balanced_terminal, "score", adapter.STATUS, prior.BALANCED_WORKER,
        (prior.BALANCED_RESULT, prior.BALANCED_TERMINAL))
    source.require(old["parent_score_sha256"] == result["parent_score_sha256"] == parent_sha
                   and old["parent_terminal_sha256"] == result["parent_terminal_sha256"] == parent_terminal
                   and old["protocol_sha256"] == adapter.LOCK and result["balanced_score_sha256"] == old_sha
                   and result["balanced_terminal_sha256"] == old_terminal, "Published reports must refer to the same archived runs")
    history = source.historical(args)
    comp, scorer, geom, reference, _, roster, audits, original, blocks = history
    _, lineage, inference = source.build_inputs(args, history)
    source.require(all(parent[key] == result[key] == value for key, value in lineage.items())
                   and inference["threshold_projection_exact"] and parent["protocol_sha256"] == source.LOCK,
                   "Closed probability, request or detector lineage differs from published baselines")
    request = json.loads(source.pin(args.request / "REQUEST_PRIVATE.json", lineage["request_sha256"]))
    tape = source.pin(args.inference / "NARROW_POSTERIORS_PRIVATE.jsonl", lineage["narrow_posteriors_sha256"])
    source.require(adapter.balanced(source, request, map(json.loads, tape.splitlines()), blocks, roster) == old["counts"],
                   "Historical independent mask counts differ")
    prior_counts, prior_ledger = prior.construct_pairs(source, request, map(json.loads, tape.splitlines()), blocks, roster)
    source.require(prior_counts == result["counts"] and prior_ledger == result["paired_rle_ledger_sha256"]
                   and prior_counts["canonical_assignment_matches_archive"] == 11054, "Historical fixed writer is not identical")
    counts, detail, ledger = extend_candidates(prior, source, matching, request, map(json.loads, tape.splitlines()), blocks, roster)
    print(json.dumps(dict(phase="verified_writebacks_before_labels", bounded_rle_ledger_sha256=ledger, **counts)), flush=True)

    gt = json.loads(source.pin(args.author_labels, scorer.GT_SHA))
    source.require(len(gt["images"]) == 232 and len(gt["annotations"]) == 546
                   and source.digest(("\n".join(scorer.EXCLUDED) + "\n").encode("ascii")) == scorer.EXCLUSION_SHA,
                   "Original author polygons and sensitivity roster required")
    scorer.ARMS += (INDEPENDENT, FIXED, BOUNDED)
    scored = scorer.score_populations(gt, blocks, scorer.EXCLUDED)
    base = geom.load(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", geom.OLD)
    continuous = geom.load(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    modules = continuous.geometry_modules(args.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    source.require(len(truth) == 546 and not neutral, "Continuous reference support differs")
    ids = {image["file_name"]: image["id"] for image in gt["images"]}
    records = {}
    for row, audit, edit in zip(roster, audits, original, strict=True):
        name = row["image_ref"]["path"]
        stock, changes, _ = blocks[name]
        records[ids[name]] = dict(name=name, rgb=edit["rgb_sha256"], native=stock,
                                  audit=audit["candidates"][:100], edit=changes)

    source.PAIRS = ((BOUNDED, FIXED), (INDEPENDENT, BOUNDED), (INDEPENDENT, FIXED))
    prior.PAIRED = BOUNDED
    ref_arms = ("raw", *comp.NEW_ARMS, INDEPENDENT, FIXED, BOUNDED)
    views, cache, anchors = {}, {}, None
    for view, scored_view in scored.items():
        expected = result["views"][view]
        metrics = scored_view["metrics_fraction"]
        source.require(all(scored_view[k] == expected[k] for k in ("images", "annotations", "candidates"))
                       and all(metrics[k] == val for k, val in expected["metrics_fraction"].items()), "Historical COCO vectors differ")
        eg = expected["geometry"]
        expected_geo = dict(eg, arms={**eg["arms"], BOUNDED: {}})
        subset = sorted(i for i, row in records.items() if view == "official" or row["name"] not in scorer.EXCLUDED)
        geometry, exposures, _, coverage, current = source.fixed_view(
            gt, subset, records, metrics, expected_geo, geom, base, modules, truth, cache, ref_arms, anchors)
        if anchors is None:
            anchors = current
        for key in ("eligible_pairs", "decoded_image_groups_with_pairs", "original_native_matches"):
            source.require(geometry[key] == eg[key], "Historical fixed cohort differs")
        source.require(all(geometry["arms"][arm] == val for arm, val in eg["arms"].items())
                       and geometry["pairwise"][INDEPENDENT + " minus " + FIXED] == eg["pairwise"][INDEPENDENT + " minus " + FIXED],
                       "Historical fixed-pair and independent geometry differs")
        coverage.update({a: expected["coverage"][a] for a in (INDEPENDENT, FIXED)})
        coverage[BOUNDED] = prior.new_coverage(source, geom, gt, subset, records, current, metrics)
        contrasts = {}
        for left, right in source.PAIRS:
            effect = source.difference(metrics[left], metrics[right])
            e = exposures[left + "_minus_" + right]
            g = geometry["pairwise"][left + " minus " + right]
            contrasts[left + "_minus_" + right] = dict(COCO_points=effect, exposure=e, paired_geometry=g,
                                                        criteria=prior.gate(effect["AP"], g, e))
        source.require(contrasts[INDEPENDENT + "_minus_" + FIXED] == expected["independent_minus_paired"],
                       "Historical exposure or paired scoring differs")
        recovery = assess_recovery(contrasts[BOUNDED + "_minus_" + FIXED], contrasts[INDEPENDENT + "_minus_" + BOUNDED])
        views[view] = dict(**scored_view, geometry=geometry, coverage=coverage, contrasts=contrasts, practical_recovery=recovery)
        print(json.dumps(dict(phase="scored_view", view=view, images=len(subset))), flush=True)
    return dict(status=STATUS, phase="score", worker_sha256=source.digest(Path(__file__).read_bytes()),
                protocol_sha256=protocol_sha, matching_sha256=args.matching_sha256,
                historical_worker_sha256=PRIOR_WORKER, prior_pair_score_sha256=PRIOR_RESULT,
                prior_pair_mask_ledger_sha256=prior_ledger, bounded_rle_ledger_sha256=ledger,
                parent_score_sha256=parent_sha, parent_terminal_sha256=parent_terminal,
                balanced_score_sha256=old_sha, balanced_terminal_sha256=old_terminal, **lineage,
                historical_COCO_and_geometry_parity=True, detector_forwards=0, phoenix_forwards=0,
                optimizer_updates=0, author_labels_open_after_verified_writebacks=True,
                bootstrap_draws=4096, bootstrap_seed=5217, counts=counts, matching_counts=detail,
                primary_view="official", sensitivity_view="exclude_exact_fit_overlap", views=views,
                score_seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("factorial-source", "balanced-source", "pair-operator-source", "prior-score", "code", "owner", "capture",
                 "classical-root", "request", "request-terminal", "inference", "inference-terminal", "author-labels",
                 "parent-score", "parent-terminal", "balanced-score", "balanced-terminal", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol-sha256", "matching-sha256"):
        parser.add_argument("--" + name, required=True)
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
