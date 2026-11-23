"""Score fixed-quota bounded Phoenix writes on the frozen Instance-RSDD reserve."""
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
import scipy
import shapely


LOCK_NAME = "PHOENIX_BOUNDED_MATCH_PRELOCK_20261123.md"
LOCK = "176ff6bd2bfb99f7474a4ffb89e2c634d9e955ee7af7d3b8a921d2485e7019f9"
PAIR_WORKER = "3cbdbf62076e52ab65425228302241c975a35a48d99151b76a8157794f7d5c06"
PAIR_RESULT = "e6725bba2d3848b2dbeb5857cd00ceaccd54b1d9b1b9f98ed0008784dfe3319a"
PAIR_RESULT_NAME = "RSDD74_PHOENIX_PAIR_OPERATOR_ACTUAL_SAFE_20261123.json"
STATUS = "RSDD74_PHOENIX_BOUNDED_MATCH_SCORE_COMPLETE"
CONTRASTS = (("balance", "bounded"), ("bounded", "paired"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def load_worker(path, expected_sha256=None):
    raw = path.read_bytes()
    stamp = digest(raw)
    require(expected_sha256 is None or stamp == expected_sha256, "Frozen source digest differs for " + path.name)
    worker = ModuleType(path.stem)
    worker.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), worker.__dict__)
    return worker, stamp


def add_bounded_masks(a, pair, matcher, api, blocks, chain, frozen_counts):
    request = json.loads(api.pin(a.request / "REQUEST_PRIVATE.json", chain["request_sha256"]))
    endpoints = api.pin(a.endpoints, chain["endpoint_sha256"])
    written = Counter()
    diagnostic = Counter()
    for image, scores in zip(request["images"], map(json.loads, endpoints.splitlines()), strict=True):
        image_id = image["image_id"]
        changes = blocks[image_id]["changes"]
        require(image_id == scores["image_id"] and len(image["candidates"]) == len(changes),
                "Frozen endpoint and candidate roster differ")
        written["images"] += 1
        for candidate, record in zip(image["candidates"], scores["candidates"], strict=True):
            position, k = candidate["position"], candidate["k"]
            edits = changes[position]
            require(position == record["position"] and "bounded" not in edits, "Frozen endpoint order differs")
            coords = np.asarray(candidate["coords"])
            probs = np.asarray(record["probabilities"], dtype=np.float64)
            native = codec.decode(candidate["segmentation"]).astype(bool)
            paired_add, paired_drop, ia, idrop, _ = pair.select_endpoints(coords, probs, k, native.shape[1])
            add, drop, stats = matcher.bounded_endpoints(coords, probs, k, native.shape[1],
                                                       paired_add, paired_drop, ia, idrop)
            add, drop = np.asarray(add), np.asarray(drop)
            require(add.shape == drop.shape == (k,) and add.dtype.kind in "iu" and drop.dtype.kind in "iu"
                    and np.all((0 <= add) & (add < len(coords))) and np.all((0 <= drop) & (drop < len(coords))),
                    "Bounded match must select the original k endpoint ordinals on both signs")
            bounded = pair.verified_mask(native, coords, add, drop)
            edits["bounded"] = {"rle": bounded}
            written.update(positive_budget_candidates=1, endpoint_pairs=len(coords), adds=k, drops=k,
                           differing_bounded_vs_paired=int(bounded != edits["paired"]["rle"]),
                           differing_balance_vs_bounded=int(edits["balance"]["rle"] != bounded))
            require(all(type(v) is int or isinstance(v, np.integer) for v in stats.values()),
                    "Bounded matcher diagnostics must be additive integers")
            diagnostic.update({key: int(value) for key, value in stats.items()})
        require(all("bounded" in edit for edit in changes.values()), "Incomplete bounded mask support")
    require(all(written[name] == frozen_counts[name]
                for name in ("images", "positive_budget_candidates", "endpoint_pairs", "adds", "drops")),
            "Bounded editor misses original candidates, signed pixels or endpoint census")
    return dict(written), dict(diagnostic)


def decision(view):
    residual = view["contrasts"]["balance_minus_bounded"]
    recovery = view["contrasts"]["bounded_minus_paired"]
    geometry = view["geometry"]["paired"].get("balance_minus_bounded")
    interval = geometry["continuous_iou_points"]["interval_95"] if geometry else None
    ap = residual["COCO_points"]["AP"]
    residual_ap_within_tolerance = ap is not None and abs(ap) <= .15
    residual_geometry_within_tolerance = bool(interval and -.15 <= interval[0] <= interval[1] <= .15)
    recovery_supported = recovery["criteria"]["supported_advantage"]
    return dict(
        practical_recovery=bool(recovery_supported and residual_ap_within_tolerance
                                and residual_geometry_within_tolerance),
        bounded_advantage_over_fixed=recovery_supported,
        independent_advantage_over_bounded=residual["criteria"]["supported_advantage"],
        residual_ap_within_practical_tolerance=residual_ap_within_tolerance,
        residual_geometry_within_practical_tolerance=residual_geometry_within_tolerance,
    )


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "stock-root", "point-root", "matched-root", "area-root", "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-score", type=Path)
    parser.add_argument("--matching-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key not in {"author_labels", "output"}))
    if a.prior_score is None:
        a.prior_score = a.code.parent / "results" / PAIR_RESULT_NAME
    for out, phase in (("phx_edits", "edits"), ("phx_score", "score"), ("request", "request"), ("inference", "inference")):
        setattr(a, out, a.matched_root / phase)
        setattr(a, out + "_terminal", a.matched_root / ("observed_" + phase) / "TERMINAL.json")
    a.endpoints = a.inference / "ENDPOINT_PROBABILITIES_PRIVATE.jsonl"
    a.writer = a.area_root / "writer"
    a.writer_terminal = a.area_root / "observed_edits" / "TERMINAL.json"
    a.cohort, a.support_root = 74, None
    return a


def main():
    a = arguments()
    require(a.protocol.name == LOCK_NAME and a.protocol_sha256 == digest(a.protocol.read_bytes()) == LOCK
            and not a.output.exists(), "Published bounded matching protocol and fresh output required")
    require((np.__version__, scipy.__version__, version("pycocotools"), shapely.__version__, shapely.geos_version_string)
            == ("1.26.4", "1.15.3", "2.0.11", "2.0.2", "3.11.2"), "Historical assignment and geometry runtime required")
    start = perf_counter()
    pair, _ = load_worker(a.code / "PHOENIX_PAIR_OPERATOR_20261123/score_rsdd.py", PAIR_WORKER)
    matcher, matching_sha256 = load_worker(Path(__file__).with_name("matching.py"), a.matching_sha256)
    area = pair.load_area(a.code)
    whole, sides, deps = area.sources(a)
    phx, api, _, history, reserve, old, _, _, _ = deps
    pair_score = json.loads(api.pin(a.prior_score, PAIR_RESULT))
    area_score, asha, aterm = api.observed(a.area_root / "score", a.area_root / "observed_score" / "TERMINAL.json",
                                          "score", area.STATUS, pair.AREA_RESULT, pair.AREA_TERMINAL, pair.AREA_WORKER)
    blocks, prior, chain, counts, _ = area.observed_inputs(a, sides, deps)
    require(all(area_score[k] == pair_score[k] == v for k, v in chain.items())
            and area_score["protocol_sha256"] == area.LOCK
            and area_score["historical_balanced_masks_and_scores_exact"] is True
            and pair_score["status"] == pair.STATUS and pair_score["worker_sha256"] == PAIR_WORKER
            and pair_score["protocol_sha256"] == pair.LOCK
            and pair_score["historical_area_score_sha256"] == asha
            and pair_score["historical_area_score_terminal_sha256"] == aterm
            and pair_score["exact_binary32_gains"] is pair_score["both_arms_verified_before_target_labels"] is True
            and pair_score["independent_masks_and_history_exact"] is True
            and all(control[name] == 0 for control in (area_score, pair_score)
                    for name in ("detector_forwards", "phoenix_forwards", "optimizer_updates")),
            "Closed area or pairing reference lineage differs")
    totals = pair.add_paired_masks(a, area, api, blocks, chain, counts)
    require(totals == pair_score["counts"], "Paired arm does not reproduce the published exact replay census")
    bounded, diagnostics = add_bounded_masks(a, pair, matcher, api, blocks, chain, totals)
    print(json.dumps(dict(phase="all_operator_masks_verified", images=totals["images"],
                          positive_budget_candidates=totals["positive_budget_candidates"],
                          target_labels_opened=False)), flush=True)

    # All three label-free arms have passed the original candidate and exact signed write checks.
    a.author_labels = a.author_labels.resolve(strict=True)
    gt, rekeyed, polygons, sets, dates, label_sha = history.views_and_labels(a, old, reserve, blocks)
    require(label_sha == area_score["author_labels_sha256"] == pair_score["author_labels_sha256"],
            "Historical label input differs")
    area.PAIRS = whole.PAIRS = CONTRASTS
    whole.ARMS = ("raw", "paired", "bounded", "balance")
    views = {}
    for name, (images, eligible, groups) in pair.VIEWS.items():
        view = area.score_view(sets[name], gt, blocks, polygons, dates, prior["views"][name], whole, deps)
        reference = pair_score["views"][name]
        phx.parity(view, reference, ("raw", "paired", "balance"))
        geometry = view["geometry"]
        require(view["images"] == images and geometry["counts"]["eligible_matches"] == eligible
                and geometry["sensor_date_blocks"] == groups and geometry["counts"] == reference["geometry"]["counts"]
                and all(view["coverage"][arm] == reference["coverage"][arm] for arm in ("raw", "paired", "balance")),
                "Published fixed-match and rematched coverage differs")
        view["bounded_match_decision"] = decision(view)
        views[name] = view
        print(json.dumps(dict(phase="scored_view", view=name, images=images)), flush=True)

    result = dict(
        status=STATUS, phase="score", protocol_sha256=a.protocol_sha256, worker_sha256=digest(Path(__file__).read_bytes()),
        matching_sha256=matching_sha256, **chain,
        historical_area_score_sha256=asha, historical_area_score_terminal_sha256=aterm,
        historical_pair_score_sha256=PAIR_RESULT, historical_pair_worker_sha256=PAIR_WORKER,
        counts=totals, bounded_counts=bounded, matching_diagnostics=diagnostics, arms=list(whole.ARMS),
        exact_binary32_gains=True, paired_and_independent_history_exact=True,
        all_arms_verified_before_target_labels=True, target_labels_opened=True, author_labels_sha256=label_sha,
        author_ids_rekeyed=rekeyed, detector_forwards=0, phoenix_forwards=0, optimizer_updates=0,
        bootstrap_draws=4096, bootstrap_seed=5217, cluster_unit="sensor/date; equal parent weight",
        primary_view="full", sensitivity_view="date_separated",
        primary_contrast="balance_minus_bounded", explanatory_contrast="bounded_minus_paired",
        contrasts=[f"{a}_minus_{b}" for a, b in CONTRASTS],
        views=views, score_seconds=perf_counter() - start,
    )
    a.output.mkdir()
    stamp = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=stamp)), flush=True)


if __name__ == "__main__":
    main()
