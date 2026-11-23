"""Replay paired and independent Phoenix writes on the frozen Instance-RSDD reserve."""
import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
from itertools import pairwise
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import scipy
from scipy.optimize import linear_sum_assignment
import shapely


LOCK_NAME = "PHOENIX_PAIR_OPERATOR_PRELOCK_20261123.md"
LOCK = "a688a92f88d47307a29760fd84aba57180c79bb8cb6dde21c0ecdd92b1825d92"
AREA_WORKER = "a7f2a920dd43955fb810d325cc79b518aaa63194f88e25d70777ab7d5180caf9"
AREA_RESULT = "5eade7925b66b56772c576cd37c92f08ba49ec8eac69458affb8031c78ceba82"
AREA_TERMINAL = "1581eb12702bd1bb0e79b5fee7fbe6fa33fe1fa644e9471a0289ecf84ed8f07e"
STATUS = "RSDD74_PHOENIX_PAIR_OPERATOR_SCORE_COMPLETE"
CONTRAST = "balance_minus_paired"
VIEWS = {"full": (74, 47, 11), "date_separated": (42, 32, 8)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def select_endpoints(coords, probabilities, k, width):
    require(coords.ndim == 2 and coords.shape == probabilities.shape and coords.shape[1] == 2
            and coords.dtype.kind in "iu" and np.all(coords >= 0)
            and np.unique(coords).size == coords.size and width > 0
            and type(k) is int and 0 < k <= min(8, len(coords)),
            "Invalid frozen endpoint support or quota")
    require(np.isfinite(probabilities).all() and np.all((0 <= probabilities) & (probabilities <= 1))
            and np.array_equal(probabilities, probabilities.astype(np.float32).astype(np.float64)),
            "Frozen Phoenix values must be exact binary32 posteriors")

    add_order = np.argsort(coords[:, 0], kind="stable")
    drop_order = np.argsort(coords[:, 1], kind="stable")
    ay, ax = np.divmod(coords[add_order, 0].astype(np.int64), width)
    dy, dx = np.divmod(coords[drop_order, 1].astype(np.int64), width)
    cost = (ay[:, None] - dy[None, :]) ** 2 + (ax[:, None] - dx[None, :]) ** 2
    rows, cols = linear_sum_assignment(cost)
    ordinals = np.arange(len(coords))
    require(np.array_equal(rows, ordinals) and np.array_equal(np.sort(cols), ordinals),
            "Minimum-distance assignment must cover both frozen signs exactly")

    add = [Fraction.from_float(float(p)) for p in probabilities[:, 0]]
    drop = [Fraction.from_float(float(p)) for p in probabilities[:, 1]]
    pair_add, pair_drop = add_order[rows], drop_order[cols]
    gains = [add[i] - drop[j] for i, j in zip(pair_add, pair_drop, strict=True)]
    ranked_pairs = sorted(range(len(gains)), key=lambda i: (-gains[i], int(coords[pair_add[i], 0])))
    ranked_adds = np.argsort(-probabilities[:, 0], kind="stable")
    ranked_drops = np.argsort(probabilities[:, 1], kind="stable")
    ia, idrop = ranked_adds[:k], ranked_drops[:k]
    selected = ranked_pairs[:k]
    paired_gain = sum(gains[i] for i in selected)
    independent_gain = sum(add[i] - drop[j] for i, j in zip(ia, idrop, strict=True))
    require(independent_gain >= paired_gain, "Independent selection cannot have smaller exact predicted gain")

    stats = dict(
        equal_pair_gains=sum(gains[i] == gains[j] for i, j in pairwise(ranked_pairs)),
        pair_gain_cutoff_ties=int(k < len(gains) and gains[ranked_pairs[k-1]] == gains[ranked_pairs[k]]),
        independent_add_cutoff_ties=int(k < len(gains) and add[ranked_adds[k-1]] == add[ranked_adds[k]]),
        independent_drop_cutoff_ties=int(k < len(gains) and drop[ranked_drops[k-1]] == drop[ranked_drops[k]]),
        selected_negative_gain_pairs=sum(gains[i] < 0 for i in selected),
        selected_zero_gain_pairs=sum(gains[i] == 0 for i in selected),
        equal_objective_candidates=int(independent_gain == paired_gain),
        higher_independent_objective_candidates=int(independent_gain > paired_gain),
    )
    return pair_add[selected], pair_drop[selected], ia, idrop, stats


def encode_mask(binary):
    mask = codec.encode(np.asfortranarray(binary, dtype=np.uint8))
    return {"size": mask["size"], "counts": mask["counts"].decode("ascii")}


def verified_mask(native, coords, add, drop):
    k = len(add)
    add_pixels, drop_pixels = coords[add, 0], coords[drop, 1]
    require(len(drop) == k and np.all((0 <= coords) & (coords < native.size))
            and not native.flat[coords[:, 0]].any() and native.flat[coords[:, 1]].all(),
            "Selected endpoint support or frozen mask signs differ")
    updated = native.copy(order="C")
    updated.flat[add_pixels] = True
    updated.flat[drop_pixels] = False
    require(np.array_equal(np.flatnonzero(updated & ~native), np.sort(add_pixels))
            and np.array_equal(np.flatnonzero(native & ~updated), np.sort(drop_pixels))
            and int(updated.sum()) == int(native.sum()),
            "Written mask lacks the exact frozen flips, signs or area")
    encoded = encode_mask(updated)
    require(np.array_equal(codec.decode(encoded), updated), "COCO mask round trip differs")
    return encoded


def load_area(code):
    path = code / "RSDD74_PHOENIX_AREA_ABLATION_20261123/score_ablation.py"
    raw = path.read_bytes()
    require(digest(raw) == AREA_WORKER, "Frozen area-ablation verifier differs")
    area = ModuleType("frozen_rsdd_area_ablation")
    area.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), area.__dict__)
    return area


def add_paired_masks(a, area, api, blocks, chain, historical_counts):
    request = json.loads(api.pin(a.request / "REQUEST_PRIVATE.json", chain["request_sha256"]))
    endpoints = api.pin(a.endpoints, chain["endpoint_sha256"])
    totals = Counter()
    for image, scored in zip(request["images"], map(json.loads, endpoints.splitlines()), strict=True):
        image_id = image["image_id"]
        changes = blocks[image_id]["changes"]
        require(image_id == scored["image_id"]
                and len(image["candidates"]) == len(changes), "Endpoint image or candidate roster differs")
        totals["images"] += 1
        for candidate, record in zip(image["candidates"], scored["candidates"], strict=True):
            position, k = candidate["position"], candidate["k"]
            require(position == record["position"] and "paired" not in changes[position],
                    "Endpoint order differs or repeats a frozen candidate")
            coords = np.asarray(candidate["coords"])
            probs = np.asarray(record["probabilities"], dtype=np.float64)
            native = codec.decode(candidate["segmentation"]).astype(bool)
            add, drop, ia, idrop, stats = select_endpoints(coords, probs, k, native.shape[1])
            archived_add, archived_drop = area.selected_ordinals(probs, k, "balance")
            require(ia.tolist() == archived_add and idrop.tolist() == archived_drop,
                    "Independent endpoint indices differ from the original accepted tie order")
            archived = changes[position]["balance"]["rle"]
            require(verified_mask(native, coords, ia, idrop) == archived,
                    "Replayed independent mask differs from original accepted RLE")
            paired = verified_mask(native, coords, add, drop)
            changes[position]["paired"] = {"rle": paired}
            totals.update(stats)
            totals.update(positive_budget_candidates=1, endpoint_pairs=len(coords), adds=k, drops=k,
                          differing_native_candidates=int(paired != archived))
        require(all("paired" in entry for entry in changes.values()), "Incomplete paired support")
    require(totals["images"] == historical_counts["images"] == 74
            and totals["positive_budget_candidates"] == historical_counts["positive_budget_candidates"] == 4107
            and totals["endpoint_pairs"] == historical_counts["endpoint_pairs"] == 129621
            and totals["adds"] == totals["drops"] == historical_counts["per_arm"]["balance"]["adds"] == 29900
            and totals["equal_objective_candidates"] + totals["higher_independent_objective_candidates"] == 4107,
            "Original candidate, probability or exact signed budget census differs")
    totals.update(native_candidates=historical_counts["native_candidates"],
                  zero_budget_candidates=historical_counts["zero_budget_candidates"],
                  endpoint_probabilities=2 * totals["endpoint_pairs"],
                  flips_per_arm=totals["adds"] + totals["drops"])
    return dict(totals)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "stock-root", "point-root", "matched-root", "area-root", "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key not in {"author_labels", "output"}))
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
            and not a.output.exists(), "Published pairing protocol and fresh output required")
    require((np.__version__, scipy.__version__, version("pycocotools"), shapely.__version__, shapely.geos_version_string)
            == ("1.26.4", "1.15.3", "2.0.11", "2.0.2", "3.11.2"), "Historical assignment and geometry runtime required")
    start = perf_counter()
    area = load_area(a.code)
    whole, sides, deps = area.sources(a)
    phx, api, _, history, reserve, old, _, _, _ = deps
    area_score, asha, aterm = api.observed(a.area_root / "score", a.area_root / "observed_score" / "TERMINAL.json",
                                          "score", area.STATUS, AREA_RESULT, AREA_TERMINAL, AREA_WORKER)
    blocks, prior, chain, counts, _ = area.observed_inputs(a, sides, deps)
    require(all(area_score[k] == v for k, v in chain.items())
            and area_score["protocol_sha256"] == area.LOCK
            and area_score["historical_balanced_masks_and_scores_exact"] is True
            and area_score["detector_forwards"] == area_score["phoenix_forwards"] == area_score["optimizer_updates"] == 0,
            "Closed area control score or original lineage differs")
    totals = add_paired_masks(a, area, api, blocks, chain, counts)
    print(json.dumps(dict(phase="all_operator_masks_verified", images=totals["images"],
                          positive_budget_candidates=totals["positive_budget_candidates"],
                          target_labels_opened=False)), flush=True)

    # Only these fixed masks and the closed original scoring chain may cross the label boundary.
    a.author_labels = a.author_labels.resolve(strict=True)
    gt, rekeyed, polygons, sets, dates, label_sha = history.views_and_labels(a, old, reserve, blocks)
    require(label_sha == area_score["author_labels_sha256"], "Historical label input differs")
    area.PAIRS = whole.PAIRS = (("balance", "paired"),)
    whole.ARMS = ("raw", "balance", "paired")
    views = {}
    for name, (images, eligible, groups) in VIEWS.items():
        view = area.score_view(sets[name], gt, blocks, polygons, dates, prior["views"][name], whole, deps)
        phx.parity(view, area_score["views"][name], ("raw", "balance"))
        geometry = view["geometry"]
        require(view["images"] == images and geometry["counts"]["eligible_matches"] == eligible
                and geometry["sensor_date_blocks"] == groups, "Published view denominators differ")
        views[name] = view
        print(json.dumps(dict(phase="scored_view", view=name, images=images)), flush=True)

    result = dict(
        status=STATUS, phase="score", protocol_sha256=LOCK, worker_sha256=digest(Path(__file__).read_bytes()),
        **chain, historical_area_score_sha256=asha, historical_area_score_terminal_sha256=aterm,
        counts=totals, arms=list(whole.ARMS), exact_binary32_gains=True, independent_masks_and_history_exact=True,
        both_arms_verified_before_target_labels=True, target_labels_opened=True, author_labels_sha256=label_sha,
        author_ids_rekeyed=rekeyed, detector_forwards=0, phoenix_forwards=0, optimizer_updates=0,
        assignment_distance="squared Euclidean", assignment_order="native row-major on both signs",
        paired_gain_tie_order="row-major addition", independent_probability_tie_order="admitted stored order",
        bootstrap_draws=4096, bootstrap_seed=5217, cluster_unit="sensor/date; equal parent weight",
        primary_view="full", sensitivity_view="date_separated", contrast=CONTRAST,
        views=views, score_seconds=perf_counter() - start,
    )
    a.output.mkdir()
    stamp = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=stamp)), flush=True)


if __name__ == "__main__":
    main()
