"""Replay FIT109 and compare learned occupancy with spatial attraction."""

import argparse
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


LEARNED = "learned_at_attraction_count"
DYNAMIC = "attraction_dynamic"
STATIC = "attraction_static_at_dynamic_count"
NEW_ARMS = (LEARNED, DYNAMIC, STATIC)
MODES = ("raw", "fixed", "free_natural", *NEW_ARMS, "free_matched_count")
PILOT_SHA = "d5f62c0996b67ee6b3becbea12ab8bc52db75d88bc28d6d91622b7fc6c11dbe9"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_bytes(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, f"Unexpected source bytes: {path}")
    return raw


def import_checked(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(checked_bytes(path, expected), str(path), "exec"), module.__dict__)
    return module


def observe(slots, m):
    slot_by_feature = {id(s["data"]["features"]): i for i, s in enumerate(slots) if s["data"] is not None}
    observed = set()
    extra = {arm: [[] for _ in slots] for arm in NEW_ARMS}
    budgets = Counter()
    prior_probability = m.learner.probabilities

    def add_readouts(model, features, paired, center, scale):
        probabilities = prior_probability(model, features, paired, center, scale)
        i = slot_by_feature[id(features)]
        need(i not in observed and paired is features, "One original held-out score per supported candidate")
        observed.add(i)
        row = slots[i]
        cap = min(8, row["area"] // 10, len(probabilities))
        coordinates = row["data"]["coordinates"]
        stock = codec.decode(row["row"]["segmentation"])
        dynamic, static = m.topology.select(stock, coordinates, cap)
        count = len(dynamic)
        learned = m.free.choose_free(probabilities, cap, fixed_count=count)
        need(len(learned) == len(static) == count <= cap, "Per-candidate matched count")
        for arm, selected in zip(NEW_ARMS, (learned, dynamic, static), strict=True):
            for side in (0, 1):
                need(len({pair[side] for pair in selected}) == count, "Distinct edit endpoints")
            extra[arm][i] = selected
        natural_count = len(m.free.choose_free(probabilities, cap))
        budgets[(natural_count, count)] += 1
        return probabilities

    return prior_probability, add_readouts, observed, slot_by_feature, extra, budgets


@contextmanager
def verify_materialization(base):
    original_apply = base.apply
    counts = Counter()

    def audited_apply(original, action):
        encoded = original_apply(original, action)
        old = codec.decode(original).ravel()
        new = codec.decode(encoded).ravel()
        add, drop = map(np.asarray, action)
        need(not old[add].any() and old[drop].all(), "Correct original edit signs")
        need(new[add].all() and not new[drop].any(), "Correct final edit signs")
        need(np.count_nonzero(np.bitwise_xor(old, new)) == 2 * len(add), "Exact realized XOR count")
        counts.update(verified_masks=1, verified_changed_pixels=2 * len(add))
        return encoded

    base.apply = audited_apply
    try:
        yield counts
    finally:
        base.apply = original_apply


def summarize_geometry(pair, slots, actions, math):
    # The preserved evaluator boots paired effects from the arm at index three.
    result = pair.fixed_geometry(slots, actions, math)
    result["learned_at_attraction_count_minus"] = result.pop("free_natural_minus")
    dynamic_first = tuple(DYNAMIC if arm == LEARNED else LEARNED if arm == DYNAMIC else arm for arm in MODES)
    pair.MODES, pair.GEOMETRY = dynamic_first, (*dynamic_first, "oracle")
    try:
        dynamic = pair.fixed_geometry(slots, actions, math)
    finally:
        pair.MODES, pair.GEOMETRY = MODES, (*MODES, "oracle")
    result["dynamic_minus_static_at_dynamic_count"] = dynamic["free_natural_minus"][STATIC]
    return result


def run(args, m, pilot):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    historical = json.loads(pilot.pinned(args.historical, pilot.INPUTS["historical"][1]))
    prior = json.loads(pilot.pinned(args.previous, pilot.INPUTS["previous"][1]))
    _, blocks = m.base.read_core(args)
    receipt, planned = m.pair.sealed_points(args, m.base, m.math, blocks)
    need(receipt["points_manifest_sha256"] == pilot.INPUTS["points_manifest"][1], "Kept all original point sets")
    args.output.mkdir()
    before = m.base.seal(args.output / "PRELABEL_SAFE.json", m.base.encoded({
        "status": "FIT109_ATTRACTION_PRELABEL", "protocol_sha256": args.prelock_sha256,
        "ranking_source_sha256": args.helper_sha256, "replay_sha256": PILOT_SHA,
        "worker_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {k: v[1] for k, v in pilot.INPUTS.items()}, "point_counts": receipt["counts"],
        "ground_truth_opened": False,
    }))
    gt, annotations = m.pair.read_labels(args, m.base, blocks)
    slots, scope = m.pair.fixed_slots(args, m.base, blocks, planned, gt, annotations)
    prior_probability, capture, observed, feature_slots, extra, count_table = observe(slots, m)
    m.learner.probabilities = capture
    try:
        actions, training, usage = m.free.fit(slots, m.learner, m.math, prior)
    finally:
        m.learner.probabilities = prior_probability
    need(len(observed) == len(feature_slots), "Complete scope for all supported masks")
    count_table[(0, 0)] += len(slots) - len(observed)
    need(sum(count_table.values()) == 10900, "All original candidates represented")
    actions.update(extra)
    need(training == historical["training"] and scope == historical["match_scope"] and usage == historical["usage"],
         "Original five-fold training with same evidence population")
    with verify_materialization(m.base) as audited:
        coco, budgets = m.gray.materialize(slots, actions, m.base, m.official, gt)
    geometry = summarize_geometry(m.pair, slots, actions, m.math)
    transitions, witness = m.common.threshold(slots, actions)
    m.common.validate_transitions(transitions, geometry)
    need((geometry["pairs"], geometry["source_groups"]) == (412, 93), "Original geometry population")
    for arm in m.free.MODES:
        need(coco[arm] == historical["cohorts"][arm] and budgets[arm] == historical["budgets"][arm]
             and geometry["metrics"][arm] == historical["geometry"]["metrics"][arm]
             and transitions[arm] == historical["strict90_transitions"][arm], f"Original result parity: {arm}")
    need(budgets[LEARNED] == budgets[DYNAMIC] == budgets[STATIC], "Realized count and roles agree")
    witness_digest = m.base.seal(args.output / "STRICT_TRANSITIONS_PRIVATE.jsonl",
                                 b"".join(m.base.encoded(row) for row in witness))
    contrasts = {}
    for arm in (DYNAMIC, STATIC):
        contrasts[arm] = {
            "AP_points": coco[LEARNED]["AP"] - coco[arm]["AP"],
            "AP75_points": coco[LEARNED]["AP75"] - coco[arm]["AP75"],
            "continuous_iou": geometry["learned_at_attraction_count_minus"][arm],
        }
    dynamic_minus_static = {
        "AP_points": coco[DYNAMIC]["AP"] - coco[STATIC]["AP"],
        "AP75_points": coco[DYNAMIC]["AP75"] - coco[STATIC]["AP75"],
        "continuous_iou": geometry["dynamic_minus_static_at_dynamic_count"],
    }
    result = {
        "status": "FIT109_ATTRACTION_MATCHED_COUNT_ACTUAL",
        "FIT": True, "DEV": False, "TEST": False, "detector_forwards": 0, "detector_updates": 0,
        "prelabel_sha256": before, "transition_witness_sha256": witness_digest,
        "training": training, "point_optimizer_updates": 1000, "fixed_original_matches": scope,
        "historical_parity": True, "verified_xor": audited, "cohorts": coco, "budgets": budgets,
        "count_table": [{"learned_natural": a, "dynamic": d, "candidates": count}
                        for (a, d), count in sorted(count_table.items())],
        "geometry": geometry, "strict90_inclusive": True, "strict90_transitions": transitions,
        "learned_minus": contrasts, "dynamic_minus_static": dynamic_minus_static,
    }
    digest = m.base.seal(args.output / "RESULT_SAFE.json", m.base.encoded(result))
    print(json.dumps({"status": result["status"], "result_sha256": digest}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run"))
    for key in ("pilot", "helper", "prelock", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("helper-sha256", "prelock-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only environment")
    need(not args.output.exists(), "Fresh output directory")
    checked_bytes(args.prelock, args.prelock_sha256)
    pilot = import_checked(args.pilot, PILOT_SHA)
    for package, version in pilot.VERSIONS.items():
        need(importlib.metadata.version(package) == version, f"Pinned version for {package}")
    args.private_root, args.code = pilot.PRIVATE, pilot.CODE
    modules = pilot.sources(args)
    modules.common.MODES = modules.gray.MODES = modules.pair.MODES = MODES
    modules.pair.GEOMETRY = (*MODES, "oracle")
    if args.mode == "run":
        run(args, modules, pilot)
    else:
        print(json.dumps({"status": "FIT109_ATTRACTION_IMPORTS_ACCEPTED", "closed_inputs_opened": 0}))


if __name__ == "__main__":
    main()
