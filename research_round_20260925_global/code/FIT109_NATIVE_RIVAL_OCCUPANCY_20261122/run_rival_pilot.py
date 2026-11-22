"""FIT109 source-only replay of frozen native rival evidence."""

import argparse
from collections import defaultdict
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np


INDEPENDENT = "free_matched_count"
TRUE = "rival_true"
PLACEBO = "rival_q_rank_permutation"
MODES = ("raw", "fixed", "free_natural", TRUE, PLACEBO, INDEPENDENT)
CLASSICAL_SHA = "acb7e00e7dfd2c60e5afc4397e073a6fabcb6a4ed977f1c8eaf89115724b52ed"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def module(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Unexpected ordinary source " + path.name)
    out = ModuleType(path.stem)
    out.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), out.__dict__)
    return out


def require_match_identity(fixed_slots):
    keywords = (
        keys for keys in fixed_slots.__code__.co_consts
        if isinstance(keys, tuple)
    )
    need(any({"area", "role", "truth", "raw", "gt_id"}.issubset(keys) for keys in keywords),
         "Original identity-attaching fixed-slot source was compiled")


def sealed_penalties(rival, blocks, planned):
    """Use only the original accepted, pre-label masks and point coordinates."""
    penalties = {}
    digest = sha256()
    for block, plan in zip(blocks, planned, strict=True):
        rows = block["coco"][:100]
        need(len(rows) == 100 and all(row["image_id"] == block["image_id"] for row in rows),
             "Original native image and rank population")
        scene = rival.NativeRivals(rows)
        for rank, data in sorted(plan["data"].items()):
            values = scene.penalties(rank - 1, data["coordinates"][:, 0])
            key = (block["image_id"], rank)
            need(key not in penalties, "Unique mask candidate")
            penalties[key] = values
            digest.update(np.array([*key, len(values)], dtype="<i8").tobytes())
            digest.update(values.astype("<f8", copy=False).tobytes())
    return penalties, digest.hexdigest()


def frozen_edits(m, slots, penalties, previous):
    feature_slots = {id(s["data"]["features"]): i for i, s in enumerate(slots) if s["data"] is not None}
    previous_probability = m.learner.probabilities
    metadata = {}
    extra = {arm: [[] for _ in slots] for arm in (TRUE, PLACEBO)}

    def capture(model, features, paired, center, scale):
        p = previous_probability(model, features, paired, center, scale)
        i = feature_slots[id(features)]
        need(i not in metadata and paired is features, "One original held-out probability vector")
        s = slots[i]
        key = (s["row"]["image_id"], i % 100 + 1)
        evidence = penalties[key]
        cap = min(8, s["area"] // 10, len(p))
        count = len(m.math.choose_pairs(p[:, 0]-p[:, 1], cap))
        normal = m.free.choose_free(p, cap, fixed_count=count)
        placebo = m.topology.placebo_penalties(p[:, 0], evidence, *key)
        need(np.array_equal(np.sort(evidence), np.sort(placebo)), "Placebo matches exact penalty multiset")
        drop = [b for _, b in normal]
        for arm, values in ((TRUE, evidence), (PLACEBO, placebo)):
            selected = m.topology.rank_additions(p, values, cap, count, m.free.choose_free)
            need(len(selected) == count and len({a for a, _ in selected}) == count
                 and [b for _, b in selected] == drop, "Exact balanced count and unchanged deletion ordinals")
            extra[arm][i] = selected
        nominal, measured, shuffled = ({a for a, _ in arm} for arm in (normal, extra[TRUE][i], extra[PLACEBO][i]))
        metadata[i] = dict(native_rival=bool(np.any(evidence)), count=count,
                           true_changed=measured != nominal, placebo_changed=shuffled != nominal,
                           placebo_displaced=not np.array_equal(placebo, evidence),
                           true_placebo_changed=measured != shuffled)
        return p

    m.learner.probabilities = capture
    try:
        actions, training, usage = m.free.fit(slots, m.learner, m.math, previous)
    finally:
        m.learner.probabilities = previous_probability
    need(len(metadata) == len(feature_slots) == len(penalties), "Complete original endpoint support")
    actions.update(extra)
    return actions, training, usage, metadata


def geometry_rows(a, m, gt, slots, actions, metadata):
    import shapely

    engine = m.base.load(a.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                         m.base.GEOM, "_rival_geom")
    geom = engine.geometry_modules(a.code)
    truth, neutral, _ = geom.geometry.polygon_support(gt, geom.vector)
    need(not set(truth).intersection(neutral), "Disjoint valid and neutral reference sets")
    by_image = defaultdict(dict)
    for annotation in gt["annotations"]:
        aid = annotation["id"]
        if aid in truth:
            by_image[annotation["image_id"]][aid] = truth[aid][0]
    rows = []
    for i, slot in enumerate(slots):
        if slot["raw"] is None:
            continue
        iid, aid = slot["row"]["image_id"], slot["gt_id"]
        need(slot["role"] == "valid" and aid in by_image[iid], "Original same-image fixed valid identity")
        own = by_image[iid][aid]
        others = [poly for ref, poly in by_image[iid].items() if ref != aid]
        if min((own.distance(poly) for poly in others), default=float("inf")) > 4.0:
            continue
        neighbor = shapely.union_all(others).difference(own)
        target_area, stock_intersection = slot["raw"]
        need(abs(own.area-target_area) <= 1e-8, "Original frozen matched polygon")
        values = {}
        for arm in (INDEPENDENT, TRUE, PLACEBO):
            selected = actions[arm][i]
            shift = intrude = 0.0
            if selected:
                q = slot["truth"]
                selection = np.asarray(selected, dtype=np.intp)
                shift = float((q[selection[:, 0], 0] - q[selection[:, 1], 1]).sum())
                coordinates = slot["data"]["coordinates"]
                cells = coordinates[selection[:, 0], 0]
                y, x = np.divmod(cells, slot["row"]["segmentation"]["size"][1])
                boxes = shapely.box(x, y, x + 1, y + 1)
                intrude = float(shapely.area(shapely.intersection(boxes, neighbor)).sum())
            intersection = float(np.clip(stock_intersection + shift, 0, min(slot["area"], target_area)))
            iou = 100 * intersection / (slot["area"] + target_area - intersection)
            values[arm] = (intrude, iou)
        status = metadata.get(i, {})
        rows.append(dict(group=slot["group"], exposed=status.get("native_rival", False)
                         and status.get("count", 0) > 0, true_changed=status.get("true_changed", False),
                         placebo_changed=status.get("placebo_changed", False),
                         placebo_displaced=status.get("placebo_displaced", False),
                         true_placebo_changed=status.get("true_placebo_changed", False), arms=values))
    return rows


def near_contrasts(rows, coco):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    exposed = [r for r in rows if r["exposed"]]
    exposed_groups = len({r["group"] for r in exposed})
    coverage = dict(matches=len(rows), prefix_groups=len(groups), exposed_matches=len(exposed),
                    exposed_prefix_groups=exposed_groups,
                    true_changed=sum(r["true_changed"] for r in rows),
                    placebo_changed=sum(r["placebo_changed"] for r in rows),
                    placebo_displaced=sum(r["placebo_displaced"] for r in rows),
                    true_placebo_changed=sum(r["true_placebo_changed"] for r in rows))
    inferential = len(exposed) >= 12 and exposed_groups >= 8
    results = {}
    for control in (INDEPENDENT, PLACEBO):
        effect = []
        for group in sorted(groups):
            pairs = []
            for row in groups[group]:
                c_other, c_own = row["arms"][control]
                t_other, t_own = row["arms"][TRUE]
                pairs.append((c_other-t_other, t_own-c_own))
            effect.append(np.mean(pairs, axis=0))
        average = np.mean(effect, axis=0).tolist() if effect else [None, None]
        interval = [None, None]
        if inferential:
            draws = np.random.Generator(np.random.PCG64(5217)).integers(len(effect), size=(4096, len(effect)))
            interval = np.percentile(np.asarray(effect)[draws].mean(axis=1), [2.5, 97.5], axis=0).T.tolist()
        ap_diff = coco[TRUE]["AP"] - coco[control]["AP"]
        passed = bool(inferential and interval[0][0] > 0 and interval[1][0] > -0.20 and ap_diff > -0.30)
        results[control] = dict(neighbor_intrusion_reduction_pixels=average[0], own_iou_gain_points=average[1],
                                neighbor_reduction_95_percent=interval[0], own_iou_95_percent=interval[1],
                                full_cohort_ap_gain_points=ap_diff, passes_all=passed)
    return dict(coverage=coverage, inference_enabled=inferential, contrasts=results,
                advances=all(result["passes_all"] for result in results.values()))


def run(a, m, prior, classical):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    historical = json.loads(prior.pinned(a.historical, prior.INPUTS["historical"][1]))
    previous = json.loads(prior.pinned(a.previous, prior.INPUTS["previous"][1]))
    _, blocks = m.base.read_core(a)
    receipt, planned = m.pair.sealed_points(a, m.base, m.math, blocks)
    need(receipt["points_manifest_sha256"] == prior.INPUTS["points_manifest"][1], "Kept all original point sets")
    penalties, digest = sealed_penalties(m.topology, blocks, planned)
    a.output.mkdir()
    seal = m.base.seal(a.output / "PRELABEL_SAFE.json", m.base.encoded({
        "status": "FIT109_RIVAL_PENALTY_PRELABEL",
        "ground_truth_opened": False, "penalty_digest": digest, "supported_candidates": len(penalties),
        "point_occupancy_units": sum(len(p) for p in penalties.values()),
        "nonzero_rival_points": sum(int(np.count_nonzero(p)) for p in penalties.values()),
        "input_sha256": {k: v[1] for k, v in prior.INPUTS.items()},
        "protocol_sha256": a.prelock_sha256, "rival_source_sha256": a.helper_sha256,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "classical_worker_sha256": CLASSICAL_SHA,
    }))
    gt, annotations = m.pair.read_labels(a, m.base, blocks)
    slots, scope = m.pair.fixed_slots(a, m.base, blocks, planned, gt, annotations)
    actions, training, usage, metadata = frozen_edits(m, slots, penalties, previous)
    need(training == historical["training"] and scope == historical["match_scope"] and usage == historical["usage"],
         "Original five-fold point head replay")
    with classical.verify_materialization(m.base) as verified:
        coco, budgets = m.gray.materialize(slots, actions, m.base, m.official, gt)
    geometry = m.pair.fixed_geometry(slots, actions, m.math)
    geometry["rival_true_minus"] = geometry.pop("free_natural_minus")
    transitions, _ = m.common.threshold(slots, actions)
    m.common.validate_transitions(transitions, geometry)
    need((geometry["pairs"], geometry["source_groups"]) == (412, 93), "Original fixed geometry population")
    for arm in m.free.MODES:
        need((coco[arm], budgets[arm], geometry["metrics"][arm], transitions[arm])
             == (historical["cohorts"][arm], historical["budgets"][arm], historical["geometry"]["metrics"][arm],
                 historical["strict90_transitions"][arm]), "Original metrics and AP parity " + arm)
    need(budgets[TRUE] == budgets[PLACEBO] == budgets[INDEPENDENT], "Equal true and control edit counts")
    near = near_contrasts(geometry_rows(a, m, gt, slots, actions, metadata), coco)
    result = {
        "status": "FIT109_NATIVE_RIVAL_OCCUPANCY_ACTUAL",
        "FIT": True, "DEV": False, "TEST": False, "detector_updates": 0, "detector_forwards": 0,
        "point_optimizer_updates": 1000, "historical_parity": True,
        "prelabel_sha256": seal, "penalty_digest": digest, "verified_xor": verified,
        "training": training, "cohorts": coco, "budgets": budgets, "geometry": geometry,
        "strict90_inclusive": True, "strict90_transitions": transitions,
        "all_native_rival_candidates": sum(row["native_rival"] for row in metadata.values()),
        "all_candidates_true_ranking_changed": sum(row["true_changed"] for row in metadata.values()),
        "all_candidates_placebo_ranking_changed": sum(row["placebo_changed"] for row in metadata.values()),
        "near": near,
    }
    output_sha = m.base.seal(a.output / "RESULT_SAFE.json", m.base.encoded(result))
    print(json.dumps({"status": result["status"], "result_sha256": output_sha, "advances": near["advances"]}),
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "run"))
    for key in ("pilot", "classical", "helper", "prelock", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("helper-sha256", "prelock-sha256"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not a.output.exists(), "Fresh CPU-only replay")
    classical = module(a.classical, CLASSICAL_SHA)
    classical.checked_bytes(a.prelock, a.prelock_sha256)
    prior = classical.import_checked(a.pilot, classical.PILOT_SHA)
    for package, version in prior.VERSIONS.items():
        need(importlib.metadata.version(package) == version, "Original package version " + package)
    a.private_root, a.code = prior.PRIVATE, prior.CODE
    m = prior.sources(a)
    require_match_identity(m.pair.fixed_slots)
    m.common.MODES = m.gray.MODES = m.pair.MODES = MODES
    m.pair.GEOMETRY = (*MODES, "oracle")
    if a.mode == "run":
        run(a, m, prior, classical)
    else:
        print(json.dumps({"status": "FIT109_NATIVE_RIVALS_IMPORTS_ACCEPTED", "closed_inputs_opened": 0,
                          "original_slot_identity_seam_compiled": True}))


if __name__ == "__main__":
    main()
