"""Replace midpoint and disagreement terms in a closed finite action menu."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.optimize import linear_sum_assignment


LOCK = "80e6f1a6e88edf4a3b938a91e84717659f5d9f3ae5f8a4c6a53d1cfe003bd4b3"
DIAGNOSE = "c44b34518377d3a932078f10fc37e9f1da683d0ef55418be3fbe7828000c0c1c"
CLOSED_SCORE = "8ec8db72912b72774fea99a420758c0ee78cb3c8b6b8750ee30a65616712a5d4"
ARMS = ("estimated_both", "true_midpoint", "true_disagreement", "true_both")
REFERENCE = ("pseg", "spsd")


def decision_indices(predicted, observed):
    """Use a common, static-first action order under each term substitution."""
    pred_m = predicted.mean(axis=-1)
    true_m = observed.mean(axis=-1)
    pred_h = np.abs(predicted[..., 0] - predicted[..., 1]) / 2
    true_h = np.abs(observed[..., 0] - observed[..., 1]) / 2
    return tuple(int(a.argmax()) for a in (
        predicted.min(axis=-1), true_m - pred_h, pred_m - true_h, observed.min(axis=-1)
    ))


def pin(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError("Digest mismatch: " + path.name)
    return data


def load_diagnose(path):
    previous = ModuleType("closed_diagnosis")
    previous.__file__ = str(path)
    exec(compile(pin(path, DIAGNOSE), str(path), "exec"), previous.__dict__)
    return previous


def measures(base, solver, family, position, references, denominator):
    coords = np.asarray(family["coords"], dtype=np.int64)
    labels = np.stack([codec.decode(target).astype(bool).ravel() for target in references])
    truth = base.family_gains(
        labels[:, coords[:, 0]].T, labels[:, coords[:, 1]].T, family["adds"], family["drops"]
    )
    pred = np.asarray(family["ap"])[:, None, :] - np.asarray(family["dp"])[None, :, :]
    pos = decision_indices(pred - pred[0, 0], truth - truth[0, 0])
    choice = np.unravel_index(pos[0], truth.shape[:2])
    base.need(choice == tuple(family["robust_pos"]), "Preserved deployed action identity")
    native = base.intersections(position["native"], references)
    static = base.intersections(position["changes"]["static"], references)
    rendered_robust = base.intersections(position["changes"]["dual_robust"], references)
    base.need(np.array_equal(static, native + truth[0, 0]) and
              np.array_equal(rendered_robust, native + truth[choice]),
              "Exact observed incumbent and deployed action intersections")
    total = labels.sum(axis=1) + int(codec.area(position["native"]))
    selected = native + truth.reshape(-1, 2)[list(pos)]
    effects = solver.iou(selected, total) - solver.iou(static, total)
    weighted = effects / denominator
    return np.column_stack((weighted, weighted.min(axis=1))), effects, pos


def report(a):
    start = perf_counter()
    pin(a.protocol, LOCK)
    prior = load_diagnose(a.prior / "diagnose.py")
    pin(a.group_protocol, prior.GROUP_PROTOCOL)
    term = json.loads((a.prior / "SCORE_TERMINAL.json").read_bytes())
    prior.need(term["phase"] == "score" and term["actual_exit"] == 0
               and term["output_footer_verified"] is True
               and term["result_sha256"] == CLOSED_SCORE and term["worker_sha256"] == DIAGNOSE,
               "Previous source development score is closed")
    previous = json.loads(pin(a.prior / "score/RESULT_SAFE.json", CLOSED_SCORE))
    old_family = prior.load(a.prior / "family.py", prior.FAMILY)
    solver = prior.load(a.oracle, prior.ORACLE)
    reader, _, _, _, native, records, groups = old_family.closed(a)
    family, receipt, family_score, _ = prior.verify_prelabel(a.prior)
    prior.need(family_score == previous["prelabel_result_sha256"] and
               receipt["family_sha256"] == previous["prelabel_family_sha256"]
               and set(native) == set(family) == set(groups), "Original label-free family closure")
    positive_count = 0
    for iid, candidates in native.items():
        options = family[iid]["families"]
        active = {str(i) for i, p in enumerate(candidates, 1) if p["quota"]}
        prior.need(family[iid]["group"] == groups[iid] and set(options) == active,
                   "Exact frozen group and positive candidate roster")
        prior.need(all(f["k"] == candidates[int(i) - 1]["quota"] for i, f in options.items()),
                   "Exact previously realized signed quotas")
        positive_count += len(active)
    prior.need(positive_count == receipt["counts"]["active_candidates"] == 8785,
               "Complete positive candidate set")
    print("Closed source gate and all finite menus admitted before reading references.", flush=True)
    truths = reader.labels(a, records)
    stock = {key: prior.stock_matches(truth, native) for key, truth in truths.items()}
    prior.need([len(stock[key]) for key in REFERENCE] == [311, 308], "Original native assignments")
    per_image = defaultdict(lambda: [[], []])
    for position, ref in enumerate(REFERENCE):
        for instance in truths[ref]["annotations"]:
            per_image[instance["image_id"]][position].append(instance)
    multiplicity = Counter(groups.values())
    ids = {key: i for i, key in enumerate(sorted(multiplicity))}
    values = np.zeros((len(ARMS), len(ids), 3))
    counts = {arm: Counter() for arm in ARMS}
    active_groups = set()
    shared = 0
    for iid, group in groups.items():
        first, second = per_image[iid]
        masks = [[i["segmentation"] for i in rows] for rows in (first, second)]
        overlap = codec.iou(*masks, [0] * len(second))
        rows, columns = linear_sum_assignment(overlap, maximize=True)
        denominator = np.array([len(first), len(second)]) * multiplicity[group]
        for row, col in zip(rows, columns, strict=True):
            if overlap[row, col] < .5:
                continue
            match = stock["pseg"].get(first[row]["id"])
            if match is None or match != stock["spsd"].get(second[col]["id"]):
                continue
            rank = match[1]
            candidate = native[iid][rank - 1]
            if not candidate["quota"]:
                continue
            active_groups.add(group)
            shared += 1
            weighted, effects, actions = measures(
                prior, solver, family[iid]["families"][str(rank)], candidate,
                (masks[0][row], masks[1][col]), denominator,
            )
            values[:, ids[group], :] += weighted
            for arm, effects_one, action in zip(ARMS, effects, actions, strict=True):
                counts[arm].update(
                    differs_from_deployed=int(action != actions[0]),
                    nonstatic=int(action != 0),
                    both_positive=int(effects_one.min() > prior.TOL),
                    negative_minimum=int(effects_one.min() < -prior.TOL),
                )
    prior.need((len(ids), shared, len(active_groups)) == (180, 206, 107), "Closed geographic scope")
    draws = np.random.Generator(np.random.PCG64(5297)).integers(180, size=(4096, 180))
    by_arm = {arm: prior.describe(v, draws) for arm, v in zip(ARMS, values, strict=True)}
    for label in prior.LABELS:
        for new, old in (("estimated_both", "deployed_robust"), ("true_both", "true_raw_family")):
            for stat in ("points", "percentile95_points"):
                np.testing.assert_allclose(by_arm[new][label][stat], previous["ladder"][old][label][stat],
                                           atol=1e-10, rtol=0)
    baseline, replace_m, replace_h, all_true = values
    mean_contribution = ((replace_m - baseline) + (all_true - replace_h)) / 2
    gap_contribution = ((replace_h - baseline) + (all_true - replace_m)) / 2
    np.testing.assert_allclose(mean_contribution + gap_contribution, all_true - baseline, atol=1e-12)
    contrasts = {
        "replace_midpoint_only": replace_m - baseline,
        "replace_disagreement_only": replace_h - baseline,
        "total_oracle_recovery": all_true - baseline,
        "symmetric_midpoint_contribution": mean_contribution,
        "symmetric_disagreement_contribution": gap_contribution,
        "paired_midpoint_minus_disagreement": mean_contribution - gap_contribution,
        "factorial_interaction": all_true - replace_m - replace_h + baseline,
    }
    status = "SPSD_PSEG_MIDPOINT_DISAGREEMENT_ATTRIBUTION_COMPLETE"
    output = dict(status=status, protocol_sha256=LOCK,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_diagnostic_sha256=CLOSED_SCORE, family_result_sha256=family_score,
                  original_score_parity=True, image_groups=180, image_records=183,
                  contributing_groups=107, shared_positive_quota_hulls=206,
                  author_denominators=[411, 410], bootstrap_draws=4096, scored_ap=False,
                  detector_forwards=0, optimizer_updates=0, fit_labels_opened=False,
                  historical_test_opened=False, label_assisted_retrospective=True,
                  arms=by_arm, selection_counts=counts,
                  contrasts={k: prior.describe(v, draws) for k, v in contrasts.items()},
                  elapsed_worker_seconds=perf_counter() - start)
    payload = (json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    a.output.mkdir()
    with (a.output / "RESULT_SAFE.json").open("xb") as result:
        result.write(payload)
    print(json.dumps(dict(status=status, result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for flag in ("prior", "gate", "labels", "protocol", "group-protocol", "oracle", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    report(parser.parse_args())
