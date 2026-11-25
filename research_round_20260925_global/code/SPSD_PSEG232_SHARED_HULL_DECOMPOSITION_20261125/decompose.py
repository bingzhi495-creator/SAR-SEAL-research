"""Attribute an observed reference shift on complete, previously closed replay outputs."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
from scipy.optimize import linear_sum_assignment

STATUS = "SPSD_PSEG232_SHARED_HULL_DECOMPOSITION_COMPLETE"
PRELOCK = "408603442e7e28a7ce5241b95628ba10d3fbcfd5376844fe4a16c25a732a4c5b"
SCORER = "9baf865d510b3af95510ca55017943e5147a9c9cc68384665f14cb40b07434a9"
PRIOR_RESULT = "dd152bd587d64453b874a9a34e6531adacbd7bf163de4f9b22ffb25c4dea52fc"
ARMS = (
    ("free_natural", "raw"),
    ("free_matched_count", "fixed"),
    ("learned_at_attraction_count", "attraction_static_at_dynamic_count"),
)
TERMS = (
    "same_candidate_label_disagreement",
    "same_candidate_iou_normalization",
    "different_assignment_reference_shape",
    "assignment_present_vs_absent",
    "assignment_rank_switch",
    "paired_reference_count_weight",
    "low_overlap_or_unpaired_composition",
)
CLASSES = ("common_background", "pseg_only", "spsd_only", "common_foreground")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_checked(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact pinned input: " + path.name)
    return raw


def load_prior(args):
    observed = json.loads((args.prior_run / "observed_score_corrected/TERMINAL.json").read_bytes())
    require(observed["phase"] == "score" and observed["actual_exit"] == 0 and observed["ended_utc"]
            and observed["output_footer_verified"] is True and observed["result_sha256"] == PRIOR_RESULT
            and observed["worker_sha256"] == SCORER, "Original result naturally terminal and accepted")
    prior = json.loads(read_checked(args.prior_run / "RESULT_SAFE.json", PRIOR_RESULT))
    legacy_path = args.code / "SPSD_PSEG232_ANNOTATION_SENSITIVITY_20261125/score_reference.py"
    legacy = ModuleType("locked_spsd_reference")
    legacy.__file__ = str(legacy_path)
    exec(compile(read_checked(legacy_path, SCORER), str(legacy_path), "exec"), legacy.__dict__)
    require(prior["status"] == legacy.STATUS and prior["worker_sha256"] == SCORER
            and prior["protocol_sha256"] == legacy.LOCK and prior["erratum_sha256"] == legacy.ERRATUM,
            "Original prelock and input-admission amendment")
    return prior, legacy


def to_rle(annotation, canvas):
    source = annotation["segmentation"]
    return mask.merge(mask.frPyObjects(source, *canvas)) if isinstance(source, list) else source


def decode(rle):
    return mask.decode(rle).astype(bool)


def correspondences(pool):
    lhs, rhs = pool
    matrix = mask.iou([rle for _, rle in lhs], [rle for _, rle in rhs], [0] * len(rhs))
    return [(int(i), int(j), float(matrix[i, j]))
            for i, j in zip(*linear_sum_assignment(-matrix), strict=True)]


def changed_cells(earlier, later):
    require(earlier.shape == later.shape and earlier.dtype == later.dtype == np.bool_, "aligned native masks")
    add = later & ~earlier
    drop = earlier & ~later
    require(np.count_nonzero(add) == np.count_nonzero(drop), "Fixed-area treatment and control")
    return add, drop


def effect(reference, baseline, add, drop):
    pred_size = int(np.count_nonzero(baseline))
    true_size = int(np.count_nonzero(reference))
    overlap_before = int(np.count_nonzero(reference & baseline))
    gain = int(np.count_nonzero(reference & add)) - int(np.count_nonzero(reference & drop))
    total = pred_size + true_size
    require(true_size > 0 and total - overlap_before - gain > 0, "Nonempty author hull and unions")
    factor = 100.0 * total / ((total - overlap_before) * (total - overlap_before - gain))
    return factor * gain, factor, gain


def symmetric(fp0, fp1, fs0, fs1):
    return 0.5 * ((fs0 - fp0) + (fs1 - fp1)), 0.5 * ((fp1 - fp0) + (fs1 - fs0))


def same_rank(effect_p, effect_s, pair, add, drop):
    p, s = pair
    pp, ap, gp = effect_p
    ps, az, gs = effect_s
    labels = p.astype(np.uint8) + 2 * s.astype(np.uint8)
    plus = np.bincount(labels[add], minlength=4)
    minus = np.bincount(labels[drop], minlength=4)
    g = plus - minus
    require(g[2] - g[1] == gs - gp, "Exact signed foreground disagreement")
    edit_signal = 0.5 * (ap + az) * (gs - gp)
    normalization = 0.5 * (gp + gs) * (az - ap)
    require(abs((ps - pp) - edit_signal - normalization) <= 1e-10, "Same-rank IoU decomposition")
    return edit_signal, normalization, plus, minus


def fixed_ranks(legacy, scorer, geom, prior, authors, blocks):
    result = []
    for name, gt in zip(("PSeg_SSDD", "SPSD"), authors, strict=True):
        ids = sorted(row["id"] for row in gt["images"])
        images = {row["id"]: {"native": blocks[row["file_name"]][0]} for row in gt["images"]}
        native_fraction = scorer.evaluate(gt, legacy.rows_for(gt, blocks, "raw"))
        matches, _ = geom.evaluate_raw(gt, images, ids, native_fraction)
        require(len(matches) == prior["references"][name]["matched_hulls"], "Prior native match population")
        result.append(matches)
    return result


def membership(p_rank, s_rank):
    if p_rank is None and s_rank is None:
        return "both_missing"
    if p_rank is None or s_rank is None:
        return "one_missing"
    if p_rank == s_rank:
        return "same_rank"
    return "different_rank"


def calculate(authors, blocks, rgbs, ranks, prior_deltas, legacy, draws):
    images = sorted(authors[0]["images"], key=lambda row: rgbs[row["file_name"]])
    pools = [defaultdict(list), defaultdict(list)]
    canvases = {i["id"]: (i["height"], i["width"]) for i in images}
    for pool, gt in zip(pools, authors, strict=True):
        for ann in gt["annotations"]:
            pool[ann["image_id"]].append((ann["id"], to_rle(ann, canvases[ann["image_id"]])))

    results = {}
    for treatment, control in ARMS:
        key = treatment + "_minus_" + control
        parts = np.zeros((len(TERMS), len(images)))
        common_only = np.zeros((3, len(images)))
        eligibility = Counter()
        cell_counts = Counter()
        signs = Counter()
        changed = set()
        changed_strict = set()
        largest_error = 0.0

        for column, image in enumerate(images):
            iid = image["id"]
            native_rows, actions, _ = blocks[image["file_name"]]
            pool = [source[iid] for source in pools]
            refs = [[decode(rle) for _, rle in side] for side in pool]
            matches = [
                [per_ref.get(index, (None, None)) for index, _ in side]
                for per_ref, side in zip(ranks, pool, strict=True)
            ]
            for i, rank in [item for side in matches for item in side]:
                require((rank is None and i is None) or (i == iid and 1 <= rank <= len(native_rows)),
                        "Same-image native match and original candidate order")

            total = list(map(len, refs))
            wp, ws = (1.0 / n for n in total)
            weight = 0.5 * (wp + ws)
            cached = {}

            def edit_pair(rank):
                if rank not in cached:
                    native = native_rows[rank - 1]["segmentation"]
                    choice = actions.get(str(rank), {})
                    before = decode(choice.get(control, {}).get("rle", native))
                    after = decode(choice.get(treatment, {}).get("rle", native))
                    add, drop = changed_cells(before, after)
                    cached[rank] = before, add, drop
                return cached[rank]

            def delta(ref, rank):
                if rank is None:
                    return 0.0, 0.0, 0
                return effect(ref, *edit_pair(rank))

            occupied = [set(), set()]
            for i, j, overlap in correspondences(pool):
                eligibility["assigned"] += 1
                eligibility["assigned_at_least_075"] += overlap >= 0.75
                eligibility["assigned_at_least_090"] += overlap >= 0.90
                if overlap < 0.50:
                    eligibility["assigned_below_050"] += 1
                    continue
                occupied[0].add(i)
                occupied[1].add(j)
                rank_p, rank_s = matches[0][i][1], matches[1][j][1]
                state = membership(rank_p, rank_s)
                eligibility[state] += 1
                p, s = refs[0][i], refs[1][j]
                pp, ps = delta(p, rank_p), delta(p, rank_s)
                sp, ss = delta(s, rank_p), delta(s, rank_s)
                style, assign = symmetric(pp[0], ps[0], sp[0], ss[0])
                parts[5, column] += 0.5 * (ws - wp) * (ss[0] + pp[0])
                if state == "same_rank":
                    ordinal, add, drop = edit_pair(rank_p)
                    assert ordinal.shape == p.shape
                    signal, normalization, plus, minus = same_rank(pp, ss, (p, s), add, drop)
                    require(abs(style - signal - normalization) <= 1e-10 and assign == 0.0, "Shared assignment identity")
                    parts[0, column] += weight * signal
                    parts[1, column] += weight * normalization
                    for name, pos, neg in zip(CLASSES, plus, minus, strict=True):
                        cell_counts[name + "_added"] += int(pos)
                        cell_counts[name + "_removed"] += int(neg)
                    if plus.sum():
                        changed.add((iid, i))
                        signs[f"P{int(np.sign(pp[2]))}_S{int(np.sign(ss[2]))}"] += 1
                    if overlap >= .75:
                        common_only[:, column] += [weight * signal, weight * normalization, weight * style]
                        if plus.sum():
                            changed_strict.add((iid, i))
                else:
                    parts[2, column] += weight * style
                    if state == "one_missing":
                        parts[3, column] += weight * assign
                    elif state == "different_rank":
                        parts[4, column] += weight * assign
                    else:
                        require(style == assign == 0.0, "No assigned candidate implies zero effect")
            for side, coefficient in enumerate((-wp, ws)):
                for i, ref in enumerate(refs[side]):
                    if i not in occupied[side]:
                        eligibility[f"reference_{side}_outside_primary_pairs"] += 1
                        parts[6, column] += coefficient * delta(ref, matches[side][i][1])[0]
            largest_error = max(largest_error, abs(float(parts[:, column].sum() - prior_deltas[key][column])))

        require(largest_error <= 1e-10 and eligibility["assigned"] == 544
                and eligibility["assigned_below_050"] == 51, "Exact observed per-image shift reconstruction")
        results[key] = dict(
            components={term: legacy.interval(row, draws) for term, row in zip(TERMS, parts, strict=True)},
            full_shift=legacy.interval(parts.sum(axis=0), draws),
            maximum_image_error=largest_error,
            correspondence_counts=dict(eligibility),
            common_changed_hulls=len(changed),
            common_changed_groups=len({i for i, _ in changed}),
            common_changed_signs=dict(signs),
            identical_candidate_cell_edits=dict(cell_counts),
            secondary_common_075=dict(
                changed_hulls=len(changed_strict), changed_groups=len({i for i, _ in changed_strict}),
                components={k: legacy.interval(row, draws) for k, row in zip(
                    ("label_disagreement", "iou_normalization", "total"), common_only, strict=True)},
            ),
        )
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("protocol", "official", "slgsea", "owner", "capture", "old-readback", "original-replay",
                 "classical-root", "code", "prior-run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    start = perf_counter()
    read_checked(args.protocol, PRELOCK)
    prior, legacy = load_prior(args)
    scorer = legacy.module(args.code / "PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py", legacy.SCORER)
    geom = legacy.module(args.code / "PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py", legacy.GEOMETRY)
    roster, blocks, rgbs, parity, identity, swaps, candidates = legacy.originals(args, scorer)
    require(identity == prior["identity"] and swaps == prior["realized_swaps"] and candidates == prior["edited_candidates"],
            "Exact same closed masks and realized budgets")
    pseg, spsd, pan, metadata = legacy.read_references(args, scorer, roster)
    require(pan == prior["spsd_annotation_json_sha256"] and metadata == prior["spsd_metadata_disagreement"],
            "Same two admitted author references")
    draws = np.random.Generator(np.random.PCG64(5297)).integers(232, size=(4096, 232))
    older, pdelta = legacy.score_reference(pseg, blocks, rgbs, scorer, geom, draws, parity)
    newer, sdelta = legacy.score_reference(spsd, blocks, rgbs, scorer, geom, draws)
    require(prior["references"] == {"PSeg_SSDD": older, "SPSD": newer}, "Both reference results exactly reproduce")
    shifts = {key: sdelta[key] - pdelta[key] for key in pdelta}
    ranks = fixed_ranks(legacy, scorer, geom, prior, (pseg, spsd), blocks)
    result = calculate((pseg, spsd), blocks, rgbs, ranks, shifts, legacy, draws)
    natural = result["free_natural_minus_raw"]
    part = natural["components"]["same_candidate_label_disagreement"]
    proceed = (part["points"] <= -0.20 and part["percentile95_points"][1] < 0
               and natural["common_changed_hulls"] >= 20 and natural["common_changed_groups"] >= 8)
    doc = dict(status=STATUS, protocol_sha256=PRELOCK, prior_result_sha256=PRIOR_RESULT,
               worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), exact_both_reference_parity=True,
               images=232, native_candidates=23200, comparisons=result, bootstrap_seed=5297,
               bootstrap_replicates=4096, model_forwards=0, optimizer_updates=0,
               natural_common_disagreement_advances_source_design=proceed, elapsed_seconds=perf_counter()-start)
    payload = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
