"""Group out-of-fold signed editing with frozen native logits and rendered SAR intensity."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from types import ModuleType

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


MIB = 1024**2
PRIOR_SHA = "261f533956651b6cd3868ebfa91624ed5a6b8efa7b40b69cb26254d3b09dd05d"
FOLDS_SHA = "c406a476e4b0f59fd2a1e8538de5e3e7f53182741ea9634708acd3f1ceb86b11"
SALT = "HRSID-FIT114-P319-RADIOMETRY-v1"
MODELS = ("ordered", "radiometry", "permuted")
PRIMARY = ("raw", *MODELS, "abs_logit", "random")
SECONDARY = ("raw", *MODELS)


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def keyed(*parts):
    return sha256(canonical([SALT, *parts])).digest()


def load_prior(path):
    source = path.read_bytes()
    require(sha256(source).hexdigest() == PRIOR_SHA, "Exact published P317 scorer required")
    prior = ModuleType("_p319_pinned_p317")
    exec(compile(source, str(path), "exec"), prior.__dict__)
    return prior


def five_folds(image_groups):
    groups = sorted(set(image_groups.values()), key=lambda group: (keyed("fold", group), group))
    require(len(groups) >= 5, "At least five independent source components required")
    return {group: i % 5 for i, group in enumerate(groups)}


def fixed_quota(area, inside):
    cap = min(16, int(area)//10)
    out, inn = int((~inside).sum()), int(inside.sum())
    if cap == 0:
        return {False: 0, True: 0}
    if cap == 1:
        return {False: int(out >= inn and out > 0), True: int(inn > out)}
    return {False: int(out > 0), True: int(inn > 0)}


def rank_points(score, inside, pixels, quota):
    selected = []
    for side in (False, True):
        options = np.flatnonzero(inside == side)
        order = np.lexsort((pixels[options], -score[options]))
        selected.extend(options[order[:quota[side]]])
    return np.asarray(selected, dtype=np.intp)


def permuted_radiometry(values, inside, key):
    permuted = values.copy()
    for side in (False, True):
        index = np.flatnonzero(inside == side)
        seed = int.from_bytes(keyed("side_permutation", key, side)[:8], "big")
        permuted[index] = values[np.random.default_rng(seed).permutation(index)]
    return permuted


def paired_values(base, augmented):
    """Extract the four new fields and require every original value to agree."""
    require(isinstance(augmented, dict) and augmented.keys() == base.keys()
            and isinstance(augmented["candidates"], list)
            and len(augmented["candidates"]) == len(base["candidates"]), "Feature/base stream image differs")
    stripped = dict(augmented, candidates=[])
    features = []
    for a, b in zip(base["candidates"], augmented["candidates"], strict=True):
        require(isinstance(b, dict) and b.keys() == a.keys() and isinstance(b["points"], list)
                and len(a["points"]) == len(b["points"]), "Feature/base stream candidates differ")
        points, values = [], []
        for point in b["points"]:
            require(isinstance(point, dict) and "radiometric" in point, "Radiometry field missing")
            stripped_point = dict(point)
            row = stripped_point.pop("radiometric")
            require(isinstance(row, list) and len(row) == 4 and all(type(v) in (float, int) for v in row)
                    and all(-4 <= v <= 4 for v in row[:3]) and 0 <= row[3] <= 1,
                    "Frozen radiometry feature range differs")
            points.append(stripped_point)
            values.append(row)
        stripped["candidates"].append(dict(b, points=points))
        features.append(np.asarray(values, dtype=np.float64).reshape(-1, 4))
    require(canonical(stripped) == canonical(base), "Feature/base stream original values differ")
    return features


@dataclass
class Candidate:
    key: str
    area: int
    inside: np.ndarray
    pixels: np.ndarray
    data: dict
    truth: np.ndarray | None
    intersection: int
    union: int


def image_candidates(prior, image, radio, truth):
    rows = image["candidates"]
    predictions, areas = prior.image_masks(rows)
    matches = prior.original_matches(predictions, truth)
    candidates = []
    for i, (row, area, values) in enumerate(zip(rows, areas, radio, strict=True)):
        key = f"{image['image_id']}/{row['query_id']}"
        xy, inside, pixels, _, data = prior.point_bank(row, area, key)
        gt = matches.get(i)
        bits = None
        intersection = union = 0
        if gt is not None:
            require(np.array_equal(prior.truth_bits(predictions[i], xy), inside), "Matched native RLE signs differ")
            bits = prior.truth_bits(gt, xy)
            intersection = int(prior.codec.area(prior.codec.merge([predictions[i], gt], intersect=True)))
            union = int(area) + int(prior.codec.area(gt)) - intersection
            require(2*intersection >= union > 0, "Original exact fixed match is below .50")
        ordered = data["ordered"]
        inputs = dict(ordered=ordered, radiometry=np.column_stack((ordered, values)),
                      permuted=np.column_stack((ordered, permuted_radiometry(values, inside, key))))
        candidates.append(Candidate(key, int(area), inside, pixels, inputs, bits, intersection, union))
    return candidates


def read_inputs(prior, cohort, paths, expected, groups, truth):
    grouped = defaultdict(list)
    ground_counts = Counter()
    census = Counter()
    with paths[0].open("rb") as capture, paths[1].open("rb") as feature:
        require(all(0 < os.fstat(stream.fileno()).st_size <= 384*MIB for stream in (capture, feature)),
                "Capture or feature stream length out of bounds")
        feature_hash = sha256()
        for image in prior.stream_capture(capture, cohort, expected[0]):
            raw = feature.readline(5*MIB+1)
            require(0 < len(raw) <= 5*MIB and raw.endswith(b"\n"), "Bounded feature image row required")
            feature_hash.update(raw)
            radiometry = paired_values(image, json.loads(raw))
            census["verified_images"] += 1
            census["verified_candidate_rows"] += len(radiometry)
            census["verified_sparse_points"] += sum(len(r) for r in radiometry)
            if image["stress"]:
                census["omitted_stress_images"] += 1
                continue
            image_id = image["image_id"]
            group = groups[image_id]
            labels = truth[image_id]
            grouped[group].extend(image_candidates(prior, image, radiometry, labels))
            ground_counts[group] += len(labels)
            census["scored_core_images"] += 1
        require(not feature.read(1) and feature_hash.hexdigest() == expected[1], "Exact closed feature digest differs")
    return grouped, ground_counts, census


def fit_models(training):
    fit_rows = []
    fit_groups = 0
    for candidates in training:
        matched = [row for row in candidates if row.truth is not None and len(row.inside)]
        if not matched:
            continue
        fit_groups += 1
        for row in matched:
            weights = np.empty(len(row.inside))
            sides = [row.inside == side for side in (False, True) if (row.inside == side).any()]
            for side in sides:
                weights[side] = 1/(len(matched)*len(sides)*side.sum())
            fit_rows.append((row.data, row.truth != row.inside, weights))
    require(fit_groups >= 2, "At least two nonempty fitting source groups required")
    label = np.concatenate([row[1] for row in fit_rows]).astype(float)
    require(label.min() == 0 and label.max() == 1, "Both edit labels must occur in fitted groups")
    weight = np.concatenate([row[2] for row in fit_rows])
    weight /= weight.sum()
    models = {}
    for model in MODELS:
        inputs = np.concatenate([row[0][model] for row in fit_rows])
        center = np.average(inputs, weights=weight, axis=0)
        scale = np.sqrt(np.average((inputs-center)**2, weights=weight, axis=0))
        scale[scale < 1e-8] = 1.
        design = np.column_stack(((inputs-center)/scale, np.ones(len(inputs))))

        def objective(params):
            logits = design @ params
            cost = weight @ (np.logaddexp(0., logits)-label*logits) + .005*(params[:-1] @ params[:-1])
            gradient = design.T @ (weight*(expit(logits)-label))
            gradient[:-1] += .01*params[:-1]
            return cost, gradient

        fit = minimize(objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
                       options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-7})
        require(fit.success and np.isfinite(fit.x).all(), "Fixed cross-fit logistic fit failed")
        models[model] = (center, scale, fit.x)
    return models, dict(fitting_groups=fit_groups, fitting_rows=len(fit_rows), fitting_points=len(label))


def edit_sets(prior, area, inside, pixels, data, candidate_key, models):
    """Return selections using frozen predictions and learned parameters only."""
    confidence = {key: expit(((data[key]-center)/scale) @ params[:-1] + params[-1])
                  for key, (center, scale, params) in models.items()}
    none = np.empty(0, dtype=np.intp)
    primary, secondary = {"raw": none}, {"raw": none}
    quota = fixed_quota(area, inside)
    if not any(quota.values()):
        return ({key: none for key in PRIMARY}, {key: none for key in SECONDARY})
    seed = int.from_bytes(keyed("fixed_random", candidate_key)[:8], "big")
    confidence["abs_logit"] = -np.abs(data["ordered"][:, 0])
    confidence["random"] = np.random.default_rng(seed).random(len(inside))
    for arm in PRIMARY[1:]:
        primary[arm] = rank_points(confidence[arm], inside, pixels, quota)
    for arm in SECONDARY[1:]:
        secondary[arm] = prior.choose_points(confidence[arm], inside, pixels, area)
    return primary, secondary


def score_group(prior, candidates, models):
    primary = {arm: Counter() for arm in PRIMARY}
    secondary = {arm: Counter() for arm in SECONDARY}
    census = Counter(candidates=len(candidates))
    for row in candidates:
        selected, selective = edit_sets(prior, row.area, row.inside, row.pixels, row.data, row.key, models)
        if row.truth is not None:
            census["fixed_matches"] += 1
            for threshold in (75, 90):
                census[f"original_{threshold}"] += int(100*row.intersection >= threshold*row.union)
        for actions, totals in ((selected, primary), (selective, secondary)):
            for arm, indices in actions.items():
                prior.score_candidate(totals[arm], indices, row.inside, row.truth, row.intersection, row.union)
    census["unmatched_candidates"] = census["candidates"] - census["fixed_matches"]
    return primary, secondary, census


def report_arms(prior, arm_totals, matches):
    result = {}
    for arm, totals in arm_totals.items():
        row = {key: int(totals[key]) for key in prior.NUMERIC}
        for action in ("add", "delete"):
            row[f"matched_wrong_{action}"] = row[action] - row[f"unmatched_{action}"] - row[f"matched_correct_{action}"]
        for threshold in (75, 90):
            row[f"net_{threshold}"] = row[f"recover_{threshold}"] - row[f"lose_{threshold}"]
        row["mean_fixed_iou"] = totals["fixed_iou_sum"]/matches if matches else None
        result[arm] = row
    return result


def intervals(group_results, arms, policy):
    selected = [group_results[g] for g in sorted(group_results) if group_results[g][2]["fixed_matches"]]
    if not selected:
        return {}
    groups = len(selected)
    draws = np.random.default_rng(319114).integers(0, groups, size=(4096, groups))
    result = {}
    for arm in arms:
        if arm == "radiometry":
            continue
        differences = np.asarray([
            (item[policy]["radiometry"]["margin_gain"]-item[policy][arm]["margin_gain"])/item[2]["fixed_matches"]
            for item in selected
        ])
        low, high = np.quantile(differences[draws].mean(axis=1), [.025, .975])
        result[arm] = dict(groups=groups, mean=float(differences.mean()), lower=float(low), upper=float(high),
                           positive_groups=int((differences > 0).sum()))
    return result


def cross_fit(prior, folds, rows_by_group, gt_counts):
    require(set(folds) == set(rows_by_group), "Exactly one held evaluation per core source component required")
    all_census = Counter()
    total_primary = {arm: Counter() for arm in PRIMARY}
    total_secondary = {arm: Counter() for arm in SECONDARY}
    fold_summaries, group_results = [], {}
    for fold in range(5):
        held = sorted(g for g, assigned in folds.items() if assigned == fold)
        training = sorted(set(folds)-set(held))
        models, fitted = fit_models(rows_by_group[g] for g in training)
        fold_census = Counter()
        for group in held:
            p, s, census = score_group(prior, rows_by_group[group], models)
            group_results[group] = (p, s, census)
            fold_census.update(census)
            for totals, actual in ((total_primary, p), (total_secondary, s)):
                for arm in totals:
                    totals[arm].update(actual[arm])
        all_census.update(fold_census)
        fold_summaries.append(dict(fold=fold, train_components=len(training), held_components=len(held),
                                   held_fixed_matches=fold_census["fixed_matches"], **fitted))
    all_census["eligible_gt"] = sum(gt_counts.values())
    denominator = all_census["fixed_matches"]
    return dict(census=dict(all_census), folds=fold_summaries,
                primary=report_arms(prior, total_primary, denominator),
                secondary_selective=report_arms(prior, total_secondary, denominator),
                primary_group_intervals=intervals(group_results, PRIMARY, 0),
                selective_group_intervals=intervals(group_results, SECONDARY, 1))


def run(prior, paths, expected, bundle):
    require(all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in expected), "Exact closed stream SHAs required")
    cohort, index = prior.load_scope(bundle)
    roles, groups = prior.split_roles(cohort)
    require(prior.split_digest(roles, groups) == prior.ROLE_SHA, "Exact sealed FIT group components required")
    core_groups = {row["image_id"]: groups[row["image_id"]] for row in cohort if row["core"]}
    folds = five_folds(core_groups)
    fold_digest = sha256(canonical(folds)).hexdigest()
    require(fold_digest == FOLDS_SHA, "Independent prelocked core fold digest differs")
    truth = prior.ground_truth(index, cohort)
    rows, gt_counts, census = read_inputs(prior, cohort, paths, expected, groups, truth)
    return dict(status="HRSID_FIT114_RADIOMETRIC_SIGNED_EDIT_OOF_COMPLETE_V1",
                prior_source_sha256=PRIOR_SHA, capture_sha256=expected[0], features_sha256=expected[1],
                fit_bundle_sha256=prior.BUNDLE_SHA, fit_index_sha256=prior.INDEX_SHA,
                fold_design_sha256=fold_digest, capture_census=dict(census),
                scope=dict(FIT=True, DEV=False, TEST=False, AP=False, model_forward_calls=0,
                           detector_optimizer_updates=0, readout_fitted_on_complementary_source_groups=True,
                           prediction_only_edit_selection=True, frozen_detector_previously_trained_on_FIT=True,
                           exploratory_oof=True, stress_omitted_from_fitting_and_scoring=True,
                           intervals="paired held-group bootstrap on observed cross-fit scores without refitting"),
                **cross_fit(prior, folds, rows, gt_counts))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("prior-source", "capture", "features", "bundle", "output"):
        cli.add_argument("--"+key, type=Path, required=True)
    for key in ("capture-sha256", "features-sha256"):
        cli.add_argument("--"+key, required=True)
    args = cli.parse_args()
    for path in (args.prior_source, args.capture, args.features, args.bundle):
        require(path.is_absolute() and path.resolve(strict=True) == path, "Canonical input file required")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical output file required")
    prior = load_prior(args.prior_source)
    result = run(prior, (args.capture, args.features), (args.capture_sha256, args.features_sha256), args.bundle)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
