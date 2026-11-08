"""Fixed FIT-only audit of signed boundary ranking on closed original matches."""
from collections import Counter, defaultdict
from importlib.metadata import version
import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELEASE_CODE = ROOT / "github_release_20261011/research_round_20260925_global/code"
HELPERS = RELEASE_CODE / "RFDETR_E24_FIT_EQUAL_HEAD_SCOUT_ACTUAL_20261108"
MATERIAL = ROOT / "RFDETR_E24_FIT_SELECTOR_MATERIALIZED_ACTUAL_PRIVATE_20261108_01"
OLD = ROOT / "RFDETR_E24_FIT_SELECTOR_SCOUT_ACTUAL_PRIVATE_20261108_01"
SOLVER = ROOT / "RFDETR_E24_FIT_EQUAL_HEAD_SOLVER_CANDIDATE_20261108_01/solver.py"

PINS = {
    "material_api": (HELPERS / "fit_selector_materialize.py", "c3f19a7dab49729c37737bf12f56683ba9ff0bef1432607fdb42c4691234530e"),
    "scout_api": (HELPERS / "fit_selector_scout.py", "961ee68d5f067c9f7ffa08d2650a2b45554c76b60aa9554e661a13b58b86b6d4"),
    "solver": (SOLVER, "e375d7a06ef7bc652e51a47cc2c16ee06fc280a6172fbd4eb5ed16ba3ce5738c"),
    "material_spec": (MATERIAL / "SPEC.json", "a8f1ebefc3e67ed2a857e7ae88beb1031aa55c37428de4e8588eaa93d387c617"),
    "material_acceptance": (MATERIAL / "ROOT_INDEPENDENT_MATERIALIZATION_ACCEPTED_PRIVATE.json", "742053fc509b9168b02727a8ed58120a471569b597916da764ea8a06e0fee9c4"),
    "material_index": (MATERIAL / "MATERIALIZED_INDEX_PRIVATE.json", "2db33d79ec28b949ccafaf6868ff26b164df344bf049f4db212bea3df88a89ea"),
    "old_spec": (OLD / "SPEC.json", "a45dd6bd99a097d69adb7f85c6007d2c53674b014009722f6cf50ba3a0c78534"),
    "old_acceptance": (OLD / "ROOT_INDEPENDENT_FIT_SCOUT_ACCEPTED_PRIVATE.json", "0b8d95e2d7a8546917beb85f37be4fd7eb0ecacd8f3d3eb2a6bed603d66b5364"),
    "old_result": (OLD / "RESULT.json", "f03d8559c7b6a65628e8d19973eafa54ab77734ae1440d6c002bfb81ca6740e4"),
    "old_models": (OLD / "HEAD_MODELS_PRIVATE.json", "1959137a3e232136d81b7f575ae47263b922d5ac8a2d4820588d2fa9017bd83b"),
    "image_manifest": (ROOT / "RFDETR_E24_FIT_IMAGE_ONLY_LOCAL_PRIVATE_20261108_01/FIT741_IMAGE_ONLY_MANIFEST_PRIVATE.json", "6dd6dd35f8418285696a224eab0ada6538224a0603c9eeb14c5ae98ea9fbcf95"),
    "FIT_bundle": (ROOT / "FULL_FIT_INPUT_BUNDLE_20261107_01/SSDD_FIT741_ORIGINAL_INPUTS_20261107_01.zip", "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"),
    "capture_index": (ROOT / "RFDETR_E24_FIT741_NATIVE_WINDOWS_CUDA_ACTUAL_PRIVATE_20261108_01/IMAGE_INDEX_PRIVATE.json", "12640ac80aa5a3a23dd135d9ca76f5358294ec601393bd70476ab005d4f1e9ed"),
}
DIRECTIONS = {1: "addition", -1: "removal"}
ARMS = ("stock", "frozen_mean", "frozen_rank", "rank_disagreement")
CONTROLS = ("stock", "frozen_mean")
PROBABILITY_ARMS = {"frozen_mean", "frozen_rank"}
SCOPES = ("all_original_matched_small", "top10_original_matched_small")
BUDGETS = (16, 32)
BOOTSTRAPS = 2048
BOOTSTRAP_SEED = 5201
MARGIN_PENALTY = {1: 9, -1: 10}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pair_auc(scores, labels):
    """Return exact tied pair wins and count within one instance and action."""
    correct = labels.astype(bool, copy=False)
    positive = scores[correct]
    negative = np.sort(scores[~correct])
    pairs = len(positive) * len(negative)
    if not pairs:
        return 0.0, 0
    before = np.searchsorted(negative, positive, side="left")
    after = np.searchsorted(negative, positive, side="right")
    return float((before + after).sum(dtype=np.int64)) / 2, pairs


def prefix(positions, scores, budget):
    return np.lexsort((positions, -scores))[:budget]


def rank_disagreement(rank, valid):
    """Prioritize local reversals of bright inside vs outside, with a fixed fallback."""
    return np.where(valid, -rank, -2.0)


def tally(selected, correct, sign):
    return dict(selected=selected, correct=correct, wrong=selected - correct,
                net_correct=2 * correct - selected,
                margin=19 * correct - MARGIN_PENALTY[sign] * selected)


def applied_rows(rows, sizes):
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    need(np.isfinite(scores).all(), "Finite original detector scores")
    return {int(i) for i in np.argsort(-scores, kind="stable")[:10] if sizes[i]}


def ratio(total, n):
    return float(total / n) if n else None


def paired_image_means(pairs):
    """Compare aligned per-instance records, averaging within image first."""
    image_pairs = []
    for arm, control in pairs:
        need(len(arm) == len(control), "Paired outcomes on the same instances")
        if len(arm):
            image_pairs.append((np.mean(arm), np.mean(control)))
    if not image_pairs:
        return dict(images=0, arm=None, control=None, delta=None, percentile95=None,
                    positive_images=0, negative_images=0, tied_images=0)
    values = np.asarray(image_pairs, dtype=np.float64)
    delta = values[:, 0] - values[:, 1]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(len(delta), size=(BOOTSTRAPS, len(delta)))
    intervals = np.quantile(delta[picks].mean(axis=1), (.025, .975)).tolist()
    return dict(images=len(delta), arm=float(values[:, 0].mean()),
                control=float(values[:, 1].mean()), delta=float(delta.mean()),
                percentile95=intervals, positive_images=int((delta > 0).sum()),
                negative_images=int((delta < 0).sum()), tied_images=int((delta == 0).sum()))


class ImageMetrics:
    def __init__(self):
        self.population = Counter()
        self.band = defaultdict(Counter)
        self.pixel = defaultdict(Counter)
        self.margin_by_instance = defaultdict(list)
        self.auc_by_instance = defaultdict(list)
        self.auc_pairs = defaultdict(Counter)
        self.application = defaultdict(Counter)
        self.application_rows = Counter()

    def record_application(self, arm, budget, sign, category, count):
        self.application[(arm, budget, sign)][category] += count

    def record_matched(self, scope, positions, signs, valid, labels, scores):
        for sign in DIRECTIONS:
            action = signs == sign
            positive = labels & action
            n, hits, valid_n = map(int, (action.sum(), positive.sum(), (valid & action).sum()))
            self.band[(scope, sign)].update(
                dict(instances=1, nonempty_instances=int(n > 0), hit_instances=int(hits > 0),
                     pixels=n, hits=hits, valid_pixels=valid_n, valid_hits=int((positive & valid).sum()),
                     no_hit_nonempty_instances=int(n > 0 and hits == 0),
                     both_class_instances=int(0 < hits < n)))
            for arm, score in scores.items():
                wins, pairs = pair_auc(score[action], labels[action])
                if pairs:
                    key = (scope, arm, sign)
                    self.auc_by_instance[key].append(wins / pairs)
                    self.auc_pairs[key].update(dict(wins=wins, pairs=pairs))

        for arm, score in scores.items():
            ordered = prefix(positions, score, max(BUDGETS))
            for budget in BUDGETS:
                edits = ordered[:budget]
                for sign in DIRECTIONS:
                    selected = edits[signs[edits] == sign]
                    counts = tally(len(selected), int(labels[selected].sum()), sign)
                    if arm in PROBABILITY_ARMS:
                        p = score[selected]
                        counts["predicted_hit_sum"] = float(p.sum(dtype=np.float64))
                        counts["predicted_margin_sum"] = float((19 * p - MARGIN_PENALTY[sign]).sum(dtype=np.float64))
                    key = (scope, arm, budget, sign)
                    self.pixel[key].update(counts)
                    self.margin_by_instance[key].append(counts["margin"])

    def add(self, rows, a, gt, scores):
        offsets = a["offsets"]
        label_offsets = a["label_offsets"]
        sizes = np.diff(offsets)
        applied = applied_rows(rows, sizes)
        self.population.update(retained=len(rows), empty_bands=int((sizes == 0).sum()),
                               top10_nonempty=len(applied))
        for i, row in enumerate(rows):
            lo, hi = map(int, offsets[i:i + 2])
            yl, yh = map(int, label_offsets[i:i + 2])
            need((lo, hi) == (row["start"], row["stop"]), "Original band identity")
            gt_id = row["GT_annotation_id"]
            if row["status"] == "matched":
                need(gt_id in gt and yh - yl == hi - lo, "Original signed label alignment")
                category = "matched_small" if gt[gt_id] else "matched_other"
            else:
                need(row["status"] == "unmatched" and gt_id is None and yl == yh,
                     "No incorrect-pixel imputation for unmatched candidates")
                category = "unmatched"
            self.population[category] += 1
            self.population["labeled_pixels"] += yh - yl
            self.population["correct_labeled_pixels"] += int(a["y"][yl:yh].sum())
            positions = a["positions"][lo:hi]
            signs = a["edit_sign"][lo:hi]
            valid = a["valid"][lo:hi]
            labels = a["y"][yl:yh]
            need(np.all((signs == 1) | (signs == -1)), "Only addition and removal")
            if category != "matched_small" and i not in applied:
                continue
            scores_for_row = {arm: score[lo:hi] for arm, score in scores.items()}
            if category == "matched_small":
                self.record_matched(SCOPES[0], positions, signs, valid, labels, scores_for_row)
                if i in applied:
                    self.record_matched(SCOPES[1], positions, signs, valid, labels, scores_for_row)
            if i in applied:
                self.application_rows[category] += 1
                for arm, score in scores_for_row.items():
                    ordered = prefix(positions, score, max(BUDGETS))
                    for budget in BUDGETS:
                        signs_selected = signs[ordered[:budget]]
                        for sign in DIRECTIONS:
                            self.record_application(arm, budget, sign, category,
                                                    int((signs_selected == sign).sum()))


def counters(values):
    total = Counter()
    for value in values:
        total.update(value)
    return total


def summarize_scope(images, scope):
    out = {}
    for sign, action in DIRECTIONS.items():
        direction = (scope, sign)
        band = counters(image.band[direction] for image in images)
        out[action] = dict(
            band=dict(band, hit_rate=ratio(band["hits"], band["pixels"]),
                      images_with_pixels=sum(image.band[direction]["pixels"] > 0 for image in images),
                      images_with_hits=sum(image.band[direction]["hits"] > 0 for image in images)),
            arms={},
        )
        for arm in ARMS:
            key = (scope, arm, sign)
            auc = counters(image.auc_pairs[key] for image in images)
            grouped = [np.mean(v) for image in images if (v := image.auc_by_instance[key])]
            arm_out = dict(
                auc=dict(pair_weighted=ratio(auc["wins"], auc["pairs"]),
                         comparable_pairs=auc["pairs"], image_then_instance_mean=ratio(sum(grouped), len(grouped)),
                         images=len(grouped), instances=sum(len(image.auc_by_instance[key]) for image in images),
                         paired_vs={control: paired_image_means(
                             (image.auc_by_instance[key], image.auc_by_instance[(scope, control, sign)])
                             for image in images) for control in CONTROLS if control != arm}),
                mixed_direction_prefix={},
            )
            for budget in BUDGETS:
                prefix_key = (scope, arm, budget, sign)
                count = counters(image.pixel[prefix_key] for image in images)
                result = dict(count, hit_rate=ratio(count["correct"], count["selected"]),
                              actual_margin_per_selected=ratio(count["margin"], count["selected"]),
                              images_with_selected=sum(image.pixel[prefix_key]["selected"] > 0 for image in images),
                              paired_margin_per_original_instance_vs={control: paired_image_means(
                                  (image.margin_by_instance[prefix_key],
                                   image.margin_by_instance[(scope, control, budget, sign)])
                                  for image in images) for control in CONTROLS if control != arm})
                if arm in PROBABILITY_ARMS:
                    result["predicted_margin_per_selected"] = ratio(count["predicted_margin_sum"], count["selected"])
                    result["mean_selected_model_probability"] = ratio(count["predicted_hit_sum"], count["selected"])
                arm_out["mixed_direction_prefix"][str(budget)] = result
            out[action]["arms"][arm] = arm_out
    return out


def summarize_combined(images, scope):
    """Combine the two actions on the same original instance before resampling."""
    combined = {}
    for budget in BUDGETS:
        margins = {
            arm: [
                tuple(add + remove for add, remove in zip(
                    image.margin_by_instance[(scope, arm, budget, 1)],
                    image.margin_by_instance[(scope, arm, budget, -1)], strict=True))
                for image in images
            ]
            for arm in ARMS
        }
        for arm in ARMS:
            counts = counters(
                image.pixel[(scope, arm, budget, sign)]
                for image in images for sign in DIRECTIONS)
            result = dict(
                counts,
                hit_rate=ratio(counts["correct"], counts["selected"]),
                actual_margin_per_selected=ratio(counts["margin"], counts["selected"]),
                paired_margin_per_original_instance_vs={
                    control: paired_image_means(zip(margins[arm], margins[control], strict=True))
                    for control in CONTROLS if control != arm
                },
            )
            if arm in PROBABILITY_ARMS:
                result["predicted_margin_per_selected"] = ratio(
                    counts["predicted_margin_sum"], counts["selected"])
            combined.setdefault(arm, {})[str(budget)] = result
    return combined

def summarize_application(images):
    rows = counters(image.application_rows for image in images)
    arms = {}
    for arm in ARMS:
        arms[arm] = {}
        for budget in BUDGETS:
            actions = {}
            for sign, action in DIRECTIONS.items():
                selected = counters(image.application[(arm, budget, sign)] for image in images)
                total = sum(selected.values())
                actions[action] = dict(selected, total=total,
                                       unmatched_fraction=ratio(selected["unmatched"], total),
                                       original_matched_small_fraction=ratio(selected["matched_small"], total))
            arms[arm][str(budget)] = dict(actions, total_selected=sum(a["total"] for a in actions.values()))
    return dict(candidates=dict(rows), arms=arms)


def file_pin(path, sha256):
    need(path.is_file() and path.resolve(strict=True) == path and not path.is_symlink(),
         "Canonical pin file")
    return dict(path=path.as_posix(), bytes=path.stat().st_size, sha256=sha256)


def verify(ref):
    path = Path(ref["path"])
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    need(signature(before) == signature(after) and before.st_size == ref["bytes"] == len(raw)
         and hashlib.sha256(raw).hexdigest() == ref["sha256"], "Exact stable executable input")
    return raw


def load_inputs():
    refs = {name: file_pin(*pin) for name, pin in PINS.items()}
    source = refs["material_api"]
    api = types.ModuleType("fit_selector_materialize")
    api.__file__ = source["path"]
    exec(compile(verify(source), api.__file__, "exec"), api.__dict__)
    sys.modules[api.__name__] = api
    scout = api.load(refs["scout_api"], "_frozen_signed_scout")
    solver = api.load(refs["solver"], "_frozen_signed_solver")
    spec = api.doc(refs["material_spec"])
    accepted = api.doc(refs["material_acceptance"])
    old_spec = api.doc(refs["old_spec"])
    old_accepted = api.doc(refs["old_acceptance"])
    old_result = api.doc(refs["old_result"])
    need(accepted["status"] == "ACTUAL_FULL_FIT741_FIXED_MATCH_FEATURES_ACCEPTED"
         and old_accepted["status"] == "ACTUAL_FIT741_EQUAL_HEAD_SCOUT_ACCEPTED"
         and all(not x["DEV"] and not x["TEST"] for x in (accepted, old_accepted, spec, old_spec))
         and accepted["material_index_ref"] == refs["material_index"]
         and old_accepted["material_acceptance_ref"] == refs["material_acceptance"]
         and old_accepted["original_result_ref"] == refs["old_result"]
         and old_result["model_ref"] == refs["old_models"]
         and old_spec["sources"]["solver"] == refs["solver"]
         and spec["FIT_bundle"] == refs["FIT_bundle"]
         and spec["capture_index"] == refs["capture_index"]
         and spec["image_manifest"] == refs["image_manifest"], "Closed FIT qualification chain")
    models = api.doc(refs["old_models"])["models"]
    need({k: models[k]["branch"] for k in ("mean", "rank")} == {"mean": "mean", "rank": "rank"},
         "Frozen means and rank head identities")
    manifest = api.doc(refs["image_manifest"])["items"]
    index = api.doc(refs["material_index"])
    capture = api.doc(refs["capture_index"])
    need(len(manifest) == len(index) == len(capture) == 741
         and Counter(x["partition"] for x in manifest) == {"head_fit": 516, "calibration": 114, "head_holdout": 111},
         "Exact FIT741 partition populations")
    gt = api.original_gt(spec, manifest)
    return api, scout, solver, models, manifest, index, capture, gt, refs


def run(protocol, output_dir):
    need(sys.version_info[:3] == (3, 10, 13) and version("numpy") == "1.26.4"
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Locked CPU-only runtime")
    need(re.fullmatch(r"[0-9a-f]{40}", protocol), "Exact lowercase protocol commit")
    need(output_dir.is_absolute() and output_dir.parent.resolve(strict=True) == ROOT
         and output_dir.name.startswith("RFDETR_E24_FIT_SIGNED_BOUNDARY_EVIDENCE_ACTUAL_PRIVATE_")
         and not output_dir.exists(), "Fresh private output beside the candidate")
    start = time.perf_counter()
    api, scout, solver, models, manifest, index, capture, gt, refs = load_inputs()
    per_image = defaultdict(dict)
    for annotation in gt["annotations"]:
        per_image[annotation["image_id"]][annotation["id"]] = (
            not annotation["iscrowd"] and annotation["area"] < 1024)
    diagnostics = []
    for item, meta, native in zip(manifest, index, capture, strict=True):
        need(all(item[key] == meta[key] == native[key] for key in ("image_id", "dataset_index", "partition")),
             "Immutable FIT741 image order and partition")
        if item["partition"] != "head_holdout":
            continue
        rows, a = scout.load_image(np, meta)
        f = scout.features(a)
        detector_scores = scout.detector_scores(np, rows, a)
        scores = {
            "stock": solver.stock_uncertainty(f),
            "frozen_mean": solver.predict_head(models["mean"], f, detector_scores),
            "frozen_rank": solver.predict_head(models["rank"], f, detector_scores),
            "rank_disagreement": rank_disagreement(f["rank"], f["valid"]),
        }
        need(all(np.isfinite(s).all() and s.shape == a["positions"].shape for s in scores.values()),
             "Finite scores aligned to original bands")
        image_gt = per_image[item["image_id"]]
        image = ImageMetrics()
        image.population["small_GT"] = sum(image_gt.values())
        image.population["distinct_matched_small_GT"] = len({
            row["GT_annotation_id"] for row in rows if image_gt.get(row["GT_annotation_id"], False)
        })
        image.add(rows, a, image_gt, scores)
        diagnostics.append(image)
    population = counters(image.population for image in diagnostics)
    need(len(diagnostics) == 111 and population["retained"] == 9083
         and population["empty_bands"] == 3087 and population["unmatched"] == 8792
         and population["labeled_pixels"] == 149923 and population["correct_labeled_pixels"] == 24813
         and population["top10_nonempty"] == 929, "Accepted FIT and prior top-ten accounting")
    application = summarize_application(diagnostics)
    need(all(v["16"]["total_selected"] == 14864 for v in application["arms"].values()),
         "Identical known P109 budget at top 16")
    population["small_GT_without_original_match"] = population["small_GT"] - population["distinct_matched_small_GT"]
    result = dict(
        status="FIT_HEAD_HOLDOUT_SIGNED_BOUNDARY_EVIDENCE_COMPLETE", protocol_commit=protocol,
        inputs=refs, source=api.file_ref(Path(__file__).resolve()), DEV=False, TEST=False,
        training=False, AP=False, GPU=False, thresholds_selected=False, mask_changes=False,
        images=len(diagnostics), population=dict(population), bootstrap=dict(draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED),
        scopes={scope: summarize_scope(diagnostics, scope) for scope in SCOPES},
        combined_mixed_direction_prefix={scope: summarize_combined(diagnostics, scope) for scope in SCOPES},
        applied_top10_gap=application,
        seconds=time.perf_counter() - start,
    )
    output_dir.mkdir()
    print(api.put(output_dir / "RESULT.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
