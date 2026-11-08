"""Compare oriented SAR gray profiles with marginal-preserving and axis controls."""
import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIGNED_HELPER = ROOT / "RFDETR_E24_FIT_SIGNED_BOUNDARY_EVIDENCE_CANDIDATE_20261109_01/signed_evidence.py"
SIGNED_SHA = "291d1e6a1175fc2a06fdfcdf6468e0f3dab1e406c62166cb839a87cfe067f1e4"
CAPTURE = ROOT / "RFDETR_E24_FIT741_NATIVE_WINDOWS_CUDA_ACTUAL_PRIVATE_20261108_01"
CAPTURE_ACCEPTANCE = CAPTURE / "ROOT_INDEPENDENT_FULL_FIT_CUDA_CAPTURE_ACCEPTED_PRIVATE.json"
CAPTURE_ACCEPTANCE_SHA = "24cb8e74b2a12f404ad397d90385c72283656a1304cf3f23e88faa0ad0d865cd"
MODES = ("aligned", "permuted", "axis")
BASELINES = ("stock", "frozen_mean", "frozen_rank")
ARMS = (*BASELINES, *(f"{kind}_{mode}" for kind in ("analytic", "learned") for mode in MODES))
CONTROLS = {
    "analytic_aligned": ("analytic_permuted", "analytic_axis", *BASELINES),
    "learned_aligned": ("learned_permuted", "learned_axis", *BASELINES),
}
BUDGETS = (16, 32)
ANALYTIC_INDEX = 2
FEATURE_NAMES = (
    "log1p_abs_logit", "edit_sign", "detector_score", "normal_strength",
    "center_gray", "inside_mean", "outside_mean", "signed_rail_vote_median",
    "signed_rail_vote_mean", "signed_rail_vote_sd", "absolute_rail_contrast_mean",
    "rail_polarity_agreement",
)
ITERATIONS = 96
STEP = .15
L2 = .02


def need(condition, message):
    if not condition:
        raise ValueError(message)


def inward_normal(mask, positions):
    """Centered differences on two three-pixel lines of the binary prediction."""
    y, x = np.divmod(positions.astype(np.intp), mask.shape[1])
    p = np.pad(mask, 2, mode="edge").astype(np.int8)
    y, x = y + 2, x + 2
    gy = sum(p[y + 2, x + j] - p[y - 2, x + j] for j in (-1, 0, 1))
    gx = sum(p[y + j, x + 2] - p[y + j, x - 2] for j in (-1, 0, 1))
    strength = np.hypot(gy.astype(np.float64), gx)
    normal_y = np.divide(gy, strength, out=np.zeros_like(strength), where=strength != 0)
    normal_x = np.divide(gx, strength, out=np.ones_like(strength), where=strength != 0)
    return normal_y, normal_x, strength / np.sqrt(18.0)


def bilinear(gray, y, x):
    yy = np.clip(y, 0, gray.shape[0] - 1)
    xx = np.clip(x, 0, gray.shape[1] - 1)
    y0 = yy.astype(np.intp)
    x0 = xx.astype(np.intp)
    y1 = np.minimum(y0 + 1, gray.shape[0] - 1)
    x1 = np.minimum(x0 + 1, gray.shape[1] - 1)
    fy, fx = yy - y0, xx - x0
    values = ((1 - fy) * ((1 - fx) * gray[y0, x0] + fx * gray[y0, x1])
              + fy * ((1 - fx) * gray[y1, x0] + fx * gray[y1, x1]))
    return values, (yy != y) | (xx != x)


def rails(gray, positions, ny, nx):
    """Take the same twelve Euclidean gray taps on each geometry arm."""
    y, x = np.divmod(positions.astype(np.intp), gray.shape[1])
    t = np.array([-1., 0., 1.])[None, :, None]
    n = np.array([1., 2., -1., -2.])[None, None, :]
    yy = y[:, None, None] - nx[:, None, None] * t + ny[:, None, None] * n
    xx = x[:, None, None] + ny[:, None, None] * t + nx[:, None, None] * n
    values, clipped = bilinear(gray, yy, xx)
    return values[..., :2], values[..., 2:], clipped.any(axis=(1, 2))


def permute_outer(outside, positions):
    """Use cyclic derangements, preserving every per-patch outer gray sample."""
    shift = 1 + positions.astype(np.intp) % 2
    indices = (np.arange(3)[None, :] + shift[:, None]) % 3
    return outside[np.arange(len(outside))[:, None], indices]


def pair_features(inside, outside, center, sign):
    """Symmetric reductions conceal the rail identities from the estimator."""
    inner = inside.mean(axis=2)
    outer = outside.mean(axis=2)
    contrast = (inner - outer) / 255.
    polarity = np.sign(contrast)
    signed_vote = (center[:, None] - .5 * (inner + outer)) * polarity * sign[:, None] / 255.
    return np.column_stack((
        np.sort(inner, axis=1).mean(axis=1) / 255.,
        np.sort(outer, axis=1).mean(axis=1) / 255.,
        np.median(signed_vote, axis=1), signed_vote.mean(axis=1), signed_vote.std(axis=1),
        np.abs(contrast).mean(axis=1), np.abs(polarity.mean(axis=1)),
    ))


def design(gray, mask, positions, logits, signs, detector_score):
    ny, nx, strength = inward_normal(mask, positions)
    inside, outside, clipped = rails(gray, positions, ny, nx)
    axis_inside, axis_outside, axis_clipped = rails(gray, positions, np.zeros_like(ny), np.ones_like(nx))
    center = gray.reshape(-1)[positions].astype(np.float64)
    common = np.column_stack((np.log1p(np.abs(logits)), signs, np.full(len(logits), detector_score),
                              strength, center / 255.))
    paired = {
        "aligned": pair_features(inside, outside, center, signs),
        "permuted": pair_features(inside, permute_outer(outside, positions), center, signs),
        "axis": pair_features(axis_inside, axis_outside, center, signs),
    }
    need(np.array_equal(paired["aligned"][:, :2], paired["permuted"][:, :2]),
         "Aligned/permuted gray marginals exact")
    features = {key: np.column_stack((common, value)).astype(np.float32) for key, value in paired.items()}
    analytic = {key: value[:, ANALYTIC_INDEX] for key, value in paired.items()}
    counts = dict(zero_normals=int((strength == 0).sum()), aligned_clipped=int(clipped.sum()),
                  axis_clipped=int(axis_clipped.sum()), pixels=len(positions))
    return features, analytic, counts


def sigmoid(value):
    return np.exp(-np.logaddexp(0., -value))


def fit_head(rows, labels, weights):
    """Fixed logistic estimator with identical regularization, rows and updates."""
    x = rows.astype(np.float64, copy=True)
    need(np.isfinite(x).all() and np.isfinite(weights).all()
         and np.all(weights > 0) and len(x) == len(labels) == len(weights), "Finite fit design and common weights")
    weights = weights / weights.sum()
    prior = np.dot(weights, labels)
    need(0 < prior < 1, "Both target classes")
    center = np.einsum("i,ij->j", weights, x)
    x -= center
    variance = np.einsum("i,ij,ij->j", weights, x, x)
    scale = np.where(variance > 1e-24, np.sqrt(variance), 1.)
    x /= scale
    beta = np.zeros(x.shape[1])
    bias = np.log(prior) - np.log1p(-prior)
    for _ in range(ITERATIONS):
        p = sigmoid(bias + np.einsum("ij,j->i", x, beta))
        residual = (p - labels) * weights
        bias -= STEP * residual.sum()
        beta -= STEP * (np.einsum("i,ij->j", residual, x) + L2 * beta)
    need(np.isfinite(bias) and np.isfinite(beta).all(), "Finite frozen probe fit")
    return dict(center=center, scale=scale, beta=beta, bias=bias)


def predict(rows, model):
    return sigmoid(model["bias"] + np.einsum("ij,j->i",
                  (rows - model["center"]) / model["scale"], model["beta"]))


def read_signed_helper():
    raw = SIGNED_HELPER.read_bytes()
    need(SIGNED_HELPER.resolve(strict=True) == SIGNED_HELPER
         and hashlib.sha256(raw).hexdigest() == SIGNED_SHA, "Exact previous signed-evidence helper")
    signed = types.ModuleType("_closed_signed_fit_helpers")
    signed.__file__ = SIGNED_HELPER.as_posix()
    exec(compile(raw, signed.__file__, "exec"), signed.__dict__)
    return signed


def original_records(api, item, native):
    ref = native["records"]
    name = f"image-{item['dataset_index']:04d}"
    need(native["dirname"] == name and ref["filename"] == "ROWS.json", "Exact closed original row artifact role")
    path = (CAPTURE / name / ref["filename"]).as_posix()
    data = json.loads(api.checked(dict(path=path, bytes=ref["bytes"], sha256=ref["sha256"])))
    need((data["image_id"], data["original_hw"], data["image_sha256"])
         == (item["image_id"], item["original_hw"], item["image_ref"]["sha256"]), "Original source image identity")
    return data


def load_instances(item, meta, native, small_ids, api, scout, codec, signed):
    from PIL import Image
    rows, arrays = scout.load_image(np, meta)
    sizes = np.diff(arrays["offsets"])
    eligible = signed.applied_rows(rows, sizes)
    accounting = Counter(retained=len(rows), top10_nonempty=len(eligible))
    selected = []
    for i, row in enumerate(rows):
        is_small = row["status"] == "matched" and row["GT_annotation_id"] in small_ids
        if i in eligible:
            category = "matched_small" if is_small else row["status"]
            accounting[f"top10_{category}"] += 1
        if is_small and sizes[i] and (item["partition"] == "head_fit" or i in eligible):
            selected.append(i)
    if not selected:
        return [], accounting
    original = original_records(api, item, native)
    with Image.open(BytesIO(api.checked(item["image_ref"]))) as image:
        gray = np.ascontiguousarray(image.convert("RGB").convert("L"), dtype=np.uint8)
    need(gray.shape == tuple(item["original_hw"]) and len(original["records"]) == len(rows)
         and hashlib.sha256(gray.tobytes()).hexdigest() == original["gray_sha256"], "Exact closed original gray grid")
    instances = []
    for i in selected:
        row, native_row = rows[i], original["records"][i]
        need(all(row[k] == native_row[k] for k in ("native_rank", "query_id", "score")),
             "Exact original candidate and score")
        rle = native_row["segmentation"]
        mask = codec.decode(dict(size=rle["size"], counts=rle["counts"].encode("ascii"))).astype(bool)
        lo, hi = map(int, arrays["offsets"][i:i + 2])
        yl, yh = map(int, arrays["label_offsets"][i:i + 2])
        pos = arrays["positions"][lo:hi]
        sign = arrays["edit_sign"][lo:hi]
        logits = arrays["logits"][lo:hi]
        y = arrays["y"][yl:yh]
        need((lo, hi) == (row["start"], row["stop"]) and len(y) == len(pos) > 0
             and mask.shape == gray.shape
             and np.array_equal(mask.reshape(-1)[pos], logits > 0)
             and np.array_equal(sign, np.where(logits > 0, -1, 1)),
             "Exact sparse grid, signed action and original matched label")
        features, analytic, diagnostics = design(gray, mask, pos, logits, sign, row["score"])
        accounting.update(diagnostics)
        accounting["instances"] += 1
        accounting["correct_pixels"] += int(y.sum())
        instance = dict(features=features, labels=y.copy())
        if item["partition"] == "head_holdout":
            instance.update(analytic=analytic, positions=pos.copy(), signs=sign.copy(), scores=row["score"],
                            stock_features=dict(
                                abs_logit=arrays["features"][lo:hi, 0].copy(),
                                mean_contrast=arrays["features"][lo:hi, 1].copy(),
                                rank=arrays["features"][lo:hi, 2].copy(),
                                edit_sign=sign.copy(), valid=arrays["valid"][lo:hi].copy()))
        instances.append(instance)
    return instances, accounting


def train(groups):
    blocks = {mode: [] for mode in MODES}
    labels, weights = [], []
    for group in groups:
        for item in group:
            label = item["labels"]
            labels.append(label)
            weights.append(np.full(len(label), 1. / (len(group) * len(label))))
            for mode in MODES:
                blocks[mode].append(item["features"][mode])
    targets = np.concatenate(labels)
    weight = np.concatenate(weights)
    return {mode: fit_head(np.concatenate(blocks[mode]), targets, weight) for mode in MODES}


def arm_scores(item, solver, frozen, fitted):
    f = item["stock_features"]
    return {
        "stock": solver.stock_uncertainty(f),
        "frozen_mean": solver.predict_head(frozen["mean"], f, item["scores"]),
        "frozen_rank": solver.predict_head(frozen["rank"], f, item["scores"]),
        **{f"analytic_{mode}": item["analytic"][mode] for mode in MODES},
        **{f"learned_{mode}": predict(item["features"][mode], fitted[mode]) for mode in MODES},
    }


def measure(groups, solver, frozen, fitted, signed):
    auc = defaultdict(list)
    auc_pairs = defaultdict(Counter)
    margin = defaultdict(list)
    totals = defaultdict(Counter)
    for group in groups:
        image_auc = defaultdict(list)
        image_margin = defaultdict(list)
        for item in group:
            signs, labels, positions = (item[k] for k in ("signs", "labels", "positions"))
            scores = arm_scores(item, solver, frozen, fitted)
            for arm, value in scores.items():
                need(np.isfinite(value).all(), "Finite paired scores")
                for sign in (-1, 1):
                    take = signs == sign
                    wins, pairs = signed.pair_auc(value[take], labels[take])
                    if pairs:
                        image_auc[arm, sign].append(wins / pairs)
                        auc_pairs[arm, sign].update(wins=wins, pairs=pairs)
                ordered = signed.prefix(positions, value, max(BUDGETS))
                for budget in BUDGETS:
                    selected = ordered[:budget]
                    combined = 0
                    for sign in (-1, 1):
                        subset = selected[signs[selected] == sign]
                        count = signed.tally(len(subset), int(labels[subset].sum()), sign)
                        totals[arm, budget, sign].update(count)
                        combined += count["margin"]
                    image_margin[arm, budget].append(combined)
        for arm in ARMS:
            for sign in (-1, 1):
                auc[arm, sign].append(image_auc[arm, sign])
            for budget in BUDGETS:
                margin[arm, budget].append(image_margin[arm, budget])
    out = {}
    for arm in ARMS:
        out[arm] = dict(auc={}, mixed_direction_prefix={})
        for sign, action in ((1, "addition"), (-1, "removal")):
            grouped = [float(np.mean(a)) for a in auc[arm, sign] if a]
            counts = auc_pairs[arm, sign]
            out[arm]["auc"][action] = dict(
                image_then_instance_mean=signed.ratio(sum(grouped), len(grouped)),
                comparable_instances=sum(len(a) for a in auc[arm, sign]),
                images=len(grouped), comparable_pairs=counts["pairs"],
                pair_weighted=signed.ratio(counts["wins"], counts["pairs"]),
                paired_vs={control: signed.paired_image_means(zip(auc[arm, sign], auc[control, sign], strict=True))
                           for control in CONTROLS.get(arm, ())},
            )
        for budget in BUDGETS:
            combined = totals[arm, budget, 1].copy()
            combined.update(totals[arm, budget, -1])
            combined["margin"] = sum(sum(v) for v in margin[arm, budget])
            combined["net_correct"] = combined["correct"] - combined["wrong"]
            out[arm]["mixed_direction_prefix"][str(budget)] = dict(
                combined=dict(combined), addition=dict(totals[arm, budget, 1]), removal=dict(totals[arm, budget, -1]),
                paired_margin_per_original_instance_vs={
                    control: signed.paired_image_means(zip(margin[arm, budget], margin[control, budget], strict=True))
                    for control in CONTROLS.get(arm, ())},
            )
    return out


def run(protocol, output):
    need(sys.version_info[:3] == (3, 10, 13) and np.__version__ == "1.26.4"
         and version("Pillow") == "12.3.0" and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and "torch" not in sys.modules, "Locked CPU-only NumPy/Pillow runtime")
    need(re.fullmatch(r"[0-9a-f]{40}", protocol), "Exact lowercase protocol commit")
    need(output.is_absolute() and output.parent.resolve(strict=True) == ROOT
         and output.name.startswith("RFDETR_E24_FIT_CONTOUR_CORRESPONDENCE_ACTUAL_PRIVATE_")
         and not output.exists(), "Fresh private result directory")
    start = time.perf_counter()
    signed = read_signed_helper()
    api, scout, solver, frozen, manifest, index, capture, gt, refs = signed.load_inputs()
    spec = api.doc(refs["material_spec"])
    capture_ref = signed.file_pin(CAPTURE_ACCEPTANCE, CAPTURE_ACCEPTANCE_SHA)
    closed = api.doc(capture_ref)
    need(spec["capture_acceptance"] == capture_ref and Path(spec["capture_root"]) == CAPTURE
         and closed["status"] == "ACTUAL_FULL_FIT741_NATIVE_CUDA_CAPTURE_ACCEPTED"
         and closed["original_index_ref"] == refs["capture_index"]
         and closed["DEV"] is closed["TEST"] is False, "Closed FIT capture chain")
    _, _, _, codec = api.runtime(spec)
    refs.update(signed_helper=signed.file_pin(SIGNED_HELPER, SIGNED_SHA), capture_acceptance=capture_ref)
    small_ids = {a["id"] for a in gt["annotations"] if not a["iscrowd"] and a["area"] < 1024}
    grouped, accounting = defaultdict(list), defaultdict(Counter)
    for item, meta, native in zip(manifest, index, capture, strict=True):
        need(all(item[k] == meta[k] == native[k] for k in ("image_id", "dataset_index", "partition")),
             "Immutable image-unique FIT partition order")
        part = item["partition"]
        if part == "calibration":
            continue
        instances, counts = load_instances(item, meta, native, small_ids, api, scout, codec, signed)
        grouped[part].append(instances)
        accounting[part].update(counts)
        accounting[part]["images"] += 1
        accounting[part]["images_with_instances"] += bool(instances)
    need(accounting["head_fit"]["images"] == 516
         and accounting["head_holdout"]["images"] == 111
         and accounting["head_holdout"]["retained"] == 9083
         and accounting["head_holdout"]["top10_nonempty"] == 929
         and accounting["head_holdout"]["instances"] == 229
         and accounting["head_holdout"]["top10_unmatched"] == 657,
         "Prior closed holdout and candidate population")
    fitted = train(grouped.pop("head_fit"))
    metrics = measure(grouped["head_holdout"], solver, frozen, fitted, signed)
    need(all(values["mixed_direction_prefix"]["16"]["combined"]["selected"] == 3664
             for values in metrics.values())
         and metrics["stock"]["mixed_direction_prefix"]["16"]["combined"]["margin"] == -3200
         and metrics["frozen_mean"]["mixed_direction_prefix"]["16"]["combined"]["margin"] == -2943
         and metrics["frozen_rank"]["mixed_direction_prefix"]["16"]["combined"]["margin"] == -2798,
         "Identical fixed support and original signed baseline anchor")
    result = dict(
        status="FIT_ONLY_CONTOUR_CORRESPONDENCE_PHASE_A_COMPLETE", protocol_commit=protocol,
        inputs=refs, source=api.file_ref(Path(__file__).resolve()),
        DEV=False, TEST=False, GPU=False, detector_retraining=False, mask_changes=False,
        AP=False, thresholds_selected=False, calibration_images_opened=0,
        feature_names=list(FEATURE_NAMES), bootstraps=signed.BOOTSTRAPS, bootstrap_seed=signed.BOOTSTRAP_SEED,
        fit=dict(iterations=ITERATIONS, step=STEP, l2=L2, equal_image_then_instance_weight=True),
        accounting={key: dict(value) for key, value in accounting.items()}, metrics=metrics,
        seconds=time.perf_counter() - start,
    )
    output.mkdir()
    print(api.put(output / "RESULT.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
