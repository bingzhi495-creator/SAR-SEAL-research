"""Expose SSDD target differences at newly seeded stock uncertainty points on real archived matches."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch


BASE_SOURCE_SHA = "ca3f0595343945ebb5db9a7f8b4f8cb9ee529985ebc89cb96883d64c3833d37d"
BASE_RESULT_SHA = "d13b16956f4ca8518f89cc9f0d32f168d6acd19b24931cff6538a9df66104508"
CRITERION_SHA = "fe3a1b05acc7edc4476c03954d75e2b9c91d5612da333782aecd23d0d1cea5c3"
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
SEED_BASE = 530270000
K, GROUPS, WIDTH = 1296, 13, 100


def require(ok, message):
    if not ok:
        raise ValueError(message)


def assert_assignment(torch, src, dst, targets, groups, width):
    require(src.shape == dst.shape == (targets * groups,) and src.dtype == dst.dtype == torch.int64
            and src.unique().numel() == targets * groups
            and bool(((0 <= src) & (src < groups * width)).all()),
            "Original matched-query census and bounds")
    exact = torch.arange(targets)
    require(all(torch.equal(dst[src // width == group].sort().values, exact) for group in range(groups)),
            "Original one-to-one targets in each group")


def uniform_expected(a, b, points):
    require(a.shape == b.shape and a.dtype == b.dtype and a.ndim == 3, "Aligned original target grids")
    return (a ^ b).flatten(1).sum(1).double() * (points / (a.shape[1] * a.shape[2]))


def losses(torch, logits, labels):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * labels).sum(1) + 1) / (probability.sum(1) + labels.sum(1) + 1)
    return bce, dice


def observe_stock(torch, api, packet, packed, src, dst):
    frame = {}
    stock_sampler, stock_labels = api.get_uncertain_point_coords_with_randomness, api._sample_target_masks_at_points

    def sampler(logits, uncertainty, points, oversample, fraction):
        require(not frame and (points, oversample, fraction) == (K, 3, .75), "Original stock uncertainty settings")
        xy = stock_sampler(logits, uncertainty, points, oversample, fraction)
        frame.update(coarse=logits, xy=xy)
        return xy

    def labels(targets, indices, xy):
        require(frame["xy"] is xy and "labels" not in frame, "Actual stock criterion coordinates")
        value = stock_labels(targets, indices, xy)
        frame["labels"] = value
        return value

    class Permutation:
        mask_point_sample_ratio = 16
        _get_src_permutation_idx = api.SetCriterion._get_src_permutation_idx

    raw = packet["sparse"]
    require(raw.keys() == {"spatial_features", "query_features", "bias"}
            and all(t.is_floating_point() and bool(torch.isfinite(t).all()) for t in raw.values()),
            "Finite original sparse model factors")
    sparse = {key: tensor.float() for key, tensor in raw.items()}
    denominator = torch.tensor(len(src), dtype=torch.float32)
    with (torch.no_grad(), torch.random.fork_rng(devices=[]),
          patch.object(api, "get_uncertain_point_coords_with_randomness", sampler),
          patch.object(api, "_sample_target_masks_at_points", labels)):
        torch.manual_seed(SEED_BASE + packet["ordinal"])
        native = api.SetCriterion.loss_masks(Permutation(), {"pred_masks": sparse}, [packed], [(src, dst)], denominator)
    coarse, xy, a = (frame[key] for key in ("coarse", "xy", "labels"))
    require(coarse.shape == (len(src), 1, 144, 144) and xy.shape == (len(src), K, 2)
            and coarse.dtype == xy.dtype == a.dtype == torch.float32 and a.shape == (len(src), K)
            and bool(torch.isfinite(coarse).all()) and bool(((xy >= 0) & (xy < 1)).all()),
            "Exact final criterion shape and finite CPU FP32 maps and coordinates")
    logits = api.point_sample(coarse, xy, align_corners=False).squeeze(1)
    native_bce, native_dice = losses(torch, logits, a)
    for key, observed in (("loss_mask_ce", native_bce), ("loss_mask_dice", native_dice)):
        require(torch.allclose(native[key], observed.mean(), rtol=2e-6, atol=2e-7), "Native raw A " + key)
    return xy, logits, a, stock_labels


def observe_view(torch, api, packet, packed, fixed, keep, weights, cap):
    src, dst = packet["native_src"], packet["native_dst"]
    assert_assignment(torch, src, dst, len(keep), GROUPS, WIDTH)
    xy, logits, a_native, stock_labels = observe_stock(torch, api, packet, packed, src, dst)

    side = packed["masks"].shape[-1]
    require(side == 576 and packed["masks"].shape[-2] == side, "Exact original square mask canvas")
    grid = torch.arange(side * side, dtype=torch.float32).reshape(1, 1, side, side)
    pixels = api.point_sample(grid, xy.reshape(1, -1, 2), mode="nearest", align_corners=False)
    pixels = pixels.reshape(len(src), K).long()
    require(torch.equal(packed["masks"].flatten(1)[dst[:, None], pixels].float(), a_native),
            "All independently reconstructed A nearest labels at new stock points")

    selected = keep[dst]
    fixed_dst = keep.long().cumsum(0)[dst[selected]] - 1
    sampled_xy, sampled_logits, sampled_pixels = xy[selected], logits[selected], pixels[selected]
    indices = [(src[selected], fixed_dst)]
    masks = [target["masks"] for target in fixed]
    labels = [a_native[selected]]
    if len(fixed_dst):
        for mask in masks[1:]:
            sampled = stock_labels([{"masks": mask}], indices, sampled_xy)
            require(torch.equal(sampled, mask.flatten(1)[fixed_dst[:, None], sampled_pixels].float()),
                    "Stock alternate labels independently agree with original nearest grid")
            labels.append(sampled)
    else:
        labels += [a_native[selected], a_native[selected]]
    small = packet["is_small"][keep]
    counts = {}
    terms = [losses(torch, sampled_logits, y) for y in labels]
    for area, rows in (("all", torch.ones_like(small)), ("small", small), ("non_small", ~small)):
        matches = rows[fixed_dst]
        n = int(matches.sum())
        scope = dict(rows=int(rows.sum()), matches=n, target_points=n*K, loss_sums={})
        for arm, (bce, dice) in zip("abc", terms, strict=True):
            sums = dict(bce=float(bce[matches].double().sum()), dice=float(dice[matches].double().sum()))
            sums["weighted_bce"] = weights[0] * sums["bce"]
            sums["weighted_dice"] = weights[1] * sums["dice"]
            sums["weighted_combined"] = sums["weighted_bce"] + sums["weighted_dice"]
            scope["loss_sums"][arm] = sums
        scope["transitions"] = {}
        for key, i, j in (("a_to_b", 0, 1), ("b_to_c", 1, 2)):
            lhs, rhs = (labels[k][matches].bool() for k in (i, j))
            changed = lhs ^ rhs
            added, removed = ~lhs & rhs, lhs & ~rhs
            picked = fixed_dst[matches]
            expected = uniform_expected(masks[i], masks[j], K)
            scope["transitions"][key] = dict(
                grid_changed_rows=int((expected[rows] > 0).sum()),
                uniform_expected_changed_labels=float(expected[picked].sum()),
                added=int(added.sum()), removed=int(removed.sum()), changed_labels=int(changed.sum()),
                changed_matches=int(changed.any(1).sum()),
                changed_rows=int(picked[changed.any(1)].unique().numel()),
                gained_hit_matches=int((~lhs.any(1) & rhs.any(1)).sum()),
                lost_hit_matches=int((lhs.any(1) & ~rhs.any(1)).sum()),
                flips_toward_frozen_binary_prediction=int(((added & (sampled_logits[matches] >= 0))
                                                            | (removed & (sampled_logits[matches] < 0))).sum()),
            )
        counts[area] = scope
    hashes = cap.state_digest(dict(src=src, dst=dst, xy=xy, logits=logits, a_native=a_native))
    return counts, hashes


def canonical_bytes(path, digest):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Exact canonical file")
    raw = path.read_bytes()
    require(0 < len(raw) < 2**20 and sha256(raw).hexdigest() == digest, "Exact pinned bytes")
    return raw


def run(prior, geometry, anchor_source, anchor_report, out, expected_python):
    import yaml
    from rfdetr.models import criterion as api

    require(sha256(Path(api.__file__).read_bytes()).hexdigest() == CRITERION_SHA, "Exact stock criterion source")
    recipe = yaml.safe_load(canonical_bytes(prior / "RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml", RECIPE_SHA))
    train = recipe["model"]["train_config"]["init_args"]
    weights = (train["mask_ce_loss_coef"], train["mask_dice_loss_coef"])
    require(weights == (5., 5.) and train["mask_point_sample_ratio"] == 16, "Exact original criterion coefficients")
    anchor = json.loads(canonical_bytes(anchor_report, BASE_RESULT_SHA))
    base = ModuleType("p269_archived_exact_replay")
    base.__file__ = str(anchor_source)
    exec(compile(canonical_bytes(anchor_source, BASE_SOURCE_SHA), str(anchor_source), "exec"), base.__dict__)
    require(not out.exists() and not out.is_symlink(), "Fresh output only")
    accepted_base, replay_arms = base.observed_exposure, base.replay_arms
    results, stream = defaultdict(list), []
    branch = None

    def tracked_arms(*args):
        nonlocal branch
        item = replay_arms(*args)
        branch = item[-1]
        return item

    def tracked_observed(packet, packed, fixed, keep, cap, torch):
        stock_readback = accepted_base(packet, packed, fixed, keep, cap, torch)
        rows, digest = observe_view(torch, api, packet, packed, fixed, keep, weights, cap)
        results[branch].append(rows)
        stream.append(digest)
        return stock_readback

    with patch.object(base, "replay_arms", tracked_arms), patch.object(base, "observed_exposure", tracked_observed):
        base.run(prior, geometry, out.parent / "A_ANCHOR", expected_python)
    observed = json.loads((out.parent / "A_ANCHOR/RESULT_SAFE.json").read_bytes())
    require({k: v for k, v in observed.items() if k != "elapsed_seconds"}
            == {k: v for k, v in anchor.items() if k != "elapsed_seconds"},
            "Complete actual original P269 uniform draw anchor and view order")
    require(len(stream) == 30, "All exact archived final-layer packet views")
    result = {}
    for path, views in sorted(results.items()):
        result[path] = dict(views=len(views), areas={})
        for area in ("all", "small", "non_small"):
            values = base.sum_dicts([x[area] for x in views])
            require(values["matches"] == GROUPS*values["rows"] and values["target_points"] == K*values["matches"],
                    "All common rows contribute original match in each group")
            values["loss_means"] = {arm: {key: v / values["matches"] for key, v in fields.items()}
                                    for arm, fields in values.pop("loss_sums").items()}
            values["loss_delta"] = {
                key: {term: values["loss_means"][right][term] - v for term, v in values["loss_means"][left].items()}
                for key, left, right in (("a_to_b", "a", "b"), ("b_to_c", "b", "c"))}
            result[path]["areas"][area] = values
    direct = result["direct"]["areas"]["all"]
    require(direct["transitions"]["b_to_c"]["changed_labels"] == 0
            and all(v == 0 for v in direct["loss_delta"]["b_to_c"].values()),
            "Direct B/C negative control for points and both raw losses")
    payload = dict(
        status="P270_SSDD_FIT_ARCHIVED_CRITERION_TARGET_EXPOSURE_COMPLETE",
        scope="only_archived_final_layer_native_assignments_with_new_cpu_fp32_stock_uncertainty_draws",
        historical_criterion_coordinates_recovered=False, capture_protocol=anchor["capture_protocol"],
        original_capture_report_sha256=anchor["capture_report_sha256"],
        p269_uniform_anchor_sha256=BASE_RESULT_SHA,
        archived_uniform_points_logits_and_A_labels_stream_sha256=anchor["archived_input_points_logits_labels_stream_sha256"],
        new_assignment_points_logits_A_labels_stream_sha256=sha256("".join(stream).encode()).hexdigest(),
        stock_criterion_sha256=CRITERION_SHA, original_recipe_sha256=RECIPE_SHA, views=30,
        count_only_empty_views=2, seed_base=SEED_BASE, seed="base_plus_original_packet_ordinal", K=K, groups=GROUPS,
        oversample=3, importance_fraction=.75, stock_mask_loss_weights=dict(bce=weights[0], dice=weights[1]),
        loss_normalization="per_matched_row_mean_on_fixed_common_support_before_weighting",
        uniform_reference="K_times_full_final_grid_target_XOR_fraction_summed_per_original_fixed_match",
        common_support=anchor["common_support"], area_policy=anchor["area_policy"], branches=result,
        FIT_only=True, DEV=False, TEST=False, AP=False, model_forward_calls=0, optimizer_updates=0, reassignment=False,
    )
    out.mkdir()
    with (out / "RESULT_SAFE.json").open("x", encoding="utf-8", newline="\n") as stream_file:
        stream_file.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream_file.flush()
        os.fsync(stream_file.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "geometry", "anchor_source", "anchor_report", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--expected-python", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.sched_getaffinity(0) == {2, 3},
            "Bounded CPU only")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError("Closed diagnostic is offline")
    sys.addaudithook(offline)
    started = perf_counter()
    run(args.prior, args.geometry, args.anchor_source, args.anchor_report, args.output, args.expected_python)
    print(json.dumps(dict(event="P270_TARGET_EXPOSURE_WRITTEN", seconds=perf_counter()-started)), flush=True)


if __name__ == "__main__":
    main()
