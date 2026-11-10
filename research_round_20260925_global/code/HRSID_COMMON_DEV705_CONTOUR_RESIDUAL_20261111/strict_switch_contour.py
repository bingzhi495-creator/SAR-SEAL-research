"""Compute a truth-assisted contour-edit ceiling on fixed HRSID DEV705 masks."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import median
import sys
from types import ModuleType


P312_SOURCE_SHA = "ef819eb9c81a30845a400839bbba1ebde4541aeff7e30d96a8d2565505f9c40d"
P312_RESULT_SHA = "0aec5ab8ddf6f21259e7787482ef25ae725f5bd3f9409534ea61c832bf5445dd"
STATUS = "P313_HRSID_COMMON_STRICT_CONTOUR_RESIDUAL_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE"
RADII = (1, 2, 4)
RADIUS = 2
BUDGETS = (4, 8, 16, 32)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, digest, maximum):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical released P312 source or result path required")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and sha256(raw).hexdigest() == digest,
            "Exact released P312 source or result digest required")
    return raw


def load_p312(path):
    raw = pinned(path, P312_SOURCE_SHA, 64 * 1024)
    module = ModuleType("_p313_exact_released_p312")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def minimum_edits(deficit, fn_capacity, fp_capacity):
    """An FN addition pays 10 units of deficit and an FP deletion pays nine."""
    add = min(fn_capacity, (deficit + 9) // 10)
    delete = (max(deficit - 10 * add, 0) + 8) // 9
    return add + delete if delete <= fp_capacity else None


def measure_bands(pred, gt, inside):
    """Count exact Boolean edit capacities on the full padded mask geometry."""
    import numpy as np
    from scipy.ndimage import maximum_filter, minimum_filter

    require(pred.ndim == 2 and pred.shape == gt.shape == inside.shape
            and pred.dtype == gt.dtype == inside.dtype == np.bool_ and not (gt & ~inside).any()
            and pred.any() and gt.any(), "Exact Boolean padded prediction, truth and original canvas required")
    fn, fp = gt & ~pred, pred & ~gt
    intersection, union = int((gt & pred).sum()), int((gt | pred).sum())
    fn_total, fp_total = int(fn.sum()), int(fp.sum())
    off_canvas_fp = int((fp & ~inside).sum())
    deficit = max(9 * union - 10 * intersection, 0)
    bands = {}
    for radius in RADII:
        width = 2 * radius + 1
        outer = maximum_filter(pred, size=width, mode="constant", cval=0)
        inner = ~minimum_filter(pred, size=width, mode="constant", cval=0)
        editable_fn = int((fn & outer & inside).sum())
        editable_fp = int((fp & inner & inside).sum())
        gt_near = maximum_filter(gt, size=width, mode="constant", cval=0)
        bands[radius] = dict(editable_fn=editable_fn, editable_fp=editable_fp, fp_near_gt=int((fp & gt_near).sum()),
                             min_edits=minimum_edits(deficit, editable_fn, editable_fp))
    return dict(intersection=intersection, union=union, fn=fn_total, fp=fp_total, off_canvas_fp=off_canvas_fp,
                deficit=deficit, unrestricted_in_canvas_min_edits=minimum_edits(deficit, fn_total, fp_total - off_canvas_fp),
                bands=bands)


def measure_prediction(pred, gt):
    """Use P312's overlap grid and retain the least-movement maximum-IoU shift."""
    import numpy as np

    require(pred.ndim == 2 and pred.shape == gt.shape and pred.dtype == gt.dtype == np.bool_,
            "Exact original Boolean prediction and truth required")
    py, px = np.nonzero(pred)
    gy, gx = np.nonzero(gt)
    require(min(len(py), len(gy)) > 0, "Two nonempty original masks required")
    height, width = gt.shape
    intersections = {}
    for dy in range(-RADIUS, RADIUS + 1):
        for dx in range(-RADIUS, RADIUS + 1):
            y, x = py + dy, px + dx
            in_frame = (0 <= y) & (y < height) & (0 <= x) & (x < width)
            intersections[dy, dx] = int(gt[y[in_frame], x[in_frame]].sum())
    shift = min(intersections, key=lambda d: (-intersections[d], d[0] * d[0] + d[1] * d[1], d[0], d[1]))
    overlaps = dict(raw=intersections[0, 0],
                    vertical=max(n for (_, dx), n in intersections.items() if dx == 0),
                    horizontal=max(n for (dy, _), n in intersections.items() if dy == 0),
                    joint=intersections[shift])
    y0, x0 = min(py.min(), gy.min()) - RADIUS, min(px.min(), gx.min()) - RADIUS
    y1, x1 = max(py.max(), gy.max()) + RADIUS + 1, max(px.max(), gx.max()) + RADIUS + 1
    size = int(y1 - y0), int(x1 - x0)
    truth = np.zeros(size, dtype=bool)
    truth[gy - y0, gx - x0] = True
    yy, xx = np.ogrid[y0:y1, x0:x1]
    inside = (0 <= yy) & (yy < height) & (0 <= xx) & (xx < width)

    scopes, masks = {}, {}
    for name, (dy, dx), expected in (("raw", (0, 0), overlaps["raw"]), ("aligned", shift, overlaps["joint"])):
        shifted = np.zeros(size, dtype=bool)
        shifted[py - y0 + dy, px - x0 + dx] = True
        row = measure_bands(shifted, truth, inside)
        require(int(shifted.sum()) == len(py) and row["intersection"] == expected,
                "Padded contour crop must retain all predicted pixels and exact translation overlap")
        scopes[name] = row
        masks[name] = shifted
    return dict(pred_area=len(py), gt_area=len(gy), area_impossible=10 * min(len(py), len(gy)) < 9 * max(len(py), len(gy)),
                overlaps=overlaps, shift=shift, contour=scopes,
                edit_budget=min(16, scopes["raw"]["union"] // 10),
                translation_xor_pixels=int((masks["raw"] ^ masks["aligned"]).sum()))


def measured(pred_rle, gt_rle, value, p311):
    from pycocotools import mask

    expected = p311.fixed_edit(pred_rle, gt_rle, value)
    row = measure_prediction(mask.decode(pred_rle).astype(bool), mask.decode(gt_rle).astype(bool))
    require((row["pred_area"], row["gt_area"], row["overlaps"]["raw"], row["area_impossible"])
            == (expected["prediction_area"], expected["gt_area"], expected["intersection"],
                not expected["area_ratio_can_reach_090"]),
            "Exact original P311 mask areas, overlap and area ceiling required")
    return row, expected


def scope_summary(pairs):
    budgets = [original["edit_budget"] for original, _ in pairs]
    scope = dict(targets=len(pairs), K_median=median(budgets) if budgets else None,
                 K_total=sum(budgets),
                 same_K_unrestricted_in_canvas_recoverable=sum(
                    row["unrestricted_in_canvas_min_edits"] is not None
                    and row["unrestricted_in_canvas_min_edits"] <= original["edit_budget"] for original, row in pairs),
                  off_canvas_fp_pixels=sum(row["off_canvas_fp"] for _, row in pairs),
                 masks_with_off_canvas_fp=sum(row["off_canvas_fp"] > 0 for _, row in pairs),
                 total_fp=sum(row["fp"] for _, row in pairs), total_fn=sum(row["fn"] for _, row in pairs),
                 originally_area_impossible_targets=sum(original["area_impossible"] for original, _ in pairs),
                 unrestricted_in_canvas_reachable=sum(row["unrestricted_in_canvas_min_edits"] is not None
                                                      for _, row in pairs))
    bands = {}
    previous = 0
    for radius in RADII:
        cells = [(original, row["bands"][radius]) for original, row in pairs]
        counts = [value["min_edits"] for _, value in cells if value["min_edits"] is not None]
        require(previous <= len(counts) <= scope["unrestricted_in_canvas_reachable"],
                "Nested contour bands and unrestricted edit ceiling must reconcile")
        previous = len(counts)
        bands[str(radius)] = dict(
            recoverable=len(counts), blocked=len(pairs) - len(counts),
            same_K_recoverable=sum(value["min_edits"] is not None and value["min_edits"] <= original["edit_budget"]
                                   for original, value in cells),
            same_K_recoverable_of_original_area_impossible=sum(
                original["area_impossible"] and value["min_edits"] is not None
                and value["min_edits"] <= original["edit_budget"] for original, value in cells),
            min_edits_median=median(counts) if counts else None,
            min_edits_at_most={str(budget): sum(value <= budget for value in counts) for budget in BUDGETS},
            recoverable_of_original_area_impossible=sum(original["area_impossible"] and value["min_edits"] is not None
                                                        for original, value in cells),
            editable_fn_pixels=sum(row["editable_fn"] for _, row in cells),
            editable_fp_pixels=sum(row["editable_fp"] for _, row in cells),
            fp_near_gt_pixels_including_off_canvas=sum(row["fp_near_gt"] for _, row in cells),
        )
    cap = [bands[str(radius)]["same_K_recoverable"] for radius in RADII]
    require(cap == sorted(cap) and cap[-1] <= scope["same_K_unrestricted_in_canvas_recoverable"],
            "Same per-mask edit budgets must preserve nested recoverability")
    return dict(scope, by_radius=bands)


def contour_summary(rows):
    raw = [(row, row["contour"]["raw"]) for row in rows]
    aligned = [(row, row["contour"]["aligned"]) for row in rows if row["contour"]["aligned"]["deficit"] > 0]
    require(all(value["deficit"] > 0 for _, value in raw), "All original switch masks must fail .90 before edits")
    return dict(raw=scope_summary(raw), best_translation_remaining=scope_summary(aligned),
                already_recovered_by_translation=len(raw) - len(aligned),
                identity_chosen_as_best_translation=sum(row["shift"] == (0, 0) for row in rows),
                best_translation_xor_pixels_total=sum(row["translation_xor_pixels"] for row in rows),
                remaining_translation_xor_pixels_total=sum(row["translation_xor_pixels"] for row, _ in aligned))


def diagnose(args):
    p312 = load_p312(args.p312_source)
    prior = json.loads(pinned(args.p312_result, P312_RESULT_SHA, 1024**2))
    require(prior["status"] == p312.STATUS and prior["closed_predictions_only"] is True
            and prior["official_TEST_opened"] is prior["COCO_AP"] is False and p312.RADIUS == RADIUS,
            "Exact prior released fixed-mask translation result required")
    original_summary = p312.summarize
    p312.measure = measured

    def augment_summary(rows):
        return dict(original_summary(rows), contour=contour_summary(rows))

    p312.summarize = augment_summary
    reconstructed = p312.diagnose(args)
    result = {}
    for stratum, entries in reconstructed["strata"].items():
        result[stratum] = {}
        for direction, entry in entries.items():
            summary = entry["totals"]
            contour = summary.pop("contour")
            require(contour["already_recovered_by_translation"] == summary["by_family"]["joint"]["recovered"],
                    "Best translated residual and original P312 strict recovery must reconcile")
            result[stratum][direction] = dict(failing_arm=entry["failing_arm"], totals=contour,
                by_failing_raw_iou_band={band: group.pop("contour") for band, group in entry["by_failing_raw_iou_band"].items()})
    require(reconstructed == prior, "Exact released P312 aggregate must reproduce before emitting P313")
    return dict(status=STATUS, p312_source_sha256=P312_SOURCE_SHA, p312_result_sha256=P312_RESULT_SHA,
                p311_result_sha256=prior["p311_result_sha256"], provenance=prior["provenance"],
                original_p312_reproduced=True, closed_predictions_only=True, official_TEST_opened=False,
                COCO_AP=False, model_forward_calls=0, training=False, device="CPU", training_seed=prior["training_seed"],
                equal_nonempty_prefix_rows_per_arm=prior["equal_nonempty_prefix_rows_per_arm"], transitions=prior["transitions"],
                radii_original_pixels=RADII, primary_radius_original_pixels=2, distance="Chebyshev",
                fixed_edit_budget="K=min(16,floor(original unshifted union/10)); same K in both scopes",
                translation_budget_scope="translation is outside the edit quota; its whole-mask XOR pixel count is separate",
                translation_tie_rule="maximum integer intersection, then minimum squared movement, then signed dy, dx",
                support="in-canvas prediction exterior dilation for FN and prediction interior erosion band for FP",
                truth_assistance="ground truth identifies correct edits inside a prediction-defined pixel set",
                post_translation_scope="truth-selected translations and still-failing masks form an oracle diagnostic",
                off_canvas_policy="translated off-canvas foreground remains FP and cannot be edited",
                descriptive_fp_near_gt_scope="distance to GT foreground, including off-canvas FP, does not define edit support",
                strata=result)


def deny_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("Offline fixed-mask diagnostic required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    paths = ("code-root", "pair", "native-arm-receipt", "center-arm-receipt", "accepted-result", "p311-result",
             "p311-source", "p312-result", "p312-source", "output")
    for name in paths:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("pair-sha256", "training-protocol", "accepted-result-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode, "Exact offline isolated CPU invocation required")
    require(args.code_root.is_absolute() and args.code_root.resolve(strict=True) == args.code_root
            and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Canonical sources and fresh output required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    require({name: version(name) for name in ("numpy", "pycocotools", "scipy")}
            == dict(numpy="1.26.4", pycocotools="2.0.11", scipy="1.15.3"), "Exact original mask runtime required")
    sys.addaudithook(deny_network)
    result = diagnose(args)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
