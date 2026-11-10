"""Score one-pixel binary morphology and a same-count sham on frozen HRSID DEV705."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import math
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO


P311 = "HRSID_COMMON_DEV705_STRICT_RESIDUAL_20261111/strict_residual.py"
P311_SHA = "6fbefd4e62e7715518c1e460db9b0c32f5c407f11adef8de6cd5737b65346b54"
STATUS = "HRSID_COMMON_E24_DEV705_RLE_MORPHOLOGY_BASELINE_COMPLETE_V1"
VARIANTS = ("identity", "morphology", "sham")
SHAM_DOMAIN = "HRSID_DEV705_RLE_ONE_PIXEL_SIDE_MATCHED_SHAM_V1"
RULE_DOMAIN = "HRSID_DEV705_RLE_ONE_PIXEL_MORPHOLOGY_TIES_V1"
EDGES = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def hash_rows(rows):
    digest = sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    return digest.hexdigest()


def encode_mask(bitmap):
    encoded = mask.encode(np.asfortranarray(bitmap, dtype=np.uint8))
    return dict(size=encoded["size"], counts=encoded["counts"].decode("ascii"))


def edit_rle(rle, image_id, mapped_rank):
    """Read a single original prediction, then generate two independent edited masks."""
    area = int(mask.area(rle))
    cap = min(16, area // 10)
    counts = dict(area=area, cap=cap, add=0, delete=0)
    if not cap:
        return rle, rle, counts
    height, width = rle["size"]
    bbox = tuple(mask.toBbox(rle))
    x, y, w, h = map(int, bbox)
    require(bbox == (x, y, w, h) and 0 <= x < x+w <= width and 0 <= y < y+h <= height,
            "Exact integer mask bounding box required")
    left, top = max(0, x-1), max(0, y-1)
    right, bottom = min(width, x+w+1), min(height, y+h+1)
    bitmap = mask.decode(rle)
    local = bitmap[top:bottom, left:right]
    require(bitmap.shape == (height, width) and int(local.sum()) == area, "Exact foreground in original bounding box required")
    ph, pw = local.shape
    padded = np.pad(local, 1, mode="constant")
    neighbors = np.zeros(local.shape, dtype=np.uint8)
    for dy, dx in EDGES:
        neighbors += padded[1+dy:1+dy+ph, 1+dx:1+dx+pw]
    foreground = local != 0
    holes = ~foreground & (neighbors == 8)
    isolated = foreground & (neighbors == 0)
    signed = [(int(j)+top, int(i)+left, value) for positions, value in ((holes, 1), (isolated, 0))
              for j, i in np.argwhere(positions)]
    if not signed:
        return rle, rle, counts

    def priority(position):
        row, col, value = position
        raw = f"{RULE_DOMAIN}|{image_id}|{mapped_rank}|{row}|{col}|{value}".encode("ascii")
        return sha256(raw).digest(), row, col, value

    selected = sorted(signed, key=priority)[:cap]
    counts["add"] = sum(point[2] for point in selected)
    counts["delete"] = len(selected) - counts["add"]
    # Sham samples the same number of positions from each original boundary side.
    support = {1: ~foreground & (neighbors > 0), 0: foreground & (neighbors < 8)}
    placebo = []
    for value, count in ((1, counts["add"]), (0, counts["delete"])):
        if not count:
            continue
        pixels = np.flatnonzero(support[value])
        require(len(pixels) >= count, "Sham support must contain all rule positions")
        domain = f"{SHAM_DOMAIN}|{image_id}|{mapped_rank}|{value}".encode("ascii")
        rng = np.random.default_rng(int.from_bytes(sha256(domain).digest()[:16], "big"))
        for index in rng.choice(pixels, size=count, replace=False):
            row, col = divmod(int(index), pw)
            placebo.append((row+top, col+left, value))

    def materialize(points):
        for row, col, value in points:
            require(bitmap[row, col] == 1-value, "Each point must flip the original side exactly once")
            bitmap[row, col] = value
        encoded = encode_mask(bitmap)
        for row, col, value in points:
            bitmap[row, col] = 1-value
        require(int(mask.area(encoded)) == area+counts["add"]-counts["delete"] > 0,
                "Original cap, sign totals, and a nonempty output must be preserved")
        return encoded

    return materialize(selected), materialize(placebo), counts


def build_predictions(original):
    """Transform prediction rows before any scorer is called, without a GT argument."""
    variants = {name: [] for name in VARIANTS}
    totals = Counter(mapped_rows=0, empty_rows=0, cap_zero_rows=0, rows_with_edits=0, add=0, delete=0,
                     total_cap=0, sham_same_rle_as_morphology_on_edited_rows=0)
    ordinals = Counter()
    for row in original:
        image = row["image_id"]
        ordinals[image] += 1
        conservative, sham, c = edit_rle(row["segmentation"], image, ordinals[image])
        variants["identity"].append(row)
        variants["morphology"].append(row if conservative is row["segmentation"] else dict(row, segmentation=conservative))
        variants["sham"].append(row if sham is row["segmentation"] else dict(row, segmentation=sham))
        changed = c["add"]+c["delete"]
        totals.update(mapped_rows=1, empty_rows=int(c["area"] == 0), cap_zero_rows=int(c["cap"] == 0),
                      rows_with_edits=int(changed > 0), add=c["add"], delete=c["delete"], total_cap=c["cap"],
                      sham_same_rle_as_morphology_on_edited_rows=int(changed > 0 and conservative == sham))
    return {name: tuple(rows) for name, rows in variants.items()}, dict(totals)


def exact_pass_090(rle, target):
    intersection = int(mask.area(mask.merge([rle, target], intersect=True)))
    union = int(mask.area(rle)) + int(mask.area(target)) - intersection
    return union > 0 and 10*intersection >= 9*union


def strict_fixed_assignments(gt, variants, identity_evals, matcher):
    """Re-read fixed original COCO IoU .50 assignments without edited rematching."""
    original = variants["identity"]
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    annotation_images = {aid: row["image_id"] for aid, row in coco.anns.items()}
    assigned = matcher.match_official(identity_evals, original, annotation_images)
    ordinal = {id(row["segmentation"]): index for index, row in enumerate(original)}
    require(len(ordinal) == len(original), "One original RLE object per closed row required")
    result = {name: Counter(original_strict=0, recovered=0, lost=0, kept=0, still_below=0) for name in VARIANTS}
    for annotation_id, prediction in assigned.items():
        index = ordinal[id(prediction)]
        target = coco.annToRLE(coco.anns[annotation_id])
        baseline = exact_pass_090(prediction, target)
        for name in VARIANTS:
            strict = exact_pass_090(variants[name][index]["segmentation"], target)
            result[name].update(original_strict=int(baseline), recovered=int(strict and not baseline),
                                lost=int(baseline and not strict), kept=int(baseline and strict),
                                still_below=int(not baseline and not strict))
    require(all(r["original_strict"] == r["lost"]+r["kept"]
                and len(assigned) == r["kept"]+r["lost"]+r["recovered"]+r["still_below"] for r in result.values()),
            "Exact fixed strict match populations must reconcile")
    return dict(original_050_matches=len(assigned), variants={name: dict(r, net=r["recovered"]-r["lost"])
                                                              for name, r in result.items()})


def load_p311(code_root):
    source = code_root / P311
    require(source.is_file() and source.resolve(strict=True) == source, "Canonical released P311 source required")
    raw = source.read_bytes()
    require(sha256(raw).hexdigest() == P311_SHA, "Exact immutable released P311 source required")
    module = ModuleType("_p314_p311")
    module.__file__ = source.as_posix()
    exec(compile(raw, str(source), "exec"), module.__dict__)
    return module


def run(args):
    start = perf_counter()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and version("numpy") == "1.26.4"
            and version("pycocotools") == "2.0.11", "Exact closed CPU COCO runtime required")
    require(args.code_root.is_absolute() and args.code_root.resolve(strict=True) == args.code_root
            and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Canonical code and fresh private output required")
    p311 = load_p311(args.code_root)
    require((args.pair_sha256, args.training_protocol, args.accepted_result_sha256)
            == (p311.PAIR_SHA, p311.TRAINING_PROTOCOL, p311.PRIMARY_SHA),
            "Exact previously accepted common-support seed 5202 lineage required")
    closed = p311.load_closed(args.code_root)
    driver = closed.load_gate(args.code_root).load_driver()
    primary = driver.read_json(args.accepted_result, args.accepted_result_sha256)
    require(primary["status"] == p311.PRIMARY_STATUS and primary["images"] == 705 and primary["primary_annotations"] == 1782
            and primary["official_TEST_opened"] is False and primary["training_protocol"] == p311.TRAINING_PROTOCOL
            and primary["training_seed"] == 5202, "Exact prior independent common-fit DEV705 score acceptance required")
    gt, native, center, provenance = closed.verified_inputs(
        args.code_root, args.pair, args.pair_sha256, native_arm_receipt=args.native_arm_receipt,
        center_arm_receipt=args.center_arm_receipt, training_protocol=args.training_protocol)
    del native
    require(provenance["training_pair_receipt_sha256"] == primary["independent_training_pair_sha256"]
            and provenance["evaluation_protocol"] == primary["evaluation_protocol"]
            and provenance["original_dev_gt_sha256"] == primary["original_dev_gt_sha256"],
            "Exact common-center score lineage required")

    variants, work = build_predictions(center)
    frozen = {name: hash_rows(rows) for name, rows in variants.items()}
    del center
    print(json.dumps(dict(event="P314_PREDICTION_VARIANTS_FROZEN", variant_prediction_sha256s=frozen)), flush=True)
    build_seconds = perf_counter()-start
    diagnostics = closed.load_diagnostics(args.code_root)
    score = diagnostics["dev705_score_template"]
    matcher = diagnostics["dev705_paired_mask_geometry"]
    adapter = diagnostics["dev705_group_sensitivity"].load_official_adapter()
    image_ids = [row["id"] for row in gt["images"]]
    metrics, identity_evals, identity_ranks = {}, None, None
    for name in VARIANTS:
        values, evaluations, ranks = score.evaluate_cell(gt, list(variants[name]), image_ids, adapter)
        metrics[name] = values
        if name == "identity":
            identity_evals, identity_ranks = evaluations, ranks
            expected = primary["scores"]["center"]["primary"]
            require(all(math.isclose(values[key], expected[key], rel_tol=0, abs_tol=1e-12)
                        for key in ("AP", "AP75", "AP90", "APs")), "Original center COCO scores must reproduce")
        else:
            require(ranks == identity_ranks, "Original image, scores and detection rank must remain unchanged")
    require(frozen == {name: hash_rows(rows) for name, rows in variants.items()}, "Frozen scored rows changed")
    matched = strict_fixed_assignments(gt, variants, identity_evals, matcher)
    report = dict(
        status=STATUS, accepted_result_sha256=p311.PRIMARY_SHA, provenance=provenance, arm="center",
        images=len(image_ids), annotations=len(gt["annotations"]), closed_predictions_only=True,
        COCO_AP=True, official_TEST_opened=False, model_forward_calls=0, training=False,
        source_sha256={"strict_residual.py": P311_SHA, "morphology_baseline.py": sha256(Path(__file__).read_bytes()).hexdigest()},
        policy="fill background with 8 foreground neighbors; remove foreground with zero foreground neighbors",
        cap="min(16,floor(original predicted foreground pixels/10)); zero padding at image borders",
        sham="fixed keyed sampling within the same original one-pixel side, with exact rule add/delete counts",
        gt_used_for_prediction_edits=False, prediction_streams_frozen_before_coco_scoring=True,
        original_rows_edited=work, prediction_sha256s=frozen, metrics=metrics,
        deltas_vs_identity_points={name: {key: 100*(metrics[name][key]-metrics["identity"][key]) for key in ("AP", "AP75", "AP90", "APs")}
                                   for name in ("morphology", "sham")},
        strict_on_fixed_original_matches=matched, build_including_admission_seconds=build_seconds,
        total_seconds=perf_counter()-start,
    )
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, bytes=len(raw), sha256=sha256(raw).hexdigest())), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "pair", "native-arm-receipt", "center-arm-receipt", "accepted-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("pair-sha256", "training-protocol", "accepted-result-sha256"):
        parser.add_argument("--" + name, required=True)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
