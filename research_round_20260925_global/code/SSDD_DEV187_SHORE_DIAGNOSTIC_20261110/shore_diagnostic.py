"""Frozen SSDD DEV187 shore-stratified diagnostics on closed P250 outputs."""
import argparse
from collections import Counter
from hashlib import sha1, sha256
from io import BytesIO
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
from types import ModuleType
from zipfile import ZipFile


SOURCE_COMMIT = "e9b3ffd35c72067206af95239447aebbdbca3c77"
SOURCE_BLOB = "14ae89d4793b2a00a8d72443fc4c38e750327a63"
SOURCE_SHA = "da66b1807ea91a1f593cd62d4987f9d2ce4b410ede8ea97e231ad6f5f9455989"
IMAGE_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
RANK_SHA = "162eb595d9a3356c868ab4c94229622faf3f85dfc08d0d9a7e6da4ff82ffd252"
PRIOR_RANK_SHA = "f51f68cf67ce36e4105999c40f842bdf9c0b8a7deebb291f4af6d55ecb6f488e"
ARMS = ("native", "center")
STRATA = ("all_water", "land_containing")
ZONES = ("land_interior", "shore_4px", "water_interior")
SAFE_SCOPE = dict(DEV=True, TEST=False, official_TEST_opened=False, training=False, model_forward_calls=0)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def stable(path, expected, limit):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
         and 0 < path.stat().st_size <= limit, "Exact bounded canonical input required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
    need(0 < len(raw) == before.st_size <= limit and sha256(raw).hexdigest() == expected
         and identity(before) == identity(after) == identity(path.stat()), "Exact stable original bytes required")
    return raw


def zones(land):
    import numpy as np
    from scipy.ndimage import binary_dilation

    need(land.ndim == 2 and land.dtype == np.bool_, "Exact original 2D sea–land grid required")
    footprint = np.ones((3, 3), dtype=bool)
    near_sea = binary_dilation(~land, structure=footprint, iterations=4, border_value=0)
    near_land = binary_dilation(land, structure=footprint, iterations=4, border_value=0)
    shore = (land & near_sea) | (~land & near_land)
    return np.where(shore, 1, np.where(land, 0, 2)).astype(np.uint8)


def mask_population(archive, items):
    import numpy as np
    from PIL import Image

    original = stable(archive, SOURCE_SHA, 542554)
    header = ("blob " + str(len(original)) + "\0").encode("ascii")
    need(len(original) == 542554 and sha1(header + original).hexdigest() == SOURCE_BLOB,
         "Exact author Git blob required")
    grids, counts = {}, Counter()
    with ZipFile(BytesIO(original)) as source:
        names = source.namelist()
        need(len(names) == len(set(names)) == 1163, "Exact author archive roster")
        paths = []
        for item in items:
            name = PureWindowsPath(item["image_ref"]["path"]).stem
            need(re.fullmatch(r"[0-9]{6}", name) is not None, "Expected direct original SSDD basename")
            path = "SL-SSDD/train/" + name + ".png"
            need(path in names, "Every DEV image requires an exact author train-only mask")
            paths.append(path)
        need(len(paths) == len(set(paths)) == len(items) == 187, "Exact unique DEV187 name scope required")
        for item, path in zip(items, paths, strict=True):
            member = source.getinfo(path)
            need(0 < member.file_size <= 512 * 1024, "Unexpected original sea–land PNG size")
            with Image.open(BytesIO(source.read(member))) as image:
                need(image.format == "PNG" and image.mode == "L"
                     and image.size == tuple(item["original_hw"][::-1]), "Exact 8-bit original-grid PNG required")
                data = np.asarray(image)
                need(data.dtype == np.uint8 and ((data == 0) | (data == 255)).all(), "Exact binary author sea–land values")
            land = data == 0
            group = "land_containing" if land.any() else "all_water"
            counts[group] += 1
            counts["all_land"] += int(land.all())
            grids[item["image_id"]] = group, zones(land)
    return grids, dict(counts)


def bbox_independent_fp(records, predictions, grids):
    import numpy as np
    from pycocotools import mask

    masses, false_positives, evaluated = Counter(), 0, 0
    by_image = {image_id: 0 for image_id in grids}
    total_pixels = sum(grid.size for _, grid in grids.values())
    available = Counter({zone: sum(int(np.count_nonzero(grid == i)) for _, grid in grids.values())
                         for i, zone in enumerate(ZONES)})
    regions = {}
    for row in records[:len(grids)]:
        if row is None:
            continue
        iid = row["image_id"]
        grid = grids[iid][1]
        for result_id, matched, ignored in zip(row["dtIds"], row["dtMatches"][0], row["dtIgnore"][0], strict=True):
            prediction = predictions[result_id - 1]
            need(prediction["image_id"] == iid, "Original COCO result index and image differ")
            evaluated += 1
            if matched or ignored:
                continue
            rle = prediction["segmentation"]
            if iid not in regions:
                regions[iid] = [mask.encode(np.asfortranarray(grid == i, dtype=np.uint8)) for i in range(len(ZONES))]
            values = [int(mask.area(mask.merge([rle, region], intersect=True))) for region in regions[iid]]
            need(sum(values) == int(mask.area(rle)) > 0, "Each nonempty FP mask must partition exactly")
            masses.update(dict(zip(ZONES, values, strict=True)))
            false_positives += 1
            by_image[iid] += sum(values)
    need(evaluated == len(predictions), "Exact all-area equal-prefix COCO rows required")
    total = masses.total()
    return dict(false_positive_masks=false_positives, evaluated_predictions=evaluated,
                pixels_total=total, pixel_mass_by_zone={zone: masses[zone] for zone in ZONES},
                available_pixels_by_zone={zone: available[zone] for zone in ZONES},
                mass_by_zone_over_available_pixels={zone: masses[zone] / available[zone] if available[zone] else None
                                                    for zone in ZONES},
                false_positive_pixel_mass_over_all_pixels=total / total_pixels,
                image_mean_false_positive_pixel_mass_over_image_pixels=sum(
                    pixels / grids[iid][1].size for iid, pixels in by_image.items()) / len(grids))


def evaluate_stratum(gt, rows, selected, grids, support, rank):
    _, template, coverage, _, official = support
    ids = [item["id"] for item in gt["images"]]
    metrics, burdens = {}, {}
    for mode in ARMS:
        metrics[mode], _, _ = template.evaluate_cell(gt, rows[mode], ids, official)
        _, records, _ = template.evaluate_cell(gt, selected[mode], ids, official)
        burdens[mode] = bbox_independent_fp(records, selected[mode], grids)
    return dict(images=len(ids), annotations=len(gt["annotations"]), raw_arm_metrics=metrics,
                center_minus_native_points=template.differences(metrics["native"], metrics["center"]),
                score_blind_coverage=rank.blind_coverage(gt, selected, coverage), false_positive_burden=burdens)


def closed_analysis(args, grids, census):
    source = args.code / "SSDD_DEV187_POST_PRIMARY_RANK_20261110/p250_rank_diagnostic.py"
    rank = ModuleType("_exact_p255")
    rank.__file__ = str(source)
    exec(compile(stable(source, RANK_SHA, 65536), str(source), "exec"), rank.__dict__)
    support = rank.load_support(args.code)
    p250, template, _, scoring, official = support
    plan = p250.document(stable(args.plan, rank.PLAN_SHA, p250.MIB))
    accepted = p250.document(stable(args.accepted_score, rank.ACCEPTED_SHA, p250.MIB))
    prior = p250.document(stable(args.rank_score, PRIOR_RANK_SHA, p250.MIB))
    need(plan["status"] == p250.PLAN_STATUS and plan["evaluation_protocol"] == rank.PROTOCOL
         and accepted["closed_plan_sha256"] == rank.PLAN_SHA and accepted["image_manifest_sha256"] == IMAGE_SHA
         and plan["image_manifest"]["sha256"] == IMAGE_SHA and plan["integer_gt"]["sha256"] == p250.INTEGER_GT_SHA
         and prior["closed_plan_sha256"] == rank.PLAN_SHA, "Exact P250 integer DEV and P255 prior required")
    refs, arms = rank.bound_prediction_refs(p250, plan, accepted)
    manifest = plan["image_manifest"]
    _, ids, items = scoring.fixed_images(scoring.Artifact(manifest, p250.read_ref(manifest, p250.MIB)))
    need(list(grids) == ids, "Masks and original closed DEV manifest must have exact logical order")
    gt = p250.document(p250.read_ref(plan["integer_gt"], p250.MIB))
    scoring.validate_gt(gt, ids, items)
    by_seed = {}
    for seed in p250.TRAIN_PROTOCOLS:
        rows = {}
        for mode in ARMS:
            key = seed, mode
            arm = arms[key]
            report = dict(mapped_rows=arm["mapped_rows"], unmapped_slot1_rows=18700-arm["mapped_rows"],
                          empty_image_predictions=arm["empty_image_predictions"])
            rows[mode] = p250.score_rows(p250.read_ref(refs[key], 64 * p250.MIB), items, report)
            primary = scoring.score_one(official, gt, rows[mode], ids)
            need(all(abs(primary[field] - arm["metrics"][field]) < 1e-12 for field in p250.METRICS),
                 "Recomputed original all-DEV primary score must reproduce independently accepted P250")
        selected, counts = template.prefix_rows(*(template.nonempty_by_image(rows[mode], gt["images"]) for mode in ARMS))
        need(sum(counts) == prior["by_seed"][str(seed)]["prefix_counts"]["retained_per_arm"],
             "Exact P255 equal-prefix population required")
        result = {}
        for name in STRATA:
            local = {iid: entry for iid, entry in grids.items() if entry[0] == name}
            need(bool(local), "Both predeclared strata must exist")
            subset = dict(gt, images=[row for row in gt["images"] if row["id"] in local],
                          annotations=[row for row in gt["annotations"] if row["image_id"] in local])
            full_rows = {mode: [row for row in rows[mode] if row["image_id"] in local] for mode in ARMS}
            prefix_rows = {mode: [row for row in selected[mode] if row["image_id"] in local] for mode in ARMS}
            result[name] = evaluate_stratum(subset, full_rows, prefix_rows, local, support, rank)
        for mode in ARMS:
            for threshold in ("0.75", "0.90"):
                count = sum(row["score_blind_coverage"]["arms"][mode][threshold]["maximum_one_to_one"]
                            for row in result.values())
                need(count == prior["by_seed"][str(seed)]["score_blind_coverage"]["arms"][mode][threshold]["maximum_one_to_one"],
                     "Stratum coverage must add back to exact P255 all-DEV match count")
        by_seed[str(seed)] = result
    return dict(status="SLSSDD_P250_DEV187_SHORE_DIAGNOSTIC_COMPLETE_V1", by_seed=by_seed, AP=True,
                primary_score_reproduced=True, p250_accepted_sha256=rank.ACCEPTED_SHA, p250_plan_sha256=rank.PLAN_SHA,
                p255_result_sha256=PRIOR_RANK_SHA, integer_gt_sha256=p250.INTEGER_GT_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("source-zip", "image-manifest", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("code", "plan", "accepted-score", "rank-score"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
         and sys.dont_write_bytecode, "Offline isolated CPU runtime required")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("Offline CPU diagnostic required")
    sys.addaudithook(offline)
    manifest = json.loads(stable(args.image_manifest, IMAGE_SHA, 1024**2))
    need(manifest["TEST"] is False and manifest["expected_ids"] == [row["image_id"] for row in manifest["items"]],
         "Exact original DEV-only ordered manifest required")
    grids, census = mask_population(args.source_zip, manifest["items"])
    scope = dict(author_commit=SOURCE_COMMIT, author_git_blob=SOURCE_BLOB, author_zip_sha256=SOURCE_SHA,
                 image_manifest_sha256=IMAGE_SHA, images=len(grids), strata_census=census,
                 author_TRAIN_mask_members_read=len(grids), author_TEST_mask_members_read=0,
                 shore_radius_original_pixels=4, shore_grid_distance="Chebyshev", **SAFE_SCOPE)
    controls = (args.code, args.plan, args.accepted_score, args.rank_score)
    if any(controls):
        need(all(controls), "All four closed P250/P255 scoring inputs required")
        result = closed_analysis(args, grids, census)
    else:
        result = dict(status="SLSSDD_DEV187_AUTHOR_MASK_QUALIFICATION_COMPLETE_V1", AP=False)
    raw = (json.dumps(dict(result, **scope), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent,
         "Fresh canonical output required")
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
