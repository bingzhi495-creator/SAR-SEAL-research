"""Read-only replay of four frozen HRSID FIT views for polygon-first geometry."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZIP_STORED, ZipFile


BUNDLE_SHA = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
BUNDLE_BYTES = 151_320_868
COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
SUBSET_SHA = "6f25f80ed0e0b9c3f0c57ab6cc3b25e947a102353757a33bca4d3823df3d547d"
HISTORY_SHA = "318408a43f2c107b329638b8be3132717f7d366b046e98aa2190318f3aaca053"
BASE_SEED, REPEATS = 520198000, 4
REPLAY = "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
AUDIT = "HRSID_PNG_FIT_NATIVE_AUG_AUDIT_20261109/audit_hrsid_fit_native_aug.py"
SUPPORT_SOURCES = {
    f"{REPLAY}/paired_fit_transform.py": "64f3f63c6ac89b66578f0ada42fa3eb8b4db3b2f1ae2ea9113debe6ab9b270e0",
    f"{REPLAY}/common_mask_supervision.py": "59ea82e24353eeb4a14115058d5115333a4e7ea517a0abef581bee137a9a3a57",
    f"{REPLAY}/one_pass_fit_transform.py": "05cad566f9b5cc761e91b5283585a5392a7450c42c3133349e6dc21e8bf7496d",
    f"{REPLAY}/reference_resize.py": "3ca1d4a6400324c13c777cb4d5f7627713bc625580195d8e05363950872cc8df",
    f"{REPLAY}/replay_fit.py": "4f95f1b5b84fcf89b005f999cfb8c5fd647a7b262aedd2622860cf8785a14157",
    f"{REPLAY}/trace_resize.py": "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f",
    "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py":
        "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5",
    "SSDD_MASK_PHASE_TRANSFER_20261110/ssdd_pixel_footprint.py":
        "cd688be9937a2c3429eafb540157b1b5b99d319c7236cfe3a5891e25fd7e7913",
    AUDIT: "6d1b1b94c14cc8f6c3cf7d5eb70ea8db83a6c0f4c0219c68b4dbf24d773aca14",
}
ARMS = ("stock", "stepwise_center", "one_pass_center", "vector_first")
REFERENCES = ("vector_raster", "continuous_vector", "continuous_source_pixel_footprints")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def fingerprint(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def pinned_member(bundle, name, expected, limit):
    item = bundle.getinfo(name)
    require(item.compress_type == ZIP_STORED and 0 < item.file_size <= limit, "Exact bounded FIT member required")
    raw = bundle.read(item)
    require(len(raw) == item.file_size and digest(raw) == expected, "Sealed FIT member bytes differ")
    return raw


def install_support(code, shapely_site):
    import numpy as np
    import torch

    require(code.resolve(strict=True) == code and shapely_site.resolve(strict=True) == shapely_site,
            "Canonical code and dependency directories required")
    for name, expected in SUPPORT_SOURCES.items():
        source = code / name
        require(source.resolve(strict=True) == source and digest(source.read_bytes()) == expected,
                "Pinned ordinary geometry dependency differs: " + source.name)
    sys.path[:0] = list(dict.fromkeys(str((code / name).parent) for name in SUPPORT_SOURCES))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.append(str(shapely_site))
    import shapely

    require(Path(shapely.__file__).resolve().is_relative_to(shapely_site)
            and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
            "Exact Shapely and GEOS versions required")
    module = ModuleType("_p251_pinned_stock_fit_audit")
    audit_path = code / AUDIT
    module.__file__ = str(audit_path)
    source = audit_path.read_bytes()
    require(digest(source) == SUPPORT_SOURCES[AUDIT], "Pinned FIT audit source changed")
    exec(compile(source, str(audit_path), "exec"), module.__dict__)
    sys.addaudithook(module.offline)
    require(all(module.version(name) == expected for name, expected in module.VERSIONS.items())
            and all(digest(Path(mod.__file__).read_bytes()) == expected for mod, expected in module.SOURCE_SHA.items()),
            "Pinned Windows RF-DETR runtime differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "GPU must remain hidden")
    return module, np, torch, shapely


def source_annotations(rows, target, hw, torch):
    """Rebuild the stock source converter's stable annotation-row identity."""
    height, width = hw
    rows = [row for row in rows if row.get("iscrowd", 0) == 0]
    boxes = torch.as_tensor([row["bbox"] for row in rows], dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=width)
    boxes[:, 1::2].clamp_(min=0, max=height)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    admitted = [row for row, retain in zip(rows, keep.tolist(), strict=True) if retain]
    require(torch.equal(boxes[keep], target["boxes"]) and len(admitted) == len(target["masks"])
            and all(row["category_id"] == 1 for row in admitted)
            and torch.equal(target["labels"], torch.zeros(len(admitted), dtype=torch.int64))
            and torch.equal(target["area"], torch.tensor([row["area"] for row in admitted], dtype=torch.float32)),
            "Exact ordered source annotation-row membership required")
    return admitted


def compare_view(native, audit, image, targets, source_rows, old, torch, np):
    from common_mask_supervision import ROW_ID, common_targets
    from one_pass_fit_transform import substitute_one_pass_masks
    from paired_fit_transform import _check_output, _prepare, _rng, _rng_bytes
    from reference_resize import _compose
    from replay_fit import normalized_trace
    from ssdd_pixel_footprint import measure_masks
    from trace_resize import capture_transform
    from vector_label_geometry import UnsupportedPolygon, binary_iou, final_polygons, polygon_raster, vector_footprint_metrics

    original = _prepare(image, targets)
    results, images, traces, after_rng = [], [], [], []
    for mode in ("native", "center"):
        native.set_seed(old["seed"])
        with capture_transform(native._torchvision, mode=mode) as trace:
            pixels, target = audit.transform(image.copy(), {name: value.clone() for name, value in original.items()})
        _check_output(pixels, target, original)
        images.append(pixels)
        results.append(target)
        traces.append(trace)
        after_rng.append(_rng_bytes(_rng()))
    a, b = results
    trace = traces[0]
    steps = normalized_trace(trace)
    a_live, b_live = (set(t[ROW_ID][~native.mask_empty(t)].tolist()) for t in results)
    legacy = old["rows"]
    require(traces[0] == traces[1] and torch.equal(images[0], images[1]) and after_rng[0] == after_rng[1]
            and old["steps"] == steps and old["flip"] == any(row["op"] == "hflip" and row["applied"] for row in trace)
            and (legacy["raw_rows"], legacy["legacy_present"], legacy["center_present"], legacy["legacy_nonempty"],
                 legacy["center_nonempty"], legacy["common_nonempty"])
            == (len(original[ROW_ID]), len(a[ROW_ID]), len(b[ROW_ID]), len(a_live), len(b_live), len(a_live & b_live)),
            "Exact closed historical view, geometry, rows and CPU RNG required")
    c = substitute_one_pass_masks(original["masks"], b, trace)
    a, b, c = common_targets(a, b, c)
    common_ids = a[ROW_ID].tolist()
    require(common_ids == b[ROW_ID].tolist() == c[ROW_ID].tolist(), "Fixed common annotation row order")
    require(old["path"] != "direct" or torch.equal(b["masks"], c["masks"]), "Direct center mask exactness")
    image_hw = (image.height, image.width)
    original_rows = source_annotations(source_rows, original, image_hw, torch)
    records, skipped = [], Counter()
    for position, row_id in enumerate(common_ids):
        annotation = original_rows[row_id]
        segmentation = annotation["segmentation"]
        try:
            parts, hw = final_polygons(segmentation, image_hw, steps)
        except UnsupportedPolygon as reason:
            skipped[str(reason)] += 1
            continue
        require(hw == tuple(b["size"].tolist()), "Vector and native final canvases differ")
        raw_mask = original["masks"][row_id].numpy()
        from vector_label_geometry import polygon_rings
        require(np.array_equal(raw_mask, polygon_raster(polygon_rings(segmentation), image_hw)),
                "Original COCO polygon must reproduce the actual prepared input mask")
        masks = {name: target["masks"][position].numpy() for name, target in zip(ARMS, (a, b, c))}
        masks[ARMS[3]] = polygon_raster(parts, hw)
        vector_measure = vector_footprint_metrics(parts, hw, masks)
        small = float(annotation["area"]) < 1024
        bitmap_measure = measure_masks(raw_mask[None], {k: v[None] for k, v in masks.items()},
                                       np.asarray([small], dtype=np.bool_), steps, _compose)
        vector_raster = masks[ARMS[3]]
        records.append(dict(
            small=small, vector_empty=not bool(vector_raster.any()),
            vector_polygon_area=vector_measure["visible_polygon_area"],
            labels_changed={name: int(np.count_nonzero(mask != vector_raster)) for name, mask in masks.items() if name != ARMS[3]},
            vector_raster={name: binary_iou(mask, vector_raster) for name, mask in masks.items() if name != ARMS[3]},
            continuous_vector={name: value["iou"] for name, value in vector_measure["by_arm"].items()},
            continuous_source_pixel_footprints={
                name: float(value["iou"][0]) if math.isfinite(value["iou"][0]) else None
                for name, value in bitmap_measure["by_arm"].items()
            },
        ))
    return records, dict(common=len(common_ids), polygon_eligible=len(records), skipped=dict(skipped),
                         stock_and_stepwise_common=len(a_live & b_live))


def summary(rows):
    result = dict(polygon_rows=len(rows), vector_empty=sum(row["vector_empty"] for row in rows))
    for reference in REFERENCES:
        arms = ARMS[:-1] if reference == "vector_raster" else ARMS
        by_arm = {}
        for arm in arms:
            values = [row[reference][arm] for row in rows if row[reference][arm] is not None]
            by_arm[arm] = dict(mean_iou=fmean(values) if values else None, rows=len(values))
        comparisons = {}
        for new, previous in ((ARMS[2], ARMS[1]), (ARMS[3], ARMS[2])):
            if new not in arms:
                continue
            values = [row[reference][new] - row[reference][previous] for row in rows
                      if row[reference][new] is not None and row[reference][previous] is not None]
            comparisons[new + "_minus_" + previous] = dict(
                mean_iou_delta=fmean(values) if values else None,
                wins=sum(value > 1e-10 for value in values), losses=sum(value < -1e-10 for value in values),
                ties=sum(abs(value) <= 1e-10 for value in values),
            )
        result[reference] = dict(arms=by_arm, comparisons=comparisons)
    return result


def run(args):
    started = perf_counter()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and sys.version_info[:3] == (3, 10, 13), "Exact isolated Windows CPU scope required")
    native, np, torch, shapely = install_support(args.code, args.shapely_site)
    audit = native.NativeInputAudit()
    bundle_path, output = args.bundle, args.output
    require(bundle_path.is_absolute() and bundle_path.resolve(strict=True) == bundle_path
            and output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Exact closed FIT input and fresh output required")
    records, views, unique_images = defaultdict(list), [], set()
    with bundle_path.open("rb") as stream:
        before = fingerprint(os.fstat(stream.fileno()))
        require(before[2] == BUNDLE_BYTES, "Exact previously closed FIT114 archive length required")
        identity = sha256()
        for chunk in iter(lambda: stream.read(1024**2), b""):
            identity.update(chunk)
        require(identity.hexdigest() == BUNDLE_SHA, "Exact previously closed FIT114 archive bytes required")
        stream.seek(0)
        with ZipFile(stream) as bundle:
            cohort = json.loads(pinned_member(bundle, "COHORT_PRIVATE.json", COHORT_SHA, 1024**2))
            index = json.loads(pinned_member(bundle, "FIT114_COCO_PRIVATE.json", SUBSET_SHA, 4 * 1024**2))
            history = [json.loads(line) for line in pinned_member(
                bundle, "HISTORICAL_VIEWS_PRIVATE.jsonl", HISTORY_SHA, 1024**2).splitlines()]
            images = {row["id"]: row["file_name"] for row in index["images"]}
            annotations = defaultdict(list)
            for row in index["annotations"]:
                require(row["image_id"] in images, "Only frozen FIT annotations allowed")
                annotations[row["image_id"]].append(row)
            members = {"COHORT_PRIVATE.json", "FIT114_COCO_PRIVATE.json", "HISTORICAL_VIEWS_PRIVATE.jsonl",
                       *("images/" + row["native_basename"] for row in cohort["images"])}
            require(cohort["fit_only"] is True and len(cohort["images"]) == len(images) == 114
                    and len(index["annotations"]) == 1109 and len(history) == 456
                    and len(bundle.namelist()) == len(members) and set(bundle.namelist()) == members,
                    "Exact frozen FIT-only population and file membership")
            for path in ("direct", "crop"):
                cursors = [i for i, item in enumerate(history) if item["layer"] == "core" and item["path"] == path
                           and item["rows"]["center_nonempty"] > 0][:2]
                require(len(cursors) == 2, "Two preselected historical core views per path required")
                for cursor in cursors:
                    old = history[cursor]
                    ordinal = cursor // REPEATS
                    row = cohort["images"][ordinal]
                    image_id, name = row["image_id"], row["native_basename"]
                    require(row["core"] is True and old["ordinal"] == ordinal and old["seed"] == BASE_SEED + cursor
                            and old["group"] == row["group"] and images[image_id] == name,
                            "Exact preselected historical source view identity")
                    pixels = pinned_member(bundle, "images/" + name, row["encoded_sha256"], 4 * 1024**2)
                    image, scales = native.io_utils.decode_pil_image_bytes(pixels, draft_size=None)
                    require(len(pixels) == row["encoded_bytes"] and image.size == (800, 800) and image.mode == "RGB"
                            and scales == (1., 1.) and digest(np.asarray(image).tobytes()) == row["decoded_rgb_sha256"],
                            "Exact preselected original decoded pixels")
                    image, target = audit.convert(image, dict(image_id=image_id, annotations=annotations[image_id]))
                    native.target_summary(target, 800, image_id, normalized=False)
                    scores, counts = compare_view(native, audit, image, target, annotations[image_id], old, torch, np)
                    records[path].extend(scores)
                    views.append(dict(path=path, historical_cursor=cursor, flip=old["flip"], **counts))
                    unique_images.add(image_id)
        require(fingerprint(os.fstat(stream.fileno())) == fingerprint(bundle_path.stat()) == before,
                "Closed FIT-only archive changed during the replay")
    require(not torch.cuda.is_initialized() and torch.get_num_threads() == 1
            and len(views) == 4 and all(records[path] for path in ("direct", "crop")),
            "Exact four-view, single-thread CPU replay")
    result = dict(
        status="HRSID_FIT114_VECTOR_FIRST_FOUR_VIEW_GEOMETRY_COMPLETE",
        selection="first_two_core_B_nonempty_views_per_historical_path",
        fit_bundle_sha256=BUNDLE_SHA, cohort_sha256=COHORT_SHA, subset_sha256=SUBSET_SHA, history_sha256=HISTORY_SHA,
        images=len(unique_images), views=views, source_sha256=SUPPORT_SOURCES,
        vector_helper_sha256=digest(Path(__file__).with_name("vector_label_geometry.py").read_bytes()),
        runner_sha256=digest(Path(__file__).read_bytes()),
        shapely_version=shapely.__version__, geos_version=shapely.geos_version_string,
        shapely_extension_sha256=digest(Path(shapely.lib.__file__).read_bytes()),
        metrics={path: dict(all=summary(rows), small=summary([row for row in rows if row["small"]]))
                 for path, rows in records.items()},
        FIT=True, DEV=False, TEST=False, AP=False, model_forward_calls=0, optimizer_updates=0,
        wall_seconds=perf_counter() - started,
    )
    output.mkdir()
    with (output / "RESULT_SAFE.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=digest((output / "RESULT_SAFE.json").read_bytes()),
                          wall_seconds=result["wall_seconds"]), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("bundle", "code", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
