"""Replay fixed HRSID FIT large-bearing images without model or DEV access."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
LAYOUT_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
BASE_SEED = 520198000
REPEATS = 4
AREA_THRESHOLD = 9216
SOURCES = {
    "fit_query_export": ("HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py",
                         "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5"),
    "reference_resize": ("HRSID_FIT_RESIZE_PHASE_REPLAY_20261109/reference_resize.py",
                         "3ca1d4a6400324c13c777cb4d5f7627713bc625580195d8e05363950872cc8df"),
    "trace_resize": ("HRSID_FIT_RESIZE_PHASE_REPLAY_20261109/trace_resize.py",
                     "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f"),
    "replay_fit": ("HRSID_FIT_RESIZE_PHASE_REPLAY_20261109/replay_fit.py",
                   "4f95f1b5b84fcf89b005f999cfb8c5fd647a7b262aedd2622860cf8785a14157"),
    "ssdd_pixel_footprint": ("SSDD_MASK_PHASE_TRANSFER_20261110/ssdd_pixel_footprint.py",
                             "cd688be9937a2c3429eafb540157b1b5b99d319c7236cfe3a5891e25fd7e7913"),
    "native_audit": ("HRSID_PNG_FIT_NATIVE_AUG_AUDIT_20261109/audit_hrsid_fit_native_aug.py",
                     "6d1b1b94c14cc8f6c3cf7d5eb70ea8db83a6c0f4c0219c68b4dbf24d773aca14"),
}
ARMS = ("native", "center")
AREA_METRICS = ("signed_area", "fn", "fp")
METRICS = ("signed_area_fraction", "fn_fraction", "fp_fraction", "iou")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected, limit):
    require(path.is_absolute() and path.resolve(strict=True) == path, "Expected canonical closed path")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit and sha256(data).hexdigest() == expected, "Closed input pin differs")
    return data


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")


def view_seeds(ordinal):
    return tuple(BASE_SEED + REPEATS * ordinal + rep for rep in range(REPEATS))


def build_population(index, layout_images):
    """Preserve original FIT image/annotation order and select images by raw COCO area."""
    images = index["images"]
    layout = {row["image_id"]: row for row in layout_images}
    image_ids = {row["id"] for row in images}
    require(len(image_ids) == len(images) == len(layout_images) == len(layout) and image_ids == layout.keys(),
            "FIT index and native layout must have exactly the same images")
    annotations = defaultdict(list)
    ann_ids = set()
    for ann in index["annotations"]:
        require(ann["id"] not in ann_ids and ann["image_id"] in image_ids and ann["category_id"] == 1
                and ann["iscrowd"] == 0 and math.isfinite(ann["area"]) and ann["area"] > 0,
                "Expected unique valid native FIT annotation")
        ann_ids.add(ann["id"])
        annotations[ann["image_id"]].append(ann)

    selected = []
    for ordinal, image in enumerate(images):
        name = image["file_name"]
        row = layout[image["id"]]
        require(re.fullmatch(r"P\d+_[A-Za-z0-9_]+\.png", name) and row["native_basename"] == name
                and image["height"] == image["width"] == 800, "Native FIT PNG name or dimensions differ")
        anns = annotations[image["id"]]
        large = [i for i, ann in enumerate(anns) if ann["area"] >= AREA_THRESHOLD]
        if large:
            selected.append(dict(image=image, ordinal=ordinal, prefix=name.split("_", 1)[0],
                                 layout=row, annotations=anns, large_indices=large))
    return selected


def load_runtime(code_root):
    """Load only byte-pinned ordinary sources and require the original isolated Windows CPU runtime."""
    require(sys.platform == "win32" and sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated
            and sys.flags.utf8_mode and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Use original Windows CPython with -I -X utf8 -B and CUDA hidden")
    loaded = {}
    for name, (relative, digest) in SOURCES.items():
        path = code_root / relative
        body = pinned(path, digest, 128 * 1024)
        module = ModuleType(name)
        module.__file__ = str(path)
        require(name not in sys.modules, "Start this replay in a fresh interpreter")
        sys.modules[name] = module
        exec(compile(body, str(path), "exec"), module.__dict__)
        loaded[name] = module

    source = loaded["native_audit"]
    torch = source.torch
    require(not torch.cuda.is_available() and not torch.cuda.is_initialized()
            and all(source.version(package) == expected for package, expected in source.VERSIONS.items())
            and all(sha256(Path(m.__file__).read_bytes()).hexdigest() == digest
                    for m, digest in source.SOURCE_SHA.items())
            and loaded["fit_query_export"].INDEX_SHA == INDEX_SHA
            and loaded["fit_query_export"].LAYOUT_SHA == LAYOUT_SHA
            and loaded["replay_fit"].AUG_SHA == SOURCES["native_audit"][1],
            "Expected original native input audit package/source pins and CPU-only state")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.addaudithook(source.offline)
    return SimpleNamespace(source=source, audit=source.NativeInputAudit(), np=source.np,
                           replay=loaded["replay_fit"], compose=loaded["reference_resize"]._compose,
                           footprint=loaded["ssdd_pixel_footprint"].measure_masks)


def prepare_image(runtime, image, item):
    anns = item["annotations"]
    image_id = item["image"]["id"]
    im, source_target = runtime.audit.convert(image, dict(image_id=image_id, annotations=anns))
    runtime.source.target_summary(source_target, 800, image_id, normalized=False)
    torch = runtime.source.torch
    actual = source_target["area"]
    require(len(actual) == len(anns) and torch.equal(actual, torch.tensor([a["area"] for a in anns],
                                                                         dtype=actual.dtype))
            and not bool(runtime.source.mask_empty(source_target).any()),
            "Full ordered original annotation rows must survive native conversion")
    source_target[runtime.replay.ROW_ID] = torch.arange(len(anns), dtype=torch.int64)
    return im, source_target


def measure_view(runtime, item, image, target, seed):
    """Replay both full views and measure only preselected large A∩B rows."""
    np = runtime.np
    observed = []
    source = runtime.source

    def capture(row, *args, **kwargs):
        source.target_summary(row, *args, **kwargs)
        observed.append(row)

    context = SimpleNamespace(_torchvision=source._torchvision, set_seed=source.set_seed, target_summary=capture)
    result = runtime.replay.replay_view(runtime.audit, context, image, target, seed)
    require(len(observed) == 2, "Expected two independently validated native augmentation targets")
    id_key = runtime.replay.ROW_ID
    large = set(item["large_indices"])
    live = []
    for output in observed:
        ids = output[id_key].numpy()
        mask = output["masks"].numpy()
        live.append({int(key): pos for pos, key in enumerate(ids)
                     if key in large and bool(mask[pos].any())})
    a, b = live
    common = sorted(a.keys() & b.keys())
    support = dict(potential_large_rows=len(large), native_nonempty=len(a), center_nonempty=len(b),
                   common_nonempty=len(common), native_only=len(a.keys() - b.keys()),
                   center_only=len(b.keys() - a.keys()), neither=len(large - (a.keys() | b.keys())))
    if not common:
        return result["path"], support, []
    arms = {name: output["masks"].numpy()[[positions[i] for i in common]]
            for name, output, positions in zip(ARMS, observed, live, strict=True)}
    geometry = runtime.footprint(target["masks"].numpy()[common], arms, np.zeros(len(common), dtype=np.bool_),
                                 result["steps"], runtime.compose)
    rows = []
    for position, raw_index in enumerate(common):
        row = dict(prefix=item["prefix"], image_id=item["image"]["id"], path=result["path"],
                   annotation_id=item["annotations"][raw_index]["id"],
                   visible=float(geometry["truth_area"][position]),
                   cropped=bool(geometry["crop_truncated"][position]),
                   edge=bool(geometry["original_edge"][position]))
        if row["visible"] > 0.:
            for name in ARMS:
                data = geometry["by_arm"][name]
                row[name] = dict(signed_area=float(data["area_signed_error"][position]), fn=float(data["fn"][position]),
                                 fp=float(data["fp"][position]), iou=float(data["iou"][position]))
        rows.append(row)
    return result["path"], support, rows


def mean(values):
    return math.fsum(values) / len(values) if values else None


def summarize_rows(rows):
    """Report row- and prefix-weighted paired effects and distinct denominators."""
    groups = defaultdict(list)
    for row in rows:
        if row["visible"] > 0.:
            groups[row["prefix"]].append(row)
    valid = [row for rows_of_group in groups.values() for row in rows_of_group]

    def measure(row, name, metric):
        return (row[name][metric] if metric == "iou"
                else row[name][metric.removesuffix("_fraction")] / row["visible"])

    def group_value(group, name, metric):
        return mean([measure(row, name, metric) for row in group])

    arm_row = {name: {metric: group_value(valid, name, metric) for metric in METRICS} for name in ARMS}
    arm_group = {name: {metric: mean([group_value(group, name, metric) for group in groups.values()])
                        for metric in METRICS} for name in ARMS}
    area_totals = {name: {metric: math.fsum(row[name][metric] for row in valid) for metric in AREA_METRICS}
                   for name in ARMS}

    def contrasts(values, metrics):
        return {metric: values["center"][metric] - values["native"][metric] if valid else None
                for metric in metrics}

    group_summaries = []
    for group in groups.values():
        arm = {name: {metric: group_value(group, name, metric) for metric in METRICS} for name in ARMS}
        group_summaries.append(dict(valid_rows=len(group), unique_images=len({r["image_id"] for r in group}),
                                    unique_annotations=len({r["annotation_id"] for r in group}),
                                    delta=contrasts(arm, METRICS)))
    return dict(common_rows=len(rows), valid_rows=len(valid), no_visible_reference_rows=len(rows) - len(valid),
                unique_annotations=len({row["annotation_id"] for row in valid}),
                unique_images=len({row["image_id"] for row in valid}), prefix_groups=len(groups),
                visible_area_total_source_px2=math.fsum(row["visible"] for row in valid),
                row_equal=arm_row, prefix_equal=arm_group, area_total_source_px2=area_totals,
                delta_row_equal=contrasts(arm_row, METRICS), delta_prefix_equal=contrasts(arm_group, METRICS),
                delta_area_total_source_px2=contrasts(area_totals, AREA_METRICS), groups=group_summaries)


def run(args):
    start = perf_counter()
    runtime = load_runtime(args.code_root)
    require(args.train.is_dir() and args.train.is_absolute() and args.train.resolve(strict=True) == args.train
            and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_relative_to(args.train),
            "Expected canonical closed FIT root and fresh output outside it")
    index_bytes = pinned(args.train / "_annotations.coco.json", INDEX_SHA, 32 * 1024 ** 2)
    index = json.loads(index_bytes)
    layout = json.loads(pinned(args.layout, LAYOUT_SHA, 2 * 1024 ** 2))
    require(len(index["images"]) == 2937 and len(index["annotations"]) == 9260
            and [row["id"] for row in index["categories"]] == [1]
            and layout["native_index_sha256"] == INDEX_SHA and layout["native_index_bytes"] == len(index_bytes),
            "Expected original FIT2937 native index and layout relationship")
    population = build_population(index, layout["images"])
    prefix_counts = Counter(item["prefix"] for item in population)
    labels = sum(len(item["large_indices"]) for item in population)
    require(len(population) == 28 and labels == 50 and len(prefix_counts) == 4,
            "Expected prelocked 28 FIT images, 50 large labels and four prefix groups")

    planned = [[item["image"]["id"], item["ordinal"], item["prefix"], list(view_seeds(item["ordinal"])),
                [item["annotations"][i]["id"] for i in item["large_indices"]]] for item in population]
    design_digest = sha256(encoded(planned)).hexdigest()
    views = Counter()
    counts = {name: Counter() for name in ("all", "direct", "crop")}
    observations = []
    for item in population:
        reference = item["layout"]
        data = pinned(args.train / item["image"]["file_name"], reference["encoded_sha256"], 4 * 1024 ** 2)
        image, scales = runtime.source.io_utils.decode_pil_image_bytes(data, draft_size=None)
        require(data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) == reference["encoded_bytes"]
                and image.mode == "RGB" and image.size == (800, 800) and scales == (1., 1.)
                and sha256(runtime.np.asarray(image).tobytes(order="C")).hexdigest() == reference["decoded_rgb_sha256"],
                "Expected original closed PNG and RGB fingerprints")
        image, target = prepare_image(runtime, image, item)
        for seed in view_seeds(item["ordinal"]):
            path, support, rows = measure_view(runtime, item, image, target, seed)
            views[path] += 1
            counts[path].update(support)
            counts["all"].update(support)
            observations.extend(rows)
    require(sum(views.values()) == 112 and counts["all"]["potential_large_rows"] == 200
            and counts["all"]["common_nonempty"] == len(observations)
            and not runtime.source.torch.cuda.is_initialized(), "Complete frozen view and support census required")
    partitions = {
        "all": observations,
        "direct": [row for row in observations if row["path"] == "direct"],
        "crop": [row for row in observations if row["path"] == "crop"],
        "crop_truncated": [row for row in observations if row["path"] == "crop" and row["cropped"]],
        "crop_intact": [row for row in observations if row["path"] == "crop" and not row["cropped"]],
        "source_edge": [row for row in observations if row["edge"]],
        "source_interior": [row for row in observations if not row["edge"]],
    }
    safe = dict(status="HRSID_FIT_LARGE_FOOTPRINT_PAIRED_ACTUAL_V1", FIT=True, DEV=False, TEST=False, AP=False,
                cuda_initialized=False, model_forward_calls=0, optimizer_updates=0,
                index_sha256=INDEX_SHA, layout_sha256=LAYOUT_SHA, design_sha256=design_digest,
                ordinary_source_sha256={path: digest for path, digest in SOURCES.values()},
                fit_image_count=len(index["images"]), selected_images=len(population), selected_large_labels=labels,
                raw_cohort_prefix_groups=len(prefix_counts), source_images_per_prefix=list(prefix_counts.values()),
                raw_area_threshold_px2=AREA_THRESHOLD, view_seed_base=BASE_SEED, views_per_image=REPEATS,
                views_by_path=dict(views), support={path: dict(counter) for path, counter in counts.items()},
                population_summary={name: summarize_rows(rows) for name, rows in partitions.items()},
                continuous_reference="clipped original Boolean foreground pixel squares",
                elapsed_seconds=perf_counter() - start)
    with args.output.open("xb") as stream:
        stream.write(encoded(safe))
    print(json.dumps(dict(status=safe["status"], design_sha256=design_digest,
                          safe_sha256=sha256(encoded(safe)).hexdigest())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for arg in ("code-root", "train", "layout", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
