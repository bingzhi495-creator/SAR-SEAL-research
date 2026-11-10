"""Replicate the three-way FIT mask-phase direction test on the frozen SSDD views."""

import argparse
import builtins
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
HISTORICAL_WORKER = HERE / "ssdd_fit_replay_dgx.py"
HISTORICAL_WORKER_SHA = "892d7e66114c4c78f0f55ffc2e7b666bdae363ec727780e97a7465b70bed4dee"
DIRECTION_SOURCES = {
    "common_mask_supervision": "59ea82e24353eeb4a14115058d5115333a4e7ea517a0abef581bee137a9a3a57",
    "vertical_phase_resize": "27404870132e5c25d27f9fd4fa5fd9dff3b72e0986a5742b10a55d2b25fe648e",
    "direction_replay": "4ca87b7ce29e00b002b03090fe73c1688e7dfb853625c987e6d7ee1d1c3e7661",
}
FIT_IMAGES, CONTENT_GROUPS = 741, 730
HISTORICAL_PATHS = dict(direct=354, crop=387)
FIT_ORDER_SHA = "42b93cbf2424e1c1ffa45c9dc4b32c3537fca6fe3a224e81762fcc339276e4cd"
SEED_BASE = 5311
STATUS = "SSDD_FIT741_VERTICAL_DIRECTION_REPLAY_COMPLETE"
PROGRESS = "SSDD_FIT_VERTICAL_DIRECTION_REPLAY_PROGRESS"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def pinned_module(path, expected, namespace=None):
    require(path.is_file() and path.resolve(strict=True) == path, "Canonical frozen source required")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == expected, "Source pin differs: " + path.name)
    module = ModuleType("_ssdd_fit_direction_" + path.stem)
    module.__file__ = str(path)
    if namespace is not None:
        def pinned_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in namespace:
                return namespace[name]
            return builtins.__import__(name, globals, locals, fromlist, level)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=pinned_import)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def direction_module(historical):
    namespace = {}
    sources = dict(historical.REPLAY_SOURCES)
    sources.update({name: (REPLAY / (name + ".py"), expected) for name, expected in DIRECTION_SOURCES.items()})
    for name, (path, expected) in sources.items():
        namespace[name] = pinned_module(path, expected, namespace)
    return namespace["direction_replay"]


def verified_groups(archive_path, input_readback, readback):
    """Read image-group identities from the same verified portable manifest."""
    before = readback.identity(archive_path.stat())
    with ZipFile(archive_path) as archive:
        manifest = json.loads(readback.read_member(archive, "manifest.json", readback.MANIFEST_BYTES, readback.MANIFEST_SHA))
    require(before == readback.identity(archive_path.stat()), "Original closed FIT archive changed")
    roster = [[row["image_id"], row["original_basename"], row["original_image_ref"]["bytes"],
               row["original_image_ref"]["sha256"]] for row in manifest["rows"]]
    require(sha256(compact(roster)).hexdigest() == input_readback["content_roster_sha256"],
            "Exact independently read-back JPEG group lineage required")
    groups = {image_id: encoded_sha for image_id, _, _, encoded_sha in roster}
    require(len(roster) == len(groups) == FIT_IMAGES and len(set(groups.values())) == CONTENT_GROUPS
            and sha256(compact(sorted(groups))).hexdigest() == input_readback["native_sorted_id_order_sha256"] == FIT_ORDER_SHA,
            "Exact historical FIT741 ID roster and 730 JPEG-byte groups required")
    return groups


def replay_population(dataset, groups, *, view_sink):
    """Run the stock FIT transform at each fixed seed, retaining logical duplicates."""
    import numpy as np
    from PIL import Image
    import random
    import torch
    import ssdd_phase_fit_replay as historical

    require(type(groups) is dict and len(groups) == FIT_IMAGES
            and all(type(key) is int and key >= 0 and type(value) is str and re.fullmatch("[0-9a-f]{64}", value)
                    for key, value in groups.items()) and len(set(groups.values())) == CONTENT_GROUPS,
            "Exact FIT image to JPEG content groups required")
    ids = tuple(sorted(groups))
    require(type(dataset) is historical.CocoDetection and tuple(dataset.ids) == ids and len(dataset) == len(ids)
            and callable(dataset._transforms) and callable(view_sink), "Exact stock FIT index order required")
    direction = direction_module(historical)
    transform = dataset._transforms
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    views, paths, capacity = [], Counter(), Counter()
    rectangular = 0
    if not historical._SERIAL.acquire(blocking=False):
        raise RuntimeError("FIT transform tracing must run serially")
    try:
        for ordinal, image_id in enumerate(ids):
            seed = SEED_BASE + ordinal
            stage = "stock_decode_and_prepare"
            view = None

            def observe(image, raw):
                nonlocal view, stage
                require(view is None and isinstance(image, Image.Image) and image.mode == "RGB"
                        and image.size == source_hw[::-1], "Exact original stock prepared RGB image required")
                historical._target(raw, image_id, source_hw, source_hw, with_rows=False)
                prepared = dict(raw, **{historical.ROW_ID: torch.arange(len(raw["labels"]), dtype=torch.int64)})

                def qualify(image, target):
                    image, target = transform(image, target)
                    historical._target(target, image_id, source_hw, historical.CANVAS, with_rows=True)
                    return image, target

                stage = "three_way_direction_replay"
                view = direction.replay_direction_view(historical._torchvision, qualify, historical._seed_cpu,
                                                       image, prepared, seed)
                return image, raw

            try:
                source = dataset.coco.imgs[image_id]
                source_hw = (source["height"], source["width"])
                require(all(type(value) is int and value > 0 for value in source_hw), "Positive original FIT dimensions")
                dataset._transforms = observe
                dataset[ordinal]
                require(view is not None and view["path"] in historical.PATHS
                        and view["steps"][0]["before"] == list(source_hw)
                        and view["steps"][-1]["after"] == list(historical.CANVAS), "Exact historical geometry trace required")
                for mode in direction.MODES:
                    present = view["membership"][mode + "_present"]
                    require(present <= historical.QUERY_CAPACITY, "Native query capacity exceeded")
                    capacity[mode] = max(capacity[mode], present)
            except Exception as error:
                raise historical.ReplayViewError(ordinal, image_id, seed, stage, error) from error
            rectangular += source_hw[0] != source_hw[1]
            paths[view["path"]] += 1
            view.update(image_id=image_id, group=groups[image_id], ordinal=ordinal, seed=seed)
            views.append(view)
            view_sink(view)
    finally:
        dataset._transforms = transform
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        historical._SERIAL.release()
    return dict(images=len(ids), views=len(views), groups=len(set(groups.values())), seed_base=SEED_BASE,
                paths=dict(paths), rectangular_source_views=rectangular, capacity=dict(capacity),
                direction=direction.summarize_direction(views, primary=True))


def run(archive, layout, output):
    legacy = pinned_module(HISTORICAL_WORKER, HISTORICAL_WORKER_SHA)
    legacy.runtime()
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink() and not output.is_relative_to(layout)
            and not layout.is_relative_to(output), "Fresh canonical output outside the frozen FIT layout required")
    sys.path.insert(0, str(HERE))
    from ssdd_phase_init import CONFIG, CONFIG_SHA, E1, E1_SHA, RECIPE, RECIPE_SHA, load_api, offline, pinned_bytes
    sys.addaudithook(offline)
    from rfdetr import config
    from ssdd_fit_only import FitOnlyData
    import ssdd_fit_stage_readback as readback
    from ssdd_phase_fit_replay import ReplayViewError
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "DGX CPU diagnostic must leave CUDA unused")
    before = readback.readback_fit_layout(archive, layout)
    groups = verified_groups(archive, before, readback)
    pinned_bytes(RECIPE, RECIPE_SHA)
    model, train = load_api(CONFIG, CONFIG_SHA)._load_configs(RECIPE, layout)
    load_api(E1, E1_SHA).native_recipe(model, train, config)
    data = FitOnlyData(model, train, layout)
    data.setup("fit")
    output.mkdir()
    ledger = sha256()
    start = perf_counter()
    with (output / "VIEWS_PRIVATE.jsonl").open("xb") as stream:
        def record(view):
            raw = compact(view) + b"\n"
            require(stream.write(raw) == len(raw), "Incomplete private FIT direction ledger write")
            ledger.update(raw)
            images = view["ordinal"] + 1
            if images % 50 == 0 or images == FIT_IMAGES:
                print(compact(dict(event=PROGRESS, images=images)).decode("utf-8"), flush=True)

        try:
            result = replay_population(data._dataset_train, groups, view_sink=record)
        except ReplayViewError as error:
            legacy.write_json(output / "FIRST_REPLAY_FAILURE_PRIVATE.json", error.diagnostic)
            raise
        stream.flush()
        os.fsync(stream.fileno())
    require(result["paths"] == HISTORICAL_PATHS and result["rectangular_source_views"] == FIT_IMAGES,
            "Exact historical SSDD path counts and original rectangular population required")
    require(not torch.cuda.is_initialized() and torch.get_num_threads() == torch.get_num_interop_threads() == 1
            and before == readback.readback_fit_layout(archive, layout), "CPU runtime or frozen FIT input changed")
    legacy.write_json(output / "RESULT_SAFE.json", dict(
        status=STATUS, **result, elapsed_seconds=perf_counter() - start,
        fit_archive_sha256=before["fit_archive_sha256"], fit_index_sha256=before["fit_index_sha256"],
        ordered_fit_ids_sha256=FIT_ORDER_SHA, content_roster_sha256=before["content_roster_sha256"],
        content_group_map_sha256=sha256(compact(sorted(groups.items()))).hexdigest(),
        view_ledger_sha256=ledger.hexdigest(), direction_source_sha256=DIRECTION_SOURCES["direction_replay"],
        group_unit="exact JPEG bytes", scene_independence_proven=False,
        platform="Linux_aarch64_DGX_CPU", packages=legacy.PACKAGES, cpu_affinity=[2, 3], torch_threads=1,
        cuda_initialized=False, exact_shared_geometry_rng_images=True, exact_common_nonmask_fields=True,
        FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("archive", "layout", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.archive, args.layout, args.output)


if __name__ == "__main__":
    main()
