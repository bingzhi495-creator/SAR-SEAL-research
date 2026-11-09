"""Replay the SSDD FIT mask resize pair on one stock CPU view per logical image."""

import builtins
from collections import Counter
from hashlib import sha256
from pathlib import Path
import random
from threading import Lock
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import torch
from rfdetr.datasets import _torchvision
from rfdetr.datasets.coco import CocoDetection


FIT_IMAGES = 741
SEED_BASE = 5311
QUERY_CAPACITY = 100
CANVAS = (576, 576)
ROW_ID = "_p198_row_id"
FIELDS = {"boxes", "labels", "area", "iscrowd", "image_id", "orig_size", "size", "masks"}
PATHS = ("direct", "crop")
MEAN_FIELDS = ("legacy_y", "legacy_x", "center_y", "center_x", "legacy_abs_y", "center_abs_y",
               "legacy_iou", "center_iou")
CODE = Path(__file__).resolve().parent.parent
REPLAY_DIR = CODE / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
REPLAY_SOURCES = {
    "fit_query_export": (CODE / "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109" / "fit_query_export.py",
                         "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5"),
    "reference_resize": (REPLAY_DIR / "reference_resize.py",
                         "3ca1d4a6400324c13c777cb4d5f7627713bc625580195d8e05363950872cc8df"),
    "trace_resize": (REPLAY_DIR / "trace_resize.py",
                     "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f"),
    "replay_fit": (REPLAY_DIR / "replay_fit.py",
                   "4f95f1b5b84fcf89b005f999cfb8c5fd647a7b262aedd2622860cf8785a14157"),
}
_SERIAL = Lock()


class ReplayViewError(RuntimeError):
    """The first failed logical image, available to the private caller."""

    def __init__(self, ordinal, image_id, seed, stage, cause):
        self.diagnostic = dict(ordinal=ordinal, image_id=image_id, seed=seed, stage=stage,
                               cause_type=type(cause).__name__)
        super().__init__(f"SSDD FIT replay stopped at ordinal {ordinal} in {stage}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _load_replay():
    modules = {}

    def import_source(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    for name, (path, digest) in REPLAY_SOURCES.items():
        require(path.resolve(strict=True) == path, "Exact canonical FIT replay source required")
        limit = 64 * 1024
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        require(0 < len(raw) <= limit and sha256(raw).hexdigest() == digest, "Pinned FIT replay source changed")
        module = ModuleType("_ssdd_fit_" + name)
        module.__file__ = str(path)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=import_source)
        exec(compile(raw, str(path), "exec"), module.__dict__)
        modules[name] = module
    return modules["replay_fit"]


def _seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def _target(target, image_id, source_hw, canvas, *, with_rows):
    fields = FIELDS | {ROW_ID} if with_rows else FIELDS
    require(type(target) is dict and set(target) == fields
            and all(torch.is_tensor(value) and value.device.type == "cpu" for value in target.values()),
            "Exact stock segmentation target fields on CPU required")
    count = len(target["labels"])
    require(target["image_id"].dtype == torch.int64 and target["image_id"].numel() == 1
            and int(target["image_id"].item()) == image_id
            and target["orig_size"].tolist() == list(source_hw) and target["size"].tolist() == list(canvas)
            and target["boxes"].shape == (count, 4) and target["boxes"].dtype == torch.float32
            and target["masks"].shape == (count, *canvas) and target["masks"].dtype == torch.bool
            and all(target[key].shape == (count,) for key in ("labels", "area", "iscrowd"))
            and target["labels"].dtype == torch.int64 and bool((target["labels"] == 0).all())
            and bool(torch.isfinite(target["boxes"]).all()) and bool(torch.isfinite(target["area"]).all()),
            "Original image identity and stock segmentation raster geometry required")
    if with_rows:
        require(target[ROW_ID].shape == (count,) and target[ROW_ID].dtype == torch.int64,
                "Exact retained stock row identities required")


def _view(replay, transform, image, raw, seed, image_id, source_hw):
    require(isinstance(image, Image.Image) and image.mode == "RGB" and image.size == source_hw[::-1],
            "Decoded RGB JPEG must retain original variable dimensions")
    _target(raw, image_id, source_hw, source_hw, with_rows=False)
    prepared = dict(raw, **{ROW_ID: torch.arange(len(raw["labels"]), dtype=torch.int64, device="cpu")})

    def target_summary(target, side, iid, *, normalized):
        require(normalized is True and (side, side) == CANVAS and iid == image_id,
                "Expected the original stock normalized FIT view")
        _target(target, iid, source_hw, CANVAS, with_rows=True)

    source = SimpleNamespace(_torchvision=_torchvision, set_seed=_seed_cpu, target_summary=target_summary)
    view = replay.replay_view(SimpleNamespace(transform=transform), source, image, prepared, seed)
    require(view["path"] in PATHS and view["steps"][0]["before"] == list(source_hw)
            and view["steps"][-1]["after"] == list(CANVAS), "Exact source-to-FIT geometry trace required")
    return view


def _area_summary(totals):
    rows = totals["count"]
    return dict(rows=rows, masks_changed=totals["masks_changed"], changed_pixels=totals["changed_pixels"],
                means={key: totals[key] / rows if rows else None for key in MEAN_FIELDS})


def replay_fit_population(dataset, expected_ids):
    """Use an admitted stock FIT dataset serially and return aggregate replay evidence."""
    ids = tuple(expected_ids)
    require(type(dataset) is CocoDetection and len(ids) == len(dataset) == FIT_IMAGES
            and all(type(image_id) is int for image_id in ids) and len(set(ids)) == FIT_IMAGES
            and tuple(dataset.ids) == ids and callable(dataset._transforms),
            "Exact admitted SSDD FIT logical IDs in stock index order required")
    if not _SERIAL.acquire(blocking=False):
        raise RuntimeError("SSDD FIT replay must run serially")
    original_transform = dataset._transforms
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    membership, path_counts = Counter(), Counter()
    path_membership = {name: Counter() for name in PATHS}
    area = {name: {key: Counter() for key in ("small", "non_small")} for name in PATHS}
    capacity = dict(native_max_present=0, center_max_present=0)
    rectangular = 0
    try:
        replay = _load_replay()
        for ordinal, image_id in enumerate(ids):
            seed = SEED_BASE + ordinal
            stage = "stock_decode_and_prepare"
            view = None

            def observe(image, raw):
                nonlocal stage, view
                require(view is None, "Exactly one prepared target per stock logical image required")
                stage = "paired_stock_transform"
                view = _view(replay, original_transform, image, raw, seed, image_id, source_hw)
                return image, raw

            try:
                entry = dataset.coco.imgs[image_id]
                source_hw = entry["height"], entry["width"]
                require(all(type(value) is int and value > 0 for value in source_hw),
                        "Positive original JPEG dimensions required")
                dataset._transforms = observe
                dataset[ordinal]
                require(view is not None, "Stock dataset did not present a prepared replay target")
                stage = "query_capacity"
                rows = view["rows"]
                native, center = rows["legacy_present"], rows["center_present"]
                require(max(native, center) <= QUERY_CAPACITY,
                        f"More than {QUERY_CAPACITY} retained rows (native={native}, center={center})")
            except Exception as error:
                raise ReplayViewError(ordinal, image_id, seed, stage, error) from error
            path = view["path"]
            rectangular += source_hw[0] != source_hw[1]
            path_counts[path] += 1
            membership.update(rows)
            path_membership[path].update(rows)
            for key in area[path]:
                area[path][key].update(view["metrics"][key])
            capacity["native_max_present"] = max(capacity["native_max_present"], native)
            capacity["center_max_present"] = max(capacity["center_max_present"], center)
    finally:
        dataset._transforms = original_transform
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        _SERIAL.release()

    return dict(
        status="SSDD_FIT_NATIVE_CENTER_REPLAY_COMPLETE", images=len(ids), views=len(ids), seed_base=SEED_BASE,
        rectangular_source_views=rectangular, membership=dict(membership),
        capacity=dict(limit=QUERY_CAPACITY, **capacity),
        by_path={path: dict(views=path_counts[path], membership=dict(path_membership[path]),
                           small=_area_summary(area[path]["small"]), other=_area_summary(area[path]["non_small"]))
                 for path in PATHS},
        population_unit="logical FIT images", scene_independence_proven=False,
        area_stratum="prepared original annotation area < 1024 pixels squared", geometry_weighting="rows",
        geometry_scope="common nonempty arm rows with nonempty one-pass reference",
        exact_shared_geometry_rng_images=True, exact_common_nonmask_fields=True,
        FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0,
    )
