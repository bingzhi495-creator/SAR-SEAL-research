"""Audit one prospectively seeded native 576-pixel augmentation view per official HRSID FIT image."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch
from rfdetr.datasets import _torchvision, coco, io_utils, transforms
from rfdetr.utilities import tensors


FIT_VIEW_SHA = "c7629de9afaa398d0b943fba8e4cec75c8b65a9864fd795c1465c9b4dec3c271"
FIT_VIEW_RECEIPT_SHA = "8f89295f891416eead7815e2c47477b5fc0d27df5cdf26a260e637a14d580160"
FINDINGS_SHA = "4ecc36d10a329ed93df280734e8867b0b330bb240b7ed084de86f96dc950e64f"
TRAIN_SHA = "3e240c38014526e9b93206a946105fb41c808272238932b0d4c2422978d5116f"
ARCHIVE_SHA = "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159"
ARCHIVE_RECEIPT_SHA = "4de89dff4cbf13a856e714b8952503a7e11ed2c9ef4405d72b06acdb425d52db"
CENTRAL_SHA = "4a54303e8cce24f0330cddd2e6d3c12205c8b0002859d0b5bf534efb531a2dce"
VERSIONS = {"rfdetr":"1.12.0", "torch":"2.10.0+cu130", "torchvision":"0.25.0+cu130",
            "Pillow":"12.3.0", "pycocotools":"2.0.11", "numpy":"1.26.4"}
SOURCE_SHA = {
    coco: "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9",
    _torchvision: "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56",
    transforms: "a4070434e1cfcb50040f1c84abddf13e37fb8f93953c76c035c32058f8b6d2a1",
    io_utils: "0e8d54c2867f8eec28aa25f650501a91bac8116ca2f7e40e9ddd4cd1332bcd65",
    tensors: "122f44a151db786ef4845c5d89e5cb2209624651d3638f007e3c0867f7312827",
}
BASE_SEED = 520100000
QUERY_CAP = 100


def need(ok, message):
    if not ok:
        raise ValueError(message)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mask_empty(target):
    return ~target["masks"].flatten(1).any(1)


def target_summary(target, side, image_id, *, normalized):
    n = len(target["labels"])
    boxes = target["boxes"]
    need(target["image_id"].numel() == 1 and int(target["image_id"].item()) == image_id
         and target["masks"].shape == (n, side, side) and target["masks"].dtype == torch.bool
         and boxes.shape == (n, 4) and boxes.dtype == torch.float32
         and target["labels"].shape == target["iscrowd"].shape == target["area"].shape == (n,)
         and target["labels"].dtype == torch.int64 and bool((target["labels"] == 0).all())
         and bool((target["iscrowd"] == 0).all()) and bool(torch.isfinite(target["area"]).all())
         and bool(torch.isfinite(boxes).all()) and bool((target["size"] == side).all())
         and bool((target["orig_size"] == 800).all()), "Native target fields and exact dimensions")
    if normalized:
        xy0, xy1 = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
        limit = 1.
    else:
        xy0, xy1 = boxes[:, :2], boxes[:, 2:]
        limit = float(side)
    need(bool((xy0 >= -1e-6).all() and (xy1 <= limit + 1e-6).all() and (xy1 > xy0).all())
         and bool((target["area"] > 0).all()), "Native boxes and recorded area remain valid")
    return dict(labels=n, empty_masks=int(mask_empty(target).sum()))


def common_rows_equal(base, restored):
    """The paired source may insert one retained empty row or leave outputs identical."""
    fields = ("labels", "boxes", "area", "iscrowd", "masks")
    n, m = len(base["labels"]), len(restored["labels"])
    if n == m:
        return all(torch.equal(base[k], restored[k]) for k in fields)
    if m != n + 1:
        return False
    return any(all(torch.equal(base[k][:i], restored[k][:i]) and
                   torch.equal(base[k][i:], restored[k][i + 1:]) for k in fields)
               for i in mask_empty(restored).nonzero().flatten().tolist())


class NativeInputAudit:
    def __init__(self):
        self.convert = coco.ConvertCoco(include_masks=True, include_keypoints=False,
                                        cat2label={1: 0}, num_keypoints_per_class=[])
        self.transform = coco.make_coco_transforms_square_div_64(
            "train", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
            patch_size=12, num_windows=1, aug_config=None, scale_jitter=True,
            gpu_postprocess=False, keypoint_flip_pairs=None)
        self.collate = tensors.make_collate_fn(block_size=12, pack=True, pad_targets_to=None)
        first = self.transform.transforms[0]
        need(isinstance(first, _torchvision.RandomSelect) and first.p == .5, "Native two-branch selector")

    def view(self, image, image_id, annotations, seed):
        set_seed(seed)
        im, source = self.convert(image, dict(image_id=image_id, annotations=annotations))
        before = target_summary(source, 800, image_id, normalized=False)
        image, target = self.transform(im, source)
        target_summary(target, 576, image_id, normalized=True)
        samples, packed = self.collate([(image, target)])
        sample, = packed.to_list("cpu")
        need(samples.tensors.shape == (1, 3, 576, 576) and samples.tensors.dtype == torch.float32
             and bool(torch.isfinite(samples.tensors).all()) and samples.mask.shape == (1, 576, 576)
             and not bool(samples.mask.any()), "Native 576 image tensor with no padding")
        after = target_summary(sample, 576, image_id, normalized=True)
        need(after["labels"] <= before["labels"], "Native transform cannot add labels")
        return before, after, source, sample, samples.tensors

    def observe(self, encoded, image_id, annotations, seed, *, original_annotations=None):
        image, scale = io_utils.decode_pil_image_bytes(encoded, draft_size=None)
        need(encoded.startswith(b"\x89PNG\r\n\x1a\n") and image.mode == "RGB"
             and image.size == (800, 800) and scale == (1., 1.), "Full native 800x800 PNG decode")
        branch = "direct_resize" if torch.rand((), generator=torch.Generator().manual_seed(seed)) < .5 else "resize_crop"
        before, after, source, augmented, pixels = self.view(image, image_id, annotations, seed)
        result = dict(before=before, after=after, branch=branch, source_labels=len(annotations),
                      source_crowd=sum(a.get("iscrowd", 0) for a in annotations))
        if original_annotations is not None:
            other_before, other_after, other_source, other_augmented, other_pixels = self.view(
                image, image_id, original_annotations, seed)
            need(other_before["labels"] == before["labels"] + 1
                 and other_before["empty_masks"] == before["empty_masks"] + 1
                 and common_rows_equal(source, other_source), "Restore exactly one original empty target")
            result["paired_source"] = dict(
                before=other_before, after=other_after, image_equal=torch.equal(pixels, other_pixels),
                shared_rows_equal=common_rows_equal(augmented, other_augmented))
        return result


def pinned_json(path, digest, limit):
    need(path.is_file() and 0 < path.stat().st_size <= limit, "Closed JSON size")
    data = path.read_bytes()
    need(sha256(data).hexdigest() == digest, "Closed JSON digest")
    return json.loads(data)


def read_member(z, name, limit):
    info = z.getinfo(name)
    need(0 < info.file_size <= limit, "Bounded archive member")
    raw = z.read(info)
    need(len(raw) == info.file_size, "Exact archive member length")
    return raw


def png_member(image):
    name = PurePosixPath(image["file_name"])
    need(name.name == image["file_name"] and name.suffix.lower() in {".jpg", ".png"}
         and re.fullmatch(r"P\d+_[A-Za-z0-9_]+", name.stem)
         and image["height"] == image["width"] == 800, "Exact official FIT PNG name")
    return f"images/{name.stem}.png"


def ordered_fits(view, findings):
    images = sorted(view["images"], key=lambda x: x["id"])
    ids = {r["id"] for r in images}
    groups = defaultdict(list)
    for ann in view["annotations"]:
        need(ann["category_id"] == 1 and ann["image_id"] in ids and ann.get("iscrowd", 0) in (0, 1),
             "FIT annotation category and population")
        groups[ann["image_id"]].append(ann)
    excluded = {r["annotation_id"]: r["image_id"] for r in findings}
    need(len(images) == len(ids) == len(groups) == 2937 and len(view["annotations"]) == 9260
         and len({r["id"] for r in view["annotations"]}) == 9260 and len(findings) == len(excluded) == 5
         and len(set(excluded.values())) == 5 and set(excluded.values()) <= ids
         and not set(excluded).intersection(r["id"] for r in view["annotations"])
         and all(r["kind"] == "EMPTY_COCO_RASTER" for r in findings)
         and len({png_member(r) for r in images}) == 2937, "Only the frozen FIT training view")
    return images, groups, excluded


def source_controls(z, groups, excluded):
    raw = read_member(z, "annotations/train2017.json", 24 * 1024**2)
    need(sha256(raw).hexdigest() == TRAIN_SHA, "Exact original TRAIN member")
    originals = defaultdict(list)
    affected = set(excluded.values())
    for r in json.loads(raw)["annotations"]:
        if r["image_id"] in affected:
            originals[r["image_id"]].append(r)
    for aid, iid in excluded.items():
        records = originals[iid]
        need(len(records) == len(groups[iid]) + 1 and any(r["id"] == aid for r in records)
             and [r for r in records if r["id"] != aid] == groups[iid], "Only the exact five source control restorations")
    return originals


def add_target_stats(counter, entry):
    n, empty = entry["labels"], entry["empty_masks"]
    counter.update(dict(images=1, target_rows=n, empty_mask_rows=empty,
                        images_with_empty_mask_rows=int(empty > 0), zero_target_images=int(n == 0),
                        nonempty_targets=n - empty, images_without_nonempty_targets=int(n == empty),
                        images_above_100_query_cap=int(n > QUERY_CAP),
                        min_unassigned_per_100_query_group=max(0, n - QUERY_CAP)))
    counter["max_targets_per_image"] = max(counter["max_targets_per_image"], n)


def offline(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("The native FIT input audit is offline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "archive-receipt", "fit-view", "fit-view-receipt", "findings", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    options = parser.parse_args()
    need(bool(re.fullmatch(r"[0-9a-f]{40}", options.protocol)), "Prospective exact protocol")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_available()
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
         and tuple(sys.version_info[:3]) == (3, 10, 13)
         and {key:version(key) for key in VERSIONS} == VERSIONS
         and all(sha256(Path(m.__file__).read_bytes()).hexdigest() == digest for m, digest in SOURCE_SHA.items()),
         "Exact pinned offline CPU native input runtime")
    sys.addaudithook(offline)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = perf_counter()
    receipt = pinned_json(options.archive_receipt, ARCHIVE_RECEIPT_SHA, 1024**2)
    verified = pinned_json(options.fit_view_receipt, FIT_VIEW_RECEIPT_SHA, 1024**2)
    need(receipt["archive_sha256"] == ARCHIVE_SHA and receipt["all_member_CRC_verified"] is True
         and receipt["expected_central_directory_sha256"] == CENTRAL_SHA
         and verified["derived_sha256"] == FIT_VIEW_SHA and verified["only_five_original_exclusions"] is True
         and verified["independent_closed_annotation_copy_field_readback"] is True, "Exact upstream integrity receipts")
    view = pinned_json(options.fit_view, FIT_VIEW_SHA, 16 * 1024**2)
    findings = pinned_json(options.findings, FINDINGS_SHA, 1024**2)
    images, groups, excluded = ordered_fits(view, findings)
    options.output.mkdir()
    audit = NativeInputAudit()
    totals = {key:Counter() for key in ("converter_only", "augmented")}
    by_branch = {key:{stage:Counter() for stage in totals} for key in ("direct_resize", "resize_crop")}
    high_density = {key:Counter() for key in totals}
    paired = Counter()
    source_rows = Counter()
    visits = sha256()
    def file_identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    archive_start = options.archive.stat()
    need(archive_start.st_size == 5698302315, "Closed official archive length")
    with options.archive.open("rb") as stream:
        need(file_identity(archive_start) == file_identity(os.fstat(stream.fileno())),
             "Closed archive handle identity")
        stream.seek(5697629273)
        need(sha256(stream.read(672944)).hexdigest() == CENTRAL_SHA, "Exact official ZIP central directory")
        with ZipFile(stream) as z:
            need(len(z.infolist()) == 5612, "Exact ZIP member census")
            controls = source_controls(z, groups, excluded)
            for index, image in enumerate(images):
                iid = image["id"]
                seed = BASE_SEED + index
                observed = audit.observe(read_member(z, png_member(image), 8 * 1024**2), iid, groups[iid], seed,
                                         original_annotations=controls.get(iid))
                source_rows.update(dict(annotations=observed["source_labels"], crowd=observed["source_crowd"],
                                        dropped_by_native_converter=observed["source_labels"]-observed["before"]["labels"]))
                for key, entry in (("converter_only", observed["before"]), ("augmented", observed["after"])):
                    add_target_stats(totals[key], entry)
                    add_target_stats(by_branch[observed["branch"]][key], entry)
                    if len(groups[iid]) > QUERY_CAP:
                        add_target_stats(high_density[key], entry)
                if "paired_source" in observed:
                    control = observed["paired_source"]
                    need(control["image_equal"] and control["shared_rows_equal"], "Paired common outputs unchanged")
                    paired.update(dict(images=1, equal_augmented_image=1, equal_common_target_rows=1,
                        source_extra_empty_before=control["before"]["empty_masks"]-observed["before"]["empty_masks"],
                        source_extra_empty_after=control["after"]["empty_masks"]-observed["after"]["empty_masks"],
                        source_extra_labels_after=control["after"]["labels"]-observed["after"]["labels"]))
                visits.update((json.dumps([iid, seed, observed], sort_keys=True) + "\n").encode())
                if (index + 1) % 250 == 0:
                    print(json.dumps(dict(event="HRSID_FIT_NATIVE_AUG_PROGRESS", visited=index + 1)), flush=True)
    after = options.archive.stat()
    need(file_identity(after) == file_identity(archive_start)
         and totals["augmented"]["images"] == 2937 and source_rows["annotations"] == 9260
         and high_density["converter_only"]["images"] == paired["images"] == 5,
         "Complete original FIT audit and stable archive")
    qualified = (source_rows["dropped_by_native_converter"] == 0 and
                 totals["converter_only"]["empty_mask_rows"] == totals["augmented"]["empty_mask_rows"] == 0)
    result = dict(
        status="HRSID_FIT_NATIVE_AUGMENTATION_AUDIT_COMPLETE",
        protocol=options.protocol, runtime=VERSIONS, native_sources={m.__name__:s for m,s in SOURCE_SHA.items()},
        fit_view_sha256=FIT_VIEW_SHA, original_archive_sha256=ARCHIVE_SHA, original_train_sha256=TRAIN_SHA,
        one_view_per_FIT_image=True, seed_rule="520100000 + position in ascending image ID order",
        resolution=576, scale_jitter=True, horizontal_flip=True, packed_collate_block_size=12,
        query_capacity=QUERY_CAP, complete_FIT_images=2937, source=source_rows, totals=totals,
        by_native_resize_branch=by_branch, five_high_density_images=high_density, paired_exact_exclusions=paired,
        visit_aggregate_sha256=visits.hexdigest(), audit_seconds=perf_counter()-started,
        no_empty_mask_target_gate=qualified, all_training_augmentation_seeds_qualified=False,
        training_qualification=False, model_executed=False, original_DEV_image_opened=False,
        TEST_member_opened=False, AP=False)
    with (options.output / "NATIVE_AUG_AUDIT_SAFE.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

