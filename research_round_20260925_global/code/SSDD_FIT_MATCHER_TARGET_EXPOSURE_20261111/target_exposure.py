"""Replay exact SSDD FIT augmentation against closed matcher packets, with no model forward."""

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from zipfile import ZipFile


CAPTURE_PROTOCOL = "860c243f60ddbf2796a3ce59760fabd971ed52d6"
REPLAY_PROTOCOL = "3baf237b8f643c36e26d435612de68b337eb3b8c"
CAPTURE_REPORT = "9d106f6ec4b6777cdae5d3fe8c2ac2619864aaa7000cc67248c0d12abda47ac4"
CAPTURE_CLOSURE = "7fbe03bb5816386513e1e9edeac631fdf0829cd00ae20b521739c26f53736fd5"
REPLAY_REPORT = "6d7bbcc4587cf3b79f655c1c6a60d0443f9234cb42ffcc17e9e675cc2b01b3f0"
REPLAY_CLOSURE = "bb10d4bcac4ad11bde280f1156b0f808d9901fe9c1d3d83d6c5f404f98c870f8"
PINNED_SOURCE = {
    "common_mask_supervision": "59ea82e24353eeb4a14115058d5115333a4e7ea517a0abef581bee137a9a3a57",
    "fit_query_export": "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5",
    "reference_resize": "3ca1d4a6400324c13c777cb4d5f7627713bc625580195d8e05363950872cc8df",
    "trace_resize": "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f",
    "replay_fit": "4f95f1b5b84fcf89b005f999cfb8c5fd647a7b262aedd2622860cf8785a14157",
    "paired_fit_transform": "64f3f63c6ac89b66578f0ada42fa3eb8b4db3b2f1ae2ea9113debe6ab9b270e0",
    "one_pass_fit_transform": "05cad566f9b5cc761e91b5283585a5392a7450c42c3133349e6dc21e8bf7496d",
}
CAPTURE_SOURCE = "7a39df88f849d36ff139195d6001ccf7a5ea258144d9bcd4a3440e7a1a572703"
REPLAY_SOURCE = "5a5001c18da150436576cc1cdb4e0d581a5e516e7e7e3ab0776e07b308e98c85"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def offline(event, _):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise RuntimeError("Network disabled in closed FIT target exposure")


def exposure(a, b, c):
    """Summarize Boolean target exposure on one common rows-by-points bank."""
    import numpy as np

    need(a.ndim == 2 and a.shape == b.shape == c.shape
         and a.dtype == b.dtype == c.dtype == np.bool_, "Exact common Boolean target point bank")
    hit = [x.any(axis=1) for x in (a, b, c)]

    def transitions(before, after, before_hits, after_hits):
        return dict(added=int((~before & after).sum()), removed=int((before & ~after).sum()),
                    changed_rows=int((before ^ after).any(axis=1).sum()),
                    gained_hit_rows=int((~before_hits & after_hits).sum()),
                    lost_hit_rows=int((before_hits & ~after_hits).sum()))

    return dict(rows=len(a), compared_target_points=a.size,
                zero_hit_rows=dict(zip("abc", (int((~x).sum()) for x in hit))),
                a_to_b=transitions(a, b, *hit[:2]), b_to_c=transitions(b, c, *hit[1:]))


def sum_dicts(items):
    if not items:
        return {}
    return {key: (sum_dicts([row[key] for row in items]) if isinstance(value, dict)
                  else sum(row[key] for row in items))
            for key, value in items[0].items()}


def seed_native(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def replay_arms(image, prepared, stock, native, seed):
    from common_mask_supervision import ROW_ID, common_targets
    from one_pass_fit_transform import substitute_one_pass_masks
    from paired_fit_transform import _check_output, _prepare, _rng, _rng_bytes
    from trace_resize import capture_transform
    import torch

    prepared = _prepare(image, prepared)
    views = []
    for mode in ("native", "center"):
        seed_native(seed)
        with capture_transform(native, mode=mode) as trace:
            pixels, target = stock(image.copy(), {key: val.clone() for key, val in prepared.items()})
        _check_output(pixels, target, prepared)
        views.append((pixels, target, trace, _rng_bytes(_rng())))
    pixels, a, trace, rng = views[0]
    pixels_b, b, trace_b, rng_b = views[1]
    need(torch.equal(pixels, pixels_b) and trace == trace_b and rng == rng_b,
         "A and B require identical image, geometry and post-augmentation CPU RNG")
    c = substitute_one_pass_masks(prepared["masks"], b, trace)
    direct = not any(step["op"] == "crop" for step in trace)
    need(not direct or torch.equal(b["masks"], c["masks"]), "Direct full-grid B/C negative control")
    fixed = common_targets(a, b, c)
    common_mask = torch.isin(a[ROW_ID], fixed[0][ROW_ID])
    return pixels, a, fixed, common_mask, "direct" if direct else "crop"


def observed_exposure(packet, packed, fixed, keep, cap, torch):
    from rfdetr.models.heads.segmentation import point_sample

    for name in ("masks", "boxes", "labels"):
        actual, old = packed[name], packet["target_" + name]
        need(actual.dtype == old.dtype and torch.equal(actual, old),
             "Exact original CUDA-captured A target " + name)
    points, logits, labels = [packet[name] for name in
                              ("uniform_draw", "native_pred_at_draw", "native_target_at_draw")]
    n = len(packed["labels"])
    need(points.shape == (cap.K, 2) and logits.shape == (cap.GROUPS * cap.WIDTH, cap.K)
         and labels.shape == (n, cap.K) and points.dtype == logits.dtype == labels.dtype == torch.float32
         and bool(torch.isfinite(logits).all()) and bool(((points >= 0) & (points < 1)).all()),
         "Exact archived stock uniform grid and finite original logits")

    def sample(masks):
        if len(masks) == 0:
            return torch.empty((0, cap.K), dtype=torch.float32)
        xy = points.unsqueeze(0).expand(len(masks), -1, -1)
        return point_sample(masks.float().unsqueeze(1), xy, align_corners=False, mode="nearest").squeeze(1)

    old_rebuilt = sample(packed["masks"])
    need(torch.equal(old_rebuilt, labels), "Exact independently reconstructed original A nearest point labels")
    a, b, c = (sample(target["masks"]) for target in fixed)
    need(torch.equal(a, labels[keep]), "Exact original A nearest labels on fixed common rows")
    small = packet["is_small"][keep]
    stats = {}
    for key, chosen in (("all", torch.ones_like(small)), ("small", small), ("non_small", ~small)):
        stats[key] = exposure(*(value[chosen].numpy().astype(bool) for value in (a, b, c)))
        stats[key]["b_to_c"]["grid_changed_rows"] = int((fixed[1]["masks"][chosen]
                                                        ^ fixed[2]["masks"][chosen]).flatten(1).any(1).sum())
    hashes = dict(input_sha256=packet["input_sha256"], point_sha256=cap.state_digest({"points": points}),
                  logits_sha256=cap.state_digest({"logits": logits}),
                  native_labels_sha256=cap.state_digest({"labels": labels}))
    return stats, hashes


def run(prior, geometry, output, expected_python):
    import types

    need(Path(sys.executable) == expected_python
         and sys.version_info[:3] == (3, 12, 14)
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.sched_getaffinity(0) == {2, 3},
         "Exact offline DGX CPU interpreter and CPU affinity")
    sys.addaudithook(offline)
    for path in (prior, geometry, output.parent):
        need(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
             "Existing canonical private parent")
    need(not output.exists() and not output.is_symlink(), "Fresh result directory only")

    def load(path, expected, name):
        before = path.stat()
        raw = path.read_bytes()
        need(path.resolve(strict=True) == path and 0 < len(raw) < 2**20
             and sha256(raw).hexdigest() == expected and before.st_mtime_ns == path.stat().st_mtime_ns
             and before.st_ino == path.stat().st_ino,
             "Exact immutable source " + name)
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(raw, str(path), "exec"), module.__dict__)
        return module

    cap = load(prior / "capture_matcher_cost.py", CAPTURE_SOURCE, "p125_capture")
    replay = load(prior / "REPLAY_CODE/matcher_cost_runtime.py", REPLAY_SOURCE, "p125_runtime")
    report = json.loads(replay.checked_bytes(prior / "CAPTURE/CAPTURE_SAFE.json", CAPTURE_REPORT))
    closure = json.loads(replay.checked_bytes(prior / "CONTROL/ORIGINAL_TERMINAL_PRIVATE.json", CAPTURE_CLOSURE))
    old = json.loads(replay.checked_bytes(prior / "REPLAY_ACTUAL/RESULT.json", REPLAY_REPORT))
    old_end = json.loads(replay.checked_bytes(prior / "REPLAY_CONTROL/ORIGINAL_TERMINAL_PRIVATE.json", REPLAY_CLOSURE))
    need(old["protocol"] == REPLAY_PROTOCOL and old["capture_protocol"] == CAPTURE_PROTOCOL
         and old["exact_native_anchors"] == old["exact_native_solver_pair_images"] == 30
         and old_end["exit"] == 0 and old_end["original_OS_wait"] is True and old_end["resource_stop"] is False
         and old_end["protocol"] == REPLAY_PROTOCOL and old_end["result_sha256"] == REPLAY_REPORT,
         "Closed original exact CUDA logit, label and assignment replay")
    rows = replay.validate_closed(report, closure, CAPTURE_PROTOCOL, CAPTURE_REPORT, cap)
    need(report["tranche"] == "cohort32" and len(rows) == 32 and report["exact_stock_traces"] == 30
         and sum(row["status"] == "ZERO_TARGET_AUGMENTED" for row in rows) == 2,
         "Fixed original 32-view SSDD FIT roster, including two zero-target views")

    import torch
    from rfdetr.datasets import _torchvision as native
    from rfdetr.datasets.coco import ConvertCoco, make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.utilities.tensors import make_collate_fn

    need(not torch.cuda.is_available() and {key: version(key) for key in cap.VERSIONS} == cap.VERSIONS
         and {key: version(key) for key in report["decoder_versions"]} == report["decoder_versions"]
         and all(cap.sha(Path(import_module(key).__file__).read_bytes()) == digest
                 for key, digest in cap.NATIVE_SOURCE.items()), "Original exact loader and matcher library identity")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for name, digest in PINNED_SOURCE.items():
        load(geometry / (name + ".py"), digest, name)
    from common_mask_supervision import ROW_ID

    start = perf_counter()
    items = json.loads(cap.exact_bytes(prior / "FIT741_IMAGE_ONLY_MANIFEST_PRIVATE.json",
                                      348469, cap.FIT_INDEX_SHA))["items"]
    need(len(items) == 741 and [row["dataset_index"] for row in items] == list(range(741)), "Exact SSDD FIT census")
    archive = cap.exact_bytes(prior / "SSDD_FIT741_ORIGINAL_INPUTS_20261107_01.zip", 33079338, cap.FIT_BUNDLE_SHA)
    plan = cap.view_plan(items, "cohort32")
    convert = ConvertCoco(include_masks=True, include_keypoints=False, cat2label={1: 0}, num_keypoints_per_class=[])
    stock = make_coco_transforms_square_div_64(
        "train", cap.RESOLUTION, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=12, num_windows=1, aug_config=None, scale_jitter=True, gpu_postprocess=False,
        keypoint_flip_pairs=None)
    collate = make_collate_fn(block_size=12, pack=True, pad_targets_to=None)
    results = defaultdict(list)
    replayed = []
    with ZipFile(BytesIO(archive)) as bundle:
        gt_raw = bundle.read("annotations/fit.json")
        need(len(gt_raw) == 498760 and cap.sha(gt_raw) == cap.FIT_GT_SHA, "Original sealed FIT labels")
        gt = json.loads(gt_raw)
        ids = {row["image_id"] for row in items}
        need(len(ids) == 741 and {row["id"] for row in gt["images"]} == ids, "FIT-only membership")
        by_image = defaultdict(list)
        for row in gt["annotations"]:
            need(row["image_id"] in ids and row["category_id"] == 1, "Exact FIT ship labels")
            by_image[row["image_id"]].append(row)
        for row, (item, repeat) in zip(rows, plan, strict=True):
            seed = 5301000 + 2 * item["dataset_index"] + repeat
            iid, digest = item["image_id"], item["image_sha256"]
            raw = bundle.read(f"images/{digest}.jpg")
            need(len(raw) == item["image_bytes"] and cap.sha(raw) == digest, "Exact original JPEG")
            seed_native(seed)
            image, scale = decode_pil_image_bytes(raw, draft_size=None)
            h, w = item["original_hw"]
            need(image.mode == "RGB" and image.size == (w, h) and scale == (1., 1.), "Original JPEG canvas")
            image, prepared = convert(image, dict(image_id=iid, annotations=by_image[iid]))
            pixels, a, fixed, keep, branch = replay_arms(image, prepared, stock, native, seed)
            count = len(a["labels"])
            need(count == row["target_rows"], "Original A augmented target count")
            if not count:
                need(row["status"] == "ZERO_TARGET_AUGMENTED", "Only original no-target views may omit packets")
                results["empty_" + branch].append(1)
                continue
            need(row["status"] == "STOCK_TRACE_EXACT", "Original exact accepted view required")
            packet = replay.load_packet(torch, prior / "CAPTURE", row, plan, CAPTURE_PROTOCOL, cap)
            samples, packed = collate([(pixels, {key: value for key, value in a.items() if key != ROW_ID})])
            need(samples.tensors.shape == (1, 3, cap.RESOLUTION, cap.RESOLUTION)
                 and not bool(samples.mask.any())
                 and cap.state_digest({"pixels": samples.tensors}) == packet["input_sha256"],
                 "Exact original packed input pixels")
            packed_target, = packed.to_list("cpu", non_blocking=False)
            stats, hashes = observed_exposure(packet, packed_target, fixed, keep, cap, torch)
            results[branch].append(dict(exposure=stats, native_rows=count, excluded_rows=count-int(keep.sum()),
                                        view_has_b_c_sample_change=int(stats["all"]["b_to_c"]["changed_rows"] > 0)))
            replayed.append(hashes)
            del packet
            print(json.dumps(dict(event="P269_EXACT_CAPTURE_VIEW_ACCEPTED", accepted=len(replayed)), sort_keys=True),
                  flush=True)

    summary = {branch: dict(views=len(results[branch]), **sum_dicts(results[branch])) for branch in ("direct", "crop")}
    need(len(replayed) == 30 and all(x["views"] > 0 and x["exposure"]["all"]["rows"] > 0 for x in summary.values())
         and not summary["direct"]["view_has_b_c_sample_change"], "Exact two-branch census and direct negative control")
    encoded_hashes = json.dumps(replayed, sort_keys=True, separators=(",", ":")).encode()
    result = dict(
        status="P269_SSDD_FIT_MATCHER_TARGET_EXPOSURE_COMPLETE", FIT_only=True, DEV=False, TEST=False, AP=False,
        optimizer_updates=0, model_forward_calls=0, reassignment=False, capture_protocol=CAPTURE_PROTOCOL,
        capture_report_sha256=CAPTURE_REPORT, capture_closure_sha256=CAPTURE_CLOSURE,
        historical_exact_cuda_replay_sha256=REPLAY_REPORT, historical_exact_cuda_replay_closure_sha256=REPLAY_CLOSURE,
        source_roster_sha256=sha256(json.dumps(PINNED_SOURCE, sort_keys=True).encode()).hexdigest(),
        fit_bundle_sha256=cap.FIT_BUNDLE_SHA, fit_index_sha256=cap.FIT_INDEX_SHA, fit_gt_sha256=cap.FIT_GT_SHA,
        capture_views=32, exact_input_and_target_packet_views=30, replayed_no_target_views=2,
        no_target_views_by_branch={branch: len(results["empty_" + branch]) for branch in summary},
        archived_input_points_logits_labels_stream_sha256=sha256(encoded_hashes).hexdigest(),
        point_policy="immutable_original_cuda_bfloat16_observation_and_stock_1296_uniform_points",
        common_support="original_A_intersect_stepwise_center_B_intersect_composed_C_on_B_crop_rows_all_nonempty",
        area_policy="original_A_augmented_mask_pixels_less_than_1024",
        branches=summary, elapsed_seconds=perf_counter() - start,
    )
    output.mkdir()
    with (output / "RESULT_SAFE.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-python", type=Path, required=True)
    args = parser.parse_args()
    run(args.prior, args.geometry, args.output, args.expected_python)


if __name__ == "__main__":
    main()
