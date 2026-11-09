"""Replay paired native HRSID FIT training geometry without weights or new data."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import torch

from fit_query_export import INDEX_SHA, need, pinned, write
from reference_resize import center_reference, inverse_scales
from trace_resize import capture_transform

AUG_SHA = "6d1b1b94c14cc8f6c3cf7d5eb70ea8db83a6c0f4c0219c68b4dbf24d773aca14"
COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
REPEATS = 4
BASE_SEED = 520198000
ROW_ID = "_p198_row_id"


def masks_count(masks):
    return np.count_nonzero(masks, axis=(1, 2))


def mask_centers(masks, area):
    y = masks.sum(axis=2, dtype=np.int32) @ np.arange(masks.shape[1], dtype=np.float64)
    x = masks.sum(axis=1, dtype=np.int32) @ np.arange(masks.shape[2], dtype=np.float64)
    return np.stack((y, x), axis=1) / area[:, None]


def paired_measurements(legacy, center, reference, small, scales):
    need(legacy.shape == center.shape == reference.shape and legacy.shape[0] == len(small),
         "Exact common-row raster shapes")
    na, nb, nr = [masks_count(x) for x in (legacy, center, reference)]
    valid = (na > 0) & (nb > 0) & (nr > 0)
    missing_ref = int(((na > 0) & (nb > 0) & (nr == 0)).sum())
    result = dict(count=int(valid.sum()), reference_empty_for_common_survivors=missing_ref)
    a, b, r = (x[valid] for x in (legacy, center, reference))
    na, nb, nr = (x[valid] for x in (na, nb, nr))
    orig = mask_centers(r, nr)
    d_a = (mask_centers(a, na) - orig) * np.abs(scales)
    d_b = (mask_centers(b, nb) - orig) * np.abs(scales)
    ia, ib = masks_count(a & r), masks_count(b & r)
    changed = masks_count(a ^ b)
    select = np.asarray(small, dtype=bool)[valid]
    for key, chosen in (("small", select), ("non_small", ~select)):
        result[key] = dict(count=int(chosen.sum()), legacy_y=float(d_a[chosen, 0].sum()),
                           legacy_x=float(d_a[chosen, 1].sum()), center_y=float(d_b[chosen, 0].sum()),
                           center_x=float(d_b[chosen, 1].sum()),
                           legacy_abs_y=float(np.abs(d_a[chosen, 0]).sum()),
                           center_abs_y=float(np.abs(d_b[chosen, 0]).sum()),
                           legacy_iou=float((ia[chosen]/(na[chosen]+nr[chosen]-ia[chosen])).sum()),
                           center_iou=float((ib[chosen]/(nb[chosen]+nr[chosen]-ib[chosen])).sum()),
                           masks_changed=int((changed[chosen] > 0).sum()),
                           changed_pixels=int(changed[chosen].sum()))
    return result


def aggregate_groups(views, area_key):
    grouped = defaultdict(Counter)
    for view in views:
        values = view["metrics"][area_key]
        for key, value in values.items():
            grouped[view["group"]][key] += value
    good = [value for _, value in sorted(grouped.items()) if value["count"] > 0]
    metrics = ("legacy_y", "center_y", "legacy_x", "center_x", "legacy_abs_y", "center_abs_y", "legacy_iou", "center_iou")
    per_group = np.array([[row[k]/row["count"] for k in metrics] for row in good]).reshape(-1, len(metrics))
    result = dict(groups=len(good), rows=sum(row["count"] for row in good),
                  masks_changed=sum(row["masks_changed"] for row in good),
                  changed_pixels=sum(row["changed_pixels"] for row in good))
    if len(good):
        result.update(dict(zip(metrics, per_group.mean(axis=0).tolist(), strict=True)))
    return result, per_group


def summarize(views, *, primary):
    rows = Counter()
    for view in views:
        rows.update(view["rows"])
    result = dict(views=len(views), groups=len({v["group"] for v in views}), membership=dict(rows),
                  small=aggregate_groups(views, "small")[0], non_small=aggregate_groups(views, "non_small")[0],
                  by_path={})
    for path in ("direct", "crop"):
        for flip in (False, True):
            partition = [v for v in views if v["path"] == path and v["flip"] == flip]
            result["by_path"][f"{path}_{'flipped' if flip else 'plain'}"] = dict(
                views=len(partition), small=aggregate_groups(partition, "small")[0],
                non_small=aggregate_groups(partition, "non_small")[0])
    if primary:
        small, table = aggregate_groups(views, "small")
        need(len(table) > 0 and rows["legacy_nonempty"] > 0, "Primary small group and row denominator")
        idx = np.random.default_rng(5219).integers(0, len(table), size=(4096, len(table)))
        bootstrap = table[idx, :2].mean(axis=1)
        delta = abs(small["legacy_y"]) - abs(small["center_y"])
        distribution = np.abs(bootstrap[:, 0]) - np.abs(bootstrap[:, 1])
        interval = np.quantile(distribution, [.025, .975], method="linear").tolist()
        lost_rate = rows["legacy_nonempty_lost_center"] / rows["legacy_nonempty"]
        iou_delta = small["center_iou"] - small["legacy_iou"]
        gate = (small["groups"] >= 30 and small["legacy_y"] >= .25 and delta >= .25 and interval[0] > 0
                and iou_delta >= -.005 and lost_rate <= .005)
        result.update(primary_abs_y_reduction=delta, primary_abs_y_reduction_interval=interval,
                      primary_reference_iou_delta=iou_delta, legacy_nonempty_lost_fraction=lost_rate,
                      gate=gate, decision="ADVANCE_CENTER_RESIZE_TO_PAIRED_TRAINING_QUALIFICATION" if gate
                      else "STOP_CENTER_RESIZE_ADVANCEMENT_FROM_THIS_REPLAY")
    return result


def normalized_trace(trace):
    steps = []
    for event in trace:
        if event["op"] == "hflip" and not event["applied"]:
            continue
        step = dict(kind=event["op"], before=list(event["in_hw"]), after=list(event["out_hw"]))
        if event["op"] == "crop":
            step.update(top=event["top"], left=event["left"])
        steps.append(step)
    need([step["kind"] for step in steps] in (["resize"], ["resize", "hflip"],
         ["resize", "crop", "resize"], ["resize", "crop", "resize", "hflip"]), "Exact native augmentation paths")
    return steps


def rng_state():
    n = np.random.get_state()
    return random.getstate(), n[0], n[1].tobytes(), n[2:], torch.get_rng_state().numpy().tobytes()


def replay_view(audit, source, image, raw, seed):
    images, targets, traces, rng = [], [], [], []
    for mode in ("native", "center"):
        source.set_seed(seed)
        target_input = {key: value.clone() if torch.is_tensor(value) else value for key, value in raw.items()}
        with capture_transform(source._torchvision, mode=mode) as trace:
            im, target = audit.transform(image.copy(), target_input)
        images.append(im)
        targets.append(target)
        traces.append(trace)
        rng.append(rng_state())
    need(traces[0] == traces[1] and rng[0] == rng[1] and torch.equal(*images)
         and images[0].shape == (3, 576, 576) and images[0].dtype == torch.float32,
         "Exact shared augmentation trace, RNG and normalized image tensor")
    a, b = targets
    for target in targets:
        source.target_summary(target, 576, int(raw["image_id"].item()), normalized=True)
    aid, bid = (target[ROW_ID].numpy() for target in targets)
    need(len(set(aid)) == len(aid) and len(set(bid)) == len(bid)
         and all(0 <= r < len(raw["labels"]) for r in (*aid, *bid)), "Native unique raw target row identity")
    common, pa, pb = np.intersect1d(aid, bid, assume_unique=True, return_indices=True)
    need(set(a) == set(b) and all(torch.equal(a[k][pa], b[k][pb]) for k in ("boxes", "labels", "area", "iscrowd"))
         and all(torch.equal(a[k], b[k]) for k in ("image_id", "size", "orig_size")),
         "Every common nonmask field exactly matches")
    am, bm = (target["masks"].numpy() for target in targets)
    nonempty_a, nonempty_b = (masks_count(x) > 0 for x in (am, bm))
    a_live, b_live = set(aid[nonempty_a]), set(bid[nonempty_b])
    steps = normalized_trace(traces[0])
    reference = center_reference(raw["masks"][common].numpy(), steps)
    small = raw["area"][common].numpy() < 1024
    metrics = paired_measurements(am[pa], bm[pb], reference, small, inverse_scales(steps))
    membership = dict(raw_rows=len(raw["labels"]), legacy_present=len(aid), center_present=len(bid),
                      legacy_empty=int((~nonempty_a).sum()), center_empty=int((~nonempty_b).sum()),
                      legacy_nonempty=len(a_live), center_nonempty=len(b_live),
                      legacy_nonempty_lost_center=len(a_live - b_live), center_nonempty_added=len(b_live - a_live),
                      legacy_only_present=len(set(aid)-set(bid)), center_only_present=len(set(bid)-set(aid)),
                      identical_retained_ids_views=int(np.array_equal(aid, bid)),
                      common_nonempty=len(a_live & b_live),
                      common_reference_empty=metrics["reference_empty_for_common_survivors"])
    need(membership["common_nonempty"] == metrics["count"] + membership["common_reference_empty"],
         "Exact common nonempty and reference census")
    return dict(rows=membership, metrics=metrics, path="crop" if len([s for s in steps if s["kind"]=="resize"])==2 else "direct",
                flip=steps[-1]["kind"] == "hflip", steps=steps)


def run(args):
    start = perf_counter()
    source = ModuleType("_qualified_native_fit_augmentation")
    source.__file__ = str(args.audit_source)
    exec(compile(pinned(args.audit_source, AUG_SHA, 65536), str(args.audit_source), "exec"), source.__dict__)
    sys.addaudithook(source.offline)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.version_info[:3] == (3, 10, 13)
         and not torch.cuda.is_available() and all(source.version(k) == v for k, v in source.VERSIONS.items())
         and all(sha256(Path(m.__file__).read_bytes()).hexdigest() == v for m, v in source.SOURCE_SHA.items()),
         "Exact frozen CPU package and native augmentation sources")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cohort = json.loads(pinned(args.cohort, COHORT_SHA, 1024**2))
    data = json.loads(pinned(args.train / "_annotations.coco.json", INDEX_SHA, 32 * 1024**2))
    fit_images = {row["id"]: row for row in data["images"]}
    anns = defaultdict(list)
    for ann in data["annotations"]:
        anns[ann["image_id"]].append(ann)
    need(len(fit_images) == 2937 and len(data["annotations"]) == 9260 and len(cohort["images"]) == 114
         and cohort["fit_only"] is True, "Exact qualification and 114 locked FIT images")
    audit = source.NativeInputAudit()
    args.output.mkdir()
    layers = {"core": [], "stress": []}
    step_pairs = Counter()
    with (args.output / "VIEWS_PRIVATE.jsonl").open("x", encoding="utf8", newline="\n") as stream:
        for ordinal, row in enumerate(cohort["images"]):
            iid, name = row["image_id"], row["native_basename"]
            need(fit_images[iid]["file_name"] == name and row["core"] != row["stress"], "Exact frozen FIT row")
            png = pinned(args.train / name, row["encoded_sha256"], 4 * 1024**2)
            image, scales = source.io_utils.decode_pil_image_bytes(png, draft_size=None)
            need(image.mode == "RGB" and image.size == (800, 800) and scales == (1., 1.)
                 and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
                 "Exact qualified image and pixels")
            im, raw = audit.convert(image, dict(image_id=iid, annotations=anns[iid]))
            source.target_summary(raw, 800, iid, normalized=False)
            need(not bool(source.mask_empty(raw).any()), "Qualified original FIT masks are nonempty")
            raw[ROW_ID] = torch.arange(len(raw["labels"]), dtype=torch.int64)
            layer = "core" if row["core"] else "stress"
            for rep in range(REPEATS):
                seed = BASE_SEED + ordinal * REPEATS + rep
                view = replay_view(audit, source, im, raw, seed)
                view.update(group=row["group"], layer=layer)
                layers[layer].append(view)
                step_pairs.update(f"{s['before'][0]}->{s['after'][0]}" for s in view["steps"] if s["kind"] == "resize")
                stream.write(json.dumps(dict(view, ordinal=ordinal, seed=seed), sort_keys=True, allow_nan=False)+"\n")
            if (ordinal + 1) % 10 == 0:
                print(json.dumps(dict(event="HRSID_RESIZE_PHASE_REPLAY_PROGRESS", images=ordinal+1)), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    need(len(layers["core"]) == 436 and len(layers["stress"]) == 20, "Exact fixed native replay views")
    result = dict(status="ACTUAL_HRSID_FIT_PAIRED_RESIZE_PHASE_REPLAY", protocol=args.protocol,
                  cohort_sha256=COHORT_SHA, fit_index_sha256=INDEX_SHA, groups_main=109, images=114, views=456,
                  resize_operations=dict(step_pairs),
                  core=summarize(layers["core"], primary=True), stress=summarize(layers["stress"], primary=False),
                  exact_shared_geometry_rng_images=True, exact_common_nonmask_fields=True,
                  elapsed_seconds=perf_counter() - start, FIT=True, DEV=False, TEST=False, AP=False,
                  forward_calls=0, optimizer_updates=0)
    write(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cohort", "train", "audit-source", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    need(re.fullmatch(r"[0-9a-f]{40}", args.protocol) and args.output.is_absolute()
         and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists()
         and not args.output.is_relative_to(args.train.parent), "Exact prelock and private fresh output outside FIT")
    run(args)


if __name__ == "__main__":
    main()
