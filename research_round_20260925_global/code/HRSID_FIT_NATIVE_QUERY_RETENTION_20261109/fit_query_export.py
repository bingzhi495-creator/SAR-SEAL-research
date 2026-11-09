"""Freeze a FIT-only cohort and export aligned native queries without opening GT."""
import argparse
from collections import Counter
from functools import partial
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType

INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
LAYOUT_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
ASSIGNMENT_SHA = "c92b85dc278bc49cc8e34634fd7d354018844ae2a2692ea451b014a082bbaa5a"
SALT = b"HRSID-FIT-Q164-P192-v1\0"
WIDTH = 164


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, digest, limit):
    need(path.is_absolute() and path.resolve(strict=True) == path, "Exact closed input path")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    need(0 < len(data) <= limit and sha256(data).hexdigest() == digest, "Exact closed input bytes")
    return data


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def make_cohort(args):
    data = json.loads(pinned(args.index, INDEX_SHA, 32 * 1024**2))
    layout = json.loads(pinned(args.layout, LAYOUT_SHA, 2 * 1024**2))
    assignment = json.loads(pinned(args.assignment, ASSIGNMENT_SHA, 2 * 1024**2))
    roles = {r["image_id"]: r for r in assignment if r["split"] == "fit"}
    image_rows = {r["image_id"]: r for r in layout["images"]}
    images = {r["id"]: r for r in data["images"]}
    need(len(roles) == len(image_rows) == len(images) == 2937 and roles.keys() == image_rows.keys() == images.keys()
         and len(data["annotations"]) == 9260 and layout["native_index_sha256"] == INDEX_SHA,
         "Exact qualified FIT partition and image identities")
    counts = Counter(r["image_id"] for r in data["annotations"])
    need(counts.keys() <= images.keys() and max(counts.values()) == 163, "Exact FIT annotation counts")
    groups = {}
    for image_id, item in image_rows.items():
        name = item["native_basename"]
        group = roles[image_id]["group"]
        need(re.fullmatch(r"P\d+_[A-Za-z0-9_]+\.png", name) and group == name.split("_", 1)[0]
             and Path(roles[image_id]["file_name"]).stem == Path(name).stem and images[image_id]["file_name"] == name
             and images[image_id]["width"] == images[image_id]["height"] == 800,
             "Closed roster, official P-prefix and native PNG agree")
        order = (sha256(SALT + bytes.fromhex(item["encoded_sha256"])).digest(), name)
        if group not in groups or order < groups[group][0]:
            groups[group] = (order, image_id)
    core = {pair[1] for pair in groups.values()}
    stress = {image_id for image_id, count in counts.items() if count > 100}
    need(len(core) == len(groups) == 109 and len(stress) == 5, "Prelocked FIT cohort and dense layer")
    rows = [
        {**{k: image_rows[i][k] for k in ("image_id", "native_basename", "encoded_bytes",
                                          "encoded_sha256", "decoded_rgb_sha256")},
         "group": roles[i]["group"], "core": i in core, "stress": i in stress}
        for i in sorted(core | stress)
    ]
    payload = dict(protocol=args.protocol, images=rows, core_images=109, stress_images=5,
                   shared_images=len(core & stress), fit_only=True)
    write(args.output, payload)
    print(json.dumps(dict(status="HRSID_FIT_COHORT_LOCKED", images=len(rows), core_images=109,
                          stress_images=5, shared_images=len(core & stress), sha256=sha256(encoded(payload)).hexdigest())),
          flush=True)


def aligned_query_rows(raw, hw, *, torch, np, codec, processor):
    class ObservedPostProcess(processor):
        def _select_topk(self, logits):
            self.selection = super()._select_topk(logits)
            return self.selection

    logits, boxes, masks = (raw[key] for key in ("pred_logits", "pred_boxes", "pred_masks"))
    need(tuple(logits.shape) == (1, WIDTH, 2) and tuple(boxes.shape) == (1, WIDTH, 4)
         and tuple(masks.shape) == (1, WIDTH, 144, 144)
         and all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in (logits, boxes, masks)),
         "Exact finite two-slot FP32 raw outputs")
    size = torch.tensor([hw], dtype=torch.int64, device=logits.device)
    joint = ObservedPostProcess(num_select=100, upsample_masks_to_image_size=True)
    ship = ObservedPostProcess(num_select=WIDTH, upsample_masks_to_image_size=True)
    native, = joint(raw, size, score_threshold=None)
    ship_raw = dict(raw, pred_logits=logits[:, :, :1].contiguous())
    all_ship, = ship(ship_raw, size, score_threshold=None)
    joint_scores, joint_labels, joint_ids = (x[0] for x in joint.selection)
    ship_scores, ship_labels, ship_ids = (x[0] for x in ship.selection)
    probs = logits.sigmoid()
    expected = torch.argsort(probs.flatten(), descending=True, stable=True)[:100]
    ship_expected = torch.argsort(probs[0, :, 0], descending=True, stable=True)
    need(torch.equal(joint_ids, expected // 2) and torch.equal(joint_labels, expected % 2)
         and torch.equal(joint_scores, probs.flatten().index_select(0, expected))
         and torch.equal(ship_ids, ship_expected) and not bool(ship_labels.any())
         and torch.equal(ship_scores, probs[0, :, 0].index_select(0, ship_expected))
         and torch.equal(native["scores"], joint_scores) and torch.equal(native["labels"], joint_labels)
         and tuple(native["masks"].shape) == (100, 1, *hw)
         and tuple(all_ship["masks"].shape) == (WIDTH, 1, *hw)
         and native["masks"].dtype == all_ship["masks"].dtype == torch.bool,
         "Native two-slot and ship-only stable rank, score and raster invariants")
    raw_order = torch.argsort(ship_ids)
    inverse = raw_order.index_select(0, joint_ids)
    need(torch.equal(native["masks"], all_ship["masks"].index_select(0, inverse)),
         "Bitwise original-grid mask alignment for all native rows")

    def rle(plane):
        item = codec.encode(np.asfortranarray(plane, dtype=np.uint8))
        return dict(size=item["size"], counts=item["counts"].decode("ascii"))

    planes = all_ship["masks"][:, 0].detach().cpu().numpy()
    by_query = [None] * WIDTH
    for query, plane in zip(ship_ids.tolist(), planes, strict=True):
        by_query[query] = rle(plane)
    positions = (joint_labels == 0).nonzero(as_tuple=True)[0]
    native_ids = joint_ids.index_select(0, positions).tolist()
    native_planes = native["masks"].index_select(0, positions)[:, 0].detach().cpu().numpy()
    need(all(by_query[query] == rle(plane) for query, plane in zip(native_ids, native_planes, strict=True)),
         "Native mapped RLE byte equality")
    return dict(native_ids=native_ids, native_ranks=(positions + 1).tolist(),
                ship_scores=probs[0, :, 0].tolist(), all_rles=by_query)


def load_source(path, digest):
    module = ModuleType("_hrsid_pinned_filtered_evaluation")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(pinned(path, digest, 65536), str(path), "exec"), module.__dict__)
    return module


def run_export(args):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    need(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
         and version("rfdetr") == "1.12.0" and version("pycocotools") == "2.0.11"
         and version("Pillow") == "12.3.0" and np.__version__ == "1.26.4", "Pinned native runtime")
    wrapper = load_source(args.wrapper, args.wrapper_sha256)
    base = wrapper.load_original_evaluator(args.base_source)
    for name, digest in base.MODEL_SOURCES.items():
        base.pinned_bytes(Path(import_module(name).__file__), digest, 1024**2)
    accepted = wrapper.read_acceptance(args.acceptance, args.acceptance_sha256)
    base.check_endpoint = partial(wrapper.check_filtered_endpoint, original_check=base.check_endpoint,
                                  qualification=accepted)
    cohort = json.loads(pinned(args.cohort, args.cohort_sha256, 1024**2))
    rows = cohort["images"]
    need(cohort["protocol"] == args.protocol and cohort["fit_only"] is True
         and sum(row["core"] for row in rows) == 109 and sum(row["stress"] for row in rows) == 5
         and len(rows) == 114 - cohort["shared_images"] and len({r["image_id"] for r in rows}) == len(rows),
         "Exact locked GT-free FIT manifest")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One isolated CUDA device")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    need(free >= base.CAP, "GPU admission floor")
    torch.cuda.set_per_process_memory_fraction(min(.35, base.CAP / total), device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    start = perf_counter()
    model, config, train, _ = base.load_model(args.checkpoint, accepted.checkpoint_sha256, torch,
                                              args.output, wrapper.TRAINING_PROTOCOL)
    model = model.to(device).eval()
    transform = make_coco_transforms_square_div_64(
        "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=config.patch_size, num_windows=config.num_windows, aug_config=None,
        scale_jitter=train.scale_jitter, gpu_postprocess=False,
    )
    args.output.mkdir()
    torch.cuda.reset_peak_memory_stats(device)
    digest = sha256()
    mapped = empty = 0
    with (args.output / "QUERIES_PRIVATE.jsonl").open("xb") as stream, torch.inference_mode():
        for ordinal, row in enumerate(rows, 1):
            name = row["native_basename"]
            need(re.fullmatch(r"P\d+_[A-Za-z0-9_]+\.png", name) and name.split("_", 1)[0] == row["group"],
                 "Admitted FIT basename")
            b = pinned(args.png_root / name, row["encoded_sha256"], 4 * 1024**2)
            need(len(b) == row["encoded_bytes"], "Exact qualified PNG length")
            image, scales = decode_pil_image_bytes(b, draft_size=None)
            need(image.mode == "RGB" and image.size == (800, 800) and scales == (1., 1.)
                 and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
                 "Exact qualified FIT PNG and decoded pixels")
            tensor, target = transform(image, None)
            need(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32,
                 "Native GT-free image transform")
            sample = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            need(sample.mask is not None and not bool(sample.mask.any()), "Native input without padding")
            raw = model(sample.to(device))
            query = aligned_query_rows(raw, (800, 800), torch=torch, np=np, codec=codec, processor=PostProcess)
            mapped += len(query["native_ids"])
            empty += int(np.count_nonzero(codec.area(query["all_rles"]) == 0))
            line = encoded(dict(image_id=row["image_id"], **query))
            stream.write(line)
            digest.update(line)
            if ordinal % 20 == 0:
                print(json.dumps(dict(event="HRSID_FIT_Q164_PROGRESS", images=ordinal)), flush=True)
    torch.cuda.synchronize(device)
    report = dict(status="HRSID_FIT_Q164_EXPORT_COMPLETE", protocol=args.protocol, cohort_sha256=args.cohort_sha256,
                  checkpoint_sha256=accepted.checkpoint_sha256, predictions_sha256=digest.hexdigest(),
                  images=len(rows), forward_calls=len(rows), native_joint_rows=100 * len(rows),
                  native_mapped_rows=mapped, all_ship_queries=WIDTH * len(rows), all_empty=empty,
                  native_rows_bitwise_verified=True, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  elapsed_seconds=perf_counter() - start, GT_opened=False, DEV=False, TEST=False, AP=False)
    write(args.output / "EXPORT_SAFE.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    cohort = modes.add_parser("cohort")
    cohort.set_defaults(run=make_cohort)
    for key in ("index", "layout", "assignment"):
        cohort.add_argument("--" + key, required=True, type=Path)
    export = modes.add_parser("export")
    export.set_defaults(run=run_export)
    for key in ("cohort", "wrapper", "base-source", "acceptance", "checkpoint", "png-root"):
        export.add_argument("--" + key, required=True, type=Path)
    for key in ("cohort-sha256", "wrapper-sha256", "acceptance-sha256"):
        export.add_argument("--" + key, required=True)
    for sub in (cohort, export):
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--protocol", required=True)
    args = parser.parse_args()
    need(re.fullmatch(r"[0-9a-f]{40}", args.protocol) and args.output.is_absolute()
         and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists(),
         "Prelocked protocol and fresh output")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("HRSID FIT query audit is offline")
    sys.addaudithook(offline)
    args.run(args)


if __name__ == "__main__":
    main()
