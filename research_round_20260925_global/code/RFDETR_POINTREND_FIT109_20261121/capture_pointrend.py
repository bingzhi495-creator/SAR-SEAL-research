"""Capture frozen RF-DETR spatial maps and aligned raw FIT109 mask logits on CPU."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType


PILOT = "b49b9cd10a906c9141caceeb68ace08499305505986b7ea705579bb7b604b8a0"
FORMAL = "9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e"
ACCEPTANCE = "7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f"
PRELOCK = "648db29c9f841f33eb54fdcb6d883c743442940dcd7cea475ad1a91b424c1aa4"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def eligible_arrays(block, closed_logits, observed, spot, area, np, torch):
    """Return the complete eligible subset in its original joint-rank order."""
    mapped = spot.native_candidate_rows(block)
    eligible = []
    for row in mapped:
        n_pixels = int(area(row["rle"]))
        if 32 <= n_pixels < 1024:
            eligible.append((row["rank"], row["query_id"], row["score"], n_pixels))
    proof = block["logit_capture"]["candidates"]
    keys = ("native_rank", "query_id", "score", "predicted_area")
    need(len(proof) == len(eligible) and all(
        p["logit_row"] == i and tuple(p[key] for key in keys) == r
        for i, (p, r) in enumerate(zip(proof, eligible))
    ), "Stored rank / query / score / area identity")
    ids = np.array([r[1] for r in eligible], dtype=np.int64)
    need(tuple(observed.shape) == (164, 144, 144) and observed.device.type == "cpu"
         and observed.dtype == torch.float32, "Full observed CPU FP32 query logits")
    logits = observed.detach().index_select(0, torch.from_numpy(ids)).contiguous().numpy()
    need(closed_logits.dtype == np.float32 and np.array_equal(logits, closed_logits)
         and np.isfinite(logits).all(), "Previously admitted retained logits")
    native_boxes = spot.author_boxes(block["raw_boxes_cxcywh"], torch)[ids]
    boxes = native_boxes / np.float32(800)
    need(boxes.dtype == np.float32 and np.isfinite(boxes).all()
         and np.all((boxes >= 0) & (boxes <= 1)) and np.all(boxes[:, 2:] >= boxes[:, :2]),
         "Canonical clipped author boxes normalized to the 800-square source extent")
    return dict(image_id=np.int64(block["image_id"]),
                ranks=np.array([r[0] for r in eligible], dtype=np.int64),
                query_ids=ids, areas=np.array([r[3] for r in eligible], dtype=np.int64),
                scores=np.array([r[2] for r in eligible], dtype=np.float64),
                boxes_xyxy=boxes, raw_logits=logits), len(mapped)


def capture_image(args, row, context):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    item, ordinal, old = row
    cap, pilot, spot, helper, basis, core, post, model, transform, cfg = context
    image = cap.original_image(args, item, core, None, None)
    tensor, target = transform(image, None)
    need(target is None and tensor.shape == (3,576,576) and tensor.dtype == torch.float32,
         "Original 576-square transform")
    sample = nested_tensor_from_tensor_list([tensor], block_size=cfg.patch_size * cfg.num_windows)
    need(sample.mask is not None and not bool(sample.mask.any()), "Unpadded stock forward")
    with torch.inference_mode():
        with basis.capture_frozen_basis(model.segmentation_head, helper.capture_final_query) as obs:
            raw = model(sample)
    captured = obs.seal(raw["pred_masks"])
    spatial = captured.spatial[0].contiguous().numpy()
    need(spatial.shape == (256,144,144) and spatial.dtype == np.float32 and np.isfinite(spatial).all(),
         "Verified full 256x144x144 projected spatial basis")
    out = post.aligned_query_rows(raw, (800,800), torch=torch, np=np, codec=codec, processor=PostProcess)
    probability = raw["pred_logits"][0].sigmoid().flatten()
    top = torch.argsort(probability, descending=True, stable=True)[:100]
    pilot.compare_original(old, out, raw["pred_logits"][0].tolist(), raw["pred_boxes"][0].tolist(),
                           (top // 2).tolist(), (top % 2).tolist(), probability.index_select(0, top).tolist())
    ref = old["logit_capture"]
    need(ref["relative_path"] == f"RAW_LOGITS/{ordinal:03d}.npy", "Original ordinal of retained logits")
    original = spot.pinned(args.capture / ref["relative_path"], ref["sha256"], 9 * spot.MIB)
    closed = np.load(BytesIO(original), allow_pickle=False)
    need(ref["bytes"] == len(original) and ref["shape"] == list(closed.shape), "Sealed raw archive shape")
    rows, mapped = eligible_arrays(old, closed, raw["pred_masks"][0], spot, codec.area, np, torch)
    return spatial, rows, mapped


def write_numpy(path, writer, *args, **kwargs):
    """Close the exclusive per-image file, then digest it using bounded memory."""
    with path.open("xb") as dest:
        writer(dest, *args, **kwargs)
        dest.flush()
        os.fsync(dest.fileno())
    digest = sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "capture", "acceptance", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "Run the admitted CPU-only interpreter with -I -B and CUDA_VISIBLE_DEVICES=-1")
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Local FIT-only capture")
    sys.addaudithook(offline)
    args.root = args.root.resolve(strict=True)
    need(args.output.is_absolute() and args.output.parent == args.output.parent.resolve(strict=True)
         and not args.output.exists(), "New private output directory")
    args.code = args.root / "github_release_20261011/research_round_20260925_global/code"
    args.mode = "fit"
    args.acceptance_sha256 = ACCEPTANCE
    args.source_cohort = args.root / "HRSID_FIT_NATIVE_QUERY_RETENTION_ACTUAL_PRIVATE_20261109_01/COHORT_PRIVATE.json"
    module_dir = args.code / "RFDETR_QUERY_FEATURE_FIT109_20261121"
    pilot = ModuleType("_pr_closed_core4")
    pilot.__file__ = str(module_dir / "run_core4.py")
    source = Path(pilot.__file__).read_bytes()
    need(sha256(source).hexdigest() == PILOT, "Pinned ordinary CORE4 source")
    exec(compile(source, pilot.__file__, "exec"), pilot.__dict__)
    need(args.protocol_sha256 == PRELOCK, "Fixed FIT109 PointRend prelock")
    protocol = json.loads(pilot.pinned(args.protocol, PRELOCK))
    need(protocol["protocol"] == "RFDETR_POINTREND_FIT109_20261121_V1"
         and protocol["phase"] == "FIT_ONLY", "Fixed FIT phase")
    import numpy as np
    import torch
    need(sys.version_info[:3] == (3,10,13) and torch.__version__ == "2.10.0+cu130"
         and np.__version__ == "1.26.4" and not torch.cuda.is_initialized(), "Previously admitted numerical runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    modules = {Path(name).name: pilot.load(args.code / name, digest, "_pr_" + Path(name).stem)
               for name, digest in pilot.PINS.items()
               if Path(name).name in ("capture.py", "run_spot.py", "cached_final_query.py")}
    cap, spot = modules["capture.py"], modules["run_spot.py"]
    formal = pilot.load(args.code / "RFDETR_MATCHED_POINT_FORMAL_20261120/run_fit109.py", FORMAL, "_pr_formal")
    basis = pilot.load(module_dir / "basis_adapter.py", pilot.BASIS_SHA, "_pr_basis")
    roster, receipt = formal.read_population(args, spot)
    need(len(roster) == 109 and [r[1] for r in roster] == list(range(1,110))
         and len({r[0]["image_id"] for r in roster}) == 109, "Exact fixed FIT109")
    args.output.mkdir()
    core = cap.load_core(args.root)
    base, post = cap.setup(args, core)
    model, transform, cfg = cap.model_and_transform(args, core, base, torch)
    context = (cap, pilot, spot, modules["cached_final_query.py"], basis, core, post, model, transform, cfg)
    images = []
    total_rows = total_mapped = total_zero = 0
    for row in roster:
        item, ordinal, _ = row
        spatial, arrays, mapped = capture_image(args, row, context)
        n = len(arrays["query_ids"])
        total_rows += n
        total_mapped += mapped
        boxes = arrays["boxes_xyxy"]
        total_zero += int(np.any(boxes[:,2:] == boxes[:,:2], axis=1).sum())
        images.append(dict(ordinal=ordinal, image_id=item["image_id"], group=item["group"], rows=n,
                           spatial=write_numpy(args.output / f"SPATIAL_{ordinal:03d}_PRIVATE.npy",
                                               np.save, spatial, allow_pickle=False),
                           row_archive=write_numpy(args.output / f"ROWS_{ordinal:03d}_PRIVATE.npz",
                                                   np.savez, **arrays)))
        del spatial, arrays
        if ordinal == 1 or ordinal % 10 == 0:
            print(json.dumps(dict(event="FIT109_POINTREND_CAPTURE_PROGRESS", images=ordinal,
                                  eligible_rows=total_rows)), flush=True)
    need(total_rows == receipt["same_device_raster_exact"] == 6738 and total_mapped == receipt["native_mapped_rows"]
         == 10488 and total_zero == 262 and not torch.cuda.is_initialized(), "Exact complete fixed population")
    index_sha = spot.write(args.output / "INDEX_PRIVATE.json", dict(images=images))
    result = dict(status="RFDETR_POINTREND_FIT109_CARRIER_CAPTURE_COMPLETE", FIT=True, DEV=False, TEST=False,
                  AP=False, GT_opened=False, head_updates=0, model_forwards=len(images), images=len(images),
                  mapped=total_mapped, eligible_rows=total_rows, zero_area_author_boxes=total_zero,
                  index_sha256=index_sha, protocol_sha256=args.protocol_sha256, raw_acceptance_sha256=ACCEPTANCE,
                  exact_original_candidates_and_masks=True, exact_original_eligible_logits=True,
                  exact_all_query_basis_contraction=True)
    digest = spot.write(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest)), flush=True)


if __name__ == "__main__":
    main()
