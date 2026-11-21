"""Capture frozen RF-DETR query/spatial features on the original FIT109 population."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


CORE4_SHA = "b49b9cd10a906c9141caceeb68ace08499305505986b7ea705579bb7b604b8a0"
CORE4_CLOSURE_SHA = "896963c572cfb482dbb005aaf01b721ffc156f88666ecd1377cbf3bb2d96e1fa"
FORMAL_SHA = "9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e"
RAW_ACCEPTANCE_SHA = "7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f"
PRELABEL_SHA = "33fc0e63fa5deff68aa384e03a3f00fd78430e63a7680b7a5a8025e33422acf4"
FEATURE_SHA = "1cea85eb8c30b11bfa847292c55d2231fb4d16dd2a551c3da52ddffdd8d3830e"
COUNTS = dict(candidate_point_sets=6421, eligible=6738, exact_original_masks=6738, mapped=10488,
              original_empty_masks=2017, pairs=203903, zero_area_author_boxes=262)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def verify_prior_prelabel(counts, feature_digest, encode):
    need(counts == COUNTS and feature_digest == FEATURE_SHA, "Exact previously closed FIT109 features")
    raw = encode(dict(status="RFDETR_MATCHED_POINTS_FIT109_PRELABEL", features=counts,
                      feature_sha256=feature_digest, accepted_capture_sha256=RAW_ACCEPTANCE_SHA,
                      FIT=True, DEV=False, TEST=False, GT_opened=False))
    need(sha256(raw).hexdigest() == PRELABEL_SHA, "Byte-identical historical FIT109 prelabel")
    return raw


def verify_arrays(arrays, image_id, points, np):
    """Check archive identity, exact native endpoint order, and original point features."""
    ranks = np.asarray([row["rank"] for row in points], dtype=int)
    qids = np.asarray([row["query_id"] for row in points], dtype=int)
    expected = dict(image_id=np.asarray(image_id), ranks=ranks, query_ids=qids)
    if points:
        expected.update(coords=np.concatenate([row["coordinates"] for row in points]),
                        old14=np.concatenate([row["features"] for row in points]),
                        layers=np.concatenate([row["layers"] for row in points]),
                        offsets=np.concatenate(([0], np.cumsum([len(row["coordinates"]) for row in points]))))
    else:
        expected.update(coords=np.empty((0, 2), np.int32), old14=np.empty((0, 2, 14), np.float32),
                        layers=np.empty((0, 2), np.int8), offsets=np.array([0], np.int64))
    need(set(arrays) == set(expected) | {"spatial", "projected_queries"}, "Exact endpoint archive fields")
    for key, value in expected.items():
        actual = arrays[key]
        need(actual.dtype == value.dtype and actual.shape == value.shape
             and actual.tobytes(order="C") == value.tobytes(order="C"), "Exact original bytes for " + key)
    for key, shape in (("spatial", (len(expected["coords"]), 2, 256)), ("projected_queries", (len(points), 256))):
        value = arrays[key]
        need(value.dtype == np.float32 and value.shape == shape and bool(np.isfinite(value).all()),
             "Finite original FP32 basis for " + key)


def signature(ordinal, raw):
    return dict(ordinal=ordinal, sha256=sha256(raw).hexdigest(), bytes=len(raw))


def archive_name(ordinal):
    return f"NATIVE_ENDPOINT_BASIS_{ordinal:02d}_PRIVATE.npz"


def admit_core4(args, roster, data, pilot, spot, np, residual):
    flags = (args.core4_output, args.core4_result_sha256, args.core4_closure, args.core4_closure_sha256)
    if not any(flags):
        return [], {}
    need(all(flags), "Supply all four CORE4 reuse arguments together")
    need(args.core4_closure_sha256 == CORE4_CLOSURE_SHA, "Qualified independent CORE4 closure")
    acceptance = json.loads(pilot.pinned(args.core4_closure, args.core4_closure_sha256))
    report = json.loads(pilot.pinned(args.core4_output/"RESULT_SAFE.json", args.core4_result_sha256))
    need(acceptance["status"] == "RFDETR_FIT_QUERY_BASIS_CORE4_INDEPENDENTLY_ACCEPTED"
         and acceptance["result_sha256"] == args.core4_result_sha256
         and acceptance["original_os_exit"] == 0 and acceptance["job_drained"] is True
         and acceptance["independent_readback_accepted"] is True
         and acceptance["entry_sha256"] == report["entry_sha256"] == CORE4_SHA
         and acceptance["private_output_sha256"] == report["private_output_sha256"]
         and acceptance["private_file_count"] == report["private_file_count"] == 4
         and acceptance["model_forwards"] == report["model_forwards"] == 4, "Closed CORE4 reuse witness")
    need(report["status"] == "RFDETR_FIT_QUERY_BASIS_CORE4_CAPTURE_COMPLETE"
         and all(report[k] is True for k in ("FIT", "original_candidate_and_mask_exact",
                 "exact_original_eligible_logits", "full_basis_logits_exact", "exact_dense_sample_witness"))
         and all(report[k] is False for k in ("DEV", "TEST", "GT_opened", "AP"))
         and report["head_updates"] == 0 and report["sampling_channel_chunk"] == 8,
         "Complete label-free original CORE4 forward and exact endpoint witness")
    original_args = SimpleNamespace(capture=args.root/pilot.CAPTURE_DIR,
                                    acceptance=args.root/pilot.MANIFEST_DIR/"INDEPENDENT_READBACK_SAFE.json",
                                    manifest=args.root/pilot.MANIFEST_DIR/"IMAGE_ONLY_MANIFEST_PRIVATE.json")
    original = spot.read_cohort(original_args, lambda rows, count: rows[:count])
    need(len(original) == 4, "Exact four CORE4 source images")
    payloads, entries = [], []
    totals = dict(candidate_sets=0, point_pairs=0, no_support_images=0, informative_points=0, moved_points=0)
    for (item, ordinal, old), (full, i, current), (prepared, _, points) in zip(original, roster[:4], data[:4], strict=True):
        need(ordinal == i == len(entries)+1 and full == prepared
             and all(item[k] == full[k] for k in ("image_id", "native_basename", "encoded_bytes",
                     "encoded_sha256", "decoded_rgb_sha256", "group")), "Same four original FIT109 source images in order")
        for key in ("image_id", "native_basename", "encoded_sha256", "image_content_sha256", "source_prefix",
                    "native_ids", "native_ranks", "ship_scores", "all_rles", "raw_classifier_logits",
                    "raw_boxes_cxcywh", "joint_query_ids", "joint_labels", "joint_scores"):
            need(old[key] == current[key], "Same accepted FIT109 carrier in reused position " + key)
        for key in ("sha256", "bytes", "shape"):
            need(old["logit_capture"][key] == current["logit_capture"][key], "Same accepted retained logits " + key)
        keys = ("logit_row", "query_id", "native_rank", "score", "predicted_area")
        proof = lambda block: [tuple(entry[k] for k in keys) for entry in block["logit_capture"]["candidates"]]
        need(proof(old) == proof(current), "Original retained logit plane identifiers, order, scores, areas")
        path = args.core4_output/archive_name(i)
        with path.open("rb") as f:
            raw = f.read(32*spot.MIB+1)
        need(0 < len(raw) <= 32*spot.MIB, "Bounded CORE4 endpoint payload")
        entries.append(signature(i, raw))
        payloads.append(raw)
    need(sha256(spot.encode(entries)).hexdigest() == report["private_output_sha256"],
         "Original four ordered archive byte signatures")
    for raw, (item, _, points) in zip(payloads, data[:4], strict=True):
        with np.load(BytesIO(raw), allow_pickle=False) as bundle:
            arrays = {k: bundle[k] for k in bundle.files}
        image_id = item["image_id"]
        verify_arrays(arrays, image_id, points, np)
        totals["no_support_images"] += int(not points)
        totals["candidate_sets"] += len(points)
        totals["point_pairs"] += len(arrays["coords"])
        for j, point in enumerate(points):
            part = slice(*arrays["offsets"][j:j+2])
            plan = residual.point_feature_controls(arrays["old14"][part], arrays["spatial"][part],
                       arrays["projected_queries"][j], arrays["coords"][part], arrays["layers"][part],
                       candidate_id=(image_id, point["rank"], point["query_id"]))
            moved, distinct = pilot.witness_controls(plan, arrays["old14"][part], np)
            totals["moved_points"] += moved
            totals["informative_points"] += distinct
    need(all(value == report[key] for key, value in totals.items()), "Replayed four complete CORE4 controls")
    return payloads, report


def capture_new(args, row, points, context):
    c = context
    item, ordinal, old = row
    torch, np = c.torch, c.np
    image = c.cap.original_image(args, item, c.core, c.fit, None)
    tensor, target = c.transform(image, None)
    need(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32, "Original stock val input")
    sample = c.nested([tensor], block_size=c.cfg.patch_size*c.cfg.num_windows)
    need(sample.mask is not None and not bool(sample.mask.any()), "Original no-padding view")
    with torch.inference_mode():
        with c.basis.capture_frozen_basis(c.model.segmentation_head, c.helper.capture_final_query) as obs:
            raw = c.model(sample)
    captured = obs.seal(raw["pred_masks"])
    out = c.post.aligned_query_rows(raw, (800, 800), torch=torch, np=np, codec=c.codec, processor=c.processor)
    probability = raw["pred_logits"][0].sigmoid().flatten()
    selected = torch.argsort(probability, descending=True, stable=True)[:100]
    c.pilot.compare_original(old, out, raw["pred_logits"][0].tolist(), raw["pred_boxes"][0].tolist(),
                             (selected//2).tolist(), (selected%2).tolist(), probability.index_select(0, selected).tolist())
    ref = old["logit_capture"]
    need(ref["relative_path"] == f"RAW_LOGITS/{ordinal:03d}.npy", "Original retained logit path")
    closed_logits = np.load(BytesIO(c.spot.pinned(args.capture/ref["relative_path"], ref["sha256"], 9*c.spot.MIB)),
                            allow_pickle=False)
    eligible = [entry["query_id"] for entry in ref["candidates"]]
    need(np.array_equal(closed_logits, raw["pred_masks"][0, eligible].contiguous().numpy()), "All retained logits exact")
    arrays, metrics = c.pilot.capture_image_endpoints(points, item["image_id"], captured, c.basis,
                                                      c.residual, c.channels, np, torch)
    verify_arrays({k: np.asarray(v) for k, v in arrays.items()}, item["image_id"], points, np)
    buffer = BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue(), metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "code", "core4-source", "basis", "residual", "capture", "acceptance", "source-cohort",
                 "protocol", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--core4-output", type=Path)
    parser.add_argument("--core4-closure", type=Path)
    parser.add_argument("--core4-result-sha256")
    parser.add_argument("--core4-closure-sha256")
    args = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "Isolated CPU-only source capture")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists(), "New exclusive capture directory")
    args.root = args.root.resolve(strict=True)
    args.mode = "fit"
    args.acceptance_sha256 = RAW_ACCEPTANCE_SHA
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Local pinned FIT inputs only")
    sys.addaudithook(offline)
    raw_source = args.core4_source.read_bytes()
    need(sha256(raw_source).hexdigest() == CORE4_SHA, "Exact closed CORE4 v04 source")
    pilot = ModuleType("_rfdetr_closed_core4")
    pilot.__file__ = str(args.core4_source)
    exec(compile(raw_source, pilot.__file__, "exec"), pilot.__dict__)
    pilot.pinned(args.protocol, args.protocol_sha, 64*1024)
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    need(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
         and np.__version__ == "1.26.4" and not torch.cuda.is_initialized(), "Original CPU runtime")
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cap, spot, step, feat, helper = [pilot.load(args.code/name, digest, f"_rfd_prior_{i}")
                                     for i, (name, digest) in enumerate(pilot.PINS.items())]
    formal = pilot.load(args.code/"RFDETR_MATCHED_POINT_FORMAL_20261120/run_fit109.py", FORMAL_SHA, "_rfd_fit109")
    basis = pilot.load(args.basis, pilot.BASIS_SHA, "_rfd_basis")
    residual = pilot.load(args.residual, pilot.RESIDUAL_SHA, "_rfd_residual")
    posterior = step.load(args.code/"RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py", spot.ADAPTER_SHA, "_rfd_post")
    core = cap.load_core(args.root)
    args.fit_images = core.LAYOUT/"train"
    roster, accepted = formal.read_population(args, spot)
    need([r[1] for r in roster] == list(range(1,110)) and len({r[0]["image_id"] for r in roster}) == 109,
         "Exactly the complete ordered FIT109 population")
    started = perf_counter()
    data, counts, feature_digest = spot.prepare(args, roster, step, feat, posterior, np, torch, codec)
    prior = verify_prior_prelabel(counts, feature_digest, spot.encode)
    need(accepted["native_mapped_rows"] == counts["mapped"] and accepted["same_device_raster_exact"] == counts["eligible"],
         "Retained raster and mapped counts match old independent acceptance")
    cached, old_report = admit_core4(args, roster, data, pilot, spot, np, residual)
    args.output.mkdir()
    with (args.output/"PRELABEL_SAFE.json").open("xb") as dest:
        dest.write(prior)
        dest.flush()
        os.fsync(dest.fileno())
    base, post = cap.setup(args, core)
    model, transform, cfg = cap.model_and_transform(args, core, base, torch)
    context = SimpleNamespace(core=core, cap=cap, fit=core.load("fit_audit"), helper=helper, pilot=pilot,
                              torch=torch, np=np, codec=codec, model=model, cfg=cfg, transform=transform,
                              nested=nested_tensor_from_tensor_list, processor=PostProcess, post=post,
                              basis=basis, residual=residual, spot=spot,
                              channels=torch.tensor([0, 1, 7, 31, 63, 95, 127, 255]))
    moved = old_report.get("moved_points", 0)
    distinct = old_report.get("informative_points", 0)
    grid_max = old_report.get("earlier_grid_maximum_delta", 0.)
    grid_exceeded = old_report.get("earlier_grid_tolerance_exceedances", 0)
    pairs = sets = empty = forwards = 0
    entries = []
    for row, (item, _, points) in zip(roster, data, strict=True):
        original, ordinal, _ = row
        need(original == item and ordinal == len(entries)+1, "All prepared images in exact original order")
        if ordinal <= len(cached):
            raw = cached[ordinal-1]
        else:
            raw, metrics = capture_new(args, row, points, context)
            moved += metrics["moved"]
            distinct += metrics["informative"]
            grid_max = max(grid_max, metrics["grid_max"])
            grid_exceeded += metrics["grid_exceedances"]
            forwards += 1
        path = args.output/archive_name(ordinal)
        with path.open("xb") as dest:
            need(dest.write(raw) == len(raw), "Complete per-image basis write")
            dest.flush()
            os.fsync(dest.fileno())
        entry = signature(ordinal, raw)
        need(path.stat().st_size == entry["bytes"] and sha256(path.read_bytes()).hexdigest() == entry["sha256"],
             "Read back closed per-image endpoint bytes")
        entries.append(entry)
        sets += len(points)
        pairs += sum(len(point["coordinates"]) for point in points)
        empty += int(not points)
        if ordinal % 10 == 0:
            print(json.dumps(dict(event="FIT109_FROZEN_QUERY_CAPTURE_PROGRESS", images=ordinal,
                                  model_forwards=forwards, reused=len(cached))), flush=True)
    need(len(entries) == len(cached)+forwards == 109 and sets == counts["candidate_point_sets"]
         and pairs == counts["pairs"] and moved > 0 and distinct > 0 and not torch.cuda.is_initialized(),
         "Complete FIT109 census and effective controls")
    result = dict(status="RFDETR_FIT109_QUERY_BASIS_CAPTURE_COMPLETE", FIT=True, DEV=False, TEST=False, GT_opened=False,
                  AP=False, head_updates=0, images=109, model_forwards=forwards, reused_core4=len(cached),
                  candidate_sets=sets, point_pairs=pairs, no_support_images=empty, moved_points=moved,
                  informative_points=distinct, earlier_grid_maximum_delta=grid_max,
                  earlier_grid_tolerance_exceedances=grid_exceeded, full_basis_logits_exact=True,
                  exact_original_eligible_logits=True, original_candidate_and_mask_exact=True,
                  exact_dense_sample_witness=True, sampling_channel_chunk=8, prelabel_sha256=PRELABEL_SHA,
                  feature_sha256=feature_digest, original_feature_counts=counts, accepted_capture_sha256=RAW_ACCEPTANCE_SHA,
                  private_file_count=len(entries), private_output_sha256=sha256(spot.encode(entries)).hexdigest(),
                  core4_result_sha256=args.core4_result_sha256, core4_closure_sha256=args.core4_closure_sha256,
                  core4_source_sha256=CORE4_SHA, basis_source_sha256=pilot.BASIS_SHA, residual_source_sha256=pilot.RESIDUAL_SHA,
                  protocol_sha256=args.protocol_sha, entry_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  wall_seconds=perf_counter()-started)
    digest = spot.write(args.output/"RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest)), flush=True)


if __name__ == "__main__":
    main()
