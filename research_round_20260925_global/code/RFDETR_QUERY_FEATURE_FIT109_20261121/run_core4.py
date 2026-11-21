"""Qualify four original FIT source images for the frozen query-channel features."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

PINS = {
    "RFDETR_CENTER5202_FULL_CAPTURE_20261120/capture.py": "97449f58d1ad1a62c52c6f100fc872690e51d465e85e241660befba7a9690a71",
    "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py": "75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee",
    "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py": "94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1",
    "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/feature_adapter.py": "747c4c4b43561f8d073e01854adf241f1cb66c4efa1d29f304ae8837ecd67b73",
    "HRSID_FIT_NATIVE_POINT_UPDATE_20261110/cached_final_query.py": "bc6bf64a7ff01a716fcd4fe97da9fa292a892bbc5699ff57b46c615581ce4adc",
}
BASIS_SHA = "d6a5db68a8bc6ad173294791deb3dcbe5e610de68cedafa355ead689c604fb12"
RESIDUAL_SHA = "4ca7f2345fd4bc3d0faee39745292df07757a083dcb7f123af3643057a5d21ce"
MANIFEST_DIR = "HRSID_FIT114_CORE36_CENTER5202_RAW_CAPTURE_DRAFT_PRIVATE_20261112_01"
CAPTURE_DIR = "HRSID_FIT114_CORE36_CENTER5202_RAW_CAPTURE_ACTUAL_PRIVATE_20261112_01/center/capture"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, expected, limit=1024**2):
    with path.open("rb") as stream:
        b = stream.read(limit + 1)
    need(0 < len(b) <= limit and sha256(b).hexdigest() == expected, "Exact admissible bytes " + path.name)
    return b


def load(path, digest, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(pinned(path, digest), str(path), "exec"), module.__dict__)
    return module


def compare_original(block, query, logits, boxes, joint_ids, joint_labels, joint_scores):
    for key in ("native_ids", "native_ranks", "ship_scores", "all_rles"):
        need(query[key] == block[key], "Prior native carrier mismatch " + key)
    for key, current in (("raw_classifier_logits", logits), ("raw_boxes_cxcywh", boxes),
                         ("joint_query_ids", joint_ids), ("joint_labels", joint_labels),
                         ("joint_scores", joint_scores)):
        need(current == block[key], "Prior raw carrier mismatch " + key)


def sample_dense_endpoint_basis(spatial, coords, np, torch, chunk=8):
    """Gather the original CPU interpolate values from channel-bounded blocks."""
    need(tuple(spatial.shape[:2]) == (1,256) and spatial.device.type=="cpu"
         and spatial.dtype==torch.float32, "Frozen complete CPU spatial basis")
    need(coords.ndim==2 and coords.shape[1]==2 and coords.dtype.kind in "iu"
         and (coords.size==0 or (int(coords.min())>=0 and int(coords.max())<800*800)),
         "Sealed native endpoint indices")
    flat = torch.as_tensor(coords.reshape(-1), dtype=torch.int64)
    output = torch.empty((len(flat),256), dtype=torch.float32)
    for start in range(0,256,chunk):
        stop = min(start+chunk,256)
        dense = torch.nn.functional.interpolate(spatial[:,start:stop].contiguous(),
                                                 size=(800,800), mode="bilinear",align_corners=False)
        output[:,start:stop] = dense[0].reshape(stop-start,-1).index_select(1,flat).T
    return output.reshape(len(coords),2,256)


def witness_controls(plan, original, np):
    controls = plan["features"]
    need(set(controls) == {"zero", "ordinary_centered", "query_conditioned_centered", "query_conditioned_permuted"}
         and all(x.shape == (len(original),2,30) and np.array_equal(x[:,:,:14],original)
                 for x in controls.values()), "Exact four controls and unchanged original features")
    focus = controls["query_conditioned_centered"][:,:,14:]
    moved = np.count_nonzero(np.any(focus != controls["query_conditioned_permuted"][:,:,14:],axis=-1))
    distinct = np.count_nonzero(np.any(focus != controls["ordinary_centered"][:,:,14:],axis=-1))
    return int(moved), int(distinct)


def capture_image_endpoints(points, image_id, captured, basis_module, residual, channel_witness, np, torch):
    """Package the native endpoint archive for one carrier-verified image."""
    ids = [row["query_id"] for row in points]
    arrays = dict(image_id=image_id, ranks=np.asarray([row["rank"] for row in points], dtype=int),
                  query_ids=np.asarray(ids, dtype=int))
    if not points:
        arrays.update(coords=np.empty((0,2), dtype=np.int32), old14=np.empty((0,2,14), dtype=np.float32),
                      layers=np.empty((0,2), dtype=np.int8), spatial=np.empty((0,2,256), dtype=np.float32),
                      projected_queries=np.empty((0,256), dtype=np.float32), offsets=np.array([0], dtype=np.int64))
        return arrays, dict(moved=0, informative=0, grid_max=0., grid_exceedances=0)

    coords = np.concatenate([row["coordinates"] for row in points])
    original = np.concatenate([row["features"] for row in points])
    layers = np.concatenate([row["layers"] for row in points])
    vectors = sample_dense_endpoint_basis(captured.spatial, coords, np, torch)
    earlier_grid = basis_module.sample_sealed_endpoints(captured.spatial, coords, (800,800))
    sampled = vectors.numpy()
    projected_query = basis_module.select_projected_queries(captured, ids).numpy()
    witness = torch.nn.functional.interpolate(captured.spatial.index_select(1,channel_witness),
                                              size=(800,800), mode="bilinear", align_corners=False)
    dense = witness[0].reshape(len(channel_witness), -1)[:,torch.from_numpy(coords.reshape(-1))].T
    dense = dense.reshape(len(coords),2,len(channel_witness))
    actual = vectors[:,:,channel_witness]
    need(torch.equal(actual,dense), "Exact original dense-sampling reference across witness channels")
    earlier_delta = (earlier_grid[:,:,channel_witness]-dense).abs()
    earlier_threshold = basis_module.REFERENCE_ATOL+basis_module.REFERENCE_RTOL*dense.abs()

    moved = informative = 0
    offsets = np.concatenate(([0],np.cumsum([len(row["coordinates"]) for row in points])))
    for i, row in enumerate(points):
        part = slice(offsets[i], offsets[i+1])
        plan = residual.point_feature_controls(original[part], sampled[part], projected_query[i], coords[part],
                     layers[part], candidate_id=(image_id, row["rank"],row["query_id"]))
        added_moved, added_informative = witness_controls(plan, original[part], np)
        moved += added_moved
        informative += added_informative
    arrays.update(coords=coords, old14=original, layers=layers, spatial=sampled,
                  projected_queries=projected_query, offsets=offsets)
    return arrays, dict(moved=moved, informative=informative, grid_max=float(earlier_delta.max()),
                        grid_exceedances=int((earlier_delta>earlier_threshold).sum()))




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "code", "basis", "residual", "output", "protocol"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--protocol-sha", required=True)
    args = ap.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only admission")
    args.mode = "fit"
    need(args.output.is_absolute() and not args.output.exists() and args.output.parent.is_dir(),
         "New exclusive output only")
    pinned(args.protocol, args.protocol_sha, 64*1024)
    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Local pinned FIT inputs only")
    sys.addaudithook(offline)
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    need(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
         and np.__version__ == "1.26.4" and not torch.cuda.is_initialized(),
         "Original CPU runtime")
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cap, spot, fit_step, feat, helper = [
        load(args.code / name, digest, "_rf_qfeature_prior_" + str(i))
        for i, (name, digest) in enumerate(PINS.items())
    ]
    basis_module = load(args.basis, BASIS_SHA, "_rf_qfeature_basis")
    residual = load(args.residual, RESIDUAL_SHA, "_rf_qfeature_residual")
    posterior = fit_step.load(args.code / "RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py",
                              spot.ADAPTER_SHA, "_rf_qfeature_native_posterior")
    core = cap.load_core(args.root.resolve(strict=True))
    args.capture = args.root / CAPTURE_DIR
    args.manifest = args.root / MANIFEST_DIR / "IMAGE_ONLY_MANIFEST_PRIVATE.json"
    args.acceptance = args.root / MANIFEST_DIR / "INDEPENDENT_READBACK_SAFE.json"
    args.fit_images = core.LAYOUT / "train"
    roster = spot.read_cohort(args, lambda rows, count: rows[:count])
    need(len(roster) == 4 and [int(row[1]) for row in roster] == [1, 2, 3, 4], "First closed CORE36 four only")
    started = perf_counter()
    data, counts, feature_digest = spot.prepare(args, roster, fit_step, feat, posterior, np, torch, codec)
    need(counts["exact_original_masks"] == counts["eligible"] and counts["candidate_point_sets"] > 0,
         "All original eligible masks exact and nonempty point support")
    args.output.mkdir()
    prelabel = spot.write(args.output / "PRELABEL_SAFE.json", dict(
        status="RFDETR_FIT_QUERY_BASIS_CORE4_PRELABEL", FIT=True, DEV=False, TEST=False, GT_opened=False,
        AP=False, images=4, feature_sha256=feature_digest, original_prior_capture_acceptance_sha256=spot.ACCEPTANCE_SHA,
        protocol_sha256=args.protocol_sha, original_feature_counts=counts))
    base, post = cap.setup(args, core)
    model, transform, cfg = cap.model_and_transform(args, core, base, torch)
    fit = core.load("fit_audit")

    total_pairs = total_sets = moved = informative = 0
    dense_max = 0.
    old_grid_tolerance_exceedances = 0
    no_support_images = 0
    outputs = []
    channel_witness = torch.tensor([0, 1, 7, 31, 63, 95, 127, 255])
    for (item, ordinal, old), (same_item, _, points) in zip(roster, data, strict=True):
        need(item == same_item, "Same original image including images without endpoint support")
        image = cap.original_image(args, item, core, fit, None)
        tensor, target = transform(image, None)
        need(target is None and tensor.shape == (3,576,576) and tensor.dtype == torch.float32,
             "Original stock val input")
        sample = nested_tensor_from_tensor_list([tensor], block_size=cfg.patch_size*cfg.num_windows)
        need(sample.mask is not None and not bool(sample.mask.any()), "Original no-padding view")
        with torch.inference_mode():
            with basis_module.capture_frozen_basis(model.segmentation_head, helper.capture_final_query) as obs:
                raw = model(sample)
        captured = obs.seal(raw["pred_masks"])
        out = post.aligned_query_rows(raw, (800,800), torch=torch, np=np, codec=codec, processor=PostProcess)
        probability = raw["pred_logits"][0].sigmoid()
        selected = torch.argsort(probability.flatten(), descending=True, stable=True)[:100]
        compare_original(old, out, raw["pred_logits"][0].tolist(), raw["pred_boxes"][0].tolist(),
                         (selected//2).tolist(), (selected%2).tolist(),
                         probability.flatten().index_select(0,selected).tolist())
        ref = old["logit_capture"]
        closed = spot.pinned(args.capture/ref["relative_path"], ref["sha256"], 9*spot.MIB)
        closed_logits = np.load(BytesIO(closed), allow_pickle=False)
        eligible = [entry["query_id"] for entry in ref["candidates"]]
        need(np.array_equal(closed_logits, raw["pred_masks"][0,eligible].contiguous().numpy()),
             "All retained original raw logits exact")

        arrays, metrics = capture_image_endpoints(
            points, item["image_id"], captured, basis_module, residual, channel_witness, np, torch
        )
        no_support_images += int(not points)
        moved += metrics["moved"]
        informative += metrics["informative"]
        dense_max = max(dense_max, metrics["grid_max"])
        old_grid_tolerance_exceedances += metrics["grid_exceedances"]

        path = args.output / f"NATIVE_ENDPOINT_BASIS_{ordinal:02d}_PRIVATE.npz"
        with path.open("xb") as output:
            np.savez(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        outputs.append(dict(ordinal=ordinal,sha256=sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
        total_pairs += len(arrays["coords"])
        total_sets += len(points)

    need(moved > 0 and informative > 0 and total_sets==counts["candidate_point_sets"]
         and total_pairs==counts["pairs"] and not torch.cuda.is_initialized(), "Four real complete feature controls")
    report = dict(status="RFDETR_FIT_QUERY_BASIS_CORE4_CAPTURE_COMPLETE", FIT=True, DEV=False, TEST=False, GT_opened=False,
                  AP=False, original_candidate_and_mask_exact=True, exact_original_eligible_logits=True,
                  full_basis_logits_exact=True, head_updates=0, model_forwards=4, candidate_sets=total_sets,
                  point_pairs=total_pairs, informative_points=informative, moved_points=moved,
                  no_support_images=no_support_images,
                  exact_dense_sample_witness=True, sampling_channel_chunk=8,
                  earlier_grid_maximum_delta=dense_max, earlier_grid_tolerance_exceedances=old_grid_tolerance_exceedances, prelabel_sha256=prelabel, private_file_count=len(outputs),
                  private_output_sha256=sha256(spot.encode(outputs)).hexdigest(), wall_seconds=perf_counter()-started,
                  entry_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    digest = spot.write(args.output / "RESULT_SAFE.json",report)
    print(json.dumps({"status": report["status"], "result_sha256": digest}),flush=True)

if __name__ == "__main__":
    main()
