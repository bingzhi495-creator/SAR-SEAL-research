"""Capture center5202 E24 on DEV705 and regenerate its single native CUDA probability repair."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile


DETECTOR_LOADER = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
INPUT_MANIFEST = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
OUTPUT_MANIFEST = "e41e77d470a27988e3b831ed7d74845053e6ef87232425688463b79996bcc697"
QUERIES_SHA = "3533d397589a0e615d93ccf92a4bc8f6beea711dfcf08912ec654968c756ed70"
QUERIES = "QUERIES_PRIVATE.jsonl"
STOCK = "STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl"
PROBABILITY = "NATIVE_PROBABILITY_PRIVATE.npy"
REPAIR = "RENDER_REPAIR_PRIVATE.json"
# Numerical reference values from the closed repair bound by public digest ce2b2726…
REPAIR_EXPECTED = {
    "source_raw_plane_sha256": "693379041fca23b7c6738b8b2359214c0ea769a262e4e1c53cab3523aa45ed9b",
    "strict_cuda_logit_sign_tensor_sha256": "3494160ca6185a541aff9cef867d60802ed2a1507ef3a87551ea5817a3b5def5",
    "native_probability_tensor_sha256": "b62f6b1877f7d503d116354754bd4a580a0fc50cfa945c62b9d0767a50110e5c",
    "stock_rle_sha256": "b92b2b3a1a774205938caeb71bf421b80be738786e49168724db299d5b399fcd",
}
REPAIR_FILE_SHA = "aa90d5c24eaedc9d38e637aba74c3bbbf82670dfa7fa2666be86443c967e7ae3"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(data):
    return sha256(data).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
        return digest.hexdigest()


def loader(code):
    path = code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py"
    raw = path.read_bytes()
    require(sha(raw) == DETECTOR_LOADER, "Published portable E24 loader differs")
    module = ModuleType("_clean_detector")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def npy(np, value):
    buffer = BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def interpolate(torch, planes):
    return torch.nn.functional.interpolate(planes, size=(800, 800), mode="bilinear", align_corners=False)


def image_block(c, row, image, ordinal):
    tensor, target = c.transform(image, None)
    require(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == c.torch.float32, "Original val576 transform")
    sample = c.nest([tensor], block_size=c.config.patch_size * c.config.num_windows)
    require(sample.mask is not None and not bool(sample.mask.any()), "Original unpadded input")
    raw = c.model(sample.to("cuda"))
    query = c.post.aligned_query_rows(raw, (800,800), torch=c.torch, np=c.np, codec=c.codec, processor=c.post_class)
    selected = c.full.eligible_candidates(query, c.codec)
    ids = c.torch.tensor([q for q, _, _ in selected], device=raw["pred_masks"].device, dtype=c.torch.long)
    values = raw["pred_masks"][0].index_select(0, ids).contiguous().cpu().numpy().astype("<f4", copy=False)
    require(values.shape == (len(selected), 144, 144) and c.np.isfinite(values).all(), "Exact selected query planes")
    for j in range(0, len(selected), 8):
        batch = c.torch.from_numpy(values[j:j+8, None]).to("cuda")
        signs = (interpolate(c.torch, batch) > 0)[:, 0].cpu().numpy()
        for binary, (q, _, _) in zip(signs, selected[j:j+8], strict=True):
            require(c.np.array_equal(binary, c.codec.decode(query["all_rles"][q])), "CUDA strict-sign native raster")
    probs = raw["pred_logits"][0].sigmoid()
    joint = c.torch.argsort(probs.flatten(), descending=True, stable=True)[:100]
    block = dict(image_id=row["image_id"], source_prefix=Path(row["file_name"]).name.split("_", 1)[0],
                 native_basename=Path(row["file_name"]).stem + ".png", image_content_sha256=row["decoded_rgb_sha256"],
                 encoded_sha256=row["encoded_sha256"], **query, raw_boxes_cxcywh=raw["pred_boxes"][0].tolist(),
                 raw_classifier_logits=raw["pred_logits"][0].tolist(), joint_query_ids=(joint // 2).tolist(),
                 joint_labels=(joint % 2).tolist(), joint_scores=probs.flatten().index_select(0, joint).tolist())
    rel = f"RAW_LOGITS/{ordinal:03d}.npy"
    data = npy(c.np, values)
    block["logit_capture"] = dict(relative_path=rel, sha256=c.full.seal(c.output / rel, data), bytes=len(data),
        shape=list(values.shape), candidates=[dict(logit_row=i, query_id=q, native_rank=rank,
            score=block["ship_scores"][q], predicted_area=area) for i, (q, rank, area) in enumerate(selected)])
    return block


def render_one(c, ordinal, block, index, plane, stock, difference):
    ref = block["logit_capture"]
    candidate = ref["candidates"][index]
    q = candidate["query_id"]
    # This is the original native CUDA one-plane operation from render.py:
    # bilinear raw FP32 logits at 800x800, then sigmoid on CUDA before copying to CPU.
    logit = interpolate(c.torch, c.torch.from_numpy(plane[None, None]).to("cuda"))[0, 0]
    sign = (logit > 0).to(c.torch.uint8).cpu().numpy()
    require(c.np.array_equal(sign, stock), "Corrected raw logit sign differs from native mask")
    prob = logit.sigmoid().cpu().numpy().astype("<f4", copy=False)
    require(prob.shape == (800,800) and c.np.isfinite(prob).all() and 0 <= prob.min() <= prob.max() <= 1,
            "Native CUDA probability field")
    digests = dict(source_raw_plane_sha256=sha(plane.tobytes(order="C")),
                   strict_cuda_logit_sign_tensor_sha256=sha(sign.tobytes(order="C")),
                   native_probability_tensor_sha256=sha(prob.tobytes(order="C")),
                   stock_rle_sha256=sha(c.full.encoded(block["all_rles"][q])))
    require(digests == REPAIR_EXPECTED, "Regenerated native CUDA correction differs from the original")
    data = npy(c.np, prob)
    require(sha(data) == REPAIR_FILE_SHA, "Portable probability array encoding differs")
    c.full.seal(c.output / PROBABILITY, data)
    require(file_hash(c.output / PROBABILITY) == REPAIR_FILE_SHA, "Probability file readback differs")
    return dict(ordinal=ordinal, image_id=block["image_id"], query_id=q, native_rank=candidate["native_rank"],
                logit_row=index, source_logit_ref={k:ref[k] for k in ("relative_path","bytes","sha256")},
                native_probability_ref=dict(relative_path=PROBABILITY, bytes=len(data), sha256=REPAIR_FILE_SHA,
                                            shape=[800,800], dtype="<f4"),
                original_hw=[800,800], cpu_disagreement_rows=1, cpu_disagreement_pixels=1,
                discrepancy_flat_index=int(difference), strict_cuda_logit_sign_RLE_exact=True, **digests)


def check_saved(c, rows):
    c.torch.set_num_threads(1)
    repair = None
    planes = mapped = size = 0
    with (c.output / QUERIES).open("rb") as queries, (c.output / STOCK).open("rb") as native:
        for ordinal, (row, query_line, stock_line) in enumerate(zip(rows, queries, native, strict=True), 1):
            block = json.loads(query_line)
            identity = row["image_id"]
            require(block["image_id"] == identity and block["image_content_sha256"] == row["decoded_rgb_sha256"] and
                    block["encoded_sha256"] == row["encoded_sha256"] and
                    c.full.native_block(identity, block) == json.loads(stock_line), "Saved candidate identity/score/raster")
            ref = block["logit_capture"]
            require(ref["relative_path"] == f"RAW_LOGITS/{ordinal:03d}.npy", "Plane order")
            raw = c.full.exact(c.output / ref["relative_path"], ref["sha256"], 9 * 1024**2)
            buf = BytesIO(raw)
            values = c.np.load(buf, allow_pickle=False)
            selected = c.full.eligible_candidates(block, c.codec)
            require(not buf.read(1) and len(raw) == ref["bytes"] and values.dtype == c.np.dtype("<f4") and
                    values.shape == (len(selected), 144, 144) and list(values.shape) == ref["shape"] and
                    c.np.isfinite(values).all(), "Saved original raw logits")
            for j in range(0, len(selected), 8):
                signs = (interpolate(c.torch, c.torch.from_numpy(values[j:j+8, None])) > 0)[:, 0].numpy()
                for i, (sign, (q, rank, area)) in enumerate(zip(signs, selected[j:j+8], strict=True), j):
                    candidate = ref["candidates"][i]
                    require(candidate == dict(logit_row=i, query_id=q, native_rank=rank, predicted_area=area,
                                              score=block["ship_scores"][q]), "Saved raw-plane identity")
                    stock = c.codec.decode(block["all_rles"][q])
                    differences = c.np.flatnonzero(sign != stock)
                    if differences.size:
                        require(differences.size == 1 and repair is None, "Unexpected additional CPU-sign disagreements")
                        repair = render_one(c, ordinal, block, i, values[i], stock, differences[0])
            planes += len(selected)
            mapped += len(block["native_ids"])
            size += len(raw)
            if ordinal % 100 == 0:
                print(json.dumps(dict(phase="readback", images=ordinal, planes=planes)), flush=True)
    require(repair is not None and (mapped, planes, size) == (67386, 42325, 3510695040),
            "Complete original carrier census and exactly one repair")
    repair_sha = c.full.seal(c.output / REPAIR, c.full.encoded(repair))
    return planes, mapped, size, repair_sha


def run(args):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    require(sys.version_info[:3] == (3,10,13) and torch.__version__ == "2.10.0+cu130" and np.__version__ == "1.26.4"
            and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0" and version("pycocotools") == "2.0.11",
            "Original RF-DETR inference environment")
    require(not args.output.exists(), "Use a fresh capture output directory")
    raw = (args.intake / "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json").read_bytes()
    require(sha(raw) == INPUT_MANIFEST, "Original DEV705 image manifest")
    rows = json.loads(raw)
    require(len(rows) == len({r["image_id"] for r in rows}) == 705, "Original DEV705 population")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and not torch.backends.cuda.matmul.allow_tf32
            and not torch.are_deterministic_algorithms_enabled(), "Original single CUDA native bilinear and FP32 settings")
    portable = loader(args.code)
    full, core, base, post = portable.detector_components(args.code, args.checkpoint)
    available, total = torch.cuda.mem_get_info()
    require(available >= base.CAP, "Original 6 GiB available GPU budget")
    torch.cuda.set_per_process_memory_fraction(min(.35, base.CAP / total))
    paths = SimpleNamespace(mode="dev", output=args.output)
    fit = base.pinned_module(base.FIT_AUDIT, base.FIT_AUDIT_SHA)
    model, transform, config = full.model_and_transform(paths, core, base, torch)
    require(not model.training and all(p.device.type == "cuda" for p in model.parameters()), "Frozen CUDA detector")
    c = SimpleNamespace(np=np, torch=torch, codec=codec, full=full, model=model, transform=transform, config=config,
                        post=post, post_class=PostProcess, nest=nested_tensor_from_tensor_list, output=args.output)
    args.output.mkdir()
    (args.output / "RAW_LOGITS").mkdir()
    manifest = []
    start = perf_counter()
    with args.archive.open("rb") as handle, torch.inference_mode():
        require(args.archive.stat().st_size == fit.ARCHIVE_BYTES, "Official PNG archive size")
        handle.seek(fit.CENTRAL_OFFSET)
        require(sha(handle.read(fit.CENTRAL_BYTES)) == fit.CENTRAL_SHA, "Original ZIP directory")
        with ZipFile(handle) as archive, (args.output / QUERIES).open("xb") as queries, (args.output / STOCK).open("xb") as stock:
            for ordinal, row in enumerate(rows, 1):
                image = full.original_image(paths, row, core, fit, archive)
                block = image_block(c, row, image, ordinal)
                stock.write(base.encoded(full.native_block(row["image_id"], block)))
                queries.write(full.encoded(block))
                manifest.append(dict(image_id=row["image_id"], encoded_sha256=row["encoded_sha256"],
                    decoded_rgb_sha256=row["decoded_rgb_sha256"], origin="new_original_CUDA_FP32"))
                if ordinal == 1 or ordinal % 25 == 0:
                    print(json.dumps(dict(phase="capture", images=ordinal)), flush=True)
    capture_seconds = perf_counter() - start
    native_hash, query_hash = file_hash(args.output / STOCK), file_hash(args.output / QUERIES)
    manifest_hash = full.seal(args.output / "IMAGE_MANIFEST_PRIVATE.json", full.encoded(manifest))
    require((native_hash, query_hash, manifest_hash) == (full.DEV_STOCK, QUERIES_SHA, OUTPUT_MANIFEST),
            f"Historical stream parity differs: stock={native_hash}, queries={query_hash}, manifest={manifest_hash}")
    del c.model, model
    torch.cuda.empty_cache()
    with torch.inference_mode():
        planes, mapped, size, repair_sha = check_saved(c, rows)
    result = dict(status="B5_RFDETR_CENTER5202_DEV705_CAPTURE_COMPLETE", mode="dev", images=len(rows),
                  model_forwards=len(rows), model_forward_calls=len(rows), optimizer_updates=0, cache_hits=0,
                  GT_opened=False, AP=False, FIT=False, DEV=True, TEST=False, checkpoint_sha256=full.CKPT,
                  source_checkpoint_sha256=full.CKPT, input_manifest_sha256=INPUT_MANIFEST,
                  image_manifest_sha256=manifest_hash, stock_predictions_sha256=native_hash, queries_sha256=query_hash,
                  native_joint_rows=len(rows)*100, native_mapped_rows=mapped, raw_logit_rows=planes,
                  raw_logit_files=len(rows), raw_logit_bytes=size, raw_logit_raster_RLE_exact=planes,
                  historical_CUDA_stock_exact=True, same_device_cuda_raster_exact=planes,
                  independent_CPU_raster_disagree_rows=1, independent_CPU_raster_disagree_pixels=1,
                  repair_sha256=repair_sha, native_probability_file_sha256=REPAIR_FILE_SHA,
                  native_probability_tensor_sha256=REPAIR_EXPECTED["native_probability_tensor_sha256"],
                  device=torch.cuda.get_device_name(0), precision="FP32_TF32_disabled",
                  capture_seconds=capture_seconds, elapsed_seconds=perf_counter()-start,
                  worker_sha256=file_hash(Path(__file__)))
    digest = full.seal(args.output / "RESULT_SAFE.json", full.encoded(result))
    print(json.dumps(dict(status=result["status"], result_sha256=digest)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("archive", "intake", "checkpoint", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--code", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    for key, path in vars(args).items():
        setattr(args, key, path.resolve(strict=key != "output"))
    run(args)

