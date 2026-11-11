"""Capture prediction-only one-pixel signed logits on frozen HRSID FIT114."""

import argparse
from collections import Counter
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile


STATUS = "P316_HRSID_FIT114_SPARSE_SIGNED_LOGIT_CAPTURE_COMPLETE"
P314_SHA = "fd94d4024268bd79f5a44eee05c359fa862a276ba9131c3ff2c3096523a174f8"
ACCEPT_SHA = "1af57a267fd594d506aee7244c6dc8a10414abad9c525d2d5dbed6380a22ba3c"
SOURCES = {
    "filtered": ("HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py",
                 "9a69500d10d5b0a371ea504f1ba89b7335aca130c1a28d96c4f93a87b2a4b9c0"),
    "base": ("HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py",
             "1ce0eaa2106fe52452129fad534d0fc11bb8ad5fc4611f70858dbdf6f87a5ec3"),
    "fit_audit": ("HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py",
                  "eb5f91a333ea7c7a1c12a87574d0c192f678aa80a742107a510654e3d5512df5"),
    "input_shift": ("HRSID_FIT114_INPUT_SHIFT_20261111/input_shift.py",
                    "bf32a994b00ce9e9e30c976b8aa7e16b0dd2dce3d4e4942c2305b1f7d07f9c71"),
    "windows": ("SCHWEGMANN_FROZEN_TRANSFER_20261110/export_schwegmann.py",
                "79fe4da8fca69589421bd726b0ad0417bdfb52aef3957747c59c51548ab1ecef"),
    "p250": ("HRSID_TO_SSDD_FROZEN_DEV187_20261110/export_frozen_hrsid_ssdd.py",
             "4acb5fbcaf35c5a32510f5f93ccf51f34e08641fd6d007c78c85becde7496727"),
}
OFFSETS = ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
POINT_SALT = b"P316-HRSID-FIT114-SIGNED-BAND-v1\0"
CAP, SIZE, QUERIES, TOP, CHUNK = 32, 800, 164, 100, 32
MIB = 2**20


def need(ok, message):
    if not ok:
        raise ValueError(message)


def band_points(pred, signed, image_sha, query_id):
    """Read up to 32 points per side of the prediction-defined Chebyshev band."""
    import numpy as np

    need(pred.ndim == 2 and pred.shape == signed.shape and pred.dtype == np.bool_ and signed.dtype == np.float32,
         "Original boolean mask and FP32 logit grid required")
    ys, xs = np.flatnonzero(pred.any(axis=1)), np.flatnonzero(pred.any(axis=0))
    if not len(ys):
        return []
    height, width = pred.shape
    y0, y1 = max(0, int(ys[0])-1), min(height, int(ys[-1])+2)
    x0, x1 = max(0, int(xs[0])-1), min(width, int(xs[-1])+2)
    support = pred[y0:y1, x0:x1]
    padded = np.pad(support, 1, constant_values=False)
    exterior = np.zeros_like(support)
    interior = np.ones_like(support)
    nh, nw = support.shape
    for dy, dx in OFFSETS:
        neighbor = padded[1+dy:1+dy+nh, 1+dx:1+dx+nw]
        np.logical_or(exterior, neighbor, out=exterior)
        np.logical_and(interior, neighbor, out=interior)
    result = []
    prefix = POINT_SALT + bytes.fromhex(image_sha) + query_id.to_bytes(2, "little")
    for inside, band in ((True, support & ~interior), (False, exterior & ~support)):
        coords = [(int(y)+y0, int(x)+x0) for y, x in np.argwhere(band)]
        if len(coords) > CAP:
            side = prefix + bytes([inside])
            coords.sort(key=lambda p: (sha256(side + (p[0]*width+p[1]).to_bytes(4, "little")).digest(), p))
            coords = sorted(coords[:CAP])
        for y, x in coords:
            logits = [float(signed[min(height-1, max(0, y+dy)), min(width-1, max(0, x+dx))])
                      for dy, dx in OFFSETS]
            need((logits[0] > 0) is inside, "Point logit sign must equal original CPU stock mask")
            result.append(dict(y=y, x=x, inside=inside, logits=logits))
    return result


def parity_counts(historical, cpu):
    old_by_query = {r["query_id"]: r for r in historical}
    new_by_query = {r["query_id"]: r for r in cpu}
    common = old_by_query.keys() & new_by_query.keys()
    result = dict(historical_rows=len(historical), cpu_rows=len(cpu), common_rows=len(common),
                  missing_old_rows=len(old_by_query.keys()-common), new_cpu_rows=len(new_by_query.keys()-common),
                  ordered_query_list_exact=int(list(old_by_query) == list(new_by_query)))
    for name, field in (("rank", "native_rank"), ("score", "score"), ("rle", "segmentation")):
        result[name+"_different"] = sum(old_by_query[q][field] != new_by_query[q][field] for q in common)
    result["complete_row_stream_exact"] = int(result["ordered_query_list_exact"] == 1
        and all(result[key] == 0 for key in ("rank_different", "score_different", "rle_different")))
    return result


def load(path, digest, label):
    need(path.is_absolute() and path.resolve(strict=True) == path and 0 < path.stat().st_size <= 64*MIB,
         "Canonical bounded frozen source required")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Frozen source digest differs: "+label)
    module = ModuleType("_p316_" + label)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def source_model(args):
    loaded = {key: load(args.code_root / path, digest, key) for key, (path, digest) in SOURCES.items()
              if key not in ("base", "fit_audit")}
    filtered = loaded["filtered"]
    base = filtered.load_original_evaluator(args.code_root / SOURCES["base"][0])
    base.FIT_AUDIT = args.code_root / SOURCES["fit_audit"][0]
    base.RECIPE = args.code_root.parent / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
    acceptance = filtered.read_acceptance(args.acceptance, ACCEPT_SHA)
    need(acceptance.checkpoint_sha256 == args.p314.CHECKPOINT_SHA, "Exact original P192 filtered endpoint")
    base.check_endpoint = partial(filtered.check_filtered_endpoint, original_check=base.check_endpoint,
                                  qualification=acceptance)
    loaded["windows"].check_runtime(base, loaded["p250"])
    return base, filtered, loaded["input_shift"]


def image_rows(archive, row, shift, np, decode):
    name = row["native_basename"]
    need(re.fullmatch(r"P\d+_[A-Za-z0-9_]+\.png", name) and name.split("_", 1)[0] == row["group"],
         "Exact FIT-only native basename")
    data = shift.member(archive, "images/"+name, row["encoded_sha256"], 4*MIB)
    image, scales = decode(data, draft_size=None)
    need(image.mode == "RGB" and image.size == (SIZE, SIZE) and scales == (1., 1.)
         and len(data) == row["encoded_bytes"]
         and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
         "Exact closed original P192 RGB image")
    return image


def observed_rows(raw, *, image_id, encoded_sha, torch, np, codec, base, postprocess):
    class Observed(postprocess):
        def _select_topk(self, logits):
            self.selected = super()._select_topk(logits)
            return self.selected

        def forward(self, *values, **kwargs):
            output = super().forward(*values, **kwargs)
            self.stock_masks = output[0]["masks"]
            return output

    stock = Observed(num_select=TOP, upsample_masks_to_image_size=True)
    native = base.native_rows(raw, image_id=image_id, hw=(SIZE, SIZE), torch=torch, postprocess=stock, np=np, codec=codec)
    scores, labels, query_ids = stock.selected
    probs = raw["pred_logits"].sigmoid()
    expected = torch.argsort(probs.flatten(), descending=True, stable=True)[:TOP]
    need(tuple(raw["pred_masks"].shape) == (1, QUERIES, 144, 144) and raw["pred_masks"].device.type == "cpu"
         and torch.equal(query_ids[0], expected//2) and torch.equal(labels[0], expected%2)
         and torch.equal(scores[0], probs.flatten().index_select(0, expected)), "Exact stock CPU stable selector")
    mapped = {rank: row for rank, row in zip(native["native_ranks"], native["coco"], strict=True)}
    selected = raw["pred_masks"][0].index_select(0, query_ids[0])
    output = []
    empties = 0
    for start in range(0, TOP, CHUNK):
        end = min(start+CHUNK, TOP)
        up = torch.nn.functional.interpolate(selected[start:end, None], size=(SIZE, SIZE),
                                               mode="bilinear", align_corners=False)
        stock_chunk = stock.stock_masks[start:end]
        need(up.dtype == torch.float32 and tuple(up.shape) == tuple(stock_chunk.shape)
             and bool(torch.isfinite(up).all()) and torch.equal(up > 0, stock_chunk),
             "Every original-grid signed FP32 logit must reproduce its stock CPU native mask")
        for position in range(start, end):
            rank = position+1
            if rank not in mapped:
                continue
            query = int(query_ids[0, position])
            plane = stock_chunk[position-start, 0].numpy()
            points = band_points(plane, up[position-start, 0].numpy(), encoded_sha, query)
            row = mapped[rank]
            empties += not bool(plane.any())
            need(row["score"] == float(scores[0, position]) and row["category_id"] == 1
                 and int(labels[0, position]) == 0, "Stock CPU native rank, query and class score identity")
            output.append(dict(native_rank=rank, query_id=query, score=row["score"], category_id=1,
                               segmentation=row["segmentation"], points=points))
    need(len(output) == len(mapped) and len({r["query_id"] for r in output}) == len(output),
         "Every CPU stock native mapped candidate must be retained exactly once")
    return output, empties


def stream_capture(args, p192, cohort_bytes, blocks, source_report, base, filtered, shift):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    need(not torch.cuda.is_initialized(), "No CUDA context permitted")
    rows = json.loads(cohort_bytes)["images"]
    need(len(rows) == len(blocks) == 114 and sum(row["core"] for row in rows) == 109
         and sum(row["stress"] for row in rows) == 5 and not any(row["core"] and row["stress"] for row in rows)
         and len({row["group"] for row in rows if row["core"]}) == 109
         and len({row["image_id"] for row in rows}) == 114
         and [row["image_id"] for row in rows] == [block["image_id"] for block in blocks]
         and sum(len(block["native_ids"]) for block in blocks) == source_report["native_mapped_rows"] == 10572,
         "Exact ordered historical FIT114 native candidate population required")
    need(shift.BUNDLE_SHA == args.p314.BUNDLE_SHA and shift.COHORT_SHA == args.p314.COHORT_SHA,
         "Independent closed FIT bundle source pins must agree")
    started = perf_counter()
    model, config, train, _ = base.load_model(args.checkpoint, args.p314.CHECKPOINT_SHA, torch,
                                              args.output.parent, filtered.TRAINING_PROTOCOL)
    model = model.cpu().eval()
    need(not model.training and model.num_queries == QUERIES and model.group_detr == 13
         and all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
         "Exact frozen CPU FP32 Q164 source model required")
    transform = make_coco_transforms_square_div_64(
        "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=config.patch_size, num_windows=config.num_windows, aug_config=None,
        scale_jitter=train.scale_jitter, gpu_postprocess=False)
    identity = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mtime_ns)
    digest, parity, counters = sha256(), Counter(), Counter()
    args.output.mkdir()
    with args.bundle.open("rb") as bundle:
        before = os.fstat(bundle.fileno())
        bundle_sha = sha256()
        for chunk in iter(lambda: bundle.read(MIB), b""):
            bundle_sha.update(chunk)
        need(before.st_size == 151_320_868 and bundle_sha.hexdigest() == shift.BUNDLE_SHA,
             "Exact closed FIT114 bundle SHA and byte length required")
        bundle.seek(0)
        with ZipFile(bundle) as archive, (args.output/"CANDIDATES_PRIVATE.jsonl").open("xb") as out, torch.inference_mode():
            need(shift.member(archive, "COHORT_PRIVATE.json", shift.COHORT_SHA, MIB) == cohort_bytes,
                 "Exact original P192 GT-free cohort in FIT bundle")
            for ordinal, (row, block) in enumerate(zip(rows, blocks, strict=True), 1):
                image = image_rows(archive, row, shift, np, decode_pil_image_bytes)
                tensor, target = transform(image, None)
                need(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32,
                     "Exact original image-only RF-DETR validation transform")
                sample = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size*config.num_windows)
                need(sample.mask is not None and not bool(sample.mask.any()), "Exact original no-padding sample")
                raw = model(sample)
                candidates, empties = observed_rows(raw, image_id=row["image_id"], encoded_sha=row["encoded_sha256"],
                                                    torch=torch, np=np, codec=codec, base=base, postprocess=PostProcess)
                old = args.p314.candidate_rows(block)
                parity.update(parity_counts(old, candidates))
                counters.update(dict(images=1, empty_candidates=empties, stock_sign_verified_rows=TOP,
                                     inner_points=sum(p["inside"] for c in candidates for p in c["points"]),
                                     outer_points=sum(not p["inside"] for c in candidates for p in c["points"])))
                record = {name: row[name] for name in ("image_id", "group", "encoded_sha256", "core", "stress")}
                line = p192.encoded(dict(record, candidates=candidates))
                need(out.write(line) == len(line), "Full native-candidate capture row required")
                digest.update(line)
                if ordinal % 10 == 0:
                    print(json.dumps(dict(event="P316_FROZEN_FORWARD_PROGRESS", images=ordinal,
                                          cpu_native_candidates=parity["cpu_rows"])), flush=True)
            out.flush()
            os.fsync(out.fileno())
        need(identity(before) == identity(os.fstat(bundle.fileno())) == identity(args.bundle.stat()),
             "Closed FIT114 bundle must remain unchanged")
    need(counters["images"] == 114 and counters["stock_sign_verified_rows"] == 11400
         and parity["historical_rows"] == 10572 and not torch.cuda.is_initialized(),
         "Exact 114 frozen CPU forwards and whole native sign raster verification required")
    return dict(status=STATUS, device="cpu", images=114, core_images=109, stress_images=5, training=False,
                FIT=True, DEV=False, TEST=False, AP=False, GT_opened=False, forward_calls=114, optimizer_updates=0,
                candidate_stream_sha256=digest.hexdigest(),
                candidate_stream_bytes=(args.output/"CANDIDATES_PRIVATE.jsonl").stat().st_size,
                counters=dict(counters), historical_gpu_vs_current_cpu=dict(parity),
                historical_gpu_parity_exact=parity["complete_row_stream_exact"] == 114,
                cpu_native_signed_raster_exact=True, logit_offsets_yx=OFFSETS, boundary_radius_original_pixels=1,
                max_points_per_side=CAP, point_salt=POINT_SALT[:-1].decode("ascii"),
                boundary_distance="Chebyshev", edge_neighbor="clamp_to_original_grid",
                source_training_protocol=filtered.TRAINING_PROTOCOL, source_checkpoint_sha256=args.p314.CHECKPOINT_SHA,
                source_checkpoint_acceptance_sha256=ACCEPT_SHA, historical_predictions_sha256=args.p314.QUERY_SHA,
                historical_export_terminal_sha256=args.p314.TERM_SHA, historical_cohort_sha256=shift.COHORT_SHA,
                fit_bundle_sha256=shift.BUNDLE_SHA, source_sha256={k:v[1] for k,v in SOURCES.items()},
                p314_source_sha256=P314_SHA, elapsed_seconds=perf_counter()-started)


def deny_network(event, _):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise PermissionError("Offline frozen FIT capture required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "original", "bundle", "p314-source", "acceptance", "checkpoint", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.code_root, args.original, args.bundle, args.p314_source, args.acceptance, args.checkpoint):
        need(path.is_absolute() and path.resolve(strict=True) == path, "Exact canonical frozen source path required")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical output required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.addaudithook(deny_network)
    args.p314 = load(args.p314_source, P314_SHA, "p314")
    p192, cohort, blocks, report = args.p314.read_original(args.code_root, args.original)
    base, filtered, shift = source_model(args)
    result = stream_capture(args, p192, cohort, blocks, report, base, filtered, shift)
    with (args.output/"CAPTURE_SAFE.json").open("xb") as stream:
        raw = p192.encoded(result)
        need(stream.write(raw) == len(raw), "Complete capture receipt required")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, report_sha256=sha256(raw).hexdigest(),
                          candidate_stream_sha256=result["candidate_stream_sha256"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
