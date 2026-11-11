"""Compare native versus globally low-pass crop features for fixed RF-DETR queries."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile


STATUS = "P326_HRSID_FIT114_TILED_BAND_CAPTURE_COMPLETE"
SALT = b"P326-HRSID-FIT114-TILED-BAND-v1\0"
COUNT, SIZE, TILE, RADIUS, MIB = 16, 800, 400, 2, 2**20
EXPECTED_SELECTION_SHA = "b95c5b44950996055301d51955806473197014a42bf87d9854b47f5057169897"
SOURCES = {
    "p316": ("HRSID_FIT114_SIGNED_PHASE_20261111/sparse_fit_capture.py",
             "c129e18d6ffaab843f2901581425711f6daca12deeaace0ee0f1150b3dcd7ba1"),
    "p318": ("HRSID_FIT114_RENDERED_INTENSITY_20261111/rendered_features.py",
             "2bbe3e14b95f7a115e81c9598c24f0bb4617aa1ea78dd63c60ea8313aa3bece5"),
    "p314": ("PHOENIX_DGX_FULL_AND_FIT_20261111/build_request.py",
             "fd94d4024268bd79f5a44eee05c359fa862a276ba9131c3ff2c3096523a174f8"),
    "query": ("HRSID_FIT_NATIVE_POINT_UPDATE_20261110/cached_final_query.py",
              "bc6bf64a7ff01a716fcd4fe97da9fa292a892bbc5699ff57b46c615581ce4adc"),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load(code_root, key):
    subpath, digest = SOURCES[key]
    path = code_root / subpath
    raw = path.read_bytes()
    need(path.resolve(strict=True) == path and sha256(raw).hexdigest() == digest,
         "Exact released source required: "+key)
    module = ModuleType("_p326_"+key)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def band_window(mask):
    """Center a half-open tile and require the entire in-canvas band to fit."""
    import numpy as np
    from scipy.ndimage import maximum_filter, minimum_filter

    need(mask.shape == (SIZE, SIZE) and mask.dtype == np.bool_ and bool(mask.any()),
         "Exact nonempty original native mask required")
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    x0 = min(SIZE-TILE, max(0, (int(xs[0])+int(xs[-1])+1-TILE)//2))
    y0 = min(SIZE-TILE, max(0, (int(ys[0])+int(ys[-1])+1-TILE)//2))
    x1, y1 = x0+TILE, y0+TILE
    halo = (max(0,int(xs[0])-RADIUS), max(0,int(ys[0])-RADIUS),
            min(SIZE,int(xs[-1])+1+RADIUS), min(SIZE,int(ys[-1])+1+RADIUS))
    if not (x0 <= halo[0] and y0 <= halo[1] and halo[2] <= x1 and halo[3] <= y1):
        return None
    window = mask[y0:y1, x0:x1]
    width = 2*RADIUS+1
    band = maximum_filter(window, size=width, mode="constant", cval=0) & ~minimum_filter(
        window, size=width, mode="constant", cval=0)
    yx = np.argwhere(band) + [y0,x0]
    need(len(yx) > 0, "The exact contour band must be nonempty")
    return (x0,y0,x1,y1), yx


def registered_values(field, window, yx):
    """Render logits at pixel centers and read the same original-image indices."""
    import torch
    from torch.nn.functional import interpolate

    x0,y0,x1,y1 = window
    y, x = torch.as_tensor(yx.T, dtype=torch.int64)
    need(field.ndim == 2 and field.dtype == torch.float32 and field.device.type == "cpu"
         and bool((y>=y0).all() and (y<y1).all() and (x>=x0).all() and (x<x1).all()),
         "FP32 mask and exact common in-window original coordinates required")
    dense = interpolate(field[None,None], size=(y1-y0,x1-x0), mode="bilinear", align_corners=False)[0,0]
    return dense[y-y0,x-x0]


def choose(roster, stream, codec):
    """Choose fixed core source groups and one existing proposal each, with no GT."""
    import numpy as np

    offsets = {}
    for row in roster:
        offsets[row["image_id"]] = stream.tell()
        line = stream.readline(4*MIB+1)
        need(0 < len(line) <= 4*MIB and line.endswith(b"\n"), "Exact original P316 candidate stream row")
    need(not stream.read(1), "Exact closed P316 stream length and roster")
    core = [r for r in roster if r["core"] is True and r["stress"] is False]
    need(len(core) == 109, "Exact original FIT114 core population")
    ordered = sorted(core, key=lambda r: (sha256(SALT+bytes.fromhex(r["encoded_sha256"])).digest(),
                                          r["group"], r["image_id"]))
    selected, groups, digests, inspected = [], set(), set(), 0
    for row in ordered:
        if row["group"] in groups or row["encoded_sha256"] in digests:
            continue
        stream.seek(offsets[row["image_id"]])
        record = json.loads(stream.readline(4*MIB+1))
        metadata = ("image_id","group","encoded_sha256","core","stress")
        need(all(record[k] == row[k] for k in metadata), "Exact P316 image identity and order")
        inspected += 1
        baseline = [{k:v for k,v in c.items() if k != "points"} for c in record["candidates"]]
        need([c["native_rank"] for c in baseline] == sorted({c["native_rank"] for c in baseline}),
             "Exact P316 native rank order")
        for candidate in baseline:
            rle = candidate["segmentation"]
            rle_bytes = dict(rle, counts=rle["counts"].encode("ascii"))
            area = int(codec.area(rle_bytes))
            if not 20 <= area <= 1023:
                continue
            pred = codec.decode(rle_bytes).astype(np.bool_, copy=False)
            window = band_window(pred)
            if window is None:
                continue
            crop, yx = window
            selected.append((row, baseline, candidate, pred, crop, yx))
            groups.add(row["group"])
            digests.add(row["encoded_sha256"])
            break
        if len(selected) == COUNT:
            break
    need(len(selected) == COUNT, "Exactly 16 distinct prelocked core source groups required")
    return selected, inspected


def lowpass_image(image, transform):
    """Verify the stock antialiased PIL resize before reconstructing the control."""
    from PIL import Image

    stock, target = transform.transforms[0](image, None)
    witness = image.resize((576,576), resample=Image.Resampling.BILINEAR)
    need(target is None and stock.mode == witness.mode == "RGB" and stock.size == witness.size
         and stock.tobytes() == witness.tobytes(), "Stock first step must be exact PIL bilinear antialiased 800 to 576")
    return stock.resize((SIZE,SIZE), resample=Image.Resampling.BILINEAR)


def run(args, p316, p318, p314, query):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    started = perf_counter()
    args.p314 = p314
    p192, cohort_bytes, _, _ = p314.read_original(args.code_root,args.original)
    reader = p318.load_scope(args.code_root)
    report = p318.read_closed_capture(SimpleNamespace(p316_report_sha256=p318.P316_REPORT_SHA,
        p316_terminal_sha256=p318.P316_TERMINAL_SHA, p316_root=args.p316_root, p316_terminal=args.p316_terminal), reader)
    roster = json.loads(cohort_bytes)["images"]
    source = args.p316_root/"CANDIDATES_PRIVATE.jsonl"
    with source.open("rb") as stream:
        identity = p318.verify_stream(source,stream,report["candidate_stream_bytes"],report["candidate_stream_sha256"])
        chosen, inspected = choose(roster,stream,codec)
        p318.require_unchanged(source,stream,identity)
    selection = [[row["image_id"],cand["query_id"],cand["native_rank"],crop[0],crop[1]]
                 for row, _, cand, _, crop, _ in chosen]
    selection_raw = json.dumps(selection,separators=(",",":")).encode("utf-8")
    selection_sha = sha256(selection_raw).hexdigest()
    need(EXPECTED_SELECTION_SHA in (selection_sha,sha256(selection_raw+b"\n").hexdigest())
         and sum(len(yx) for *_,yx in chosen) == 9429,
         "Frozen independent P326 selection digest and full band count differ")

    base, filtered, shift = p316.source_model(args)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    need(not torch.cuda.is_initialized(), "CPU-only frozen forward required")
    model, config, train, _ = base.load_model(args.checkpoint,p314.CHECKPOINT_SHA,torch,args.output.parent,
                                             filtered.TRAINING_PROTOCOL)
    model = model.cpu().eval()
    need(not model.training and model.num_queries == 164 and model.group_detr == 13,
         "Exact original frozen CPU Q164 carrier")
    head = model.segmentation_head
    transform = make_coco_transforms_square_div_64("val",576,multi_scale=False,expanded_scales=False,
        skip_random_resize=True,patch_size=config.patch_size,num_windows=config.num_windows,aug_config=None,
        scale_jitter=train.scale_jitter,gpu_postprocess=False)

    def forward(image):
        tensor, target = transform(image,None)
        need(target is None and tensor.shape == (3,576,576) and tensor.dtype == torch.float32,
             "Exact stock image-only val576 input required")
        sample = nested_tensor_from_tensor_list([tensor],block_size=config.patch_size*config.num_windows)
        need(sample.mask is not None and not bool(sample.mask.any()), "Exact stock no-padding inference")
        with query.capture_final_query(head) as capture:
            raw = model(sample)
        cache = capture.seal(raw["pred_masks"])
        need(cache.spatial.shape[-2:] == (144,144), "Same original and tile 144 mask lattice")
        return raw, cache

    calls, points_total, candidate_count = 0, 0, 0
    equal_input_crops, absolute_input_sum, absolute_input_max = 0, 0, 0
    digest = sha256()
    args.output.mkdir()
    with args.bundle.open("rb") as stream:
        identity = p318.verify_stream(args.bundle,stream,151_320_868,reader.BUNDLE_SHA)
        path = args.output/"TILED_BAND_PRIVATE.jsonl"
        with ZipFile(stream) as archive, path.open("xb") as out, torch.inference_mode():
            need(reader.member(archive,"COHORT_PRIVATE.json",reader.COHORT_SHA,MIB) == cohort_bytes,
                 "Exact P316 historical FIT cohort in original image bundle")
            for row, original, candidate, pred, crop, yx in chosen:
                image = p316.image_rows(archive,row,shift,np,decode_pil_image_bytes)
                reconstructed = lowpass_image(image,transform)
                raw, full = forward(image)
                observed, _ = p316.observed_rows(raw,image_id=row["image_id"],encoded_sha=row["encoded_sha256"],
                                                 torch=torch,np=np,codec=codec,base=base,postprocess=PostProcess)
                need([{k:v for k,v in c.items() if k != "points"} for c in observed] == original,
                     "All same-run full native candidate fields must exactly replay closed P316 CPU rows")
                candidate_count += len(observed)
                q = candidate["query_id"]
                fields = [registered_values(raw["pred_masks"][0,q],(0,0,SIZE,SIZE),yx)]
                views = image.crop(crop), reconstructed.crop(crop)
                rgb_difference = np.abs(np.asarray(views[0],dtype=np.int16)-np.asarray(views[1],dtype=np.int16))
                local_max = int(rgb_difference.max())
                equal_input_crops += int(local_max == 0)
                absolute_input_sum += int(rgb_difference.sum())
                absolute_input_max = max(absolute_input_max, local_max)
                for view in views:
                    _, local = forward(view)
                    need(torch.equal(local.bias,full.bias), "Frozen native segmentation bias must be identical")
                    mixed = query.FinalQueryCache(local.spatial,full.queries,full.bias)
                    tile_field = query.matched_dense_logits(head.query_features_proj,mixed,[q])[0]
                    fields.append(registered_values(tile_field,crop,yx))
                calls += 3
                y,x = yx.T
                inside = pred[y,x].tolist()
                need((fields[0]>0).tolist() == inside and all(bool(torch.isfinite(z).all()) for z in fields),
                     "Every signed band point must be finite and stock full sign must match native")
                points = [dict(y=int(y),x=int(x),inside=side,full=base_z,native_crop=native_z,lowpass_crop=lowpass_z)
                          for (y,x),side,base_z,native_z,lowpass_z in
                          zip(yx,inside,*(z.tolist() for z in fields),strict=True)]
                record = dict(image_id=row["image_id"],group=row["group"],encoded_sha256=row["encoded_sha256"],
                              crop_xyxy=crop,candidate=candidate,points=points)
                line = p192.encoded(record)
                need(out.write(line)==len(line), "Complete fixed tile band row required")
                digest.update(line)
                points_total += len(points)
                if calls % 12 == 0:
                    print(json.dumps(dict(event="P326_FROZEN_TILE_PROGRESS",images=calls//3,forward_calls=calls)),flush=True)
            out.flush()
            os.fsync(out.fileno())
        p318.require_unchanged(args.bundle,stream,identity)
    need(calls==48 and points_total==9429 and not torch.cuda.is_initialized(),
         "Exact fixed three-forwards-per-image CPU and full Cheb2 point population")
    safe = dict(status=STATUS, images=COUNT, source_groups=COUNT, inspected_core_images=inspected,
                FIT=True, DEV=False, TEST=False, GT_opened=False, training=False, optimizer_updates=0,
                device="cpu", forward_calls=calls, band_points=points_total, native_full_replay_exact=True,
                native_full_candidate_rows_verified=candidate_count, original_size=SIZE,tile_size=TILE,
                band_radius_original_pixels=RADIUS,area_inclusive=[20,1023],selection_salt=SALT[:-1].decode(),
                selection_sha256=EXPECTED_SELECTION_SHA, actual_compact_selection_sha256=selection_sha,
                source_checkpoint_sha256=p314.CHECKPOINT_SHA, p316_candidate_stream_sha256=report["candidate_stream_sha256"],
                p316_report_sha256=p318.P316_REPORT_SHA,p316_original_wait_sha256=p318.P316_TERMINAL_SHA,
                tiled_band_stream_sha256=digest.hexdigest(),tiled_band_stream_bytes=path.stat().st_size,
                cohort_sha256=reader.COHORT_SHA,fit_bundle_sha256=reader.BUNDLE_SHA,
                pixel_render="bilinear align_corners=False to half-open original crop extent",
                lowpass="verified stock PIL bilinear antialiased 800->576 then PIL bilinear 576->800 before identical crop",
                readout="frozen full-image native query and bias on each crop spatial basis",
                paired_input_byte_identical_crops=equal_input_crops,paired_input_max_abs_rgb=absolute_input_max,
                paired_input_mean_abs_rgb=absolute_input_sum/(COUNT*TILE*TILE*3),
                support_source_sha256={k:v[1] for k,v in SOURCES.items()},elapsed_seconds=perf_counter()-started)
    p192.write(args.output/"CAPTURE_SAFE.json",safe)
    print(json.dumps(dict(status=STATUS,report_sha256=sha256(p192.encoded(safe)).hexdigest(),
                          tiled_band_stream_sha256=digest.hexdigest()),sort_keys=True),flush=True)


def deny_network(event, _):
    if event in ("socket.connect","socket.getaddrinfo"):
        raise PermissionError("Offline fixed FIT-only tiled capture required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code-root","original","bundle","acceptance","checkpoint","p316-root","p316-terminal","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args(argv)
    for name,path in vars(args).items():
        if name != "output":
            need(path.is_absolute() and path.resolve(strict=True)==path, "Exact canonical closed input path required")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True)==args.output.parent
         and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical output required")
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.addaudithook(deny_network)
    run(args,*(load(args.code_root,key) for key in ("p316","p318","p314","query")))


if __name__ == "__main__":
    main()
