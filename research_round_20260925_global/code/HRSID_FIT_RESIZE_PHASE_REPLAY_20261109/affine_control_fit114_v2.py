"""Compare a generic composed affine and a pinned Kornia warp on frozen FIT masks."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


PILOT_SHA = "0b9726556a27dcca40ca9571390a8d04bb6c0dea7f4d982c29235283791858d9"
KORNIA_WARP_SHA = "f91c87e6a32150c06a7ea310a25df4e1af228776e74077a29827c95eb6172b89"
EXPECTED = {"direct": (248, 2770), "crop": (208, 1375)}
DIRECT_KERNEL_WITNESSES = (0, 1, 2, 5)
TIE_TOLERANCE = 1e-10


def need(condition, message):
    if not condition:
        raise ValueError(message)


def forward_edges(steps):
    """Compose ordinary forward affine matrices in (x,y) edge coordinates."""
    current = tuple(steps[0]["before"])
    result = np.eye(3, dtype=np.float64)
    for step in steps:
        before, after = tuple(step["before"]), tuple(step["after"])
        need(current == before and min(*before, *after) > 0, "Invalid transform chain")
        change = np.eye(3, dtype=np.float64)
        kind = step["kind"]
        if kind == "resize":
            change[0, 0], change[1, 1] = after[1] / before[1], after[0] / before[0]
        elif kind == "crop":
            left, top = step["left"], step["top"]
            need(0 <= left <= before[1] - after[1] and 0 <= top <= before[0] - after[0], "Crop outside canvas")
            change[0, 2], change[1, 2] = -left, -top
        elif kind == "hflip":
            need(before == after, "Flip changed the canvas")
            change[0, 0], change[0, 2] = -1., before[1]
        else:
            raise ValueError("Unknown frozen geometry")
        result = change @ result
        current = after
    return result


def inverse_edges(matrix, output_hw):
    inverse = np.linalg.inv(matrix)
    h, w = output_hw
    return (inverse[1, 1] * (np.arange(h, dtype=np.float64) + .5) + inverse[1, 2],
            inverse[0, 0] * (np.arange(w, dtype=np.float64) + .5) + inverse[0, 2])


def float_center_indices(matrix, output_hw, *, match_ties):
    axes = inverse_edges(matrix, output_hw)
    if match_ties:
        axes = [np.where(np.abs(x - np.rint(x)) <= TIE_TOLERANCE, np.rint(x), x) for x in axes]
    return tuple(np.floor(x).astype(np.intp) for x in axes)


def center_matrix(edges):
    plus = np.eye(3, dtype=np.float64)
    plus[:2, 2] = .5
    minus = np.eye(3, dtype=np.float64)
    minus[:2, 2] = -.5
    return (minus @ edges @ plus)[:2]


def pinned_module(path, digest):
    need(path.resolve(strict=True) == path and sha256(path.read_bytes()).hexdigest() == digest,
         "Original pilot source differs")
    spec = importlib.util.spec_from_file_location("_closed_fit_geometry_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_arrays(raw, target, trace, steps, *, ref, one, row_id, torch, kornia, tally, cursor):
    need(raw.ndim == 3 and raw.dtype == torch.bool, "Expected exact native Boolean input masks")
    ids = target[row_id]
    input_hw = tuple(raw.shape[-2:])
    output_hw = tuple(target["size"].tolist())
    source = raw[ids].numpy()
    reference = one(raw, target, trace)["masks"].numpy()
    rows, cols = ref.source_indices(steps, input_hw)
    need(np.array_equal(reference, source[:, rows[:, None], cols]), "Existing one-pass raster changed")
    expected_grid = rows[:, None] * input_hw[1] + cols

    affine = forward_edges(steps)
    _, _, scales, offsets = ref._compose(steps, input_hw)
    ties = tuple(np.array([(a*Fraction(2*j+1,2)+b).denominator == 1 for j in range(n)])
                 for a,b,n in zip(scales,offsets,output_hw,strict=True))
    tie_grid = ties[0][:, None] | ties[1]
    tally["exact_tie_locations"] += int(tie_grid.sum())
    coordinates = inverse_edges(affine, output_hw)
    need(all(np.array_equal(np.abs(x - np.rint(x)) <= TIE_TOLERANCE, tie)
             for x,tie in zip(coordinates,ties,strict=True)), "Floating-point tie tolerance misclassifies this trace")
    grids = {}
    for name, control in (("float_floor", False), ("float_match_ties", True)):
        rr, cc = float_center_indices(affine, output_hw, match_ties=control)
        need(rr.min() >= 0 and rr.max() < input_hw[0] and cc.min() >= 0 and cc.max() < input_hw[1],
             "Generic affine sampled outside the source")
        grids[name] = rr[:, None] * input_hw[1] + cc

    transform = torch.as_tensor(center_matrix(affine), dtype=torch.float64)[None]
    ramp = torch.arange(1, input_hw[0]*input_hw[1]+1, dtype=torch.float64).view(1, 1, *input_hw)
    warped = kornia.warp_affine(ramp, transform, output_hw, flags="nearest", padding_mode="zeros",
                               align_corners=True)[0,0].numpy()
    need(warped.min() >= 1 and warped.max() <= ramp.numel() and np.array_equal(warped, np.rint(warped)),
         "Kornia warped source IDs outside the valid integral source canvas")
    grids["kornia031_aligned"] = warped.astype(np.intp) - 1
    flat = source.reshape(len(source), input_hw[0]*input_hw[1])
    area = output_hw[0]*output_hw[1]
    reference_live = reference.reshape(len(source), area).any(1)
    tally["views"] += 1
    tally["rows"] += len(source)
    for arm, grid in grids.items():
        differed = grid != expected_grid
        values = flat[:,grid]
        xor = values != reference
        differs = xor.reshape(len(source), area).any(1)
        tally.update({arm+"_index_differences":int(differed.sum()),
                      arm+"_non_tie_index_differences":int((differed & ~tie_grid).sum()),
                      arm+"_views_changed":int(differs.any()),
                      arm+"_mask_rows_changed":int(differs.sum()),
                      arm+"_mask_xor":int(xor.sum()),
                      arm+"_non_tie_mask_xor":int((xor & ~tie_grid).sum()),
                      arm+"_lost_nonempty":int((reference_live & ~values.reshape(len(source), area).any(1)).sum())})
        if arm == "kornia031_aligned" and len(source) and cursor in DIRECT_KERNEL_WITNESSES:
            direct = kornia.warp_affine(torch.from_numpy(source).to(torch.float64)[None],
                                       transform, output_hw, flags="nearest", padding_mode="zeros",
                                       align_corners=True)[0].numpy()
            need(np.array_equal(direct, values), "Kornia mask and source-index probes differ")
            tally["direct_kernel_mask_witnesses"] += len(source)


def run(args):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.environ.get("OMP_NUM_THREADS") == "1",
         "One-thread CPU-only FIT scope required")
    sys.path.insert(0,str(args.kornia_site))
    import kornia
    path = args.kornia_site / "kornia/geometry/transform/imgwarp.py"
    need(kornia.__version__ == "0.3.1" and Path(kornia.__file__).resolve().is_relative_to(args.kornia_site)
         and sha256(path.read_bytes()).hexdigest() == KORNIA_WARP_SHA, "Exact Kornia runtime required")
    pilot = pinned_module(args.pilot, PILOT_SHA)
    native, _, torch, _ = pilot.install_support(args.code, args.shapely_site)
    from common_mask_supervision import ROW_ID
    from one_pass_fit_transform import substitute_one_pass_masks
    from paired_fit_transform import _check_output, _prepare
    import reference_resize
    from replay_fit import normalized_trace
    from trace_resize import capture_transform

    audit = native.NativeInputAudit()
    counts = defaultdict(Counter)
    with args.bundle.open("rb") as stream:
        before = pilot.fingerprint(os.fstat(stream.fileno()))
        need(before[2] == pilot.BUNDLE_BYTES, "Wrong FIT archive length")
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
        need(digest.hexdigest() == pilot.BUNDLE_SHA, "Wrong frozen FIT archive")
        stream.seek(0)
        with ZipFile(stream) as archive:
            def member(name, expected, bound):
                return pilot.pinned_member(archive, name, expected, bound)
            cohort = json.loads(member("COHORT_PRIVATE.json",pilot.COHORT_SHA,1024**2))
            index = json.loads(member("FIT114_COCO_PRIVATE.json",pilot.SUBSET_SHA,4*1024**2))
            history = [json.loads(line) for line in member(
                "HISTORICAL_VIEWS_PRIVATE.jsonl",pilot.HISTORY_SHA,1024**2).splitlines()]
            rows = cohort["images"]
            images = {x["id"]:x["file_name"] for x in index["images"]}
            labels = defaultdict(list)
            for label in index["annotations"]:
                need(label["image_id"] in images, "Source label outside frozen FIT cohort")
                labels[label["image_id"]].append(label)
            need(cohort["fit_only"] and len(rows) == len(images) == 114 and len(history) == 456
                 and len(index["annotations"]) == 1109 and len(archive.namelist()) == 117
                 and set(archive.namelist()) == {"COHORT_PRIVATE.json","FIT114_COCO_PRIVATE.json",
                     "HISTORICAL_VIEWS_PRIVATE.jsonl",*("images/"+r["native_basename"] for r in rows)},
                 "Exact closed FIT-only scope required")
            for ordinal, row in enumerate(rows):
                image_id, name = row["image_id"], row["native_basename"]
                need(images[image_id] == name, "Original image roster differs")
                png = member("images/"+name,row["encoded_sha256"],4*1024**2)
                image, scale = native.io_utils.decode_pil_image_bytes(png,draft_size=None)
                need(len(png) == row["encoded_bytes"] and image.size == (800,800) and image.mode == "RGB"
                     and scale == (1.,1.) and pilot.digest(np.asarray(image).tobytes()) == row["decoded_rgb_sha256"],
                     "Original PNG conversion differs")
                image, target = audit.convert(image, dict(image_id=image_id,annotations=labels[image_id]))
                prepared = _prepare(image,target)
                for rep in range(pilot.REPEATS):
                    cursor = ordinal*pilot.REPEATS+rep
                    old = history[cursor]
                    seed = pilot.BASE_SEED+cursor
                    need(old["ordinal"] == ordinal and old["seed"] == seed and old["group"] == row["group"],
                         "Wrong original augmentation schedule")
                    native.set_seed(seed)
                    with capture_transform(native._torchvision,mode="center") as trace:
                        b_image, b = audit.transform(image.copy(),{k:v.clone() for k,v in prepared.items()})
                    _check_output(b_image,b,prepared)
                    steps = normalized_trace(trace)
                    path_type = "crop" if any(step["kind"] == "crop" for step in steps) else "direct"
                    need(old["steps"] == steps and old["path"] == path_type and
                         (len(prepared[ROW_ID]),len(b[ROW_ID]),int((~native.mask_empty(b)).sum()))
                         == tuple(old["rows"][k] for k in ("raw_rows","center_present","center_nonempty")),
                         "Frozen trace and exact B row census required")
                    collect_arrays(prepared["masks"], b, trace, steps, ref=reference_resize,
                                   one=substitute_one_pass_masks,row_id=ROW_ID,torch=torch,kornia=kornia,
                                   tally=counts[path_type],cursor=cursor)
                if (ordinal+1)%20 == 0 or ordinal+1 == len(rows):
                    print(json.dumps(dict(event="COMPOSED_AFFINE_FIT_IMAGES_DONE",images=ordinal+1)),flush=True)
        need(before == pilot.fingerprint(os.fstat(stream.fileno())) == pilot.fingerprint(args.bundle.stat()),
             "Frozen FIT archive changed")
    need({p:(v["views"],v["rows"]) for p,v in counts.items()} == EXPECTED
         and not torch.cuda.is_initialized(),"Incomplete original B-conditioned mask population")
    report = dict(status="HRSID_FIT114_COMPOSED_AFFINE_CONTROL_ACTUAL", scope="B-conditioned FIT masks",
                  views=456, source_images=114, tie_tolerance=TIE_TOLERANCE, kornia="0.3.1",
                  kornia_explicit_align_corners=True,kornia_cpu_dtype="float64",
                  trace_and_b_row_census_match=True,counts={p:dict(v) for p,v in counts.items()},
                  model_forward_calls=0,optimizer_updates=0,FIT=True,DEV=False,TEST=False,AP=False)
    raw = (json.dumps(report,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as output:
        output.write(raw)
    print(json.dumps(dict(event=report["status"],sha256=sha256(raw).hexdigest())),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle","code","pilot","shapely-site","kornia-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
