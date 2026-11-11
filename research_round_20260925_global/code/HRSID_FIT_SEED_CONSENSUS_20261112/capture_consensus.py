"""Capture the unchanged three-seed FIT masks and assess candidate match coverage."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL_SHA = "38c26159cc9ef1fab493b01f509962d922999882e010af1ac91796af09ece41b"
SOURCE_PLAN_SHA = "272d89c6fe9a5fb14809db52a715634dcf3419ebd481ed169dc22855e12132a9"
ADMISSION_SHA = "4acb5fbcaf35c5a32510f5f93ccf51f34e08641fd6d007c78c85becde7496727"
POST_SHA = "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5"
MATCH_SHA = "caf79d33d7897999215c04c01dd6d0b55fc6fb22a1de6168dd0c9da647f39e6b"
SEEDS = (5201, 5202, 5203)
SOURCE_DIRS = {
    "phase": "HRSID_MASK_PHASE_E24_DEV705_20261109",
    "repeat": "HRSID_MASK_PHASE_REPEAT_DEV705_20261109",
    "filtered": "HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109",
    "base": "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109",
    "fit_audit": "HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109",
    "recipe": "../configs",
}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(path, digest, limit):
    need(path.resolve(strict=True) == path and 0 < path.stat().st_size <= limit, "Canonical pinned input " + path.name)
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Input digest " + path.name)
    return raw


def module(path, digest, name):
    m = ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(exact(path, digest, 2**20), str(path), "exec"), m.__dict__)
    return m


def encode(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, data):
    with path.open("xb") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    return sha256(data).hexdigest()


def source_refs(code, admission):
    refs = {}
    for role, (name, digest) in admission.SOURCE_PINS.items():
        path = (code / SOURCE_DIRS[role] / name).resolve(strict=True)
        exact(path, digest, 2**20)
        refs[role] = dict(path=path.as_posix(), bytes=path.stat().st_size, sha256=digest)
    return refs


def manifest_rows(manifest):
    rows = manifest["images"]
    need(len(rows) == len({r["group"] for r in rows}) == 36
         and len({r["encoded_sha256"] for r in rows}) == len({r["decoded_rgb_sha256"] for r in rows}) == 36
         and [r["ordinal"] for r in rows] == list(range(1, 37)), "Exact 36 unique ordered source prefixes and images")
    for row in rows:
        name = row["basename"]
        need(re.fullmatch(r"P\d+_\d+_\d+_\d+_\d+\.png", name) and name.split("_")[0] == row["group"],
             "Canonical native source filename")
    return rows


def arm_run(args, arm, seed, refs, rows, admission, post):
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    start = perf_counter()
    center = arm["arms"]["center"]
    checkpoint = center["checkpoint"]
    lock = dict(seed=seed, mode="center", checkpoint=checkpoint, pair_receipt=arm["pair_receipt"],
                native_receipt=arm["arms"]["native"]["receipt"], center_receipt=center["receipt"], sources=refs)
    base = admission.admit_source(lock)
    for name, digest in base.MODEL_SOURCES.items():
        exact(Path(import_module(name).__file__), digest, 2**20)
    output = args.output / str(seed)
    output.mkdir()
    model, config, train, _ = base.load_model(Path(checkpoint["path"]), checkpoint["sha256"], torch,
                                              output, admission.TRAINING_PROTOCOLS[seed])
    model.eval()
    need(model.num_queries == 164 and model.group_detr == 13
         and all(value.device.type == "cpu" for value in model.parameters()), "Accepted frozen Q164 CPU model")
    transform = make_coco_transforms_square_div_64(
        "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=config.patch_size, num_windows=config.num_windows, aug_config=None,
        scale_jitter=train.scale_jitter, gpu_postprocess=False,
    )
    digest = sha256()
    all_rows = []
    logit_total = logit_bytes = mapped = nonempty = 0
    with (output / "CANDIDATES_PRIVATE.jsonl").open("xb") as stream, torch.inference_mode():
        for row in rows:
            ordinal = row["ordinal"]
            raw_png = exact(args.fit_images / row["basename"], row["encoded_sha256"], 4*2**20)
            image, scales = decode_pil_image_bytes(raw_png, draft_size=None)
            need(image.mode == "RGB" and image.size == (800, 800) and scales == (1., 1.)
                 and len(raw_png) == row["encoded_bytes"]
                 and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
                 "Exact selected PNG and decoded identity")
            tensor, target = transform(image, None)
            need(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32,
                 "Frozen image-only val576 transform")
            sample = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            need(sample.mask is not None and not bool(sample.mask.any()), "No native canvas padding")
            raw = model(sample)
            observed = post.aligned_query_rows(raw, (800, 800), torch=torch, np=np, codec=codec, processor=PostProcess)
            areas = codec.area(observed["all_rles"])
            mapped += len(observed["native_ids"])
            active = []
            logit_ids = []
            for query, rank in zip(observed["native_ids"], observed["native_ranks"], strict=True):
                area = int(areas[query])
                if area == 0:
                    continue
                keep = 16 <= area < 2048
                active.append(dict(query=query, rank=rank, score=observed["ship_scores"][query],
                                   area=area, rle=observed["all_rles"][query],
                                   logit_row=len(logit_ids) if keep else None))
                if keep:
                    logit_ids.append(query)
            nonempty += len(active)
            indices = torch.tensor(logit_ids, dtype=torch.long)
            planes = raw["pred_masks"][0].index_select(0, indices).contiguous().numpy().astype("<f4", copy=False)
            need(planes.shape == (len(logit_ids), 144, 144) and bool(np.isfinite(planes).all()), "Native raw plane roster")
            logit_path = output / f"{ordinal:03d}.npy"
            with logit_path.open("xb") as handle:
                np.save(handle, planes, allow_pickle=False)
            binary = logit_path.read_bytes()
            logit_total += len(logit_ids)
            logit_bytes += len(binary)
            block = dict(ordinal=ordinal, group=row["group"], basename=row["basename"],
                         rgb_sha256=row["decoded_rgb_sha256"], mapped=len(observed["native_ids"]),
                         candidates=active, logit=dict(filename=logit_path.name, sha256=sha256(binary).hexdigest(),
                                                       bytes=len(binary), rows=len(logit_ids)))
            line = encode(block)
            need(stream.write(line) == len(line), "Exact prediction stream write")
            digest.update(line)
            all_rows.append(block)
            if ordinal % 12 == 0:
                print(json.dumps(dict(event="FIT_SEED_CONSENSUS_PROGRESS", seed=seed, images=ordinal)), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    arm_result = dict(seed=seed, images=len(all_rows), model_forwards=len(all_rows), raw_queries=164*len(all_rows),
                      native_rows=100*len(all_rows), mapped=mapped, nonempty=nonempty, raw_logit_planes=logit_total,
                      logit_bytes=logit_bytes, checkpoint_sha256=checkpoint["sha256"],
                      prediction_sha256=digest.hexdigest(), worker_seconds=perf_counter()-start)
    write(output / "ARM_SAFE.json", encode(arm_result))
    return all_rows, arm_result


def coverage(blocks, match, output):
    from pycocotools import mask as codec

    census = {seed: Counter() for seed in (5201, 5203)}
    cover_groups, nonzero_groups = set(), set()
    offsets = Counter()
    eligible = covered = 0
    lines = []
    for anchor, left, right in zip(blocks[5202], blocks[5201], blocks[5203], strict=True):
        need(anchor["ordinal"] == left["ordinal"] == right["ordinal"]
             and anchor["rgb_sha256"] == left["rgb_sha256"] == right["rgb_sha256"], "Exact shared image identity")
        a = anchor["candidates"]
        donor_rows = {5201:left["candidates"], 5203:right["candidates"]}
        matched = {}
        for seed, candidates in donor_rows.items():
            if a and candidates:
                matrix = codec.iou([p["rle"] for p in a], [p["rle"] for p in candidates], [0]*len(candidates))
            else:
                matrix = np.empty((len(a), len(candidates)), dtype=np.float64)
            matched[seed], states = match.reciprocal_matches(matrix)
            census[seed].update(state for p, state in zip(a, states, strict=True) if 32 <= p["area"] < 1024)
        kept = []
        for i, point in enumerate(a):
            if not 32 <= point["area"] < 1024:
                continue
            eligible += 1
            if not all(i in matched[seed] for seed in donor_rows):
                continue
            donors = {seed: donor_rows[seed][matched[seed][i]] for seed in donor_rows}
            need(point["logit_row"] is not None and all(p["logit_row"] is not None for p in donors.values()),
                 "Every covered candidate has all three native planes")
            offset = sorted([point["area"], *[p["area"] for p in donors.values()]])[1] - point["area"]
            offsets["add" if offset > 0 else "drop" if offset < 0 else "zero"] += 1
            covered += 1
            cover_groups.add(anchor["group"])
            if offset:
                nonzero_groups.add(anchor["group"])
            kept.append(dict(anchor_rank=point["rank"], donor_ranks={seed:p["rank"] for seed,p in donors.items()},
                             median_area_offset=offset))
        lines.append(encode(dict(ordinal=anchor["ordinal"], matches=kept)))
    assoc_digest = write(output / "ASSOCIATIONS_PRIVATE.jsonl", b"".join(lines))
    nonzero = offsets["add"] + offsets["drop"]
    gate = covered >= 60 and len(cover_groups) >= 20 and nonzero >= 30 and len(nonzero_groups) >= 15
    return dict(eligible=eligible, covered_both=covered, covered_groups=len(cover_groups),
                nonzero_median_area_offset=nonzero, nonzero_groups=len(nonzero_groups),
                offsets={k:offsets[k] for k in ("add","drop","zero")}, per_donor=census,
                associations_sha256=assoc_digest, gate="COVERAGE_PASS" if gate else "COVERAGE_STOP")


def run(args):
    from time import perf_counter
    import torch

    began = perf_counter()
    source = Path(__file__).resolve().parent
    exact(source / "PROTOCOL.md", PROTOCOL_SHA, 65536)
    need(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(),
         "Isolated CPU process with hidden CUDA")
    need(torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0"
         and version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4", "Exact frozen CPU package versions")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    need(args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists()
         and shutil.disk_usage(args.output.parent).free >= 4 * 2**30, "Fresh CPU output with 4 GiB free disk")
    admission = module(args.code / "HRSID_TO_SSDD_FROZEN_DEV187_20261110/export_frozen_hrsid_ssdd.py", ADMISSION_SHA,
                       "_consensus_admission")
    post = module(args.code / "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py", POST_SHA, "_consensus_post")
    match = module(source / "consensus_match.py", MATCH_SHA, "_consensus_match")
    plan = json.loads(exact(args.source_plan, SOURCE_PLAN_SHA, 2**20))
    manifest = json.loads(exact(args.manifest, args.manifest_sha, 65536))
    need(manifest["status"] == "FIT_SEED_CONSENSUS_IMAGE_MANIFEST_V1" and manifest["variant"] == "P316_COORDINATE_DISJOINT_ORACLE36_PREFIX_EXCLUDED",
         "Image-only FIT manifest")
    rows = manifest_rows(manifest)
    refs = source_refs(args.code, admission)
    args.output.mkdir()
    blocks, arms = {}, []
    for seed in SEEDS:
        blocks[seed], result = arm_run(args, plan["pairs"][str(seed)], seed, refs, rows, admission, post)
        arms.append(result)
    summary = coverage(blocks, match, args.output)
    result = dict(status="HRSID_FIT_SEED_CONSENSUS_COVERAGE_ACTUAL", FIT=True, DEV=False, TEST=False,
                  GT_opened=False, AP=False, optimizer_updates=0, images=36, model_forwards=108,
                  arms=arms, coverage=summary, manifest_sha256=args.manifest_sha, protocol_sha256=PROTOCOL_SHA,
                  worker_seconds=perf_counter()-began)
    result_sha = write(args.output / "RESULT_SAFE.json", encode(result))
    print(json.dumps(dict(status=result["status"], gate=summary["gate"], result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("code", "source-plan", "manifest", "fit-images", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    def offline(event, _args):
        if event.startswith("socket."):
            raise PermissionError("FIT capture cannot use network")
    sys.addaudithook(offline)
    run(parser.parse_args())
