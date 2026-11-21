"""Paired stage latency for all 232 closed target images, without target labels."""
from collections import Counter
from hashlib import sha256
from importlib import metadata, util
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from time import get_clock_info, perf_counter, perf_counter_ns, process_time_ns
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as codec
import torch


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["PSEG_COST_SOURCE_ROOT"]).resolve(strict=True)
OWNER = Path(os.environ["PSEG_COST_AUDIT_OWNER"]).resolve(strict=True)
REPLAY = ROOT / "PSEG_TEST232_DFINE_FROZEN_REPLAY_ACTUAL_20261121_01"
ACCEPT = "e99d0f2bbd9e1839e260a88075f6af187e05ffc2b86abaf6cc33b788372b7b71"
REPLAY_SOURCE = "f1b6e83e98851bf867bd72f3ecb082ca2997632bdd1aedaaf64110cf06b04fd5"


def check(value, msg):
    if not value:
        raise ValueError(msg)


def pin(path, digest):
    data = path.read_bytes()
    check(sha256(data).hexdigest() == digest, f"Closed identity for {path.name}")
    return data


def read_closed_inputs():
    verified = json.loads(pin(ROOT/"PSEG_TEST232_DFINE_FROZEN_REPLAY_READBACK_ACTUAL_20261121_01/RESULT_SAFE.json", ACCEPT))
    path = Path(os.environ["PSEG_COST_REPLAY_SOURCE"]).resolve(strict=True)
    source = pin(path, REPLAY_SOURCE)
    replay = util.module_from_spec(util.spec_from_file_location("_closed_replay", path))
    exec(compile(source,str(path),"exec"),replay.__dict__)
    check(verified["status"] == replay.VERIFIED and verified["worker_sha256"] == REPLAY_SOURCE
          and verified["exact_native_support"] and verified["images"] == 232, "Closed five-arm independent replay")
    terminal = json.loads(pin(OWNER/"REPLAY_TERMINAL_PRIVATE.json", verified["original_tool_wait_sha256"]))
    replay.check_terminal(terminal, verified["replay_sha256"])
    result = json.loads(pin(REPLAY/"RESULT_SAFE.json", verified["replay_sha256"]))
    check(result["status"] == replay.PREPARED and result["edits_sha256"] == verified["edits_sha256"],
          "Closed original edit stream")
    refit = ROOT / "HRSID_DFINE_E24_FIT109_BCE_FINAL_REFIT_DRAFT_20261112_01"
    a = SimpleNamespace(capture=ROOT/"DFINE_HRSID_E24_TO_PSEG_SSDD_TEST232_CAPTURE_ACTUAL_20261121_02",
                        manifest=OWNER/"TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json",
                        manifest_sha=verified["manifest_sha256"], accepted=OWNER/"CAPTURE_READBACK/INDEPENDENT_SAFE.json",
                        accepted_sha=verified["acceptance_sha256"],images=OWNER/"IMAGE_ONLY",
                        refit_result=refit/"ACTUAL/RESULT_SAFE.json",refit_acceptance=refit/"INDEPENDENT/RESULT_SAFE.json",
                        weights=refit/"ACTUAL/POINT_HEAD_PRIVATE.npz")
    modules = replay.modules(ROOT/"github_release_20261011/research_round_20260925_global/code", a)
    modules.author.cpu_runtime()
    roster, receipt = replay.admit(modules, a)
    capture = json.loads(pin(a.capture/"RESULT_SAFE.json",receipt["export_result_sha256"]))
    edits = pin(REPLAY/"EDITS_PRIVATE.jsonl", verified["edits_sha256"])
    evidence = [json.loads(line) for line in edits.splitlines()]
    check(len(evidence) == len(roster) == 232, "All frozen target edit blocks")
    models = modules.old.restore(a)
    models[0].raw = modules.legacy.pin(a.pair_math,modules.native.PAIR)
    return a, modules, roster, receipt, capture, evidence, models, verified


def natural(m, models, coco, audit, planes, rgb):
    base, rank, free, learner, head, center, scale = models
    pair, sampling = m.native.make_native(base, rank, *rgb.shape[:2])
    image = base.image_features(rgb)
    output = list(coco)
    for i, (row, detail, plane) in enumerate(zip(coco, audit, planes, strict=True)):
        area = detail["mask_area"]
        if not 32 <= area < 1024:
            continue
        stock = codec.decode(row["segmentation"]).astype(bool, order="C")
        support = pair.paired_points(stock, detail["box_xyxy"], area, row["score"], plane, image, sampling)
        if support is None:
            continue
        coordinates, features = support[:2]
        probabilities = learner.probabilities(head, features, features, center, scale)
        actions = free.choose_free(probabilities, min(8, area//10, len(features)))
        if actions:
            output[i] = dict(row, segmentation=m.old.edit(stock, coordinates, actions))
    return output


def timed(operation):
    cpu = process_time_ns()
    wall = perf_counter_ns()
    result = operation()
    return result, (perf_counter_ns()-wall)*1e-6, (process_time_ns()-cpu)*1e-6


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(mean_ms=float(v.mean()),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),
                summed_seconds=float(v.sum()/1000))


def run():
    start = perf_counter()
    check(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
          "Isolated CPU experiment")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    a,m,roster,receipt,capture,evidence,models,verified = read_closed_inputs()
    check(not torch.cuda.is_initialized(), "CPU-only head and inputs")
    stream = iter(m.legacy.blocks(a,m.author,roster,receipt,True))
    totals = Counter()
    startup_seconds = perf_counter()-start
    rows = []
    for ordinal, expected in enumerate(evidence, 1):
        t = perf_counter()
        item,coco,audit,planes = next(stream)
        totals["capture_stream_read_parse_validate_seconds"] += perf_counter()-t
        t = perf_counter()
        ref = item["image_ref"]
        name = PurePosixPath(ref["path"]).name
        encoded = pin(a.images/name, ref["sha256"])
        check(len(encoded) == ref["bytes"], "JPEG bytes")
        totals["jpeg_read_and_encoded_digest_seconds"] += perf_counter()-t
        t = perf_counter()
        with Image.open(BytesIO(encoded)) as image:
            check(image.format == "JPEG", "JPEG decoding")
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"),dtype=np.uint8)
        totals["jpeg_decode_orient_rgb_seconds"] += perf_counter()-t
        check(tuple(item["original_hw"]) == rgb.shape[:2] and
              sha256(rgb.tobytes(order="C")).hexdigest() == expected["rgb_sha256"] and
              item["image_id"] == expected["image_id"], "Original pixels and image identity")
        reference = list(coco)
        for key, arms in expected["changes"].items():
            edit = arms.get("free_natural")
            if edit is not None:
                index = int(key)-1
                reference[index] = dict(coco[index],segmentation=edit["rle"])
                totals["edited_candidates"] += 1
                totals["swaps"] += edit["swaps"]
        operations = {"raw":lambda:list(coco), "natural":lambda:natural(m,models,coco,audit,planes,rgb)}
        methods = ("raw","natural") if ordinal % 2 else ("natural","raw")
        per_image = {}
        for label in methods:
            check(operations[label]() == (coco if label=="raw" else reference), "Warm exact " + label)
        for repeat in range(2):
            for label in methods if repeat == 0 else methods[::-1]:
                values, wall_ms, cpu_ms = timed(operations[label])
                check(values == (coco if label=="raw" else reference), "Timed exact " + label)
                per_image.setdefault(label,[]).append([wall_ms,cpu_ms])
        rows.append(per_image)
        totals["candidates"] += len(coco)
        if ordinal % 32 == 0 or ordinal == 232:
            print(json.dumps(dict(event="PSEG_CPU_COST_PROGRESS",images=ordinal)),flush=True)
    check(next(stream,None) is None, "All source streams verified through EOF")
    check(totals["edited_candidates"] == verified["counts"]["free_natural_edited"]
          and totals["swaps"] == verified["counts"]["free_natural_swaps"]
          and totals["candidates"] == verified["counts"]["candidates"]
          and not torch.cuda.is_initialized(), "All closed natural edits and CPU-only finish")
    raw, method = (np.asarray([row[name] for row in rows]) for name in ("raw","natural"))
    timings = {}
    for j,clock in enumerate(("wall","process_cpu")):
        for name,array in (("raw",raw),("natural",method),("incremental",method-raw)):
            timings[f"{clock}_{name}"] = summary(array[:,:,j].mean(axis=1))
    measured = dict(schema="PSEG_TEST232_DFINE_FROZEN_NATURAL_CPU_COST_V1",status="PAIRED_CPU_STAGE_TIMINGS_COMPLETE",
         prelock_sha256=sha256((HERE/"BENCHMARK_PROTOCOL.md").read_bytes()).hexdigest(),
         worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),independent_replay_sha256=ACCEPT,
         original_replay_sha256=verified["replay_sha256"],original_approved_capture_sha256=a.accepted_sha,
         source_endpoint_sha256=receipt["ema_sha256"],source_head_acceptance_sha256=verified["refit_accepted_sha256"],
         images=len(rows),warmup_pairs_per_image=1,measured_pairs_per_image=2,per_image_order="alternating AB/BA",
         complete_rle_score_rank_parity=True,raw_natural_output_checks_per_arm=3*len(rows),
         output_latency_includes="native size adapter setup, 5 image features, eligible proposals and occupancy, natural selection and edited COCO RLE",
         timings=timings, aggregate_incremental_wall_seconds_per_repetition=((method-raw)[:,:,0].sum(axis=0)/1000).tolist(),
         original_capture=dict(export_result_sha256=receipt["export_result_sha256"],images=capture["images"],
                               forward_postprocess_seconds=capture["forward_postprocess_seconds"],
                               total_capture_seconds=capture["elapsed_seconds"],model_precision=capture["model_precision"],
                               mask_resize_precision=capture["mask_resize_precision"]),
         stage_counts_and_input_times=dict(totals),startup_and_source_verification_seconds=startup_seconds,
         benchmark_body_seconds=perf_counter()-start,python=platform.python_version(),
         packages={n:metadata.version(n) for n in ("torch","numpy","scipy","Pillow","pycocotools")},
         cpu=os.environ.get("PSEG_COST_CPU_DESCRIPTION",platform.processor()),os=platform.platform(),torch_intra_threads=torch.get_num_threads(),
         torch_interop_threads=torch.get_num_interop_threads(), thread_environment={n:os.environ[n] for n in
         ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")},
         wall_clock=get_clock_info("perf_counter").implementation,other_HRSID_CPU_task_active_at_benchmark_setup=os.environ.get("PSEG_COST_SHARED_HOST") == "1",
         GT_opened=False,AP=False,optimizer_updates=0,model_forwards=0,cuda_initialized=False)
    observations=dict(units="milliseconds",axes=["image_in_original_order","repeat","wall_and_process_CPU"],
                      raw=raw.tolist(),natural=method.tolist())
    obs = json.dumps(observations,allow_nan=False,separators=(",",":")).encode()+b"\n"
    with (HERE/"PAIRED_TIMES_SAFE.json").open("xb") as out:
        out.write(obs)
    measured["paired_times_sha256"] = sha256(obs).hexdigest()
    payload = json.dumps(measured,allow_nan=False,indent=2).encode()+b"\n"
    with (HERE/"RESULT_SAFE.json").open("xb") as out:
        out.write(payload)
    print(json.dumps(dict(status=measured["status"],result_sha256=sha256(payload).hexdigest())),flush=True)


if __name__ == "__main__":
    run()
