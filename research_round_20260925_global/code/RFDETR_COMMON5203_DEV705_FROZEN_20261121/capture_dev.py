"""Image-only RF-DETR common5203 DEV705 native reference and frozen point-head replay."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile
import zlib


SELF = Path(__file__).resolve()
PRELOCK = SELF.with_name("PRELOCK.json")
SCOPE = dict(detector_seed=5203, DEV=True, FIT=False, TEST=False, AP=False, GT_opened=False, optimizer_updates=0)
FULL = "97449f58d1ad1a62c52c6f100fc872690e51d465e85e241660befba7a9690a71"
TREATMENT = "a50da3d740d9620a23c98b978b7cfaa383040f1e7542e2eb5da32ca4b0106da6"
FIT = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
MANIFEST = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
INIT = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
FIT_INDEX = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
OLD_CKPT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
PHASE = "RFDETR_COMMON5203_DEV705_FROZEN_PRELABEL_V1"
SEEDS = (5217, 5218, 5219)
ARMS = ("SF", "SU=F", "SU", "LW=S")
SHAPES = {"pred_logits": (1, 164, 2), "pred_boxes": (1, 164, 4), "pred_masks": (1, 164, 144, 144)}
NATIVE_STATUS = "RFDETR_COMMON5203_DEV705_NATIVE_CAPTURED_V1"
REFINE_STATUS = "RFDETR_COMMON5203_DEV705_FROZEN_CAPTURED_V1"


def need(test, message):
    if not test:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def hash_bytes(data):
    return sha256(data).hexdigest()


def is_sha(s, length=64):
    return type(s) is str and len(s) == length and all(ch in "0123456789abcdef" for ch in s)


def write(path, raw):
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "Short exclusive write")
        stream.flush()
        os.fsync(stream.fileno())
    return hash_bytes(raw)


def checked(path, expected, limit=8*1024**2):
    bound = Path(path)
    with bound.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        need(0 < size <= limit, "Closed input size: " + bound.name)
        data = stream.read(size + 1)
    need(len(data) == size and hash_bytes(data) == expected, "Closed input digest: " + bound.name)
    return data


def module(path, digest, name):
    result = ModuleType(name)
    result.__file__ = str(path)
    exec(compile(checked(path, digest), str(path), "exec"), result.__dict__)
    return result


def endpoint(receipt):
    need(receipt["status"] == "COMMON5203_SINGLE_CENTER_E24_ACCEPTED"
         and receipt["FIT"] is True and all(receipt[k] is False for k in ("TEST", "DEV", "AP")),
         "Independent single-center training acceptance required")
    arm = receipt["arm"]
    need(arm["status"] == "HRSID_MASK_COMMON_FIT_E24_ARM_ACCEPTED_V1" and
         arm["original_os_wait"] is True and type(arm["original_os_exit"]) is int and arm["original_os_exit"] == 0
         and arm["resource_stop"] is False and arm["completed_epochs"] == 24
         and arm["optimizer_updates"] == arm["ema_updates"] == 17640 and arm["filtered_empty_rows_total"] == 0
         and is_sha(arm["checkpoint_sha256"]) and arm["checkpoint_sha256"] != OLD_CKPT,
         "Accepted 5203 E24 fixed endpoint")
    support, phase = arm["common_support"], arm["phase"]
    need(support["policy"] == "ordered_common_nonempty_prepared_rows_v1" and support["mode"] == "center"
         and support["stage"] == "e24" and support["training_seed"] == phase["seed"] == 5203
         and support["initialization_seed"] == 5201 and support["initial_expanded_sha256"] == INIT
         and support["fit_index_sha256"] == FIT_INDEX and phase["mask_resize_mode"] == "center"
         and phase["stage"] == "e24" and phase["protocol"] == support["training_protocol"]
         and is_sha(phase["protocol"], 40) and is_sha(support["qualification_pair_sha256"]),
         "Exact common-row, seed and initial-state provenance")
    return arm


def input_path(variable):
    need(bool(os.environ.get(variable)), "Set " + variable)
    return Path(os.environ[variable]).expanduser().resolve(strict=True)


def release_code():
    root = os.environ.get("SAR_SEAL_RELEASE_ROOT")
    return (Path(root).expanduser().resolve(strict=True) if root else SELF.parents[2]) / "code"


def load_core(code, full):
    # The historical pinned helper resolves a source-index directory at import.
    # Its invoked source lookups below use the relocated public release tree.
    key = "SAR_SEAL_CLOSED_FIT_ROOT"
    original = os.environ.get(key)
    os.environ[key] = str(code.parent)
    try:
        core = module(code / "HRSID_FIT114_CORE36_CENTER5202_RAW_CAPTURE_20261112/capture.py",
                      full.CORE_SOURCE, "_portable_core")
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
    core.CODE = code
    return core


def admission(args):
    sources = release_code()
    receipt_path = input_path("SAR_SEAL_TRAINING_RECEIPT")
    arm = endpoint(json.loads(checked(receipt_path, args.training_acceptance_sha256)))
    paths = dict(python=Path(sys.executable).resolve(strict=True), code=sources, training_acceptance=receipt_path,
        checkpoint=input_path("SAR_SEAL_CHECKPOINT"),
        manifest=input_path("SAR_SEAL_DEV705_INPUTS") / "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json",
        archive=input_path("SAR_SEAL_PNG_ARCHIVE"), official_capture=input_path("SAR_SEAL_TREATMENT_SOURCE"),
        fit_result=input_path("SAR_SEAL_FIT_RESULT"), fit_acceptance=input_path("SAR_SEAL_FIT_ACCEPTANCE"),
        heads=input_path("SAR_SEAL_POINT_HEADS"), output=args.output.resolve())
    need(not paths["output"].exists() and paths["output"].parent.is_dir(), "Fresh capture output")
    fit = json.loads(checked(paths["fit_acceptance"], FIT))
    need(is_sha(fit["fit_result_sha256"]), "Accepted original point fit")
    return dict(status="RFDETR_COMMON5203_DEV705_FROZEN_CAPTURE_ADMITTED_V1", phase=PHASE,
        images=705, training_acceptance_sha256=args.training_acceptance_sha256, endpoint_sha256=arm["checkpoint_sha256"],
        fit_accepted_sha256=FIT, fit_result_sha256=fit["fit_result_sha256"], manifest_sha256=MANIFEST,
        capture_source_sha256=hash_bytes(SELF.read_bytes()), protocol_sha256=hash_bytes(PRELOCK.read_bytes()),
        treatment_source_sha256=TREATMENT, paths={k: str(v) for k, v in paths.items()}, **SCOPE)


def raw_hashes(raw, torch, image_id):
    identity = {"image_id": image_id}
    cpu = {}
    for key, shape in SHAPES.items():
        tensor = raw[key].detach()
        need(tuple(tensor.shape) == shape and tensor.dtype == torch.float32, "Author raw shape and dtype")
        tensor = tensor.to("cpu").contiguous()
        data = tensor.numpy().astype("<f4", copy=False)
        identity[key + "_sha256"] = sha256(memoryview(data).cast("B")).hexdigest()
        cpu[key] = tensor
    return cpu, encode(identity)


def render_with_backend(original, raw_cpu, raw_device, query_ids, ranks, log, torch, np, image_id, proof_dir):
    """Keep original CPU probability math; use native-device posterior only on a verified raster disagreement."""
    index = {raw_cpu[0, q].data_ptr(): (q, rank) for q, rank in zip(query_ids, ranks, strict=True)}
    def render(t, n, adapter, plane, stock):
        try:
            return original(t, n, adapter, plane, stock)
        except ValueError as err:
            if str(err) != "Exact native stock parity":
                raise
        q, rank = index[plane.__array_interface__["data"][0]]
        interpol = torch.nn.functional.interpolate
        cpu_mask = interpol(raw_cpu[:, q:q+1], (800, 800), mode="bilinear", align_corners=False)[0, 0] > 0
        diff = int(np.count_nonzero(cpu_mask.numpy() != stock))
        # RF-DETR PostProcess uses the native CUDA kernel. Deterministic mode
        # dispatches CUDA bilinear interpolation to a different decomposition.
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            native = interpol(raw_device[:, q:q+1], (800, 800), mode="bilinear", align_corners=False)[0, 0]
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        need(diff > 0 and np.array_equal((native > 0).cpu().numpy(), stock),
             "Same-device strict-sign raster must reproduce every stock pixel")
        probability = native.sigmoid().cpu().numpy()
        need(probability.dtype == np.float32 and np.isfinite(probability).all() and
             probability.min() >= 0 and probability.max() <= 1, "Finite native-device FP32 posterior")
        proof_dir.mkdir(exist_ok=True)
        raw_name = f"{image_id}_{q}_raw144.npy"
        prob_name = f"{image_id}_{q}_posterior800.npy"
        for name, value in ((raw_name, plane), (prob_name, probability)):
            with (proof_dir / name).open("xb") as f:
                np.save(f, np.ascontiguousarray(value, dtype="<f4"), allow_pickle=False)
        log.append(dict(image_id=image_id, native_rank=rank, query_id=q, cpu_disagreement_pixels=diff,
                        plane_file=raw_name, probability_file=prob_name, plane_sha256=hash_bytes(plane.tobytes()),
                        probability_sha256=hash_bytes(probability.tobytes())))
        return probability
    return render


def pack(index, line, predictions, witness, timings):
    rows = {}
    for seed in SEEDS:
        block = predictions[seed]
        need(set(block) == {"N", *ARMS} and encode(block["N"]) == line, "Shared exact native under all heads")
        rows[str(seed)] = {arm: block[arm] for arm in ARMS}
    return zlib.compress(encode(dict(index_zero_based=index, image_id=witness["image_id"],
        stock_block_sha256=hash_bytes(line), predictions=rows, witness=witness, timings=timings)), level=1)


def bind_loader(core, code, arm):
    loader = core.load("admission")
    phase = loader.load_phase(core.source("phase")[0])
    phase.TRAINING_PROTOCOL = arm["phase"]["protocol"]
    loader.TRAINING_SEED = 5203
    filtered, base = phase.load_evaluator(core.source("filtered")[0], core.source("base")[0])
    loader.bind_evaluator(phase, base, filtered, arm)
    base.FIT_AUDIT = core.source("fit_audit")[0]
    base.RECIPE = code.parent / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
    return base


def setup(record, paths, torch):
    code = paths["code"]
    full = module(code / "RFDETR_CENTER5202_FULL_CAPTURE_20261120/capture.py", FULL, "_dev_original_capture")
    core = load_core(code, full)
    receipt = json.loads(checked(paths["training_acceptance"], record["training_acceptance_sha256"]))
    arm = endpoint(receipt)
    need(arm["checkpoint_sha256"] == record["endpoint_sha256"], "Own independent checkpoint receipt")
    base = bind_loader(core, code, arm)
    for name, digest in base.MODEL_SOURCES.items():
        checked(Path(import_module(name).__file__), digest, 1024**2)
    model, config, train, fit = base.load_model(paths["checkpoint"], arm["checkpoint_sha256"], torch,
                                               paths["output"], arm["phase"]["protocol"])
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    transform = make_coco_transforms_square_div_64("val", 576, multi_scale=False, expanded_scales=False,
        skip_random_resize=True, patch_size=config.patch_size, num_windows=config.num_windows,
        aug_config=None, scale_jitter=train.scale_jitter, gpu_postprocess=False)
    return full, base, fit, model.to("cuda").eval(), transform, config


def scientific_worker(stage, ad_path, ad_sha):
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    record = json.loads(checked(ad_path, ad_sha))
    paths = {key: Path(path) for key, path in record["paths"].items()}
    need(record["capture_source_sha256"] == hash_bytes(SELF.read_bytes()) and record["phase"] == PHASE
         and record["protocol_sha256"] == hash_bytes(PRELOCK.read_bytes()) and record["fit_accepted_sha256"] == FIT
         and record["treatment_source_sha256"] == TREATMENT and record["manifest_sha256"] == MANIFEST
         and record["images"] == 705 and all(record[k] == v for k, v in SCOPE.items()), "Bound DEV-only worker")
    need(sys.version_info[:3] == (3, 10, 13) and Path(sys.executable).resolve() == paths["python"]
         and sys.flags.isolated and sys.dont_write_bytecode and sys.flags.utf8_mode and np.__version__ == "1.26.4"
         and torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
         and version("Pillow") == "12.3.0" and version("pycocotools") == "2.0.11", "Historical runtime")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    available, total = torch.cuda.mem_get_info()
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and available >= 6 * 1024**3,
         "Native single-device inference")
    torch.cuda.set_per_process_memory_fraction(min(.35, 6 * 1024**3 / total))
    full, base, fit, model, transform, config = setup(record, paths, torch)
    manifest = json.loads(checked(paths["manifest"], MANIFEST))
    need(len(manifest) == len({row["image_id"] for row in manifest}) == 705, "Original DEV705 roster")
    out = paths["output"] / stage
    out.mkdir()
    comparisons = stage == "refine"
    if comparisons:
        cap = module(paths["official_capture"], TREATMENT, "_historical_official_point_treatment")
        policy = cap.load_policy(paths["code"])
        heads = cap.restore_refits(policy, paths["fit_result"], paths["fit_acceptance"], paths["heads"],
                                   fit_result_sha=record["fit_result_sha256"], fit_acceptance_sha=FIT)
        standard_render = policy.features.render_posterior
        image_dir = out / "IMAGE_ROWS_PRIVATE"
        image_dir.mkdir()
    counts = Counter()
    times = Counter()
    native_digest, raw_digest, row_digest = sha256(), sha256(), sha256()
    corrections = []
    start = perf_counter()
    archive_identity = fit.archive_identity(paths["archive"].stat())
    need(archive_identity[2] == fit.ARCHIVE_BYTES, "Original archive size")
    with ExitStack() as stack, torch.inference_mode():
        handle = stack.enter_context(paths["archive"].open("rb"))
        need(fit.archive_identity(os.fstat(handle.fileno())) == archive_identity, "Original image archive handle")
        handle.seek(fit.CENTRAL_OFFSET)
        need(hash_bytes(handle.read(fit.CENTRAL_BYTES)) == fit.CENTRAL_SHA, "Original central directory")
        zipfile = stack.enter_context(ZipFile(handle))
        raw_stream = stack.enter_context((out / "RAW_HASHES_PRIVATE.jsonl").open("xb"))
        if comparisons:
            # Parent observed the original native process's actual zero exit before this process.
            stock_stream = stack.enter_context((paths["output"] / "native/DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl").open("rb"))
            raw_reference = stack.enter_context((paths["output"] / "native/RAW_HASHES_PRIVATE.jsonl").open("rb"))
        else:
            stock_stream = stack.enter_context((out / "DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl").open("xb"))
        for idx, row in enumerate(manifest):
            identity = row["image_id"]
            need((row["height"], row["width"]) == (800, 800), "Original dimensions")
            image = full.original_image(SimpleNamespace(mode="dev"), row, None, fit, zipfile)
            tensor, target = transform(image, None)
            need(target is None and tensor.dtype == torch.float32 and tuple(tensor.shape) == (3, 576, 576), "Native transform")
            sample = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            need(sample.mask is not None and not bool(sample.mask.any()), "Unpadded original image")
            begin = perf_counter()
            raw = model(sample.to("cuda"))
            torch.cuda.synchronize()
            forwarded = perf_counter()
            cpu, raw_line = raw_hashes(raw, torch, identity)
            raw_stream.write(raw_line)
            raw_digest.update(raw_line)
            if comparisons:
                line = stock_stream.readline()
                need(raw_reference.readline() == raw_line, "Two independent detector forwards differ")
                native, qids = cap.mapped_queries(raw, identity, line, base.native_rows, PostProcess)
            else:
                post = PostProcess(num_select=100, upsample_masks_to_image_size=True)
                native = base.native_rows(raw, image_id=identity, hw=(800, 800), torch=torch, postprocess=post,
                                          np=np, codec=codec)
                line = base.encoded(native)
                stock_stream.write(line)
            native_digest.update(line)
            counts.update(mapped=len(native["coco"]), native_bytes=len(line), images=1)
            if comparisons:
                mapped = perf_counter()
                torch.set_num_threads(1)
                torch.use_deterministic_algorithms(True)
                policy.features.render_posterior = render_with_backend(standard_render, cpu["pred_masks"], raw["pred_masks"],
                    qids, native["native_ranks"], corrections, torch, np, identity, out / "CORRECTION_PLANES_PRIVATE")
                predictions, witness = cap._treatment(cpu, np.asarray(image), native, qids, policy, heads)
                torch.use_deterministic_algorithms(False)
                torch.set_num_threads(2)
                witness["native_line_sha256"] = hash_bytes(line)
                witness["joint_scores_f32le_hex"] = cap.f32_bytes(raw["pred_logits"][0].sigmoid().cpu().numpy()).hex()
                timing = dict(detector_forward_seconds=forwarded-begin, native_postprocess_parity_seconds=mapped-forwarded,
                              refinement_cpu_seconds=perf_counter()-mapped)
                packed = pack(idx, line, predictions, witness, timing)
                write(image_dir / f"{idx:05d}.json.zlib", packed)
                row_digest.update(sha256(packed).digest())
                counts["row_bytes"] += len(packed)
                times.update(timing)
                # Release the prior image's GPU view before processing the next frame.
                policy.features.render_posterior = standard_render
            if (idx + 1) % 25 == 0 or idx == 0:
                print(json.dumps(dict(stage=stage, images=idx+1, mapped=counts["mapped"])), flush=True)
        if comparisons:
            need(not stock_stream.read(1) and not raw_reference.read(1), "Complete independent reference streams")
        need(fit.archive_identity(os.fstat(handle.fileno())) == fit.archive_identity(paths["archive"].stat())
             == archive_identity, "DEV archive changed while in use")
    result = dict(status=REFINE_STATUS if comparisons else NATIVE_STATUS, phase=PHASE, admission_sha256=ad_sha,
        training_acceptance_sha256=record["training_acceptance_sha256"], endpoint_sha256=record["endpoint_sha256"],
        fit_accepted_sha256=FIT, absolute_device="cuda:0", dtype="float32", tf32=False, images=counts["images"],
        mapped_rows=counts["mapped"], predictions_sha256=native_digest.hexdigest(), prediction_bytes=counts["native_bytes"],
        raw_hashes_sha256=raw_digest.hexdigest(), detector_forwards=counts["images"], wall_seconds=perf_counter()-start, **SCOPE)
    if comparisons:
        result.update(image_row_bytes=counts["row_bytes"], image_rows_sha256=row_digest.hexdigest(), phase_seconds=dict(times),
            same_device_posterior_corrections=len(corrections), correction_log_sha256=write(out / "CORRECTIONS_PRIVATE.jsonl",
            b"".join(encode(c) for c in corrections)))
    print(json.dumps(dict(status=result["status"], result_sha256=write(out / "RESULT_SAFE.json", encode(result)))),
          flush=True)


def owner(args):
    record = admission(args)
    out = Path(record["paths"]["output"])
    out.mkdir()
    ad_path = out / "ADMISSION.json"
    ad_sha = write(ad_path, encode(record))
    stamps, reports = {}, {}
    for stage, status in (("native", NATIVE_STATUS), ("refine", REFINE_STATUS)):
        stdout = out / (stage + ".stdout")
        with stdout.open("xb") as so, (out / (stage + ".stderr")).open("xb") as se:
            threads = "1" if stage == "refine" else "2"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                       OPENBLAS_NUM_THREADS=threads, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            argv = [record["paths"]["python"], "-I", "-B", "-X", "utf8", str(SELF),
                    stage, "--admission", str(ad_path), "--admission-sha256", ad_sha]
            original = subprocess.Popen(argv, stdout=so, stderr=se, env=env,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            code = original.wait()
        stamps[stage] = dict(pid=original.pid, exit=code, original_process_terminal=True)
        write(out / (stage + "_WAIT.json"), encode(stamps[stage]))
        need(code == 0, stage + " worker failed; see its stderr and original wait")
        footer = json.loads(stdout.read_bytes().splitlines()[-1])
        report = json.loads(checked(out / stage / "RESULT_SAFE.json", footer["result_sha256"]))
        need(footer["status"] == report["status"] == status and report["admission_sha256"] == ad_sha
             and report["images"] == report["detector_forwards"] == 705, "Full worker closure")
        stamps[stage]["result_sha256"] = footer["result_sha256"]
        reports[stage] = report
    a, b = reports["native"], reports["refine"]
    need(all(a[k] == b[k] for k in ("endpoint_sha256", "training_acceptance_sha256", "mapped_rows",
        "prediction_bytes", "predictions_sha256", "raw_hashes_sha256")), "Distinct forward streams disagree")
    terminal = dict(status="RFDETR_COMMON5203_DEV705_PRELABEL_TWO_STAGES_CLOSED_V1", phase=PHASE,
        original_exit=0, original_process_terminal=True, native_exit=stamps["native"]["exit"],
        refinement_exit=stamps["refine"]["exit"], native_result_sha256=stamps["native"]["result_sha256"],
        refinement_result_sha256=stamps["refine"]["result_sha256"], admission_sha256=ad_sha,
        training_acceptance_sha256=record["training_acceptance_sha256"], fit_accepted_sha256=FIT,
        endpoint_sha256=record["endpoint_sha256"], images=705, mapped_rows=a["mapped_rows"],
        raw_hashes_sha256=a["raw_hashes_sha256"], **SCOPE)
    print(json.dumps(dict(status=terminal["status"], terminal_sha256=write(out / "TERMINAL_SAFE.json", encode(terminal)))),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--training-acceptance-sha256", required=True)
    for name in ("native", "refine"):
        worker = sub.add_parser(name)
        worker.add_argument("--admission", type=Path, required=True)
        worker.add_argument("--admission-sha256", required=True)
    args = parser.parse_args()
    if args.action == "run":
        owner(args)
    else:
        scientific_worker(args.action, args.admission, args.admission_sha256)


if __name__ == "__main__":
    main()
