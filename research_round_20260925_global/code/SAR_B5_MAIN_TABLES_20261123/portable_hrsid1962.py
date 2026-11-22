"""Reproduce the official 1,962-image, three-head RF-DETR table from frozen assets."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile
import zlib


HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("hrsid_prepare", str(HERE / "hrsid_prepare.py"))
inputs = module_from_spec(spec)
spec.loader.exec_module(inputs)
check = inputs.check
HEADS = {
    5217: "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1",
    5218: "08b39b9911f77c039b26f8bb9bc34e940f5a350390911586f47290b2c3d88869",
    5219: "93a3be4356816f9d29560d40d9bff8cc5d8730e9e600b29db07959612f504604",
}
SEEDS = tuple(HEADS)
ARMS = ("SF", "SU=F", "SU", "LW=S")
NATIVE_SHA = "9c16ecb5275ae1ba23926a46aa74a5e10c979a3761838d7ff279859986a742b4"
CAPDIR = "HRSID_OFFICIAL_FROZEN_REFINEMENT_CAPTURE_READBACK_20261121"
CAP = "a50da3d740d9620a23c98b978b7cfaa383040f1e7542e2eb5da32ca4b0106da6"
READER = "5503e284690b6e0d36c09b6e427ea0cd9ca6acbc26e0ce1a20e8e84e3f58cc30"
PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
FILE = "RESULT_SAFE.json"
CTX = None


def initialize(args):
    global CTX
    import numpy as np
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    check(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130" and
          np.__version__ == "1.26.4" and version("rfdetr") == "1.12.0" and version("Pillow") == "12.3.0"
          and version("pycocotools") == "2.0.11" and not torch.cuda.is_initialized(), "Original CPU inference runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    portable = inputs.load(args.code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", "_detector", PORTABLE)
    carrier, core, base, _ = portable.detector_components(args.code, args.checkpoint)
    model, transform, config = carrier.model_and_transform(SimpleNamespace(mode="test", output=args.output), core, base, torch)
    check(not model.training and all(v.device.type == "cpu" for v in model.parameters()), "Original CPU eval model")
    capture = inputs.load(args.code / CAPDIR / "capture.py", "_policy_capture", CAP)
    readback = inputs.load(args.code / CAPDIR / "readback.py", "_independent_replay", READER)
    policy = capture.load_policy(args.code)
    heads = {s: policy.plan.restore_bundle(args.heads / (str(s) + "_all_fit.npz"), h, policy.trainer, policy.fit_step)
             for s, h in HEADS.items()}
    CTX = SimpleNamespace(np=np, torch=torch, codec=codec, post=PostProcess, decoder=decode_pil_image_bytes,
                          nest=nested_tensor_from_tensor_list, model=model, transform=transform, config=config,
                          base=base, cap=capture, readback=readback, policy=policy, heads=heads,
                          archive=ZipFile(args.archive), output=args.output)


def image_capture(task):
    c = CTX
    ordinal, row = task
    image_id = row["image_id"]
    png = c.archive.read(row["member"])
    check(sha256(png).hexdigest() == row["encoded_sha256"], "Official PNG fingerprint")
    image, scales = c.decoder(png, draft_size=None)
    rgb = c.np.asarray(image)
    check(image.mode == "RGB" and image.size == (800, 800) and scales == (1., 1.) and
          sha256(rgb.tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"], "Official RGB fingerprint")
    with c.torch.inference_mode():
        tensor, target = c.transform(image, None)
        check(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == c.torch.float32, "Original val576")
        sample = c.nest([tensor], block_size=c.config.patch_size * c.config.num_windows)
        check(sample.mask is not None and not bool(sample.mask.any()), "Unpadded CPU sample")
        raw = c.model(sample)
        post = c.post(num_select=100, upsample_masks_to_image_size=True)
        native = c.base.native_rows(raw, image_id=image_id, hw=(800, 800), torch=c.torch,
                                    np=c.np, codec=c.codec, postprocess=post)
        line = c.cap.encode_native(native)
        # Re-run original native mapping against this fresh stream, using the same single forward.
        predictions, witness, _ = c.cap.compute_image(image_id=image_id, rgb=rgb, expected_native_line=line,
            forward=lambda: raw, native_rows=c.base.native_rows, postprocess_class=c.post,
            policy=c.policy, refits=c.heads)
    selected = {str(s):{a:predictions[s][a] for a in ARMS} for s in SEEDS}
    envelope = dict(index_zero_based=ordinal, image_id=image_id, stock_block_sha256=sha256(line).hexdigest(),
                    predictions=selected, witness=witness)
    tally = Counter()
    c.readback.validate_image(envelope, ordinal, line, tally)
    features = [[image_id, v["rank"], v["query_id"], v["feature_sha256"]] for v in witness["candidates"]
                if "feature_sha256" in v]
    body = dict(index=ordinal, image_id=image_id, native=native, treatments=selected, counts=dict(tally), features=features)
    final = shard(c.output, ordinal)
    tmp = final.with_suffix(".tmp")
    tmp.write_bytes(zlib.compress(c.cap.encode_native(body), 1))
    tmp.replace(final)
    return ordinal


def shard(root, ordinal):
    return root / "IMAGES_PRIVATE" / ("%05d.json.zlib" % ordinal)


def encode(item):
    return (json.dumps(item, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def combine(args, rows):
    names = ["N"] + [str(s) + "_" + a for s in SEEDS for a in ARMS]
    hashes = {k:sha256() for k in names}
    tally, ledger = Counter(), sha256()
    # Shards are retained across interrupted runs. Rebuild the final streams in order.
    with ExitStack() as stack:
        handles = {k:stack.enter_context((args.output / (k + ".jsonl")).open("wb")) for k in names}
        for i, row in enumerate(rows):
            obj = json.loads(zlib.decompress(shard(args.output, i).read_bytes()))
            check(obj["index"] == i and obj["image_id"] == row["image_id"], "Shard identity or order differs")
            blocks = {"N":obj["native"]}
            for s, arms in obj["treatments"].items():
                blocks.update({s + "_" + a:b for a, b in arms.items()})
            check(set(blocks) == set(names), "Missing treatment")
            for k, block in blocks.items():
                raw = encode(block)
                handles[k].write(raw)
                hashes[k].update(raw)
            for descriptor in obj["features"]:
                ledger.update(encode(descriptor))
            tally.update(obj["counts"])
    check(tally["images"] == len(rows) == 1962 and tally["mapped"] == 187796, "Official native census")
    refs = {}
    for name in names:
        path = args.output / (name + ".jsonl")
        digest = inputs.file_digest(path)
        check(digest == hashes[name].hexdigest(), "Stream readback differs")
        refs[name] = dict(sha256=digest, bytes=path.stat().st_size)
    check(refs["N"]["sha256"] == NATIVE_SHA, "Complete ordered fresh native stream differs from the historical stream")
    return dict(status="B5_HRSID1962_CAPTURE_COMPLETE", images=len(rows), mapped_rows=tally["mapped"],
                model_forward_calls=len(rows), optimizer_updates=0, AP=False, GT_opened=False,
                endpoint_sha256=inputs.ENDPOINT, point_heads_sha256=HEADS, manifest_sha256=inputs.MANIFEST_SHA,
                native_predictions_sha256=NATIVE_SHA, native_prediction_bytes=refs["N"]["bytes"],
                streams={str(s):{a:refs[str(s) + "_" + a] for a in ARMS} for s in SEEDS}, counts=dict(tally),
                exact_candidate_identity=True, exact_native_per_image=True, exact_per_candidate_budgets=True,
                actual_XOR_area_RLE_verified=True, point_score_decisions_replayed=True,
                descriptor_identity_ledger_sha256=ledger.hexdigest(), device="cpu", threads_per_worker=1)


def capture(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    raw = (args.intake / inputs.MANIFEST).read_bytes()
    check(sha256(raw).hexdigest() == inputs.MANIFEST_SHA, "Original ordered image manifest")
    rows = json.loads(raw)
    check(args.workers >= 1, "Positive worker count")
    lineage = dict(archive_sha256=inputs.ARCHIVE_SHA, checkpoint_sha256=inputs.ENDPOINT,
                   heads_sha256={str(s):h for s, h in HEADS.items()}, manifest_sha256=inputs.MANIFEST_SHA,
                   worker_sha256=inputs.file_digest(Path(__file__)), portable_detector_source=PORTABLE,
                   treatment_source=CAP, independent_reader_source=READER)
    if args.resume:
        check(json.loads((args.output / "RUN.json").read_bytes()) == lineage and
              not (args.output / FILE).exists(), "Resume only the same unfinished capture")
    else:
        args.output.mkdir()
        (args.output / "IMAGES_PRIVATE").mkdir()
        (args.output / "RUN.json").write_bytes(inputs.encode(lineage))
    missing = [(i, row) for i, row in enumerate(rows) if not shard(args.output, i).is_file()]
    before = len(rows) - len(missing)
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args,)) as pool:
        for done, i in enumerate(pool.map(image_capture, missing), before + 1):
            if done <= before + 1 or done % 25 == 0:
                print(json.dumps(dict(images=done, last_index=i, seconds=round(perf_counter() - started, 2))), flush=True)
    result = combine(args, rows)
    result.update(fresh_images_this_invocation=len(missing), completed_images_from_same_run=before,
                  worker_sha256=lineage["worker_sha256"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("prepare", "capture", "score"):
        p = phases.add_parser(phase)
        p.add_argument("--archive", required=True, type=Path)
        p.add_argument("--code", type=Path, default=HERE.parent)
        p.add_argument("--output", required=True, type=Path)
        if phase != "prepare":
            p.add_argument("--intake", required=True, type=Path)
        if phase == "capture":
            p.add_argument("--checkpoint", required=True, type=Path)
            p.add_argument("--heads", required=True, type=Path)
            p.add_argument("--workers", type=int, default=1, help="Independent one-thread CPU models")
            p.add_argument("--resume", action="store_true", help="Continue the same unfinished capture after stopping its workers")
        if phase == "score":
            p.add_argument("--capture", required=True, type=Path)
    args = parser.parse_args()
    for key, v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, key, v.resolve(strict=key != "output"))
    if args.phase == "prepare":
        inputs.prepare(args)
        return
    if args.phase == "capture":
        result = capture(args)
    else:
        score = inputs.load(HERE / "hrsid_score.py", "_score1962")
        result = score.run(args, inputs)
    path = args.output / FILE
    with path.open("xb") as stream:
        stream.write(inputs.encode(result))
    print(json.dumps(dict(status=result["status"], result_sha256=inputs.file_digest(path))), flush=True)


if __name__ == "__main__":
    main()

