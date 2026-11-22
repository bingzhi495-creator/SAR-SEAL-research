"""Reproduce the original 740-image Instance-RSDD frozen transfer table, using explicit assets."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace


NATIVE = "INSTANCE_RSDD_FROZEN_HRSID_20261122"
PORTABLE = "INSTANCE_RSDD_B5_REPRO_20261123"
COMMON_SHA = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
MANIFEST_SHA = "16a546e96d6e76d3afbfbe4f380b182505494818b35f1c8bcc960e6d4d1b2184"
SCORE_SHA = "7039d7d0336f6a0cc3b64bfbe4b0c446b38b073cc90764f446a4dc6d09f90d85"
EDIT_OPERATOR = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
HEADS = {5217: "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1",
         5218: "08b39b9911f77c039b26f8bb9bc34e940f5a350390911586f47290b2c3d88869"}
CAPTURED = "B5_RSDD740_CAPTURE_COMPLETE"
REPLAYED = "B5_RSDD740_REPLAY_COMPLETE"
RESULT = "RESULT_SAFE.json"


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def read_module(path, digest):
    source = path.read_bytes()
    require(sha256(source).hexdigest() == digest, f"Source identity differs for {path.name}")
    value = ModuleType(path.stem)
    value.__file__ = str(path)
    exec(compile(source, str(path), "exec"), value.__dict__)
    return value


def native_modules(code):
    shared = read_module(code / PORTABLE / "portable_eval.py", COMMON_SHA)
    native = shared.module(code / NATIVE / "capture_instance_rsdd.py", shared.CAPTURE_SOURCE)
    return shared, native


def write_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
    return sha256(payload).hexdigest()


def load_items(a, shared, native):
    manifest = json.loads(shared.pin(a.intake / shared.MANIFEST, MANIFEST_SHA))
    rows = [dict(id=r["image_id"], file_name=r["source_image_name"], height=r["original_hw"][0],
                 width=r["original_hw"][1]) for r in manifest["items"]]
    items = native.roster({"images": rows}, manifest)
    require(Counter(x["sensor"] for x in items) == {"GF-3": 583, "TerraSAR-X": 157}
            and len({x["acquisition_group"] for x in items}) == 112, "Original TEST roster differs")
    return items


def capture_info(a, shared, items):
    raw = (a.capture / RESULT).read_bytes()
    report = json.loads(raw)
    require(report["status"] == CAPTURED and report["source_partition"] == "author test"
            and report["checkpoint_sha256"] == shared.CHECKPOINT and report["manifest_sha256"] == MANIFEST_SHA
            and report["model_forward_calls"] == len(items) and report["target_labels_opened"] is False,
            "Complete new portable 740-image capture required")
    return report, sha256(raw).hexdigest()


def records(path, expected):
    digest = sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            yield json.loads(line)
    require(digest.hexdigest() == expected, f"Newly generated stream differs for {path.name}")


def capture_rows(a, shared, items, report):
    predictions = records(a.capture / shared.PREDICTIONS, report["predictions_sha256"])
    audits = records(a.capture / shared.AUDIT, report["audit_sha256"])
    for item, row, pred in zip(items, audits, predictions, strict=True):
        require(item["image_id"] == row["image_id"] == pred["image_id"] and row["image_sha256"] == item["image_ref"]["sha256"]
                and all(row[k] == item[k] for k in ("source_image_name", "rgb_sha256", "sensor", "acquisition_group", "original_hw")),
                "Fresh prediction, candidate and author image identities differ")
        yield row, pred


def initialize_replay(a, items):
    global worker
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "-1" and not torch.cuda.is_initialized(), "CPU-only frozen replay")
    shared, native = native_modules(a.code)
    native.install_guard(a.intake, items)
    edit = shared.module(a.code / shared.UEIKAP / "edit_rf.py", EDIT_OPERATOR)
    base = edit.load_module(a.code / edit.BASE / "replay.py", edit.REPLAY_SHA)
    methods = base.methods(a.code)
    step = base.read_source(a.code / base.SPOT / "fit_step.py", methods.readout.PINS["fit_step.py"])
    plan = base.read_source(a.code / base.RF / "fit_plan.py", methods.readout.PINS["fit_plan.py"])
    trainer = step.load_trainer(a.code / base.BASE / "fit109_paired_point/train_point.py")
    heads, center, scale = plan.restore_bundle(a.point_head, HEADS[a.seed], trainer, step)
    predictor = lambda features: step.predict(heads["S"], features, center, scale)
    worker = (edit, base, methods, predictor, SimpleNamespace(images=a.intake / "images", capture=a.capture))


def replay_image(pair):
    edit, base, methods, predictor, args = worker
    row, pred = pair
    first = edit.image_edits(base, methods, predictor, args, row, pred, independent=False)
    second = edit.image_edits(base, methods, predictor, args, row, pred, independent=True)
    require(first == second, "Independent endpoint choices, areas or edit counts differ")
    return base.encoded(first[0]), first[1], len(pred["coco"])


def replay(a, shared, native, items):
    native.install_guard(a.intake, items)
    report, capture_sha = capture_info(a, shared, items)
    for item in items:
        native.decode_native(a.intake, item)
    a.output.mkdir()
    counts, digest = Counter(), sha256()
    pairs = capture_rows(a, shared, items, report)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize_replay, initargs=(a, items)) as pool:
        with (a.output / shared.EDITS).open("xb") as stream:
            for data, row_counts, mapped in pool.map(replay_image, pairs, chunksize=4):
                stream.write(data)
                digest.update(data)
                counts.update(row_counts)
                counts.update(images=1, mapped_candidates=mapped)
    require(counts["images"] == len(items) and counts["mapped_candidates"] == report["counts"]["mapped"],
            "Incomplete replay")
    return dict(status=REPLAYED, counts=dict(counts), edits_sha256=digest.hexdigest(),
                capture_sha256=capture_sha, predictions_sha256=report["predictions_sha256"], audit_sha256=report["audit_sha256"],
                exact_replayed_policy_choices=True, exact_support_xor_area=True, model_forward_calls=0, target_labels_opened=False)


def new_blocks(a, shared, items, cap, replayed, adapter, core):
    pairs = capture_rows(a, shared, items, cap)
    edits = records(a.replay / shared.EDITS, replayed["edits_sha256"])
    blocks = {}
    for (audit, pred), update in zip(pairs, edits, strict=True):
        image_id = pred["image_id"]
        require(update["image_id"] == image_id, "Fresh edit stream image order differs")
        masks = adapter.native_masks(core, pred, audit, image_id)
        checked = core.verify_edits(update["changes"], masks, *audit["original_hw"])
        blocks[image_id] = dict(stock=pred["coco"], ranks=[x["position"] for x in audit["candidates"]],
                                rgb=audit["rgb_sha256"], sensor=audit["sensor"], group=audit["acquisition_group"],
                                file=audit["source_image_name"],
                                changes={k: {arm: dict(swaps=n, rle=rle) for arm, (n, rle) in options.items()}
                                         for k, options in checked.items()})
    return blocks


def score(a, shared, items):
    original = shared.module(a.code / NATIVE / "score_rsdd.py", SCORE_SHA)
    cap, cap_sha = capture_info(a, shared, items)
    payload = (a.replay / RESULT).read_bytes()
    replayed = json.loads(payload)
    require(replayed["status"] == REPLAYED and replayed["capture_sha256"] == cap_sha
            and replayed["point_head_seed"] == a.seed and replayed["point_head_sha256"] == HEADS[a.seed]
            and replayed["target_labels_opened"] is False and replayed["exact_replayed_policy_choices"] is True
            and replayed["exact_support_xor_area"] is True
            and all(replayed[k] == cap[k] for k in ("manifest_sha256", "predictions_sha256", "audit_sha256", "images",
                                                     "checkpoint_sha256", "source_partition")),
            "Seed, image cohort or fresh capture/replay lineage differs")
    # Replace receipt discovery only. Original run owns annotation loading and ID adaptation,
    # every arm and view, original-mask anchor matching, geometry, bootstrap and stopping rule.
    original.admitted_inputs = lambda _, adapter, core: new_blocks(a, shared, items, cap, replayed, adapter, core)
    output = original.run(SimpleNamespace(code=a.code, author_test=a.author_test, cap_receipt_sha=cap_sha,
                                         edit_receipt_sha=sha256(payload).hexdigest()))
    output.update(status="B5_RSDD740_SCORE_COMPLETE", capture_sha256=output.pop("capture_receipt_sha256"),
                  replay_sha256=output.pop("edit_receipt_sha256"), edits_sha256=replayed["edits_sha256"],
                  predictions_sha256=cap["predictions_sha256"], audit_sha256=cap["audit_sha256"])
    a.output.mkdir()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in ("capture", "replay", "score"):
        cmd = commands.add_parser(phase, allow_abbrev=False)
        cmd.add_argument("--code", type=Path, default=Path(__file__).resolve().parent.parent)
        cmd.add_argument("--output", required=True, type=Path, help="Fresh phase output directory")
        cmd.add_argument("--intake", required=True, type=Path, help="Directory with original manifest and images/")
        if phase == "capture":
            cmd.add_argument("--checkpoint", required=True, type=Path)
        if phase in ("score", "replay"):
            cmd.add_argument("--capture", required=True, type=Path)
            cmd.add_argument("--seed", required=True, choices=tuple(HEADS), type=int)
        if phase == "replay":
            cmd.add_argument("--point-head", required=True, type=Path)
            cmd.add_argument("--workers", type=int, choices=range(1, 9), default=4)
        if phase == "score":
            cmd.add_argument("--replay", required=True, type=Path)
            cmd.add_argument("--author-test", required=True, type=Path)
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key != "output"))
    require(not a.output.exists(), "Supply a fresh output directory")
    if a.phase != "capture":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    expected = {"numpy": "1.26.4", "pycocotools": "2.0.11"}
    if a.phase == "score":
        expected["shapely"] = "2.0.2"
    else:
        expected.update(torch="2.10.0+cu130", rfdetr="1.12.0", Pillow="12.3.0")
    runtime = {k: version(k) for k in expected}
    require(runtime == expected, f"Expected {expected}, found {runtime}")
    shared, native = native_modules(a.code)
    items = load_items(a, shared, native)
    if a.phase == "capture":
        result = shared.capture(a, items, MANIFEST_SHA, native)
        result["status"] = CAPTURED
    elif a.phase == "replay":
        result = replay(a, shared, native, items)
    else:
        result = score(a, shared, items)
    result.update(phase=a.phase, images=740, manifest_sha256=MANIFEST_SHA, source_partition="author test",
                  operator_versions=runtime, optimizer_updates=0, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    result["checkpoint_sha256"] = shared.CHECKPOINT
    if a.phase in ("replay", "score"):
        result.update(point_head_seed=a.seed, point_head_sha256=HEADS[a.seed])
    result_sha = write_json(a.output / RESULT, result)
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    main()

