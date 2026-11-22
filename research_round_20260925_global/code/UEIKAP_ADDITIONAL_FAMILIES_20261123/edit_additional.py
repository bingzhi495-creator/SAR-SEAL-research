"""Replay the frozen UEIKAP RF-DETR policies on the admitted 342-image roster."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


PROTOCOL = "fbcc4d067351a7a7d2db226e6261a13c2f399eed59be414226e6ae29b6553bf2"
ROSTER = "0d78b4611e7db88a146f5bad489b6301e64e1d1eb499d536fd414e7b134c456d"
RF_DIR = "UEIKAP_RFDETR_CARRIER_20261122"
EDITOR_SHA = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
PREPARED = "UEIKAP_ADDITIONAL_RFDETR_EDITS_PREPARED"
ACCEPTED = "UEIKAP_ADDITIONAL_RFDETR_EDITS_VERIFIED"
IMAGES = 342


def require(value, message):
    if not value:
        raise ValueError(message)


def pinned(path, digest):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, "SHA-256 differs for " + path.name)
    return data


def source(path, expected):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def run(a, verify):
    import torch
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(),
            "Run the frozen editor on CPU")
    torch.set_num_threads(1)
    pinned(a.protocol, PROTOCOL)
    manifest = json.loads(pinned(a.manifest, a.manifest_sha))
    editor = source(a.code / RF_DIR / "edit_rf.py", EDITOR_SHA)
    event = editor.observed(a.capture_terminal, a.capture_terminal_sha, a.capture_receipt_sha, "verify-capture")
    receipt = json.loads(pinned(a.capture_receipt, a.capture_receipt_sha))
    capture = json.loads(pinned(a.capture / "RESULT_SAFE.json", receipt["capture_sha256"]))
    require(capture["status"] == "UEIKAP_ADDITIONAL_RFDETR_CAPTURE_COMPLETE" and
            receipt["status"] == "UEIKAP_ADDITIONAL_RFDETR_CAPTURE_VERIFIED" and
            event["worker_sha256"] == receipt["worker_sha256"] and receipt["model_forward_calls"] == 0 and
            receipt["checkpoint_sha256"] == editor.CHECKPOINT and receipt["manifest_sha256"] == a.manifest_sha and
            receipt["protocol_sha256"] == PROTOCOL and receipt["images"] == IMAGES and receipt["groups"] == 13,
            "Observed additional-family capture with the frozen checkpoint")
    require(all(capture[key] == receipt[key] for key in ("audit_sha256", "predictions_sha256", "counts",
                "manifest_sha256", "intake_terminal_sha256", "protocol_sha256", "checkpoint_sha256", "images", "groups"))
            and all(c["target_labels_opened"] is False and c["GT_opened"] is False and c["optimizer_updates"] == 0
                    for c in (capture, receipt)), "Capture and readback lineage differ")
    predictions = editor.load_stream(a.capture / "PREDICTIONS_PRIVATE.jsonl", receipt["predictions_sha256"])
    audits = editor.load_stream(a.capture / "CANDIDATE_AUDIT_PRIVATE.jsonl", receipt["audit_sha256"])
    require(len(predictions) == len(audits) == len(manifest["items"]) == IMAGES and
            manifest["expected_ids"] == list(range(1, IMAGES + 1)), "Complete image-only population")
    base = editor.load_module(a.code / editor.BASE / "replay.py", editor.REPLAY_SHA)
    methods = base.methods(a.code)
    predict = base.predictor(a.code, methods, a.fit_result, a.fit_accepted, a.weights)
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    previous, previous_sha = [], None
    if verify:
        event = editor.observed(a.prepared_terminal, a.prepared_terminal_sha, a.prepared_receipt_sha, "prepare-edits")
        old = json.loads(pinned(a.prepared_receipt, a.prepared_receipt_sha))
        require(old["status"] == PREPARED and old["target_labels_opened"] is False and
                old["worker_sha256"] == event["worker_sha256"] == worker_sha and
                old["capture_receipt_sha256"] == a.capture_receipt_sha and old["protocol_sha256"] == PROTOCOL,
                "Prepared edits must be closed on this capture using the same source")
        previous_sha = old["edits_sha256"]
        previous = editor.load_stream(a.prepared_receipt.parent / "EDITS_PRIVATE.jsonl", previous_sha)
        require(len(previous) == IMAGES, "Incomplete prepared stream")
    counts, hasher = Counter(), sha256()
    a.output.mkdir()
    from contextlib import nullcontext
    output_stream = nullcontext(None) if verify else (a.output / "EDITS_PRIVATE.jsonl").open("xb")
    with output_stream as stream:
        for index, (native, row, item) in enumerate(zip(predictions, audits, manifest["items"], strict=True), 1):
            ref = item["image_ref"]
            require(row["image_id"] == item["image_id"] == index and row["image_sha256"] == ref["sha256"]
                    and ref["path"] == "images/" + row["source_image_name"] == f"images/{index:06d}.png"
                    and item["original_hw"] == row["original_hw"] == [512, 512], "Image identity in capture manifest")
            result, totals = editor.image_edits(base, methods, predict, a, row, native, independent=verify)
            if verify:
                require(result == previous[index-1], "Independent native endpoint choices or mask cells differ")
            data = base.encoded(result)
            hasher.update(data)
            if stream:
                stream.write(data)
            counts.update(totals)
            if index % 50 == 0 or index == IMAGES:
                print(json.dumps({"images": index, "phase": "verify-edits" if verify else "prepare-edits"}), flush=True)
    edits_sha = hasher.hexdigest()
    require(not verify or previous_sha == edits_sha, "Prepared and independently replayed edits differ")
    counts.update(images=IMAGES, mapped_candidates=sum(len(row["coco"]) for row in predictions))
    require(counts["mapped_candidates"] == receipt["counts"]["mapped"] and
            counts["editable_supports"] <= receipt["counts"]["eligible"], "Mapped and editable counts")
    return dict(status=ACCEPTED if verify else PREPARED, images=IMAGES, groups=13,
                target_labels_opened=False, GT_opened=False, optimizer_updates=0, model_forwards=0,
                exact_support_xor_area=True, exact_replayed_policy_choices=verify,
                protocol_sha256=PROTOCOL, cohort_roster_sha256=ROSTER, manifest_sha256=a.manifest_sha,
                predictions_sha256=receipt["predictions_sha256"], candidate_audit_sha256=receipt["audit_sha256"],
                capture_receipt_sha256=a.capture_receipt_sha, capture_terminal_sha256=a.capture_terminal_sha,
                intake_terminal_sha256=receipt["intake_terminal_sha256"], edits_sha256=edits_sha,
                prepared_receipt_sha256=a.prepared_receipt_sha if verify else None,
                historical_editor_sha256=EDITOR_SHA, worker_sha256=worker_sha, counts=dict(counts))


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    for key in ("authority", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--authority-sha", required=True)
    args = p.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and not args.output.exists(), "Isolated fresh phase")
    values = json.loads(pinned(args.authority, args.authority_sha))
    paths = {"protocol", "manifest", "images", "code", "capture", "capture_terminal", "capture_receipt",
             "fit_result", "fit_accepted", "weights"}
    stamps = {"capture_terminal_sha", "capture_receipt_sha", "manifest_sha"}
    verify = args.phase == "verify-edits"
    if verify:
        paths |= {"prepared_terminal", "prepared_receipt"}
        stamps |= {"prepared_terminal_sha", "prepared_receipt_sha"}
    require(set(values) == paths | stamps, "Unexpected phase authority keys")
    a = SimpleNamespace(output=args.output, **{k: Path(v) if k in paths else v for k, v in values.items()})
    start = perf_counter()
    result = run(a, verify)
    result.update(seconds=perf_counter() - start, input_authority_sha256=args.authority_sha)
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (args.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(data)
    print(json.dumps({"status": result["status"], "result_sha256": sha256(data).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
