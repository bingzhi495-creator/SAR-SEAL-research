"""Replay frozen boundary edits on the full author Instance-RSDD TEST release."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


IMAGE_COUNT = 740
ROSTER_SHA = "72c4b9af8f8b0bd0df31a3a3f7df6a082390688f0942eed23f106316a2d5d98b"
PROTOCOL_SHA = "5c0c9410db15710e34fe453271bd3a99a1f1a50eb365bd354f884e40b88c8b85"
PRELOCK_COMMIT = "6665e15ebb2c064661957b51d0ee7334bae1bb6c"
EDIT_SHA = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
CAPTURED = "INSTANCE_RSDD_TEST740_RFDETR_CAPTURE_COMPLETE_V1"
CAPTURE_VERIFIED = "INSTANCE_RSDD_TEST740_RFDETR_READBACK_COMPLETE_V1"
PREPARED = "INSTANCE_RSDD_TEST740_RFDETR_EDIT_PREPARED_V1"
ACCEPTED = "INSTANCE_RSDD_TEST740_RFDETR_EDIT_REPLAY_ACCEPTED_V1"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    data = path.read_bytes()
    check(sha256(data).hexdigest() == expected, "Pinned source or input: " + path.name)
    return data


def source_module(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def author_rows(roster, predictions, audit, count=IMAGE_COUNT):
    leaves = roster["images"]
    check(set(roster) == {"images"} and len(leaves) == len(predictions) == len(audit) == count
          and len({item["id"] for item in leaves}) == len({item["file_name"] for item in leaves}) == count,
          "Complete author TEST roster")
    for image, prediction, row in zip(leaves, predictions, audit, strict=True):
        name = image["file_name"]
        check(Path(name).name == name and name.lower().endswith(".jpg") and type(image["id"]) is int
              and image["id"] == row["image_id"] == prediction["image_id"]
              and [image["height"], image["width"]] == row["original_hw"] == [512,512]
              and name == row["source_image_name"], "Author image identity and order")
        check(all(candidate["image_id"] == image["id"] and candidate["category_id"] == 1
                  and candidate["segmentation"]["size"] == row["original_hw"] for candidate in prediction["coco"]),
              "Original mapped ship candidate identity")
    return list(zip(audit, predictions, strict=True))


def read_capture(a, edit):
    terminal = edit.observed(a.capture_terminal, a.capture_terminal_sha, a.capture_receipt_sha, "verify-capture")
    receipt = json.loads(pin(a.capture_receipt, a.capture_receipt_sha))
    original = json.loads(pin(a.capture / "RESULT_SAFE.json", receipt["capture_sha256"]))
    check(receipt["status"] == CAPTURE_VERIFIED and original["status"] == CAPTURED
          and receipt["checkpoint_sha256"] == original["checkpoint_sha256"] == CHECKPOINT
          and receipt["protocol_sha256"] == original["protocol_sha256"] == PROTOCOL_SHA
          and receipt["prelock_commit"] == original["prelock_commit"] == PRELOCK_COMMIT
          and receipt["author_roster_sha256"] == original["author_roster_sha256"] == ROSTER_SHA
          and receipt["exact_native_parity"] is True and receipt["GT_opened"] is original["GT_opened"] is False
          and receipt["author_TEST"] is original["author_TEST"] is True
          and terminal["worker_sha256"] == receipt["worker_sha256"]
          and all(receipt[key] == original[key] for key in ("predictions_sha256", "audit_sha256", "intake_sha256", "counts")),
          "Observed frozen TEST capture, native parity and input seals")
    roster = json.loads(pin(a.roster, ROSTER_SHA))
    predictions = edit.load_stream(a.capture / "PREDICTIONS_PRIVATE.jsonl", receipt["predictions_sha256"])
    audit = edit.load_stream(a.capture / "CANDIDATE_AUDIT_PRIVATE.jsonl", receipt["audit_sha256"])
    return author_rows(roster, predictions, audit), receipt


def initialize(a, independent):
    global worker
    import torch
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(), "CPU-only replay")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    edit = source_module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py", EDIT_SHA)
    base = edit.load_module(a.code / edit.BASE / "replay.py", edit.REPLAY_SHA)
    methods = base.methods(a.code)
    predictor = base.predictor(a.code, methods, a.fit_result, a.fit_accepted, a.weights)
    worker = (edit, base, methods, predictor, a, independent)


def replay_one(pair):
    row, stock = pair
    edit, base, methods, predictor, a, independent = worker
    return edit.image_edits(base, methods, predictor, a, row, stock, independent=independent)


def run(a, independent):
    pin(a.protocol, PROTOCOL_SHA)
    edit = source_module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py", EDIT_SHA)
    base = edit.load_module(a.code / edit.BASE / "replay.py", edit.REPLAY_SHA)
    rows, source = read_capture(a, edit)
    source_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    if independent:
        terminal = edit.observed(a.prepared_terminal, a.prepared_terminal_sha, a.prepared_receipt_sha, "prepare-edits")
        old = json.loads(pin(a.prepared_receipt, a.prepared_receipt_sha))
        check(old["status"] == PREPARED and old["GT_opened"] is False and old["author_TEST"] is True
              and old["protocol_sha256"] == PROTOCOL_SHA and old["capture_receipt_sha256"] == a.capture_receipt_sha
              and terminal["worker_sha256"] == source_sha,
              "Prior observed preparation on the same frozen input")
        previous = edit.load_stream(a.prepared_receipt.parent / "EDITS_PRIVATE.jsonl", old["edits_sha256"])
        check(len(previous) == len(rows), "Complete prepared author roster")

    counts = Counter(images=IMAGE_COUNT, mapped_candidates=sum(len(pred["coco"]) for _, pred in rows))
    serialized = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a, independent)) as pool:
        for index, (row, row_counts) in enumerate(pool.map(replay_one, rows, chunksize=4)):
            check(row["image_id"] == rows[index][0]["image_id"], "Exact replay image position")
            if independent:
                check(row == previous[index], "Identical independent endpoint choices and mask edits")
            counts.update(row_counts)
            serialized.append(base.encoded(row))
    payload = b"".join(serialized)
    payload_sha = sha256(payload).hexdigest()
    if independent:
        check(payload_sha == old["edits_sha256"] and dict(counts) == old["counts"], "Exact observed replay readback")
    return payload, dict(status=ACCEPTED if independent else PREPARED, GT_opened=False, author_TEST=True,
                         model_forwards=0, optimizer_updates=0, protocol_sha256=PROTOCOL_SHA,
                         manifest_sha256=ROSTER_SHA, intake_sha256=source["intake_sha256"],
                         predictions_sha256=source["predictions_sha256"], candidate_audit_sha256=source["audit_sha256"],
                         capture_receipt_sha256=a.capture_receipt_sha, edits_sha256=payload_sha,
                         exact_replayed_policy_choices=independent, exact_support_xor_area=True,
                         counts=dict(counts), source_sha256=source_sha, workers=a.workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    values = json.loads(pin(args.authority, args.authority_sha))
    paths = {"protocol", "roster", "images", "code", "capture", "capture_terminal", "capture_receipt",
             "fit_result", "fit_accepted", "weights"}
    stamps = {"capture_terminal_sha", "capture_receipt_sha", "workers"}
    independent = args.phase == "verify-edits"
    if independent:
        paths |= {"prepared_terminal", "prepared_receipt"}
        stamps |= {"prepared_terminal_sha", "prepared_receipt_sha"}
    check(set(values) == paths | stamps and not args.output.exists() and sys.flags.isolated and sys.dont_write_bytecode
          and type(values["workers"]) is int and 1 <= values["workers"] <= 8,
          "Pinned phase inputs and bounded CPU workers for a fresh output")
    a = SimpleNamespace(**{key: Path(value) if key in paths else value for key, value in values.items()})
    start = perf_counter()
    payload, result = run(a, independent)
    result.update(seconds=perf_counter()-start, input_authority_sha256=args.authority_sha)
    args.output.mkdir()
    if not independent:
        (args.output / "EDITS_PRIVATE.jsonl").write_bytes(payload)
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    (args.output / "RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
