"""Closed label-free replay for the prespecified Instance-RSDD parent reserve."""
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


LOCK = "3cc00d35d65318c3ae15c041c0f0b71f0c0dc743e0286c70955b9c3740009727"
COMMIT = "8ea9023ffb48abbd71e683d223a44087f3021d5b"
EDIT = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
OLD_REPLAY = "c5d96adec0fdc03aa51b2b098a9928391382d0a87ccf6e4ca2d3a04d57e229f5"
CAPTURE = "INSTANCE_RSDD_RESERVE74_RFDETR_READBACK_COMPLETE_V1"
PREPARED = "INSTANCE_RSDD_RESERVE74_EDIT_PREPARED_V1"
ACCEPTED = "INSTANCE_RSDD_RESERVE74_EDIT_ACCEPTED_V1"
ARMS = ("fixed","free_matched_count","free_natural")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, expected):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected, "Exact digest: " + path.name)
    return data


def module(path, expected):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pin(path,expected),str(path),"exec"),obj.__dict__)
    return obj


def closed(terminal, terminal_sha, summary, summary_sha, phase):
    receipt = json.loads(pin(terminal,terminal_sha))
    need(receipt["phase"] == phase and receipt["actual_exit"] == 0 and receipt["output_footer_verified"] is True
         and receipt["result_sha256"] == summary_sha, "Observed exact closed " + phase)
    return json.loads(pin(summary,summary_sha)), receipt


def read_capture(a):
    capture = a.capture_root
    report, receipt = closed(capture/"observed_verify-capture/TERMINAL.json",a.capture_terminal_sha,
                             capture/"verify-capture/RESULT_SAFE.json",a.capture_receipt_sha,"verify-capture")
    need(report["status"] == CAPTURE and report["GT_opened"] is False and report["protocol_sha256"] == LOCK
         and report["prelock_commit"] == COMMIT and report["intake_sha256"] == a.manifest_sha
         and report["worker_sha256"] == receipt["worker_sha256"], "New frozen reserve capture lineage")
    audit = [json.loads(x) for x in pin(capture/"capture/CANDIDATE_AUDIT_PRIVATE.jsonl",report["audit_sha256"]).splitlines()]
    stock = [json.loads(x) for x in pin(capture/"capture/PREDICTIONS_PRIVATE.jsonl",report["predictions_sha256"]).splitlines()]
    manifest = json.loads(pin(a.manifest,a.manifest_sha))["items"]
    need(len(audit) == len(stock) == len(manifest) == 74, "Entire agreed parent-scene image population")
    for row,pred,image in zip(audit,stock,manifest,strict=True):
        need(row["image_id"] == pred["image_id"] == image["image_id"]
             and row["source_image_name"] == image["source_image_name"]
             and row["rgb_sha256"] == image["rgb_sha256"]
             and row["sensor"] == image["sensor"] and row["original_hw"] == [512,512]
             and all(x["image_id"] == row["image_id"] for x in pred["coco"]), "Exact native candidate identity")
    return report, list(zip(audit,stock,strict=True))


def initialize(a, independent):
    global worker
    old = module(a.code/"INSTANCE_RSDD_EXPANDED_RANKING_20261123/replay_expanded.py",OLD_REPLAY)
    old.label_guard(a.images)
    import torch
    need(not torch.cuda.is_initialized(), "Frozen point replay is CPU-only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    edit = module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py",EDIT)
    base = edit.load_module(a.code/edit.BASE/"replay.py",edit.REPLAY_SHA)
    methods = base.methods(a.code)
    predict = base.predictor(a.code,methods,a.fit_result,a.fit_accepted,a.weights)
    req = SimpleNamespace(images=a.images,capture=a.capture_root/"capture")
    worker = (edit,base,methods,predict,req,independent)


def replay(pair):
    row, pred = pair
    edit,base,methods,predict,req,independent = worker
    result, census = edit.image_edits(base,methods,predict,req,row,pred,independent=independent)
    changes = {}
    counts = Counter(images=1,mapped_candidates=len(pred["coco"]),editable_supports=census["editable_supports"])
    for ordinal, candidates in result["changes"].items():
        picked = {arm:candidates[arm] for arm in ARMS if arm in candidates}
        need(picked.get("fixed",{}).get("swaps",0) == picked.get("free_matched_count",{}).get("swaps",0),
             "Archived same-head pairing and balanced independent budgets")
        if picked:
            changes[ordinal] = picked
            for arm,action in picked.items():
                counts[arm+"_swaps"] += action["swaps"]
                counts[arm+"_candidates"] += 1
    return {"image_id":result["image_id"],"changes":changes}, counts


def run(a, out, phase, authority_sha):
    report, items = read_capture(a)
    review = phase == "verify-edits"
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    previous = None
    if review:
        done, monitor = closed(a.prepare/"observed_prepare-edits/TERMINAL.json",a.prepare_terminal_sha,
                               a.prepare/"prepare-edits/RESULT_SAFE.json",a.prepare_receipt_sha,"prepare-edits")
        need(done["status"] == PREPARED and done["GT_opened"] is False
             and done["capture_receipt_sha256"] == a.capture_receipt_sha
             and done["protocol_sha256"] == LOCK and monitor["worker_sha256"] == worker_sha, "First reserve replay")
        previous = pin(a.prepare/"prepare-edits/EDITS_PRIVATE.jsonl",done["edits_sha256"]).splitlines(keepends=True)
        need(len(previous) == len(items), "Full prior ledger")
    start = perf_counter()
    pieces, totals = [], Counter()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a,review)) as pool:
        for index,(item,num) in enumerate(pool.map(replay,items,chunksize=2)):
            need(item["image_id"] == items[index][0]["image_id"], "Original stable order")
            raw = (json.dumps(item,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
            need(previous is None or raw == previous[index], "Independent edits agree byte by byte")
            pieces.append(raw)
            totals.update(num)
    ledger = b"".join(pieces)
    digest = sha256(ledger).hexdigest()
    need(totals["images"] == 74 and totals["mapped_candidates"] == report["counts"]["mapped"]
         and (previous is None or (digest == done["edits_sha256"] and totals == done["counts"])),
         "Exact original candidate census and accepted rerun")
    out.mkdir()
    if not review:
        (out/"EDITS_PRIVATE.jsonl").write_bytes(ledger)
    result = dict(status=ACCEPTED if review else PREPARED, protocol_sha256=LOCK,prelock_commit=COMMIT,
                  predictions_sha256=report["predictions_sha256"],candidate_audit_sha256=report["audit_sha256"],
                  capture_receipt_sha256=a.capture_receipt_sha,worker_sha256=worker_sha,input_authority_sha256=authority_sha,
                  edits_sha256=digest,GT_opened=False,author_TEST=False,author_partition="train",evaluation_only=True,
                  exact_replayed_policy_choices=review,exact_support_xor_area=True,detector_forwards=0,
                  optimizer_updates=0,counts=dict(totals),seconds=perf_counter()-start)
    content = (json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    (out/"RESULT_SAFE.json").write_bytes(content)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(content).hexdigest())),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("phase",choices=("prepare-edits","verify-edits"))
    p.add_argument("--authority",required=True,type=Path)
    p.add_argument("--authority-sha",required=True)
    p.add_argument("--output",required=True,type=Path)
    args = p.parse_args()
    values = json.loads(pin(args.authority,args.authority_sha))
    paths = {"code","protocol","capture_root","manifest","images","fit_result","fit_accepted","weights"}
    keys = {"workers","manifest_sha","capture_terminal_sha","capture_receipt_sha"}
    if args.phase == "verify-edits":
        paths.add("prepare")
        keys |= {"prepare_terminal_sha","prepare_receipt_sha"}
    need(set(values) == paths|keys and type(values["workers"]) is int and 1 <= values["workers"] <= 6
         and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and args.output.name == args.phase and not args.output.exists(), "Fresh independent CPU replay")
    auth = SimpleNamespace(**{k:Path(v).resolve(strict=True) if k in paths else v for k,v in values.items()})
    pin(auth.protocol,LOCK)
    need(auth.images == auth.manifest.parent/"images", "Closed private intake paths")
    run(auth,args.output,args.phase,args.authority_sha)


if __name__ == "__main__":
    main()
