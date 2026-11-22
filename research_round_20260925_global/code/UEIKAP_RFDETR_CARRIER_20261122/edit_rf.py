"""Evaluate frozen RF-DETR edit policies without opening the UEIKAP labels."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask


BASE = "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122"
REPLAY_SHA = "be1516489123153a018ebc721806190d83f05285617e7c108e9b3c1064f66655"
PROTOCOL = "3ceb843a570028ccd2215cb5a211915b5ed80f82640786dffbf8cb03897da6e6"
MANIFEST = "1274563170a1dd6a7bc4326cf8d14a110d184d3d1687ec9bee22b6c5265be77f"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
CAPTURE_VERIFIED = "UEIKAP_RFDETR_CENTER5202_108_INDEPENDENT_READBACK_V1"
PREPARED = "UEIKAP_RFDETR_RESERVED_EDIT_PREPARED_V1"
ACCEPTED = "UEIKAP_RFDETR_RESERVED_EDIT_REPLAY_ACCEPTED_V1"
ARMS = ("fixed", "free_matched_count", "free_natural", "native_matched_count",
        "attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count")


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def pin(path, expected):
    data = path.read_bytes()
    check(sha256(data).hexdigest() == expected, "Pinned identity: " + path.name)
    return data


def load_module(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def ordered_pairs(q, cap, k=None):
    add = sorted(range(len(q)), key=lambda i: (-q[i, 0], i))[:cap]
    drop = sorted(range(len(q)), key=lambda i: (q[i, 1], i))[:cap]
    if k is None:
        k = sum(q[a, 0] > q[d, 1] for a, d in zip(add, drop, strict=True))
    return list(zip(add[:k], drop[:k], strict=True))


def policies(m, stock, coords, q, posterior, *, independent):
    check(q.shape == coords.shape and q.dtype == np.float32 and np.isfinite(q).all()
          and np.all((0 <= q) & (q <= 1)), "Frozen endpoint probabilities")
    cap = min(8, int(stock.sum()) // 10, len(coords))
    free = ordered_pairs if independent else m.free.choose_free
    fixed = [(i, i) for i in m.pair.choose_pairs(q[:, 0] - q[:, 1], cap)]
    matched = free(q, cap, len(fixed))
    natural = free(q, cap)
    native = free(posterior.ravel()[coords], cap, len(natural))
    dynamic, static = m.attraction.select(stock, coords, cap)
    learned = free(q, cap, len(dynamic))
    return dict(zip(ARMS, (fixed, matched, natural, native, dynamic, static, learned), strict=True))


def validate_edits(stock, coords, edits):
    check(set(edits) <= set(ARMS), "Only the locked policies")
    k = [edits[key]["swaps"] if key in edits else 0 for key in ARMS]
    check(k[0] == k[1] and k[2] == k[3] and k[4] == k[5] == k[6], "Matched per-candidate budgets")
    for entry in edits.values():
        other = mask.decode(entry["rle"]).astype(bool)
        count = entry["swaps"]
        check(other.shape == stock.shape and type(count) is int and 0 < count <= min(8, len(coords), int(stock.sum())//10),
              "Locked native canvas and cap")
        added = np.flatnonzero(other & ~stock)
        removed = np.flatnonzero(stock & ~other)
        check(len(added) == len(removed) == count and np.isin(added, coords[:, 0]).all()
              and np.isin(removed, coords[:, 1]).all(), "Native XOR and allowed endpoints")


def load_stream(path, expected):
    return [json.loads(line) for line in pin(path, expected).splitlines()]


def observed(path, expected_sha, result_sha, phase):
    event = json.loads(pin(path, expected_sha))
    check(event["actual_exit"] == 0 and event["output_footer_verified"] is True
          and event["phase"] == phase and event["result_sha256"] == result_sha,
          "Independently observed worker exit and output footer")
    return event


def read_capture(a):
    observed(a.capture_terminal, a.capture_terminal_sha, a.capture_receipt_sha, "verify-capture")
    receipt = json.loads(pin(a.capture_receipt, a.capture_receipt_sha))
    original = json.loads(pin(a.capture / "RESULT_SAFE.json", receipt["capture_sha256"]))
    check(receipt["status"] == CAPTURE_VERIFIED and original["checkpoint_sha256"] == CHECKPOINT
          and receipt["intake_sha256"] == original["intake_sha256"] == MANIFEST
          and receipt["protocol_sha256"] == original["protocol_sha256"] == PROTOCOL
          and all(receipt[key] == original[key] for key in ("predictions_sha256", "audit_sha256", "counts"))
          and receipt["GT_opened"] is original["GT_opened"] is receipt["author_TEST"] is False,
          "Observed verified RF-DETR capture and frozen intake")
    prediction = load_stream(a.capture / "PREDICTIONS_PRIVATE.jsonl", receipt["predictions_sha256"])
    audit = load_stream(a.capture / "CANDIDATE_AUDIT_PRIVATE.jsonl", receipt["audit_sha256"])
    check(len(prediction) == len(audit) == 108, "Unfiltered publisher coverage")
    return prediction, audit, receipt


def image_edits(r, m, predictors, a, row, native, *, independent):
    check(row["image_id"] == native["image_id"] and row["original_hw"] == [512, 512], "Ordered original author PNGs")
    info, stock_rows = row["candidates"], native["coco"]
    check(len(info) == len(stock_rows) <= 100, "Complete native mapped candidate order")
    ranks = [i["native_rank"] for i in info]
    check(ranks == sorted(set(ranks)) and all(1 <= value <= 100 for value in ranks), "Author native joint rank gaps")
    planes = r.load_array(a.capture, row["logits_ref"])
    check(planes.shape == (len(row["logit_ranks"]),144,144), "Accepted native-resolution logit planes")
    logits = dict(zip(row["logit_ranks"], planes, strict=True))
    check(list(logits) == [i["native_rank"] for i in info if 32 <= i["mask_area"] < 1024], "Exact edit-eligible plane order")
    post_overrides = {x["native_rank"]: x for x in row["posterior_overrides"]}
    check(set(post_overrides) <= set(logits), "Only accepted CUDA posterior overrides")
    changes, counts = {}, Counter()
    if not logits:
        return dict(image_id=row["image_id"], changes=changes), counts

    image = m.pair.image_features(r.load_rgb(a.images, row))
    pair, sampler = r.canvas(m, (512,512))
    for ordinal, (details, native_row) in enumerate(zip(info, stock_rows, strict=True), 1):
        rank = details["native_rank"]
        check(details["position"] == ordinal and 0 <= details["query_id"] < 164
              and details["score"] == native_row["score"], "Native RF Q164 mapping and positional linkage")
        if rank not in logits:
            continue
        stock = mask.decode(native_row["segmentation"]).astype(bool)
        check(int(stock.sum()) == details["mask_area"], "Observed original native foreground area")
        override = post_overrides.get(rank)
        check(override is None or override["query_id"] == details["query_id"], "Original override query identity")
        probability = r.native_probability(logits[rank], stock, None if override is None else r.load_array(a.capture, override))
        box = np.asarray(details["box_xyxy"], dtype=np.float32)
        check(box.shape == (4,) and np.isfinite(box).all() and np.all((box >= 0) & (box <= 512)), "Archived native box")
        data = pair.paired_points(stock, box, details["mask_area"], native_row["score"], probability, image, sampler)
        if data is None:
            if independent:
                r.check_support(m, stock, box, probability, np.empty((0,2), dtype=np.int32))
            continue
        coords, features = data[:2]
        if independent:
            r.check_support(m, stock, box, probability, coords)
        actions = policies(m, stock, coords, predictors(features), probability, independent=independent)
        edits = {name: r.mask_edit(stock, coords, choices) for name, choices in actions.items() if choices}
        validate_edits(stock, coords, edits)
        if edits:
            changes[str(ordinal)] = edits
        counts["editable_supports"] += 1
        for name, entry in edits.items():
            counts[name + "_candidates"] += 1
            counts[name + "_swaps"] += entry["swaps"]
    return dict(image_id=row["image_id"], changes=changes), counts


def run(a, independent):
    import torch
    check(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(), "CPU label-free replay")
    torch.set_num_threads(1)
    pin(a.protocol, PROTOCOL)
    pin(a.manifest, MANIFEST)
    predictions, audit, source = read_capture(a)
    base = load_module(a.code / BASE / "replay.py", REPLAY_SHA)
    m = base.methods(a.code)
    predictor = base.predictor(a.code, m, a.fit_result, a.fit_accepted, a.weights)
    previous, previous_sha = None, None
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    if independent:
        event = observed(a.prepared_terminal, a.prepared_terminal_sha, a.prepared_receipt_sha, "prepare-edits")
        old = json.loads(pin(a.prepared_receipt, a.prepared_receipt_sha))
        check(old["status"] == PREPARED and old["GT_opened"] is False and event["worker_sha256"] == worker_sha
              and old["capture_receipt_sha256"] == a.capture_receipt_sha, "Closed first replay with same capture")
        previous_sha = old["edits_sha256"]
        previous = load_stream(a.prepared_receipt.parent / "EDITS_PRIVATE.jsonl", previous_sha)
    counts, edited = Counter(), []
    for ordinal, (native, row) in enumerate(zip(predictions, audit, strict=True), 1):
        check(row["image_id"] == ordinal, "Original 108 image indices")
        result, values = image_edits(base, m, predictor, a, row, native, independent=independent)
        if independent:
            check(result == previous[ordinal-1], "Every edit recomputed at identical native cells")
        counts.update(values)
        edited.append(result)
    payload = b"".join(base.encoded(row) for row in edited)
    edit_sha = sha256(payload).hexdigest()
    check(not independent or len(previous) == 108 and edit_sha == previous_sha, "Exact prepared stream readback")
    counts.update(images=108, mapped_candidates=sum(len(row["coco"]) for row in predictions))
    return payload, dict(status=ACCEPTED if independent else PREPARED, GT_opened=False, author_TEST=False,
                         model_forwards=0, optimizer_updates=0, manifest_sha256=MANIFEST, protocol_sha256=PROTOCOL,
                         predictions_sha256=source["predictions_sha256"], candidate_audit_sha256=source["audit_sha256"],
                         capture_receipt_sha256=a.capture_receipt_sha, edits_sha256=edit_sha,
                         exact_replayed_policy_choices=independent, exact_support_xor_area=True,
                         counts=dict(counts), source_sha256=worker_sha)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=("prepare-edits","verify-edits"))
    p.add_argument("--authority", required=True, type=Path)
    p.add_argument("--authority-sha", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    values = json.loads(pin(args.authority,args.authority_sha))
    paths = {"protocol", "manifest", "images", "code", "capture", "capture_terminal", "capture_receipt",
             "fit_result", "fit_accepted", "weights"}
    stamps = {"capture_terminal_sha", "capture_receipt_sha"}
    independent = args.phase == "verify-edits"
    if independent:
        paths |= {"prepared_terminal", "prepared_receipt"}
        stamps |= {"prepared_terminal_sha", "prepared_receipt_sha"}
    check(set(values) == paths | stamps and not args.output.exists() and sys.flags.isolated and sys.dont_write_bytecode,
          "Exact independently pinned phase authority and fresh process output")
    a = SimpleNamespace(**{k: Path(v) if k in paths else v for k,v in values.items()})
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
