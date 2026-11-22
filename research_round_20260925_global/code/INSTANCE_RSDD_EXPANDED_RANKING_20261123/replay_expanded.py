"""Replay two fixed source heads on identical expanded native boundary support."""
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

import numpy as np


PROTOCOL = "f4ea800a1b5e0672b7d0995387b56ecf68410092f477ba47837df253e2f2f6d3"
COMMIT = "e48025be91836468b025ce13a29d7ec1444db9b0"
PRIOR = "9368956a7d47e502c005a6adc1ab39edaca18cd5b3780002073ba672732a2b2f"
PRIOR_REF = "ed6e7d48b5f5ec0ba9a98f37327ed4cd075a95c5ffc3857167a79da53f6a0cbd"
POINT_PREP = "be95fce967568ec68af281429361a61aacf77537cbac45473b84711cae514354"
POINT_RESULT = "db6164067ce262d8b4a5fd5e020f3e62cd4a50d5d843a7c61b69535c1f4142e4"
OCC_PREP = "0869578ac9e9bdd7d20bf88b66337a8c3d10422c1c6e11ce999babc063c0d71e"
OCC_RESULT = "ede9c744262afad4ae538f021deafe3279954d725eaa6acb2be9ea89d1f5f133"
CODEC = "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"
PREPARED = "INSTANCE_RSDD_EXPANDED_RANKING_PREPARED_V1"
ACCEPTED = "INSTANCE_RSDD_EXPANDED_RANKING_ACCEPTED_V1"
ARMS = ("learned_common_native_tie", "point_expanded", "learned_expanded")


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    data = path.read_bytes()
    need(digest(data) == expected, "Exact path content: " + path.name)
    return data


def module(path, expected, name):
    unit = ModuleType(name)
    unit.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), unit.__dict__)
    return unit


def label_guard(images):
    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
            raise PermissionError("Closed offline replay")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0]))
            name = path.name.casefold()
            if (name in {"train.json", "test.json"} or name.endswith(".coco.json")
                    or {p.casefold() for p in path.parts} & {"annotation_coco", "annotations", "labels"}):
                raise PermissionError("Target label access belongs to scoring")
            if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                if path.parent != images or path.suffix.casefold() != ".jpg":
                    raise PermissionError("Only the pinned decoded source image roster is allowed")
    sys.addaudithook(audit)


class RecordedRender:
    def __init__(self, implementation):
        self.implementation = implementation
        self.logits = []

    def subdivide_binary_mask(self, *args, **kwargs):
        result = self.implementation.subdivide_binary_mask(*args, **kwargs)
        self.logits.extend(result.logits.unbind(0))
        return result


def closed_prepare(pr, root, terminal_sha, result_sha, status):
    value, term = pr.closed(root/"observed_prepare-edits/TERMINAL.json", terminal_sha,
                            root/"prepare-edits/RESULT_SAFE.json", result_sha)
    need(term["phase"] == "prepare-edits" and value["status"] == status
         and value["GT_opened"] is False, "Original observed label-free preparation")
    return value


def load_inputs(a, pr):
    prior = closed_prepare(pr, a.source_replay, POINT_PREP, POINT_RESULT, pr.PREPARED)
    point_request = pr.read_authority(json.loads(pin(a.source_replay/"AUTHORITY_prepare-edits_PRIVATE.json",
                                                     prior["input_authority_sha256"])), False)
    need(point_request.code == a.code and point_request.workers > 0, "Exact original PointRend inputs")
    capture, native, head_sha, occupancy_receipt, rows = pr.read_population(point_request)
    accepted, term = pr.closed(a.source_replay/"observed_verify-edits/TERMINAL.json", a.source_replay_terminal_sha,
                               a.source_replay/"verify-edits/RESULT_SAFE.json", PRIOR_REF)
    need(term["phase"] == "verify-edits" and term["worker_sha256"] == PRIOR
         and accepted["status"] == pr.ACCEPTED and accepted["exact_replayed_policy_choices"] is True
         and accepted["exact_support_xor_area"] is True and accepted["GT_opened"] is False
         and all(accepted[k] == prior[k] for k in ("edits_sha256", "counts", "predictions_sha256", "candidate_audit_sha256"))
         and accepted["predictions_sha256"] == native["predictions_sha256"]
         and accepted["fit_weight_sha256"] == head_sha, "Exact previously scored reference masks")
    old = closed_prepare(pr, point_request.old_replay, OCC_PREP, OCC_RESULT,
                         "INSTANCE_RSDD_TEST740_RFDETR_EDIT_PREPARED_V1")
    occ = json.loads(pin(point_request.old_replay/"AUTHORITY_prepare-edits_PRIVATE.json",
                         old["input_authority_sha256"]))
    need(Path(occ["images"]).resolve(strict=True) == a.images
         and Path(occ["capture"]).resolve(strict=True) == point_request.capture
         and Path(occ["code"]).resolve(strict=True) == a.code
         and old["edits_sha256"] == occupancy_receipt["edits_sha256"],
         "Same pre-existing image capture and admitted occupancy point head")
    former = [json.loads(line) for line in pin(a.source_replay/"prepare-edits/EDITS_PRIVATE.jsonl",
                                               accepted["edits_sha256"]).splitlines()]
    need(len(former) == len(rows) == 740, "Complete reference and candidate cohort")
    return point_request, occ, head_sha, accepted, list(zip(rows, former, strict=True))


def initialize(a, point_request, occ, head_sha):
    global worker
    label_guard(a.images)
    import torch
    from pycocotools import mask

    need(sys.version_info[:3] == (3, 10, 13) and torch.__version__ == "2.10.0+cu130"
         and np.__version__ == "1.26.4" and not torch.cuda.is_initialized(),
         "Exact CPU inference and interpolation runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    pr = module(a.code/"INSTANCE_RSDD_POINTREND_TRANSFER_20261122/replay_pointrend.py", PRIOR, "_main_point")
    capture = pr.load_capture_source(point_request)
    _, _, edit = capture.input_modules(a.code)
    base = edit.load_module(a.code/edit.BASE/"replay.py", edit.REPLAY_SHA)
    methods = base.methods(a.code)
    predict = base.predictor(a.code, methods, Path(occ["fit_result"]), Path(occ["fit_accepted"]), Path(occ["weights"]))
    infer, adapter, render, budget = pr.policies_modules(a, capture)
    head = pr.restore_head(adapter, pin(point_request.fit_weights, head_sha))
    helper = module(Path(__file__).with_name("support.py"), a.support_sha, "_native_expanded")
    editor = module(a.code/"RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py", CODEC, "_sparse_editor")
    worker = SimpleNamespace(a=point_request, base=base, m=methods, predict=predict, pr=pr,
                             infer=infer, adapter=adapter, render=render, budget=budget, head=head,
                             helper=helper, editor=editor, codec=mask, torch=torch, images=a.images)


def mask_edit(stock_rle, stock, sides, k):
    mask = worker.codec
    add, drop = sides
    flips = np.concatenate((add, drop))
    need(len(add) == len(drop) == k and len(np.unique(flips)) == 2*k
         and not stock.flat[add].any() and stock.flat[drop].all(), "Exact original addition and deletion signs")
    rle = worker.editor.xor_rle(stock_rle, flips)
    both = int(mask.area(mask.merge([stock_rle, rle], intersect=True)))
    need(int(mask.area(rle)) - both == int(mask.area(stock_rle)) - both == k,
         "Independent COCO XOR counts and exact area")
    return dict(swaps=k, rle=rle)


def own_center_outside(indices, width, box):
    y, x = np.divmod(indices, width)
    x, y = x+.5, y+.5
    return int(((x < box[0]) | (x > box[2]) | (y < box[1]) | (y > box[3])).sum())


def point_sides(logits, box, indices, observations, n, hw):
    torch = worker.torch
    result = []
    for side, old in zip(indices, observations, strict=True):
        need(len(old) == n and all(int(i) == pair[0] for i, pair in zip(side[:n], old, strict=True)),
             "Exact original PointRend S sampling coordinates")
        prefix = np.array([pair[1] for pair in old], dtype=np.float32)
        extra = torch.from_numpy(np.ascontiguousarray(side[n:], dtype=np.int64))
        if extra.numel():
            width = hw[1]
            xy = torch.stack((extra % width, extra // width), dim=-1)[None]
            tail = worker.adapter.paste_probabilities_at_pixels(logits[None], box, xy, hw).reshape(-1).numpy()
        else:
            tail = np.empty(0, np.float32)
        result.append(np.concatenate((prefix, tail)))
    return tuple(result)


def replay_image(item):
    (row_native, manifest, old_occ), published = item
    row, native = row_native
    w, m = worker, worker.m
    hw = tuple(row["original_hw"])
    need(row["image_id"] == published["image_id"] and hw == (512, 512)
         and not row["posterior_overrides"], "Same label-free native candidate stream")
    planes = w.base.load_array(w.a.capture, row["logits_ref"])
    spatial = w.base.load_array(w.a.basis, manifest["spatial"])
    need(planes.shape == (len(row["logit_ranks"]), 144, 144)
         and spatial.shape == (256, 144, 144), "Exact frozen query and spatial planes")
    by_rank = dict(zip(row["logit_ranks"], planes, strict=True))
    pixels = m.pair.image_features(w.base.load_rgb(w.images, row))
    recorded, work = [], []
    counts = Counter(images=1, mapped_candidates=len(native["coco"]), area_eligible=len(by_rank))
    history = published["changes"]
    for pos, (candidate, stock_row) in enumerate(zip(row["candidates"], native["coco"], strict=True), 1):
        area, rank, query = (candidate[k] for k in ("mask_area", "native_rank", "query_id"))
        rle = stock_row["segmentation"]
        key = str(pos)
        need(candidate["position"] == pos and stock_row["score"] == candidate["score"]
             and rank == row["native_ranks"][pos-1] and query == row["query_ids"][pos-1]
             and int(w.codec.area(rle)) == area, "Exact original candidate identity")
        former = old_occ["changes"].get(key, {}).get("free_natural")
        k = 0 if former is None else former["swaps"]
        if rank not in by_rank:
            need(not 32 <= area < 1024 and k == 0 and key not in history, "Frozen excluded area fallback")
            continue
        box = np.asarray(candidate["box_xyxy"], dtype=np.float32)
        if np.any(box[2:] == box[:2]):
            need(k == 0 and key not in history, "Frozen degenerate box fallback")
            counts["degenerate"] += 1
            continue
        stock = w.codec.decode(rle).astype(bool)
        probability = w.base.native_probability(by_rank[rank], stock, None)
        add, drop = w.base.expected_support(m.rank, stock, box, probability)
        coords = m.pair.pair_assignment(add, drop, hw[1]) if len(add) else np.empty((0, 2), dtype=np.int64)
        historical = history.get(key, {})
        need(historical.get("learned_common") == former and (k > 0) == ("point_equal" in historical),
             "Unchanged archived L and actual per-candidate allowance")
        recorded.append(dict(image_id=row["image_id"], rank=rank, query_id=query, rle=rle, raw_logits=by_rank[rank],
                             coordinates=coords, box_xyxy01=box/np.float32(hw[1]), k=k))
        work.append((key, stock, probability, box, area, former, historical))

    rendered = RecordedRender(w.render)
    old_budgets = w.pr.RecordedBudget(w.budget)
    with w.torch.inference_mode():
        result, inference = w.infer.infer_image(w.head, spatial, recorded, adapter=w.adapter,
                                                render=rendered, budget=old_budgets, image_hw=hw, chunk_size=12)
    counts["rendered"] = len(recorded)
    counts["point_evaluations"] = inference["point_evaluations"]
    need(inference["fallback"] == 0 and inference["point_evaluations"] == 3136 * len(recorded),
         "Exact original four PointRend subdivisions")
    changes = {}
    for row_input, more, logits, observed in zip(recorded, work, rendered.logits, old_budgets.observed, strict=True):
        key, stock, probability, box, area, former, historical = more
        stock_rle = row_input["rle"]
        rid = tuple(row_input[name] for name in ("image_id", "rank", "query_id"))
        need(result[rid]["P"] == historical.get("point_paste", {}).get("rle", stock_rle),
             "Re-rendered PointRend P exactly matches the original mask")
        k, coords = row_input["k"], row_input["coordinates"]
        sides, features = w.helper.expanded_features(m.rank, stock, box, area, row["candidates"][int(key)-1]["score"],
                                                    probability, pixels, coords)
        for name, available, original in zip(("add", "drop"), sides, coords.T, strict=True):
            need(np.array_equal(available[:len(coords)], original), "Historical S is an ordered prefix of E")
            counts[f"old_support_{name}"] += len(original)
            counts[f"expanded_support_{name}"] += len(available)
            counts[f"expanded_support_outside_roi_{name}"] += own_center_outside(available, hw[1], box)
        counts["support_expanded_candidates"] += any(len(s) > len(coords) for s in sides)
        if not k:
            counts["zero_k"] += 1
            continue
        n = len(coords)
        raw_added, raw_removed = result[rid]["B"][:k], result[rid]["B"][k:]
        need(mask_edit(stock_rle, stock, (raw_added, raw_removed), k) == historical["point_equal"],
             "Exact old PointRend S mask")
        q = w.helper.predict_sides(w.predict, features)
        counts["occupancy_point_evaluations"] += sum(map(len, sides))
        narrow = tuple(s[:n] for s in sides)
        qs = tuple(s[:n] for s in q)
        legacy = w.helper.select(narrow, qs, k, tie="legacy")
        need(mask_edit(stock_rle, stock, legacy, k) == former, "Exact old occupancy S mask")
        native_box = w.torch.as_tensor(box, dtype=w.torch.float32)[None]
        point = point_sides(logits, native_box, sides, observed[:2], n, hw)
        counts["point_extra_interpolations"] += sum(map(len, sides)) - 2*n
        selections = (w.helper.select(narrow, qs, k, tie="native"),
                      w.helper.select(sides, point, k, tie="native"),
                      w.helper.select(sides, q, k, tie="native"))
        changes[key] = {}
        counts["positive_k"] += 1
        counts["swaps"] += k
        for arm, selected in zip(ARMS, selections, strict=True):
            changes[key][arm] = mask_edit(stock_rle, stock, selected, k)
            for name, chosen, available, original in zip(("add", "drop"), selected, sides, narrow, strict=True):
                need(np.isin(chosen, available).all(), "Every chosen endpoint lies in E")
                outside_s = int((~np.isin(chosen, original)).sum())
                need(arm != "learned_common_native_tie" or outside_s == 0, "Tie control uses S only")
                counts[f"{arm}_outside_old_{name}"] += outside_s
                counts[f"{arm}_outside_roi_{name}"] += own_center_outside(chosen, hw[1], box)
        counts["native_tie_changed"] += changes[key]["learned_common_native_tie"] != former
    return dict(image_id=row["image_id"], changes=changes), counts


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def run(a, output, phase, authority_sha):
    pr = module(a.code/"INSTANCE_RSDD_POINTREND_TRANSFER_20261122/replay_pointrend.py", PRIOR, "_past_replay")
    point, occ, head_sha, original, rows = load_inputs(a, pr)
    independent = phase == "verify-edits"
    worker_sha = digest(Path(__file__).read_bytes())
    expected = None
    if independent:
        prepared, term = pr.closed(a.prepare/"observed_prepare-edits/TERMINAL.json", a.prepare_terminal_sha,
                                   a.prepare/"prepare-edits/RESULT_SAFE.json", a.prepare_receipt_sha)
        need(term["phase"] == "prepare-edits" and term["worker_sha256"] == worker_sha
             and prepared["status"] == PREPARED and prepared["support_sha256"] == a.support_sha
             and prepared["protocol_sha256"] == PROTOCOL and prepared["GT_opened"] is False,
             "Independent process uses exact source and observed preparation")
        expected = pin(a.prepare/"prepare-edits/EDITS_PRIVATE.jsonl", prepared["edits_sha256"]).splitlines(keepends=True)
        need(len(expected) == len(rows), "Complete previous output")
    started = perf_counter()
    pieces, counts = [], Counter()
    by_sensor = {"GF-3": Counter(), "TerraSAR-X": Counter()}
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a, point, occ, head_sha)) as jobs:
        for i, (change, subtotal) in enumerate(jobs.map(replay_image, rows, chunksize=2)):
            row = rows[i][0][0][0]
            need(change["image_id"] == row["image_id"], "Original full image order")
            packed = encode(change)
            need(expected is None or packed == expected[i], "Byte-identical independent native edits")
            counts.update(subtotal)
            by_sensor[row["sensor"]].update(subtotal)
            pieces.append(packed)
            if (i+1) % 25 == 0:
                print(json.dumps({"images":i+1, "point_evaluations":counts["point_evaluations"]}), flush=True)
    past = original["counts"]
    identical = ("images", "mapped_candidates", "area_eligible", "rendered", "point_evaluations", "degenerate")
    need(all(counts[k] == past[k] for k in identical) and counts["positive_k"] == past["point_equal_candidates"]
         and counts["swaps"] == past["point_equal_swaps"] and counts["zero_k"] == past["common_zero_swaps"]
         and {k:v["images"] for k,v in by_sensor.items()} == {"GF-3":583, "TerraSAR-X":157},
         "Original cohort, rendering effort and edit allowance")
    need(counts["old_support_add"] == past["endpoints_add"] and counts["old_support_drop"] == past["endpoints_drop"]
         and counts["expanded_support_add"] >= counts["old_support_add"]
         and counts["expanded_support_drop"] >= counts["old_support_drop"], "Exact historical and nested supports")
    ledger = b"".join(pieces)
    ledger_sha = digest(ledger)
    need(not independent or (ledger_sha == prepared["edits_sha256"] and dict(counts) == prepared["counts"]
                             and by_sensor == prepared["counts_by_sensor"]), "Exact independent ledger and exposure")
    output.mkdir()
    if not independent:
        (output/"EDITS_PRIVATE.jsonl").write_bytes(ledger)
    payload = dict(status=ACCEPTED if independent else PREPARED, protocol_sha256=PROTOCOL, prelock_commit=COMMIT,
                   edits_sha256=ledger_sha, worker_sha256=worker_sha, support_sha256=a.support_sha,
                   input_authority_sha256=authority_sha, source_replay_receipt_sha256=PRIOR_REF,
                   predictions_sha256=original["predictions_sha256"],
                   candidate_audit_sha256=original["candidate_audit_sha256"],
                   fit_weight_sha256=head_sha, images=740, author_TEST=True, GT_opened=False, AP=False,
                   model_forwards=0, optimizer_updates=0, exact_replayed_policy_choices=independent,
                   exact_support_xor_area=True, counts=dict(counts), counts_by_sensor=by_sensor,
                   seconds=perf_counter()-started, workers=a.workers)
    data = encode(payload)
    (output/"RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=payload["status"],result_sha256=digest(data))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    parser.add_argument("--authority",required=True,type=Path)
    parser.add_argument("--authority-sha",required=True)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    cfg = json.loads(pin(args.authority,args.authority_sha))
    paths = {"protocol","code","images","source_replay"}
    pins = {"source_replay_terminal_sha","support_sha"}
    if args.phase == "verify-edits":
        paths.add("prepare")
        pins |= {"prepare_terminal_sha","prepare_receipt_sha"}
    need(set(cfg) == paths|pins|{"workers"} and type(cfg["workers"]) is int and 1 <= cfg["workers"] <= 8
         and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and args.output.name == args.phase and not args.output.exists(), "Fresh isolated CPU replay")
    inputs = SimpleNamespace(**{k:Path(v).resolve(strict=True) if k in paths else v for k,v in cfg.items()})
    pin(inputs.protocol,PROTOCOL)
    label_guard(inputs.images)
    run(inputs,args.output,args.phase,args.authority_sha)


if __name__ == "__main__":
    main()
