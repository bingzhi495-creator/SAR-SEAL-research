"""Run and independently replay frozen B5 edits on the closed SAR-SSP native capture."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType


CAPTURE_WORKER_SHA = "cba02f638a15b130a0655557e5e444d44a8d24d5642b53d46fb676aa33c2f407"
CAPTURE_SHA = "c40a133d62fafdecb9736d17abbd0ddcbf31ec42672d61eb0cbfc5d5df0b643e"
READBACK_SHA = "5788cd43fdc1e404cf6cc3c611dec47bb8ed30c55b926b22aabb6d3ec80e3711"
HEAD_SHA = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
EDITOR_SHA = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
PREPARED = "SAR_SSP_VAL232_B5_EDITS_PREPARED_V1"
VERIFIED = "SAR_SSP_VAL232_B5_EDITS_VERIFIED_V1"
RESULT = "RESULT_SAFE.json"
EDITS = "EDITS_PRIVATE.jsonl"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Pinned SHA-256 differs: " + path.name)
    return raw


def module(path, expected):
    source = ModuleType(path.stem)
    source.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), source.__dict__)
    return source


def closed(root, phase, expected, worker_sha, status):
    terminal = json.loads((root / ("observed_" + phase) / "TERMINAL.json").read_bytes())
    result = json.loads(pinned(root / phase / RESULT, expected))
    require(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["phase"] == result["phase"] == phase and result["status"] == status
            and result["worker_sha256"] == terminal["worker_sha256"] == worker_sha
            and terminal["result_sha256"] == expected and result["target_labels_opened"] is False
            and result["optimizer_updates"] == 0, "Independently observed closed frozen phase: " + phase)
    return result


def image_edits(base, editor, methods, pair, sampler, predict, images, capture,
                item, row, prediction, *, independent):
    import numpy as np

    image_id = item["image_id"]
    hw = tuple(item["original_hw"])
    require(image_id == row["image_id"] == prediction["image_id"]
            and all(row[key] == item[key] for key in ("source_image_name", "rgb_sha256", "original_hw"))
            and row["image_sha256"] == item["image_ref"]["sha256"], "Closed author image order and bytes")
    candidates, stock_rows = row["candidates"], prediction["coco"]
    require(len(candidates) == len(stock_rows) <= 100 and row["native_ranks"] == prediction["native_ranks"],
            "Complete and unchanged stock candidate identities")
    planes = base.load_array(capture, row["logits_ref"])
    ranks = row["logit_ranks"]
    logits = dict(zip(ranks, planes, strict=True))
    overrides = {value["native_rank"]: value for value in row["posterior_overrides"]}
    require(planes.shape == (len(ranks), 144, 144) and len(logits) == len(ranks)
            and ranks == [c["native_rank"] for c in candidates if 32 <= c["mask_area"] < 1024]
            and len(overrides) == len(row["posterior_overrides"]) and set(overrides) <= set(logits),
            "Original [32,1024) stock area and posterior identity")
    image_features = methods.pair.image_features(base.load_rgb(images, row)) if logits else None
    counts = Counter(images=1, mapped_candidates=len(candidates), eligible_planes=len(logits))
    changes = {}
    for position, (candidate, original) in enumerate(zip(candidates, stock_rows, strict=True), 1):
        rank = candidate["native_rank"]
        query = candidate["query_id"]
        area = candidate["mask_area"]
        rle = original["segmentation"]
        require(candidate["position"] == position and rank == row["native_ranks"][position - 1]
                and query == row["query_ids"][position - 1] and 0 <= query < 164
                and original["image_id"] == image_id and original["category_id"] == 1
                and candidate["score"] == original["score"] and rle["size"] == list(hw),
                "Stock mask, detection score and rank map retained")
        if rank not in logits:
            continue
        stock = base.codec.decode(rle).astype(bool)
        require(int(stock.sum()) == area, "Original stock area")
        override = overrides.get(rank)
        require(override is None or override["query_id"] == query, "Only same-query posterior fallback")
        probability = base.native_probability(logits[rank], stock,
                                              None if override is None else base.load_array(capture, override))
        box = np.asarray(candidate["box_xyxy"], dtype=np.float32)
        require(box.shape == (4,) and np.isfinite(box).all() and np.all(box[:2] <= box[2:])
                and np.all((box >= 0) & (box <= np.asarray((*hw[::-1], *hw[::-1])))),
                "Exact admitted native ROI")
        detail = pair.paired_points(stock, box, area, original["score"], probability, image_features, sampler)
        coords = np.empty((0, 2), dtype=np.int32) if detail is None else detail[0]
        if independent:
            base.check_support(methods, stock, box, probability, coords)
        if not len(coords):
            continue
        require(coords.ndim == 2 and coords.shape[1] == 2 and 0 < len(coords) <= 32
                and np.unique(coords).size == coords.size and np.all((0 <= coords) & (coords < stock.size))
                and not stock.flat[coords[:, 0]].any() and stock.flat[coords[:, 1]].all(),
                "Original signed boundary sites and one-to-one pairing")
        q = predict(detail[1])
        actions = editor.policies(methods, stock, coords, q, probability, independent=independent)
        if independent:
            gain = q[:, 0] - q[:, 1]
            cap = min(8, area // 10, len(coords))
            expected = [(idx, idx) for idx in sorted(range(len(gain)), key=lambda idx: (-float(gain[idx]), idx))
                        if gain[idx] > 0][:cap]
            require(actions["fixed"] == expected, "Independently recovered positive-margin distance choices")
        edits = {arm: base.mask_edit(stock, coords, decision) for arm, decision in actions.items() if decision}
        editor.validate_edits(stock, coords, edits)
        counts["editable_supports"] += 1
        for arm, edit in edits.items():
            counts[arm + "_swaps"] += edit["swaps"]
            counts[arm + "_candidates"] += 1
        if edits:
            changes[str(position)] = edits
    return dict(image_id=image_id, changes=changes), counts


def run(args):
    started = perf_counter()
    pinned(args.protocol, args.protocol_sha)
    independent = args.phase == "verify-edits"
    root = args.capture_run
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    source = module(root / "capture_sar_ssp.py", CAPTURE_WORKER_SHA)
    original = closed(root, "capture", CAPTURE_SHA, CAPTURE_WORKER_SHA, source.CAPTURED)
    checked = closed(root, "verify-capture", READBACK_SHA, CAPTURE_WORKER_SHA, source.VERIFIED)
    require(checked["capture_sha256"] == CAPTURE_SHA and checked["exact_native_parity"] is True
            and checked["model_forward_calls"] == 0 and original["model_forward_calls"] == source.COUNT
            and all(original[key] == checked[key] for key in (
                "intake_sha256", "predictions_sha256", "audit_sha256", "counts", "checkpoint_sha256")),
            "Exact capture/readback linkage")
    _, native, _, items, torch = source.setup_capture(args.code, root / "intake", False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    from rfdetr.datasets.io_utils import decode_pil_image_bytes

    require(decode_pil_image_bytes is not None and torch.get_float32_matmul_precision() == "high",
            "Original B5 unchanged CPU inference policy")
    editor = module(args.code / "UEIKAP_RFDETR_CARRIER_20261122/edit_rf.py", EDITOR_SHA)
    base = module(args.code / editor.BASE / "replay.py", editor.REPLAY_SHA)
    methods = base.methods(args.code)
    pair, sampler = base.canvas(methods, source.HW)
    step = base.read_source(args.code / base.SPOT / "fit_step.py", methods.readout.PINS["fit_step.py"])
    plan = base.read_source(args.code / base.RF / "fit_plan.py", methods.readout.PINS["fit_plan.py"])
    trainer = step.load_trainer(args.code / base.BASE / "fit109_paired_point/train_point.py")
    heads, center, scale = plan.restore_bundle(args.point_head, HEAD_SHA, trainer, step)

    def predict(features):
        return step.predict(heads["S"], features, center, scale)

    capture = root / "capture"
    predictions = editor.load_stream(capture / native.PRED, original["predictions_sha256"])
    audits = editor.load_stream(capture / native.AUDIT, original["audit_sha256"])
    previous, old = None, None
    if independent:
        old = closed(args.output.parent, "prepare-edits", args.prepare_sha, worker_sha, PREPARED)
        require(old["protocol_sha256"] == args.protocol_sha and old["capture_sha256"] == CAPTURE_SHA
                and old["capture_readback_sha256"] == READBACK_SHA and old["point_head_sha256"] == HEAD_SHA,
                "Same image roster, protocol and frozen head during independent replay")
        previous = pinned(args.output.parent / "prepare-edits" / EDITS, old["edits_sha256"]).splitlines(keepends=True)
        require(len(previous) == source.COUNT, "Complete previously prepared edit ledger")

    counts = Counter()
    rows = []
    for position, (item, audit, prediction) in enumerate(zip(items, audits, predictions, strict=True), 1):
        result, census = image_edits(base, editor, methods, pair, sampler, predict, root / "intake" / "images",
                                     capture, item, audit, prediction, independent=independent)
        row = base.encoded(result)
        require(previous is None or row == previous[position - 1], "Independently identical candidate inputs and locations")
        counts.update(census)
        rows.append(row)
        if position % 20 == 0:
            print(json.dumps(dict(images=position, **{k: v for k, v in counts.items() if k != "images"})), flush=True)

    require(counts["images"] == source.COUNT and counts["mapped_candidates"] == original["counts"]["mapped"]
            and counts["eligible_planes"] == original["counts"]["eligible"]
            and counts["fixed_swaps"] == counts["free_matched_count_swaps"]
            and counts["attraction_dynamic_swaps"] == counts["attraction_static_at_dynamic_count_swaps"]
            == counts["learned_at_attraction_count_swaps"]
            and not torch.cuda.is_initialized(), "Full CPU-only ledger with exact matched budgets")
    payload = b"".join(rows)
    edits_sha = sha256(payload).hexdigest()
    require(old is None or (dict(counts) == old["counts"] and edits_sha == old["edits_sha256"]),
            "Exact original ledger and aggregate count")
    args.output.mkdir()
    if not independent:
        with (args.output / EDITS).open("xb") as stream:
            stream.write(payload)
    result = dict(status=VERIFIED if independent else PREPARED, phase=args.phase,
                  worker_sha256=worker_sha, protocol_sha256=args.protocol_sha,
                  capture_sha256=CAPTURE_SHA, capture_readback_sha256=READBACK_SHA,
                  prepare_sha256=args.prepare_sha, edits_sha256=edits_sha, point_head_sha256=HEAD_SHA,
                  intake_sha256=source.MANIFEST_SHA, counts=dict(counts), exact_support_xor_area=True,
                  exact_replayed_policy_choices=independent, target_labels_opened=False,
                  detector_forwards=0, optimizer_updates=0, seconds=perf_counter() - started)
    receipt_sha = source.write_json(args.output / RESULT, result)
    print(json.dumps(dict(status=result["status"], result_sha256=receipt_sha)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    for path in ("code", "capture-run", "protocol", "point-head", "output"):
        parser.add_argument("--" + path, required=True, type=Path)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--prepare-sha")
    args = parser.parse_args()
    for path in ("code", "capture_run", "protocol", "point_head"):
        setattr(args, path, getattr(args, path).resolve(strict=True))
    args.output = args.output.resolve()
    require(args.output.name == args.phase and args.output.parent == args.capture_run and not args.output.exists()
            and (args.prepare_sha is not None) == (args.phase == "verify-edits"),
            "Fresh phase output beside closed capture and first replay")
    run(args)


if __name__ == "__main__":
    main()
