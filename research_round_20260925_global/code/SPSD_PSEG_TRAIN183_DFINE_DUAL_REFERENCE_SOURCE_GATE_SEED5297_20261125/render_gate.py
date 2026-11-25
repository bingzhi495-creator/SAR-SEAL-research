"""Label-free replay for the exact first matched occupancy fit."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import torch


GROUP_PROTOCOL = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
FIT = "2ff91842249358aefd77915f87df3d337a814993245e0d973c057c21088ea538"
FIT_WORKER = "c00138dae34e620fe0617336309e2737ea7b247914fc8dc49b558382e278077e"
CORE = "0f1537f12ed4890bb60f71957e5d81926aeaab4760c8b01291383e4b5c53ee0d"
SELECTOR = "7c7a5415dad9c70804a0cf4dd07d0be46f8cbc5b3c0d1666e3cd067d25846205"
ATTRACTION = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
CAPTURE = "6f58ba8b486950431f4aebebf237e60029b447e5ec21a7f51ad2202d065f904a"
READBACK = "e461828b04f166d98030210fe42df0d590d24e2452ea5da8630fb0fc3219d795"
POINTS = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
SPLIT = "91095aacbbbc89003ffd6eb8f25506ce99a64eee5ac369a7753203bbf962bcf9"
ARMS = ("native", "dynamic", "static", "dual_robust", "dual_mean", "dual_cell_pessimistic",
        "pseg_repeat_mean", "spsd_repeat_mean", "mean_repeat_mean")
PROTOCOL = "33467a9ea2df47d80843ad2cdb8abc567f9349b820d0e8111dec3b069686b313"


def need(test, message):
    if not test:
        raise ValueError(message)


def hash_bytes(value):
    return sha256(value).hexdigest()


def pin(path, checksum):
    raw = path.read_bytes()
    need(hash_bytes(raw) == checksum, "Bound file digest: " + path.name)
    return raw


def unseal(path, checksum):
    return json.loads(pin(path, checksum))


def module(path, digest):
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), m.__dict__)
    return m


def terminal(path, phase, result, worker):
    raw = path.read_bytes()
    t = json.loads(raw)
    need(t["actual_exit"] == 0 and t["output_footer_verified"] is True and t["phase"] == phase
         and t["result_sha256"] == result and t["worker_sha256"] == worker, "Observed successful " + phase)
    return hash_bytes(raw)


def load_artifacts(a):
    fit = unseal(a.fit / "actual/RESULT_SAFE.json", FIT)
    fterm = terminal(a.fit / "observed_fit_retry/TERMINAL.json", "prepare-edits", FIT, FIT_WORKER)
    need(fit["worker_sha256"] == FIT_WORKER and fit["total_optimizer_updates"] == 4000
         and fit["training_core_sha256"] == CORE and fit["protocol_sha256"] == GROUP_PROTOCOL
         and fit["capture_readback_sha256"] == READBACK and fit["points_result_sha256"] == POINTS
         and fit["train_audit"]["seed"] == 5297 and fit["source_gate_labels_opened"] is False
         and fit["historical_test_opened"] is fit["source_gate_scored"] is False, "Exact seed 5297 FIT scope")

    cap = unseal(a.capture / "capture/RESULT_SAFE.json", CAPTURE)
    read = unseal(a.capture / "READBACK_SAFE.json", READBACK)
    cterm = terminal(a.capture / "observed_capture/TERMINAL.json", "capture", CAPTURE, cap["export_source_sha256"])
    rterm = terminal(a.capture / "observed_readback/TERMINAL.json", "verify", READBACK, read["readback_source_sha256"])
    pts = unseal(a.points / "actual_retry/RESULT_SAFE.json", POINTS)
    pterm = terminal(a.points / "observed_points_retry/TERMINAL.json", "prepare-edits", POINTS, pts["worker_sha256"])
    need(read["export_result_sha256"] == pts["capture_result_sha256"] == CAPTURE
         and read["export_terminal_sha256"] == cterm and pts["capture_readback_terminal_sha256"] == rterm
         and pts["capture_readback_sha256"] == READBACK and pts["group_partition_sha256"] == SPLIT
         and read["manifest_sha256"] == pts["image_only_manifest_sha256"]
         and read["predictions_sha256"] == cap["predictions_sha256"]
         and read["candidate_audit_sha256"] == cap["candidate_audit_sha256"]
         and read["identical_closed_candidate_streams"] is True and read["rows_at_100"] == 91500
         and read["GT_opened"] is read["TEST"] is pts["train_labels_opened"] is pts["test_labels_opened"] is False,
         "Exact capture, independently scored admission and separately grouped endpoints")
    core = module(a.core, CORE)
    selector = module(a.selector, SELECTOR)
    attraction = module(a.attraction, ATTRACTION)
    raw = pin(a.fit / "actual/FITTED_WEIGHTS_PRIVATE.pt", fit["output_checkpoint_sha256"])
    checkpoint = torch.load(BytesIO(raw), weights_only=True, map_location="cpu")
    need(checkpoint["seed"] == 5297 and set(checkpoint["models"]) == set(core.MODES), "Exact closed fit heads")
    models = {}
    for name, state in checkpoint["models"].items():
        model = core.PointHead()
        model.load_state_dict(state, strict=True)
        models[name] = model.eval()
        recorded = fit["train_audit"]["heads"][name]
        observed = hash_bytes(b"".join(t.detach().numpy().astype("<f4", copy=False).tobytes()
                                       for t in model.parameters()))
        need(recorded["final_tensor_sha256"] == observed and recorded["optimizer_updates"] == 1000
             and recorded["parameters"] == 1346, "Exact fitted parameter membership and steps")
    center, scale = checkpoint["center"].numpy(), checkpoint["scale"].numpy()
    receipts = dict(fit=fterm, capture=cterm, readback=rterm, points=pterm)
    return fit, read, pts, core, selector, attraction, models, center, scale, receipts


def actions(rows):
    table = np.asarray(rows, dtype=np.int32).reshape((-1, 2))
    return tuple(tuple(sorted(side.tolist())) for side in table.T)


def edited_mask(raw, stock, coordinates, pair, k):
    add, drop = (coordinates[np.asarray(side, dtype=np.intp), j] for j, side in enumerate(pair))
    need(len(add) == len(np.unique(add)) == len(drop) == len(np.unique(drop)) == k
         and not stock.ravel()[add].any() and stock.ravel()[drop].all(), "Exact original signed action budget")
    out = stock.copy(order="C")
    flat = out.ravel()
    flat[add] = True
    flat[drop] = False
    encoded = codec.encode(np.asfortranarray(out, dtype=np.uint8))
    encoded["counts"] = encoded["counts"].decode("ascii")
    old = int(codec.area(raw))
    intersection = int(codec.area(codec.merge([raw, encoded], intersect=True)))
    need(int(codec.area(encoded)) == old and old - intersection == k, "Actual equal signed native raster edits")
    return encoded


def render(a):
    start = perf_counter()
    fit, read, pts, core, selector, attraction, models, center, scale, terms = load_artifacts(a)
    gate = a.points / "actual_retry/source_gate"
    index = [json.loads(row) for row in pin(gate / "POINT_INDEX_PRIVATE.jsonl",
                                           pts["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    m = unseal(a.capture / "intake_retry/TRAIN_IMAGE_ONLY_MANIFEST_PRIVATE.json", read["manifest_sha256"])["items"]
    prediction_name = a.capture / "capture/DEV_MASK_PREDICTIONS_PRIVATE.jsonl"
    audit_name = a.capture / "capture/DEV_CANDIDATE_AUDIT_PRIVATE.jsonl"
    pred = pin(prediction_name, read["predictions_sha256"]).splitlines()
    au = pin(audit_name, read["candidate_audit_sha256"]).splitlines()
    need(len(pred) == len(au) == len(m) == 915 and len(index) == 183
         and len({x["source_group"] for x in index}) == 180
         and len({x["image_id"] for x in index}) == 183, "Exact all native captures and gate roster")
    totals = Counter()
    arm_counts = {k: Counter() for k in ARMS}
    stream_hash = sha256()
    a.output.mkdir()
    with (a.output / "GATE_RENDER_PRIVATE.jsonl").open("xb") as out:
        for item in index:
            ordinal, iid, shape = item["native_archive_ordinal"] - 1, item["image_id"], item["original_hw"]
            row, audit = json.loads(pred[ordinal]), json.loads(au[ordinal])
            manifest = m[ordinal]
            need(iid == row["image_id"] == audit["image_id"] == manifest["image_id"]
                 and item["rgb_sha256"] == manifest["rgb_sha256"] and item["source_group"] == manifest["source_group"]
                 and item["original_hw"] == manifest["original_hw"], "Exact gate image and decoded group identity")
            series = []
            for rank, (p, c) in enumerate(zip(row["coco"][:100], audit["candidates"][:100], strict=True), 1):
                need(p["image_id"] == iid and p["category_id"] == 1 and c["native_rank"] == rank
                     and c["score"] == p["score"] and p["segmentation"]["size"] == shape, "Exact owned native candidate order/score")
                series.append(dict(native=p["segmentation"], score=p["score"], quota=0, changes={}))
            need(len(series) == 100, "No native candidate dropped")
            raw_data = pin(gate / item["filename"], item["sha256"])
            need(len(raw_data) == item["bytes"], "Exact native endpoint packet length")
            with np.load(BytesIO(raw_data), allow_pickle=False) as bundle:
                ranks, starts, coords, features = (bundle[k] for k in ("native_ranks", "point_offsets",
                                                                      "signed_coordinates", "features"))
                quota, static_rows = (bundle[k] for k in ("static_offsets", "static_side_indices"))
            n, pp, qq = len(ranks), len(coords), len(static_rows)
            need(n == item["supported_candidates"] and n == len(set(ranks))
                 and pp == item["paired_sites"] and qq == item["realized_static_exchanges"]
                 and starts.shape == quota.shape == (n + 1,) and starts[0] == quota[0] == 0
                 and starts[-1] == pp and quota[-1] == qq and coords.shape == (pp, 2)
                 and features.shape == (pp, 2, 14) and static_rows.shape == (qq, 2)
                 and np.all((1 <= ranks) & (ranks <= 100)), "Exact native packet shape and supported counts")
            probabilities = {key: core.predict(value, features, center, scale) for key, value in models.items()}
            for j, r in enumerate(ranks):
                p = series[int(r) - 1]
                raw = p["native"]
                stock = codec.decode(raw).astype(bool, order="C")
                area = int(codec.area(raw))
                lo, hi, qlo, qhi = starts[j], starts[j+1], quota[j], quota[j+1]
                pair_coords = coords[lo:hi]
                need(32 <= area < 1024 and 1 <= len(pair_coords) <= 32, "Original supported candidate scope")
                real, static = attraction.select(stock, pair_coords, min(8, area//10, len(pair_coords)))
                k = int(qhi - qlo)
                need(k == len(real) == len(static) and np.array_equal(
                    static_rows[qlo:qhi], np.asarray(static, dtype=np.int16).reshape((-1, 2))),
                    "Independent historical static and dynamic quota parity")
                if k == 0:
                    continue
                inc = actions(static)
                proposed = {"dynamic": actions(real), "static": inc}
                for name, full in probabilities.items():
                    sliced = full[lo:hi]
                    choice = selector.select(sliced[:,0], sliced[:,1], *inc, k=k)
                    if name == "dual":
                        proposed.update(dual_robust=choice["robust"], dual_mean=choice["mean"],
                                        dual_cell_pessimistic=choice["cell_pessimistic"])
                    else:
                        proposed[name + "_mean"] = choice["mean"]
                need(set(proposed) == set(ARMS[1:]), "Fixed supervision and writer contrasts")
                cached = {}
                p["quota"] = k
                for arm, sig in proposed.items():
                    if sig not in cached:
                        cached[sig] = edited_mask(raw, stock, pair_coords, sig, k)
                    p["changes"][arm] = cached[sig]
                    arm_counts[arm].update(candidates=1, exchanges=k)
                    if arm != "static":
                        arm_counts[arm]["changed_vs_static"] += int(sig != inc)
            record = dict(image_id=iid, source_group=item["source_group"], original_hw=shape,
                          filename=manifest["image_ref"]["path"], candidates=series)
            payload = (json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
            need(out.write(payload) == len(payload), "Complete gate replay output row")
            stream_hash.update(payload)
            totals.update(images=1, candidates=100, supported=n, paired_sites=pp, quota=qq)
            if totals["images"] % 30 == 0:
                print("gate render images", totals["images"], flush=True)
    need(totals == dict(images=183,candidates=18300,supported=8821,paired_sites=265943,quota=57692)
         and all(t["candidates"] == 8785 and t["exchanges"] == 57692 for t in list(arm_counts.values())[1:]),
         "Original exact gate budget for every writer")
    result = dict(status="SPSD_DUAL_GATE_RENDER_COMPLETE", seed=5297,
                  protocol_sha256=PROTOCOL, group_protocol_sha256=GROUP_PROTOCOL,
                  worker_sha256=hash_bytes(Path(__file__).read_bytes()), fit_result_sha256=FIT,
                  fitted_checkpoint_sha256=fit["output_checkpoint_sha256"], points_result_sha256=POINTS,
                  capture_readback_sha256=READBACK, upstream_terminal_hashes=terms,
                  training_core_sha256=CORE, selector_sha256=SELECTOR, attraction_sha256=ATTRACTION,
                  render_stream_sha256=stream_hash.hexdigest(), counts=totals, arms=arm_counts,
                  decoded_groups=180, detector_forwards=0, optimizer_updates=0,
                  historical_test_opened=False, source_gate_labels_opened=False,
                  elapsed_seconds=perf_counter() - start)
    result_raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (a.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(result_raw)
    print(json.dumps(dict(status=result["status"], result_sha256=hash_bytes(result_raw))), flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("fit", "capture", "points", "core", "selector", "attraction", "protocol",
                 "group-protocol", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    pin(a.protocol, PROTOCOL)
    pin(a.group_protocol, GROUP_PROTOCOL)
    torch.set_num_threads(1)
    render(a)


if __name__ == "__main__":
    main()

