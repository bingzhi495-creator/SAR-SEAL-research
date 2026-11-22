"""Replay source-only PointRend and occupancy readouts on the same native candidates."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


PREPARED = "INSTANCE_RSDD_POINTREND_REPLAY_PREPARED_V1"
ACCEPTED = "INSTANCE_RSDD_POINTREND_REPLAY_ACCEPTED_V1"
OLD_RESULT = "0796f68909a5c99ab49e1805a6cb209606cf4e5a1ee5cd135a8a672ff57a6eac"
OLD_TERMINAL = "0890e2f53de83888e15cdf0ad6dbba423bf0c991ff5162d059eeb4ddc8e98708"
OLD_EDITS = "b3296ed2eb052a1a272a8457dc38bc53ee2ad8ac08501b89d6b373f0abe58eba"
PINNED = {
    "RFDETR_POINTREND_FIT109_20261121/infer_point_policies.py":
        "4f472c97f7e2ee3a4e34b3eafe3944c155be8a1d96c957143973ddfc346cd546",
    "RFDETR_POINTREND_SUBDIVISION_20261121/roi_point_head.py":
        "672bdd2aed8c0056f29c816f829bae9ceae5f9a685309beebf107aa59e547274",
    "RFDETR_POINTREND_SUBDIVISION_20261121/pointrend_subdivision.py":
        "6c6be609317dc7960ba901b690e661d92588d88ee1f3210c9156c81a4d3219ad",
    "RFDETR_POINTREND_SUBDIVISION_20261121/matched_budget.py":
        "4ca4207435c03cbd45567c4aff470f21b892c2d6889b5fad4b0f2b6e94316b7a",
}


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, f"Pinned bytes: {path.name}")
    return raw


def install_guard():
    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
            raise PermissionError("Offline raster readout")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0]))
            if (path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
                    or path.name.casefold().endswith(".coco.json")
                    or path.name.casefold() in {"train.json", "test.json"}
                    or {p.casefold() for p in path.parts} & {"annotation_coco", "annotations", "labels"}):
                raise PermissionError("PointRend replay has no image or label file input")
    sys.addaudithook(audit)


def load_capture_source(a):
    path = a.basis_worker
    source = ModuleType("_rsdd_basis_capture_for_replay")
    source.__file__ = str(path)
    exec(compile(pin(path, a.basis_worker_sha), str(path), "exec"), source.__dict__)
    return source


def read_authority(payload, independent):
    paths = {"code", "protocol", "roster", "capture", "capture_receipt", "capture_terminal", "basis", "basis_terminal",
             "basis_worker", "old_replay", "fit_weights", "fit_result", "fit_terminal"}
    stamps = {"basis_worker_sha", "basis_terminal_sha", "basis_result_sha", "fit_terminal_sha", "fit_result_sha"}
    if independent:
        paths |= {"prepare", "prepare_terminal"}
        stamps |= {"prepare_terminal_sha", "prepare_result_sha"}
    need(set(payload) == paths | stamps | {"workers"}
         and type(payload["workers"]) is int and 1 <= payload["workers"] <= 8
         and all(Path(payload[k]).is_absolute() for k in paths)
         and all(re.fullmatch("[0-9a-f]{64}", payload[k]) for k in stamps), "Complete fixed phase inputs")
    return SimpleNamespace(**{k: Path(v).resolve(strict=True) if k in paths else v for k, v in payload.items()})


def closed(terminal, terminal_sha, result, result_sha):
    obs = json.loads(pin(terminal, terminal_sha))
    need(obs["actual_exit"] == 0 and obs["output_footer_verified"] is True and obs["result_sha256"] == result_sha,
         f"Expected worker exit and output receipt: {result.parent.name}")
    return json.loads(pin(result, result_sha)), obs


def read_population(a):
    capture = load_capture_source(a)
    pin(a.protocol, capture.LOCK_SHA)
    a.capture_terminal_sha = capture.CAPTURE_TERMINAL
    a.capture_receipt_sha = capture.CAPTURE_RECEIPT
    base, replay, edit = capture.input_modules(a.code)
    rows, orig = replay.read_capture(a, edit)

    previous_root = a.old_replay
    previous, prev_terminal = closed(previous_root / "observed_verify-edits/TERMINAL.json", OLD_TERMINAL,
                                     previous_root / "verify-edits/RESULT_SAFE.json", OLD_RESULT)
    need(previous["status"] == replay.ACCEPTED and previous["source_sha256"] == capture.REPLAY_SHA
         == prev_terminal["worker_sha256"] and previous["GT_opened"] is False
         and previous["edits_sha256"] == OLD_EDITS and previous["exact_replayed_policy_choices"] is True
         and previous["capture_receipt_sha256"] == capture.CAPTURE_RECEIPT and previous["counts"]["images"] == 740
         and previous["predictions_sha256"] == orig["predictions_sha256"]
         and previous["candidate_audit_sha256"] == orig["audit_sha256"], "Observed exact former occupancy replay")
    occupancy = edit.load_stream(previous_root / "prepare-edits/EDITS_PRIVATE.jsonl", OLD_EDITS)

    basis, basis_obs = closed(a.basis_terminal, a.basis_terminal_sha, a.basis / "RESULT_SAFE.json", a.basis_result_sha)
    need(basis["status"] == capture.STATUS and basis_obs["phase"] == "capture"
         and basis["worker_sha256"] == basis_obs["worker_sha256"] == a.basis_worker_sha
         and basis["images"] == basis["model_forwards"] == 740 and basis["optimizer_updates"] == 0
         and basis["GT_opened"] is basis["AP"] is False and basis["author_TEST"] is True
         and basis["protocol_sha256"] == capture.LOCK_SHA and basis["prelock_commit"] == capture.LOCK_COMMIT
         and basis["checkpoint_sha256"] == base.SOURCE_CHECKPOINT
         and basis["capture_receipt_sha256"] == capture.CAPTURE_RECEIPT
         and basis["predictions_sha256"] == orig["predictions_sha256"]
         and basis["candidate_audit_sha256"] == orig["audit_sha256"] and basis["counts"] == orig["counts"]
         and basis["exact_original_candidates_and_masks"] is basis["exact_original_eligible_logits"]
         is basis["exact_all_query_basis_contraction"] is True, "Observed parity-verified frozen target basis capture")
    index = json.loads(pin(a.basis / "INDEX_PRIVATE.json", basis["index_sha256"]))
    for ordinal, ((row, _), file, edits) in enumerate(zip(rows, index, occupancy, strict=True), 1):
        need(row["image_id"] == edits["image_id"] == file["image_id"] and file["ordinal"] == ordinal
             and row["rgb_sha256"] == file["rgb_sha256"] and len(row["logit_ranks"]) == file["eligible_rows"]
             and file["spatial"]["path"] == f"spatial/{ordinal:06d}.npy", "Ordered image, basis and occupancy identity")
    need(len(rows) == len(index) == len(occupancy) == 740, "Full admitted analysis roster")

    fit, _ = closed(a.fit_terminal, a.fit_terminal_sha, a.fit_result, a.fit_result_sha)
    need(fit["status"] == "POINTREND_FULL_FIT109_SOURCE_REFIT_COMPLETE" and fit["scaler"] is False
         and fit["carrier_checkpoint_sha256"] == base.SOURCE_CHECKPOINT
         and fit["optimizer_updates"] == 200 and fit["native_targets"] == 39200
         and a.fit_weights.name == "HEAD_PRIVATE.npz" and a.fit_weights.parent == a.fit_result.parent,
         "Fixed all-FIT source-only PointRend head")
    pin(a.fit_weights, fit["head_sha256"])
    return capture, orig, fit["head_sha256"], previous, list(zip(rows, index, occupancy, strict=True))


def policies_modules(a, capture):
    modules = {Path(name).stem: capture.module(a.code / name, expected) for name, expected in PINNED.items()}
    return tuple(modules[name] for name in ("infer_point_policies", "roi_point_head", "pointrend_subdivision", "matched_budget"))


def restore_head(adapter, weights):
    import numpy as np
    import torch

    head = adapter.BinaryPointHead()
    expected = head.state_dict()
    state = {}
    with np.load(BytesIO(weights), allow_pickle=False) as saved:
        need(set(saved.files) == set(expected) and len(saved.files) == len(expected), "Exact source head state schema")
        for name, tensor in expected.items():
            values = saved[name]
            need(values.dtype == np.float32 and values.shape == tuple(tensor.shape) and np.isfinite(values).all(),
                 "Finite FP32 source head tensors")
            state[name] = torch.from_numpy(values)
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)


class RecordedBudget:
    """Delegate unchanged PointRend ranking and record its actual endpoint probabilities."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.observed = []

    def project_matched_budget(self, adds, drops, k):
        self.observed.append((adds, drops, k))
        return self.delegate.project_matched_budget(adds, drops, k)


def initialize(a, fit_sha):
    global worker
    install_guard()
    import numpy as np
    import torch

    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized()
         and sys.version_info[:3] == (3, 10, 13) and np.__version__ == "1.26.4" and torch.__version__ == "2.10.0+cu130",
         "Original CPU numerical runtime for PointRend inference")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    capture = load_capture_source(a)
    _, _, edit = capture.input_modules(a.code)
    base = edit.load_module(a.code / edit.BASE / "replay.py", edit.REPLAY_SHA)
    m = base.methods(a.code)
    infer, adapter, render, budget = policies_modules(a, capture)
    head = restore_head(adapter, pin(a.fit_weights, fit_sha))
    worker = SimpleNamespace(a=a, base=base, methods=m, infer=infer, adapter=adapter, render=render, budget=budget, head=head)


def counts_for_paste(stock, rle, summary):
    import numpy as np
    from pycocotools import mask as codec

    decoded = codec.decode(rle).astype(bool)
    need(decoded.shape == stock.shape, "ROI paste remains on the original native raster")
    ads = int(np.count_nonzero(decoded & ~stock))
    rem = int(np.count_nonzero(stock & ~decoded))
    need((ads, rem, ads+rem, ads-rem) == tuple(summary[k] for k in ("adds", "drops", "edits", "area_delta")),
         "ROI repaint verified against native raster XOR and area")
    return ads, rem, not decoded.any()


def verify_swap(stock, coords, record):
    import numpy as np
    from pycocotools import mask as codec

    k = record["swaps"]
    edited = codec.decode(record["rle"]).astype(bool)
    need(edited.shape == stock.shape and type(k) is int and 1 <= k <= min(8, len(coords), int(stock.sum())//10),
         "Fixed realized count on the original raster")
    adds = np.flatnonzero(edited & ~stock)
    drops = np.flatnonzero(stock & ~edited)
    need(len(adds) == len(drops) == k and np.isin(adds, coords[:, 0]).all() and np.isin(drops, coords[:, 1]).all(),
         "Both exact native edit supports and guaranteed area conservation")


def check_endpoints(item, observations, counts, image_hw):
    import numpy as np

    adds, drops, k = observations
    coords = item["coordinates"]
    need(k == item["k"] and len(coords) == len(adds) == len(drops), "Actual PointRend endpoint counts")
    bounds = np.array(item["box_xyxy01"], dtype=np.float32) * np.array([image_hw[1], image_hw[0]]*2, dtype=np.float32)
    for side, (values, name) in enumerate(zip((adds, drops), ("add", "drop"), strict=True)):
        if not values:
            continue
        idx, probs = zip(*values, strict=True)
        idx = np.asarray(idx)
        p = np.asarray(probs)
        need(np.array_equal(idx, coords[:, side]) and np.isfinite(p).all() and np.all((0 <= p) & (p <= 1)),
             "Actual rendered-paste probability queried on coincident legacy endpoint")
        uv = np.stack((idx % image_hw[1], idx // image_hw[1]), axis=1) + .5
        outside = np.any((uv < bounds[:2]) | (uv > bounds[2:]), axis=1)
        zero = p == 0
        counts[f"endpoints_{name}"] += len(idx)
        counts[f"outside_{name}"] += int(outside.sum())
        counts[f"outside_zero_{name}"] += int(np.count_nonzero(outside & zero))
        counts[f"outside_nonzero_{name}"] += int(np.count_nonzero(outside & ~zero))
        counts[f"inside_zero_{name}"] += int(np.count_nonzero(~outside & zero))


def replay_image(work):
    import numpy as np
    from pycocotools import mask as codec

    (row, native), file, previous = work
    a, m, base = worker.a, worker.methods, worker.base
    hw = tuple(row["original_hw"])
    need(hw == (512, 512), "Original square target view; normalization is exact power-of-two")
    planes = base.load_array(a.capture, row["logits_ref"])
    spatial = base.load_array(a.basis, file["spatial"])
    need(planes.shape == (len(row["logit_ranks"]), 144, 144) and spatial.shape == (256, 144, 144)
         and not row["posterior_overrides"], "Verified target basis, query planes and original CPU stock parity")
    by_rank = dict(zip(row["logit_ranks"], planes, strict=True))
    counts = Counter(images=1, mapped_candidates=len(native["coco"]), area_eligible=len(by_rank))
    data, same = [], []
    learned_positions = {p for p, modes in previous["changes"].items() if "free_natural" in modes}
    learned_seen = set()
    for pos, (info, cam) in enumerate(zip(row["candidates"], native["coco"], strict=True), 1):
        area = info["mask_area"]
        rank, query = info["native_rank"], info["query_id"]
        key = str(pos)
        need(info["position"] == pos and cam["score"] == info["score"] and rank == row["native_ranks"][pos-1]
             and query == row["query_ids"][pos-1] and int(codec.area(cam["segmentation"])) == area,
             "Identical native rank, query, score and mask area")
        if rank not in by_rank:
            need(not 32 <= area < 1024 and key not in learned_positions, "Outside area allowance, all arms stay native")
            counts["area_below_32" if area < 32 else "area_at_least_1024"] += 1
            continue
        need(32 <= area < 1024, "Exact original area-eligible candidate")
        box = np.asarray(info["box_xyxy"], dtype=np.float32)
        need(box.shape == (4,) and np.isfinite(box).all() and np.all((0 <= box) & (box <= 512))
             and np.all(box[2:] >= box[:2]), "Original clipped detector RoI")
        if np.any(box[2:] == box[:2]):
            need(key not in learned_positions, "Degenerate box lacks original occupancy edit allowance")
            counts["degenerate"] += 1
            continue
        stock = codec.decode(cam["segmentation"]).astype(bool)
        probability = base.native_probability(by_rank[rank], stock, None)
        add, drop = base.expected_support(m.rank, stock, box, probability)
        coords = m.pair.pair_assignment(add, drop, hw[1]) if len(add) else np.empty((0, 2), dtype=np.int64)
        need(coords.shape == (len(add), 2) and np.array_equal(np.sort(coords[:, 0]), add)
             and np.array_equal(np.sort(coords[:, 1]), drop), "Complete independently certified former paired endpoint sets")
        old = previous["changes"].get(key, {}).get("free_natural")
        if old is not None:
            verify_swap(stock, coords, old)
            learned_seen.add(key)
        data.append(dict(image_id=row["image_id"], rank=rank, query_id=query, rle=cam["segmentation"],
                         box_xyxy01=box/np.float32(512), raw_logits=by_rank[rank], coordinates=coords,
                         k=old["swaps"] if old is not None else 0))
        same.append((key, stock, old))
        counts["editable_supports"] += int(bool(len(coords)))
        counts["common_zero_swaps"] += int(old is None)
        counts["common_no_endpoint_support"] += int(not len(coords))
    need(learned_seen == learned_positions, "All original occupancy edits preserved inside common support")

    budget = RecordedBudget(worker.budget)
    rendered, summary = worker.infer.infer_image(worker.head, spatial, data, adapter=worker.adapter, render=worker.render,
                                                budget=budget, image_hw=hw, chunk_size=12)
    need(summary["fallback"] == 0 and summary["rendered"] == len(data) == len(budget.observed)
         and summary["point_evaluations"] == 3136*len(data), "All and only common RoIs get the fixed four-stage inference")
    counts["rendered"] = len(data)
    counts["point_evaluations"] = summary["point_evaluations"]
    edits = {}
    for (pos, stock, old), item, obs in zip(same, data, budget.observed, strict=True):
        check_endpoints(item, obs, counts, hw)
        key = tuple(item[k] for k in ("image_id", "rank", "query_id"))
        output = rendered[key]
        record = {}
        for alias, name in (("P", "point_paste"), ("C", "coarse_paste")):
            rle = output[alias]
            adds, drops, empty = counts_for_paste(stock, rle, output["diagnostics"][alias])
            if adds+drops:
                record[name] = dict(adds=adds, drops=drops, rle=rle)
                counts[f"{name}_candidates"] += 1
                counts[f"{name}_adds"] += adds
                counts[f"{name}_drops"] += drops
                counts[f"{name}_empty"] += int(empty)
        k = item["k"]
        indices = output["B"]
        need(len(indices) == 2*k and output["diagnostics"]["B"]["area_delta"] == 0, "Exact same-budget PointRend choice")
        if k:
            updated = stock.copy(order="C")
            updated.flat[indices] = ~updated.flat[indices]
            rle = codec.encode(np.asfortranarray(updated, dtype=np.uint8))
            rle["counts"] = rle["counts"].decode("ascii")
            entry = dict(swaps=k, rle=rle)
            verify_swap(stock, item["coordinates"], entry)
            record.update(learned_common=old, point_equal=entry)
            for name in ("learned_common", "point_equal"):
                counts[f"{name}_candidates"] += 1
                counts[f"{name}_swaps"] += k
            counts["point_equal_nonpositive"] += output["diagnostics"]["B"]["nonpositive_flips"]
        if record:
            edits[pos] = record
    return dict(image_id=row["image_id"], changes=edits), counts


def run(a, output, independent):
    capture, orig, fit_sha, previous_replay, population = read_population(a)
    worker_sha = digest(Path(__file__).read_bytes())
    counts = Counter()
    by_sensor = {"GF-3": Counter(), "TerraSAR-X": Counter()}
    changes = []
    if independent:
        prev, obs = closed(a.prepare_terminal, a.prepare_terminal_sha, a.prepare / "RESULT_SAFE.json", a.prepare_result_sha)
        need(prev["status"] == PREPARED and prev["worker_sha256"] == obs["worker_sha256"] == worker_sha
             and obs["phase"] == "prepare-edits" and prev["GT_opened"] is False
             and prev["basis_result_sha256"] == a.basis_result_sha and prev["fit_result_sha256"] == a.fit_result_sha
             and prev["fit_weight_sha256"] == fit_sha and prev["protocol_sha256"] == capture.LOCK_SHA,
             "Independent new execution against exact observed prepared replay")
        expected = pin(a.prepare / "EDITS_PRIVATE.jsonl", prev["edits_sha256"]).splitlines(keepends=True)
        need(len(expected) == len(population), "Same full ordered test replay")

    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a, fit_sha)) as pool:
        for i, (change, number) in enumerate(pool.map(replay_image, population, chunksize=2)):
            need(change["image_id"] == population[i][0][0]["image_id"], "Same row order and identity")
            raw = (json.dumps(change, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()
            if independent:
                need(raw == expected[i], "Every independently re-rendered mask and edit is byte-identical")
            counts.update(number)
            by_sensor[population[i][0][0]["sensor"]].update(number)
            changes.append(raw)
            if i == 0 or (i+1) % 20 == 0:
                print(json.dumps(dict(images=i+1, point_evaluations=counts["point_evaluations"])), flush=True)
    need(counts["images"] == 740 and counts["mapped_candidates"] == orig["counts"]["mapped"]
         and counts["area_eligible"] == orig["counts"]["eligible"] == counts["rendered"] + counts["degenerate"]
         and counts["area_below_32"] + counts["area_at_least_1024"] + counts["area_eligible"] == counts["mapped_candidates"]
         and counts["point_evaluations"] == 3136*counts["rendered"]
         and counts["learned_common_swaps"] == counts["point_equal_swaps"] == previous_replay["counts"]["free_natural_swaps"]
         and counts["learned_common_candidates"] == counts["point_equal_candidates"] == previous_replay["counts"]["free_natural_candidates"]
         and counts["editable_supports"] == previous_replay["counts"]["editable_supports"]
         and counts["rendered"] == counts["common_zero_swaps"] + counts["point_equal_candidates"]
         and {sensor: row["images"] for sensor, row in by_sensor.items()} == {"GF-3": 583, "TerraSAR-X": 157},
         "Full map, same eligibility, exact total replay and original occupancy budget")
    raw = b"".join(changes)
    edits_sha = digest(raw)
    need(not independent or (edits_sha == prev["edits_sha256"] and dict(counts) == prev["counts"]
                             and by_sensor == prev["counts_by_sensor"]),
         "Exact independently observed edits and count aggregates")
    output.mkdir()
    if not independent:
        (output / "EDITS_PRIVATE.jsonl").write_bytes(raw)
    return dict(status=ACCEPTED if independent else PREPARED, images=len(population), GT_opened=False, author_TEST=True,
                model_forwards=0, optimizer_updates=0, counts=dict(counts), counts_by_sensor=by_sensor,
                protocol_sha256=capture.LOCK_SHA,
                prelock_commit=capture.LOCK_COMMIT, worker_sha256=worker_sha, edits_sha256=edits_sha,
                predictions_sha256=orig["predictions_sha256"], candidate_audit_sha256=orig["audit_sha256"],
                capture_receipt_sha256=capture.CAPTURE_RECEIPT, basis_result_sha256=a.basis_result_sha,
                fit_result_sha256=a.fit_result_sha, fit_weight_sha256=fit_sha, old_edits_sha256=OLD_EDITS,
                exact_replayed_policy_choices=independent, exact_support_xor_area=True, workers=a.workers)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    p.add_argument("--authority", type=Path, required=True)
    p.add_argument("--authority-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    cfg = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and cfg.output.is_absolute() and cfg.output.name == cfg.phase and not cfg.output.exists(), "Fresh isolated CPU replay")
    install_guard()
    start = perf_counter()
    independent = cfg.phase == "verify-edits"
    a = read_authority(json.loads(pin(cfg.authority, cfg.authority_sha)), independent)
    result = run(a, cfg.output, independent)
    result.update(seconds=perf_counter()-start, input_authority_sha256=cfg.authority_sha)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (cfg.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
