"""Frozen SRNet point replay with native pixel-footprint priors."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import torch


PROTOCOL_SHA = "708c4183d61ffb36c2d3274a30eef5a429e4943d22fa98f8595bc610200dc026"
PRELOCK_COMMIT = "fbed6a496256e18640bb7c809877534b225dd9bc"
PRIOR_REPLAY_SHA = "7644c685452a71fbb2bd9ecdfc170b1a0c1825a566cdbc41196a59e86d4d440d"
CAPTURE_STATUS = "SRNET_RSDD74_AUTHOR_PREGRID_CAPTURE_COMPLETE"
CAPTURE_VERIFIED = "SRNET_RSDD74_AUTHOR_PREGRID_READBACK_COMPLETE"
INTEGRATOR_SHA = "eebf26addd6e37d31ad9a464cd66256367ea2478b3da6ceeb1c58672b063f6e5"
PREPARED = "SRNET_RSDD74_PIXEL_INTEGRAL_EDITS_PREPARED"
VERIFIED = "SRNET_RSDD74_PIXEL_INTEGRAL_EDITS_VERIFIED"
A, S = "source_pixel_integral", "native_smoothing"
CONTRASTS = ("source_vs_original", "smoothed_vs_original", "source_vs_smoothed")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Exact digest: " + path.name)
    return raw


def load(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def posteriors(head, features, source_average, native_average):
    model, center, scale = head
    standard = torch.from_numpy(np.asarray((features - center) / scale, dtype=np.float32))
    native = torch.from_numpy(features[:, :, 0].copy())
    with torch.inference_mode():
        residual = model.network(torch.cat((standard, standard), dim=-1)).squeeze(-1)
        original = torch.logit(native.clamp(1e-4, 1-1e-4)) + residual
        require(torch.equal(original, model(standard, standard, native)), "Original frozen arithmetic")
        output = [original.sigmoid().numpy()]
        for value in (source_average, native_average):
            require(value.shape == native.shape and value.dtype == np.float32 and np.isfinite(value).all()
                    and np.all((-1e-6 <= value) & (value <= 1+1e-6)), "Finite native FP32 probability prior")
            term = torch.logit(torch.from_numpy(value.copy()).clamp(1e-4, 1-1e-4))
            output.append((term + residual).sigmoid().numpy())
    return output


def checked_pregrid_capture(c, replay, a):
    capture = replay.closed(c.capture / "observed_capture/TERMINAL.json", c.capture_terminal_sha,
                            c.capture / "actual/RESULT_SAFE.json", c.capture_result_sha, "capture", c.capture_worker_sha)
    verified = replay.closed(c.capture / "observed_readback/TERMINAL.json", c.capture_readback_terminal_sha,
                             c.capture / "readback/RESULT_SAFE.json", c.capture_readback_result_sha,
                             "verify", c.capture_worker_sha)
    require(capture["status"] == CAPTURE_STATUS and verified["status"] == CAPTURE_VERIFIED
            and verified["capture_result_sha256"] == c.capture_result_sha
            and verified["capture_terminal_sha256"] == c.capture_terminal_sha
            and verified["index_sha256"] == capture["index_sha256"]
            and capture["weights_sha256"] == replay.SRNET_WEIGHTS_SHA
            and capture["model_forwards"] == 74 and verified["model_forwards"] == 0
            and verified["all_original_stock_fields_exact"] is True
            and verified["counts"]["images"] == 74
            and verified["counts"]["candidates"] == verified["counts"]["exact_native_f32_candidates"] == 89,
            "Exact observed pre-final canvas and independent native FP32 readback")
    for proof in (capture, verified):
        require(proof["protocol_sha256"] == c.protocol_sha and proof["prelock_commit"] == c.prelock_commit
                and proof["all_original_native_probability_f32_bits_exact"] is True
                and proof["original_capture_result_sha256"] == a.capture_result_sha
                and proof["original_capture_terminal_sha256"] == a.capture_terminal_sha
                and proof["image_manifest_sha256"] == replay.MANIFEST_SHA
                and proof["reference_files_opened"] == proof["optimizer_updates"] == 0
                and proof["ground_truth_opened"] is False and proof["AP"] is False,
                "Exact already accepted legacy native candidate lineage and label-free capture")
    return capture


def admit(c, prior, replay):
    previous = c.previous
    a = prior.namespaces(json.loads(pin(previous / "AUTHORITY_prepare-edits.json", prior.PREPARE_AUTHORITY_SHA)))
    replay.pin(a.protocol, a.protocol_sha)
    prepare = replay.closed(previous / "observed_prepare-edits/TERMINAL.json", prior.PREPARE_TERMINAL_SHA,
                            previous / "prepare-edits/RESULT_SAFE.json", prior.PREPARE_SHA, "prepare-edits", prior.REPLAY_SHA)
    verify = replay.closed(previous / "observed_verify-edits/TERMINAL.json", prior.VERIFY_TERMINAL_SHA,
                           previous / "verify-edits/RESULT_SAFE.json", prior.VERIFY_SHA, "verify-edits", prior.REPLAY_SHA)
    require(verify["exact_independent_replay"] and verify["prepare_result_sha256"] == prior.PREPARE_SHA
            and verify["prepare_terminal_sha256"] == prior.PREPARE_TERMINAL_SHA
            and verify["edits_sha256"] == prepare["edits_sha256"] and verify["counts"] == prepare["counts"]
            and verify["fit_weights_sha256"] == replay.WEIGHTS_SHA
            and verify["label_files_opened"] == prepare["label_files_opened"] == 0, "Original closed F/I")
    legacy, stock_rows = replay.checked_capture(a)
    capture = checked_pregrid_capture(c, replay, a)
    index = [json.loads(row) for row in pin(c.capture / "actual/INDEX_PRIVATE.jsonl", capture["index_sha256"]).splitlines()]
    prior_edits = pin(previous / "prepare-edits/EDITS_PRIVATE.jsonl", prepare["edits_sha256"]).splitlines(keepends=True)
    require(len(stock_rows) == len(index) == len(prior_edits) == 74 and legacy["counts"]["candidates"] == 89,
            "Original image and candidate denominators")
    extras = {"pregrid_file", "resized_hw", "padded_hw", "native_hw", "final_resize_called"}
    for old, new in zip(stock_rows, index, strict=True):
        require(extras <= new.keys() and all(new[key] == value for key, value in old.items())
                and new["native_hw"] == old["original_hw"] == [512, 512]
                and new["final_resize_called"] == bool(old["candidates"]),
                "Whole original metadata parity and native/pre-final mapping")
    return a, prepare, capture, stock_rows, index, prior_edits


def load_pregrid(root, row):
    ref = row["pregrid_file"]
    name = f"PREGRID_PRIVATE/{row['ordinal']:03d}.npz"
    require(ref["path"] == name, "Expected capture-local pregrid name")
    raw = pin(root / "actual" / name, ref["sha256"])
    require(ref["bytes"] == len(raw), "Exact pregrid compressed length")
    with np.load(BytesIO(raw), allow_pickle=False) as bits:
        require(bits.files == ["pregrid"], "Only original pre-final grid")
        plane = bits["pregrid"]
    require(plane.dtype == np.float32 and plane.shape == (len(row["candidates"]), *row["resized_hw"])
            and np.isfinite(plane).all() and np.all((-1e-6 <= plane) & (plane <= 1+1e-6)),
            "Exact native pregrid FP32 dimensions and values")
    return plane


def render_image(c, a, row, new, expected, replay, base, methods, point, sampler, head, integrate):
    native = replay.native_rows(row)
    probabilities = replay.native_probabilities(a, row)
    pregrids = load_pregrid(c.capture, new)
    rgb = base.load_rgb(a.manifest.parent / "images", row)
    image = methods.pair.image_features(rgb)
    choices, original = {}, {}
    counts = Counter(images=1, candidates=len(native), no_candidate_images=int(not native))
    sums = [0., 0., 0.]
    maxima = [0., 0., 0.]
    for cand, original_row, prob, pre in zip(row["candidates"], native, probabilities, pregrids, strict=True):
        stock = replay.codec.decode(original_row["segmentation"]).astype(bool)
        require(np.array_equal(prob > .5, stock), "Author strict threshold at original native grid")
        area = int(stock.sum())
        if not 32 <= area < 1024:
            continue
        box = np.asarray(cand["clipped_corner_xyxy"], dtype=np.float32)
        data = None if np.any(box[2:] == box[:2]) else point.paired_points(
            stock, box, area, cand["score"], prob, image, sampler)
        coords = data[0] if data is not None else np.empty((0,2), dtype=np.int32)
        base.check_support(methods, stock, box, prob, coords)
        if data is None:
            continue
        features = data[1]
        p = features[:, :, 0]
        require(features.dtype == np.float32 and features.shape == (len(coords), 2, 14)
                and np.isfinite(features).all() and np.array_equal(p, prob.ravel()[coords]),
                "All 14 original features and sampled prior")
        pa = integrate(pre, (512,512), coords)
        ps = integrate(prob, (512,512), coords)
        q_original, q_integral, q_smoothed = posteriors(head, features, pa, ps)
        observed = replay.render_pair(base, methods, stock, coords, q_original)
        counts["point_head_evaluations"] += 1
        if not observed:
            continue
        rank = str(cand["rank"])
        original[rank] = observed
        k = observed[replay.F]["swaps"]
        cap = min(8, area // 10, len(coords))
        opts = {key: base.mask_edit(stock, coords, methods.free.choose_free(q, cap, fixed_count=k))
                for key, q in ((A, q_integral), (S, q_smoothed))}
        require(opts[A]["swaps"] == opts[S]["swaps"] == k, "Same signed k for both footprint interventions")
        base.check_edits(stock, coords, {replay.F:opts[A], replay.I:opts[S]})
        changes = (opts[A] != observed[replay.I], opts[S] != observed[replay.I], opts[A] != opts[S])
        counts["source_prior_crosses_stock_half"] += int(np.count_nonzero((pa > .5) != (p > .5)))
        counts["smoothing_prior_crosses_stock_half"] += int(np.count_nonzero((ps > .5) != (p > .5)))
        pairs = ((pa, p), (ps, p), (pa, ps))
        for n, (name, (x, y), change) in enumerate(zip(CONTRASTS, pairs, changes, strict=True)):
            delta = np.abs(x.astype(np.float64) - y)
            counts[name + "_changed_candidates"] += int(change)
            counts[name + "_different_prior_endpoints"] += int(np.count_nonzero(x != y))
            sums[n] += float(delta.sum())
            maxima[n] = max(maxima[n], float(delta.max()))
        counts.update(positive_candidates=1, signed_swaps=k, positive_candidate_endpoint_entries=p.size)
        choices[rank] = opts
    require(replay.encoded(dict(image_id=row["image_id"], changes=original)) == expected,
            "Recomputed original F/I ledger exactly, with original score, rank, mask and zero budgets")
    return dict(image_id=row["image_id"], changes=choices), counts, sums, maxima


def run(c):
    start = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
            and torch.__version__ == "2.10.0+cu130" and not torch.cuda.is_initialized() and not c.output.exists(),
            "Fresh qualified CPU and no training")
    pin(c.protocol, c.protocol_sha)
    require(c.prelock_commit == PRELOCK_COMMIT and c.protocol_sha == PROTOCOL_SHA, "Exact verified preregistration")
    prior = load(c.prior_replay, PRIOR_REPLAY_SHA)
    require(c.integrator_sha == INTEGRATOR_SHA, "Pinned mathematical integrator")
    integrator = load(c.integrator, c.integrator_sha)
    replay = prior.old_replay(c.previous)
    a, prepare, capture, original, pregrids, expected = admit(c, prior, replay)
    worker = digest(Path(__file__).read_bytes())
    verification = c.phase == "verify-edits"
    provisional = None
    if verification:
        require(c.prepare and c.prepare_result_sha and c.prepare_terminal and c.prepare_terminal_sha,
                "Exact first replay authority")
        provisional = replay.closed(c.prepare_terminal, c.prepare_terminal_sha, c.prepare / "RESULT_SAFE.json",
                                    c.prepare_result_sha, "prepare-edits", worker)
        require(provisional["status"] == PREPARED and provisional["capture_result_sha256"] == c.capture_result_sha
                and provisional["integrator_sha256"] == c.integrator_sha
                and provisional["protocol_sha256"] == c.protocol_sha and provisional["label_files_opened"] == 0,
                "Original observed label-free run")
        prior_bytes = pin(c.prepare / "EDITS_PRIVATE.jsonl", provisional["edits_sha256"]).splitlines(keepends=True)
        require(len(prior_bytes) == 74, "Closed full first edit ledger")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base, methods = replay.edit_modules(a.code)
    head = prior.restore(base, methods, a)
    point, sampler = base.canvas(methods, (512, 512))
    pieces, totals = [], Counter()
    sums, maximum = np.zeros(3), np.zeros(3)
    for j, (row, mapped, archived) in enumerate(zip(original, pregrids, expected, strict=True)):
        item, cnt, abs_changes, max_changes = render_image(
            c, a, row, mapped, archived, replay, base, methods, point, sampler, head, integrator.pixel_cell_average)
        raw = replay.encoded(item)
        require(not verification or raw == prior_bytes[j], "Independent reproduction of candidate edit bytes")
        pieces.append(raw)
        totals.update(cnt)
        sums += abs_changes
        maximum = np.maximum(maximum, max_changes)
    require(totals["images"] == 74 and totals["candidates"] == 89
            and totals["positive_candidates"] == prepare["counts"]["positive_matched_budget_candidates"]
            and totals["signed_swaps"] == prepare["counts"]["exact_adds_per_policy"], "Exact full signed budgets")
    hist = {name: dict(mean_absolute=float(sums[j] / max(1, totals["positive_candidate_endpoint_entries"])),
                       max_absolute=float(maximum[j])) for j, name in enumerate(CONTRASTS)}
    payload = b"".join(pieces)
    hash_edits = digest(payload)
    if verification:
        require(hash_edits == provisional["edits_sha256"] and dict(totals) == provisional["counts"]
                and hist == provisional["prior_absolute_differences"], "Independent geometry, readback and quantities")
    c.output.mkdir()
    if not verification:
        (c.output / "EDITS_PRIVATE.jsonl").write_bytes(payload)
    result = dict(status=VERIFIED if verification else PREPARED, worker_sha256=worker,
                  protocol_sha256=c.protocol_sha, prelock_commit=c.prelock_commit,
                  capture_result_sha256=c.capture_result_sha, capture_terminal_sha256=c.capture_terminal_sha,
                  capture_readback_result_sha256=c.capture_readback_result_sha,
                  capture_readback_terminal_sha256=c.capture_readback_terminal_sha,
                  capture_index_sha256=capture["index_sha256"], integrator_sha256=c.integrator_sha,
                  original_prepare_sha256=prior.PREPARE_SHA, original_verify_sha256=prior.VERIFY_SHA,
                  original_edit_stream_sha256=prepare["edits_sha256"], fit_weights_sha256=replay.WEIGHTS_SHA,
                  original_fixed_and_full_edit_bytes_exact=True, original_signed_k_and_support_exact=True,
                  residual_and_features_unchanged=True, native_and_pregrid_fp32=True,
                  label_files_opened=0, detector_forwards=0, optimizer_updates=0, seed=5217,
                  edits_sha256=hash_edits, counts=dict(totals), prior_absolute_differences=hist,
                  exact_independent_replay=verification, elapsed_seconds=perf_counter()-start)
    if verification:
        result.update(prepare_result_sha256=c.prepare_result_sha, prepare_terminal_sha256=c.prepare_terminal_sha)
    raw = replay.encoded(result)
    (c.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(raw))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=("prepare-edits", "verify-edits"))
    for key in ("previous", "capture", "protocol", "prior-replay", "integrator", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("protocol-sha", "prelock-commit", "integrator-sha", "capture-result-sha", "capture-terminal-sha", "capture-worker-sha",
                "capture-readback-result-sha", "capture-readback-terminal-sha"):
        p.add_argument("--" + key, required=True)
    for key in ("prepare", "prepare-terminal"):
        p.add_argument("--" + key, type=Path)
    for key in ("prepare-result-sha", "prepare-terminal-sha"):
        p.add_argument("--" + key)
    run(p.parse_args())


if __name__ == "__main__":
    main()

