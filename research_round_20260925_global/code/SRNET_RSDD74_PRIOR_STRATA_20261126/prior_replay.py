"""Read out prior-only and residual-only ranks on the closed SRNet reserve."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch


REPLAY_SHA = "c7f5146b0901c9158ff63d719b3caf05b66272bd4f10ee331514cfcb84d93628"
PREPARE_AUTHORITY_SHA = "ca571b9960078ba9e117a17c130b2539180bcbf467c69bdc3400c4cbfbe595d9"
PREPARE_SHA = "30d186181bce6dbb5eafaf3563d7e67ac9376a8273bd6bc1e12e63e6aac4f64a"
PREPARE_TERMINAL_SHA = "a040c5c4b065b74b7f00f50c7e42ca592370eb29940e397c245da5f687acd088"
VERIFY_SHA = "e3b9404583925d16ab55428e9a80a527d7004513d462deca7b9dee7cf9280c29"
VERIFY_TERMINAL_SHA = "74d4c6e1c13132d138e7456d6f389d15328fabb6c5615dac68ecbc9dcfa8ae80"
STATUS = "SRNET_RSDD74_DIRECT_PRIOR_ABLATION_RENDER_COMPLETE"
P, R = "prior_only", "residual_only"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    from hashlib import sha256
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact digest: " + path.name)
    return raw


def posterior_arms(model, features, center, scale):
    standard = torch.from_numpy(np.asarray((features - center) / scale, dtype=np.float32))
    probability = torch.from_numpy(features[:, :, 0].copy())
    with torch.inference_mode():
        prior = torch.logit(probability.clamp(1e-4, 1 - 1e-4))
        residual = model.network(torch.cat((standard, standard), dim=-1)).squeeze(-1)
        combined = prior + residual
        require(torch.equal(combined, model(standard, standard, probability)), "Original full head arithmetic")
        return tuple(x.sigmoid().numpy() for x in (prior, residual, combined))


def old_replay(root):
    import types
    source = root / "replay.py"
    replay = types.ModuleType("closed_srnet_replay")
    replay.__file__ = str(source)
    exec(compile(pin(source, REPLAY_SHA), str(source), "exec"), replay.__dict__)
    return replay


def namespaces(raw):
    return SimpleNamespace(**{k: Path(v) if isinstance(v, str) and Path(v).is_absolute() else v for k, v in raw.items()})


def restore(base, methods, a):
    source = json.loads(base.pin(a.fit_result, base.FIT_SHA))
    closure = json.loads(base.pin(a.fit_accepted, base.FIT_ACCEPTED_SHA))
    record = methods.readout.validated_refits(source, closure, base.FIT_SHA)[5217]
    step = base.read_source(a.code / base.SPOT / "fit_step.py", methods.readout.PINS["fit_step.py"])
    plan = base.read_source(a.code / base.RF / "fit_plan.py", methods.readout.PINS["fit_plan.py"])
    trainer = step.load_trainer(a.code / base.BASE / "fit109_paired_point/train_point.py")
    require(a.weights.name == record["relative_weights"], "Locked 5217 all-source refit")
    heads, center, scale = plan.restore_bundle(a.weights, record["weights_sha256"], trainer, step)
    return heads["S"], center, scale


def render_image(a, row, replay, base, methods, point, sampler, head, expected):
    original = replay.native_rows(row)
    probabilities = replay.native_probabilities(a, row)
    rgb = base.load_rgb(a.manifest.parent / "images", row)
    image = methods.pair.image_features(rgb)
    changes, as_recorded = {}, {}
    stats = Counter(images=1, candidates=len(original))
    for cand, native, prob in zip(row["candidates"], original, probabilities, strict=True):
        stock = replay.codec.decode(native["segmentation"]).astype(bool)
        require(np.array_equal(prob > .5, stock), "Author threshold parity")
        area = int(stock.sum())
        if not 32 <= area < 1024:
            continue
        box = np.asarray(cand["clipped_corner_xyxy"], dtype=np.float32)
        require(box.shape == (4,) and np.isfinite(box).all() and np.all((box >= 0) & (box <= 512))
                and np.all(box[2:] >= box[:2]), "Original support box")
        if np.any(box[2:] == box[:2]):
            continue
        data = point.paired_points(stock, box, area, native["score"], prob, image, sampler)
        coords = data[0] if data is not None else np.empty((0, 2), dtype=np.int32)
        base.check_support(methods, stock, box, prob, coords)
        if data is None:
            continue
        features = data[1]
        require(features.dtype == np.float32 and features.shape == (len(coords), 2, 14)
                and np.isfinite(features).all(), "Original endpoint features")
        p, r, q = posterior_arms(head[0], features, head[1], head[2])
        original_options = replay.render_pair(base, methods, stock, coords, q)
        if not original_options:
            continue
        rank = str(cand["rank"])
        as_recorded[rank] = original_options
        k = original_options[replay.F]["swaps"]
        cap = min(8, area // 10, len(coords))
        plans = {key: methods.free.choose_free(x, cap, fixed_count=k) for key, x in ((P, p), (R, r))}
        options = {key: base.mask_edit(stock, coords, action) for key, action in plans.items()}
        base.check_edits(stock, coords, {replay.F: options[P], replay.I: options[R]})
        raw = features[:, :, 0]
        stats.update(positive_candidates=1, exchanges=k, endpoints=raw.size,
                     clipped_lower_adds=int((raw[:, 0] < 1e-4).sum()),
                     clipped_upper_drops=int((raw[:, 1] > 1-1e-4).sum()),
                     prior_differs_from_full=int(options[P] != original_options[replay.I]),
                     residual_differs_from_full=int(options[R] != original_options[replay.I]))
        changes[rank] = options
    require(replay.encoded(dict(image_id=row["image_id"], changes=as_recorded)) == expected,
            "Byte-exact original fixed/full independent replay including zero budgets")
    return dict(image_id=row["image_id"], changes=changes), stats


def run(cmd):
    started = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
            and torch.__version__ == "2.10.0+cu130" and not torch.cuda.is_initialized() and not cmd.output.exists(),
            "Fresh qualified CPU diagnostic")
    pin(cmd.protocol, cmd.protocol_sha256)
    require(re.fullmatch("[0-9a-f]{40}", cmd.prelock_commit), "Verified published commit format")
    replay = old_replay(cmd.previous)
    a = namespaces(json.loads(pin(cmd.previous / "AUTHORITY_prepare-edits.json", PREPARE_AUTHORITY_SHA)))
    replay.pin(a.protocol, a.protocol_sha)
    prepare = replay.closed(cmd.previous / "observed_prepare-edits/TERMINAL.json", PREPARE_TERMINAL_SHA,
                            cmd.previous / "prepare-edits/RESULT_SAFE.json", PREPARE_SHA, "prepare-edits", REPLAY_SHA)
    verify = replay.closed(cmd.previous / "observed_verify-edits/TERMINAL.json", VERIFY_TERMINAL_SHA,
                           cmd.previous / "verify-edits/RESULT_SAFE.json", VERIFY_SHA, "verify-edits", REPLAY_SHA)
    require(verify["exact_independent_replay"] is True and verify["prepare_result_sha256"] == PREPARE_SHA
            and verify["prepare_terminal_sha256"] == PREPARE_TERMINAL_SHA
            and verify["edits_sha256"] == prepare["edits_sha256"] and verify["counts"] == prepare["counts"]
            and verify["fit_weights_sha256"] == replay.WEIGHTS_SHA
            and verify["label_files_opened"] == prepare["label_files_opened"] == 0,
            "Original independent edit closure")
    expected = replay.pin(cmd.previous / "prepare-edits/EDITS_PRIVATE.jsonl", prepare["edits_sha256"]).splitlines(keepends=True)
    capture, index = replay.checked_capture(a)
    require(len(index) == len(expected) == 74 and capture["counts"]["candidates"] == 89, "Whole original count")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base, methods = replay.edit_modules(a.code)
    head = restore(base, methods, a)
    point, sampler = base.canvas(methods, (512, 512))
    rows, counts = [], Counter()
    for row, archived in zip(index, expected, strict=True):
        new, stats = render_image(a, row, replay, base, methods, point, sampler, head, archived)
        rows.append(replay.encoded(new))
        counts.update(stats)
    require(counts["images"] == 74 and counts["candidates"] == 89 and
            counts["positive_candidates"] == prepare["counts"]["positive_matched_budget_candidates"] and
            counts["exchanges"] == prepare["counts"]["exact_adds_per_policy"], "Full original signed budget flow")
    payload = b"".join(rows)
    cmd.output.mkdir()
    (cmd.output / "EDITS_PRIVATE.jsonl").write_bytes(payload)
    result = dict(status=STATUS, worker_sha256=replay.digest(Path(__file__).read_bytes()),
                  protocol_sha256=cmd.protocol_sha256, prelock_commit=cmd.prelock_commit,
                  original_prepare_sha256=PREPARE_SHA, original_verify_sha256=VERIFY_SHA,
                  original_edit_stream_sha256=prepare["edits_sha256"], fit_weights_sha256=replay.WEIGHTS_SHA,
                  original_fixed_and_full_edit_bytes_exact=True, original_signed_k_and_support_exact=True,
                  label_files_opened=0, detector_forwards=0, optimizer_updates=0, seed=5217,
                  edits_sha256=replay.digest(payload), counts=dict(counts), elapsed_seconds=perf_counter()-started)
    (cmd.output / "RESULT_SAFE.json").write_bytes(replay.encoded(result))
    print(json.dumps(dict(status=STATUS, result_sha256=replay.digest(replay.encoded(result)))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--prelock-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

