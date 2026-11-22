"""Render the frozen same-carrier PointRend comparison twice on the 74 parents."""

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

PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
TRANSFER = "INSTANCE_RSDD_POINTREND_TRANSFER_20261122"
REPLAY_OP = "9368956a7d47e502c005a6adc1ab39edaca18cd5b3780002073ba672732a2b2f"
ORIGINAL_BASIS = "476a8ff4a9c705df3fa6530fca2c4f58d020dc7995b444480476eeac1e98b8a1"
HEAD = "7401de08367ef31b909ea9c92daa27be02d4efb04460acbf96eab33648fdd7cc"
OCCUPANCY = "353362e0efabc0374eaaf0b9c27fdabbcad195402a1090230c4ef56ca2a10398"
LOCK = "4cdad2535417233eca5a959201ee8fabc0cc37a10d3e81baa03f594d0646fc82"
STATUS = "INSTANCE_RSDD_RESERVE74_POINTREND_REPLAY_ACCEPTED_V1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"Changed input {path.name}")
    return raw


def source(path, expected):
    module = ModuleType("_reserve_replay_" + path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def initialize(args):
    global render_image
    original = source(args.code / TRANSFER / "replay_pointrend.py", REPLAY_OP)
    original.initialize(args, HEAD)
    render_image = original.replay_image


def render(work):
    return render_image(work)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_population(a):
    pin(a.protocol, LOCK)
    pin(a.fit_weights, HEAD)
    portable = source(a.code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", PORTABLE)
    native = portable.module(a.code / portable.NATIVE / "capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    items, manifest = portable.load_intake(a, native)
    cap, predictions = portable.read_capture(a, items, manifest)
    cap_sha = digest((a.capture / portable.RESULT).read_bytes())
    lineage = {k: cap[k] for k in ("roster_sha256", "manifest_sha256", "checkpoint_sha256",
                                  "predictions_sha256", "audit_sha256")}
    old = json.loads((a.occupancy / portable.RESULT).read_bytes())
    require(old["status"] == "B5_RESERVE74_REPLAY_COMPLETE" and old["edits_sha256"] == OCCUPANCY
            and old["independent_decisions_equal"] is True and old["target_labels_opened"] is False
            and old["point_head_sha256"] == portable.POINT_HEAD
            and all(old[key] == value for key, value in lineage.items()), "Accepted original occupancy replay required")
    original = [json.loads(line) for line in pin(a.occupancy / portable.EDITS, OCCUPANCY).splitlines()]
    basis_raw = (a.basis / "RESULT_SAFE.json").read_bytes()
    basis_sha = digest(basis_raw)
    basis = json.loads(basis_raw)
    terminal_raw = a.basis_terminal.read_bytes()
    terminal = json.loads(terminal_raw)
    require(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == basis_sha and terminal["worker_sha256"] == basis["worker_sha256"],
            "Observed external feature-capture exit and footer")
    require(basis["status"] == "INSTANCE_RSDD_RESERVE74_POINTREND_BASIS_CAPTURED_V1"
            and basis["capture_receipt_sha256"] == cap_sha and basis["images"] == basis["model_forwards"] == 74
            and basis["optimizer_updates"] == 0 and basis["counts"] == cap["counts"] and basis["GT_opened"] is False
            and all(basis[key] == value for key, value in lineage.items())
            and all(basis[key] is True for key in ("exact_original_candidates_and_masks",
                "exact_original_eligible_logits", "exact_all_query_basis_contraction")),
            "Fresh detector features must match the original reserve capture")
    index = json.loads(pin(a.basis / "INDEX_PRIVATE.json", basis["index_sha256"]))
    for ordinal, (item, idx, before, (audit, stock)) in enumerate(zip(items, index, original, predictions, strict=True), 1):
        require(item["image_id"] == idx["image_id"] == before["image_id"] == audit["image_id"] == stock["image_id"]
                and item["rgb_sha256"] == idx["rgb_sha256"] and idx["ordinal"] == ordinal
                and idx["eligible_rows"] == len(audit["logit_ranks"])
                and idx["spatial"]["path"] == f"spatial/{ordinal:06d}.npy", "Exact ordered reserve basis and candidate")
    return list(zip(predictions, index, original, strict=True)), cap, old, dict(
        lineage, capture_result_sha256=cap_sha, basis_result_sha256=basis_sha,
        basis_terminal_sha256=digest(terminal_raw))


def render_all(a, population):
    rows, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a,)) as pool:
        for ordinal, (edit, tally) in enumerate(pool.map(render, population, chunksize=2), 1):
            require(edit["image_id"] == population[ordinal-1][0][0]["image_id"], "Render order or image changed")
            rows.append(encode(edit))
            counts.update(tally)
    return b"".join(rows), counts


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("intake", "roster", "capture", "basis", "basis_terminal", "occupancy", "fit_weights", "protocol", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    a = p.parse_args()
    for name, path in vars(a).items():
        if isinstance(path, Path):
            setattr(a, name, path.resolve(strict=name != "output"))
    require(not a.output.exists(), "Fresh output directory required")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    start = perf_counter()
    population, native, old, lineage = read_population(a)
    a.basis_worker = a.code / TRANSFER / "capture_basis.py"
    a.basis_worker_sha = ORIGINAL_BASIS
    data, counts = render_all(a, population)
    print(json.dumps(dict(phase="first_render", images=counts["images"],
                          point_evaluations=counts["point_evaluations"])), flush=True)
    repeated, checked = render_all(a, population)
    require(data == repeated and counts == checked, "Independent render outputs differ")
    require(counts["images"] == 74 and counts["mapped_candidates"] == native["counts"]["mapped"]
            and counts["area_eligible"] == native["counts"]["eligible"] == counts["rendered"] + counts["degenerate"]
            and counts["editable_supports"] == old["counts"]["editable_supports"]
            and counts["rendered"] == counts["common_zero_swaps"] + counts["point_equal_candidates"]
            and counts["point_evaluations"] == 3136 * counts["rendered"]
            and counts["learned_common_candidates"] == counts["point_equal_candidates"] == old["counts"]["free_natural_candidates"]
            and counts["learned_common_swaps"] == counts["point_equal_swaps"] == old["counts"]["free_natural_swaps"],
            "Native population and equal per-candidate occupancy budgets")
    a.output.mkdir()
    (a.output / "EDITS_PRIVATE.jsonl").write_bytes(data)
    result = dict(lineage, status=STATUS, images=counts["images"], counts=dict(counts), edits_sha256=digest(data),
                  fit_weight_sha256=HEAD, occupancy_edits_sha256=OCCUPANCY, protocol_sha256=LOCK,
                  exact_independent_render=True, same_endpoints_exact_realized_adds_drops=True,
                  GT_opened=False, model_forwards=0, optimizer_updates=0, AP=False,
                  worker_sha256=digest(Path(__file__).read_bytes()), wall_seconds=perf_counter()-start)
    safe = encode(result)
    (a.output / "RESULT_SAFE.json").write_bytes(safe)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(safe))), flush=True)


if __name__ == "__main__":
    main()
