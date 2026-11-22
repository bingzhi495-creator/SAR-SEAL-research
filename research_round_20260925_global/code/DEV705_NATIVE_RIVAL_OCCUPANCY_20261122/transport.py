"""Apply the FIT109 rival-mask rule to an immutable D-FINE DEV705 replay."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np


E24 = Path("DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24")
E = E24 / "dev705_free_endpoint"
HELPER_SHA = "a6f927db39800f2a251eee265032e6cb8a4b8c5318c1b77ef6c3417432a4a4d1"
FIT_WORKER_SHA = "21ac90236e1451fb501d5ebe8d32942e0fb6eb5c0425bf91361d25f2e6b89ba9"
FIT_RESULT_SHA = "ae044680ae52192dd4ed14dc5beb2902dab5e63f1dd76f95de1ecd69e475aac5"
SIDE_SHA = "277965bef4230ec7fadac30e103095af3ce68f99d1b69f74a8e348c391cef035"
OLD_REPLAY_SHA = "878d2a791da39d0941a9ea57067c8c3141c524fd0d690a0cadf01888946d46a3"
OLD_ACCEPT_SHA = "c1753af526c63b7b3f37825127e8a3cb82baf90bb3d584b51a9b2bdeee5d1f7d"
OLD_SCORE_SHA = "e32f2e6f45fb2e1a777765a57b021877e0803588285c6868aa2a915c5c6542e8"
CAPTURE = "DFINE_DEV705_RIVAL_FROZEN_CAPTURE_COMPLETE"
READBACK = "DFINE_DEV705_RIVAL_FROZEN_READBACK_ACCEPTED"
STREAM = "RIVAL_EDITS_PRIVATE.jsonl"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Unexpected original digest: " + path.name)
    return raw


def module(path, digest):
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(pinned(path, digest), str(path), "exec"), m.__dict__)
    return m


def sources(a):
    old = module(a.code / E / "dev_endpoint.py", "6681d3903199214cd109efd9a8bf92681d838901e6c9c8780e4f19f7e16703b7")
    score = module(a.code / E / "score_endpoint.py", "77159ea493a95973fe422dac1d303eaeba603e6d8af7ae179c72aecc2e4301b0")
    dev = module(a.code / E24 / "dev705/dfine_dev705.py", old.DEV_SOURCE)
    rival = module(a.rival, HELPER_SHA)
    fit = module(a.fit_worker, FIT_WORKER_SHA)
    return SimpleNamespace(old=old, score=score, dev=dev, rival=rival, fit=fit,
                           arms=(fit.TRUE, fit.PLACEBO), control=fit.INDEPENDENT)


def inputs(a, m):
    m.dev.cpu_runtime()
    b = SimpleNamespace()
    local = json.loads(a.inputs.read_bytes())
    for name in ("archive", "dev_root", "sidecar", "sidecar_accepted", "refit_result", "refit_acceptance",
                 "weights", "replay", "replay_accepted", "shapely_site", "fit_result"):
        setattr(b, name, Path(local[name]))
    for name, rel in {
        "refit": "fit109_final_refit/frozen_refit.py", "common": "fit109_gray_equal_count/equal_count.py",
        "gray": "fit109_gray_ablation/gray_ablation.py", "pair_math": "fit109_paired_point/pair_math.py",
        "rank_math": "fit109_probability_rank/rank_math.py", "free": "fit109_free_pairing/free_pairing.py",
        "training": "fit109_paired_point/train_point.py",
    }.items():
        setattr(b, name, a.code / E24 / rel)
    b.replay_sha, b.sidecar_accepted_sha = OLD_REPLAY_SHA, SIDE_SHA
    fit = json.loads(pinned(b.fit_result, FIT_RESULT_SHA))
    need(fit["status"] == "FIT109_NATIVE_RIVAL_OCCUPANCY_ACTUAL" and fit["near"]["advances"] is True
         and fit["historical_parity"] is True, "Original FIT-source gate")
    receipt = m.old.probability_receipt(b, m.dev)
    manifest = m.dev.read_manifest(b.dev_root)
    old = m.old.require_prelabel(b)
    check = json.loads(pinned(b.replay_accepted, OLD_ACCEPT_SHA))
    need(check["status"] == m.old.VERIFIED and check["original_exit"] == 0
         and check["edits_sha256"] == old["edits_sha256"] and check["replay_sha256"] == OLD_REPLAY_SHA
         and check["sidecar_accepted_sha256"] == SIDE_SHA, "Original closed label-free readback")
    return b, manifest, receipt, old


def select(m, scene, rank, coordinates, p, cap, matched, free, image_id):
    values = scene.penalties(rank - 1, coordinates[:, 0])
    placebo = m.rival.placebo_penalties(p[:, 0], values, image_id, rank)
    need(np.array_equal(np.sort(values), np.sort(placebo)), "Exact penalty multiset")
    arms = {arm: m.rival.rank_additions(p, signal, cap, len(matched), free.choose_free)
            for arm, signal in zip(m.arms, (values, placebo), strict=True)}
    removed = [j for _, j in matched]
    need(all([j for _, j in picks] == removed for picks in arms.values()), "Identical exact drop ordinals")
    control, real, fake = ({i for i, _ in picks} for picks in (matched, *arms.values()))
    return arms, dict(native_rival=bool(np.any(values)), count=len(matched), true_changed=real != control,
                      placebo_changed=fake != control, placebo_displaced=not np.array_equal(values, placebo),
                      true_placebo_changed=real != fake)


def image_replay(a, m):
    path = a.code / E / "dev_endpoint.py"
    text = pinned(path, m.score.REPLAY_SOURCE).decode("utf-8").replace("\r\n", "\n")
    body = text[text.index("def image_edits("):text.index("\n\ndef prelabel(")]
    replacements = {
        "    result={}\n": "    result={}\n    coverage={}\n    scene=rival_module.NativeRivals(coco)\n",
        "        changes={}\n": (
            '        if matched:\n'
            '            extra,state=choose(scene,i,coordinates,p,cap,matched,free,row["image_id"])\n'
            '            actions.update(extra)\n'
            '            coverage[str(i)]=state\n'
            '        changes={}\n'
        ),
        "    return result": "    return result,coverage",
    }
    for before, after in replacements.items():
        need(body.count(before) == 1, "Expected exactly one seam in original readout")
        body = body.replace(before, after)
    context = dict(m.old.__dict__, rival_module=m.rival, choose=lambda *args: select(m, *args))
    exec(compile(body, str(path), "exec"), context)
    return context["image_edits"]


def verify_image(m, item, coco, audit, original, row):
    need(set(row) == {"image_id", "changes", "coverage"} and row["image_id"] == item["image_id"],
         "Original image identity")
    changes = row["changes"]
    old = {rank: {arm: v for arm, v in arms.items() if arm not in m.arms} for rank, arms in changes.items()}
    need(old == original, "Bit-identical historical arm masks and counts")
    edited = {rank for rank, arms in original.items() if m.control in arms}
    need(set(row["coverage"]) == edited and all(set(changes[rank]).intersection(m.arms)
         == (set(m.arms) if rank in edited else set()) for rank in changes), "Matched candidate budgets including zero")
    codec = m.old.codec
    for rank in edited:
        i = int(rank) - 1
        base = coco[i]["segmentation"]
        anchor = original[rank][m.control]
        k, area = anchor["swaps"], audit[i]["mask_area"]
        state = row["coverage"][rank]
        flags = ("native_rival", "true_changed", "placebo_changed", "placebo_displaced", "true_placebo_changed")
        need(set(state) == {*flags, "count"} and state["count"] == k and all(type(state[k]) is bool for k in flags),
             "Exact coverage witness")
        for arm, flag in zip(m.arms, ("true_changed", "placebo_changed"), strict=True):
            change = changes[rank][arm]
            m.old.validate_edit(base, change["rle"], area, k)
            kept = codec.merge([base, anchor["rle"], change["rle"]], intersect=True)
            need(change["swaps"] == k and int(codec.area(kept)) == area-k
                 and state[flag] == (change["rle"] != anchor["rle"]),
                 "Exact XOR and identical removed native pixels")
        need(state["true_placebo_changed"] == (changes[rank][m.arms[0]]["rle"] != changes[rank][m.arms[1]]["rle"]),
             "Exact changed mask identity")


def emit(a, m, result):
    digest = m.dev.write_new(a.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest)), flush=True)


def capture(a, m, b, manifest, receipt, original):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = perf_counter()
    counts, digest = Counter(), sha256()
    models = m.old.restore(b)
    make_edits = image_replay(a, m)
    historic = m.old.replay_rows(b, m.dev, manifest, receipt, original)
    inputs = m.old.source_blocks(b, m.dev, manifest, receipt, True)
    a.output.mkdir()
    with b.archive.open("rb") as f, (a.output / STREAM).open("xb") as out:
        need(b.archive.stat().st_size == m.dev.ARCHIVE_BYTES, "Original PNG archive size")
        f.seek(m.dev.CENTRAL_OFFSET)
        need(sha256(f.read(m.dev.CENTRAL_BYTES)).hexdigest() == m.dev.CENTRAL_SHA, "Original archive directory")
        with ZipFile(f) as archive:
            for j, ((item, coco, audit, planes), old) in enumerate(zip(inputs, historic, strict=True), 1):
                need((item, coco, audit) == old[:3], "Original exact candidate order")
                rgb, shape = m.dev.read_png(archive, item)
                need(shape == (800, 800), "Original native canvas")
                changes, coverage = make_edits(coco, audit, planes, models[0].image_features(rgb), models, counts)
                row = dict(image_id=item["image_id"], changes=changes, coverage=coverage)
                verify_image(m, item, coco, audit, old[3], row)
                line = m.dev.encoded(row)
                out.write(line)
                digest.update(line)
                if j % 25 == 0 or j == 705:
                    print(json.dumps(dict(event="DFINE_DEV705_RIVAL_CAPTURE_PROGRESS", images=j)), flush=True)
        out.flush()
        os.fsync(out.fileno())
    need(j == 705 and all(counts[k] == original["counts"][k] for k in original["counts"]),
         "Original support and budget tallies")
    emit(a, m, dict(status=CAPTURE, protocol_sha256=a.protocol_sha, images=705, candidates=70500,
                    gt_opened=False, detector_forwards=0, fit_result_sha256=FIT_RESULT_SHA,
                    worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    edits_sha256=digest.hexdigest(), counts=dict(counts), wall_seconds=perf_counter()-start))


def captured_rows(a, m, b, manifest, receipt, original, observed):
    digest = sha256()
    with (a.capture / STREAM).open("rb") as f:
        for item, coco, audit, old in m.old.replay_rows(b, m.dev, manifest, receipt, original):
            line = f.readline(4*1024**2)
            need(line.endswith(b"\n"), "Complete captured line")
            digest.update(line)
            row = json.loads(line)
            verify_image(m, item, coco, audit, old, row)
            yield item, coco, audit, row["changes"], row["coverage"]
        need(not f.read(1) and digest.hexdigest() == observed["edits_sha256"], "Exact closed label-free stream")


def terminal(path, expected):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    need(data["original_exit"] == 0 and data["result_sha256"] == expected, "Observed original process exit")
    return sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "capture", "readback", "score"))
    for name in ("code", "rival", "fit-worker", "protocol"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--protocol-sha", required=True)
    for name in ("inputs", "output", "capture", "capture-terminal", "readback", "readback-terminal", "reference",
                 "score-source"):
        p.add_argument("--" + name, type=Path)
    for name in ("capture-sha", "readback-sha", "score-source-sha"):
        p.add_argument("--" + name)
    a = p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only")
    pinned(a.protocol, a.protocol_sha)
    m = sources(a)
    image_replay(a, m)
    if a.mode == "check":
        print(json.dumps(dict(status="DFINE_DEV705_RIVAL_IMPORTS_ACCEPTED", closed_inputs_opened=0)))
        return
    need(a.output is not None and not a.output.exists(), "Fresh result directory")
    b, manifest, receipt, original = inputs(a, m)
    if a.mode == "capture":
        capture(a, m, b, manifest, receipt, original)
        return
    observed = json.loads(pinned(a.capture / "RESULT_SAFE.json", a.capture_sha))
    cap_term = terminal(a.capture_terminal, a.capture_sha)
    need(observed["status"] == CAPTURE and observed["protocol_sha256"] == a.protocol_sha
         and observed["worker_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest()
         and observed["gt_opened"] is False and observed["images"] == 705
         and observed["candidates"] == 70500, "Original exact closed capture")
    rows = captured_rows(a, m, b, manifest, receipt, original, observed)
    if a.mode == "readback":
        counts = Counter()
        for _, _, _, changes, coverage in rows:
            counts["images"] += 1
            for rank in coverage:
                for arm in m.arms:
                    counts[arm + "_edited"] += 1
                    counts[arm + "_swaps"] += changes[rank][arm]["swaps"]
        need(counts["images"] == 705 and all(counts[k] == observed["counts"][k] for k in counts if k != "images"),
             "Independent count parity for the new arms")
        a.output.mkdir()
        emit(a, m, dict(status=READBACK, protocol_sha256=a.protocol_sha, gt_opened=False, counts=dict(counts),
                        capture_sha256=a.capture_sha, capture_terminal_sha256=cap_term,
                        edits_sha256=observed["edits_sha256"]))
        return
    verified = json.loads(pinned(a.readback / "RESULT_SAFE.json", a.readback_sha))
    terminal(a.readback_terminal, a.readback_sha)
    need(verified["status"] == READBACK and verified["capture_sha256"] == a.capture_sha
         and verified["capture_terminal_sha256"] == cap_term and verified["edits_sha256"] == observed["edits_sha256"]
         and verified["gt_opened"] is False and verified["protocol_sha256"] == a.protocol_sha,
         "Separate label-free close and verification required")
    score = module(a.score_source, a.score_source_sha)
    rows = list(rows)
    need(len(rows) == 705, "Exact frozen evaluation population")
    result = score.evaluate(a, m, b, rows, observed, module, pinned, OLD_SCORE_SHA)
    a.output.mkdir()
    emit(a, m, result)


if __name__ == "__main__":
    main()
