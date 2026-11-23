"""Bind the original RF-DETR carrier and equal-count replay to the Gulf intake."""
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

LOCK = "6001a3d9be9cf5cccb71e4156603d91c2dec1858ba46816a8380f2d315272cbb"
INTAKE = "3ae9fad2f45129347d962541184e405ed7408fb6b4926d05e8a2540aa76a8016"
NASTAR = "aa90c6116719cac85a2aaff6187a531399163fdd3ba8325d8442cd85b41b7238"
CAPTURE = "GULF_AIS_FROZEN_CAPTURE_COMPLETE"
REPLAY = "GULF_AIS_FROZEN_REPLAY_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Pinned bytes changed: " + path.name)
    return raw


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def engine_for(code, roster=None):
    path = code / "NASTAR_AIS_FROZEN_20261123/frozen_nastar.py"
    engine = ModuleType("_gulf_nastar_carrier")
    engine.__file__ = str(path)
    exec(compile(pin(path, NASTAR), str(path), "exec"), engine.__dict__)
    engine.CAPTURE = CAPTURE
    engine.PROTOCOL = LOCK
    engine.ROSTER = roster
    return engine


def observed(root, terminal_path, phase, status, worker_sha=None):
    term_bytes = terminal_path.read_bytes()
    terminal = json.loads(term_bytes)
    data = pin(root / "RESULT_SAFE.json", terminal["result_sha256"])
    result = json.loads(data)
    require(terminal["phase"] == result["phase"] == phase
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["worker_sha256"] == result["worker_sha256"]
            and (worker_sha is None or result["worker_sha256"] == worker_sha)
            and result["status"] == status and result["protocol_sha256"] == LOCK,
            "Observed worker exit and immutable result required")
    return result, digest(data), digest(term_bytes)


def load_items(a):
    from PIL import Image

    require(LOCK == a.protocol_sha256 and digest(a.protocol.read_bytes()) == LOCK, "Exact published prelock")
    intake, result_sha, terminal_sha = observed(a.intake, a.intake_terminal, "intake", "GULF_AIS_FROZEN_INTAKE_COMPLETE", INTAKE)
    require(intake["target_labels_opened"] is intake["ais_fields_opened"] is False
            and intake["detector_forwards"] == intake["optimizer_updates"] == 0,
            "Image-only admission is required")
    manifest = json.loads(pin(a.intake / "INTAKE_PRIVATE.json", intake["intake_sha256"]))
    require(all(manifest[k] == intake[k] for k in ("protocol_sha256", "roster_sha256", "census_result_sha256")),
            "Independent study roster and intake provenance")
    items, seen_scenes = [], set()
    selections = manifest["items"]
    require(len(selections) == intake["counts"]["selected"] <= 256 and len(selections) > 0, "Complete selected roster")
    for ordinal, row in enumerate(selections, 1):
        image_id = row["image_id"]
        require(image_id == ordinal and isinstance(row["modeled"], bool) and row["status"] in ("ready", "failed")
                and row["modeled"] == (row["status"] == "ready") and re.fullmatch(r"[0-9]{6}-[0-9]{8}", row["pass_id"]),
                "Retain ordered success or failure of every chosen chip")
        if not row["modeled"]:
            continue
        scene = row["scene_id"]
        name = f"{image_id:06d}.png"
        require(row["image_path"] == "images/" + name and row["archive_path"] == f"archives/{image_id:06d}.npz",
                "Canonical original chip paths")
        geometry = manifest["scene_geometry"][scene]
        require(geometry["status"] == "ready" and geometry["range_pixel_spacing_m"] == geometry["azimuth_pixel_spacing_m"] == 10.0
                and min(geometry["width"], geometry["height"]) >= 512
                and geometry["xml_path"] == f"geometry/{scene}.xml", "Source SAR native rectangular grid")
        if scene not in seen_scenes:
            pin(a.intake / geometry["xml_path"], geometry["xml_sha256"])
            seen_scenes.add(scene)
        raw = pin(a.intake / row["image_path"], row["png_sha256"])
        with Image.open(BytesIO(raw)) as image:
            require(image.format == "PNG" and image.mode == "L" and image.size == (512, 512)
                    and digest(image.tobytes()) == row["gray_sha256"]
                    and digest(image.convert("RGB").tobytes()) == row["rgb_sha256"], "Bound native VV display bytes")
        items.append(dict(image_id=image_id, source_image_name=name, acquisition_group=row["pass_id"], sensor="Sentinel-1A",
                          original_hw=[512, 512], rgb_sha256=row["rgb_sha256"],
                          image_ref=dict(path=row["image_path"], sha256=row["png_sha256"], bytes=len(raw))))
    require(0 < len(items) == intake["counts"]["ready"], "All and only admitted chip rasters")
    require(intake["counts"]["failed"] == len(selections)-len(items), "Full admission denominator")
    a.intake_sha256 = intake["intake_sha256"]
    engine = engine_for(a.code, intake["roster_sha256"])
    return SimpleNamespace(manifest=manifest, items=items, roster_sha256=intake["roster_sha256"],
                           intake_result_sha256=result_sha, intake_terminal_sha256=terminal_sha,
                           intake_sha256=a.intake_sha256, operator=engine.portable_source(a.code), engine=engine)


def verify_capture(a, state):
    worker_sha = digest(Path(__file__).read_bytes())
    capture, h, terminal_sha = observed(a.capture, a.capture_terminal, "capture", CAPTURE, worker_sha)
    require(capture["intake_result_sha256"] == state.intake_result_sha256
            and capture["intake_terminal_sha256"] == state.intake_terminal_sha256
            and capture["lock_sha256"] == state.roster_sha256
            and capture["target_labels_opened"] is False and capture["optimizer_updates"] == 0,
            "Bound observed capture and label-free input")
    source, sha, population = state.engine.verify_capture(a, state.items, state.operator)
    require(source == capture and h == sha, "Exact independently reconstructed original capture")
    return capture, h, population


def initialize(a, items):
    global worker
    worker = engine_for(a.code).portable_source(a.code)
    worker.initialize_replay(a.code, a.intake, items, a.capture, a.point_head)


def replay_image(population):
    return worker.replay_both(population)


def replay(a, state):
    cap, cap_sha, population = verify_capture(a, state)
    a.output.mkdir()
    total, delivery = Counter(), sha256()
    with (a.output / state.operator.EDITS).open("xb") as out, ProcessPoolExecutor(
            max_workers=a.workers, initializer=initialize, initargs=(a, state.items)) as pool:
        for idx, (row, counts) in enumerate(pool.map(replay_image, population, chunksize=2), 1):
            require(row["image_id"] == state.items[idx-1]["image_id"], "Exact frozen writeback order")
            line = encode(row)
            require(out.write(line) == len(line), "Complete frozen edit block")
            delivery.update(line)
            total.update(counts)
            if idx == 1 or idx % 20 == 0 or idx == len(population):
                print(json.dumps(dict(phase="replay", images=idx)), flush=True)
        out.flush()
        os.fsync(out.fileno())
    require(total["mapped_candidates"] == cap["counts"]["mapped"] and total["images"] == len(population)
            and total["fixed_swaps"] == total["free_matched_count_swaps"], "Verified complete signed equal-count replay")
    return dict(capture_result_sha256=cap_sha, predictions_sha256=cap["predictions_sha256"], audit_sha256=cap["audit_sha256"],
                edits_sha256=delivery.hexdigest(), counts=dict(total), independent_decisions_equal=True,
                point_head_sha256=state.operator.POINT_HEAD, model_forward_calls=0, detector_forwards=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    phases = parser.add_subparsers(dest="phase", required=True)
    for name in ("capture", "replay"):
        command = phases.add_parser(name)
        for key in ("code", "intake", "intake-terminal", "protocol", "output"):
            command.add_argument("--" + key, type=Path, required=True)
        command.add_argument("--protocol-sha256", required=True)
        if name == "capture":
            command.add_argument("--checkpoint", type=Path, required=True)
        else:
            for key in ("capture", "capture-terminal", "point-head"):
                command.add_argument("--" + key, type=Path, required=True)
            command.add_argument("--workers", type=int, choices=range(1, 9), default=5)
    a = parser.parse_args()
    for key, item in vars(a).items():
        if isinstance(item, Path):
            setattr(a, key, item.resolve(strict=key != "output"))
    require(not a.output.exists() and sys.version_info[:3] == (3, 10, 13), "Fresh output and qualified interpreter")
    if a.phase == "replay":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    start = perf_counter()
    state = load_items(a)
    result = state.engine.capture(a, state.items, state.operator) if a.phase == "capture" else replay(a, state)
    result.update(status=CAPTURE if a.phase == "capture" else REPLAY, phase=a.phase, images=len(state.items),
                  manifest_sha256=state.intake_sha256, intake_result_sha256=state.intake_result_sha256,
                  intake_terminal_sha256=state.intake_terminal_sha256, lock_sha256=state.roster_sha256,
                  protocol_sha256=LOCK, checkpoint_sha256=state.operator.CHECKPOINT, target_labels_opened=False,
                  ais_fields_opened=False, optimizer_updates=0, worker_sha256=digest(Path(__file__).read_bytes()),
                  wall_seconds=perf_counter()-start)
    if a.phase == "capture":
        result["detector_forwards"] = result["model_forward_calls"]
    raw = encode(result)
    with (a.output / "RESULT_SAFE.json").open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
