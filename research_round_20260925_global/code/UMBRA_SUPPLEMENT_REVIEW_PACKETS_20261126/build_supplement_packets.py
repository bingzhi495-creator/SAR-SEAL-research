"""Build two verified, role-specific blind Umbra supplement packets."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from types import ModuleType


PRELOCK = "775451046ef9ff3e084e80f7fd0df8bbe9a5883cc22243ed0583d3cfec923d4d"
BASE_PINS = {
    "build_packets.py": "3b88c6c1c4ab070341eb8fffdc1c682a5f1520b7ff5e7decab1ee5d3037134bd",
    "annotate.html": "6935d4d23f15aa44ba7c02e0ef590f78a69b814ff133f4b632b0d70df60331ed",
    "REVIEWER_GUIDE.md": "5e0fb4d3c9b958d04bdf8ae9871daaf4dc82569d897b37f151e7fba5b2edefe8",
}
PHASES = {"intake": "native-window-intake", "readback": "native-window-readback"}
COMPLETE = "UMBRA_SUPPLEMENT_NATIVE_WINDOWS_COMPLETE"
PARTIAL = "UMBRA_SUPPLEMENT_NATIVE_WINDOWS_PARTIAL"
READBACK = "UMBRA_SUPPLEMENT_NATIVE_READBACK_COMPLETE"
HW = [1024, 1024]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "Expected SHA-256 pin")
    require(path.resolve(strict=True) == path and path.is_file(), "Expected an exact original input path")
    raw = path.read_bytes()
    require(digest(raw) == expected, "Original input bytes differ from the pinned closure")
    return raw


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Expected unique JSON keys")
        result[key] = value
    return result


def read_json(path, expected):
    return json.loads(pin(path, expected), object_pairs_hook=unique_object)


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
            "Expected an observed UTC timestamp")
    return parsed


def observe(role, lock, terminal_path, result_path, worker_path):
    require(set(lock) == {"phase", "terminal_sha256", "result_sha256", "worker_sha256"}
            and lock["phase"] == PHASES[role], "Expected exact observed stage pins")
    pin(worker_path, lock["worker_sha256"])
    terminal = read_json(terminal_path, lock["terminal_sha256"])
    allowed_exits = (0, 1) if role == "intake" else (0,)
    require(terminal["phase"] == lock["phase"] and type(terminal["actual_exit"]) is int
            and terminal["actual_exit"] in allowed_exits and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == lock["result_sha256"]
            and terminal["worker_sha256"] == lock["worker_sha256"]
            and utc(terminal["started_utc"]) < utc(terminal["ended_utc"]),
            "Expected independently witnessed closed result")
    return terminal, read_json(result_path, lock["result_sha256"])


def checked_rows(args, admission):
    require(set(admission) == {"manifest_sha256", "intake", "readback"},
            "Expected separate pinned stage closures and manifest")
    manifest_sha = admission["manifest_sha256"]
    intake_lock, readback_lock = admission["intake"], admission["readback"]
    require(intake_lock["worker_sha256"] != readback_lock["worker_sha256"], "Independent readback worker required")
    intake_terminal, intake = observe("intake", intake_lock, args.intake_terminal,
                                      args.intake / "RESULT_SAFE.json", args.intake_worker)
    readback_terminal, readback = observe("readback", readback_lock, args.readback_terminal,
                                          args.readback_result, args.readback_worker)
    require(utc(intake_terminal["ended_utc"]) <= utc(readback_terminal["started_utc"]),
            "Independent readback must start after intake has closed")
    manifest = read_json(args.manifest, manifest_sha)
    require(manifest["schema"] == "umbra_gec_supplement_v1", "Expected the supplement acquisition manifest")
    requested = len(manifest["items"])
    identifiers = [item["window_id"] for item in manifest["items"]]
    require(18 <= requested <= 24 and identifiers == [f"UMB-{i:03d}" for i in range(101, 101 + requested)],
            "Expected the exact ordered supplemental roster")
    verified = intake["verified_native_windows"]
    require(type(verified) is int and 18 <= verified <= requested, "At least eighteen unique verified windows required")
    failed = requested - verified
    expected_status = PARTIAL if failed else COMPLETE
    require(intake["manifest_sha256"] == readback["manifest_sha256"] == manifest_sha
            and intake["status"] == expected_status and intake_terminal["actual_exit"] == int(failed > 0)
            and intake["requested_windows"] == readback["requested_windows"] == requested
            and intake["failed_window_count"] == readback["failed_window_count"] == failed
            and intake["decoded_unique_windows"] == readback["readback_windows"] == readback["unique_pixel_arrays"] == verified
            and intake["native_window_shape_yx"] == readback["native_window_shape_yx"] == HW
            and intake["annotation_files"] == intake["detector_forwards"] == intake["optimizer_updates"] == 0,
            "Closed supplemental intake and planned-flow counts differ")
    require(readback["status"] == READBACK and readback["eligible_for_detector"] is True
            and readback["extraction_actual_exit"] == intake_terminal["actual_exit"]
            and readback["extraction_result_sha256"] == intake_lock["result_sha256"]
            and readback["extraction_terminal_sha256"] == intake_lock["terminal_sha256"]
            and readback["matching_uncompressed_ranges"] == 4 * verified
            and readback["source_coords_exposed"] is False
            and readback["annotations_read"] == readback["detector_forwards"] == readback["optimizer_updates"] == 0,
            "Independent readback does not bind all exact eligible input windows")

    provenance = read_json(args.intake / "PROVENANCE_PRIVATE.json", intake["provenance_sha256"])
    all_rows = provenance["results"]
    require(provenance["manifest_sha256"] == manifest_sha and [row["window_id"] for row in all_rows] == identifiers
            and all(row["status"] in ("complete", "failed") for row in all_rows),
            "The private intake outcome must preserve the full requested roster")
    rows = [row for row in all_rows if row["status"] == "complete"]
    require(len(rows) == verified and len({row["native_array_sha256"] for row in rows}) == verified
            and all(row["native_shape_yx"] == HW and row["native_dtype"] == "|u1" for row in rows),
            "Only the exact unique verified native images are eligible")
    image_dir = args.intake / "images"
    require(image_dir.resolve(strict=True) == image_dir
            and {entry.name for entry in image_dir.iterdir()} == {row["window_id"] + ".png" for row in rows},
            "Anonymous PNG roster differs from the admitted rows")
    return rows, failed


def load_original(base):
    assets = {name: pin(base / name, expected) for name, expected in BASE_PINS.items()}
    path = base / "build_packets.py"
    original = ModuleType("verified_umbra_packet_builder")
    original.__file__ = str(path)
    exec(compile(assets["build_packets.py"], str(path), "exec"), original.__dict__)
    return original, assets


def build(args):
    pin(args.prelock, PRELOCK)
    require(not args.output.exists() and args.output.parent.resolve(strict=True) == args.output.parent,
            "Use a fresh packet directory")
    require(args.intake not in (args.output.parent, *args.output.parent.parents)
            and args.base not in (args.output.parent, *args.output.parent.parents),
            "Keep packet outputs separate from original data and source")
    admission = read_json(args.admission, args.admission_sha256)
    rows, omitted = checked_rows(args, admission)
    original, assets = load_original(args.base)
    roster, images = [], {}
    for row in rows:
        identifier = row["window_id"]
        name = f"images/{identifier}.png"
        raw = pin(args.intake / name, row["png_sha256"])
        original.check_png(raw, row)
        images[name] = raw
        roster.append({"image_id": identifier, "width": 1024, "height": 1024, "png_sha256": row["png_sha256"]})
    roster_bytes = original.encode({"schema": "sar-hull-roster-v1", "images": roster})
    template = assets["annotate.html"].decode()
    markers = {
        original.MANIFEST_MARKER: roster_bytes.decode().strip(),
        '<title>SAR 船体判读</title>': '<title>Umbra 补充队列 SAR 船体判读</title>',
        '<h1>SAR 船体判读</h1>': '<h1>Umbra 补充队列 SAR 船体判读</h1>',
        'a.download="SAR_ship_review_"': 'a.download="SAR_ship_supplement_review_"',
    }
    require(template.count(original.REVIEWER_MARKER) == 1
            and all(template.count(marker) == 1 for marker in markers), "Expected pinned offline template markers")
    for marker, replacement in markers.items():
        template = template.replace(marker, replacement)
    instructions = assets["REVIEWER_GUIDE.md"] + (
        "\n补充队列请单独回传网页生成的本组 JSON。文件名为 "
        "SAR_ship_supplement_review_A.json 或 SAR_ship_supplement_review_B.json。"
        "请保留原文件和原文件名。名单里的编号按采集顺序保留，逐张完成包内全部影像即可。\n"
    ).encode()
    args.output.mkdir()
    archives = {}
    for role in ("A", "B"):
        interface = template.replace(original.REVIEWER_MARKER, f'data-reviewer="{role}"').encode()
        members = {"打开判读.html": interface, "判读说明.md": instructions, "ROSTER.json": roster_bytes, **images}
        archives[role] = original.archive(args.output / f"UMBRA_SUPPLEMENT_REVIEWER_{role}.zip", members)

    with (args.output / "ROSTER.json").open("xb") as file:
        file.write(roster_bytes)
        file.flush()
        os.fsync(file.fileno())

    result = {
        "status": "TWO_BLIND_SUPPLEMENT_REVIEW_PACKETS_VERIFIED",
        "phase": "packet-build",
        "worker_sha256": digest(Path(__file__).read_bytes()),
        "frozen_manifest_sha256": admission["manifest_sha256"],
        "intake_result_sha256": admission["intake"]["result_sha256"],
        "intake_terminal_sha256": admission["intake"]["terminal_sha256"],
        "readback_result_sha256": admission["readback"]["result_sha256"],
        "readback_terminal_sha256": admission["readback"]["terminal_sha256"],
        "prelock_sha256": PRELOCK,
        "admission_sha256": args.admission_sha256,
        "anonymous_roster_sha256": digest(roster_bytes),
        "requested_windows": len(images) + omitted,
        "windows_per_packet": len(images),
        "unavailable_windows": omitted,
        "labels_present": 0,
        "model_outputs_present": 0,
        "source_coordinates_present": 0,
        "archives": archives,
    }
    raw_result = original.encode(result)
    with (args.output / "RESULT_SAFE.json").open("xb") as file:
        file.write(raw_result)
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps({"status": result["status"], "result_sha256": digest(raw_result)}), flush=True)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("base", "prelock", "admission", "manifest", "intake", "intake-terminal", "intake-worker",
                 "readback-result", "readback-terminal", "readback-worker", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    try:
        build(parser.parse_args())
    except (ValueError, KeyError, TypeError, OSError) as error:
        if type(error) is ValueError:
            parser.exit(1, f"Supplementary packet intake failed: {error}\n")
        parser.exit(1, "Supplementary packet intake failed on schema or input access.\n")


if __name__ == "__main__":
    main()
