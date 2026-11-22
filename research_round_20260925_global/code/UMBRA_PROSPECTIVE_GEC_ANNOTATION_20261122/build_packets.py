"""Create two isolated human review packages from a closed native-window intake."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image

MANIFEST_MARKER = '{"schema":"sar-hull-roster-v1","images":[]}'
REVIEWER_MARKER = 'data-reviewer="A"'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Identity: " + path.name)
    return raw


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()


def check_png(raw, item):
    require(digest(raw) == item["png_sha256"], "PNG integrity")
    with Image.open(BytesIO(raw)) as img:
        img.load()
        require(img.format == "PNG" and img.mode == "L" and img.size == (1024, 1024) and not img.info,
                "Lossless native grayscale PNG without source metadata")
        require(digest(img.tobytes()) == item["native_array_sha256"], "Exact source-grid pixels")


def read_intake(intake, terminal, result_sha, manifest_sha):
    observed = json.loads(terminal.read_bytes())
    require(observed["actual_exit"] == 0 and observed["output_footer_verified"] is True
            and observed["result_sha256"] == result_sha, "Actual native-window closure")
    safe = json.loads(pinned(intake / "RESULT_SAFE.json", result_sha))
    require(safe["manifest_sha256"] == manifest_sha, "Frozen cohort")
    provenance = json.loads(pinned(intake / "PROVENANCE_PRIVATE.json", safe["provenance_sha256"]))
    rows = provenance["results"]
    require(provenance["manifest_sha256"] == manifest_sha and len(rows) == safe["verified_native_windows"]
            == safe["decoded_unique_windows"] and 8 <= len(rows) <= 12, "Complete ordered cohort within the preregistered range")
    require([item["window_id"] for item in rows] == [f"UMB-{i:03d}" for i in range(1, len(rows) + 1)],
            "Exactly the fixed IDs in cohort order")
    images = {}
    roster = []
    for item in rows:
        require(item["status"] == "complete" and item["native_shape_yx"] == [1024, 1024]
                and item["native_dtype"] == "|u1", "Verified native resolution")
        image_id = item["window_id"]
        name = f"images/{image_id}.png"
        raw = (intake / name).read_bytes()
        check_png(raw, item)
        images[name] = raw
        roster.append({"image_id": image_id, "width": 1024, "height": 1024, "png_sha256": item["png_sha256"]})
    return {"schema": "sar-hull-roster-v1", "images": roster}, images


def archive(path, members):
    with ZipFile(path, "x", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name, blob in members.items():
            item = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            z.writestr(item, blob)
    with ZipFile(path, "r") as z:
        require(z.namelist() == list(members) and z.testzip() is None, "Archive roster and CRC")
        require(all(z.read(name) == blob for name, blob in members.items()), "Archive exact readback")
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path.read_bytes()),
            "members": len(members)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake", "terminal", "template", "instructions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("result-sha256", "manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(all(re.fullmatch(r"[0-9a-f]{64}", v) for v in (args.result_sha256, args.manifest_sha256)), "Expected SHA references")
    roster, images = read_intake(args.intake, args.terminal, args.result_sha256, args.manifest_sha256)
    roster_json = encode(roster)
    template = args.template.read_text(encoding="utf-8")
    require(template.count(MANIFEST_MARKER) == template.count(REVIEWER_MARKER) == 1, "Ordinary reviewer template")
    template = template.replace(MANIFEST_MARKER, roster_json.decode().strip())
    instructions = args.instructions.read_bytes()
    args.output.mkdir()
    results = {}
    for role in ("A", "B"):
        app = template.replace(REVIEWER_MARKER, f'data-reviewer="{role}"').encode()
        members = {"打开判读.html": app, "判读说明.md": instructions, "ROSTER.json": roster_json, **images}
        results[role] = archive(args.output / f"UMBRA_SAR_REVIEWER_{role}.zip", members)
    result = {"status": "TWO_BLIND_REVIEW_PACKETS_VERIFIED", "frozen_manifest_sha256": args.manifest_sha256,
              "verified_source_result_sha256": args.result_sha256, "anonymous_roster_sha256": digest(roster_json),
              "windows_per_packet": len(images), "labels_present": 0, "model_outputs_present": 0,
              "archives": results}
    with (args.output / "RESULT_SAFE.json").open("xb") as f:
        f.write(encode(result))
    print(digest(encode(result)), flush=True)


if __name__ == "__main__":
    main()

