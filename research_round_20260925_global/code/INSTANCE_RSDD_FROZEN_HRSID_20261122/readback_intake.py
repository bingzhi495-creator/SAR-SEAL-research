"""Verify completed Instance-RSDD JPEGs with the qualified RF-DETR decoder."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
from time import perf_counter
import zlib

from rfdetr.datasets.io_utils import decode_pil_image_bytes


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(data):
    return sha256(data).hexdigest()


def load(path, expected):
    raw = path.read_bytes()
    check(digest(raw) == expected, "Unexpected input fingerprint: " + path.name)
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake", "terminal", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    for name in ("intake-sha", "terminal-sha"):
        parser.add_argument("--"+name, required=True)
    a = parser.parse_args()
    check(not a.output.exists(), "Fresh readback output required")
    start = perf_counter()
    original = load(a.intake / "RESULT_SAFE.json", a.intake_sha)
    observed = load(a.terminal, a.terminal_sha)
    check(original["status"] == "INSTANCE_RSDD_TEST740_IMAGE_INTAKE_COMPLETE_V1"
          and observed["actual_exit"] == 0 and observed["output_footer_verified"] is True
          and observed["result_sha256"] == a.intake_sha, "Original intake actual exit or output identity differs")
    manifest_sha = original["image_manifest_sha256"]
    manifest = load(a.intake / "IMAGE_MANIFEST_PRIVATE.json", manifest_sha)
    items = manifest["items"]
    check(len(items) == original["test_images"] == 740 and manifest["expected_ids"] == [x["image_id"] for x in items],
          "Author exact ID order differs")
    filenames, encoded, decoded = set(), set(), set()
    sensors, groups = Counter(), {}
    for item in items:
        name = item["source_image_name"]
        ref = item["image_ref"]
        check(Path(name).name == name and ref["path"] == "images/" + name and name not in filenames,
              "Unbound or duplicate source JPEG path")
        filenames.add(name)
        raw = (a.intake / ref["path"]).read_bytes()
        encoded_hash = digest(raw)
        check(len(raw) == ref["bytes"] and encoded_hash == ref["sha256"] and f"{zlib.crc32(raw):08x}" == item["zip_crc32"],
              "Original compressed archive member identity differs")
        im, scales = decode_pil_image_bytes(raw, draft_size=None)
        rgb_hash = digest(im.tobytes())
        check(im.mode == "RGB" and list(reversed(im.size)) == item["original_hw"] == [512,512]
              and scales == (1.0,1.0) and rgb_hash == item["rgb_sha256"], "Qualified RF decoder differs from admitted RGB")
        encoded.add(encoded_hash)
        decoded.add(rgb_hash)
        sensor = item["sensor"]
        sensors[sensor] += 1
        group = item["acquisition_group"]
        check(groups.setdefault(group,sensor) == sensor, "Acquisition group mixes source sensors")
    check(len(encoded) == original["encoded_unique"] == 740 and len(decoded) == original["rgb_unique"] == 740
          and len(groups) == original["acquisition_groups"] == 112 and dict(sensors) == original["sensor_counts"],
          "Aggregate admission fingerprints differ")
    check({p.name for p in (a.intake/"images").iterdir()} == filenames, "Original image directory set differs")
    record = dict(status="INSTANCE_RSDD_TEST740_IMAGE_INTAKE_READBACK_V1",test_images=740,
                  original_intake_sha256=a.intake_sha,image_manifest_sha256=manifest_sha,
                  exact_qualified_rf_rgb=740,encoded_unique=740,rgb_unique=740,sensor_counts=dict(sensors),
                  acquisition_groups_by_sensor=dict(Counter(groups.values())), acquisition_groups=len(groups),
                  model_forwards=0,optimizer_updates=0,target_polygon_rows_opened=0,seconds=perf_counter()-start)
    output = (json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    a.output.mkdir()
    with (a.output/"RESULT_SAFE.json").open("xb") as f:
        f.write(output)
    print(json.dumps(dict(status=record["status"],result_sha256=digest(output))),flush=True)


if __name__ == "__main__":
    main()
