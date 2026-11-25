"""Image-only SAR-SSP adapter to existing exact D-FINE author inference."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

CAPTURE_SOURCE = "11df18cb1eb0dbcff66b576d0c082b45896e5e254adb66b98fdfaff366d7391a"
MANIFEST = "16c8e6b7655e51f208c3a9708acd419a60126e55a9075cdea7d4b9e4edbb9425"
PACKAGE = "23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5"


def load(path, checksum, name):
    path = Path(path)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != checksum:
        raise ValueError(f"Source digest mismatch: {path.name}")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def validate_manifest(path, claimed_sha, base):
    base.need(claimed_sha == MANIFEST, "Expected closed target image-only intake")
    data = json.loads(base.pinned(path, MANIFEST, 1024**2))
    rows = data["images"]
    base.need(data["archive_sha256"] == PACKAGE and len(rows) == 232,
              "Expected author cohort and archive identity")
    names = [i["source_image_name"] for i in rows]
    base.need(names == sorted(set(names)), "Sorted unique author JPEGs")
    out = []
    for row, filename in zip(rows, names, strict=True):
        ref = row["image_ref"]
        base.need(len(filename) == 16 and filename.endswith(".jpg")
                  and filename[:12].isascii() and filename[:12].isdigit()
                  and int(filename[:12]) == row["image_id"]
                  and row["original_hw"] == [500, 500]
                  and ref["path"] == f"images/{filename}"
                  and base.digest(ref["sha256"]) and base.digest(row["rgb_sha256"])
                  and type(ref["bytes"]) is int and 0 < ref["bytes"] < 16 * 1024**2,
                  "Expected SAR-SSP native image record")
        out.append(dict(image_id=row["image_id"], original_hw=row["original_hw"],
                        image_ref=dict(ref, path=filename), rgb_sha256=row["rgb_sha256"]))
    return out


def main():
    source = sys.argv.pop(1)
    original = load(source, CAPTURE_SOURCE, "_dfine_psg_capture")
    read_rgb = original.read_rgb

    def sar_rgb(root, row, base):
        image = read_rgb(root, row, base)
        base.need(hashlib.sha256(image.tobytes(order="C")).hexdigest() == row["rgb_sha256"],
                  "Exact original SAR-SSP decoded RGB")
        return image

    original.manifest_rows = validate_manifest
    original.read_rgb = sar_rgb
    original.INVENTORY_SHA = MANIFEST
    original.STATUS = "DFINE_HRSID_E24_SAR_SSP_VAL232_PROBABILITY_EXPORT_COMPLETE_V1"
    original.main()


if __name__ == "__main__":
    main()
