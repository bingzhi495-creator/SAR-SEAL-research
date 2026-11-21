"""Independently check the prelocked extracted PSeg-SSDD TEST232 inputs on CPU."""

import collections
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zlib

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from shapely.geometry import LinearRing


BASE = Path(__file__).resolve().parent
PRELOCK = "3dbc4e540678349bb57b4d80702f993c2fa69ce26dad769fadcada90748b4b89"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def verify():
    check(digest((BASE / "PRELOCK.md").read_bytes()) == PRELOCK, "prelock identity")
    src = BASE / "PRIVATE_INPUTS"
    raw_json = (src / "test.json").read_bytes()
    check(len(raw_json) == 397010 and zlib.crc32(raw_json) == 0xdacc1bcd, "original annotation record")
    data = json.loads(raw_json)
    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    check(len(coco.anns) == len(data["annotations"]), "unique annotation IDs")
    check(len(coco.imgs) == len(data["images"]), "unique image IDs")
    names = {n["id"]: PurePosixPath(n["file_name"]).name for n in data["images"]}
    expected = {f"{i:06d}.jpg" for i in range(1, 1161) if i % 10 in (1, 9)}
    check(set(names.values()) == expected and len(names) == len(expected) == 232, "official JPEG roster")
    manifest, image_digests = {}, set()
    widths, heights = [], []
    modes = collections.Counter()
    for image_id, name in names.items():
        payload = (src / name).read_bytes()
        sha = digest(payload)
        image_digests.add(sha)
        manifest[name] = f"{name}\t{len(payload)}\t{sha}\n"
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            metadata = coco.imgs[image_id]
            check(image.format == "JPEG" and image.size == (metadata["width"], metadata["height"]),
                  "JPEG decode and COCO dimensions")
            modes[image.mode] += 1
            widths.append(image.width)
            heights.append(image.height)
    lines = "".join(manifest[name] for name in sorted(manifest)).encode("ascii")
    check(lines == (BASE / "IMAGE_IDENTITIES_PRIVATE.tsv").read_bytes(), "exact private JPEG manifest")

    counts = collections.Counter()
    has_ann, has_raster, has_geometry = set(), set(), set()
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        check(image_id in coco.imgs and ann["category_id"] in coco.cats, "valid annotation references")
        h, w = coco.imgs[image_id]["height"], coco.imgs[image_id]["width"]
        has_ann.add(image_id)
        counts["crowd"] += bool(ann.get("iscrowd", False))
        counts["ignored"] += bool(ann.get("ignore", False))
        counts["declared_area_nonpositive_or_missing"] += not (ann.get("area", 0) > 0)
        rings = ann["segmentation"]
        check(isinstance(rings, list) and rings, "nonempty original polygon list required")
        ann_fractional, ann_valid = False, True
        for ring in rings:
            check(isinstance(ring, list) and len(ring) >= 6 and len(ring) % 2 == 0,
                  "malformed original polygon ring")
            check(all(type(c) in (int, float) and math.isfinite(c) for c in ring), "finite native vertices")
            xy = np.array(ring, dtype=np.float64).reshape(-1, 2)
            frac = xy != np.trunc(xy)
            outside = np.any((xy < 0) | (xy > (w, h)), axis=1)
            x, y = xy.T
            twice_area = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
            valid = LinearRing(xy).is_valid and LinearRing(xy).is_simple
            counts["polygon_rings"] += 1
            counts["coordinate_values"] += len(ring)
            counts["coordinate_values_json_float"] += sum(type(c) is float for c in ring)
            counts["vertices"] += len(xy)
            counts["nonintegral_coordinate_values"] += int(frac.sum())
            counts["nonintegral_vertices"] += int(frac.any(axis=1).sum())
            counts["polygons_with_nonintegral_vertex"] += bool(frac.any())
            counts["out_of_canvas_vertices"] += int(outside.sum())
            counts["out_of_canvas_rings"] += bool(outside.any())
            counts["zero_shoelace_area_rings"] += twice_area == 0
            counts["invalid_shapely_rings"] += not valid
            ann_fractional |= bool(frac.any())
            ann_valid &= bool(valid and twice_area != 0 and not outside.any())
        counts["annotations_with_nonintegral_vertex"] += ann_fractional
        counts["annotations_with_invalid_or_outside_geometry"] += not ann_valid
        mask = coco.annToMask(ann)
        check(mask.ndim == 2 and mask.shape == (h, w), "native COCO mask")
        occupied = bool(np.any(mask))
        counts["empty_masks"] += not occupied
        counts["nonempty_masks"] += occupied
        if occupied:
            has_raster.add(image_id)
            if ann_valid:
                has_geometry.add(image_id)
    actual = {key: int(v) for key, v in counts.items()}
    actual.update(json_bytes=len(raw_json), json_crc32=f"{zlib.crc32(raw_json):08x}", json_sha256=digest(raw_json),
                  images=len(names), annotations=len(coco.anns), original_category_ids=sorted(coco.cats),
                  image_inventory_sha256=digest(lines), with_annotation=len(has_ann), with_nonempty_raster=len(has_raster),
                  images_with_nonempty_valid_in_bounds_geometry=len(has_geometry), images_fully_jpeg_decoded=len(names),
                  unique_encoded_images=len(image_digests), image_modes=dict(modes),
                  width_range=[min(widths), max(widths)], height_range=[min(heights), max(heights)],
                  malformed_annotations=0, malformed_rings=0, raster_failures=0, prelock_sha256=PRELOCK)
    evidence = (BASE / "LOCAL_SAFE.json").read_bytes()
    primary = json.loads(evidence)
    check(all(primary.get(k) == v for k, v in actual.items()), "independent aggregate differs from local audit")
    output = dict(status="PSEG_SSDD_TEST232_INDEPENDENT_INPUT_READBACK_PASSED",
                  **actual, schema_fields_recomputed=len(actual), byte_identical_image_manifest=True,
                  local_safe_sha256=digest(evidence), model_forwards=0, AP_computed=False,
                  independent_source_sha256=digest(Path(__file__).read_bytes()))
    raw = (json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    with (BASE / "INDEPENDENT_SAFE.json").open("xb") as dest:
        dest.write(raw)
    print(json.dumps({"status": output["status"], "result_sha256": digest(raw),
                      "verified_fields": len(actual), "images": len(names), "annotations": len(coco.anns)}))


if __name__ == "__main__":
    verify()
