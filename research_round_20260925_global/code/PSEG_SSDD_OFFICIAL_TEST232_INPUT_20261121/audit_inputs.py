"""Qualify only the prelocked official COCO test232 image and annotation members."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
from zlib import crc32

import numpy as np
from PIL import Image
from pycocotools import mask as mask_codec
from shapely.geometry import Polygon


LOCK = "3dbc4e540678349bb57b4d80702f993c2fa69ce26dad769fadcada90748b4b89"
PREFIX = "Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style/"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def select_inputs(archive, unrar, directory, names):
    require(archive.stat().st_size == 1043241197, "wrong size local archive")
    stat = archive.stat()
    original = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    listing = subprocess.check_output([str(unrar), "lb", str(archive)])
    members = [item.replace(b"\\", b"/") for item in listing.splitlines()]
    paths = [PREFIX+"annotations/test.json"] + [PREFIX+"images/test/"+name for name in names]
    image_prefix = (PREFIX+"images/test/").encode()
    selected = [item for item in members if item.startswith(image_prefix) and item.lower().endswith(b".jpg")]
    require(sorted(selected) == sorted(s.encode() for s in paths[1:]), "unexpected archived TEST image population")
    require(members.count(paths[0].encode()) == 1, "expected exactly one combined TEST COCO member")
    directory.mkdir()
    cmd = [str(unrar), "e", "-idq", "-y", "-o-", str(archive)]
    cmd.extend(s.replace("/", "\\") for s in paths)
    cmd.append(str(directory)+"\\")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
    require({p.name for p in directory.iterdir()} == set(names) | {"test.json"}, "extracted member mismatch")
    stat = archive.stat()
    require(original == (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns), "archive changed")


def check_images(coco, names, directory):
    by_name = {im["file_name"]: im for im in coco["images"]}
    require(len(by_name) == len(coco["images"]) == 232 and set(by_name) == set(names), "COCO image names differ")
    by_id, lines, dims, modes = {}, [], [], Counter()
    for name in names:
        raw = (directory/name).read_bytes()
        im = by_name[name]
        image_id = im["id"]
        require(type(image_id) is int and image_id not in by_id, "duplicate/non-integer image ID")
        with Image.open(BytesIO(raw)) as pic:
            require(pic.format == "JPEG" and pic.size == (im["width"], im["height"]), "JPEG and COCO canvas mismatch")
            pic.load()
            modes[pic.mode] += 1
        dims.append((im["width"], im["height"]))
        by_id[image_id] = im
        lines.append(f"{name}\t{len(raw)}\t{sha256(raw).hexdigest()}\n")
    listing = "".join(lines).encode("ascii")
    with (directory.parent/"IMAGE_IDENTITIES_PRIVATE.tsv").open("xb") as stream:
        stream.write(listing)
    widths, heights = zip(*dims)
    return by_id, dict(images=len(names), image_inventory_sha256=sha256(listing).hexdigest(),
                       width_range=[min(widths), max(widths)], height_range=[min(heights), max(heights)],
                       image_modes=dict(modes), images_fully_jpeg_decoded=len(names),
                       unique_encoded_images=len({line.split("\t")[2] for line in lines}))


def check_annotations(coco, by_id):
    classes = {row["id"] for row in coco["categories"]}
    require(len(classes) == len(coco["categories"]) == 1 and
            coco["categories"][0]["name"].strip().lower() == "ship", "expected a single declared ship class")
    ids, positive, valid_geom, annotated = set(), set(), set(), set()
    counts = Counter()
    for row in coco["annotations"]:
        aid, iid = row["id"], row["image_id"]
        require(type(aid) is int and aid not in ids and iid in by_id and row["category_id"] in classes,
                "annotation ID, image_id or class mismatch")
        ids.add(aid)
        annotated.add(iid)
        counts["crowd"] += bool(row.get("iscrowd", 0))
        counts["ignored"] += bool(row.get("ignore", 0))
        counts["declared_area_nonpositive_or_missing"] += not (isinstance(row.get("area"), (int, float))
                                                               and row["area"] > 0)
        h, w = by_id[iid]["height"], by_id[iid]["width"]
        polys = row.get("segmentation")
        if not isinstance(polys, list) or not polys:
            counts["malformed_annotations"] += 1
            continue
        malformed = unsuitable = fractional = False
        for ring in polys:
            if (not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in ring)):
                malformed = True
                counts["malformed_rings"] += 1
                continue
            arr = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
            fract = arr != np.trunc(arr)
            counts["coordinate_values"] += arr.size
            counts["coordinate_values_json_float"] += sum(isinstance(v, float) for v in ring)
            counts["vertices"] += len(arr)
            counts["polygon_rings"] += 1
            counts["nonintegral_coordinate_values"] += int(fract.sum())
            counts["nonintegral_vertices"] += int(fract.any(axis=1).sum())
            counts["polygons_with_nonintegral_vertex"] += bool(fract.any())
            fractional |= bool(fract.any())
            outside = ((arr < 0) | (arr > [w, h])).any(axis=1)
            counts["out_of_canvas_vertices"] += int(outside.sum())
            counts["out_of_canvas_rings"] += bool(outside.any())
            poly = Polygon(arr)
            counts["invalid_shapely_rings"] += not poly.is_valid
            counts["zero_shoelace_area_rings"] += poly.area <= 0
            unsuitable |= not poly.is_valid or poly.area <= 0 or bool(outside.any())
        counts["annotations_with_nonintegral_vertex"] += fractional
        if malformed:
            counts["malformed_annotations"] += 1
            continue
        try:
            rle = mask_codec.merge(mask_codec.frPyObjects(polys, h, w))
            area = int(mask_codec.area(rle))
            mask = mask_codec.decode(rle)
            require(mask.shape == (h, w) and int(mask.sum()) == area, "COCO decoded area or dimensions differ")
        except (TypeError, ValueError, OverflowError):
            counts["raster_failures"] += 1
            continue
        if area == 0:
            counts["empty_masks"] += 1
        else:
            positive.add(iid)
            counts["nonempty_masks"] += 1
            if not unsuitable:
                valid_geom.add(iid)
        counts["annotations_with_invalid_or_outside_geometry"] += unsuitable
    result = {key: int(counts[key]) for key in (
        "crowd","ignored","declared_area_nonpositive_or_missing","malformed_annotations","malformed_rings",
        "coordinate_values","coordinate_values_json_float","vertices","polygon_rings",
        "nonintegral_coordinate_values","nonintegral_vertices","polygons_with_nonintegral_vertex",
        "annotations_with_nonintegral_vertex","out_of_canvas_vertices","out_of_canvas_rings","invalid_shapely_rings",
        "zero_shoelace_area_rings","raster_failures","empty_masks","nonempty_masks",
        "annotations_with_invalid_or_outside_geometry")}
    result.update(annotations=len(ids), with_annotation=len(annotated), with_nonempty_raster=len(positive),
                  images_with_nonempty_valid_in_bounds_geometry=len(valid_geom),
                  original_category_ids=sorted(classes))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--unrar", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()
    require(sha256((args.work/"PRELOCK.md").read_bytes()).hexdigest() == LOCK, "prelock mismatch")
    names = [f"{i:06}.jpg" for i in range(1, 1161) if i % 10 in (1, 9)]
    private = args.work/"PRIVATE_INPUTS"
    select_inputs(args.archive, args.unrar, private, names)
    raw = (private/"test.json").read_bytes()
    require(len(raw) == 397010 and crc32(raw) == 0xdacc1bcd, "original COCO annotation identity mismatch")
    coco = json.loads(raw)
    by_id, images = check_images(coco, names, private)
    semantic = check_annotations(coco, by_id)
    result = dict(status="PSEG_SSDD_TEST232_LOCAL_INPUT_SEMANTIC_AUDIT_COMPLETE", prelock_sha256=LOCK,
                  json_sha256=sha256(raw).hexdigest(), json_bytes=len(raw), json_crc32="dacc1bcd",
                  scoring_or_model_forwards=0, whole_archive_authenticated=False,
                  historical_overlap_source="PSEG_SSDD_CANDIDATE_ARCHIVE_ROLES_AND_EXACT_OVERLAP_SAFE_20261111",
                  inherited_train_overlap_test_images=10, inherited_train_side_copies=12,
                  remote_status="separate member-level result", **images, **semantic)
    write_new(args.work/"LOCAL_SAFE.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
