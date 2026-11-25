"""Rebuild private train-only COCO references from the two author archives."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
from time import perf_counter
import zipfile

import numpy as np
from PIL import Image
from pycocotools import mask

LOCK = "2dc622eaed8ac6b262ed70a46547e4d0d05beb1eba66af0ba541ff83c4204536"
ADMISSION = "e40636ec2cfb4c42718673a553f069f5f4e2833e3e81264c9d44a43ece7b9927"
READBACK = "f6be62a8e860fa5c135bcadef88d1d57317b9bb8a12ae32c7af13aab2140d841"
ROSTER = "f4e23e5eb7f58a881db48e4a89d2902a253aa945e7a92ab7d1674c8b02b3fb17"
PARTITION = "91095aacbbbc89003ffd6eb8f25506ce99a64eee5ac369a7753203bbf962bcf9"
SPLIT_SAFE = "a02f0ac9e48bc4a54fd55a7d05853ddceee04fa69c4887ea279f85a09ae05872"
PSEG_TRAIN = "Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style/annotations/train.json"
SPSD_TRAIN = "datasets/SPSD/annotations/panoptic_train2017.json"
SPSD_PREFIX = "datasets/SPSD/annotations/panoptic_train2017/"
NAMES = {"fit": (732, 721), "source_gate": (183, 180)}
OUTPUT = {"pseg": "PSeg_SSDD_COCO_PRIVATE.json", "spsd": "SPSD_COCO_PRIVATE.json", "binding": "IMAGE_BINDINGS_PRIVATE.json"}
STATUS = "SPSD_PSEG_TRAIN_ONLY_LABEL_EXPORT_COMPLETE"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def file_hash(path):
    h = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def json_checked(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Unchanged admitted file " + path.name)
    return json.loads(raw)


def unique_index(rows, field):
    index = {row[field]: row for row in rows}
    need(len(index) == len(rows), "Unique indexed author field " + field)
    return index


def read_partition(args):
    admission = json_checked(args.admission, ADMISSION)
    readback = json_checked(args.admission_readback, READBACK)
    need(readback["partition_and_hashes_verified"] is True and readback["result_sha256"] == ADMISSION and
         admission["admitted_roster_private_sha256"] == ROSTER, "Verified closed source roster")
    roster = json_checked(args.roster, ROSTER)["admitted_train"]
    split_stats = json_checked(args.split_safe, SPLIT_SAFE)
    split = json_checked(args.partition, PARTITION)
    need(split_stats["private_partition_sha256"] == PARTITION and split["parent_roster_sha256"] == ROSTER and
         split_stats["fit_gate_group_overlap"] == 0, "Exactly verified clean group split")
    partitions = {key: split["partition"][key] for key in NAMES}
    groups = {}
    for key, (images, groups_expected) in NAMES.items():
        names = partitions[key]
        lists = split["decoded_groups"][key]
        members = [name for g in lists for name in g]
        need(len(names) == len(set(names)) == images and len(lists) == groups_expected and
             len(members) == len(set(members)) and set(names) == set(members), "Complete prescribed partition")
        hashes = [set(roster[name]["rgb_size_prefixed_sha256"] for name in group) for group in lists]
        need(all(len(g) == 1 for g in hashes), "Preserved duplicate-image groups")
        groups[key] = {next(iter(g)) for g in hashes}
        need(len(groups[key]) == groups_expected, "Unique decoded identities across partition groups")
    f, g = set(partitions["fit"]), set(partitions["source_gate"])
    need(not (f & g) and f | g == set(roster) and len(roster) == 915 and
         not (groups["fit"] & groups["source_gate"]) and len(groups["fit"] | groups["source_gate"]) == 901,
         "Disjoint, complete clean source population")
    for item in roster.values():
        need(not item["cross_test_decoded_overlap"] and not item["cross_test_encoded_overlap"] and
             not item["invalid_dual_reference"], "Only admitted train-only image and reference")
    return admission, roster, partitions


def serialized_rle(rle):
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def raster_digest(rles):
    return digest(canonical(sorted((tuple(rle["size"]), rle["counts"]) for rle in rles)))


def make_reference_records(args, admitted, roster):
    need(file_hash(args.official) == admitted["pseg_author_rar_sha256"], "Exact Official-SSDD author RAR")
    need(file_hash(args.slgsea) == admitted["slgsea_zip_sha256"], "Exact SLGSea author ZIP")
    pseg_raw = subprocess.run(["tar", "-xOf", str(args.official), PSEG_TRAIN], capture_output=True, check=True).stdout
    need(digest(pseg_raw) == admitted["pseg_train_float_annotation_sha256"], "Exact original train polygons")
    left = json.loads(pseg_raw)
    p_images = unique_index(left["images"], "file_name")
    p_annotations = defaultdict(list)
    for a in unique_index(left["annotations"], "id").values():
        p_annotations[a["image_id"]].append(a)
    made = {}
    sid = 0
    with zipfile.ZipFile(args.slgsea) as archive:
        right_raw = archive.read(SPSD_TRAIN)
        need(digest(right_raw) == admitted["spsd_train_panoptic_annotation_sha256"], "Exact SPSD panoptic train roster")
        right = json.loads(right_raw)
        s_images = unique_index(right["images"], "file_name")
        s_annotations = unique_index(right["annotations"], "image_id")
        need(left["categories"] == [{"supercategory": "none", "id": 0, "name": "ship"}] and
             [a["id"] for a in right["categories"] if a["isthing"]] == [0], "Common ship scoring class")
        for name in sorted(roster, key=lambda k: roster[k]["pseg_image_id"]):
            row = roster[name]
            p_im, s_im = p_images[name], s_images[name]
            shape = (row["height"], row["width"])
            need(p_im["id"] == row["pseg_image_id"] and s_im["id"] == row["spsd_image_id"] and
                 shape == (p_im["height"], p_im["width"]) == (s_im["height"], s_im["width"]),
                 "Exact source image owner and native grid")
            original_poly = p_annotations[p_im["id"]]
            need([a["id"] for a in original_poly] == row["pseg_annotation_ids"] and
                 digest(canonical(original_poly)) == row["pseg_annotation_row_digest"] and
                 len(original_poly) == row["pseg_count"], "Exact original train float polygon rows")
            pseg = []
            for a in original_poly:
                rle = mask.merge(mask.frPyObjects(a["segmentation"], *shape))
                need(int(mask.area(rle)) > 0, "Nonempty PSeg scoring raster")
                pseg.append(dict(a, segmentation=serialized_rle(rle)))
            need(raster_digest([a["segmentation"] for a in pseg]) == row["pseg_raster_set_digest"],
                 "Exact admitted PSeg raster digests")
            record = s_annotations[s_im["id"]]
            need(digest(canonical(record)) == row["spsd_panoptic_record_digest"] and
                 row["spsd_png_member"] == SPSD_PREFIX + record["file_name"], "Exact SPSD ship-mask ownership")
            png = archive.read(row["spsd_png_member"])
            need(digest(png) == row["spsd_png_sha256"], "Exact native SPSD PNG bytes")
            with Image.open(BytesIO(png)) as image:
                need(image.mode == "RGB" and image.size == shape[::-1], "Exact SPSD raster geometry")
                rgb = np.asarray(image, dtype=np.uint32)
            native = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
            segments = [a for a in unique_index(record["segments_info"], "id").values() if a["category_id"] == 0]
            need(len(segments) == row["spsd_count"] and [a["id"] for a in segments] == row["spsd_ship_segment_ids"],
                 "Exact original author ship segment IDs")
            spsd, mapping = [], []
            for a in sorted(segments, key=lambda i: i["id"]):
                rle = mask.encode(np.asfortranarray(native == a["id"], dtype=np.uint8))
                area = int(mask.area(rle))
                need(not a["iscrowd"] and area > 0, "Nonempty noncrowd SPSD reference")
                sid += 1
                spsd.append(dict(id=sid, image_id=p_im["id"], category_id=0, iscrowd=0, area=area, bbox=mask.toBbox(rle).tolist(),
                                 segmentation=serialized_rle(rle)))
                mapping.append((a["id"], sid))
            need(raster_digest([a["segmentation"] for a in spsd]) == row["spsd_ship_raster_set_digest"],
                 "Exact admitted SPSD native raster digests")
            binding = dict(image_id=p_im["id"], spsd_author_image_id=s_im["id"], width=shape[1], height=shape[0],
                           decoded_group_representative=row["decoded_group_representative"],
                           decoded_group_sha256=row["rgb_size_prefixed_sha256"], jpeg_sha256=row["jpeg_sha256"],
                           pseg_original_scoring_rows_sha256=row["pseg_annotation_row_digest"],
                           pseg_raster_set_sha256=row["pseg_raster_set_digest"], spsd_png_sha256=row["spsd_png_sha256"],
                           spsd_author_scoring_row_sha256=row["spsd_panoptic_record_digest"],
                           spsd_raster_set_sha256=row["spsd_ship_raster_set_digest"], pseg_original_ids=row["pseg_annotation_ids"],
                           spsd_author_segment_to_export_id=mapping, pseg_export_rows_sha256=digest(canonical(pseg)),
                           spsd_export_rows_sha256=digest(canonical(spsd)))
            made[name] = (p_im, pseg, spsd, binding)
    need(sid == sum(len(values[1]) for values in made.values()) == 1990, "1,990 individual train references per release")
    return left["categories"], made


def put(path, obj):
    data = canonical(obj) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
    return digest(data)


def save(out, partitions, categories, records):
    out.mkdir()
    stats = {}
    seen = {"image": set(), "pseg": set(), "spsd": set(), "encoded": set(), "decoded": set()}
    for key, names in partitions.items():
        dest = out / key
        dest.mkdir()
        selected = sorted((records[name] for name in names), key=lambda v: v[0]["id"])
        images = [a[0] for a in selected]
        ps = [r for a in selected for r in a[1]]
        sp = [r for a in selected for r in a[2]]
        binds = {a[0]["file_name"]: a[3] for a in selected}
        image_ids = {i["id"] for i in images}
        for ann, index in ((ps, "pseg"), (sp, "spsd")):
            keys = {r["id"] for r in ann}
            need(len(keys) == len(ann) and not keys & seen[index] and {r["image_id"] for r in ann} == image_ids,
                 "Partition preserves unique valid reference IDs and owners")
            seen[index].update(keys)
        current = {"image": image_ids, "encoded": {b["jpeg_sha256"] for b in binds.values()},
                   "decoded": {b["decoded_group_sha256"] for b in binds.values()}}
        for group, values in current.items():
            need(not (values & seen[group]), "No train group or source content across partitions")
            seen[group].update(values)
        hashes = {tag: put(dest / OUTPUT[tag], dict(type="instances", images=images, categories=categories, annotations=ann))
                  for tag, ann in (("pseg", ps), ("spsd", sp))}
        hashes["binding"] = put(dest / OUTPUT["binding"], dict(partition=key, source_partition_sha256=PARTITION,
                                                                image_bindings=binds, label_sha256=hashes.copy()))
        stats[key] = dict(images=len(images), decoded_groups=len(current["decoded"]), pseg_hulls=len(ps), spsd_hulls=len(sp),
                          artifact_hashes=hashes)
    need(len(seen["image"]) == 915 and len(seen["decoded"]) == 901 and len(seen["pseg"]) == len(seen["spsd"]) == 1990,
         "Complete exact train population and both hull totals")
    return stats


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("protocol", "admission", "admission-readback", "roster", "split-safe", "partition", "official", "slgsea", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    need(file_hash(args.protocol) == LOCK and not args.output.exists(), "Exact export prelock and fresh output path")
    admitted, roster, folds = read_partition(args)
    classes, records = make_reference_records(args, admitted, roster)
    stats = save(args.output, folds, classes, records)
    result = dict(status=STATUS, partitions=stats, image_count=915, decoded_groups=901, pseg_hulls=1990, spsd_hulls=1990,
                  fit_gate_encoded_overlap=0, fit_gate_decoded_overlap=0, test_images_opened=0, test_annotation_members_opened=0,
                  optimizer_updates=0, model_forwards=0, admission_sha256=ADMISSION, roster_sha256=ROSTER,
                  readback_sha256=READBACK, split_safe_sha256=SPLIT_SAFE, partition_sha256=PARTITION, protocol_sha256=LOCK,
                  worker_sha256=file_hash(Path(__file__)), seconds=perf_counter()-started)
    result_sha = put(args.output / "RESULT_SAFE.json", result)
    print(STATUS, result_sha, flush=True)


if __name__ == "__main__":
    main()
