"""Admission of two author ship reference releases. No model or test labels."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import struct
import subprocess
from time import perf_counter
import zipfile

import numpy as np
import PIL
from PIL import Image
from pycocotools import mask
import scipy
from scipy.optimize import linear_sum_assignment

PS = "Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style"
SP = "datasets/SPSD"
PS_TRAIN = "4565548ce05136dba8072a4b1c1fb668f3d415e3c980ebfe05b5a8998dec1ac5"
SLGSEA = "23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5"
LOCK = "bb9d30de3265ffce7bcf031958fe3b9735a5c3d51e6e0600ed9041531458615e"
ERRATUM = "a62de3a5db1e014230fe35d8bd64e3eab25e643c3fcace9103cd56805e7fb6fa"
ACQUISITION = re.compile(r"(?:^|_)(?:parent(?:_id|_scene)?|source_(?:scene|image|id|product)|acquisition(?:_id|_time|_date)?|tile_[xy]|product_id|scene_id|geotransform|crs)(?:$|_)")
JPEG = re.compile(r"^\d+\.jpg$")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_hash(path):
    h = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def tar_bytes(archive, flag, *patterns):
    return subprocess.run(["tar", flag, str(archive), *patterns], capture_output=True, check=True).stdout


def data_by(items, key):
    result = {item[key]: item for item in items}
    need(len(result) == len(items), f"Duplicate {key} in the author roster")
    return result


def zip_members(z, prefix, suffix):
    members = [name.removeprefix(prefix) for name in z.namelist() if name.startswith(prefix) and name.endswith(suffix)]
    need(len(members) == len(set(members)) and all("/" not in name for name in members), f"Invalid member roster {prefix}")
    return set(members)


def jpeg_identity(data):
    with Image.open(BytesIO(data)) as image:
        need(image.format == "JPEG", "Original chip is JPEG")
        size = image.size
        exif = tuple(map(int, image.getexif().keys()))
        pixels = image.convert("RGB").tobytes()
    key = digest(struct.pack(">II", *size) + pixels)
    return size, key, exif


def archive_images(official, z, listing, split):
    pseg_prefix = f"{PS}/images/{split}/"
    spsd_prefix = f"{SP}/{'train' if split == 'train' else 'val'}2017/"
    pattern = re.compile(rb"^\S+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+.* " + pseg_prefix.encode() + rb"(\d+\.jpg)$")
    members = [pattern.fullmatch(line) for line in listing]
    members = [(int(m[1]), m[2].decode("ascii")) for m in members if m]
    names = {name for _, name in members}
    need(len(names) == len(members) == (928 if split == "train" else 232), "Exact PSeg author image population")
    need(names == zip_members(z, spsd_prefix, ".jpg") and all(JPEG.fullmatch(n) for n in names), "Identical SPSD/PSeg JPEG member names")
    blob = tar_bytes(official, "-xOf", "--include", pseg_prefix + "*.jpg")
    need(len(blob) == sum(length for length, _ in members), "Exact complete sequential RAR JPEG bytes")
    offset = 0
    rows = {}
    for length, name in members:
        data = blob[offset:offset + length]
        offset += length
        need(data == z.read(spsd_prefix + name), "Byte-identical paired JPEG")
        size, decoded, exif = jpeg_identity(data)
        rows[name] = {"width": size[0], "height": size[1], "jpeg_sha256": digest(data), "rgb_size_prefixed_sha256": decoded,
                      "pseg_image_member": pseg_prefix + name, "spsd_image_member": spsd_prefix + name, "exif_tag_ids": exif}
    return rows


def collect_keys(x):
    if isinstance(x, dict):
        for key, value in x.items():
            yield key
            yield from collect_keys(value)
    elif isinstance(x, list):
        for row in x:
            yield from collect_keys(row)


def mask_signatures(rows):
    ordered = sorted((tuple(rle["size"]), rle["counts"].decode("ascii")) for rle in rows)
    return digest(canonical(ordered))


def statistics(items):
    vals = [iou for r in items for iou in r["assigned_iou"]]
    ratios = [value for r in items for value in r["assigned_area_ratios"]]
    return {"images": len(items), "pseg_hulls": sum(r["pseg_count"] for r in items),
             "spsd_hulls": sum(r["spsd_count"] for r in items), "assigned": len(vals),
            "spsd_declared_ship_without_pixels": sum(r["spsd_missing_ship_id_count"] for r in items),
            "assignment_iou_mean": float(np.mean(vals)) if vals else None,
            "assignment_iou_median": float(np.median(vals)) if vals else None,
            "iou_ge_50": sum(i >= .5 for i in vals), "iou_ge_90": sum(i >= .9 for i in vals),
            "equal_mask_pairs": sum(i == 1. for i in vals),
            "equal_union_images": sum(r["equal_union"] for r in items),
            "equal_count_images": sum(r["pseg_count"] == r["spsd_count"] for r in items),
             "median_spsd_to_pseg_area_ratio": float(np.median(ratios)) if ratios else None,
            "spsd_metadata_area_disagreement": sum(r["spsd_meta_area_disagreement"] for r in items),
            "spsd_metadata_box_disagreement": sum(r["spsd_meta_bbox_disagreement"] for r in items)}


def annotations(official, z, train, listing):
    ps_member = PS + "/annotations/train.json"
    sp_member = SP + "/annotations/panoptic_train2017.json"
    ps_bytes = tar_bytes(official, "-xOf", ps_member)
    sp_bytes = z.read(sp_member)
    need(digest(ps_bytes) == PS_TRAIN, "Accepted float Official-SSDD training labels")
    left = json.loads(ps_bytes)
    right = json.loads(sp_bytes)
    ps_roster = data_by(left["images"], "file_name")
    sp_roster = data_by(right["images"], "file_name")
    need(set(ps_roster) == set(sp_roster) == set(train), "Both complete train annotation rosters")
    need([c["id"] for c in left["categories"] if c["name"] == "ship"] == [0] and
         [(c["id"], c["name"]) for c in right["categories"] if c["isthing"]] == [(0, "ship")], "Exact thing category alignment")
    ps_owned = defaultdict(list)
    ps_ids, sp_ids = data_by(left["images"], "id"), data_by(right["images"], "id")
    for entry in data_by(left["annotations"], "id").values():
        need(entry["image_id"] in ps_ids and entry["category_id"] == 0 and not entry["iscrowd"] and
             entry["area"] > 0 and isinstance(entry["segmentation"], list) and entry["segmentation"], "Original valid PSeg polygon reference")
        ps_owned[entry["image_id"]].append(entry)
    source = data_by(right["annotations"], "image_id")
    need(set(ps_owned) == set(ps_ids) and set(source) == set(sp_ids), "Complete positive author image ownership")
    png_prefix = SP + "/annotations/panoptic_train2017/"
    labels = data_by(right["annotations"], "file_name")
    need(set(labels) == zip_members(z, png_prefix, ".png"), "Complete unique SPSD train PNG owners")
    text_member = SP + "/text_library/spsd_text_library_train.json"
    text = json.loads(z.read(text_member))
    need(set(data_by(text, "file_name")) == set(train) and all(t["split"] == "train" for t in text),
         "Complete aligned SPSD training metadata records")
    keys = {name: sorted(set(collect_keys(items))) for name, items in
            (("pseg_image", left["images"]), ("spsd_image", right["images"]),
             ("spsd_annotations", right["annotations"]), ("spsd_text_library", text))}
    documents = sorted(set(s for s in z.namelist() if s.startswith(SP + "/") and s.endswith((".csv", ".xml", ".md", ".txt", ".json"))))
    docnames = re.compile(r"(?:metadata|scene|parent|acquisition|product|georef|mapping|readme|info)", re.I)
    official_docs = sorted(set(s.split(maxsplit=8)[-1] for s in listing if docnames.search(s) and
                               s.endswith((".csv", ".md", ".txt", ".json"))))
    source_meta = {"field_names": keys, "acquisition_fields": {k: [field for field in v if ACQUISITION.search(field)] for k, v in keys.items()},
                   "spsd_text_and_annotation_archive_members": documents,
                   "official_candidate_metadata_member_count": len(official_docs),
                   "official_candidate_metadata_members": official_docs[:32]}
    for name in sorted(train):
        item, p, q = train[name], ps_roster[name], sp_roster[name]
        shape = item["height"], item["width"]
        need(shape == (p["height"], p["width"]) == (q["height"], q["width"]), "Both labels align to native JPEG")
        old = ps_owned[p["id"]]
        a = [mask.merge(mask.frPyObjects(v["segmentation"], *shape)) for v in old]
        need(all(int(mask.area(x)) > 0 for x in a), "Nonempty native PSeg reference")
        record = source[q["id"]]
        png_name = png_prefix + record["file_name"]
        png = z.read(png_name)
        with Image.open(BytesIO(png)) as im:
            need(im.mode == "RGB" and im.size == shape[::-1], "Original aligned SPSD RGB raster")
            rgb = np.asarray(im, dtype=np.uint32)
        ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        segment = data_by(record["segments_info"], "id")
        present = set(np.unique(ids).tolist())
        need(present <= set(segment) | {0}, "All nonvoid native pixels have an author segment ID")
        missing = sorted(set(segment) - present)
        need(all(segment[j]["category_id"] == 0 and segment[j]["area"] > 0 for j in missing),
             "Only the documented missing positive-area ship IDs")
        missing_set = set(missing)
        b = []
        area_diff = box_diff = 0
        for v in record["segments_info"]:
            if v["category_id"] != 0 or v["id"] in missing_set:
                continue
            rle = mask.encode(np.asfortranarray(ids == v["id"], dtype=np.uint8))
            area, bbox = int(mask.area(rle)), mask.toBbox(rle).tolist()
            need(area > 0 and not v["iscrowd"] and v["area"] > 0 and len(v["bbox"]) == 4, "Nonempty SPSD ship ID")
            area_diff += area != v["area"]
            box_diff += bbox != v["bbox"]
            b.append(rle)
        need(bool(b), "All aligned author images have SPSD ship reference")
        matrix = mask.iou(a, b, [0] * len(b))
        i, j = linear_sum_assignment(-matrix)
        ratios = np.asarray(mask.area(b), dtype=np.float64)[j] / np.asarray(mask.area(a), dtype=np.float64)[i]
        item.update(pseg_image_id=p["id"], spsd_image_id=q["id"], pseg_annotation_ids=[v["id"] for v in old],
                    pseg_annotation_row_digest=digest(canonical(old)), pseg_raster_set_digest=mask_signatures(a),
                    spsd_panoptic_record_digest=digest(canonical(record)), spsd_png_member=png_name, spsd_png_sha256=digest(png),
                    spsd_ship_raster_set_digest=mask_signatures(b), spsd_ship_segment_ids=[v["id"] for v in record["segments_info"]
                                                                                      if v["category_id"] == 0],
                    pseg_count=len(a), spsd_count=len(b), spsd_missing_ship_ids=missing,
                    spsd_missing_ship_id_count=len(missing), assigned_iou=matrix[i, j].tolist(),
                    assigned_area_ratios=ratios.tolist(), equal_union=bool(mask.iou([mask.merge(a)], [mask.merge(b)], [0]).item() == 1.),
                    spsd_meta_area_disagreement=area_diff, spsd_meta_bbox_disagreement=box_diff)
    need(sum(item["pseg_count"] for item in train.values()) == 2041, "Accepted PSeg train instance count")
    sizes = [item["spsd_missing_ship_id_count"] for item in train.values() if item["spsd_missing_ship_id_count"]]
    need(sizes == [2], "Exactly two ship IDs absent in the one admitted author raster discrepancy")
    return digest(ps_bytes), digest(sp_bytes), source_meta


def main():
    p = ArgumentParser(description=__doc__)
    for k in ("official", "slgsea", "protocol", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    started = perf_counter()
    need(not (a.output / "RESULT_SAFE.json").exists() and not (a.output / "TRAIN_ROSTER_PRIVATE.json").exists(), "Choose fresh output")
    need(file_hash(a.protocol) == LOCK, "Frozen plain-text admission")
    need(file_hash(a.protocol.with_name("ERRATUM.md")) == ERRATUM, "Frozen prescore native-raster erratum")
    need(a.official.stat().st_size == 1043241197 and a.slgsea.stat().st_size == 866821756, "Accepted original archive lengths")
    zip_sha = file_hash(a.slgsea)
    need(zip_sha == SLGSEA, "Accepted exact SLGSea package")
    listing = tar_bytes(a.official, "-tvf").decode("utf-8", errors="replace").splitlines()
    with zipfile.ZipFile(a.slgsea) as z:
        train = archive_images(a.official, z, [s.encode("utf-8") for s in listing], "train")
        test = archive_images(a.official, z, [s.encode("utf-8") for s in listing], "test")
        need(not (train.keys() & test.keys()), "Distinct author roster filenames")
        ps_sha, sp_sha, metadata = annotations(a.official, z, train, listing)

    test_encoded = {r["jpeg_sha256"] for r in test.values()}
    test_decoded = {r["rgb_size_prefixed_sha256"] for r in test.values()}
    need(len(test_decoded) == 232, "Exactly 232 prior unique decoded TEST image groups")
    groups = defaultdict(list)
    retained, removed = [], []
    for name, r in sorted(train.items()):
        encoded = r["jpeg_sha256"] in test_encoded
        decoded = r["rgb_size_prefixed_sha256"] in test_decoded
        need(not encoded or decoded, "Identical JPEG always decodes identically")
        r["cross_test_encoded_overlap"], r["cross_test_decoded_overlap"] = encoded, decoded
        r["invalid_dual_reference"] = bool(r["spsd_missing_ship_id_count"])
        if encoded or decoded or r["invalid_dual_reference"]:
            removed.append(r)
        else:
            retained.append(r)
            groups[r["rgb_size_prefixed_sha256"]].append(name)

    for g in groups.values():
        for name in g:
            row = train[name]
            row["decoded_group_representative"] = g[0]
            row["decoded_group_size"] = len(g)
    sizes = Counter(map(len, groups.values()))
    disagreements = {"pseg": 0, "spsd": 0}
    for g in groups.values():
        if len(g) > 1:
            for label, sig in (("pseg", "pseg_raster_set_digest"), ("spsd", "spsd_ship_raster_set_digest")):
                disagreements[label] += len({train[name][sig] for name in g}) != 1
    metadata["all_jpeg_exif_tag_census"] = dict(Counter(i for row in [*train.values(), *test.values()] for i in row["exif_tag_ids"]))
    metadata["acquisition_grouping_admitted"] = False
    roster = dict(version=1, groups_by_first_filename=sorted(groups.values()),
                  admitted_train={n: v for n, v in sorted(train.items()) if "decoded_group_representative" in v},
                  excluded_train={n: v for n, v in sorted(train.items()) if "decoded_group_representative" not in v})
    roster_bytes = canonical(roster)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "TRAIN_ROSTER_PRIVATE.json").write_bytes(roster_bytes)
    crossover = [row for row in removed if row["cross_test_decoded_overlap"]]
    invalid = [row for row in removed if row["invalid_dual_reference"]]
    result = dict(status="SPSD_PSEG_DUAL_REFERENCE_TRAIN_ROSTER_ADMITTED", model_forwards=0,
                  optimizer_updates=0, test_annotation_members_read=0, full_source_train=statistics(list(train.values())),
                  admitted_train=statistics(retained), removed_train=statistics(removed),
                  removed_cross_test=statistics(crossover), removed_invalid_reference=statistics(invalid),
                  invalid_reference_also_cross_test_count=sum(row["cross_test_decoded_overlap"] for row in invalid),
                  test_identity_images=len(test), test_distinct_encoded_groups=len(test_encoded),
                  test_distinct_decoded_groups=len(test_decoded),
                  train_cross_test_both_enc_dec=sum(r["cross_test_encoded_overlap"] for r in crossover),
                  train_cross_test_decoded_only=sum(not r["cross_test_encoded_overlap"] for r in crossover),
                  train_all_exact_decoded_groups=len({r["rgb_size_prefixed_sha256"] for r in train.values()}),
                  train_admitted_decoded_group_sizes=dict(sorted(sizes.items())),
                  duplicate_groups_with_differing_reference_rasters=disagreements, source_metadata=metadata,
                  protocol_sha256=LOCK, erratum_sha256=ERRATUM, worker_sha256=file_hash(Path(__file__)),
                  slgsea_zip_sha256=zip_sha, pseg_author_rar_sha256=file_hash(a.official),
                  pseg_train_float_annotation_sha256=ps_sha, spsd_train_panoptic_annotation_sha256=sp_sha,
                  admitted_roster_private_sha256=digest(roster_bytes), runtime=dict(pillow=PIL.__version__, numpy=np.__version__, scipy=scipy.__version__),
                  elapsed_seconds=perf_counter() - started)
    data=canonical(result)+b"\n"
    (a.output / "RESULT_SAFE.json").write_bytes(data)
    print("SPSD_PSEG_DUAL_REFERENCE_TRAIN_ADMISSION_COMPLETE", digest(data), flush=True)


if __name__ == "__main__":
    main()
