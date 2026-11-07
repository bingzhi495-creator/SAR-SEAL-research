"""Build a read-only RF-DETR COCO layout from two selected independent archives."""
import hashlib
import json
import os
import re
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

PROTOCOL = {
    "train": {
        "archive": (33079338, "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"),
        "annotation_name": "annotations/fit.json",
        "annotation": (498760, "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"),
        "manifest": (861009, "5bf021523d1ba3507a5ef7ba6a731da8699c37eb32d6b8ea962c52dcfbaafb47"),
        "manifest_schema": "SSDD-FIT741-portable-original-byte-input-manifest-v1",
        "other": {},
        "images": 741,
        "instances": 1661,
        "unique_encoded_images": 730,
    },
    "valid": {
        "archive": (7439654, "250d72ec705ced1fdf775c1331ce2b439e023cb29f039a4b7b0d50313f92e794"),
        "annotation_name": "annotations/dev.json",
        "annotation": (118319, "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"),
        "manifest": (78137, "525c4d741c8c5c0d5dd5e460206eb220465743007c03e5692fbf053b0dfdd9da"),
        "manifest_schema": "Root-DiffSAR-DEV187-original-image-input-v1",
        "other": {"ORIGINAL_LOCAL_CANDIDATE_MANIFEST.json": (
            83195, "454f2def1ffb2ce5baf68904922f5fdba4827291c8af4fc525ab49c0db93b1f4"
        )},
        "images": 187,
        "instances": 380,
        "unique_encoded_images": 183,
    },
}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def byte_identity(raw):
    return len(raw), hashlib.sha256(raw).hexdigest()


def file_identity(path):
    digest = hashlib.sha256()
    size = 0
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after), "file changed during hash")
    return size, digest.hexdigest()


def exact_member(archive, name, expected):
    item = archive.getinfo(name)
    need(item.compress_type == ZIP_STORED and item.file_size == expected[0], "bound member length")
    raw = archive.read(item)
    need(byte_identity(raw) == expected, "bound member checksum")
    return raw


def freeze(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
    os.chmod(path, 0o400)


def assemble_split(split, archive_path, destination):
    selected = PROTOCOL[split]
    need(file_identity(archive_path) == selected["archive"], "exact selected archive bytes")
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = {item.filename for item in infos}
        need(len(names) == len(infos), "duplicate archive member")
        annotation_raw = exact_member(archive, selected["annotation_name"], selected["annotation"])
        annotations = json.loads(annotation_raw)
        manifest = json.loads(exact_member(archive, "manifest.json", selected["manifest"]))
        for name, expected in selected["other"].items():
            exact_member(archive, name, expected)
        need(manifest["schema"] == selected["manifest_schema"], "selected split manifest schema")
        rows = manifest["rows" if split == "train" else "images"]
        images = annotations["images"]
        instances = annotations["annotations"]
        categories = annotations["categories"]
        need(len(rows) == len(images) == selected["images"], "exact split image count")
        need(len(instances) == selected["instances"] and [c["id"] for c in categories] == [1],
             "exact instance count and ship category")
        ids = {image["id"] for image in images}
        need(len(ids) == len(images) and all(type(image_id) is int for image_id in ids), "unique integer image IDs")
        need(len({ann["id"] for ann in instances}) == len(instances), "unique annotations within split")
        need(all(ann["image_id"] in ids and ann["category_id"] == 1 and ann["iscrowd"] == 0
                 and ann["segmentation"] for ann in instances), "annotation population and valid target references")
        positives = {ann["image_id"] for ann in instances
                     if ann["bbox"][2] > 0 and ann["bbox"][3] > 0 and ann["area"] > 0}
        need(bool(positives), "preselected positive probe witness")

        output_names = set()
        encoded_ids = set()
        member_names = {selected["annotation_name"], "manifest.json", *selected["other"]}
        records = []
        for index, (image, row) in enumerate(zip(images, rows)):
            need(row["dataset_index"] == index and row["image_id"] == image["id"], "same frozen image order")
            name = image["file_name"]
            need(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.-]+\.jpg", name)
                 and name not in output_names, "unique safe JPEG basename")
            if split == "train":
                ref = row["original_image_ref"]
                need(row["original_basename"] == name, "fit original filename")
                need(row["original_shape"] == [image["height"], image["width"]], "fit original dimensions")
            else:
                ref = row["image_ref"]
                need((row["height"], row["width"]) == (image["height"], image["width"]), "valid original dimensions")
            encoded_id = ref["sha256"]
            need(re.fullmatch(r"[0-9a-f]{64}", encoded_id) and type(ref["bytes"]) is int
                 and 0 < ref["bytes"] <= 16 * 1024 * 1024, "exact JPEG reference")
            member = f"images/{encoded_id}.jpg"
            if split == "train":
                need(row["archive_image_entry"] == member, "fit original member")
            output_names.add(name)
            encoded_ids.add(encoded_id)
            member_names.add(member)
            records.append((name, member, (ref["bytes"], encoded_id)))
        need(names == member_names and len(encoded_ids) == selected["unique_encoded_images"],
             "exact archive membership and byte groups")

        destination.mkdir()
        for name, member, expected in records:
            raw = exact_member(archive, member, expected)
            freeze(destination / name, raw)
            need(file_identity(destination / name) == expected, "native copy checksum")
        freeze(destination / "_annotations.coco.json", annotation_raw)
        need(file_identity(destination / "_annotations.coco.json") == selected["annotation"], "native annotation bytes")
        os.chmod(destination, 0o500)

    return {
        "image_ids": [image["id"] for image in images],
        "positive_witness_image_id": min(positives),
        "images": len(images),
        "instances": len(instances),
        "category_ids": [1],
        "unique_encoded_images": len(encoded_ids),
        "annotation_bytes": selected["annotation"][0],
        "annotation_sha256": selected["annotation"][1],
    }, encoded_ids


def build_native_view(archives, output):
    """Use only explicit archive files. A receipt is emitted by the caller on success."""
    output = Path(output)
    need(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
         and not output.exists(), "fresh canonical native dataset destination")
    need(set(archives) == set(PROTOCOL), "explicit train and valid archives")
    output.mkdir()
    summary = {}
    checksums = {}
    for split, path in archives.items():
        path = Path(path)
        need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
             "canonical ordinary input archive")
        summary[split], checksums[split] = assemble_split(split, path, output / split)
    need(not set(summary["train"]["image_ids"]) & set(summary["valid"]["image_ids"]),
         "train and valid image IDs overlap")
    need(not checksums["train"] & checksums["valid"], "train and valid encoded JPEG hashes overlap")
    os.chmod(output, 0o500)
    return {
        "status": "NATIVE_TRAIN_VALID_LAYOUT_BYTES_VERIFIED",
        "splits": summary,
        "cross_split_image_id_overlap": 0,
        "cross_split_encoded_jpeg_sha_overlap": 0,
        "scene_identity_established": False,
        "native_dataset_executed": False,
        "training": False,
        "AP": None,
    }
