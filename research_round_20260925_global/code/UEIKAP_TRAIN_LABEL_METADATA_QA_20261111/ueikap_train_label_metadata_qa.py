"""Qualify the public UEIKAP label format on four previously unused TRAIN families."""

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from math import floor, isfinite
from pathlib import Path, PurePosixPath
import re
import struct
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZipFile


DIRECTORY_LIMIT = 1024**2
MEMBER_LIMIT = 64 * 1024
TRANSFER_LIMIT = 2 * 1024**2
ZIP_END = struct.Struct("<4s4H2IH")
ZIP_LOCAL = struct.Struct("<4s5H3I2H")
URL = "https://zenodo.org/api/records/18391027/files/UEIKAP_SAR.zip/content"
ARCHIVE_BYTES = 1571288093
DIRECTORY_SHA256 = "e7a969a39b4f6723d9293acdc9297e917d2830fcba54eeac3dd22b26b4f56610"
RECOVERED = (9066, "fb58d7a0c4b7ac1f81751c2c8955c61cf8ce5bb0e6e2bccd738cec9ee1def732")
RECOVERY = (1274, "c2195a679688590284c1f269b843dd88d7986a29607502bc0d5225aa47cc9b01")
CATALOGUE = (370, "7a72d54dc5b67031a02474089442cde7f3b57c1a20ba28664971b4978648793d")
PREVIOUS_FAMILY_SHA256 = "5c0e6f1af01f41a2776ab1cefe5592f13d7755ac8d28c24fa1f4fdbda5634612"
CONTROL_STEM_SHA256 = "78f37f5a849ea25884011146508cda8dbfb05e793c091f526c4cd6b10b0d0bf6"
CONTROL_MEMBERS = {
    "txt": (408, None, "699d355ac1ea749545a19c6664f0255f3733f07fd0c87d258637b4a841de7c15"),
    "coco": (2404, 0xC77B95B7, "c94fd3885645f323c2eb540efa0c271f78581d459e13ec020d5c352c1f3143e2"),
    "ais": (196, 0xBBA92A97, "5e6ae66b035ff0fe578732ab0bd8ef2576f20eff3407a5f990fb2c7f831d91bc"),
}
STATUS = "P305_UEIKAP_TRAIN_LABEL_METADATA_QA_V3_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE"
FALSE_COLOR = "_FalseColorVVVH_"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class SparseZipReader:
    """Expose only the pinned archive directory and one selected member to ZipFile."""

    def __init__(self, size, directory_start, directory):
        self.size = size
        self.directory = (directory_start, directory)
        self.member = (0, b"")
        self.offset = 0

    def seek(self, offset, whence=0):
        origin = (0, self.offset, self.size)[whence]
        self.offset = origin + offset
        require(0 <= self.offset <= self.size, "ZIP seek leaves the fixed archive")
        return self.offset

    def tell(self):
        return self.offset

    def seekable(self):
        return True

    def read(self, count=-1):
        start = self.offset
        end = self.size if count < 0 else min(start + count, self.size)
        if start == end:
            return b""
        for offset, data in (self.directory, self.member):
            if offset <= start < end <= offset + len(data):
                self.offset = end
                return data[start - offset:end - offset]
        raise ValueError("ZIP read leaves the admitted directory or current member")


class BoundedZip:
    """Read a fixed central directory and CRC-verified, explicitly admitted members."""

    def __init__(self, fetch, size, directory_sha256, permitted):
        self.fetch = fetch
        self.permitted = permitted if callable(permitted) else frozenset(permitted).__contains__
        end = size - ZIP_END.size
        footer = fetch(end, size)
        magic, disk, start_disk, local_count, count, length, start, comment = ZIP_END.unpack(footer)
        require(magic == b"PK\x05\x06" and disk == start_disk == comment == 0
                and 0 < local_count == count < 65535 and 0 < length <= DIRECTORY_LIMIT and start + length == end,
                "Fixed single-volume classic ZIP with no comment required")
        directory = fetch(start, end)
        require(sha256(directory).hexdigest() == directory_sha256, "Fixed author ZIP central directory digest differs")
        self.reader = SparseZipReader(size, start, directory + footer)
        self.archive = ZipFile(self.reader)
        members = self.archive.infolist()
        self.members = {item.filename: item for item in members}
        require(len(self.members) == len(members) == count, "ZIP central member identities must be unique")
        self.offsets = sorted({item.header_offset for item in members} | {start})
        require(len(self.offsets) == count + 1 and 0 <= self.offsets[0] < self.offsets[-1] == start,
                "ZIP central member offsets must be distinct and bounded")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.archive.close()

    def read_member(self, name):
        require(self.permitted(name), "Member outside the fixed TRAIN metadata scope")
        item = self.members.get(name)
        require(item is not None and 0 < item.file_size <= MEMBER_LIMIT and 0 < item.compress_size <= MEMBER_LIMIT
                and not item.flag_bits & 1, "Bounded nonencrypted original metadata member required")
        start = item.header_offset
        limit = self.offsets[bisect_right(self.offsets, start)]
        require(start + ZIP_LOCAL.size < limit, "Local metadata header cannot cross into another member")
        header = self.fetch(start, start + ZIP_LOCAL.size)
        magic, _, flags, method, _, _, _, _, _, name_len, extra_len = ZIP_LOCAL.unpack(header)
        length = name_len + extra_len + item.compress_size
        end = start + ZIP_LOCAL.size + length
        require(magic == b"PK\x03\x04" and flags == item.flag_bits and method == item.compress_type
                and length <= MEMBER_LIMIT * 2 and end <= limit, "Exact bounded local metadata envelope required")
        body = self.fetch(start + ZIP_LOCAL.size, end)
        self.reader.member = (start, header + body)
        try:
            raw = self.archive.read(item)  # ZipFile verifies the original CRC after decoding.
        finally:
            self.reader.member = (0, b"")
        require(len(raw) == item.file_size, "Original metadata member decoded length differs")
        return raw


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class AuthorRange:
    def __init__(self):
        self.opener = build_opener(NoRedirect)
        self.requests = 0
        self.bytes = 0

    def fetch(self, start, end):
        length = end - start
        require(0 <= start < end <= ARCHIVE_BYTES and length <= DIRECTORY_LIMIT
                and self.bytes + length <= TRANSFER_LIMIT and self.requests < 32,
                "Original train-only range transfer budget exceeded")
        request = Request(URL, headers={"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity",
                                        "User-Agent": "UEIKAP-train-metadata-qualification/1.0"})
        self.requests += 1
        with self.opener.open(request, timeout=18) as response:
            require(response.status == 206 and response.geturl() == URL
                    and response.headers.get("Content-Range") == f"bytes {start}-{end - 1}/{ARCHIVE_BYTES}"
                    and response.headers.get("Content-Encoding", "identity").lower() == "identity",
                    "Exact unencoded official 206 Content-Range required")
            data = response.read(length + 1)
        require(len(data) == length, "Original bounded range byte count differs")
        self.bytes += len(data)
        return data


def document(raw):
    def unique(pairs):
        value = dict(pairs)
        require(len(value) == len(pairs), "Duplicate metadata JSON keys")
        return value

    value = json.loads(raw, object_pairs_hook=unique)
    require(type(value) is dict, "Original metadata must be a JSON object")
    return value


def pinned_document(path, pin):
    size, digest = pin
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Canonical closed provenance path required")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    require(len(raw) == size and sha256(raw).hexdigest() == digest, "Closed provenance bytes or SHA-256 differ")
    return document(raw)


def select_train_headers(recovered, recovery, catalogue):
    require(recovery.get("status") == "ACTUAL_FIXED_UEIKAP_MISSING_TIFF_HEADER_RECOVERY"
            and recovery.get("archive_directory_sha256") == DIRECTORY_SHA256
            and recovery.get("archive_total_bytes") == ARCHIVE_BYTES
            and recovery.get("recovered_headers") == recovery.get("newly_unique_acquisition_assignments") == 12
            and recovery.get("recovered_unique_groups_in_qa_families") == 5
            and recovery.get("recovered_unique_groups_in_reserved_families") == 0
            and recovery.get("additional_quarantined_rows_in_reserved_families") == 0
            and recovery.get("fixed_quality_and_reserved_roles_changed") is False
            and recovery.get("formal_TEST_opened") is False,
            "Exact original recovered TRAIN families must be disjoint from all reserved families")
    require(catalogue.get("status") == "CURRENT_OFFICIAL_CATALOGUE_INDEPENDENT_READBACK_MATCHED"
            and catalogue.get("safe_recovery_sha256") == RECOVERY[1]
            and catalogue.get("full_recovered_headers_checked") == 12
            and catalogue.get("matched_groups") == 5 and catalogue.get("group_assignment_mismatches") == 0
            and catalogue.get("bytes_of_image_or_label_read") == 0, "Exact independent catalogue readback required")
    rows = recovered["rows"]
    require(type(rows) is list and len(rows) == 12 and recovered["failed"] == [],
            "Exact original twelve recovered headers required")
    by_family, names = {}, set()
    for row in rows:
        family, header = row["unique"], row["header"]
        path = PurePosixPath(header["name"])
        require(re.fullmatch(r"S1[ABCD]_\d{6}_[0-9A-F]{6}", family) is not None
                and len(path.parts) == 4 and path.parts[:3] == ("UEIKAP_SAR", "images_tifs", "train")
                and path.suffix == ".tif" and path.name not in names and header["split"] == "train"
                and header["dimensions"] == [512, 512] and header["crs"] == "EPSG:4326",
                "Only unique original non-reserved TRAIN header metadata admitted")
        names.add(path.name)
        by_family.setdefault(family, []).append(header)
    previous = [family for family in by_family if sha256(family.encode("ascii")).hexdigest() == PREVIOUS_FAMILY_SHA256]
    require(len(by_family) == 5 and len(previous) == 1, "Exact five recovered TRAIN families and prior QA required")
    control = [header for header in by_family[previous[0]]
               if sha256(PurePosixPath(header["name"]).stem.encode("ascii")).hexdigest() == CONTROL_STEM_SHA256]
    require(len(control) == 1, "Exact original TRAIN positive-control header required")
    headers = [min(by_family[family], key=lambda item: item["name"])
               for family in sorted(set(by_family) - set(previous))]
    return control[0], headers


def expected_paths(stem):
    require(stem.count(FALSE_COLOR) == 1, "Exact original FalseColor TRAIN filename convention required")
    vv = stem.replace(FALSE_COLOR, "_VV_")
    annotated = stem.replace(FALSE_COLOR, "_VV_annotations_")
    return dict(txt=f"UEIKAP_SAR/labels/train/{stem}.txt",
                coco=f"UEIKAP_SAR/labels_jsons/{annotated}_coco.json",
                ais=f"UEIKAP_SAR/AIS_interpolated_SAR_UEIKAP/{vv}.tif.csv")


def match_train_metadata(archive, stem):
    paths = expected_paths(stem)
    matching_stems = {stem, stem.replace(FALSE_COLOR, "_VV_")}
    for name in archive.members:
        path = PurePosixPath(name)
        if (len(path.parts) >= 4 and path.parts[0] == "UEIKAP_SAR"
                and path.parts[1] in ("labels", "images", "images_tifs")
                and path.parts[2] != "train" and path.stem in matching_stems):
            return paths, {"other_split_matches": True}
    return paths, {role: 0 for role, path in paths.items() if path not in archive.members}


def verify_positive_control(archive, header, expected=CONTROL_MEMBERS):
    stem = PurePosixPath(header["name"]).stem
    paths, unresolved = match_train_metadata(archive, stem)
    require(not unresolved, "Exact unique original TRAIN positive-control metadata required")
    for role, name in paths.items():
        item = archive.members[name]
        size, crc, _ = expected[role]
        require(item.file_size == size and (crc is None or item.CRC == crc),
                "Original positive-control size or central-directory CRC differs")
    blobs = {role: archive.read_member(name) for role, name in paths.items()}
    require(all(sha256(raw).hexdigest() == expected[role][2] for role, raw in blobs.items()),
            "Original positive-control member SHA-256 differs")
    return {role: digest for role, (_, _, digest) in expected.items()}


def polygon(values, width, height, *, normalized):
    require(len(values) >= 6 and len(values) % 2 == 0, "Polygon must contain at least three finite XY pairs")
    result = [float(value) for value in values]
    scales = (width, height)
    require(all(isfinite(value) and 0 <= value <= (1 if normalized else scales[index % 2])
                for index, value in enumerate(result)), "Polygon coordinates outside the admitted original canvas")
    return [value * scales[index % 2] for index, value in enumerate(result)] if normalized else result


def audit_labels(txt, coco, stem, width, height):
    from pycocotools import mask

    labels = {0: [], 1: []}
    for line in txt.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        category, *points = line.split()
        require(category in ("0", "1"), "Unexpected author YOLO category")
        rle, = mask.frPyObjects([polygon(points, width, height, normalized=True)], height, width)
        labels[int(category)].append(rle)
    image, = coco["images"]
    category, = coco["categories"]
    annotation_stem = PurePosixPath(expected_paths(stem)["coco"]).stem.removesuffix("_coco")
    allowed_image_stems = {stem, stem.replace(FALSE_COLOR, "_VV_"), annotation_stem}
    require(PurePosixPath(image["file_name"]).stem in allowed_image_stems
            and image["width"] == width and image["height"] == height
            and category["name"].strip().casefold() == "wake", "Exact same-chip named author COCO wake required")
    wakes, ids = [], set()
    for row in coco["annotations"]:
        require(row["image_id"] == image["id"] and row["category_id"] == category["id"] and row["id"] not in ids,
                "Original COCO wake annotation identities differ")
        ids.add(row["id"])
        polygons = [polygon(points, width, height, normalized=False) for points in row["segmentation"]]
        require(polygons, "Author COCO wake requires polygon coordinates")
        wakes.append(mask.merge(mask.frPyObjects(polygons, height, width)))

    def nonempty(rows):
        return [row for row in rows if int(mask.area(row)) > 0]

    def counts(rows):
        return Counter(row["counts"] for row in nonempty(rows))

    body, yolo_wake = labels[0], labels[1]
    overlap = mask.iou(body, wakes, [0] * len(wakes)) if body and wakes else None
    exact = (counts(yolo_wake) & counts(wakes)).total()
    return dict(txt_class0=len(body), txt_class1=len(yolo_wake), coco_wake=len(wakes),
                coco_wake_category_id=category["id"], wake_exact_nonempty_raster_matches=exact,
                all_wakes_match_exactly=exact == len(yolo_wake) == len(wakes) > 0,
                class0_wake_positive_iou_pairs=int((overlap > 0).sum()) if overlap is not None else 0,
                empty_txt_class0=len(body) - len(nonempty(body)), empty_txt_class1=len(yolo_wake) - len(nonempty(yolo_wake)),
                empty_coco_wakes=len(wakes) - len(nonempty(wakes))), (body, yolo_wake)


def utc_proxy(value):
    value = value.strip()
    match = re.fullmatch(r"(?:S1[ABCD]_)?(\d{8}T\d{6})", value)
    if match:
        return datetime.strptime(match[1], "%Y%m%dT%H%M%S")
    try:
        point = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return point.astimezone(timezone.utc).replace(tzinfo=None) if point.tzinfo else point


def audit_ais(raw, header, masks):
    import numpy as np
    from pycocotools import mask

    table = csv.DictReader(StringIO(raw.decode("utf-8-sig")))
    rows = list(table)
    fields = table.fieldnames or []
    require(len(set(fields)) == len(fields) and all(None not in row for row in rows), "Invalid original AIS CSV structure")
    normalized = {field: re.sub(r"[^a-z0-9]", "", field.casefold()) for field in fields}
    aliases = (("lat", "latitude", "latinterp", "interplat", "latinterpolated", "interpolatedlatitude"),
               ("lon", "lng", "longitude", "loninterp", "interplon", "loninterpolated", "interpolatedlongitude"),
               ("timestamp", "time", "datetime", "basedatetime", "timestamputc", "datetimeutc", "interpolatedtimestamp"))
    candidates = [[field for field in fields if normalized[field] in choices] for choices in aliases]
    result = dict(rows=len(rows), columns=len(fields), coordinate_columns_identified=all(len(items) == 1 for items in candidates[:2]),
                  timestamp_column_identified=len(candidates[2]) == 1)
    if len(rows) != 1 or any(len(items) != 1 for items in candidates):
        return dict(result, status="NO_UNIQUE_AIS_ROW_OR_FIELD_MAPPING")
    row, = rows
    lat, lon, time = (row[items[0]] or "" for items in candidates)
    reference, observed = utc_proxy(header["timestamp_proxy"]), utc_proxy(time)
    try:
        lat, lon = float(lat), float(lon)
    except ValueError:
        return dict(result, status="INVALID_AIS_COORDINATES")
    if reference is None or observed is None or not all(isfinite(value) for value in (lat, lon)):
        return dict(result, status="UNPARSEABLE_AIS_TIME_OR_COORDINATES")
    difference = abs((observed - reference).total_seconds())
    if difference > 120:
        return dict(result, status="AIS_TIME_GAP_OVER_120_SECONDS", absolute_time_gap_seconds=difference)
    west, south, east, north = header["bounds_lonlat"]
    width, height = header["dimensions"]
    require(all(isfinite(value) for value in (west, south, east, north)) and west < east and south < north,
            "Exact valid closed georeferencing bounds required")
    x = width * (lon - west) / (east - west)
    y = height * (north - lat) / (north - south)
    if not 0 <= x < width or not 0 <= y < height:
        return dict(result, status="AIS_OUTSIDE_HEADER_BOUNDS", absolute_time_gap_seconds=difference)
    distances, inside = [], []
    for group in masks:
        if not group:
            distances.append(None)
            inside.append(False)
            continue
        bitmap = mask.decode(mask.merge(group))
        rows, columns = np.nonzero(bitmap)
        distances.append(float(np.hypot(columns + .5 - x, rows + .5 - y).min()) if rows.size else None)
        inside.append(bool(bitmap[floor(y), floor(x)]))
    return dict(result, status="ONE_AIS_PROXIMITY_DIAGNOSTIC", absolute_time_gap_seconds=difference,
                nearest_foreground_center_pixels_class0=distances[0], nearest_foreground_center_pixels_class1=distances[1],
                point_in_raster_class0=inside[0], point_in_raster_class1=inside[1])


def examine(archive, headers):
    results = []
    for ordinal, header in enumerate(headers, 1):
        stem = PurePosixPath(header["name"]).stem
        paths, unresolved = match_train_metadata(archive, stem)
        if unresolved:
            results.append(dict(sample=ordinal, status="AMBIGUOUS_OR_MISSING_TRAIN_METADATA", unresolved=unresolved))
            continue
        blobs = {role: archive.read_member(name) for role, name in paths.items()}
        width, height = header["dimensions"]
        labels, masks = audit_labels(blobs["txt"], document(blobs["coco"]), stem, width, height)
        ais = audit_ais(blobs["ais"], header, masks)
        members = {role: dict(bytes=len(raw), sha256=sha256(raw).hexdigest(),
                              original_zip_crc32=f"{archive.members[paths[role]].CRC:08x}")
                   for role, raw in blobs.items()}
        results.append(dict(sample=ordinal, status="TRAIN_METADATA_COMPARED", labels=labels, ais=ais, members=members))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("recovered", "recovery-safe", "catalogue-safe", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical safe output required")
    control, headers = select_train_headers(pinned_document(args.recovered, RECOVERED),
                                            pinned_document(args.recovery_safe, RECOVERY),
                                            pinned_document(args.catalogue_safe, CATALOGUE))
    permitted = {name for header in (control, *headers)
                 for name in expected_paths(PurePosixPath(header["name"]).stem).values()}
    transport = AuthorRange()
    with BoundedZip(transport.fetch, ARCHIVE_BYTES, DIRECTORY_SHA256, permitted) as archive:
        control_digests = verify_positive_control(archive, control)
        rows = examine(archive, headers)
    result = dict(status=STATUS, archive_source=URL, source_record_id=18391027, archive_bytes=ARCHIVE_BYTES,
                  archive_directory_sha256=DIRECTORY_SHA256, recovered_proof_sha256=RECOVERED[1],
                  recovery_acceptance_sha256=RECOVERY[1], catalogue_readback_sha256=CATALOGUE[1],
                  reserved_family_overlap=0, new_train_families=len(headers), prior_qa_families=1,
                  positive_control_exact_three_member_sha256_verified=True,
                  positive_control_member_sha256=control_digests,
                  compared_train_members=sum(row["status"] == "TRAIN_METADATA_COMPARED" for row in rows),
                  http_206_requests=transport.requests, received_bytes=transport.bytes, samples=rows,
                  metadata_only=True, image_member_reads=0, tiff_pixel_decoding=False,
                  official_test_member_reads=0, reserved_member_reads=0, model_forward_calls=0,
                  annotation_completeness_established=False, class0_official_mapping_established=False,
                  ais_projection_scope="north-up header bounds approximation; unspecified timestamp offset assumed UTC")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(STATUS)


if __name__ == "__main__":
    main()
