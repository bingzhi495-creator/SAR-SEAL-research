"""Locate and compare only the official PSeg-SSDD RAR5 TRAIN annotation records."""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPException
from io import BytesIO
import json
from os import fstat, fsync
from pathlib import Path
import re
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zlib import crc32

SIZE = 1_043_241_197
URL = "https://drive.usercontent.google.com/download?id=1glNJUGotrbEyk43twwB9556AdngJsynZ&export=download&confirm=t"
TRAIN_JSON = "Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style/annotations/train.json"
TRAIN_XML = re.compile(r"^Official-SSDD-OPEN/(PSeg_SSDD|BBox_RBox_PSeg_SSDD)/voc_style/Annotations_train/([^/]+)[.]xml$")
SIGNATURE = b"Rar!\x1a\x07\x01\x00"
LIMIT = 8 * 1024**2


@dataclass(frozen=True)
class Member:
    role: str
    start: int
    end: int
    packed: int
    unpacked: int
    crc: int


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def read_exact(stream, count):
    payload = stream.read(count)
    need(len(payload) == count, "Truncated RAR field")
    return payload


def vint(stream, width=10):
    octets = bytearray()
    value = 0
    for index in range(width):
        byte = read_exact(stream, 1)[0]
        octets.append(byte)
        value |= (byte & 127) << (index * 7)
        if byte < 128:
            return value, octets
    raise ValueError("Overlong RAR vint")


def value(stream):
    return vint(stream)[0]


def role_for(name):
    if name == TRAIN_JSON:
        return "combined_coco_train"
    match = TRAIN_XML.fullmatch(name)
    return match[1] + "_voc_train" if match else None


def source_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def local_members(stream):
    need(read_exact(stream, len(SIGNATURE)) == SIGNATURE, "Exact unencrypted RAR5 signature")
    verified_headers = 0
    main_seen = False
    selected = []
    while stream.tell() < SIZE:
        start = stream.tell()
        expected_crc = int.from_bytes(read_exact(stream, 4), "little")
        header_size, size_bytes = vint(stream, 3)
        need(0 < header_size <= 2**21 - 1, "RAR5 header size")
        raw = read_exact(stream, header_size)
        need(crc32(size_bytes + raw) == expected_crc, "RAR5 header CRC32")
        header = BytesIO(raw)
        kind, flags = value(header), value(header)
        extra = value(header) if flags & 1 else 0
        packed = value(header) if flags & 2 else 0
        end = stream.tell() + packed
        verified_headers += 1
        need(0 <= extra <= header_size and end <= SIZE and kind in (1, 2, 3, 5), "RAR5 block bounds and known type")
        if kind == 1:
            need(not main_seen and not (value(header) & 0x5), "Single non-solid, non-split archive")
            main_seen = True
        elif kind == 2:
            need(main_seen, "Missing RAR5 main header")
            file_flags, unpacked, _attrs = value(header), value(header), value(header)
            if file_flags & 2:
                read_exact(header, 4)
            crc = int.from_bytes(read_exact(header, 4), "little") if file_flags & 4 else None
            compression, _host, name_size = value(header), value(header), value(header)
            name = read_exact(header, name_size).decode("utf-8").replace("\\", "/")
            need(header.tell() <= header_size - extra, "RAR5 file field bounds")
            role = role_for(name)
            if role:
                extra_records = BytesIO(raw[header_size-extra:])
                while extra_records.tell() < extra:
                    length = value(extra_records)
                    record = BytesIO(read_exact(extra_records, length))
                    need(length > 0 and value(record) != 1, "No encrypted TRAIN member")
                need(flags & 0x38 == 0 and file_flags & 9 == 0 and not compression & 0x40 and
                     crc is not None and 0 < packed <= LIMIT, "TRAIN member independently packed, CRC known")
                selected.append(Member(role, start, end, packed, unpacked, crc))
        elif kind == 5:
            need(end == SIZE and main_seen, "Exact archive end")
            return selected, verified_headers
        stream.seek(end)
    raise ValueError("Missing RAR5 end-of-archive marker")


def coalesced(records):
    spans = []
    for member in records:
        if spans and spans[-1][-1].end == member.start and spans[-1][-1].role == member.role:
            spans[-1].append(member)
        else:
            spans.append([member])
    return spans


def fetch(start, end):
    count = end - start
    need(0 < count <= LIMIT, "Bounded admitted TRAIN span")
    request = Request(URL + "&range_version=" + str(start),
                      headers={"Range": f"bytes={start}-{end-1}", "Accept-Encoding": "identity",
                               "User-Agent": "Python/3 SSDD public training record verification"})
    try:
        with urlopen(request, timeout=18) as response:
            scope = response.headers.get("Content-Range", "")
            if response.status != 206 or scope != f"bytes {start}-{end-1}/{SIZE}":
                return dict(status=response.status, content_range=scope, matched=False, failure="Unexpected HTTP range")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                return dict(status=response.status, content_range=scope, matched=False, failure="Unexpected content encoding")
            body = response.read(count + 1)
            return dict(status=response.status, content_range=scope, remote_sha256=sha256(body).hexdigest(),
                        remote_bytes=len(body), matched=False, last_modified=response.headers.get("Last-Modified"))
    except HTTPError as error:
        return dict(status=error.code, matched=False, failure="HTTP error")
    except (URLError, TimeoutError, HTTPException) as error:
        return dict(status=None, matched=False, failure=type(error).__name__)


def main():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("map", "verify"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "verify" and (args.output is None or args.output.exists()):
        parser.error("--mode verify requires a fresh --output path")
    before = source_identity(args.archive.stat())
    need(before[2] == SIZE, "Exact candidate archive size")
    started = perf_counter()
    with args.archive.open("rb") as stream:
        need(source_identity(fstat(stream.fileno())) == before, "Stable opened archive")
        records, count = local_members(stream)
        expected = {"combined_coco_train": 1, "PSeg_SSDD_voc_train": 928, "BBox_RBox_PSeg_SSDD_voc_train": 928}
        need({role: sum(row.role == role for row in records) for role in expected} == expected and
             len(records) == sum(expected.values()), "Exact trained annotation populations")
        coco, = [row for row in records if row.role == "combined_coco_train"]
        need(coco.unpacked == 1_526_458 and coco.crc == 0xba72d160, "Same qualified COCO TRAIN annotation")
        spans = coalesced(records)
        summary = [dict(role=items[0].role, records=len(items), start=items[0].start, end=items[-1].end,
                        unpacked_bytes=sum(row.unpacked for row in items)) for items in spans]
        if args.mode == "map":
            print(json.dumps(dict(verified_headers=count, spans=len(spans), first=summary[:7]), sort_keys=True))
            return
        need(len(spans) <= 12 and all(row["end"] - row["start"] <= LIMIT for row in summary),
             "Only a small set of contiguous authorized TRAIN spans may be fetched")
        matches = []
        for row in sorted(summary, key=lambda item: (item["role"] != "combined_coco_train", item["start"])):
            start, end = row["start"], row["end"]
            stream.seek(start)
            local_digest = sha256(read_exact(stream, end-start)).hexdigest()
            remote = fetch(start, end)
            remote["matched"] = remote.get("remote_bytes") == end-start and remote.get("remote_sha256") == local_digest
            matches.append(dict(**row, local_sha256=local_digest, **remote))
        need(source_identity(fstat(stream.fileno())) == before == source_identity(args.archive.stat()),
             "Archive identity remained stable")
    result = dict(status="P393_OFFICIAL_SSDD_TRAIN_RAR_RANGE_COMPARISON_ACTUAL", archive_bytes=SIZE,
                  verified_local_headers=count, TRAIN_only=True, TEST_member_payload_opened=False,
                  image_payload_opened=False, raw_labels_written=False, whole_archive_authenticated=False,
                  annotation_records_by_role=expected, matches=matches,
                  matched_records=sum(row["records"] for row in matches if row["matched"]),
                  elapsed_seconds=perf_counter()-started)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as output:
        output.write(raw)
        output.flush()
        fsync(output.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest(),
                          matched_records=result["matched_records"], spans=matches), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
