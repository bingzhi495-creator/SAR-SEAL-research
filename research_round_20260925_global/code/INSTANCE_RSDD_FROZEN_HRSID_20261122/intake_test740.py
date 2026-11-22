"""Retrieve and verify the exact JPEGs in the public Instance-RSDD test roster."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import struct
import threading
from time import perf_counter, sleep
import zipfile
import zlib

import requests
from PIL import Image, ImageOps


ARCHIVE_ID = "1PJxr7Tbr_ZAzuG8MNloDa4mLaRYCD3qc"
ARCHIVE_BYTES = 1_334_274_952
DIRECTORY_START = 1_333_234_226
DIRECTORY_SHA = "0d270fdb35120a341dff3940790d13173d603ee05b99ec9dd57efa1f82521d9e"
ROSTER_SHA = "72c4b9af8f8b0bd0df31a3a3f7df6a082390688f0942eed23f106316a2d5d98b"
PAPER_TEXT_SHA = "b9c25ef594c1bccd7dd84aa4b08b5deaab40a5188c6a5c1c1458fbe52681c396"
EXPECTED_SENSORS = {"GF-3": 583, "TerraSAR-X": 157}


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def checked_bytes(path, expected):
    data = path.read_bytes()
    check(digest(data) == expected, "Source fingerprint differs: " + path.name)
    return data


def read_directory(earlier):
    data = checked_bytes(earlier / "gdrive-zip-directory.bin", DIRECTORY_SHA)
    tail = (earlier / "gdrive-zip-tail.bin").read_bytes()
    end = tail[tail.rindex(b"PK\x05\x06"):]
    check(len(end)==22 and struct.unpack("<4s4H2IH", end)[1:] ==
          (0,0,14015,14015,len(data),DIRECTORY_START,0), "Unexpected ZIP archive metadata")
    with zipfile.ZipFile(BytesIO(data+end)) as source:
        entries = source.infolist()
        for member in entries:
            member.header_offset += DIRECTORY_START
    return entries


def read_parent_groups(earlier):
    data = checked_bytes(earlier / "rsdd-paper.txt", PAPER_TEXT_SHA).decode("utf-8")
    appendix = data.split("附表 1 RSDD-SAR数据集详细信息")[-1]
    pattern = r"^\s*(\d+)\s+(GF-3|TerraSAR-X)\s+.*?\s+(\d+)(?:-(\d+))?\s*$"
    groups = {}
    for line in appendix.splitlines():
        match = re.match(pattern, line)
        if match:
            group, sensor, first, last = match.groups()
            for pid in range(int(first), int(last or first)+1):
                check(pid not in groups, "Duplicate product in author appendix")
                groups[pid] = int(group), sensor
    check(set(groups)==set(range(193)) and {group for group,_ in groups.values()}==set(range(1,126)),
          "Incomplete Table 1 acquisition mapping")
    return groups


def coalesce_ranges(spans, gap, max_size):
    merged = []
    for lo, hi, obj in spans:
        check(0 <= lo < hi and hi - lo <= max_size, "Single source range exceeds the transfer bound")
        if merged and lo - merged[-1][1] <= gap and hi - merged[-1][0] <= max_size:
            begin, end, contents = merged[-1]
            check(end <= lo, "Overlapping ZIP member extents")
            merged[-1] = begin, hi, contents + [(lo, hi, obj)]
        else:
            merged.append((lo, hi, [(lo, hi, obj)]))
    return merged


def extract_member(member, span):
    header = struct.Struct("<4s5H3I2H")
    check(len(span) >= header.size, "ZIP local member is truncated")
    magic, _, flags, method, _, _, _, _, _, name_len, extra_len = header.unpack_from(span)
    check(magic == b"PK\x03\x04" and not flags & 1 and method == member.compress_type,
          "ZIP local member flags/method differ")
    start = header.size + name_len + extra_len
    check(span[header.size:header.size+name_len] == member.filename.encode("ascii"), "ZIP local member name differs")
    end = start + member.compress_size
    check(end <= len(span), "ZIP compressed bytes incomplete")
    compressed = span[start:end]
    data = zlib.decompress(compressed, -15) if method == zipfile.ZIP_DEFLATED else compressed
    check(len(data) == member.file_size and zlib.crc32(data) == member.CRC, "ZIP member CRC or size differs")
    return data


def public_download_url(earlier):
    form = (earlier / "gdrive-response.bin").read_text(encoding="utf-8")
    match = re.search(r'name="uuid" value="([a-z0-9-]{36})"', form)
    check(match is not None and "rsdd.zip" in form, "Expected standard public Drive download form")
    return "https://drive.usercontent.google.com/download?id=" + ARCHIVE_ID + "&export=download&confirm=t&uuid=" + match.group(1)


class RangeReader:
    def __init__(self, args):
        self.local = threading.local()
        self.proxy = args.proxy
        self.archive = args.archive
        self.url = None if self.archive else public_download_url(args.earlier)
        if self.archive:
            check(self.archive.stat().st_size == ARCHIVE_BYTES, "Supplied archive length differs")
            with self.archive.open("rb") as f:
                f.seek(DIRECTORY_START)
                check(digest(f.read(1_040_704)) == DIRECTORY_SHA, "Supplied archive directory differs")

    def read(self, start, end):
        if self.archive:
            with self.archive.open("rb") as f:
                f.seek(start)
                data = f.read(end-start)
            check(len(data)==end-start, "Local archive short read")
            return data
        if not hasattr(self.local,"session"):
            self.local.session = requests.Session()
            self.local.session.trust_env = False
            if self.proxy:
                self.local.session.proxies.update(http=self.proxy,https=self.proxy)
        for attempt in range(4):
            try:
                expect = end-start
                with self.local.session.get(self.url, stream=True, timeout=(12,30),
                                            headers={"Range":f"bytes={start}-{end-1}","Accept-Encoding":"identity"}) as r:
                    check(r.status_code == 206, f"Public range returned HTTP {r.status_code}")
                    check(r.headers.get("Content-Range") == f"bytes {start}-{end-1}/{ARCHIVE_BYTES}", "Source range differs")
                    data = bytearray()
                    for chunk in r.iter_content(65536):
                        data.extend(chunk)
                        check(len(data) <= expect, "Source returned bytes beyond requested range")
                    check(len(data) == expect, "Source returned short range")
                    return data
            except (requests.RequestException,ValueError) as e:
                if attempt == 3:
                    why = str(e) if isinstance(e, ValueError) else type(e).__name__
                    raise ValueError(f"Public archive range [{start},{end}) failed: {why}") from None
                sleep(.5*(attempt+1))


def write_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with path.open("xb") as f:
        f.write(data)
    return digest(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("earlier","roster","output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--proxy")
    parser.add_argument("--workers",type=int,default=12)
    args = parser.parse_args()
    started = perf_counter()
    check(not args.output.exists() and 1 <= args.workers <= 16, "Fresh and bounded intake")
    entries = read_directory(args.earlier)
    groups = read_parent_groups(args.earlier)
    roster = json.loads(checked_bytes(args.roster,ROSTER_SHA))["images"]
    check(len(roster)==len({row["id"] for row in roster})==len({row["file_name"] for row in roster})==740,
          "Unexpected author TEST population")
    by_name = {row["file_name"]:row for row in roster}
    ordered = sorted(entries,key=lambda m:m.header_offset)
    spans = []
    for idx, member in enumerate(ordered):
        path = PurePosixPath(member.filename)
        if path.name not in by_name or path.parent.name != "JPEGImages":
            continue
        check(idx+1<len(ordered) and re.fullmatch(r"\d+_\d+_\d+\.jpg",path.name)
              and member.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) and not member.flag_bits & 1,
              "Unexpected test ZIP member metadata")
        spans.append((member.header_offset,ordered[idx+1].header_offset,member))
    check(len(spans) == len({PurePosixPath(x[2].filename).name for x in spans}) == len(roster), "Incomplete test member join")
    ranges = coalesce_ranges(spans,gap=32768,max_size=2**20)
    fetch = RangeReader(args)
    (args.output/"images").mkdir(parents=True)
    print(json.dumps(dict(status="INTAKE_STARTED",requested_images=len(roster),transfer_ranges=len(ranges),
                          transfer_bytes=sum(hi-lo for lo,hi,_ in ranges))),flush=True)

    def collect(batch):
        lo,hi,parts = batch
        data = fetch.read(lo,hi)
        gathered = []
        for a,b,member in parts:
            raw = extract_member(member,data[a-lo:b-lo])
            name = PurePosixPath(member.filename).name
            wanted = by_name[name]
            with Image.open(BytesIO(raw)) as img:
                check(img.format=="JPEG" and img.size==(wanted["width"],wanted["height"]), "Author JPEG dimensions differ")
                rgb = ImageOps.exif_transpose(img).convert("RGB")
                check(rgb.size==img.size, "Decoded RGB orientation changes dimensions")
                rgb_sha = digest(rgb.tobytes())
            pid,row,col = map(int,Path(name).stem.split("_"))
            group,sensor = groups[pid]
            path = args.output/"images"/name
            with path.open("xb") as output:
                output.write(raw)
            gathered.append(dict(image_id=wanted["id"],source_image_name=name,
                                 original_hw=[wanted["height"],wanted["width"]],
                                 image_ref=dict(path="images/"+name,sha256=digest(raw),bytes=len(raw)),
                                 rgb_sha256=rgb_sha,acquisition_group=group,sensor=sensor,
                                 product_id=pid,row=row,column=col,zip_member=member.filename,zip_crc32=f"{member.CRC:08x}"))
        return gathered

    results, checkpoint = {}, 40
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(collect,r) for r in ranges]
        for future in as_completed(pending):
            for item in future.result():
                results[item["source_image_name"]] = item
            if len(results)>=checkpoint:
                print(json.dumps(dict(verified_images=len(results),elapsed_seconds=round(perf_counter()-started,2))),flush=True)
                checkpoint += 40
    items = [results[entry["file_name"]] for entry in roster]
    sensors = dict(Counter(x["sensor"] for x in items))
    parent = {x["acquisition_group"] for x in items}
    check(len(items)==740 and sensors==EXPECTED_SENSORS and len(parent)==112, "Author TEST sensor/group population differs")
    image_manifest = dict(expected_ids=[x["image_id"] for x in items],items=items)
    manifest_sha = write_json(args.output/"IMAGE_MANIFEST_PRIVATE.json",image_manifest)
    summary = dict(status="INSTANCE_RSDD_TEST740_IMAGE_INTAKE_COMPLETE_V1",test_images=len(items),
                   test_roster_sha256=ROSTER_SHA,zip_archive_bytes=ARCHIVE_BYTES,zip_central_directory_sha256=DIRECTORY_SHA,
                   source_paper_layout_sha256=PAPER_TEXT_SHA,member_crc32_verified=len(items),rgb_decode_verified=len(items),
                   sensor_counts=sensors,acquisition_groups=len(parent),
                   encoded_unique=len({x["image_ref"]["sha256"] for x in items}),
                   rgb_unique=len({x["rgb_sha256"] for x in items}),image_manifest_sha256=manifest_sha,
                   optimizer_updates=0,model_forwards=0,target_polygon_rows_opened=0,seconds=perf_counter()-started)
    safe_sha = write_json(args.output/"RESULT_SAFE.json",summary)
    print(json.dumps(dict(status=summary["status"],result_sha256=safe_sha)),flush=True)


if __name__ == "__main__":
    main()
