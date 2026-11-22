"""Recover pinned publisher chips and labels using exact public ZIP ranges."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import struct
from time import monotonic, sleep
from zipfile import ZipFile

from PIL import Image
import requests


ARCHIVE = "https://zenodo.org/api/records/18391027/files/UEIKAP_SAR.zip/content"
LENGTH = 1_571_288_093
CENTRAL = "e7a969a39b4f6723d9293acdc9297e917d2830fcba54eeac3dd22b26b4f56610"
ROSTER = "0d78b4611e7db88a146f5bad489b6301e64e1d1eb499d536fd414e7b134c456d"
ROSTER_WORKER = "99f6eca5e3018e16c31d01a1148476adf83d521af420043eb6b2fce2fc97e2af"
PROTOCOL = "fbcc4d067351a7a7d2db226e6261a13c2f399eed59be414226e6ae29b6553bf2"
OLD = "abe5ec8c385ec9874b12de22192fce0fbf0eb3c977568513193d87e1b9f8cfa8"
MAX_RANGE = 24 * 1024 ** 2
MAX_GAP = 1024 ** 2
TRANSFER_CAP = 440_000_000


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def save(path, data):
    with path.open("xb") as f:
        f.write(data)
    return digest(data)


def save_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return save(path, raw)


class Ranges:
    def __init__(self, session, log):
        self.session = session
        self.log = log
        self.charged = 0
        self.received = 0
        self.requests = 0
        self.last = 0

    def get(self, start, stop):
        length = stop - start
        need(0 <= start < stop <= LENGTH and length <= MAX_RANGE, "Unapproved range dimensions")
        value = f"bytes={start}-{stop - 1}"
        for attempt in range(1, 4):
            need(self.charged + length <= TRANSFER_CAP, "Bounded transfer reached")
            sleep(max(0, self.last + 0.75 - monotonic()))
            self.last = monotonic()
            self.charged += length
            self.requests += 1
            event = dict(start=start, stop=stop, attempt=attempt)
            pause = 8 * attempt
            try:
                with self.session.get(ARCHIVE, headers={"Range": value, "Accept-Encoding": "identity"},
                                      stream=True, timeout=(20, 45), allow_redirects=False) as res:
                    event["response"] = res.status_code
                    if res.status_code in (429, 503):
                        retry = res.headers.get("Retry-After", "")
                        pause = max(60, int(retry)) if retry.isdecimal() else 60
                    else:
                        need(res.status_code == 206 and
                             res.headers.get("Content-Range") == f"bytes {start}-{stop - 1}/{LENGTH}" and
                             int(res.headers.get("Content-Length", "-1")) == length and
                             res.headers.get("Content-Encoding", "identity") == "identity", "Nonexact ZIP range")
                        data = bytearray()
                        for chunk in res.iter_content(1024 * 1024):
                            self.received += len(chunk)
                            data.extend(chunk)
                            need(len(data) <= length, "Oversized range body")
                        if len(data) != length:
                            raise requests.exceptions.ChunkedEncodingError("Truncated exact range")
                        event["bytes"] = length
                        return bytes(data)
            except requests.RequestException as exc:
                event["error"] = type(exc).__name__
            finally:
                self.log.write(json.dumps(event) + "\n")
                self.log.flush()
            if attempt < 3:
                sleep(pause)
        raise RuntimeError("Publisher range exhausted its bounded retries")


class SparseArchive:
    """Allow ZIP reads only inside the fixed directory or one verified batch."""
    def __init__(self, offset, tail):
        self.directory = (offset, tail)
        self.batch = (0, b"")
        self.pos = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, LENGTH)[whence] + offset
        need(0 <= self.pos <= LENGTH, "ZIP seek exceeded archive")
        return self.pos

    def read(self, size=-1):
        end = LENGTH if size < 0 else min(LENGTH, self.pos + size)
        if self.pos == end:
            return b""
        for start, data in (self.directory, self.batch):
            if start <= self.pos <= end <= start + len(data):
                block = data[self.pos - start:end - start]
                self.pos = end
                return block
        raise ValueError("ZIP attempted to read outside verified ranges")


def central(ranges):
    footer = ranges.get(LENGTH - 22, LENGTH)
    magic, disk, central_disk, n_disk, n, size, offset, note = struct.unpack("<4s4H2LH", footer)
    need((magic, disk, central_disk, n_disk, n, note) == (b"PK\x05\x06", 0, 0, 2615, 2615, 0)
         and size == 427_621 and offset + size == LENGTH - 22, "Published ZIP directory layout differs")
    data = ranges.get(offset, offset + size)
    need(digest(data) == CENTRAL, "Exact publisher central directory required")
    return offset, SparseArchive(offset, data + footer)


def batches(entries, selected, offset):
    entries = sorted(entries, key=lambda item: item.header_offset)
    offsets = [e.header_offset for e in entries] + [offset]
    need(len(offsets) == len(set(offsets)), "ZIP local headers are not unique")
    runs = []
    for left, right, entry in zip(offsets, offsets[1:], entries):
        if entry.filename not in selected:
            continue
        need(0 < right - left <= MAX_RANGE and not entry.flag_bits & 1 and
             entry.compress_type in (0, 8) and 0 < entry.file_size <= 4_000_000, "Unsupported ZIP member")
        if runs and left - runs[-1][1] <= MAX_GAP and right - runs[-1][0] <= MAX_RANGE:
            runs[-1][1] = right
            runs[-1][2].append(entry)
        else:
            runs.append([left, right, [entry]])
    need(sum(len(group) for _, _, group in runs) == len(selected), "Selected intake member missing")
    return runs


def check_png(raw):
    with Image.open(BytesIO(raw), formats=["PNG"]) as picture:
        need(picture.mode == "RGB" and picture.size == (512, 512) and
             picture.n_frames == 1, "Exact publisher 512-square RGB chip required")
        picture.verify()
    with Image.open(BytesIO(raw), formats=["PNG"]) as picture:
        return digest(picture.tobytes())


def check_label(raw):
    counts = Counter()
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            category, *tokens = line.split()
            coordinates = [float(token) for token in tokens]
            need(category in ("0", "1") and len(coordinates) >= 6 and len(coordinates) % 2 == 0 and
                 all(math.isfinite(x) and 0 <= x <= 1 for x in coordinates), "Invalid publisher voxel coordinates")
            counts[category] += 1
    need(counts["0"] > 0, "The locked publication census requires a class-0 hull per chip")
    return counts


def intake(a):
    roster_raw = a.roster.read_bytes()
    receipt_raw = a.roster_result.read_bytes()
    observed = json.loads(a.roster_terminal.read_bytes())
    need(observed["phase"] == "roster" and observed["actual_exit"] == 0 and observed["output_footer_verified"]
         and observed["result_sha256"] == digest(receipt_raw) and observed["worker_sha256"] == ROSTER_WORKER
         and digest(roster_raw) == ROSTER and digest(a.protocol.read_bytes()) == PROTOCOL, "Frozen intake authority")
    roster = json.loads(roster_raw)
    ids = list(range(1, 343))
    need(roster["expected_ids"] == ids == [r["image_id"] for r in roster["items"]]
         and roster["archive_url"] == ARCHIVE and roster["archive_bytes"] == LENGTH
         and roster["central_directory_sha256"] == CENTRAL, "Roster lineage differs")
    old_raw = a.prior_provenance.read_bytes()
    need(digest(old_raw) == OLD, "Prior scored member provenance differs")
    old = json.loads(old_raw)
    previous_pixels = {item["decoded_rgb_sha256"] for item in old if "decoded_rgb_sha256" in item}
    need(len(previous_pixels) == 108, "Prior 108 decoded-image hashes are required")
    new_pixels = set()
    out = a.output
    out.mkdir()
    (out / "images").mkdir()
    (out / "labels").mkdir()

    output = {"image": {}, "label": {}}
    prov = {}
    with requests.Session() as session, (out / "RANGES_PRIVATE.jsonl").open("x", encoding="utf-8") as log:
        # The input URL is the public publisher resource, and requires no authentication.
        session.trust_env = False
        ranges = Ranges(session, log)
        offset, sparse = central(ranges)
        with ZipFile(sparse) as archive:
            entries = archive.infolist()
            need(len(entries) == 2615 and len({e.filename for e in entries}) == 2615, "Duplicate ZIP members")
            for kind, directory, extension in (("image", "images", ".png"), ("label", "labels", ".txt")):
                lookup = {r[f"{kind}_zip_name"]: r for r in roster["items"]}
                need(len(lookup) == len(ids), "Duplicated roster member")
                for start, stop, group in batches(entries, lookup, offset):
                    sparse.batch = (start, ranges.get(start, stop))
                    for member in group:
                        chosen = lookup[member.filename]
                        num = chosen["image_id"]
                        path = f"{directory}/{num:06d}{extension}"
                        raw = archive.read(member)  # Standard ZIP reader also verifies the publisher CRC.
                        info = dict(path=path, bytes=len(raw), sha256=digest(raw))
                        row = dict(image_id=num, acquisition_group=chosen["acquisition_group"])
                        published = dict(image_id=num, zip_member=member.filename, sha256=info["sha256"],
                                         bytes=len(raw), zip_crc32=f"{member.CRC:08x}", kind=kind)
                        if kind == "image":
                            pixels = check_png(raw)
                            need(pixels not in new_pixels and pixels not in previous_pixels, "Duplicate decoded chip")
                            new_pixels.add(pixels)
                            row.update(original_hw=[512, 512], image_ref=info, rgb_sha256=pixels)
                            published["decoded_rgb_sha256"] = pixels
                        else:
                            row.update(count_by_class=check_label(raw), label_ref=info)
                        need(save(out / path, raw) == info["sha256"], "Saved member identity")
                        output[kind][num] = row
                        prov[(num, kind)] = published
                    print(json.dumps(dict(kind=kind, admitted=len(output[kind]), total=len(ids))), flush=True)
                need(sorted(output[kind]) == ids, "Intake did not recover every locked member")

    group_count = len({r["acquisition_group"] for r in output["image"].values()})
    hulls = sum(r["count_by_class"]["0"] for r in output["label"].values())
    need(group_count == 13 and hulls == 528, "Full author hull/acquisition count differs")
    result = dict(status="UEIKAP_ADDITIONAL_AUTHOR_INTAKE_COMPLETE", source_record=18391027,
                  archive_bytes=LENGTH, central_directory_sha256=CENTRAL, cohort_roster_sha256=ROSTER,
                  roster_result_sha256=digest(receipt_raw), protocol_sha256=PROTOCOL, images=len(ids),
                  labeled_hulls=hulls, acquisition_groups=group_count, decoded_rgb_exact_overlap_previous_108=0,
                  received_bytes=ranges.received, charged_range_bytes=ranges.charged, http_requests=ranges.requests,
                  worker_sha256=digest(Path(__file__).read_bytes()))
    for kind in ("image", "label"):
        name = kind + "s_roster"
        result[name + "_sha256"] = save_json(out / (name + ".json"),
            dict(expected_ids=ids, items=[output[kind][i] for i in ids]))
    result["member_provenance_sha256"] = save_json(out / "member_provenance.json", [prov[key] for key in sorted(prov)])
    got = save_json(out / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=got)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("roster", "roster-result", "roster-terminal", "protocol", "prior-provenance", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    intake(parser.parse_args())


if __name__ == "__main__":
    main()

