"""Admit the 74 locked Instance-RSDD reserve JPEGs and compare decoded pixels with the historical test."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath
import re
from time import perf_counter
from types import ModuleType
import zipfile

from PIL import Image, ImageOps


SOURCE_SHA = "8162a78e3a05457180225a38422f5f2637f803781099c10cadd4f32c4ae854da"
RESERVE_SHA = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
TEST_MANIFEST_SHA = "16a546e96d6e76d3afbfbe4f380b182505494818b35f1c8bcc960e6d4d1b2184"
EXPECTED = {"GF-3":42, "TerraSAR-X":32}
STATUS = "INSTANCE_RSDD_RESERVE74_IMAGE_INTAKE_COMPLETE_V1"


def load_source(path):
    source = path.read_bytes()
    if sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError("Original archive reader fingerprint differs")
    old = ModuleType("rsdd_archive_intake")
    old.__file__ = str(path)
    exec(compile(source, str(path), "exec"), old.__dict__)
    return old


def population(a, old):
    groups = old.read_parent_groups(a.earlier)
    test = json.loads(old.checked_bytes(a.test_manifest, TEST_MANIFEST_SHA))
    prior = test["items"]
    prior_groups = {row["acquisition_group"] for row in prior}
    old.check(len(prior) == len({r["rgb_sha256"] for r in prior}) == 740 and len(prior_groups) == 112
              and test["expected_ids"] == [r["image_id"] for r in prior], "Historical decoded RGB accession")
    reader = csv.DictReader(StringIO(old.checked_bytes(a.roster, RESERVE_SHA).decode("utf-8")))
    old.check(reader.fieldnames == ["author_train_image_id", "file_name", "paper_scene", "sensor"], "Locked CSV schema")
    roster = list(reader)
    old.check(len(roster) == len({r["file_name"] for r in roster}) == len({int(r["author_train_image_id"]) for r in roster}) == 74,
              "Complete ordered author reserve accession")
    for r in roster:
        name = r["file_name"]
        old.check(re.fullmatch(r"\d+_\d+_\d+\.jpg", name) and groups[int(name.split("_")[0])] == (int(r["paper_scene"]),r["sensor"]),
                  "Original source name, product, parent and sensor")
    parents = {int(r["paper_scene"]):r["sensor"] for r in roster}
    old.check(set(parents).isdisjoint(prior_groups) and set(parents) | prior_groups == set(range(1,126))
              and Counter(parents.values()) == {"GF-3":8,"TerraSAR-X":5} and Counter(r["sensor"] for r in roster) == EXPECTED
              and {r["file_name"] for r in roster}.isdisjoint(r["source_image_name"] for r in prior), "Frozen acquisition split")
    return roster, prior


def selected_ranges(entries, wanted, old):
    entries = sorted(entries, key=lambda m:m.header_offset)
    spans = []
    for idx, entry in enumerate(entries):
        path = PurePosixPath(entry.filename)
        if path.parent.name == "JPEGImages" and path.name in wanted:
            end = entries[idx+1].header_offset if idx+1 < len(entries) else old.DIRECTORY_START
            old.check(entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) and not entry.flag_bits & 1,
                      "Unencrypted original JPEG archive member")
            spans.append((entry.header_offset, end, entry))
    old.check(len(spans) == len(wanted) and {PurePosixPath(e.filename).name for _,_,e in spans} == set(wanted),
              "One archive record per selected JPEG")
    # Consecutive selected members may share one range. A gap could contain a different member.
    return old.coalesce_ranges(spans, gap=0, max_size=2**20)


def intake(a, decoder):
    started = perf_counter()
    old = load_source(a.source)
    old.check(not a.output.exists() and 1 <= a.workers <= 16, "Fresh and bounded worker")
    roster, prior = population(a, old)
    wanted = {r["file_name"]:r for r in roster}
    batches = selected_ranges(old.read_directory(a.earlier), wanted, old)
    fetch = old.RangeReader(a)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/"images").mkdir()

    def collect(batch):
        lo, hi, parts = batch
        span = fetch.read(lo, hi)
        items = []
        for start, end, entry in parts:
            raw = old.extract_member(entry, span[start-lo:end-lo])
            name = PurePosixPath(entry.filename).name
            row = wanted[name]
            with Image.open(BytesIO(raw)) as image:
                old.check(image.format == "JPEG" and image.size == (512,512), "Original JPEG format and dimensions")
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb_hash = old.digest(rgb.tobytes())
            decoded, scales = decoder(raw, draft_size=None)
            old.check(decoded.mode == "RGB" and rgb.size == decoded.size == (512,512) and scales == (1.0,1.0)
                      and rgb_hash == old.digest(decoded.tobytes()), "Qualified RF-DETR decoder parity")
            with (a.output/"images"/name).open("xb") as out:
                out.write(raw)
            pid, y, x = map(int, Path(name).stem.split("_"))
            items.append(dict(image_id=int(row["author_train_image_id"]), source_image_name=name, original_hw=[512,512],
                              image_ref=dict(path="images/"+name, sha256=old.digest(raw), bytes=len(raw)), rgb_sha256=rgb_hash,
                              acquisition_group=int(row["paper_scene"]), sensor=row["sensor"], product_id=pid, row=y, column=x,
                              zip_member=entry.filename, zip_crc32=f"{entry.CRC:08x}"))
        return items

    print(json.dumps(dict(status="RESERVE_INTAKE_STARTED", requested_images=len(roster), ranges=len(batches))),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        collected = {r["source_image_name"]:r for batch in pool.map(collect, batches) for r in batch}
    items = [collected[name] for name in wanted]
    image_manifest = dict(expected_ids=[r["image_id"] for r in items], items=items)
    manifest_sha = old.write_json(a.output/"IMAGE_MANIFEST_PRIVATE.json", image_manifest)
    old_rgb = {r["rgb_sha256"] for r in prior}
    rgb = Counter(r["rgb_sha256"] for r in items)
    cross = [r for r in items if r["rgb_sha256"] in old_rgb]
    record = dict(status=STATUS, worker_sha256=old.digest(Path(__file__).read_bytes()), archive_reader_sha256=SOURCE_SHA,
                  roster_sha256=RESERVE_SHA, historical_test_manifest_sha256=TEST_MANIFEST_SHA,
                  image_manifest_sha256=manifest_sha, zip_archive_bytes=old.ARCHIVE_BYTES,
                  zip_central_directory_sha256=old.DIRECTORY_SHA, source_paper_layout_sha256=old.PAPER_TEXT_SHA,
                  transfer_method="local archive ranges" if a.archive else "public Google Drive HTTPS ranges",
                  transfer_ranges=len(batches), transfer_bytes=sum(hi-lo for lo,hi,_ in batches),
                  source_split="author train", comparison_split="complete historical author test", images=len(items),
                  sensor_counts=dict(EXPECTED), acquisition_groups=13, historical_test_images=len(prior),
                  historical_test_groups=112, common_test_parents=0, member_crc32_verified=len(items),
                  rgb_decode_verified=len(items), exact_qualified_rf_rgb=len(items),
                  decoded_rgb_method="SHA-256 of native 512x512 RGB bytes, Pillow EXIF transpose and full RF-DETR decoder agree",
                  encoded_unique=len({r["image_ref"]["sha256"] for r in items}), rgb_unique=len(rgb),
                  within_reserve_duplicate_extra_images=sum(n-1 for n in rgb.values()),
                  historical_test_rgb_matching_reserve_images=len(cross), historical_test_rgb_matching_unique=len({r['rgb_sha256'] for r in cross}),
                  historical_test_rgb_matching_reserve_images_by_sensor={s:sum(r['sensor']==s for r in cross) for s in EXPECTED},
                  hrsid_training_rgb_check="pending separate pinned complete training image comparison",
                  primary_exposure_qualified=False, target_polygon_rows_opened=0, model_forwards=0, optimizer_updates=0,
                  elapsed_seconds=perf_counter()-started)
    result_sha = old.write_json(a.output/"RESULT_SAFE.json", record)
    print(json.dumps(dict(status=STATUS, result_sha256=result_sha)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("earlier", "source", "roster", "test-manifest", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--proxy")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    intake(args, decode_pil_image_bytes)


if __name__ == "__main__":
    main()
