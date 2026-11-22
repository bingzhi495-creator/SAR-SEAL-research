"""Extract the locked reserve JPEGs from official RSDD-SAR inputs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath
from time import perf_counter
from types import ModuleType
import zipfile

from PIL import Image, ImageOps


ROSTER_SHA = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
ARCHIVE_SOURCE_SHA = "8162a78e3a05457180225a38422f5f2637f803781099c10cadd4f32c4ae854da"
STATUS = "INSTANCE_RSDD_RESERVE74_CLEAN_CONVERSION_COMPLETE_V1"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def metadata_reader():
    code = Path(__file__).resolve().parent.parent
    path = code/"INSTANCE_RSDD_FROZEN_HRSID_20261122/intake_test740.py"
    data = path.read_bytes()
    need(digest(data) == ARCHIVE_SOURCE_SHA, "Exact published archive operations")
    reader = ModuleType("public_rsdd_archive")
    reader.__file__ = str(path)
    exec(compile(data,str(path),"exec"),reader.__dict__)
    return reader


def roster(path):
    source = path.read_bytes()
    need(digest(source) == ROSTER_SHA, "Rebuilt author roster differs from the published roster builder")
    reader = csv.DictReader(StringIO(source.decode("utf-8")))
    need(reader.fieldnames == ["author_train_image_id","file_name","paper_scene","sensor"], "Author roster schema")
    rows = {row["file_name"]:row for row in reader}
    need(len(rows) == 74 and len({int(row["author_train_image_id"]) for row in rows.values()}) == 74
         and len({row["paper_scene"] for row in rows.values()}) == 13
         and Counter(row["sensor"] for row in rows.values()) == {"GF-3":42,"TerraSAR-X":32},
         "Complete image-only reserve")
    return rows


def local_members(path, reader):
    need(path.stat().st_size == reader.ARCHIVE_BYTES, "Exact original public archive size")
    with path.open("rb") as stream:
        stream.seek(reader.DIRECTORY_START)
        directory = stream.read(1_040_704)
    need(digest(directory) == reader.DIRECTORY_SHA, "Exact original ZIP central directory")
    with zipfile.ZipFile(path) as z:
        return z.infolist()


def image_record(raw, entry, row, output, decoder):
    name = row["file_name"]
    with Image.open(BytesIO(raw)) as image:
        need(image.format == "JPEG" and image.size == (512,512), "Original native JPEG dimensions")
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb_sha = digest(rgb.tobytes())
    decoded, scale = decoder(raw,draft_size=None)
    need(decoded.mode == "RGB" and decoded.size == (512,512) and scale == (1.0,1.0)
         and digest(decoded.tobytes()) == rgb_sha, "Native decode parity with RF-DETR")
    (output/"images"/name).write_bytes(raw)
    product, y, x = map(int,Path(name).stem.split("_"))
    return dict(image_id=int(row["author_train_image_id"]),source_image_name=name,original_hw=[512,512],
                image_ref=dict(path="images/"+name,sha256=digest(raw),bytes=len(raw)),rgb_sha256=rgb_sha,
                acquisition_group=int(row["paper_scene"]),sensor=row["sensor"],product_id=product,row=y,column=x,
                zip_member=entry.filename,zip_crc32=f"{entry.CRC:08x}")


def convert(a, decoder):
    started = perf_counter()
    reader = metadata_reader()
    wanted = roster(a.roster)
    entries = local_members(a.archive,reader) if a.archive else reader.read_directory(a.public_range_cache)
    ordered = sorted(entries,key=lambda e:e.header_offset)
    spans = []
    for j,entry in enumerate(ordered):
        part = PurePosixPath(entry.filename)
        if part.name in wanted and part.parent.name == "JPEGImages":
            end = ordered[j+1].header_offset if j+1 < len(ordered) else reader.DIRECTORY_START
            spans.append((entry.header_offset,end,entry))
    need(len(spans) == len(wanted) and {PurePosixPath(e.filename).name for _,_,e in spans} == set(wanted),
         "Exact author JPEG join")
    batches = reader.coalesce_ranges(spans,gap=0,max_size=2**20)
    fetch = reader.RangeReader(argparse.Namespace(earlier=a.public_range_cache,archive=a.archive,proxy=None))
    a.output.mkdir(parents=True)
    (a.output/"images").mkdir()
    def collect(batch):
        lo, hi, pieces = batch
        data = fetch.read(lo,hi)
        results = {}
        for start,end,entry in pieces:
            name=PurePosixPath(entry.filename).name
            raw=reader.extract_member(entry,data[start-lo:end-lo])
            results[name]=image_record(raw,entry,wanted[name],a.output,decoder)
        return results
    print(json.dumps({"status":"CONVERSION_STARTED","ranges":len(batches),"requested_images":len(wanted)}),flush=True)
    found = {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for group in pool.map(collect,batches):
            need(set(group).isdisjoint(found),"One fresh extracted copy per filename")
            found.update(group)
    images = [found[name] for name in wanted]
    need(len({x["rgb_sha256"] for x in images}) == len(images), "No decoded duplicate within the reserve")
    manifest = dict(expected_ids=[x["image_id"] for x in images],items=images)
    manifest_sha = reader.write_json(a.output/"IMAGE_MANIFEST_PRIVATE.json",manifest)
    receipt = dict(status=STATUS,roster_sha256=ROSTER_SHA,image_manifest_sha256=manifest_sha,images=len(images),
                   parents=len({x["acquisition_group"] for x in images}),exact_crc_and_rgb=len(images),
                   sensor_counts=dict(Counter(x["sensor"] for x in images)),
                   private_annotations_opened=0,detector_forwards=0,rgb_unique=len(images),
                   zip_central_directory_sha256=reader.DIRECTORY_SHA,
                   archive_layout="full local ZIP" if a.archive else "official HTTPS ranges",
                   transfer_ranges=len(batches),transfer_bytes=sum(hi-lo for lo,hi,_ in batches),
                   seconds=perf_counter()-started)
    stamp=reader.write_json(a.output/"RESULT_SAFE.json",receipt)
    print(json.dumps(dict(status=STATUS,result_sha256=stamp)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--roster",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--workers",type=int,default=8)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive",type=Path)
    source.add_argument("--public-range-cache",type=Path,help="Public archive central directory, tail, and Drive download form")
    a=p.parse_args()
    need(not a.output.exists() and 1 <= a.workers <= 16,"Fresh conversion and bounded readers")
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    convert(a,decode_pil_image_bytes)


if __name__ == "__main__":
    main()
