"""Rebuild the released image-only Instance-RSDD test inputs from author sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import struct
from time import perf_counter
from types import ModuleType
import zipfile
import zlib

from PIL import Image, ImageOps


BUILDER = "a783fdeb9851711e1bbcbdc776ceb0d0d01269846db11f4cb15c23f0c2923601"
STATUS = "RSDD740_CLEAN_IMAGE_INPUT_COMPLETE"
AUTHOR_ROSTER = "72c4b9af8f8b0bd0df31a3a3f7df6a082390688f0942eed23f106316a2d5d98b"
MANIFEST = "16a546e96d6e76d3afbfbe4f380b182505494818b35f1c8bcc960e6d4d1b2184"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def builder(code):
    path = code/"INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/build_reserve_roster.py"
    raw = path.read_bytes()
    need(digest(raw) == BUILDER, "Exact published author metadata extractor")
    m = ModuleType("published_rsdd_roster")
    m.__file__ = str(path)
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def save(path, record):
    raw = (json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as file:
        file.write(raw)
    return digest(raw)


def jpeg_directory(a, m):
    if a.archive:
        need(a.archive.stat().st_size == m.ZIP_BYTES, "Original archive size")
        with a.archive.open("rb") as file:
            file.seek(m.DIRECTORY_START)
            raw = file.read(m.DIRECTORY_BYTES)
    else:
        raw = a.zip_directory.read_bytes()
    need(len(raw) == m.DIRECTORY_BYTES and digest(raw) == m.DIRECTORY_SHA, "Exact original archive directory")
    ending = struct.pack("<4s4H2IH",b"PK\x05\x06",0,0,14015,14015,len(raw),m.DIRECTORY_START,0)
    with zipfile.ZipFile(BytesIO(raw+ending)) as z:
        files = [(PurePosixPath(e.filename),e) for e in z.infolist()]
    selected = [(p.name,e) for p,e in files if p.parent.name == "JPEGImages" and p.suffix == ".jpg"]
    need(len(selected) == len({p for p,_ in selected}) == 7000, "Unique public JPEG archive entries")
    return dict(selected)


def run(a):
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    m = builder(a.code)
    source = m.image_rows(a.author_test,"test")
    rows = [{k:r[k] for k in ("id","file_name","height","width")} for r in source]
    products = m.paper_products(a.paper_pdf,a.pdftotext)
    directory = jpeg_directory(a,m)
    need({r["file_name"] for r in rows} <= directory.keys(), "Every author filename must be an original ZIP member")
    a.output.mkdir(parents=True)
    image_dir = a.output/"images"
    image_dir.mkdir()
    author_sha = save(a.output/"test_image_roster.json",{"images":rows})
    need(author_sha == AUTHOR_ROSTER, "Exact original author image-only roster")
    started = perf_counter()

    def collect(row):
        name = row["file_name"]
        original = directory[name]
        if a.archive:
            with zipfile.ZipFile(a.archive) as z:
                raw = z.read(original.filename)
        else:
            raw = (a.image_dir/name).read_bytes()
        need(len(raw) == original.file_size and zlib.crc32(raw) == original.CRC, "Original public JPEG size and CRC")
        with Image.open(BytesIO(raw)) as opened:
            need(opened.format == "JPEG" and opened.size == (512,512), "Original author JPEG canvas")
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            rgb_sha = digest(rgb.tobytes())
        decoded, scales = decode_pil_image_bytes(raw,draft_size=None)
        need(decoded.mode == "RGB" and decoded.size == (512,512) and scales == (1.,1.)
             and digest(decoded.tobytes()) == rgb_sha, "Independent decoding equals actual RF-DETR input")
        with (image_dir/name).open("xb") as out:
            out.write(raw)
        product,y,x = map(int,Path(name).stem.split("_"))
        parent,sensor = products[product]
        return dict(image_id=row["id"],source_image_name=name,original_hw=[512,512],
                    image_ref=dict(path="images/"+name,sha256=digest(raw),bytes=len(raw)),
                    rgb_sha256=rgb_sha,acquisition_group=parent,sensor=sensor,product_id=product,row=y,column=x,
                    zip_member=original.filename,zip_crc32=f"{original.CRC:08x}")

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        items = list(pool.map(collect,rows))
    per_sensor = dict(Counter(x["sensor"] for x in items))
    parents = len({x["acquisition_group"] for x in items})
    need(len(items) == len({x["rgb_sha256"] for x in items}) == 740
         and per_sensor == {"GF-3":583,"TerraSAR-X":157} and parents == 112, "Exact complete author test census")
    manifest_sha = save(a.output/"IMAGE_MANIFEST_PRIVATE.json",dict(expected_ids=[x["image_id"] for x in items],items=items))
    need(manifest_sha == MANIFEST, "Fresh decoded image manifest must match the earlier canonical digest")
    summary = dict(status=STATUS,test_roster_sha256=author_sha,image_manifest_sha256=manifest_sha,
                   images=740,sensor_counts=per_sensor,acquisition_groups=parents,rgb_unique=740,
                   encoded_unique=len({x["image_ref"]["sha256"] for x in items}),exact_crc_and_rgb=740,source="author test",target_polygon_rows_opened=0,
                   source_archive_directory_sha256=m.DIRECTORY_SHA,
                   image_source="original ZIP" if a.archive else "original JPEGs with official ZIP CRC verification",
                   elapsed_seconds=perf_counter()-started)
    stamp = save(a.output/"RESULT_SAFE.json",summary)
    print(json.dumps(dict(status=STATUS,result_sha256=stamp)),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--code",type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument("--author-test",type=Path,required=True)
    p.add_argument("--paper-pdf",type=Path,required=True)
    p.add_argument("--pdftotext",default="pdftotext")
    p.add_argument("--zip-directory",type=Path,help="Public original ZIP directory bytes for an already extracted JPEG input")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=8)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive",type=Path)
    source.add_argument("--image-dir",type=Path)
    a = p.parse_args()
    need(not a.output.exists() and 1 <= a.workers <= 16
         and (a.archive is not None or a.zip_directory is not None), "Fresh output and official CRC authority")
    run(a)


if __name__ == "__main__":
    main()
