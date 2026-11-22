"""Build the frozen 74-image Instance-RSDD reserve roster from author files without decoding annotations or pixels."""
import argparse
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import zipfile


# Exact image metadata boundaries for the admitted release. Each prefix ends at the closing
# bracket of "images". Only "info", "licenses" and "images" precede that point.
SPLITS = {
    "train": (1933970, 341503, "852d193b38d69e813934b218b246f51498c400b474095f7fc7d72b20e8193a38", 2810),
    "test": (540834, 89510, "6667bfa7c864afe37728bb71e0e7a8beca156365255c3fef8847e410d4d25f1b", 740),
}
PAPER_SHA = "2ea1bf0486b701a385829128a0274f4e54f2fcea991eb1d942eda37d4c71bc6a"
MAPPING_SHA = "359e2d99ea6b872ee5b8d87e3e4742b7f8648a78dc9dadff1d9250bac53bf0c4"
DIRECTORY_SHA = "0d270fdb35120a341dff3940790d13173d603ee05b99ec9dd57efa1f82521d9e"
ROSTER_SHA = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
ZIP_BYTES, DIRECTORY_START, DIRECTORY_BYTES = 1334274952, 1333234226, 1040704


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def image_rows(path, split):
    size, endpoint, expected_sha, count = SPLITS[split]
    require(path.stat().st_size == size, "Author file size differs for " + split)
    with path.open("rb", buffering=0) as stream:
        prefix = stream.read(endpoint)
    require(sha256(prefix).hexdigest() == expected_sha, "Image roster prefix differs for " + split)
    header = json.loads(prefix + b"}")
    require(set(header) == {"info", "licenses", "images"}, "Unexpected image metadata structure")
    rows = header["images"]
    require(len(rows) == len({r["id"] for r in rows}) == len({r["file_name"] for r in rows}) == count,
            "Author image roster count or uniqueness differs for " + split)
    require(all(r["height"] == r["width"] == 512 and re.fullmatch(r"\d+_\d+_\d+\.jpg", r["file_name"])
                for r in rows), "Unexpected image canvas or filename")
    return rows


def paper_products(pdf, pdftotext):
    require(sha256(pdf.read_bytes()).hexdigest() == PAPER_SHA, "A different RSDD-SAR paper was supplied")
    result = subprocess.run([pdftotext, "-layout", "-enc", "UTF-8", str(pdf.resolve()), "-"],
                            stdout=subprocess.PIPE, check=True)
    products = {}
    appendix = result.stdout.decode("utf-8").split("附表 1 RSDD-SAR数据集详细信息")[-1]
    for line in appendix.splitlines():
        row = re.fullmatch(r"\s*(\d+)\s+(GF-3|TerraSAR-X)\s+.*?\s+(\d+)(?:-(\d+))?\s*", line)
        if row:
            scene, sensor, first, last = row.groups()
            for product in range(int(first), int(last or first) + 1):
                require(product not in products, "Duplicate product number in Appendix Table 1")
                products[product] = int(scene), sensor
    canonical = json.dumps(sorted(products.items()), separators=(",", ":")).encode()
    require(set(products) == set(range(193)) and sha256(canonical).hexdigest() == MAPPING_SHA,
            "Could not reproduce the 193-product mapping in Appendix Table 1; check pdftotext layout extraction")
    return products


def jpeg_names(archive, directory_file):
    if archive is not None:
        require(archive.stat().st_size == ZIP_BYTES, "Unexpected RSDD-SAR archive size")
        with archive.open("rb", buffering=0) as stream:
            stream.seek(DIRECTORY_START)
            directory = stream.read(DIRECTORY_BYTES)
    else:
        require(directory_file.stat().st_size == DIRECTORY_BYTES, "Unexpected ZIP directory size")
        directory = directory_file.read_bytes()
    require(sha256(directory).hexdigest() == DIRECTORY_SHA, "Unexpected RSDD-SAR archive directory")
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 14015, 14015, len(directory), 0, 0)
    with zipfile.ZipFile(BytesIO(directory + end)) as z:
        paths = [PurePosixPath(m.filename) for m in z.infolist()]
    names = [p.name for p in paths if p.parent.name == "JPEGImages" and re.fullmatch(r"\d+_\d+_\d+\.jpg", p.name)]
    require(len(names) == len(set(names)) == 7000, "Expected 7,000 distinct JPEG member names")
    return set(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--author-dir", type=Path, required=True, help="Author annotation_coco directory containing train.json and test.json")
    parser.add_argument("--paper-pdf", type=Path, required=True, help="Original RSDD-SAR publication PDF including Appendix Table 1")
    parser.add_argument("--pdftotext", default="pdftotext", help="pdftotext executable on PATH or an explicit path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-archive", type=Path, help="Original 1,334,274,952-byte RSDD-SAR ZIP; reads its central directory only")
    source.add_argument("--zip-directory", type=Path, help="The same ZIP's verified central-directory bytes (offset 1333234226, length 1040704)")
    parser.add_argument("--output", type=Path, required=True, help="New local CSV file; its parent directory must already exist")
    args = parser.parse_args()
    train = image_rows(args.author_dir / "train.json", "train")
    test = image_rows(args.author_dir / "test.json", "test")
    products = paper_products(args.paper_pdf, args.pdftotext)
    names = jpeg_names(args.image_archive, args.zip_directory)
    train_names = {r["file_name"] for r in train}
    test_names = {r["file_name"] for r in test}
    require(train_names.isdisjoint(test_names) and (train_names | test_names) <= names, "Incomplete author/archive join")
    def parent(name):
        return products[int(name.split("_", 1)[0])]
    test_parents = {parent(name)[0] for name in test_names}
    require(len(test_parents) == 112, "Expected 112 previously evaluated parent scenes")
    selected = []
    for row in train:
        name = row["file_name"]
        scene, sensor = parent(name)
        if scene not in test_parents:
            selected.append((row["id"], name, scene, sensor))
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["author_train_image_id", "file_name", "paper_scene", "sensor"])
    writer.writerows(selected)
    raw = buffer.getvalue().encode("utf-8")
    require(len(selected) == 74 and sha256(raw).hexdigest() == ROSTER_SHA, "Frozen roster identity differs")
    with args.output.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(images=74, parent_scenes=13, sha256=ROSTER_SHA, bytes=len(raw))))


if __name__ == "__main__":
    main()
