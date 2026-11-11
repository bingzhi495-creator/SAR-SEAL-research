"""Bounded TRAIN-only intake for a public OpenSAR paired-geometry pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
from time import monotonic
import xml.etree.ElementTree as ET

import requests
import zstandard

MIB = 1024 * 1024
BASE = ("https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/"
        "resolve/e90a0c0465cfb8484b4f356ef968d976054f453a/train/")
SPECS = (
    ("focused", "train_patches.tar.zst", 3709705662, 640*MIB),
    ("rc", "train_range_compressed_rescaled.tar.zst", 10395354466, 512*MIB),
)
FOCUSED = re.compile(r"(?:^|[_-])(?:VD|DVD)_(\d+)_(GRD|SLC)_VH\.tiff?$", re.I)
FOCUSED_VESSEL = re.compile(r"[-_]VD_(\d+)(?:_[A-Z]+)?\.tiff?$", re.I)
RC = re.compile(r"s1[ab]-iw-raw-s-vh-[a-z0-9-]+-IW[123]-VD_(\d+)_rescaled\.npy$", re.I)
LABEL = re.compile(r"DB_OPENSAR_VD_(\d+)\.xml$")
PIN = 395


def kind_and_id(name, kind):
    if kind == "rc":
        m = RC.fullmatch(name)
        return ("rc", int(m[1])) if m else None
    p = Path(name)
    if p.suffix.lower() not in (".tif", ".tiff", ".png"):
        return None
    tokens = p.stem.split("_")
    product = [t.lower() for t in tokens if t.upper() in ("GRD", "SLC")]
    id_ = next((int(t) for t in reversed(tokens) if t.isascii() and t.isdigit()), None)
    if len(product) == 1 and id_ is not None and "VH" in {t.upper() for t in tokens}:
        return product[0], id_
    m = FOCUSED.search(name)
    if m:
        return m[2].lower(), int(m[1])
    m = FOCUSED_VESSEL.search(name)
    if m and re.search(r"-slc-vh-", name, re.I):
        return "slc", int(m[1])
    return None


def labels_from(directory):
    labels = {}
    docs = 0
    for p in directory.glob("*.xml"):
        m = LABEL.fullmatch(p.name)
        if not m or p.stat().st_size > 128*1024:
            raise ValueError("unexpected TRAIN label archive member")
        docs += 1
        root = ET.parse(p).getroot()
        for ship in root.findall("./ProcessingData/List_of_ships/Ship"):
            if ship.findtext("Is_Vessel") != "True" or ship.findtext("Confidence") != "HIGH":
                continue
            boxes = [ship.find(key) for key in ("BoundingBox", "BoundingBox_RC_Local", "BoundingBox_RC_Global")]
            if any(box is None for box in boxes):
                continue
            values = [[int(box.findtext(side)) for side in ("Top", "Left", "Bottom", "Right")] for box in boxes]
            if all(0 <= top < bottom and 0 <= left < right for top,left,bottom,right in values):
                labels[int(m[1])] = p
                break
    if docs != 2671 or PIN not in labels:
        raise ValueError("exact closed TRAIN label archive and pinned positive required")
    return labels


class Counted(io.RawIOBase):
    def __init__(self, raw, size):
        self.raw = raw
        self.limit = size
        self.count = 0
        self.digest = sha256()
        self.start = monotonic()
        self.stopped = False

    def readable(self):
        return True

    def readinto(self, buf):
        if monotonic()-self.start > 205:
            self.stopped = True
            return 0
        count = min(len(buf), self.limit-self.count)
        if count == 0:
            return 0
        data = self.raw.read(count)
        buf[:len(data)] = data
        self.count += len(data)
        self.digest.update(data)
        return len(data)


def scan(spec, labels, dest, stop_for=None):
    arm, archive, total, cap = spec
    members, retained = {}, {}
    path = dest / arm
    path.mkdir()
    with requests.get(BASE+archive+"?download=true", headers={"Range": f"bytes=0-{cap-1}"},
                      stream=True, timeout=(10,12)) as reply:
        if reply.status_code != 206 or reply.headers.get("Content-Range") != f"bytes 0-{cap-1}/{total}":
            raise RuntimeError(f"{arm}: unexpected HTTP response/range")
        with Counted(reply.raw, cap) as counted:
            with zstandard.ZstdDecompressor().stream_reader(counted, closefd=False) as decoded:
                with tarfile.open(fileobj=decoded, mode="r|", bufsize=65536) as tar:
                    try:
                        for entry in tar:
                            if not entry.isfile():
                                continue
                            name = Path(entry.name).name
                            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not (0 < entry.size <= 16*MIB):
                                raise ValueError(f"{arm}: invalid member")
                            group = kind_and_id(name, arm)
                            if group is None:
                                members["other"] = members.get("other", 0)+1
                                continue
                            tag, id_ = group
                            members[tag] = members.get(tag, 0)+1
                            if id_ not in labels:
                                continue
                            if group in retained:
                                raise ValueError(f"{arm}: repeated image/product type")
                            with tar.extractfile(entry) as stream:
                                data = stream.read()
                            if len(data) != entry.size:
                                raise ValueError(f"{arm}: short member")
                            with (path/name).open("xb") as target:
                                target.write(data)
                            retained[group] = path/name
                            if group == stop_for:
                                break
                    except tarfile.ReadError as error:
                        if "unexpected end" not in str(error) or counted.count != cap and not counted.stopped:
                            raise
            if counted.count != cap and not counted.stopped and stop_for not in retained:
                raise ValueError(f"{arm}: prefix terminated before byte bound")
            return dict(bytes=counted.count, budget_bytes=cap, wall_stop=counted.stopped, early_stop_for_requested_member=stop_for in retained, compressed_sha256=counted.digest.hexdigest(), members=members,
                        eligible_members=len(retained), pinned_kinds=sorted(tag for tag, id_ in retained if id_ == PIN)), retained


def run(args):
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("fresh output required")
    labels = labels_from(args.labels)
    args.output.mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = list(pool.map(lambda spec: scan(spec, labels, args.output), SPECS))
    summaries, paths = zip(*pairs)
    focused, rc = paths
    slc_ids = sorted(id_ for kind,id_ in focused if kind=="slc" and ("rc",id_) in rc)
    grd_ids = sorted(id_ for kind,id_ in focused if kind=="grd" and ("rc",id_) in rc)
    choices = [(kind,PIN) for kind in ("slc","grd") if (kind,PIN) in focused and ("rc",PIN) in rc]
    if not choices:
        for kind, ids in (("slc",slc_ids),("grd",grd_ids)):
            if ids:
                choices.append((kind,ids[0]))
                break
    selected = args.output/"selected"
    selected.mkdir()
    recorded_ids = set()
    for kind,id_ in choices:
        shutil.copy2(focused[(kind,id_)], selected)
        if id_ not in recorded_ids:
            shutil.copy2(rc[("rc",id_)], selected)
            shutil.copy2(labels[id_], selected)
            recorded_ids.add(id_)
    safe = dict(status="TRAIN_POSITIVE_PREFIX_INTAKE_CLOSED", train_only=True, dev_or_test_opened=False,
                pinned_id=PIN, matching_high_confidence_slc_count=len(slc_ids),
                matching_high_confidence_grd_count=len(grd_ids), selected=choices,
                candidate_pair_arrays_read=bool(choices), registration_run=False,
                train_label_count=2671, eligible_train_label_count=len(labels),
                focused=summaries[0], rc=summaries[1])
    report = args.output/"RESULT_SAFE.json"
    with report.open("x",encoding="utf-8") as f:
        json.dump(safe, f, sort_keys=True, indent=2)
    print(json.dumps(safe,sort_keys=True),flush=True)


def only_slc(args):
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("fresh output required")
    labels = labels_from(args.labels)
    args.output.mkdir()
    spec = ("focused", "train_patches.tar.zst", 3709705662, 3709705661)
    info, paths = scan(spec, {PIN: labels[PIN]}, args.output, stop_for=("slc", PIN))
    if ("slc", PIN) in paths:
        selected = args.output/"selected"
        selected.mkdir()
        shutil.copy2(paths[("slc", PIN)], selected)
    safe = dict(status="TRAIN_EXACT_SLC_INTAKE_CLOSED", train_only=True, dev_or_test_opened=False,
                pinned_id=PIN, pinned_slc_retrieved=("slc", PIN) in paths, focused=info)
    with (args.output/"RESULT_SAFE.json").open("x",encoding="utf-8") as f:
        json.dump(safe, f, sort_keys=True, indent=2)
    print(json.dumps(safe,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument('--focused-exact-slc', action='store_true')
    args = parser.parse_args()
    if args.focused_exact_slc:
        only_slc(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
