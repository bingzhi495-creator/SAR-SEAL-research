"""Audit only author archive identity, panoptic masks and image/source overlap."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image

EXPECTED_SHA256 = "23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5"
STATUS = "SAR_SSP_AUTHOR_ARCHIVE_INTAKE_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decoded(raw, mode):
    with Image.open(BytesIO(raw)) as image:
        return np.asarray(image.convert(mode))


def fingerprint(pixels):
    return sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def audit(zip_file):
    names = [entry.filename for entry in zip_file.infolist() if not entry.is_dir()]
    require(len(set(names)) == len(names) and
            all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
                and "\\" not in name for name in names), "Invalid member identity")
    index = json.loads(zip_file.read("datasets/DATASET_INDEX.json"))
    declared = {row["path"]: (row["bytes"], row["sha256"]) for row in index["files"]}
    checked = 0

    def read(path):
        nonlocal checked
        raw = zip_file.read("datasets/" + path)
        require((len(raw), sha256(raw).hexdigest()) == declared[path], "Archive file manifest mismatch")
        checked += 1
        return raw

    splits = {}
    for split, expected in (("train", (929,461)), ("val", (232,139))):
        root = "SAR-SSP/annotations/"
        dataset = json.loads(read(root + "panoptic_" + split + "2017.json"))
        by_image = {row["id"]: row for row in dataset["images"]}
        annotations = {ann["image_id"]: ann for ann in dataset["annotations"]}
        require(len(by_image)==len(dataset["images"])==len(annotations)==len(dataset["annotations"])==expected[0]
                and set(by_image)==set(annotations), "Author roster differs")
        cat = {row["id"]: row for row in dataset["categories"]}
        require(len(cat)==3 and cat[0]["name"]=="ship" and cat[0]["isthing"]==1
                and all(cat[i]["isthing"]==0 for i in (1,2)), "Class mapping differs")
        ships = Counter()
        source = Counter()
        source_positive = set()
        image_hash = Counter()
        image_source = {}
        tile_keys = set()
        patches = []
        for key, row in by_image.items():
            ann = annotations[key]
            group = row["source_image_no"]
            source[group] += 1
            tile = (group, row["tile_x"], row["tile_y"])
            require(tile not in tile_keys and all(row[k]==ann[k] for k in ("source_image_no","tile_x","tile_y")),
                    "Duplicate within-split tile or differing metadata")
            tile_keys.add(tile)
            patches.append((group,row["tile_x"],row["tile_y"],row["width"],row["height"]))
            original = decoded(read("SAR-SSP/" + split + "2017/" + row["file_name"]), "RGB")
            maskrgb = decoded(read(root + "panoptic_" + split + "2017/" + ann["file_name"]), "RGB")
            require(original.shape==maskrgb.shape==(row["height"],row["width"],3), "Native image/mask shape differs")
            code = maskrgb[:,:,0].astype(np.int32) + 256*maskrgb[:,:,1].astype(np.int32) + 65536*maskrgb[:,:,2].astype(np.int32)
            ids, counts = np.unique(code, return_counts=True)
            areas = dict(zip(ids.tolist(), counts.tolist(), strict=True))
            segments = {s["id"]:s for s in ann["segments_info"]}
            require(len(segments)==len(ann["segments_info"]) and set(areas)-{0}==set(segments),
                    "Panoptic segment PNG id differs from author segments_info")
            has_ship=False
            for sid, seg in segments.items():
                require(seg["category_id"] in cat and areas[sid]==seg["area"], "Segment category or area mismatch")
                if seg["category_id"] != 0:
                    continue
                require(seg["iscrowd"]==0, "Unexpected crowd vessel")
                y,x = np.nonzero(code==sid)
                box = [int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)]
                require(box==seg["bbox"], "Ship bbox differs from panoptic PNG")
                ships["ship_instances"]+=1
                has_ship=True
                ships["truth_area_512_to_1023"]+=int(512 <= len(x) < 1024)
                ships["touch_tile_edge"]+=int(x.min()==0 or y.min()==0 or x.max()==row["width"]-1 or y.max()==row["height"]-1)
            if has_ship:
                ships["ship_positive_chips"]+=1
                source_positive.add(group)
            stamp = fingerprint(original)
            image_hash[stamp]+=1
            image_source[stamp]=group
        require(ships["ship_instances"]==expected[1], "Author ship count differs")
        splits[split] = dict(counts=dict(ships, images=len(by_image), source_groups=len(source),
                                         ship_positive_source_groups=len(source_positive),
                                         duplicate_decoded_images_within=sum(n-1 for n in image_hash.values())),
                             tile_keys=tile_keys, source=set(source), hashes=image_hash,
                             patches=patches, ship_groups=source_positive, image_source=image_source)
    tr,vl = splits["train"],splits["val"]
    shared = tr["source"] & vl["source"]
    cross_tiles = tr["tile_keys"] & vl["tile_keys"]
    inter = [
        (a,b) for a in tr["patches"] for b in vl["patches"] if a[0]==b[0]
        and max(a[1],b[1]) < min(a[1]+a[3],b[1]+b[3])
        and max(a[2],b[2]) < min(a[2]+a[4],b[2]+b[4])
    ]
    result = dict(train=tr["counts"], val=vl["counts"], shared_source_groups=len(shared),
        val_independent_of_archive_train_groups=len(vl["source"]-shared),
        val_independent_of_archive_train_ship_groups=len(vl["ship_groups"]-shared),
        train_val_duplicate_decoded_images=len(set(tr["hashes"]) & set(vl["hashes"])),
        train_val_exact_same_source_tile=len(cross_tiles), train_val_intersplit_overlapping_tiles=len(inter),
        manifest_verified_SAR_SSP_members=checked, all_ship_ids_area_and_bounds_verified=True)
    result["other_archive_datasets_equal_image_pixels"] = {}
    hashes = set(tr["hashes"]) | set(vl["hashes"])
    for corpus in ("HRSID-PAN","SPSD"):
        matches = Counter()
        compared = 0
        for name in names:
            path = PurePosixPath(name)
            if len(path.parts)!=4 or path.parts[1]!=corpus or path.parts[2] not in ("train2017","val2017"):
                continue
            with Image.open(BytesIO(zip_file.read(name))) as image:
                compared+=1
                if image.size != (500,500):
                    continue
                stamp = fingerprint(np.asarray(image.convert("RGB")))
                if stamp in hashes:
                    matches["matched_images"]+=1
        result["other_archive_datasets_equal_image_pixels"][corpus] = dict(images_considered=compared,
                                                                       exact_matches=matches["matched_images"])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    with args.archive.open("rb") as stream:
        digest=sha256()
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(chunk)
    require(digest.hexdigest()==EXPECTED_SHA256,"Expected downloaded author archive")
    with ZipFile(args.archive) as package:
        result=audit(package)
    result.update(status=STATUS,phase="score",archive_sha256=digest.hexdigest(),
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), detector_forwards=0)
    raw=(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n").encode()
    args.output.mkdir()
    (args.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
