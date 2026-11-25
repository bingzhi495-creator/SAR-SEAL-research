"""Independently read back the closed private COCO label export."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from pycocotools import mask
from pycocotools.coco import COCO


def sha(payload):
    return sha256(payload).hexdigest()


def canonical(doc):
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def check():
    root = Path(__file__).parent
    output = root / "export"
    summary_raw = (output / "RESULT_SAFE.json").read_bytes()
    assert sha(summary_raw) == "bd4eb3add418d876f1defee88352aeee26c4f1b3135700436851fe73ad9b2796"
    summary = json.loads(summary_raw)
    assert sha((root / "export_train_labels.py").read_bytes()) == summary["worker_sha256"]
    assert sha((root / "PRELOCK.md").read_bytes()) == summary["protocol_sha256"]
    member = "Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style/annotations/train.json"
    source = subprocess.check_output(["tar", "-xOf", "D:/SAR/UNet++/archives/datasets/Official-SSDD-OPEN.rar", member])
    assert sha(source) == "4565548ce05136dba8072a4b1c1fb668f3d415e3c980ebfe05b5a8998dec1ac5"
    original = {a["id"]: a for a in json.loads(source)["annotations"]}
    used = {k: set() for k in ("images", "encoded", "decoded", "pseg", "spsd")}
    results = {}
    for part, stats in summary["partitions"].items():
        files = {"pseg": "PSeg_SSDD_COCO_PRIVATE.json", "spsd": "SPSD_COCO_PRIVATE.json",
                 "binding": "IMAGE_BINDINGS_PRIVATE.json"}
        docs = {}
        for tag, filename in files.items():
            raw = (output / part / filename).read_bytes()
            assert sha(raw) == stats["artifact_hashes"][tag]
            docs[tag] = json.loads(raw)
        assert docs["pseg"]["images"] == docs["spsd"]["images"] and docs["pseg"]["categories"] == docs["spsd"]["categories"]
        assert docs["binding"]["label_sha256"] == {k: stats["artifact_hashes"][k] for k in ("pseg", "spsd")}
        imgs = {rec["file_name"]: rec for rec in docs["pseg"]["images"]}
        binds = docs["binding"]["image_bindings"]
        assert set(imgs) == set(binds) and len(imgs) == stats["images"]
        by_id = {}
        for tag in ("pseg", "spsd"):
            coco = COCO()
            coco.dataset = docs[tag]
            coco.createIndex()
            ann = docs[tag]["annotations"]
            assert len(coco.imgs) == stats["images"] and len(coco.anns) == len(ann) == stats[tag + "_hulls"]
            assert not (used[tag] & set(coco.anns))
            by_id[tag] = dict(coco.imgToAnns)
            used[tag].update(coco.anns)
        units = {"images": {im["id"] for im in imgs.values()}, "decoded": {b["decoded_group_sha256"] for b in binds.values()},
                 "encoded": {b["jpeg_sha256"] for b in binds.values()}}
        assert len(units["decoded"]) == stats["decoded_groups"]
        for field, keys in units.items():
            assert not (used[field] & keys)
            used[field].update(keys)
        for nm, im in imgs.items():
            bnd = binds[nm]
            assert (im["id"], im["width"], im["height"]) == (bnd["image_id"], bnd["width"], bnd["height"])
            for tag in ("pseg", "spsd"):
                rows = by_id[tag][im["id"]]
                assert sha(canonical(rows)) == bnd[tag + "_export_rows_sha256"]
                assert all(a["image_id"] == im["id"] and a["category_id"] == 0 and a["iscrowd"] == 0 for a in rows)
                assert sha(canonical(sorted((tuple(a["segmentation"]["size"]), a["segmentation"]["counts"]) for a in rows))) == bnd[tag + "_raster_set_sha256"]
                assert all(a["segmentation"]["size"] == [im["height"], im["width"]] for a in rows)
                if tag == "pseg":
                    assert [a["id"] for a in rows] == bnd["pseg_original_ids"]
                    for a in rows:
                        old = original[a["id"]]
                        assert {k: v for k, v in old.items() if k != "segmentation"} == {k: v for k, v in a.items() if k != "segmentation"}
                        rle = mask.merge(mask.frPyObjects(old["segmentation"], im["height"], im["width"]))
                        assert a["segmentation"]["counts"] == rle["counts"].decode("ascii")
                else:
                    assert [item[1] for item in bnd["spsd_author_segment_to_export_id"]] == [a["id"] for a in rows]
                    for a in rows:
                        assert mask.area(a["segmentation"]) == a["area"]
                        assert mask.toBbox(a["segmentation"]).tolist() == a["bbox"]
        results[part] = stats
    assert len(used["images"]) == 915 and len(used["decoded"]) == 901 and len(used["pseg"]) == len(used["spsd"]) == 1990
    return {"complete_readback": True, "result_sha256": sha(summary_raw), "partitions": results,
            "test_annotations_opened": 0, "private_labels_releasable": False}


if __name__ == "__main__":
    import contextlib
    import sys
    with contextlib.redirect_stdout(sys.stderr):
        result = check()
    print(json.dumps(result, sort_keys=True))
