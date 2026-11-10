"""Strict, isolated FIT741-only input admission for the SSDD geometry repeat."""
from collections import defaultdict
from hashlib import sha256
import json
from zipfile import ZIP_STORED

ARCHIVE_SHA = "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"
INDEX_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
MANIFEST_SHA = "5bf021523d1ba3507a5ef7ba6a731da8699c37eb32d6b8ea962c52dcfbaafb47"
SALT = "PSEG_SSDD_FIT741_MASK_RTDETR_A_B_20261111_V1"
SOURCES,BATCH_SIZE,REPEATS = 24,4,4


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read_member(bundle,name,digest,expected_bytes):
    record=bundle.getinfo(name)
    require(record.compress_type==ZIP_STORED and record.file_size==expected_bytes,
            "Expected exact original stored member")
    data=bundle.read(record)
    require(len(data)==expected_bytes and sha256(data).hexdigest()==digest,"Expected exact closed member bytes")
    return data


def verify_archive(path):
    require(path.is_file() and path.stat().st_size==33079338,"Expected exact FIT741 archive length")
    with path.open("rb") as stream:
        checksum=sha256()
        for part in iter(lambda:stream.read(2**20),b""):
            checksum.update(part)
    require(checksum.hexdigest()==ARCHIVE_SHA,"Expected exact closed SSDD FIT741 byte archive")


def catalog(bundle):
    manifest=json.loads(read_member(bundle,"manifest.json",MANIFEST_SHA,861009))
    gt=json.loads(read_member(bundle,"annotations/fit.json",INDEX_SHA,498760))
    rows=manifest["rows"]
    images={row["id"]:row for row in gt["images"]}
    names={row["archive_image_entry"] for row in rows}
    require(manifest["schema"]=="SSDD-FIT741-portable-original-byte-input-manifest-v1"
            and len(rows)==len(images)==741 and len(gt["annotations"])==1661 and len(names)==730
            and len(bundle.infolist())==732 and set(bundle.namelist())==names|{"manifest.json","annotations/fit.json"}
            and gt["categories"]==[{"id":1,"name":"ship"}],"Expected exact FIT population")
    annotations=defaultdict(list)
    for annotation in gt["annotations"]:
        require(annotation["image_id"] in images and annotation["category_id"]==1
                and annotation["iscrowd"]==0 and type(annotation["segmentation"]) is list
                and annotation["segmentation"] and
                all(float(x).is_integer() for part in annotation["segmentation"] for x in part),
                "Expected FIT integer polygon rows")
        annotations[annotation["image_id"]].append(annotation)
    for i,row in enumerate(rows):
        image=images[row["image_id"]]
        require(row["dataset_index"]==i and row["original_shape"]==[image["height"],image["width"]]
                and row["original_basename"]==image["file_name"] and row["archive_image_entry"]==
                "images/"+row["original_image_ref"]["sha256"]+".jpg"
                and row["ann_ids"]==[label["id"] for label in annotations[row["image_id"]]],
                "Expected original image shape and COCO annotation order")
    return rows,annotations


def content_representatives(rows):
    result={}
    for row in sorted(rows,key=lambda item:item["image_id"]):
        result.setdefault(row["original_image_ref"]["sha256"],row)
    return result


def choose(rows):
    unique=content_representatives(rows)
    require(len(unique)==730,"Expected exact original JPEG content count")
    sorted_digests=sorted(unique,key=lambda value:sha256((SALT+value).encode("ascii")).hexdigest())
    return [unique[value] for value in sorted_digests[:SOURCES]]


def make_plan(rows):
    selected=choose(rows)
    return dict(schema="PSEG_SSDD_FIT741_MASK_RTDETR_STOCK_GEOMETRY_V1",
                fit_archive_sha256=ARCHIVE_SHA, fit_index_sha256=INDEX_SHA, fit_manifest_sha256=MANIFEST_SHA,
                salt=SALT, sources=[dict(image_id=row["image_id"],content_sha256=row["original_image_ref"]["sha256"])
                                         for row in selected],
                seeds=[712100+100*rep+batch for rep in range(REPEATS) for batch in range(SOURCES//BATCH_SIZE)],
                views=SOURCES*REPEATS,batch_size=BATCH_SIZE,area_upper_exclusive=1024,
                bootstrap_draws=2048,bootstrap_seed=31991,minimum_groups=16,
                vertical_abs_improvement_px=.1,min_iou_gain=-.002,min_nonempty_rate_gain=-.01,
                official_DEV_opened=False,official_TEST_opened=False)


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()

