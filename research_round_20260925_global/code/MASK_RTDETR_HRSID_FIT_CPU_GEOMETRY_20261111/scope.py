"""Prepare the fixed FIT-only scope, before opening any selected image."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

ZIP_SHA = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
COCO_SHA = "6f25f80ed0e0b9c3f0c57ab6cc3b25e947a102353757a33bca4d3823df3d547d"
ARCHIVE_BYTES = 151320868
SALT = "HRSID_FIT_MASK_RTDETR_STOCK_A_B_20261111_V1"
SOURCES, BATCH_SIZE, REPEATS = 24, 4, 4
SCHEDULE_SEED = 712100
BOOTSTRAP_SEED, DRAWS = 31991, 2048
MIN_GROUPS = 16
LIMIT_Y_PX, LIMIT_IOU, LIMIT_RETENTION = .1, -.002, -.01

def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()

def zip_digest(path):
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Wrong closed archive length")
    digest = sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(2**20), b""):
            digest.update(data)
    if digest.hexdigest() != ZIP_SHA:
        raise ValueError("Wrong closed archive SHA256")

def read_pinned(bundle, name, digest, max_bytes):
    info = bundle.getinfo(name)
    if info.compress_type != ZIP_STORED or not 0 < info.file_size <= max_bytes:
        raise ValueError("Noncanonical FIT archive member")
    data = bundle.read(info)
    if len(data) != info.file_size or sha256(data).hexdigest() != digest:
        raise ValueError("Unexpected FIT archive member bytes")
    return data

def read_catalog(bundle):
    cohort = json.loads(read_pinned(bundle, "COHORT_PRIVATE.json", COHORT_SHA, 2**20))
    coco = json.loads(read_pinned(bundle, "FIT114_COCO_PRIVATE.json", COCO_SHA, 4*2**20))
    rows = cohort["images"]
    names = {"COHORT_PRIVATE.json", "FIT114_COCO_PRIVATE.json", "HISTORICAL_VIEWS_PRIVATE.jsonl",
             *("images/" + row["native_basename"] for row in rows)}
    if not (cohort["fit_only"] and len(rows) == len(coco["images"]) == 114
            and len(coco["annotations"]) == 1109
            and len(bundle.infolist()) == len(names) == 117 and set(bundle.namelist()) == names
            and [row["id"] for row in coco["categories"]] == [1]):
        raise ValueError("Expected the closed FIT114 subset and no DEV/TEST")
    ids = {row["id"]: row for row in coco["images"]}
    if not (len(ids) == 114 and all((ids[row["image_id"]]["file_name"],
                                    ids[row["image_id"]]["height"],
                                    ids[row["image_id"]]["width"]) == (row["native_basename"], 800, 800)
                                   for row in rows)):
        raise ValueError("FIT114 image metadata must match the closed PNG roster")
    return rows, coco

def choose(rows):
    ordered = sorted((row for row in rows if row["core"] and not row["stress"]),
                     key=lambda row: sha256((SALT + row["encoded_sha256"]).encode("ascii")).hexdigest())
    selected, groups, hashes = [], set(), set()
    for row in ordered:
        if row["group"] in groups or row["encoded_sha256"] in hashes:
            continue
        selected.append(row)
        groups.add(row["group"])
        hashes.add(row["encoded_sha256"])
        if len(selected) == SOURCES:
            return selected
    raise ValueError("Missing 24 distinct core source groups")

def make_plan(rows):
    selected = choose(rows)
    return dict(schema="MASK_RTDETR_STOCK_HRSID_FIT_A_B_CPU_GEOMETRY_V1",
                archive_sha256=ZIP_SHA, subset_sha256=COCO_SHA, cohort_sha256=COHORT_SHA,
                salt=SALT, sources=[dict(image_id=row["image_id"], group=row["group"],
                                         encoded_sha256=row["encoded_sha256"]) for row in selected],
                seeds=[SCHEDULE_SEED+100*repeat+batch for repeat in range(REPEATS)
                       for batch in range(SOURCES//BATCH_SIZE)],
                views=SOURCES*REPEATS, batch_size=BATCH_SIZE, area_upper_exclusive=1024,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=DRAWS, minimum_groups=MIN_GROUPS,
                vertical_abs_improvement_px=LIMIT_Y_PX, min_iou_gain=LIMIT_IOU,
                min_nonempty_rate_gain=LIMIT_RETENTION, official_DEV_opened=False, official_TEST_opened=False)

