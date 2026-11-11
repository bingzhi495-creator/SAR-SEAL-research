"""Extract the exact closed native RF-DETR HRSID seed-5202 E24 image order."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import sys


EPOCHS = 24
FIT_IMAGES = 2937
PADDING = 3
BATCH = 4
UPDATES = 17640
ARM_SHA = "d08f10bb1ec8622953604f74ad8a5492d697f2bd80a22ede408f5f2816fb10c2"
VISIT_SHA = "fce4c3cba14e5a55f0f1783639f0799be3fb141f588e40755fe839108612f33f"
CHECKPOINT_SHA = "1ef686c8449621ff3179150f12ba21c81340a8da4e553b23105746c403709909"
FIT_INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
FIT_MANIFEST_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
FIT_READBACK_SHA = "0ac416d96575b9fe5b9784d2252e0bc3430277e5e88d1437d13830c2039bf7f7"
TRAIN_PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decode(raw):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), "Duplicate JSON member")
        return result
    return json.loads(raw, object_pairs_hook=unique)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()


def pinned(path, expected_sha, max_bytes):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and 0 < path.stat().st_size <= max_bytes, "Canonical bounded closed input required: "+path.name)
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected_sha, "Exact closed input digest differs: "+path.name)
    return raw


def validate_arm(arm):
    phase = arm["phase"]
    require(arm["status"] == "HRSID_MASK_PHASE_E24_ARM_ACCEPTED_V1"
            and phase["protocol"] == TRAIN_PROTOCOL and phase["seed"] == 5202
            and phase["mask_resize_mode"] == "native" and phase["stage"] == "e24"
            and arm["original_os_wait"] is True and arm["original_os_exit"] == 0
            and arm["resource_stop"] is False and arm["completed_epochs"] == EPOCHS
            and arm["optimizer_updates"] == arm["ema_updates"] == UPDATES
            and arm["checkpoint_sha256"] == CHECKPOINT_SHA and arm["visit_ledger_sha256"] == VISIT_SHA,
            "Exact closed accepted historical native E24 repeat required")


def validated_names(manifest, independent):
    require(independent["status"] == "INDEPENDENT_ALL_FIT_PNGS_AND_ANNOTATIONS_VERIFIED"
            and independent["no_DEV_TEST_directories"] is True
            and independent["private_manifest_sha256"] == FIT_MANIFEST_SHA
            and independent["native_index_sha256"] == manifest["native_index_sha256"] == FIT_INDEX_SHA
            and independent["images"] == independent["files_sha256_verified"] == len(manifest["images"]) == FIT_IMAGES
            and independent["annotations"] == 9260, "Exact independently verified original FIT identity map required")
    names, folded = {}, set()
    for record in manifest["images"]:
        image_id, basename = record["image_id"], record["native_basename"]
        require(type(image_id) is int and image_id >= 0 and image_id not in names
                and type(basename) is str and PurePosixPath(basename).name == basename
                and "\\" not in basename and PurePosixPath(basename).suffix == ".png"
                and basename.casefold() not in folded and record["member"] == "images/"+basename,
                "Exact unique FIT ID and native PNG basename required")
        names[image_id] = basename
        folded.add(basename.casefold())
    return names


def validate_epoch(row, epoch, ids, padding=PADDING):
    visits = row.get("visits")
    require(set(row) == {"epoch","visits"} and type(row["epoch"]) is int and row["epoch"] == epoch,
            "Exact chronological native epoch required")
    require(type(visits) is list and len(visits) == len(ids)+padding and len(visits)%BATCH == 0
            and all(type(v) is list and len(v) == 2 and type(v[0]) is int and type(v[1]) is int
                    and 0 <= v[1] <= 164 for v in visits), "Exact original visit structure required")
    order = [visit[0] for visit in visits]
    seen = Counter(order)
    require(set(seen) == ids, "Exact FIT identity population required every epoch")
    repeats = {image_id:n-1 for image_id,n in seen.items() if n > 1}
    require(sum(repeats.values()) == padding, "Exact original native padding visits")
    return [order[i:i+BATCH] for i in range(0,len(order),BATCH)], repeats


def write_new(path, raw):
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def extract(args):
    require(sys.flags.isolated and args.output.is_absolute()
            and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists(),
            "Isolated CPU extraction and fresh canonical output required")
    validate_arm(decode(pinned(args.acceptance, ARM_SHA, 16*1024)))
    names = validated_names(decode(pinned(args.layout_manifest, FIT_MANIFEST_SHA, 4*1024**2)),
                            decode(pinned(args.layout_readback, FIT_READBACK_SHA, 16*1024)))
    original = pinned(args.visits, VISIT_SHA, 2*1024**2)
    lines = original.splitlines(keepends=True)
    require(len(lines) == EPOCHS and all(line.endswith(b"\n") for line in lines),
            "Exact closed visit ledger must have 24 complete lines")
    roster = set(names)
    replay, repeats = [], []
    for epoch, line in enumerate(lines, 1):
        batches, padding = validate_epoch(decode(line), epoch, roster)
        replay.append(dict(epoch=epoch,batches=batches))
        repeats.append(padding)
    require(sum(len(row["batches"]) for row in replay) == UPDATES,
            "Four-image batch grouping must preserve all original optimizer update boundaries")
    args.output.mkdir(exist_ok=False)
    replay_sha = write_new(args.output/"FIT_ORDER_PRIVATE.jsonl", b"".join(map(encode,replay)))
    map_sha = write_new(args.output/"FIT_ID_BASENAME_PRIVATE.json", encode(names))
    result = dict(
        status="HRSID_DFINE_E24_EXACT_RF5202_FIT_REPLAY_EXTRACTED",
        FIT=True, DEV=False, TEST=False, labels_opened=False, images_opened=False, weights_opened=False,
        rf_training_seed=5202, rf_initialization_seed=5201, rf_arm="native", rf_train_protocol=TRAIN_PROTOCOL,
        rf_arm_accepted_sha256=ARM_SHA, rf_checkpoint_sha256=CHECKPOINT_SHA, rf_visit_sha256=VISIT_SHA,
        fit_manifest_sha256=FIT_MANIFEST_SHA, fit_manifest_independent_sha256=FIT_READBACK_SHA,
        fit_index_sha256=FIT_INDEX_SHA, fit_id_basename_sha256=map_sha, fit_batch_order_sha256=replay_sha,
        epochs=EPOCHS, unique_fit_images_per_epoch=FIT_IMAGES, visits_per_epoch=FIT_IMAGES+PADDING,
        padding_visits_per_epoch=PADDING, padding_visits_total=PADDING*EPOCHS, original_fit_visits=UPDATES*BATCH,
        batch_size=BATCH, batches_per_epoch=(FIT_IMAGES+PADDING)//BATCH, optimizer_update_boundaries=UPDATES,
        repeat_multiset_same_all_epochs=all(row == repeats[0] for row in repeats),
        repeat_ids_per_epoch_min=min(map(len,repeats)), repeat_ids_per_epoch_max=max(map(len,repeats)),
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    result_sha = write_new(args.output/"RESULT_SAFE.json", encode(result))
    print(json.dumps(dict(status=result["status"],result_sha256=result_sha),sort_keys=True),flush=True)


def main():
    parser = ArgumentParser(description=__doc__,allow_abbrev=False)
    for arg in ("acceptance","visits","layout-manifest","layout-readback","output"):
        parser.add_argument("--"+arg,type=Path,required=True)
    def offline(event, _args):
        if event.startswith("socket."):
            raise PermissionError("No network in offline FIT ledger extraction")
    sys.addaudithook(offline)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()

