"""Frozen D-FINE inference and edits on a qualified image-only PNG cohort."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import sys
from time import perf_counter
from types import ModuleType


CAPTURE_SOURCE = "11df18cb1eb0dbcff66b576d0c082b45896e5e254adb66b98fdfaff366d7391a"
REPLAY_SOURCE = "f1b6e83e98851bf867bd72f3ecb082ca2997632bdd1aedaaf64110cf06b04fd5"
CLASSICAL_SOURCE = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
CAPTURED = "UEIKAP_DFINE_RESERVED_NATIVE_CAPTURE_COMPLETE_V1"
CAPTURED_OK = "UEIKAP_DFINE_RESERVED_NATIVE_CAPTURE_ACCEPTED_V1"
EDITED = "UEIKAP_DFINE_RESERVED_EDIT_REPLAY_COMPLETE_V1"
EDITED_OK = "UEIKAP_DFINE_RESERVED_EDIT_REPLAY_ACCEPTED_V1"
ARM_EXTRA = ("attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count")
MODES = ("raw", "fixed", "free_matched_count", "free_natural", "native_matched_count", *ARM_EXTRA)
SOURCE_SCOPE = dict(FIT=False, DEV=False, TEST=True, GT_opened=False, AP=False, author_TEST=False)
MIB = 1024**2


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def pinned(path, expected, bound=16*MIB):
    need(sha(expected) and path.is_file() and path.stat().st_size <= bound, "Bounded frozen input " + path.name)
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected, "Exact frozen bytes " + path.name)
    return data


def doc(path, expected):
    return json.loads(pinned(path, expected), object_pairs_hook=unique)


def unique(items):
    out = {}
    for k, v in items:
        need(k not in out, "Duplicate JSON key " + k)
        out[k] = v
    return out


def compiled(path, digest, replacements=()):
    source = pinned(path, digest).decode()
    for before, after in replacements:
        need(source.count(before) == 1, "Exact ordinary-source seam " + before)
        source = source.replace(before, after, 1)
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(source, str(path), "exec"), obj.__dict__)
    return obj


def load_manifest(a):
    m = doc(a.manifest, a.manifest_sha)
    need(set(m) == {"expected_ids", "items"}, "Image-only manifest carries no annotation fields")
    closure = doc(a.intake_terminal, a.intake_terminal_sha)
    need(closure["status"] == "completed_verified" and type(closure["actual_exit"]) is int
         and closure["actual_exit"] == 0 and closure["images_roster_sha256"] == a.manifest_sha,
         "Independently closed original PNG intake")
    rows, items = m["expected_ids"], m["items"]
    need(rows == list(range(1,109)) and all(type(k) is int for k in rows)
         and len(items) == 108, "Exact frozen 108-member order")
    image_dir = (a.images / "images").resolve(strict=True)
    need(image_dir.parent == a.images and image_dir.is_dir(), "Consistent image-only root")
    inventory = sha256()
    for index, row in zip(rows, items, strict=True):
        need(set(row) == {"image_id","original_hw","image_ref"} and type(row["image_id"]) is int
             and row["image_id"] == index and row["original_hw"] == [512,512],
             "Original author 512-square native raster, exact identity")
        ref = row["image_ref"]
        name = f"images/{index:06}.png"
        need(set(ref) == {"path","sha256","bytes"} and ref["path"] == name and sha(ref["sha256"])
             and type(ref["bytes"]) is int and 0 < ref["bytes"] <= 16*MIB,
             "Exact author PNG bytes and enumerated path")
        inventory.update(f"{name}\t{ref['bytes']}\t{ref['sha256']}\n".encode())
    need({p.name for p in image_dir.iterdir() if p.is_file()} ==
         {PurePosixPath(row["image_ref"]["path"]).name for row in items}, "Exact image-only file inventory")
    return items, inventory.hexdigest()


def png(root, item, _base=None):
    import numpy as np
    from PIL import Image

    ref = item["image_ref"]
    path = (root / ref["path"]).resolve(strict=True)
    need(path.parent == (root / "images").resolve(strict=True), "Image remains inside intake root")
    raw = pinned(path, ref["sha256"])
    need(len(raw) == ref["bytes"], "Exact original PNG length")
    with Image.open(BytesIO(raw)) as image:
        need(image.format == "PNG" and image.mode in ("RGB","L","P") and image.getexif().get(274,1) == 1,
             "Explicit native 8-bit PNG and unchanged orientation")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    need(rgb.shape == (*item["original_hw"],3), "Exact original native canvas")
    return rgb


def capture_module(a):
    source = a.code / "PSEG_TEST232_DFINE_FROZEN_CAPTURE_20261121/target_capture.py"
    original = """base.need({p.name for p in args.test_root.iterdir()} == {r["image_ref"]["path"] for r in rows},
              "Exactly the 232 JPEGs in the image-only root")"""
    target = compiled(source, CAPTURE_SOURCE, (
        (original, 'base.need(len(rows)==108, "Exactly the reserved PNGs in the image-only root")'),
        ('stats["probability_files"] == 232', 'stats["probability_files"] == 108'),
        ('images=232, model_forwards=232', 'images=108, model_forwards=108'),
        ('raw_queries=232*300, internal_top_rows=232*300', 'raw_queries=108*300, internal_top_rows=108*300'),
        ('native_dimensions_verified=232', 'native_dimensions_verified=108'),
        ('PIL EXIF transpose then RGB, full native resolution','PIL PNG RGB decode, untransformed native resolution'),
        ('historical_target_exposure=True, FIT=False, DEV=False, TEST=True, GT_opened=False, AP=False, **stats',
         'intake_terminal_sha256=args.intake_terminal_sha, author_TEST=False, FIT=False, DEV=False, TEST=True, GT_opened=False, AP=False, **stats'),
        ('event="SSDD_TEST232_NATIVE_CAPTURE_PROGRESS"', 'event="UEIKAP_RESERVED108_NATIVE_CAPTURE_PROGRESS"'),
    ))
    target.STATUS = CAPTURED
    target.__file__ = __file__
    target.read_rgb = png
    target.manifest_rows = lambda _path, _digest, _base: load_manifest(a)[0]
    return target


def replay_modules(a):
    source = a.code / "PSEG_TEST232_DFINE_FROZEN_REPLAY_20261121/replay.py"
    historical = compiled(source, REPLAY_SOURCE)
    m = historical.modules(a.code, a)
    m.legacy.MODES = MODES
    m.legacy.rgb = lambda owner, row: png(owner.images, row)
    return historical, m


def observe_terminal(path, digest, stage, result_sha):
    t = doc(path, digest)
    need(t["phase"] == stage and t["actual_exit"] == 0 and type(t["actual_exit"]) is int
         and t["result_sha256"] == result_sha and t["worker_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest()
         and t["output_footer_verified"] is True, "Original process wait, matching footer and result")
    return t


def safe_write(path, value):
    data = (json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return sha256(data).hexdigest()


def complete(a, status, fields):
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    record = dict(**SOURCE_SCOPE, status=status, images=108, model_forwards=0, optimizer_updates=0,
                  code_sha256=worker_sha, manifest_sha256=a.manifest_sha, intake_terminal_sha256=a.intake_terminal_sha,
                  protocol_sha256=a.protocol_sha, **fields)
    digest = safe_write(a.output / "RESULT_SAFE.json", record)
    print(json.dumps(dict(status=status,result_sha256=digest)), flush=True)


def verify_capture(a, roster, inventory):
    cap = capture_module(a)
    result = doc(a.capture/"RESULT_SAFE.json", a.capture_sha)
    observe_terminal(a.capture_terminal, a.capture_terminal_sha, "capture", a.capture_sha)
    worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    need(result["status"] == CAPTURED and result["manifest_sha256"] == a.manifest_sha
         and result["image_inventory_sha256"] == inventory and result["export_source_sha256"] == worker_sha
         and result["protocol_sha256"] == a.protocol_sha and result["intake_terminal_sha256"] == a.intake_terminal_sha
         and result["ema_sha256"] == cap.EMA_SHA and result["e24_acceptance_sha256"] == cap.ACCEPT_SHA
         and result["images"] == result["model_forwards"] == result["native_dimensions_verified"] == 108
         and result["probability_files"] == 108 and result["optimizer_updates"] == 0
         and result["exact_author_binary_replay"] == result["post_nms_rows"]
         and result["exact_top100_binary_replay"] == result["probability_rows"] == result["rows_at_100"]
         and all(result[k] is v for k,v in SOURCE_SCOPE.items()), "Frozen original capture closure")
    _,m = replay_modules(a)
    m.author.cpu_runtime()
    names = {f"{i:04d}.npy" for i in range(1,109)}
    need({x.name for x in (a.capture/"PROBABILITY_PRIVATE").iterdir()} == names, "Exact probability inventory")
    rows = empty = 0
    for item,coco,audit,planes in m.legacy.blocks(a,m.author,roster,result,True):
        rows+=1
        empty+=sum(x["mask_area"] == 0 for x in audit)
        need(len(coco) == len(audit) == len(planes), "Verified native probability pair")
    need(rows == 108 and empty == result["empty_at_100"], "Independent native candidate and plane census")
    a.output.mkdir()
    fields = {key:result[key] for key in ("predictions_sha256","candidate_audit_sha256","probability_index_sha256",
             "probability_files","probability_rows","post_nms_rows","empty_at_100","ema_sha256","e24_acceptance_sha256")}
    complete(a,CAPTURED_OK,dict(fields,capture_sha256=a.capture_sha,capture_terminal_sha256=a.capture_terminal_sha,
                             independent_probability_and_candidate_census=True))


def extended_edits(a, m):
    classical = compiled(a.code / "FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py", CLASSICAL_SOURCE)
    payload = Path(m.old.__file__).read_text(encoding="utf-8")
    before = "        actions=dict(fixed=fixed,free_matched_count=matched,free_natural=natural,native_matched_count=native)\n"
    after = before + """        dynamic,static=attraction.select(stock,coordinates,cap)
        classical_matched=free.choose_free(p,cap,len(dynamic))
        actions.update(attraction_dynamic=dynamic,attraction_static_at_dynamic_count=static,
                       learned_at_attraction_count=classical_matched)
"""
    begin, end=payload.index("def image_edits("),payload.index("\n\ndef prelabel(")
    original = payload[begin:end]
    need(original.count(before) == 1, "Exact original replay method seam")
    namespace = dict(m.old.__dict__, attraction=classical)
    exec(compile(original.replace(before,after,1), str(m.old.__file__), "exec"), namespace)
    m.old.image_edits = namespace["image_edits"]


def same_count(changes, modes):
    for arms in changes.values():
        want = {arms.get(key,{}).get("swaps",0) for key in modes}
        need(len(want) == 1, "Exact per-candidate classical count and retained no-ops")


def replay(a, roster, verify):
    import os
    import torch

    old,m = replay_modules(a)
    m.author.cpu_runtime()
    accepted = doc(a.accepted/"RESULT_SAFE.json",a.accepted_sha)
    need(accepted["status"] == CAPTURED_OK and accepted["capture_sha256"] == a.capture_sha
         and accepted["manifest_sha256"] == a.manifest_sha and accepted["protocol_sha256"] == a.protocol_sha
         and accepted["intake_terminal_sha256"] == a.intake_terminal_sha and accepted["images"] == 108
         and accepted["probability_files"] == 108 and all(accepted[k] is v for k,v in SOURCE_SCOPE.items()),
         "Independently accepted native capture")
    if verify:
        prepared = doc(a.replay/"RESULT_SAFE.json",a.replay_sha)
        observe_terminal(a.replay_terminal,a.replay_terminal_sha,"prepare-edits",a.replay_sha)
        need(prepared["status"] == EDITED and prepared["accepted_sha256"] == a.accepted_sha
             and prepared["manifest_sha256"] == a.manifest_sha and prepared["protocol_sha256"] == a.protocol_sha
             and prepared["intake_terminal_sha256"] == a.intake_terminal_sha and prepared["modes"] == list(MODES)
             and all(prepared[k] is v for k,v in SOURCE_SCOPE.items()), "Closed label-free edit stream")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    extended_edits(a,m)
    models = m.old.restore(a)
    models[0].raw = m.legacy.pin(a.pair_math,m.native.PAIR)
    start = perf_counter()
    counts,sums,groups=Counter(),sha256(),set()
    support = Counter()
    a.output.mkdir()
    report = (a.replay if verify else a.output)/m.legacy.EDITS
    with report.open("rb" if verify else "xb") as stream:
        for count,(item,coco,audit,planes) in enumerate(m.legacy.blocks(a,m.author,roster,accepted,True),1):
            rgb = png(a.images,item)
            row = old.emit(m,a,item,coco,audit,planes,rgb,models,counts)
            data = m.author.encoded(row)
            same_count(row["changes"],ARM_EXTRA)
            if verify:
                need(stream.readline(4*MIB) == data, "Byte-for-byte replay of real policy choices")
                old.check_support(models[1],coco,audit,row["changes"],support)
            else:
                need(stream.write(data) == len(data), "Complete frozen policy stream")
            sums.update(data)
            groups.add(row["rgb_sha256"])
            counts["candidates"]+=len(coco)
            if count % 25 == 0 or count == 108:
                print(json.dumps(dict(event="UEIKAP_FROZEN_EDIT_REPLAY_PROGRESS",images=count)),flush=True)
        if verify:
            need(not stream.read(1) and prepared["edits_sha256"] == sums.hexdigest()
                 and prepared["counts"] == dict(counts) and prepared["rgb_groups"] == len(groups)
                 and all(counts[key] == v for key,v in support.items()), "Full independent edit and support count parity")
        else:
            stream.flush()
            os.fsync(stream.fileno())
    need(count == 108 and counts["candidates"] == accepted["probability_rows"]
         and counts["fixed_swaps"] == counts["free_matched_count_swaps"]
         and counts["free_natural_swaps"] == counts["native_matched_count_swaps"]
         and len({counts[x+"_swaps"] for x in ARM_EXTRA}) == 1, "Complete 108-cohort realized counts")
    fields = dict(modes=list(MODES), counts=dict(counts), rgb_groups=len(groups), accepted_sha256=a.accepted_sha,
                  capture_sha256=a.capture_sha, edits_sha256=sums.hexdigest(), elapsed_seconds=perf_counter()-start,
                  classical_source_sha256=CLASSICAL_SOURCE, ema_sha256=accepted["ema_sha256"],
                  predictions_sha256=accepted["predictions_sha256"], candidate_audit_sha256=accepted["candidate_audit_sha256"],
                  frozen_refit_acceptance_sha256=m.old.REFIT_ACCEPTED)
    if verify:
        fields.update(replay_sha256=a.replay_sha,replay_terminal_sha256=a.replay_terminal_sha,
                      exact_replayed_policy_choices=True,exact_support_xor_area=True)
    complete(a,EDITED_OK if verify else EDITED,fields)


def main():
    import os

    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase",choices=("check","capture","verify-capture","prepare-edits","verify-edits"))
    p.add_argument("--code",type=Path,required=True)
    for k in ("manifest","intake-terminal","images","protocol","output","acceptance","author-source","author-tree",
              "ema","codec-wheel","capture","capture-terminal","accepted","refit-result","refit-acceptance",
              "weights","replay","replay-terminal"):
        p.add_argument("--"+k,type=Path)
    for k in ("manifest","intake-terminal","protocol","acceptance","capture","capture-terminal",
              "accepted","replay","replay-terminal"):
        p.add_argument("--"+k+"-sha")
    a = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode,"Exact isolated Python invocation")
    for key,value in vars(a).items():
        if isinstance(value,Path):
            setattr(a,key,value.resolve(strict=key!="output"))
    if a.phase == "check":
        capture_module(a)
        _,m = replay_modules(a)
        extended_edits(a,m)
        print(json.dumps(dict(status="UEIKAP_TRANSFER_ORDINARY_SOURCE_PREFLIGHT_OK",target_inputs_opened=0)))
        return
    need(a.output is not None and not a.output.exists(),"Fresh result directory")
    pinned(a.protocol,a.protocol_sha,64*1024)
    roster,inventory = load_manifest(a)
    if a.phase == "capture":
        def offline(event, _):
            if event.startswith("socket."):
                raise PermissionError("Offline frozen target capture")
        sys.addaudithook(offline)
        target = capture_module(a)
        target.INVENTORY_SHA = inventory
        a.test_root = a.images
        a.reference_dir = a.code/"DFINE_HRSID_TO_SSDD_DEV187_CAPTURE_20261120"
        target.run(a)
    else:
        need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1","CPU-only independent readback and edit replay")
        if a.phase == "verify-capture":
            verify_capture(a,roster,inventory)
        else:
            replay(a,roster,a.phase=="verify-edits")


if __name__ == "__main__":
    main()
