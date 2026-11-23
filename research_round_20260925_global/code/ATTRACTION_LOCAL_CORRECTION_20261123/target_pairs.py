"""Extract label-free, matched-budget static/learned disagreement pairs on D-FINE."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec


DEV_REPLAY = "dc38fa8dcd9476599b397657961e8bbdbee1712415ff79efc1ef7dab8e5ac43f"
PSEG_REPLAY = "633da1df76185bb32177422eba0c3b51f2c0b1a01931ce432b993641c10aa6c7"
SIGNED_REPLAY = "cca7746728efc1ac5388364f9c4c47784b7665c418e99c8c62e02f68fcb15603"
ATTRACTION = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
WEIGHTS = "eca85da01c53545485928ae3c502844839083d4ebff896e3df1f91a072d9f725"
PROTOCOL = "7e5c54f9e0ea3489d158daaa114a643d1100175ce4f818696ead95f816d7b933"
GATE = "5fbc36afeaecadd2e2313dae023eb1077ba649e437251790b0e43f7a033ce128"
REFIT = "HRSID_DFINE_E24_FIT109_BCE_FINAL_REFIT_DRAFT_20261112_01"
ARMS = ("attraction_static_at_dynamic_count", "learned_at_attraction_count", "attraction_dynamic")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_module(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Source digest mismatch: " + path.name)
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), m.__dict__)
    return m


def blocks(path, expected):
    digest = sha256()
    with path.open("rb") as stream:
        for line in stream:
            require(line.endswith(b"\n"), "Incomplete historical edit row")
            digest.update(line)
            yield json.loads(line)
    require(digest.hexdigest() == expected, "Historical edit stream digest mismatch")


def dev_inputs(args, options):
    legacy = load_module(args.code/"HRSID_DEV705_CLASSICAL_ATTRACTION_20261122/replay.py", DEV_REPLAY)
    sidecar = args.root/"HRSID_DFINE_E24_DEV705_PROBABILITY_CAPTURE_20261112_01"
    options.protocol = args.code.parent/"HRSID_DEV705_CLASSICAL_ATTRACTION_PRELOCK_20261122.md"
    options.dev_source = args.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705/dfine_dev705.py"
    options.dev_root = args.root/"HRSID_OFFICIAL_PNG_DEV705_AUDIT_ACTUAL_PRIVATE_20261109_01"
    options.sidecar = sidecar/"CONTROL/EXPORT"
    options.sidecar_accepted = sidecar/"INDEPENDENT_RESULT_SAFE.json"
    options.refit_root = args.root/REFIT
    options.prior_root = args.root/"HRSID_DFINE_E24_DEV705_FREE_ENDPOINT_DRAFT_20261112_01"
    old, author, accepted, roster, _ = legacy.load_inputs(options)
    models = old.restore(options)

    def images():
        with args.archive.open("rb") as handle:
            require(args.archive.stat().st_size == author.ARCHIVE_BYTES, "Author DEV archive length differs")
            handle.seek(author.CENTRAL_OFFSET)
            require(sha256(handle.read(author.CENTRAL_BYTES)).hexdigest() == author.CENTRAL_SHA,
                    "Author DEV archive central directory differs")
            with ZipFile(handle) as archive:
                for item, pred, audit, planes in old.source_blocks(options,author,roster,accepted,True):
                    rgb, shape = author.read_png(archive,item)
                    require(shape == (800,800), "Author DEV raster size differs")
                    yield item["image_id"], pred, audit, planes, models[0].image_features(rgb), models[0], models[1], None
    return old, models, accepted, images()


def pseg_inputs(args, options):
    classical = load_module(args.code/"PSEG_TEST232_CLASSICAL_ATTRACTION_20261122/compare.py", PSEG_REPLAY)
    options.owner = args.owner
    options.refit = args.root/REFIT
    options.capture = args.root/"DFINE_HRSID_E24_TO_PSEG_SSDD_TEST232_CAPTURE_ACTUAL_20261121_02"
    options.original_replay = args.root/"PSEG_TEST232_DFINE_FROZEN_REPLAY_ACTUAL_20261121_01"
    options.original_readback = args.root/"PSEG_TEST232_DFINE_FROZEN_REPLAY_READBACK_ACTUAL_20261121_01/RESULT_SAFE.json"
    options.protocol = args.code.parent/"PSEG_TEST232_CLASSICAL_ATTRACTION_PRELOCK_20261122.md"
    classical.pinned(options.protocol, classical.PROTOCOL_SHA)
    _, m, roster, accepted, _ = classical.inputs(options)
    models = m.old.restore(options)
    models[0].raw = m.legacy.pin(options.pair_math,m.native.PAIR)

    def images():
        for item, pred, audit, planes in m.legacy.blocks(options,m.author,roster,accepted,True):
            rgb = m.legacy.rgb(options,item)
            pair,rank = m.native.make_native(models[0], models[1], *item["original_hw"])
            yield item["image_id"], pred, audit, planes, models[0].image_features(rgb), pair, rank, sha256(rgb.tobytes()).hexdigest()
    return m.old, models, accepted, images()


def signed_pairs(gate, coordinates, features, q, attraction, static, learned):
    """Verify the emitted native pixel pairs exhaust exactly the two signed disagreements."""
    rows = gate.pair_rows(coordinates, features, q, attraction, static, learned)
    if isinstance(rows, dict):
        require(set(rows) == {"add","drop"}, "Unexpected pair_rows result keys")
        rows = (rows["add"],rows["drop"])
    require(len(rows) == 2, "Both native signs required")
    result = {}
    for col, (side, pairs) in enumerate(zip(("add","drop"), rows, strict=True)):
        s_coords = {int(coordinates[r[col],col]) for r in static}
        l_coords = {int(coordinates[r[col],col]) for r in learned}
        pairs = [[int(s),int(l),[float(value) for value in x]] for s,l,x in pairs]
        require({r[0] for r in pairs} == s_coords-l_coords and {r[1] for r in pairs} == l_coords-s_coords
                and len(pairs) == len(s_coords-l_coords), "Disagreement pairs omit or duplicate original endpoints")
        require(all(r[2] and np.isfinite(r[2]).all() for r in pairs), "Invalid pair feature vector")
        result[side] = pairs
    return result


def extract_image(inputs, prior, counts, models, att, old, gate):
    image, predictions, audit, planes, rgb_features, pair, rank_math, rgb_sha = inputs
    require(image == prior["image_id"] and (rgb_sha is None or rgb_sha == prior["rgb_sha256"]),
            "Target image identity differs from the archived replay")
    require(len(predictions) == 100, "All archived top-100 slots required")
    _,_,free,head_ops,head,center,scale = models
    counts["images"] += 1
    counts["candidates"] += len(predictions)
    positive = {key for key, row in prior["changes"].items() if any(name in row for name in ARMS)}
    found, output = set(), {}
    for j, (pred, candidate, probability) in enumerate(zip(predictions,audit,planes,strict=True), 1):
        require(candidate["native_rank"] == j and pred["image_id"] == image and candidate["score"] == pred["score"],
                "Native rank and score linkage differs")
        area = candidate["mask_area"]
        if not 32 <= area < 1024:
            continue
        counts["area_eligible"] += 1
        stock = codec.decode(pred["segmentation"]).astype(bool, order="C")
        require(int(stock.sum()) == area, "Native area differs")
        data = pair.paired_points(stock,candidate["box_xyxy"],area,pred["score"],probability,rgb_features,rank_math)
        if data is None:
            continue
        coordinates, features = data[:2]
        q = head_ops.probabilities(head, features, features, center, scale)
        counts["head_forwards"] += 1
        require(q.shape == coordinates.shape and coordinates.shape == features.shape[:2] and
                coordinates.shape[1] == 2 and np.isfinite(q).all(), "Endpoint head shape or values differ")
        cap = min(8,area//10,len(features))
        dynamic, static = att.select(stock,coordinates,cap)
        k = len(dynamic)
        if not k:
            continue
        learned = free.choose_free(q,cap,k)
        key = str(j)
        found.add(key)
        choices = prior["changes"].get(key,{})
        for name, positions in zip(ARMS,(static,learned,dynamic),strict=True):
            require(len(positions) == k and name in choices and choices[name]["swaps"] == k
                    and old.edit(stock,coordinates,positions) == choices[name]["rle"],
                    "Frozen static, learned or dynamic RLE / budget did not reproduce")
        counts.update(positive_budget_candidates=1,swaps_per_arm=k)
        normalized = att._initial(stock,*np.divmod(coordinates,stock.shape[1])) / np.sum(att._WEIGHTS,dtype=np.float64)
        sides = signed_pairs(gate,coordinates,features,q,normalized,static,learned)
        if sides["add"] or sides["drop"]:
            output[key] = sides
            counts.update(disagreement_candidates=1,addition_pairs=len(sides["add"]),deletion_pairs=len(sides["drop"]))
    require(found == positive, "Archived and recomputed positive-budget rosters differ")
    return dict(image_id=image, changes=output)


def run(args):
    protocol_sha = sha256(args.protocol.read_bytes()).hexdigest()
    require(protocol_sha == args.protocol_sha256 == PROTOCOL and args.gate_model_sha256 == GATE,
            "Published prelock or common gate differs")
    gate = load_module(Path(__file__).with_name("gate_model.py"),args.gate_model_sha256)
    att = load_module(args.code/"FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py",ATTRACTION)
    history = load_module(args.code/"ATTRACTION_SIGNED_SIDES_20261123/dfine_sides.py",SIGNED_REPLAY)
    pins = history.PINS[args.cohort]
    ref, historic_result_sha, historic_terminal_sha = history.closed(
        args.root/pins["history"], "verify-edits", pins["verified"])
    require(ref["GT_opened"] is False and ref["optimizer_updates"] == ref["detector_forwards"] == 0,
            "A closed label-free historical replay is required")
    weights = args.root/REFIT/"ACTUAL/POINT_HEAD_PRIVATE.npz"
    require(sha256(weights.read_bytes()).hexdigest() == WEIGHTS,"Frozen full-FIT109 point head differs")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    options = SimpleNamespace(code=args.code)
    if args.cohort == "dev705":
        old,models,receipt,images = dev_inputs(args, options)
    else:
        old,models,receipt,images = pseg_inputs(args, options)
    require(receipt["manifest_sha256"] == pins["manifest"] and receipt["predictions_sha256"] == pins["pred"]
            and receipt["candidate_audit_sha256"] == pins["audit"], "Accepted native carrier differs from classical archives")
    source = args.root/pins["history"]/"prepare-edits/EDITS_PRIVATE.jsonl"
    counts, digest = Counter(), sha256()
    args.output.mkdir()
    with (args.output/"PAIRS_PRIVATE.jsonl").open("xb") as handle:
        for inputs, prior in zip(images,blocks(source,ref["edits_sha256"]),strict=True):
            row = extract_image(inputs,prior,counts,models,att,old,gate)
            data = (json.dumps(row, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode("utf-8")
            handle.write(data)
            digest.update(data)
            if counts["images"] % 50 == 0:
                print(json.dumps(dict(phase="prepare-pairs",images=counts["images"])),flush=True)
    require(counts["images"] == pins["images"] and counts["candidates"] == 100*pins["images"]
            and counts["swaps_per_arm"] == pins["swaps"] and counts["positive_budget_candidates"] == pins["positive"],
            "Frozen control census differs")
    return dict(status="ATTRACTION_LOCAL_CORRECTION_TARGET_PAIRS_COMPLETE", phase="prepare-pairs",
                cohort=args.cohort,images=counts["images"],counts=dict(counts),pairs_sha256=digest.hexdigest(),
                protocol_sha256=protocol_sha,gate_model_sha256=args.gate_model_sha256,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), weights_sha256=WEIGHTS,
                classical_edits_sha256=ref["edits_sha256"], classical_terminal_sha256=historic_terminal_sha,
                classical_verified_result_sha256=historic_result_sha,refit_accepted_sha256=old.REFIT_ACCEPTED,
                classical_replay_source_sha256=DEV_REPLAY if args.cohort == "dev705" else PSEG_REPLAY,
                attraction_source_sha256=ATTRACTION, historical_lineage_source_sha256=SIGNED_REPLAY,
                manifest_sha256=pins["manifest"],predictions_sha256=pins["pred"],candidate_audit_sha256=pins["audit"],
                probability_index_sha256=receipt["probability_index_sha256"], exact_SS_LL_DD=True, exact_budget=True,
                detector_forwards=0,head_forwards=counts["head_forwards"],optimizer_updates=0,
                target_labels_opened=False,GT_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cohort", choices=("dev705","pseg232"), required=True)
    for key in ("code","root","protocol","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--owner",type=Path)
    parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--gate-model-sha256",required=True)
    args = parser.parse_args()
    require((args.archive is not None) == (args.cohort == "dev705")
            and (args.owner is not None) == (args.cohort == "pseg232"),"Supply the cohort-specific archive or image owner")
    require(not args.output.exists(),"Supply a fresh output directory")
    started = perf_counter()
    result = run(args)
    result["seconds"] = perf_counter()-started
    raw = (json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    with (args.output/"RESULT_SAFE.json").open("xb") as handle:
        handle.write(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
