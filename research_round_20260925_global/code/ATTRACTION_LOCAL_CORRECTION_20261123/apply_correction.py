"""Apply a frozen FIT109 correction to closed image-only disagreement features."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


PRELOCK = "7e5c54f9e0ea3489d158daaa114a643d1100175ce4f818696ead95f816d7b933"
GATE = "5fbc36afeaecadd2e2313dae023eb1077ba649e437251790b0e43f7a033ce128"
DFINE = "cca7746728efc1ac5388364f9c4c47784b7665c418e99c8c62e02f68fcb15603"
SPARSE = "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"
PREPARED = "ATTRACTION_LOCAL_CORRECTION_TARGET_PAIRS_COMPLETE"
STATUS = "ATTRACTION_LOCAL_CORRECTION_REPLAY_COMPLETE"


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def source(path, stamp):
    code = path.read_bytes()
    require(sha256(code).hexdigest() == stamp, "Code differs for " + path.name)
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(code, str(path), "exec"), mod.__dict__)
    return mod


def membership(rle, pixels, sparse):
    coordinates = np.asarray(pixels, dtype=np.int64)
    height, width = rle["size"]
    require(np.all((0 <= coordinates) & (coordinates < height*width)), "Out-of-canvas correction coordinates")
    runs = np.cumsum(sparse.decode_runs(rle["counts"]), dtype=np.int64)
    index = (coordinates % width) * height + coordinates // width
    return np.searchsorted(runs, index, side="right") % 2


def corrected(static, learned, native, pair, model, gate, sparse, k, counts):
    activations = {key: [] for key in ("C", "CM", "Q")}
    complete = []
    for side_number, side in enumerate(gate.SIDES):
        entries = pair[side]
        n = len(entries)
        require(n <= k, "Too many disagreement locations")
        if not n:
            continue
        points = np.asarray([e[:2] for e in entries], dtype=np.int64)
        require(all(type(e[0]) is int and type(e[1]) is int for e in entries)
                and len(set(points.flat)) == 2*n and
                all(np.all(np.diff(points[:, i]) > 0) for i in (0, 1)),
                "Signed source pairs must be unique and ordered by native cell index")
        flat = points.reshape(-1)
        require(np.all(membership(native, flat, sparse) == side_number), "Pairs changed native sign")
        expected = np.array([1-side_number, side_number])
        require(np.all(membership(static, points, sparse) == expected)
                and np.all(membership(learned, points, sparse) == expected[::-1]), "SS or LL selected a different cell")
        x = np.asarray([e[2] for e in entries], dtype=np.float64)
        require(x.shape == (n, gate.WIDTH) and np.isfinite(x).all(), "Finite frozen disagreement features")
        c, q = gate.scores(x, model["C"]), gate.scores(x, model["Q"])
        c_take, q_take = np.flatnonzero(c > 0), np.flatnonzero(q > 0)
        cm_take = np.argsort(-c, kind="stable")[:len(q_take)]
        counts["disagreements_" + side] += n
        for key, take in (("C", c_take), ("CM", cm_take), ("Q", q_take)):
            counts[key + "_accepted_" + side] += len(take)
            activations[key].extend(points[take].reshape(-1).tolist())
        counts["Q_CM_shared_" + side] += len(set(q_take) & set(cm_take))
        complete.extend(flat.tolist())
    require(sparse.xor_rle(static, complete) == learned, "Correction pairs do not exactly span the old SS to LL change")
    outputs = {}
    for key, pixels in activations.items():
        rle = sparse.xor_rle(static, pixels)
        area = int(codec.area(native))
        require(int(codec.area(rle)) == area and int(codec.area(codec.merge([rle, native], intersect=True))) == area-k,
                "Actual corrected signed budget differs")
        outputs[key] = dict(rle=rle, swaps=k)
    return outputs


def run(a, gate, d, sparse):
    feat, feat_sha, feat_term_sha = gate.observed(a.features, a.feature_terminal, "prepare-pairs", PREPARED)
    fit, fit_sha, fit_term_sha = gate.observed(a.fit, a.fit_terminal, "fit", gate.FIT)
    p = d.PINS[a.cohort]
    require(feat["protocol_sha256"] == fit["protocol_sha256"] == PRELOCK and feat["cohort"] == a.cohort
            and feat["images"] == p["images"] and feat["gate_model_sha256"] == GATE
            and feat["target_labels_opened"] is False and feat["detector_forwards"] == 0
            and fit["target_labels_opened"] is False and fit["ridge_solves"] == 2
            and fit["worker_sha256"] == GATE, "Frozen target features and source models differ")
    reference = d.historical(a, "verify-edits")
    require(reference["edits_sha256"] == feat["classical_edits_sha256"], "Features do not bind the prior classical edits")
    model = json.loads(gate.pinned(a.fit/"MODEL_PRIVATE.json", fit["model_sha256"]))
    require(model["status"] == gate.FIT and model["protocol_sha256"] == PRELOCK, "Wrong model file")
    model = model["model"]
    require(set(model) == {"C", "Q"} and all(
        len(model[k]["weight"]) == len(model[k]["rms"]) == n and all(v > 0 for v in model[k]["rms"])
        for k, n in (("C", 2), ("Q", gate.WIDTH))), "Invalid frozen ridge dimensions")
    prior_path = a.root/p["history"]/"prepare-edits/EDITS_PRIVATE.jsonl"
    reader = d.rows(a, prior_path, reference["edits_sha256"])
    counts, out_sha, feat_sha_reader = Counter(), sha256(), sha256()
    a.output.mkdir()
    with (a.features/"PAIRS_PRIVATE.jsonl").open("rb") as examples, (a.output/"EDITS_PRIVATE.jsonl").open("xb") as out:
        for i, ((item, pred, audit, old), line) in enumerate(zip(reader, examples, strict=True), 1):
            require(line.endswith(b"\n"), "Incomplete feature row")
            feat_sha_reader.update(line)
            row = json.loads(line)
            require(row["image_id"] == item["image_id"] and row.get("rgb_sha256", old.get("rgb_sha256")) ==
                    old.get("rgb_sha256"), "Feature image order or RGB differs")
            choices = row["changes"]
            actions = {}
            for key, saved in old["changes"].items():
                versions = {k: saved.get(d.NAMES[k]) for k in ("SS", "LL", "DD")}
                budgets = {v["swaps"] if v else 0 for v in versions.values()}
                require(len(budgets) == 1, "Archived allocations differ")
                k = next(iter(budgets))
                if not k:
                    continue
                rank = int(key)
                require(str(rank) == key and 1 <= rank <= 100, "Original candidate identity differs")
                native = pred[rank-1]["segmentation"]
                require(int(codec.area(native)) == audit[rank-1]["mask_area"], "Original stock area differs")
                pair = choices.get(key, dict.fromkeys(gate.SIDES, []))
                require(set(pair) == set(gate.SIDES), "Incomplete signed feature record")
                new = corrected(versions["SS"]["rle"], versions["LL"]["rle"], native, pair, model, gate, sparse, k, counts)
                actions[key] = dict(arms=versions | new)
                counts.update(edited_candidates=1, swaps_per_arm=k)
            require(choices.keys() <= actions.keys(), "Features selected an inactive candidate")
            result_row = dict(image_id=item["image_id"], changes=actions)
            if a.cohort == "pseg232":
                result_row["rgb_sha256"] = old["rgb_sha256"]
            data = gate.encoded(result_row)
            out.write(data)
            out_sha.update(data)
            if i % 100 == 0:
                print(json.dumps(dict(replayed_images=i)), flush=True)
    require(feat_sha_reader.hexdigest() == feat["pairs_sha256"] and counts["edited_candidates"] == p["positive"]
            and counts["swaps_per_arm"] == p["swaps"] and
            all(counts["Q_accepted_"+side] == counts["CM_accepted_"+side] for side in gate.SIDES),
            "Observed feature bytes or per-sign swap/accept budgets differ")
    return dict(status=STATUS, protocol_sha256=PRELOCK, cohort=a.cohort, images=p["images"], counts=dict(counts),
                edits_sha256=out_sha.hexdigest(), worker_sha256=gate.digest(Path(__file__).read_bytes()),
                fit_result_sha256=fit_sha, fit_terminal_sha256=fit_term_sha, model_sha256=fit["model_sha256"],
                feature_result_sha256=feat_sha, feature_terminal_sha256=feat_term_sha,
                target_labels_opened=False, detector_forwards=0, head_forwards=0, optimizer_updates=0,
                original_pure_arms_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cohort", choices=("dev705", "pseg232"), required=True)
    for name in ("root", "code", "manifest", "protocol", "features", "feature-terminal", "fit", "fit-terminal", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    require(args.protocol_sha256 == PRELOCK and not args.output.exists(), "Locked protocol and fresh output required")
    gate = source(Path(__file__).with_name("gate_model.py"), GATE)
    gate.pinned(args.protocol, PRELOCK)
    d = source(args.code/"ATTRACTION_SIGNED_SIDES_20261123/dfine_sides.py", DFINE)
    sparse = source(args.code/"RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py", SPARSE)
    start = perf_counter()
    result = run(args, gate, d, sparse)
    result["wall_seconds_inside_worker"] = perf_counter() - start
    output = gate.encoded(result)
    (args.output/"RESULT_SAFE.json").write_bytes(output)
    print(json.dumps(dict(status=STATUS, result_sha256=gate.digest(output))), flush=True)


if __name__ == "__main__":
    main()

