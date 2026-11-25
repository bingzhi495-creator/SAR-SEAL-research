"""Retrospective candidate geometry on two exactly closed SAR-SSP captures."""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

PROTOCOL = "9a74a8c9702a56f12a0f5d9fccfec4f4e6155ecd52680b619503fa1229f31d87"
SCORER = "d4d16cdfab0903480c7e6f3686de973886b726d54f91b7e235686c1502d44735"
INTAKE = "16c8e6b7655e51f208c3a9708acd419a60126e55a9075cdea7d4b9e4edbb9425"
RF_CAPTURE = "c40a133d62fafdecb9736d17abbd0ddcbf31ec42672d61eb0cbfc5d5df0b643e"
RF_READBACK = "5788cd43fdc1e404cf6cc3c611dec47bb8ed30c55b926b22aabb6d3ec80e3711"
RF_SCORE = "fa2508ebde848235dada628406f095a330955f103fa611256964a3f714ea9972"
DF_CAPTURE = "0e15ca971ac2a6f9318b16710977ea5b86961c233a5effeca5649d348bc84155"
DF_READBACK = "223715cb6c21a6d9b20927410a914605c51e2fca566df20e44a1ec71596fc853"
DF_SCORE = "77da6f63f83e23c804a8773bb40d432531241bab68b3871bdf7ba936d79e9bc6"
ERRATUM = "aebe42e401aebd7208ed3950a7b08cb99e4ba2df6393d9fb818c812c1f845a8a"
RUNTIME_HW = [500, 500]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, f"Known input digest: {path.name}")
    return raw


def load_json(path, digest):
    return json.loads(pin(path, digest))


def closed(path, phase, result_digest):
    data = json.loads(path.read_bytes())
    require(data["phase"] == phase and data["result_sha256"] == result_digest
            and data.get("actual_exit", data.get("exit_code")) == 0
            and data.get("output_footer_verified", data.get("footer_verified")) is True,
            "Independent original terminal verification")
    return sha256(path.read_bytes()).hexdigest()


def source(path, digest):
    m = ModuleType("original_ground_truth")
    m.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), m.__dict__)
    return m


def rows(root, prediction_file, audit_file, receipt, is_rf, roster):
    preds = pin(root / prediction_file, receipt["predictions_sha256"]).splitlines()
    audits = pin(root / audit_file, receipt["audit_sha256" if is_rf else "candidate_audit_sha256"]).splitlines()
    require(len(preds) == len(audits) == len(roster) == 232, "Complete aligned capture lengths")
    records = {}
    for expected, pred_raw, audit_raw in zip(roster, preds, audits, strict=True):
        pred, audit = json.loads(pred_raw), json.loads(audit_raw)
        iid = expected["image_id"]
        require(iid == audit["image_id"] == pred["image_id"] and iid not in records,
                "Exact image identity and manifest order")
        if is_rf:
            require(audit["image_sha256"] == expected["image_ref"]["sha256"]
                    and audit["source_image_name"] == expected["source_image_name"]
                    and audit["rgb_sha256"] == expected["rgb_sha256"]
                    and audit["original_hw"] == expected["original_hw"], "Exact RF native image identity")
        prediction, metadata = pred["coco"], audit["candidates"]
        require(len(prediction) == len(metadata)
                and (len(prediction) <= 100 if is_rf else len(prediction) == audit["post_nms"] >= 100),
                "Original capture scope")
        ranks = [candidate["native_rank"] for candidate in metadata]
        require(ranks == sorted(set(ranks)), "Original unique candidate ranks")
        if is_rf:
            require(ranks == audit["native_ranks"] == pred["native_ranks"], "Exact RF ranks in both streams")
        else:
            require(ranks == list(range(1, len(prediction) + 1)), "Exact D-FINE native contiguous ranks")
        data = []
        previous = np.inf
        for stock, meta in zip(prediction[:100], metadata[:100], strict=True):
            score, mask = stock["score"], stock["segmentation"]
            require(stock["image_id"] == iid and stock["category_id"] == 1
                    and score == meta["score"] and 0 <= score <= previous
                    and mask["size"] == RUNTIME_HW and int(codec.area(mask)) == meta["mask_area"],
                    "Exact native candidate score, class and raster")
            x0, y0, x1, y1 = meta["box_xyxy"]
            require(all(0 <= v <= 500 for v in (x0, y0, x1, y1))
                    and (not is_rf or (x0 <= x1 and y0 <= y1)), "Valid original clipped box endpoints")
            data.append((score, mask, [x0, y0, max(0., x1 - x0), max(0., y1 - y0)]))
            previous = score
        records[iid] = data
    return records


def matching_matrix(stream, instances, modality):
    source = [row[1 if modality == "mask" else 2] for row in stream]
    target = [row["segmentation" if modality == "mask" else "bbox"] for row in instances]
    if not source or not target:
        return np.zeros((len(source), len(target)))
    overlaps = codec.iou(source, target, [0] * len(target))
    require(np.isfinite(overlaps).all() and overlaps.min() >= 0 and overlaps.max() <= 1, "Valid IoU matrix")
    return overlaps


def match(iou, threshold):
    qualified = iou >= threshold
    if not qualified.any():
        return set()
    dominance = min(qualified.shape) + 1
    rr, cc = linear_sum_assignment(np.where(qualified, dominance + iou, 0), maximize=True)
    return {int(c) for r, c in zip(rr, cc, strict=True) if qualified[r, c]}


def rescue(iou, matched, threshold):
    missed_columns = np.flatnonzero(~np.asarray(matched, dtype=bool))
    novel = iou[:, missed_columns]
    opportunities = (novel >= threshold).any(axis=1)
    conflicts = (iou[:, matched] >= threshold).any(axis=1)
    clean = opportunities & ~conflicts
    unrestricted_gt = {int(missed_columns[i]) for i in match(novel, threshold)}
    clean_gt = {int(missed_columns[i]) for i in match(novel[clean, :], threshold)}
    return int(opportunities.sum()), int((opportunities & conflicts).sum()), unrestricted_gt, clean_gt


def greedy(iou, threshold):
    """COCO score order, author truth ties resolve to the last available column."""
    assigned = {}
    available = np.ones(iou.shape[1], dtype=bool)
    for j, candidate in enumerate(iou):
        eligible = np.flatnonzero(available & (candidate >= threshold))
        if eligible.size:
            gt = int(eligible[np.flatnonzero(candidate[eligible] == candidate[eligible].max())[-1]])
            assigned[gt] = j
            available[gt] = False
    return assigned


def stock_coco(truth, records):
    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(truth)
        ref.createIndex()
        rows = [dict(image_id=iid, category_id=0, score=s, segmentation=m)
                for iid, stream in records.items() for s, m, _ in stream]
        pred = ref.loadRes(deepcopy(rows))
        scorer = COCOeval(ref, pred, "segm")
        scorer.params.imgIds = list(records)
        scorer.params.catIds = [0]
        scorer.params.maxDets = [1, 10, 100]
        scorer.params.iouThrs = np.array([.50])
        scorer.evaluate()
    lookup = {i: iid for i, (iid, _) in enumerate(
        ((iid, j) for iid, stream in records.items() for j in range(len(stream))), 1)}
    by_image = defaultdict(dict)
    for row in scorer.evalImgs:
        if row is None or row["maxDet"] != 100 or row["aRng"] != scorer.params.areaRng[0]:
            continue
        require(not any(row["gtIgnore"]), "No ignored original author hull")
        iid = row["image_id"]
        detection_index = {c: j for j, c in enumerate(row["dtIds"])}
        require(all(lookup[c] == iid for c in detection_index), "Original localization row identity")
        by_image[iid].update({int(tid): detection_index[int(did)]
                              for tid, did in zip(row["gtIds"], row["gtMatches"][0], strict=True) if did})
    return by_image


def group_summary(counts, totals, draws):
    positions = [counts[g] for g in totals]
    balanced = np.array([counts[g] / n for g, n in totals.items()])
    sampler = balanced[draws].mean(axis=1)
    return dict(hulls=int(sum(positions)), parent_groups_positive=int(np.count_nonzero(positions)),
                largest_parent_hull_count=max(positions, default=0),
                parent_balanced_fraction=float(balanced.mean()),
                percentile95_fraction=np.percentile(sampler, [2.5, 97.5]).tolist())


def summarize(a, b, truth, groups):
    labeled = defaultdict(list)
    parents = defaultdict(int)
    for row in truth["annotations"]:
        labeled[row["image_id"]].append(row)
        parents[groups[row["image_id"]]] += 1
    draws = np.random.Generator(np.random.PCG64(5323)).integers(0, len(parents), (4096, len(parents)))
    native_a, native_b = stock_coco(truth, a), stock_coco(truth, b)
    totals = [sum(map(len, state.values())) for state in (native_a, native_b)]
    require(totals == [92, 88], "Independent native score-ordered count parity")
    report, by_modality = {}, {}
    support_categories = ("rf_only", "dfine_only", "both", "neither")
    for modality in ("mask", "box"):
        threshold_rows = {}
        extra = dict(donor_candidates=0, overlapping_matched_hull_candidates=0,
                     unrestricted=defaultdict(int), conflict_free=defaultdict(int))
        for ii, instances in labeled.items():
            im_group = groups[ii]
            ia, ib = (matching_matrix(s[ii], instances, modality) for s in (a, b))
            if modality == "mask":
                for matrix, state in ((ia, native_a), (ib, native_b)):
                    expected = {instances[col]["id"]: row for col, row in greedy(matrix, .50).items()}
                    require(expected == state[ii], "Exact COCO-versus-independent native association")
            for t in (.25, .50, .75):
                state = threshold_rows.setdefault(t, dict.fromkeys(
                    (*support_categories, "rf_max_one_to_one", "dfine_max_one_to_one", "union_max_one_to_one",
                     "original_rf_unmatched_with_rf", "original_rf_unmatched_with_dfine"), 0))
                ca, cb = (matrix.max(axis=0, initial=0) >= t for matrix in (ia, ib))
                for tag, selected in zip(support_categories, (ca & ~cb, ~ca & cb, ca & cb, ~ca & ~cb)):
                    state[tag] += int(selected.sum())
                state["rf_max_one_to_one"] += len(match(ia, t))
                state["dfine_max_one_to_one"] += len(match(ib, t))
                state["union_max_one_to_one"] += len(match(np.concatenate((ia, ib)), t))
                matched = [row["id"] in native_a[ii] for row in instances]
                state["original_rf_unmatched_with_rf"] += int((~np.asarray(matched) & ca).sum())
                state["original_rf_unmatched_with_dfine"] += int((~np.asarray(matched) & cb).sum())
            n, conflicts, all_indices, clear_indices = rescue(ib, matched, .50)
            extra["donor_candidates"] += n
            extra["overlapping_matched_hull_candidates"] += conflicts
            extra["unrestricted"][im_group] += len(all_indices)
            extra["conflict_free"][im_group] += len(clear_indices)
        report[modality] = threshold_rows
        for label in ("unrestricted", "conflict_free"):
            extra[label] = group_summary(extra[label], parents, draws)
        by_modality[modality] = extra
    clean = by_modality["mask"]["conflict_free"]
    return dict(threshold_coverage=report, rf_locked_donor_at_050=by_modality,
                native_rf_score_matched=totals[0], native_dfine_score_matched=totals[1],
                native_rf_unmatched=139-totals[0], paired_parent_draws=len(draws),
                original_ground_truth_hulls=len(truth["annotations"]),
                ship_bearing_parents=len(parents), all_parents=len(set(groups.values())),
                opportunity_gate_pass=clean["hulls"] >= 8 and clean["parent_groups_positive"] >= 5)


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("protocol", "rf-root", "rf-score-root", "dfine-root", "scorer", "archive", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    started = perf_counter()
    pin(args.protocol, PROTOCOL)
    pin(args.protocol.with_name("ERRATUM.md"), ERRATUM)
    o = source(args.scorer, SCORER)
    rf, df, rs = args.rf_root, args.dfine_root, args.rf_score_root
    rf_capture = load_json(rf / "capture/RESULT_SAFE.json", RF_CAPTURE)
    rf_verify = load_json(rf / "verify-capture/RESULT_SAFE.json", RF_READBACK)
    rf_score = load_json(rs / "score/RESULT_SAFE.json", RF_SCORE)
    df_capture = load_json(df / "capture/RESULT_SAFE.json", DF_CAPTURE)
    df_verify = load_json(df / "verify/RESULT_SAFE.json", DF_READBACK)
    df_score = load_json(df / "score/RESULT_SAFE.json", DF_SCORE)
    terminals = dict(rf_capture=closed(rf / "observed_capture/TERMINAL.json", "capture", RF_CAPTURE),
                     rf_verify=closed(rf / "observed_verify-capture/TERMINAL.json", "verify-capture", RF_READBACK),
                     rf_score=closed(rs / "observed_score/TERMINAL.json", "score", RF_SCORE),
                     dfine_capture=closed(df / "CAPTURE_TERMINAL.json", "capture", DF_CAPTURE),
                     dfine_verify=closed(df / "VERIFY_TERMINAL.json", "verify", DF_READBACK),
                     dfine_score=closed(df / "SCORE_TERMINAL.json", "score", DF_SCORE))
    require(rf_capture["archive_sha256"] == rf_verify["archive_sha256"] == o.ARCHIVE_SHA
            and rf_capture["intake_sha256"] == rf_verify["intake_sha256"] == df_capture["manifest_sha256"] == INTAKE
            and rf_capture["predictions_sha256"] == rf_verify["predictions_sha256"]
            and rf_capture["audit_sha256"] == rf_verify["audit_sha256"]
            and rf_score["capture_sha256"] == RF_CAPTURE and rf_score["capture_readback_sha256"] == RF_READBACK
            and df_verify["capture_sha256"] == df_score["capture_sha256"] == DF_CAPTURE
            and df_score["prior_verify_sha256"] == DF_READBACK
            and rf_capture["target_labels_opened"] is rf_verify["target_labels_opened"]
            is df_capture["GT_opened"] is False, "Original native capture, readback and target-score identity")
    manifest = load_json(rf / "intake/IMAGE_MANIFEST_PRIVATE.json", INTAKE)
    roster = manifest["images"]
    names = {item["image_id"]: (item["source_image_name"], item["image_ref"]["sha256"]) for item in roster}
    require(manifest["archive_sha256"] == o.ARCHIVE_SHA and len(names) == len(roster) == 232
            and all(row["original_hw"] == RUNTIME_HW for row in roster), "Same 232-plate author intake")
    a = rows(rf / "capture", "PREDICTIONS_PRIVATE.jsonl", "CANDIDATE_AUDIT_PRIVATE.jsonl", rf_capture, True, roster)
    b = rows(df / "capture", "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl",
             df_capture, False, roster)
    require(sum(map(len, a.values())) == 22741 and sum(map(len, b.values())) == 23200, "Original top-100 scope")
    with args.archive.open("rb") as archive:
        digest = sha256()
        for block in iter(lambda: archive.read(8 * 1024**2), b""):
            digest.update(block)
        require(digest.hexdigest() == o.ARCHIVE_SHA, "Same author archive")
    with ZipFile(args.archive) as archive:
        for name, digest in names.values():
            require(sha256(archive.read("datasets/SAR-SSP/val2017/" + name)).hexdigest() == digest,
                    "Each original encoded target JPEG")
        print("Exact closed captures, sorted score/geometry and all 232 author JPEG identities accepted.",
              flush=True)
        truth, groups, edge = o.exact_ground_truth(archive, names)
    result = summarize(a, b, truth, groups)
    result.update(status="SAR_SSP_CROSS_CARRIER_NATIVE_COMPLEMENT_COMPLETE", protocol_sha256=PROTOCOL,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), erratum_sha256=ERRATUM,
                  archive_sha256=o.ARCHIVE_SHA, shared_jpegs=len(names), original_score_sha256=[RF_SCORE, DF_SCORE],
                  native_capture_sha256=[RF_CAPTURE, DF_CAPTURE], independent_readback_sha256=[RF_READBACK, DF_READBACK],
                  original_observed_terminal_sha256=terminals, target_labels_assessed_after_input_gate=True,
                  candidate_slots=[22741,23200], edge_touching_hulls=len(edge),
                  model_forwards=0, optimizer_updates=0, scored_AP=False, ensemble_executed=False,
                  prospective=False, elapsed_seconds=perf_counter() - started)
    raw = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with args.output.open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()

