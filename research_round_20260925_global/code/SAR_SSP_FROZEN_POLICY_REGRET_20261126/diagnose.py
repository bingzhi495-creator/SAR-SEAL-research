"""Retrospective native-match audit of a closed SAR-SSP transfer."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
WORKER = "83ff61406235b142b86c2bcdbebbc92b3b17a7d3eb48032cd6c6619ec7e6f8a2"
RENDER = "9cf1753d4818df64717217831d29421f329eae21014b4656e52be83ecefd1701"
SCORE = "a857f06b37ca65e38205e24c5fd024e837f533dbf1ba6c448fed5d2e4727e837"
RENDER_TERM = "1ea6b73fc7f977ad7f3cce632c97083fb2d3fa6b8c153ce4480dd212ec67c582"
SCORE_TERM = "faa2c69b75cb23bfd5077a338a6682a9fd6233e6c66720fc713c1f64108f57c3"
LOCK = "3d2e4788b3ce58517d1bd6573828c0d5caaf1815fe4df840b32d91e2555dc3f7"
ARMS = ("fixed", "variable", "rank_only", "allocation_only", "full_menu", "full_support", "exact_geometry")
CONTRASTS = (
    ("variable", "fixed"), ("rank_only", "variable"), ("full_menu", "rank_only"),
    ("full_menu", "variable"), ("full_support", "full_menu"),
    ("allocation_only", "variable"), ("full_support", "fixed"),
    ("exact_geometry", "variable"), ("exact_geometry", "fixed"),
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Bound file mismatch: " + path.name)
    return raw


def closed(root):
    for phase, result, terminal, tag in (("render", RENDER, RENDER_TERM, "capture"), ("score", SCORE, SCORE_TERM, "verify")):
        observed = json.loads(pin(root / ("observed_" + phase) / "TERMINAL.json", terminal))
        content = json.loads(pin(root / phase / "RESULT_SAFE.json", result))
        need(observed["actual_exit"] == 0 and observed["output_footer_verified"] is True
             and observed["phase"] == tag and observed["worker_sha256"] == content["worker_sha256"] == WORKER
             and observed["result_sha256"] == result, "Original transfer has a verified natural exit")
        if phase == "render":
            writeback = content
        else:
            scored = content
    need(writeback["gate_labels_opened"] is False and scored["gate_labels_opened"] is True
         and scored["render_result_sha256"] == RENDER and scored["original_control_parity"],
         "Exact previously scored target writer")
    worker = root / "transfer.py"
    pin(worker, WORKER)
    return writeback, scored, SimpleNamespace(**runpy.run_path(str(worker)))


def core_run(root):
    started = perf_counter()
    render, scored, prior = closed(root)
    codecs = prior.OLD / "capture/CODEC_PRIVATE"
    sys.path[:0] = [str(codecs), str(prior.OLD / "numpy_1_26_4_only"), str(HERE)]
    from frontier import audit
    from pycocotools import mask as codec
    import numpy as np

    old, m, args, roster, capture, checked, _, scoring, policy = prior.setup()
    frozen = scoring.read_stream(root / "render/RENDER_PRIVATE.jsonl", render["raw_sha256"])
    archived = scoring.read_stream(prior.OLD / "prepare/EDITS_PRIVATE.jsonl", checked["edits_sha256"])
    records = {}
    for j, (item, native, _, _) in enumerate(m.legacy.blocks(args, m.author, roster, capture, False)):
        iid = item["image_id"]
        now, historical = frozen[j], archived[j]
        need(iid == now["image_id"] == historical["image_id"] and len(native) == 100
             and set(now["edits"]) == {position for position, choices in historical["changes"].items()
                                       if "attraction_static_at_dynamic_count" in choices},
             "Exact four-arm edited subset at unchanged detector scores")
        records[iid] = [dict(native=row["segmentation"], score=row["score"], masks={}) for row in native]
    need(len(records) == len(frozen) == len(archived) == 232 and sum(len(row["edits"]) for row in frozen) == 10393,
         "Complete previously scored candidate roster")
    need(old.file_hash(prior.ARCHIVE) == scoring.ARCHIVE_SHA, "Exact original SAR-SSP author archive")
    names = {row["image_id"]: (Path(row["image_ref"]["path"]).name, row["image_ref"]["sha256"]) for row in roster}
    with ZipFile(prior.ARCHIVE) as archive:
        truth, groups, _ = scoring.exact_ground_truth(archive, names)
    evaluator = scoring.load_evaluator(prior.CODE / scoring.EVALUATOR_PATH)
    expected_ap = evaluator(truth, scoring.coco_rows(records, "native"))["AP"]
    need(expected_ap * 100 == scored["scoring"]["coco_mask_metric_points"]["native"]["AP"], "Original native AP parity")
    matches = scoring.native_matches(truth, records, expected_ap)
    need(len(matches) == 88 and len(truth["annotations"]) == 139, "Exact native fixed match denominator")
    by_image = defaultdict(dict)
    per_parent = defaultdict(list)
    counts = Counter()
    writeback_by_image = {row["image_id"]: row["edits"] for row in frozen}
    for annot in truth["annotations"]:
        aid, iid = annot["id"], annot["image_id"]
        if aid in matches:
            mid, rank = matches[aid]
            need(mid == iid, "Same-image fixed match")
            if str(rank) in writeback_by_image[iid]:
                by_image[iid][rank] = annot
                continue
            baseline = scoring.hull_measures(records[iid][rank-1]["native"], annot["segmentation"], annot["area"])[0]
        else:
            baseline = 0.
        per_parent[groups[iid]].append(np.full(len(ARMS), baseline))

    core, model, center, scale, _, attraction = prior.load_fit()
    feature_source, rank_source = m.native.load_old(args.pair_math, args.rank_math)
    adapted, probability = m.native.make_native(feature_source, rank_source, 500, 500)
    for ordinal, (item, native, meta, planes) in enumerate(m.legacy.blocks(args, m.author, roster, capture, True)):
        iid = item["image_id"]
        if iid not in by_image:
            continue
        image = m.legacy.rgb(args, item)
        need(digest(image.tobytes(order="C")) == frozen[ordinal]["rgb_sha256"] == item["rgb_sha256"],
             "Exact original native pixels for the point head")
        gray = feature_source.image_features(image)
        eligible = []
        for key, row in frozen[ordinal]["edits"].items():
            rank = int(key)
            main, det, plane = native[rank-1], meta[rank-1], planes[rank-1]
            area, k = det["mask_area"], row["k"]
            stock = codec.decode(main["segmentation"]).astype(bool, order="C")
            coords, features, _, _, _ = adapted.paired_points(
                stock, det["box_xyxy"], area, main["score"], plane, gray, probability)
            need(coords.tolist() == row["coordinates"], "Exact original signed endpoint positions")
            _, static = attraction.select(stock, coords, min(8, area//10, len(coords)))
            incumbent = prior.static_pair(static)
            need(len(static) == k, "Exact archived attraction quota")
            eligible.append((rank, k, area, stock, coords, features, incumbent, row))
        eligible.sort(key=lambda row: row[0])
        predictions = core.predict(model, np.concatenate([row[5] for row in eligible]), center, scale)
        cursor = 0
        for rank, k, area, stock, coords, _, incumbent, row in eligible:
            probability_rows = predictions[cursor:cursor+len(coords)]
            cursor += len(coords)
            if rank not in by_image[iid]:
                continue
            annot = by_image[iid][rank]
            gt = codec.decode(annot["segmentation"]).ravel(order="C").astype(np.int64)
            native_hit = int((gt & stock.ravel(order="C")).sum())
            observed = {name: tuple(tuple(side) for side in row["arms"][name]["action"])
                        for name in ("fixed_iou", "variable_iou")}
            for name, action in observed.items():
                mask, signed = policy.write_mask(stock, coords, action, k, codec)
                need(mask == row["arms"][name]["rle"] and list(signed) == row["arms"][name]["counts"],
                     "Exact frozen native flips on the fixed match")
            result = audit(policy, probability_rows, incumbent, k, area, annot["area"], native_hit,
                           gt[coords[:, 0]], gt[coords[:, 1]], observed)
            per_parent[groups[iid]].append(np.asarray([result[name]*100 for name in ARMS]))
            counts.update(eligible_hulls=1, feasible_quotas=result["feasible_quotas"],
                          exact_geometry_changed_quota=int(result["exact_geometry_adds"] != result["chosen_adds"]),
                          observed_fixed_variable_changed=int(observed["fixed_iou"] != observed["variable_iou"]))
        need(cursor == len(predictions), "Original candidate-batch prediction census")
    values = np.asarray([np.mean(per_parent[key], axis=0) for key in sorted(per_parent)])
    need(values.shape == (23,len(ARMS)) and sum(map(len,per_parent.values())) == 139
         and counts["observed_fixed_variable_changed"] == 48, "All original denominator and exact changed hulls")
    means = dict(zip(ARMS, map(float, values.mean(axis=0)), strict=True))
    historical = scored["scoring"]["fixed_native_match_all_truth"]["arms"]
    need(all(abs(means[arm]-historical[arm+"_iou"]["iou_points"]) < 1e-11 for arm in ("fixed","variable")),
         "Exact historical group weighting parity on fixed and variable masks")
    draws = np.random.Generator(np.random.PCG64(5217)).integers(23, size=(4096,23))
    comparisons = {}
    for left,right in CONTRASTS:
        paired = values[:,ARMS.index(left)] - values[:,ARMS.index(right)]
        intervals = np.percentile(paired[draws].mean(axis=1), [2.5,97.5]).tolist()
        comparisons[left+"_vs_"+right] = dict(delta=float(paired.mean()), interval95=intervals)
    return dict(status="SAR_SSP_FROZEN_POLICY_REGRET_COMPLETE", elapsed_seconds=perf_counter()-started,
                historical_exposure=True, author_label_assisted=True, detector_forwards=0, optimizer_updates=0,
                n_hulls=139, n_groups=23, native_matched_hulls=88, fixed_variable_exact_parity=True,
                candidate_frontier_counts=dict(counts), iou_group_equal_points=means, paired=comparisons,
                protocol_sha256=LOCK, worker_sha256=digest(Path(__file__).read_bytes()),
                prior_render_sha256=RENDER, prior_score_sha256=SCORE)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--closed-transfer",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated and sys.dont_write_bytecode,
         "Fresh isolated CPU audit")
    need(not a.output.exists(), "Fresh result path")
    pin(HERE / "PRELOCK.md", LOCK)
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("This audit is offline")
    sys.addaudithook(offline)
    result = core_run(a.closed_transfer)
    body = (json.dumps(result, sort_keys=True, allow_nan=False)+"\n").encode("ascii")
    with a.output.open("xb") as out:
        out.write(body)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=digest(body))), flush=True)

if __name__ == "__main__":
    main()
