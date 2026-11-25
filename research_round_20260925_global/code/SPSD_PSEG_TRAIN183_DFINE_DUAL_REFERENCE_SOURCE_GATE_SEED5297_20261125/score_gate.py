"""Score the sealed source-gate replay against each admitted author reference."""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment


PROTOCOL = "33467a9ea2df47d80843ad2cdb8abc567f9349b820d0e8111dec3b069686b313"
GROUP_PROTOCOL = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
FIT = "2ff91842249358aefd77915f87df3d337a814993245e0d973c057c21088ea538"
POINTS = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
LABELS = "bd4eb3add418d876f1defee88352aeee26c4f1b3135700436851fe73ad9b2796"
LABEL_READBACK = "de9455ecd6ac4db9fd3968c66e3039fb410efe3c6d5cc3f42f51c7eed893ff5e"
SPLIT = "91095aacbbbc89003ffd6eb8f25506ce99a64eee5ac369a7753203bbf962bcf9"
ARMS = ("native", "dynamic", "static", "dual_robust", "dual_mean", "dual_cell_pessimistic",
        "pseg_repeat_mean", "spsd_repeat_mean", "mean_repeat_mean")
LABEL_FILES = {"pseg": "PSeg_SSDD_COCO_PRIVATE.json", "spsd": "SPSD_COCO_PRIVATE.json",
               "binding": "IMAGE_BINDINGS_PRIVATE.json"}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(blob):
    return sha256(blob).hexdigest()


def pin(path, expected):
    data = path.read_bytes()
    need(digest(data) == expected, "Bound file digest: " + path.name)
    return data


def exact_readback(a):
    terminal_path = a.render_root / "RENDER_TERMINAL.json"
    term_raw = terminal_path.read_bytes()
    term = json.loads(term_raw)
    need(term["phase"] == "render" and term["actual_exit"] == 0
         and term["output_footer_verified"] is True, "Independent realized label-free render termination")
    accepted = json.loads(pin(a.render_root / "render/RESULT_SAFE.json", term["result_sha256"]))
    pin(a.render_root / "render_gate.py", term["worker_sha256"])
    need(accepted["worker_sha256"] == term["worker_sha256"]
         and accepted["protocol_sha256"] == PROTOCOL and accepted["group_protocol_sha256"] == GROUP_PROTOCOL
         and accepted["fit_result_sha256"] == FIT and accepted["points_result_sha256"] == POINTS
         and accepted["status"] == "SPSD_DUAL_GATE_RENDER_COMPLETE" and accepted["seed"] == 5297
         and accepted["detector_forwards"] == accepted["optimizer_updates"] == 0
         and accepted["historical_test_opened"] is accepted["source_gate_labels_opened"] is False,
         "Replay belongs to the exact proven prelabel run")
    stream = pin(a.render_root / "render/GATE_RENDER_PRIVATE.jsonl", accepted["render_stream_sha256"])
    rows = [json.loads(row) for row in stream.splitlines()]
    groups = {row["image_id"]: row["source_group"] for row in rows}
    need(len(groups) == len(rows) == accepted["counts"]["images"] == 183
         and len(set(groups.values())) == accepted["decoded_groups"] == 180, "Complete replay gate population")
    budget, changed = 0, 0
    verified = dict.fromkeys(ARMS[1:], 0)
    native, record = {}, {}
    for item in rows:
        image = item["image_id"]
        original = item["candidates"]
        native[image] = original
        record[image] = item
        need(len(original) == 100, "Original native slots")
        for p in original:
            k = p["quota"]
            area = int(codec.area(p["native"]))
            need(0 <= k <= min(8, area // 10) and p["native"]["size"] == item["original_hw"]
                 and 0 <= p["score"] <= 1 and set(p["changes"]) == (set(ARMS[1:]) if k else set()),
                 "Exact replay scope and signed quota")
            budget += k
            if k:
                changed += 1
                for arm, rle in p["changes"].items():
                    inter = int(codec.area(codec.merge([p["native"], rle], intersect=True)))
                    need(rle["size"] == item["original_hw"] and int(codec.area(rle)) == area
                         and area - inter == k, "Independent actual signed area verification")
                    verified[arm] += k
    need(changed == 8785 and budget == 57692 and set(verified.values()) == {budget}
         and sum(map(len, native.values())) == 18300, "Accepted full preserved candidate and edit counts")
    print("All 18,300 frozen slots and signed area parity independently verified before source-gate labels.", flush=True)
    return accepted, term, digest(term_raw), native, record, groups


def labels(a, record):
    exported = json.loads(pin(a.labels / "export/RESULT_SAFE.json", LABELS))
    checked = json.loads(pin(a.labels / "READBACK_SAFE.json", LABEL_READBACK))
    term = json.loads((a.labels / "LABEL_EXPORT_TERMINAL_PRIVATE.json").read_bytes())
    need(term["phase"] == "author_train_labels_export" and term["actual_exit"] == 0
         and term["result_sha256"] == LABELS and LABELS in term["terminal_output"]
         and checked["complete_readback"] is True and checked["result_sha256"] == LABELS
         and exported["partition_sha256"] == SPLIT and checked["partitions"] == exported["partitions"]
         and checked["test_annotations_opened"] == exported["test_annotation_members_opened"]
         == exported["test_images_opened"] == exported["fit_gate_decoded_overlap"]
         == exported["fit_gate_encoded_overlap"] == 0, "Closed author train-only label export")
    src = exported["partitions"]["source_gate"]
    raw = {k: json.loads(pin(a.labels / "export/source_gate" / f, src["artifact_hashes"][k]))
           for k, f in LABEL_FILES.items()}
    binding = raw["binding"]
    pseg, spsd = raw["pseg"], raw["spsd"]
    need(binding["partition"] == "source_gate" and binding["source_partition_sha256"] == SPLIT
         and src["images"] == 183 and src["decoded_groups"] == 180 and len(pseg["images"]) == 183
         and pseg["images"] == spsd["images"] and set(record) == {k["id"] for k in pseg["images"]}
         and len(pseg["annotations"]) == 411 and len(spsd["annotations"]) == 410,
         "Exact source gate-only reference population")
    for image in pseg["images"]:
        key, filename = image["id"], image["file_name"]
        source, attr = record[key], binding["image_bindings"][filename]
        dims = [image["height"], image["width"]]
        need(source["filename"] == filename and dims == source["original_hw"]
             == [attr["height"], attr["width"]] and attr["image_id"] == key, "Exact author gate alignment")
    return dict(pseg=pseg, spsd=spsd)


def scored(truth, native, arm):
    index, dt = {}, []
    for image, series in native.items():
        for rank, p in enumerate(series, 1):
            index[len(dt) + 1] = (image, rank)
            dt.append(dict(image_id=image, category_id=0, score=p["score"],
                           segmentation=p["changes"].get(arm, p["native"])))
    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = deepcopy(truth)
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(deepcopy(dt)), "segm")
        ev.params.catIds, ev.params.imgIds, ev.params.maxDets = [0], sorted(native), [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ap90_index, = np.flatnonzero(np.isclose(ev.params.iouThrs, .90))
    precision = ev.eval["precision"][ap90_index, :, 0, 0, 2]
    vals = precision[precision >= 0]
    summary = dict(AP=float(ev.stats[0]) * 100, AP50=float(ev.stats[1]) * 100,
                   AP75=float(ev.stats[2]) * 100, AP90=float(vals.mean()) * 100 if vals.size else None,
                   AR100=float(ev.stats[8]) * 100)
    match = {}
    count = 0
    if arm == "native":
        threshold, = np.flatnonzero(np.isclose(ev.params.iouThrs, .50))
        for row in ev.evalImgs:
            if not row or row["aRng"] != ev.params.areaRng[0] or row["maxDet"] != 100:
                continue
            count += len(row["gtIds"])
            for tid, did, ignore in zip(row["gtIds"], row["gtMatches"][threshold], row["gtIgnore"], strict=True):
                if did and not ignore:
                    slot = index[int(did)]
                    need(slot[0] == row["image_id"] and int(tid) not in match, "Native assignment identity parity")
                    match[int(tid)] = slot
        need(count == len(truth["annotations"]), "Native assignments account for all author rows")
    return summary, match


def fixed_values(reference, native, matches, groups, group_ids):
    hull_iou, hull_count = defaultdict(list), defaultdict(int)
    for anno in reference["annotations"]:
        iid = anno["image_id"]
        hull_count[iid] += 1
        linked = matches.get(anno["id"])
        if linked is None:
            hull_iou[iid].append(np.zeros(len(ARMS), dtype=np.float64))
        else:
            got_iid, rank = linked
            need(got_iid == iid, "Fixed original matched native candidate image")
            pos = native[iid][rank - 1]
            all_rles = [pos["changes"].get(arm, pos["native"]) for arm in ARMS]
            vec = codec.iou(all_rles, [anno["segmentation"]], [0])[:,0]
            need(np.isfinite(vec).all() and np.all((0 <= vec) & (vec <= 1)), "Valid raster IoU")
            hull_iou[iid].append(vec)
    grouped = defaultdict(list)
    for iid, g in groups.items():
        values = hull_iou[iid]
        img = np.mean(values, axis=0) if values else np.zeros(len(ARMS), dtype=np.float64)
        grouped[g].append(img)
    return np.stack([np.mean(grouped[g], axis=0) for g in group_ids]), len(matches), sum(v > 0 for v in hull_count.values())


def changed_common(truths, matches, native, groups):
    by_image = {}
    for key, truth in truths.items():
        by_image[key] = defaultdict(list)
        for entry in truth["annotations"]:
            by_image[key][entry["image_id"]].append(entry)
    joint, size, changed_groups = 0, 0, set()
    pair_count, eligible = 0, 0
    for iid, source_group in groups.items():
        a, b = by_image["pseg"][iid], by_image["spsd"][iid]
        if not a or not b:
            continue
        iou = codec.iou([x["segmentation"] for x in a], [x["segmentation"] for x in b], [0] * len(b))
        rows, cols = linear_sum_assignment(iou, maximize=True)
        pair_count += len(rows)
        for x, y in zip(rows, cols, strict=True):
            if iou[x,y] < .5:
                continue
            eligible += 1
            l = matches["pseg"].get(a[x]["id"])
            r = matches["spsd"].get(b[y]["id"])
            if l is None or l != r:
                continue
            joint += 1
            pos = native[iid][l[1] - 1]
            if pos["changes"].get("dual_robust") != pos["changes"].get("static"):
                size += 1
                changed_groups.add(source_group)
    return dict(reference_pairs=pair_count, reference_pairs_ge050=eligible, shared_native_matches=joint,
                changed_common_matched_hulls=size, changed_common_groups=len(changed_groups))


def contrast(values, ap, draws):
    robust = ARMS.index("dual_robust")
    table = {}
    for j, arm in enumerate(ARMS):
        if arm == "dual_robust":
            continue
        d = (values[:,robust] - values[:,j]) * 100
        lo, hi = np.percentile(d[draws].mean(axis=1), [2.5, 97.5])
        table[arm] = dict(iou_points=float(d.mean()), percentile95_points=[float(lo), float(hi)],
                          ap_points=ap["dual_robust"]["AP"] - ap[arm]["AP"])
    return table


def decision(refs, exposure):
    primary = {k: data["dual_robust_minus"]["static"] for k, data in refs.items()}
    mean_change = min(data["dual_robust_minus"]["dual_mean"]["iou_points"] for data in refs.values())
    pessimistic_change = min(data["dual_robust_minus"]["dual_cell_pessimistic"]["iou_points"] for data in refs.values())
    checks = dict(iou_magnitude=all(p["iou_points"] >= .15 for p in primary.values()),
                  iou_lower=all(p["percentile95_points"][0] > 0 for p in primary.values()),
                  ap_guard=all(p["ap_points"] >= -.10 for p in primary.values()),
                  exposure=exposure["changed_common_matched_hulls"] >= 20 and exposure["changed_common_groups"] >= 8,
                  vs_dual_mean=mean_change > 0, vs_dual_cell_pessimistic=pessimistic_change > 0)
    return dict(criteria=checks, first_seed_advances=all(checks.values()),
                worst_reference_iou_vs_dual_mean_points=mean_change,
                worst_reference_iou_vs_dual_cell_pessimistic_points=pessimistic_change)


def main():
    p = ArgumentParser(description=__doc__)
    for field in ("render-root", "labels", "protocol", "group-protocol", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    a = p.parse_args()
    started = perf_counter()
    pin(a.protocol, PROTOCOL)
    pin(a.group_protocol, GROUP_PROTOCOL)
    render, terminal, terminal_sha, native, record, groups = exact_readback(a)
    truths = labels(a, record)
    refs, native_assign = {}, {}
    group_ids = sorted(set(groups.values()))
    seed = 5297
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, 180, (4096, 180))
    for name, truth in truths.items():
        metrics = {}
        for arm in ARMS:
            metrics[arm], matches = scored(truth, native, arm)
            if arm == "native":
                native_assign[name] = matches
        values, matched, positive_images = fixed_values(truth, native, native_assign[name], groups, group_ids)
        refs[name] = dict(metrics=metrics, dual_robust_minus=contrast(values, metrics, draws),
                          fixed_iou_points={arm: float(g * 100) for arm, g in zip(ARMS, values.mean(axis=0), strict=True)},
                          author_hulls=len(truth["annotations"]), original_matches=matched,
                          original_unmatched=len(truth["annotations"])-matched, positive_images=positive_images)
        print("source gate scored reference", name, flush=True)
    change = changed_common(truths, native_assign, native, groups)
    result = dict(status="SPSD_DUAL_GATE_SEED5297_SCORED_COMPLETE", seed=seed, protocol_sha256=PROTOCOL,
                  group_protocol_sha256=GROUP_PROTOCOL, worker_sha256=digest(Path(__file__).read_bytes()),
                  render_result_sha256=terminal["result_sha256"], render_terminal_sha256=terminal_sha,
                  render_stream_sha256=render["render_stream_sha256"], fit_result_sha256=FIT,
                  label_export_sha256=LABELS, label_readback_sha256=LABEL_READBACK,
                  images=183, decoded_groups=180, optimizer_updates=0, detector_forwards=0,
                  point_quota=render["counts"]["quota"], bootstrap_draws=4096, candidates=18300,
                  historical_test_opened=False, fit_labels_opened=False, editable_arm_counts=render["arms"],
                  common_match_exposure=change, references=refs, primary=decision(refs, change),
                  elapsed_seconds=perf_counter()-started)
    a.output.mkdir()
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (a.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()

