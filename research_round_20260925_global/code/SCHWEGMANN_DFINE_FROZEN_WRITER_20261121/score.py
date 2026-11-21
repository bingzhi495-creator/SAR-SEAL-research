"""Score preclosed D-FINE center-mask arms on the image-disjoint Schwegmann panel."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np


ARMS = ("raw", "fixed", "free_matched_count", "free_natural", "native_matched_count")
PAIRS = (("free_matched_count", "fixed"), ("free_natural", "raw"),
         ("free_natural", "native_matched_count"))
METRICS = ("iou", "dice", "recovery_050", "recovery_075")
GROUPS = tuple(f"acq_{i:03d}" for i in range(1, 17))
MANIFEST = "2485f984db978572e5d446847444f63a8055d868aeebbdede2fe018e3997fc46"
ARCHIVE = "78ff85f49eb5102716ae6d39bdbafaf27fc9c63a630be0d86a2c434ab09f053b"
GROUP_INDEX = "609c209aac59bee168b5338010bf4965d146ea5dc513192bafa84444e088b0f4"
INTAKE = "eec82422b8d16aa464ead5d9847f2f173788798a0f26696c3b069975962a5f1d"
QUALIFIER = "41704d3d3980b6ae6442d17b92625e37ed782e417e06a585806227d881ce7d79"
HISTORIC_SCORE = "5b9c3bc50223167079e1cd62882c32debcaaaf27be758b7bd273ea4083d89d61"
PROTOCOL = "847b8f53aec9764b443963741148401e1456712f872b02af65ec6ea6a50d4db9"
EMA = "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
HEAD = "dca00310ba7398790a874e2f32c7fa26324fed1b8e071372831a85e7a83ecd3a"


def need(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, expected):
    need(re.fullmatch(r"[0-9a-f]{64}", expected), "SHA-256 required for " + path.name)
    raw = path.read_bytes()
    need(digest(raw) == expected, "Input digest mismatch for " + path.name)
    return raw


def module(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(read(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def jsonl(raw, n):
    lines = raw.splitlines()
    need(len(lines) == n and raw.endswith(b"\n"), "Exact closed complete per-image JSONL")
    return map(json.loads, lines)


def selected_row(item, captured, edited, decode):
    ident, group = item["image_id"], item["acquisition_group"]
    need(ident == captured["image_id"] == edited["image_id"] and
         group == captured["group"] == edited["group"] and set(edited["arms"]) == set(ARMS),
         "Original image, group, and five-arm enrollment")
    candidates = captured["candidates"]
    need(isinstance(candidates, list) and len(candidates) <= 100, "Original post-NMS roster")
    rank = None
    for n, candidate in enumerate(candidates, 1):
        x0, y0, x1, y1 = candidate["box_xyxy"]
        if 40 <= (x0 + x1) / 2 < 61 and 40 <= (y0 + y1) / 2 < 61:
            rank = n
            break
    need(rank == captured["rank"] == edited["rank"], "Original center rule and selected row")
    arms = edited["arms"]
    if rank is None:
        need(captured["prediction"] is None and edited["score"] is None and
             all(v is None for v in arms.values()), "Missing candidate must stay absent in all arms")
        swaps = dict.fromkeys(ARMS, 0)
        check_declared_swaps(edited, swaps)
        return None, dict.fromkeys(ARMS), swaps
    prediction = captured["prediction"]
    selected = candidates[rank - 1]
    need(type(edited["rank"]) is int and selected["score"] == edited["score"] == prediction["score"]
         and prediction["image_id"] == ident and 0 <= edited["score"] <= 1 and
         arms["raw"] == prediction["segmentation"], "Original native RLE, rank and score")
    need(all(v is not None for v in arms.values()), "Present selection requires five RLEs including empty masks")
    masks = {name: decode(arms[name]) for name in ARMS}
    raw = masks["raw"]
    area = int(raw.sum())
    need(area == selected["mask_area"], "Original native mask area")
    swaps = {}
    for name, mask in masks.items():
        adds = int((~raw & mask).sum())
        drops = int((raw & ~mask).sum())
        need(adds == drops and adds <= min(8, area//10) and (32 <= area < 1024 or adds == 0),
             "Area, edit budget and eligible size band per image")
        swaps[name] = adds
    need(swaps["fixed"] == swaps["free_matched_count"] and
         swaps["free_natural"] == swaps["native_matched_count"], "Per-image matched action counts")
    check_declared_swaps(edited, swaps)
    return area, masks, swaps


def check_declared_swaps(edited, swaps):
    if "swaps" in edited:
        actual = edited["swaps"]
        need(isinstance(actual, dict) and set(actual) == set(ARMS[1:]) and
             all(type(actual[a]) is int and actual[a] == swaps[a] for a in ARMS[1:]),
             "Declared swaps differ from observed mask geometry")


def author_truth(raw, items, lex, qualifier):
    """Bind decoded intensities and acquisition aliases, then materialize only enrolled references."""
    selected = {row["record_ordinal"]: row for row in items}
    start = re.match(rb"\s*\[\s*\[\s*", raw)
    need(start, "Original nested record list")
    pos = start.end()
    groups, spans = [], {}
    for ordinal in range(1, 1597):
        need(raw[pos:pos+1] == b"{", "Author record")
        pos += 1
        record = {}
        while True:
            key = lex.KEY.match(raw, pos)
            need(key, "Author record field")
            name = key[1].decode("ascii")
            left = key.end()
            right = lex.skip_value(raw, left)
            if name in {"id", "fullname", "sensor", "polarisation"} or (
                    name == "patchfu" and ordinal in selected):
                record[name] = json.loads(raw[left:right])
            elif name == "patchgt" and ordinal in selected:
                need(ordinal not in spans, "Exactly one reference per scored ordinal")
                spans[ordinal] = left, right
            pos = lex.WS.match(raw, right).end()
            separator = raw[pos:pos+1]
            pos += 1
            if separator == b"}":
                break
            need(separator == b",", "Original field separator")
        groups.append(qualifier.acquisition(record))
        if ordinal in selected:
            item = selected[ordinal]
            gray = qualifier.gray_levels(np.asarray(record["patchfu"], dtype="<f8"))
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            need(record["polarisation"] in {"HH", "VV"} and rgb.shape == (101, 101, 3)
                 and digest(rgb.tobytes()) == item["rgb_sha256"], "Original decoded RGB for scored ordinal")
        pos = lex.WS.match(raw, pos).end()
        if ordinal < 1596:
            need(raw[pos:pos+1] == b",", "Author row separator")
            pos = lex.WS.match(raw, pos+1).end()
    need(re.fullmatch(rb"\]\s*\]\s*", raw[pos:]) and set(spans) == set(selected), "Complete original accession")
    aliases = {key: f"acq_{i:03d}" for i, key in enumerate(sorted(set(groups)), 1)}
    ledger = "".join(f"{n}\t{aliases[key]}\n" for n, key in enumerate(groups, 1)).encode("ascii")
    need(set(aliases.values()) == set(GROUPS) and digest(ledger) == GROUP_INDEX and
         all(aliases[groups[n-1]] == item["acquisition_group"] for n, item in selected.items()),
         "Original source-ordinal to acquisition mapping")
    masks = {}
    for ordinal, (left, right) in spans.items():
        target = np.asarray(json.loads(raw[left:right]))
        need(target.shape == (101, 101) and np.isin(target, (0, 1)).all() and target.any(),
             "Original binary nonempty 101-pixel center scatter mask")
        masks[ordinal] = target.astype(bool)
    return masks


def score_panel(items, selected, truth):
    groups = sorted({row["acquisition_group"] for row in items})
    indices = {name: i for i, name in enumerate(groups)}
    totals = np.zeros((len(ARMS), len(groups), len(METRICS)), dtype=np.float64)
    population = Counter()
    counts = {name: Counter(selected=0, missing=0, selected_empty=0, selected_nonempty=0,
                            changed_images=0, added_pixels=0, removed_pixels=0, xor_pixels=0,
                            recovered_050=0, recovered_075=0) for name in ARMS}
    for item, (area, masks, swaps) in zip(items, selected, strict=True):
        group = item["acquisition_group"]
        group_i = indices[group]
        population[group] += 1
        target = truth[item["record_ordinal"]]
        target_area = int(target.sum())
        need(target_area > 0, "Positive reference required")
        for i, name in enumerate(ARMS):
            count = counts[name]
            if area is None:
                count["missing"] += 1
                continue
            count["selected"] += 1
            count["selected_empty" if area == 0 else "selected_nonempty"] += 1
            swaps_n = swaps[name]
            count.update(changed_images=int(swaps_n > 0), added_pixels=swaps_n,
                         removed_pixels=swaps_n, xor_pixels=2 * swaps_n)
            hit = int((masks[name] & target).sum())
            union = area + target_area - hit
            rec50, rec75 = int(2 * hit >= union), int(4 * hit >= 3 * union)
            totals[i, group_i] += (hit / union, 2 * hit / (area + target_area), rec50, rec75)
            count.update(recovered_050=rec50, recovered_075=rec75)
    means = totals / np.array([population[g] for g in groups])[None, :, None]
    seed, n = 254101, 4096
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, len(groups), size=(n, len(groups)))
    arms = {key: dict(counts=dict(counts[key]), acquisition_equal=dict(zip(METRICS, means[i].mean(axis=0).tolist())))
            for i, key in enumerate(ARMS)}
    pairs = {}
    for left, right in PAIRS:
        effect = means[ARMS.index(left)] - means[ARMS.index(right)]
        intervals = np.quantile(effect[draws].mean(axis=1), [.025, .975], axis=0)
        pairs[f"{left}_minus_{right}"] = dict(
            acquisition_equal_delta=dict(zip(METRICS, effect.mean(axis=0).tolist())),
            acquisition_cluster_percentile95=dict(zip(METRICS, intervals.T.tolist())),
            per_acquisition={g: dict(zip(METRICS, v.tolist())) for g, v in zip(groups, effect, strict=True)})
    return dict(images=len(items), acquisitions=len(groups), group_images=dict(sorted(population.items())),
                bootstrap=dict(seed=seed, draws=n, unit="acquisition", generator="PCG64", interval="percentile95"),
                arms=arms, paired=pairs)


def observed_terminal(terminal, role, receipt_hash):
    need(terminal.get("status") == f"SCHWEGMANN_DFINE_FROZEN_{role}_TERMINAL" and
         terminal.get("actual_process_terminal") is True and type(terminal.get("actual_exit")) is int
         and terminal["actual_exit"] == 0 and terminal.get("result_sha256") == receipt_hash and
         type(terminal.get("tool_session_handle")) is int and terminal["tool_session_handle"] > 0,
         "Required original successful process observation for " + role)


def admitted(a):
    manifest = json.loads(read(a.manifest, MANIFEST))
    items = manifest["items"]
    ids = [item["image_id"] for item in items]
    need(ids == manifest["expected_ids"] == sorted(set(ids)) and len(ids) == 750 and
         manifest["source_json_sha256"] == ARCHIVE and manifest["group_index_sha256"] == GROUP_INDEX and
         all(item["image_id"] == item["record_ordinal"] and item["original_hw"] == [101, 101] for item in items)
         and {item["acquisition_group"] for item in items} == set(GROUPS), "Original disjoint positive image roster")
    read(a.protocol, PROTOCOL)
    n = len(items)
    for role, path, signature, receipt in (
            ("CAPTURE", a.capture_terminal, a.capture_terminal_sha256, a.capture_result_sha256),
            ("EDITS", a.edit_terminal, a.edit_terminal_sha256, a.edit_receipt_sha256)):
        observed_terminal(json.loads(read(path, signature)), role, receipt)
    capture = json.loads(read(a.capture_result, a.capture_result_sha256))
    edits = json.loads(read(a.edit_receipt, a.edit_receipt_sha256))
    need(capture["status"] == "SCHWEGMANN_DFINE_FROZEN_CENTER_CAPTURE_COMPLETE" and
         capture["manifest_sha256"] == MANIFEST and capture["protocol_sha256"] == PROTOCOL and
         capture["labels_opened"] is False and capture["optimizer_updates"] == 0 and
         capture["author_E24_sha256"] == EMA and capture["counts"]["images"] == n == capture["model_forwards"],
         "Closed label-free HRSID E24 capture")
    need(edits["status"] == "SCHWEGMANN_DFINE_FROZEN_CENTER_EDITS_COMPLETE" and
         edits["labels_opened"] is False and edits["manifest_sha256"] == MANIFEST and
         edits["images"] == n and edits["capture_result_sha256"] == a.capture_result_sha256 and
         edits["head_acceptance_sha256"] == HEAD and edits["model_forwards"] == edits["optimizer_updates"] == 0,
         "Closed label-free HRSID FIT109 CPU readout")
    captured_raw = read(a.capture_rows, capture["native_rows_sha256"])
    edited_raw = read(a.predictions, edits["edits_sha256"])
    need(len(edited_raw) == edits["edits_bytes"], "Edited stream byte count")
    original = module(a.reference_code / "score_schwegmann.py", HISTORIC_SCORE)
    selected = []
    native_in_top100 = 0
    for item, gpu, cpu in zip(items, jsonl(captured_raw, n), jsonl(edited_raw, n), strict=True):
        native_in_top100 += len(gpu["candidates"])
        selected.append(selected_row(item, gpu, cpu, original.decode_rle))
    areas = [area for area, _, _ in selected if area is not None]
    counts = dict(images=n, selected=len(areas), missing=n-len(areas),
                  nonempty=sum(x > 0 for x in areas), eligible=sum(32 <= x < 1024 for x in areas))
    need(all(capture["counts"].get(k, 0) == v for k, v in counts.items()), "GPU candidate census")
    c = edits["counts"]
    need(all(c.get(k, 0) == v for k, v in counts.items() if k != "images") and
         c.get("area_eligible", 0) == counts["eligible"] == c.get("supported", 0) + c.get("zero_support", 0),
         "CPU candidate and point enrollment census")
    for name in ARMS[1:]:
        per_image = [swaps[name] for _, _, swaps in selected]
        need(c.get(name + "_swaps", 0) == sum(per_image) and
             c.get(name + "_edited", 0) == sum(x > 0 for x in per_image), "CPU swaps and edited-image census")
    return items, selected, capture, edits, dict(**counts, retained_top100_rows=native_in_top100,
        supported=c.get("supported", 0), zero_support=c.get("zero_support", 0), point_pairs=c.get("pairs", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("manifest", "protocol", "capture-terminal", "capture-result", "capture-rows",
                 "edit-terminal", "edit-receipt", "predictions", "archive", "reference-code",
                 "intake-source", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("capture-terminal", "capture-result", "edit-terminal", "edit-receipt"):
        parser.add_argument("--" + name + "-sha256", required=True)
    a = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode, "Use isolated Python -I -B")
    need(a.output.is_absolute() and a.output.parent.is_dir() and not a.output.exists(), "Fresh absolute score path")
    items, selected, capture, edits, census = admitted(a)
    lex = module(a.intake_source, INTAKE)
    qualifier = module(a.reference_code / "qualify_schwegmann.py", QUALIFIER)
    # First access to the mixed source occurs after both prelabel terminals, receipts, and all arm rows.
    truth = author_truth(read(a.archive, ARCHIVE), items, lex, qualifier)
    report = score_panel(items, selected, truth)
    report.update(status="SCHWEGMANN_DFINE_FROZEN_CENTER_SCORE_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE",
                  scope="Auxiliary descriptive center scatter-support mask scores; image-disjoint within 16 previously used acquisitions",
                  protocol_sha256=PROTOCOL, manifest_sha256=MANIFEST, source_json_sha256=ARCHIVE,
                  group_index_sha256=GROUP_INDEX, score_source_sha256=digest(Path(__file__).read_bytes()),
                  capture_terminal_sha256=a.capture_terminal_sha256, capture_result_sha256=a.capture_result_sha256,
                  capture_rows_sha256=capture["native_rows_sha256"], edit_terminal_sha256=a.edit_terminal_sha256,
                  edit_receipt_sha256=a.edit_receipt_sha256, predictions_sha256=edits["edits_sha256"],
                  author_E24_sha256=EMA, fit109_head_acceptance_sha256=HEAD, candidate_counts=census, labels_opened=True,
                  source_reference_records=len(truth), model_forwards=0, optimizer_updates=0, instance_AP=False)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with a.output.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(status=report["status"], result_sha256=digest(raw), images=len(items))))


if __name__ == "__main__":
    main()
