"""Score common pixel-center support and RoI padding against seven archived policies."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


EXPANDED_SCORER = "6e2563d7d857bdd36f932756b53146e243cfdfa3eada32545d7c05d51010d454"
EXPANDED_AUTHORITY = "0405efe9947fc41b269db64366b67fd8f2557a0907d2bad61437a9e6de59e3e4"
EXPANDED_SCORE = "5c6bb6ce97c69964b21ce0b11f39e0fa38911857f9e5d8f4ffba5ae689d71547"
ACCEPTED = "INSTANCE_RSDD_CENTER_RANKING_ACCEPTED_V1"
STATUS = "INSTANCE_RSDD_CENTER_RANKING_SCORE_COMPLETE_V1"
BC, LC, BCB = "point_center", "learned_center", "point_center_border"
NEW = (BC, LC, BCB)
COUNT_KEYS = ("positive_original_k", "positive_center_k", "center_swaps", "original_swaps", "clamped_candidates")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, f"Exact bytes required for {path.name}")
    return data


def load_module(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def join_center(blocks, rows, receipt, expanded, core):
    require(len(rows) == len(blocks) == 740, "Original author roster")
    by_sensor = defaultdict(Counter)
    for (image, frame), row in zip(blocks.items(), rows, strict=True):
        require(set(row) == {"image_id", "changes"} and row["image_id"] == image, "Ordered center roster")
        expected = {str(i) for i, arms in frame["changes"].items() if arms.get(expanded.LE, {}).get("swaps", 0) > 0}
        require(set(row["changes"]) == expected, "Center policies must cover every original positive-budget candidate")
        c = by_sensor[frame["sensor"]]
        for key, edits in row["changes"].items():
            i = int(key)
            historic = frame["changes"][i]
            k = historic[expanded.LE]["swaps"]
            stock = frame["stock"][i-1]["segmentation"]
            require(set(edits) == set(NEW), "Complete three-policy ledger")
            joint = edits[BC]["swaps"]
            require(type(joint) is int and 0 <= joint <= k == historic[expanded.BE]["swaps"], "Joint center budget")
            area = int(core.codec.area(stock))
            for arm, edit in edits.items():
                require(set(edit) == {"swaps", "rle"} and type(edit["swaps"]) is int and edit["swaps"] == joint,
                        "Same integer signed budget for all three policies")
                after = core.checked_rle(edit["rle"], 512, 512)
                overlap = int(core.codec.area(core.codec.merge([after, stock], intersect=True)))
                require(int(core.codec.area(after)) == area and area-overlap == joint and (joint or after == stock),
                        "Native area and both signed counts; exact original RLE for a zero budget")
                historic[arm] = edit
            c.update(positive_original_k=1, positive_center_k=int(joint > 0), center_swaps=joint,
                     original_swaps=k, clamped_candidates=int(joint < k))
    counts = {"full": {key:sum(c[key] for c in by_sensor.values()) for key in COUNT_KEYS},
              "by_sensor": {s:{key:c[key] for key in COUNT_KEYS} for s,c in by_sensor.items()}}
    require(set(by_sensor) == set(receipt["counts_by_sensor"]) == {"GF-3", "TerraSAR-X"}, "Archived sensors")
    for measured, replayed in [(counts["full"], receipt["counts"]),
                               *((v, receipt["counts_by_sensor"][s]) for s,v in counts["by_sensor"].items())]:
        require(all(type(replayed[k]) is int and replayed[k] == v for k,v in measured.items()),
                "Reproduce independently verified center budgets and clamp counts")
    return counts


def admit_center(inputs, old, expanded, blocks, previous, core):
    root = inputs.center_replay
    old.terminal(root/"observed_verify-edits/TERMINAL.json", inputs.center_terminal_sha,
                 inputs.center_receipt_sha, "verify-edits")
    receipt = json.loads(pin(root/"verify-edits/RESULT_SAFE.json", inputs.center_receipt_sha))
    require(receipt["status"] == ACCEPTED and receipt["images"] == 740 and receipt["GT_opened"] is False
            and receipt["exact_replayed_policy_choices"] is True and receipt["exact_support_xor_area"] is True
            and receipt["protocol_sha256"] == inputs.protocol_sha and receipt["prelock_commit"] == inputs.prelock_commit
            and receipt["expanded_replay_receipt_sha256"] == inputs.expanded_receipt_sha
            and all(receipt[k] == previous[k] for k in ("predictions_sha256", "candidate_audit_sha256")),
            "Independent center replay must anchor to the same expanded replay, frozen capture and protocol")
    rows = old.records(root/"prepare-edits/EDITS_PRIVATE.jsonl", receipt["edits_sha256"])
    return join_center(blocks, rows, receipt, expanded, core), receipt


def configure_views(expanded):
    """Reuse the private module's COCO/geometry routines with the extended policy list."""
    previous_contrasts = expanded.contrasts

    def contrasts(v):
        return {**previous_contrasts(v),
                "L_C minus B_C": v[LC] - v[BC], "L_C minus B_CB": v[LC] - v[BCB],
                "B_CB minus B_C": v[BCB] - v[BC], "L_C minus L_E": v[LC] - v[expanded.LE],
                "B_C minus B_E": v[BC] - v[expanded.BE], "L_C minus P": v[LC] - v[expanded.P],
                "B_C minus P": v[BC] - v[expanded.P], "B_CB minus P": v[BCB] - v[expanded.P]}

    expanded.OLD = expanded.ARMS
    expanded.ARMS += NEW
    expanded.contrasts = contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    cmd = parser.parse_args()
    started = perf_counter()
    paths = {"expanded_authority", "expanded_score", "protocol", "center_replay"}
    hashes = {"protocol_sha", "center_terminal_sha", "center_receipt_sha"}
    values = json.loads(pin(cmd.authority, cmd.authority_sha))
    require(set(values) == paths | hashes | {"prelock_commit"} and all(Path(values[k]).is_absolute() for k in paths)
            and all(re.fullmatch("[0-9a-f]{64}", values[k]) for k in hashes)
            and re.fullmatch("[0-9a-f]{40}", values["prelock_commit"]) and not cmd.output.exists()
            and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11", "Qualified fresh CPU scorer")
    inputs = SimpleNamespace(**{k:Path(v) if k in paths else v for k,v in values.items()})
    pin(inputs.protocol, inputs.protocol_sha)
    previous = json.loads(pin(inputs.expanded_score, EXPANDED_SCORE))
    authority = json.loads(pin(inputs.expanded_authority, EXPANDED_AUTHORITY))
    require(previous["authority_sha256"] == EXPANDED_AUTHORITY and previous["worker_sha256"] == EXPANDED_SCORER
            and previous["independent_readback_sha256"] == authority["expanded_receipt_sha"], "Archived expanded score")
    ref = json.loads(pin(Path(authority["reference_authority"]), authority["reference_authority_sha"]))
    legacy = SimpleNamespace(**{k:v if k.endswith("_sha") else Path(v) for k,v in ref.items()})
    expanded = load_module(legacy.code/"INSTANCE_RSDD_EXPANDED_RANKING_20261123/score_expanded.py", EXPANDED_SCORER)
    transfer = load_module(legacy.code/"INSTANCE_RSDD_POINTREND_TRANSFER_20261122/score_transfer.py", expanded.TRANSFER_SCORER)
    original = json.loads(pin(Path(authority["original_score"]), expanded.ORIGINAL_SCORE))
    pin(Path(authority["protocol"]), authority["protocol_sha"])
    pin(legacy.protocol, transfer.PROTOCOL)
    sys.path.append(str(legacy.shapely_site))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Archived geometry runtime")
    old = transfer.predecessor(legacy.code)
    evaluator = old.module(legacy.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(legacy.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(legacy.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    expanded_inputs = SimpleNamespace(**vars(legacy), expanded_replay=Path(authority["expanded_replay"]),
                                      expanded_terminal_sha=authority["expanded_terminal_sha"],
                                      expanded_receipt_sha=authority["expanded_receipt_sha"],
                                      protocol_sha=authority["protocol_sha"], prelock_commit=authority["prelock_commit"])
    blocks, historic, previous_receipt = expanded.admit(expanded_inputs, transfer, old, adapter, core, original)
    require(historic == previous["label_free_realized_counts"], "Reproduce archived seven-arm census")
    inputs.expanded_receipt_sha = authority["expanded_receipt_sha"]
    counts, receipt = admit_center(inputs, old, expanded, blocks, previous_receipt, core)

    # Only the closed and terminal-verified image-only ledger can reach the first annotation read.
    gt = {"info": {}, **json.loads(pin(legacy.author_test, old.AUTHOR_TEST))}
    require(len(gt["images"]) == 740 and len(gt["annotations"]) == 872 and set(blocks) == {r["id"] for r in gt["images"]}
            and all(r["width"] == r["height"] == 512 and r["file_name"] == blocks[r["id"]]["file"] for r in gt["images"])
            and all(r["category_id"] == 1 and len(r["segmentation"]) == 1 for r in gt["annotations"]), "Author population")
    gt, rekeyed = old.coco_annotation_ids(gt)
    polygons = {r["id"]:shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1, 2)) for r in gt["annotations"]}
    require(sum(not p.is_valid for p in polygons.values()) == 1, "Original polygon treatment")
    full = set(blocks)
    gf3 = {i for i,frame in blocks.items() if frame["sensor"] == "GF-3"}
    selections = dict(gf3=gf3, full=full, terrasar_x=full-gf3, gf3_deduplicated=old.representatives(blocks, gf3),
                      full_deduplicated=old.representatives(blocks, full))
    configure_views(expanded)
    views, evaluated = {}, {}
    for name, ids in selections.items():
        reference = previous["views"][name]
        key = frozenset(ids)
        if key not in evaluated:
            evaluated[key] = expanded.score_view(gt, ids, blocks, reference, evaluator, core, polygons, shapely)
        view = views[name] = evaluated[key]
        require(all(view[k] == reference[k] for k in ("images", "annotations"))
                and all(view["metrics_fraction"][a] == reference["metrics_fraction"][a] for a in expanded.OLD)
                and all(view["geometry"]["paired_contrasts_points"][c] == v
                        for c,v in reference["geometry"]["paired_contrasts_points"].items()),
                "Reproduce all historical COCO metrics, cohort sizes and paired geometry estimates")
    primary = ("L_C minus B_C", "L_C minus B_CB")
    main = views["gf3"]
    decisions = {contrast: bool(main["metrics_contrasts_points"]["AP"][contrast] > 0
        and main["geometry"]["paired_contrasts_points"][contrast][expanded.FIELDS[0]]["interval_95"][0] > 0)
        for contrast in primary}
    result = dict(status=STATUS, protocol_sha256=inputs.protocol_sha, prelock_commit=inputs.prelock_commit,
                  authority_sha256=cmd.authority_sha, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  expanded_score_sha256=EXPANDED_SCORE, expanded_replay_receipt_sha256=inputs.expanded_receipt_sha,
                  independent_readback_sha256=inputs.center_receipt_sha, center_replay_terminal_sha256=inputs.center_terminal_sha,
                  native_prediction_sha256=receipt["predictions_sha256"], edits_sha256=receipt["edits_sha256"],
                  author_test_sha256=old.AUTHOR_TEST, evaluator_annotation_ids_rekeyed=rekeyed, previously_evaluated=True,
                  detector_forwards=0, optimizer_updates=0, label_free_realized_counts=counts, views=views,
                  within_recipe_center_ranking_criterion=all(decisions.values()),
                  primary_robustness_contrasts=decisions, elapsed_seconds=perf_counter()-started)
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    cmd.output.mkdir()
    (cmd.output/"RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
