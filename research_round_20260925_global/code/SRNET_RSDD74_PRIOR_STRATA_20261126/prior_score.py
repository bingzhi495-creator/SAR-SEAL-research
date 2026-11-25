"""Score frozen prior/residual readouts with the exact archived COCO and geometry.""" 
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


SCORE_SHA = "8f26062eed2f56501f1a50a656961535b9fff3aa7ed493b7d4fe758f44b1f77a"
SCORE_TERMINAL = "799047e9123044aff40ef1bead0698ba6a7d103aa744d9d4934d766d562012e9"
SCORE_AUTHORITY = "7121fd1b7c1c98ee87ce6cf23963c09fabc437af458f57fffa43fb99a75c6339"
SCORE_WORKER = "6ba17a0bcaa897f6f56bca354c22b2db57b79e38b18f60f50cc4ae3498613889"
STATUS = "SRNET_RSDD74_DIRECT_PRIOR_ABLATION_SCORE_COMPLETE"
N, F, I, P, R = ARMS = ("raw", "fixed", "free_matched_count", "prior_only", "residual_only")
ORIGINAL_PAIRS = ((I, F), (I, N), (F, N))
NEW_PAIRS = ((I, P), (R, I))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact digest: " + path.name)
    return raw


def worker(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def load_original(previous):
    historic = json.loads(pin(previous / "score/RESULT_SAFE.json", SCORE_SHA))
    terminal = json.loads(pin(previous / "observed_score/TERMINAL.json", SCORE_TERMINAL))
    require(terminal["phase"] == "score" and terminal["result_sha256"] == SCORE_SHA
            and terminal["worker_sha256"] == historic["worker_sha256"] == SCORE_WORKER
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and historic["status"] == "SRNET_SSDD_E24_RSDD74_PAIRED_SCORE_COMPLETE"
            and historic["authority_sha256"] == SCORE_AUTHORITY and historic["author_test_opened"] is False,
            "Original naturally closed score")
    old = worker(previous / "score.py", SCORE_WORKER)
    raw = json.loads(pin(previous / "AUTHORITY_score.json", SCORE_AUTHORITY))
    a = SimpleNamespace(**{k: Path(v) if isinstance(v, str) and Path(v).is_absolute() else v for k, v in raw.items()})
    require(a.prepare.resolve() == previous.resolve() and a.protocol_sha == old.PROTOCOL_SHA
            and a.verify_result_sha == historic["independent_edits_result_sha256"]
            and a.verify_terminal_sha == historic["independent_edits_terminal_sha256"]
            and a.capture_result_sha == historic["capture_result_sha256"], "Original archive lineage")
    pin(a.protocol, a.protocol_sha)
    replay = old.load(a.prepare / "replay.py", a.replay_worker_sha)
    old.audit_source(a, replay)
    edit, blocks = old.read_closed_stream(a, replay)
    require(edit["edits_sha256"] == historic["edits_sha256"] and len(blocks) == 74
            and sum(len(r["stock"]) for r in blocks.values()) == 89, "Original exact cohort")
    return a, old, replay, edit, blocks, historic


def add_write_arms(blocks, rendered, codec):
    require(len(blocks) == len(rendered), "Same image roster")
    for (image, block), row in zip(blocks.items(), rendered, strict=True):
        changes = {int(k): v for k, v in row["changes"].items()}
        require(row["image_id"] == image and changes.keys() == block["changes"].keys(),
                "Exact original image order and positive quota roster")
        for rank, options in changes.items():
            existing = block["changes"][rank]
            native = block["stock"][rank - 1]["segmentation"]
            require(set(options) == {P, R} and all(
                opt["swaps"] == existing[F]["swaps"] and opt["rle"]["size"] == native["size"]
                and int(codec.area(opt["rle"])) == int(codec.area(native)) for opt in options.values()),
                "Original signed quota, native canvas and area for both ablations")
            existing.update(options)


def same_tree(new, old):
    if isinstance(old, dict):
        return all(k in new and same_tree(new[k], v) for k, v in old.items())
    if isinstance(old, list):
        return len(new) == len(old) and all(same_tree(x, y) for x, y in zip(new, old, strict=True))
    if isinstance(old, float):
        return abs(new - old) <= 1e-12
    return new == old


def score_view(old, reserve, evaluator, core, geometry, gt, ids, blocks, polygons, dates):
    anchors = {}
    score_one = evaluator.score_one

    def remember(*args, **kwargs):
        scores, matches = score_one(*args, **kwargs)
        anchors.update(matches)
        return scores, matches

    evaluator.score_one = remember
    try:
        view = old.score_view(reserve, evaluator, core, geometry, gt, ids, blocks, polygons, dates)
    finally:
        evaluator.score_one = score_one
    view.pop("criteria")
    view.pop("within_recipe_mechanism_criterion")
    exposure = {}
    for new, comparator in NEW_PAIRS:
        groups = set()
        changed = 0
        for label_id, (i, rank, native) in anchors.items():
            poly = polygons[label_id]
            area = int(core.codec.area(native["segmentation"]))
            if not poly.is_valid or poly.area <= 0 or not 32 <= area < 1024:
                continue
            options = blocks[i]["changes"].get(rank)
            if options and options[new]["rle"] != options[comparator]["rle"]:
                changed += 1
                groups.add(blocks[i]["group"])
        exposure[new + " minus " + comparator] = dict(anchors=changed, parents=len(groups),
                                                       sensor_date_blocks=len({dates[g] for g in groups}))
    view["changed_anchors"] = exposure
    return view


def run(cmd):
    started = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11" and not cmd.output.exists(),
            "Fresh qualified CPU score")
    pin(cmd.protocol, cmd.protocol_sha256)
    a, old, replay, edit, blocks, historic = load_original(cmd.previous)
    render = replay.closed(cmd.render_terminal, cmd.render_terminal_sha256, cmd.render / "RESULT_SAFE.json",
                           cmd.render_result_sha256, "prepare-edits", cmd.render_worker_sha256)
    require(render["status"] == "SRNET_RSDD74_DIRECT_PRIOR_ABLATION_RENDER_COMPLETE"
            and render["protocol_sha256"] == cmd.protocol_sha256 and render["prelock_commit"] == cmd.prelock_commit
            and render["original_edit_stream_sha256"] == edit["edits_sha256"]
            and render["fit_weights_sha256"] == replay.WEIGHTS_SHA
            and render["original_fixed_and_full_edit_bytes_exact"] is True
            and render["original_signed_k_and_support_exact"] is True
            and render["label_files_opened"] == render["detector_forwards"] == render["optimizer_updates"] == 0
            and render["counts"]["images"] == 74 and render["counts"]["candidates"] == 89
            and render["counts"]["positive_candidates"] == edit["counts"]["positive_matched_budget_candidates"]
            and render["counts"]["exchanges"] == edit["counts"]["exact_adds_per_policy"], "Closed label-free readout")
    verification = replay.closed(cmd.verify_terminal, cmd.verify_terminal_sha256, cmd.verify / "RESULT_SAFE.json",
                                 cmd.verify_result_sha256, "verify-edits", cmd.render_worker_sha256)
    readback_fields = {"status", "worker_sha256", "protocol_sha256", "prelock_commit",
                       "original_prepare_sha256", "original_verify_sha256", "original_edit_stream_sha256",
                       "fit_weights_sha256", "original_fixed_and_full_edit_bytes_exact",
                       "original_signed_k_and_support_exact", "label_files_opened", "detector_forwards",
                       "optimizer_updates", "seed", "edits_sha256", "counts"}
    require({key: render[key] for key in readback_fields} == {key: verification[key] for key in readback_fields},
            "Independent label-free edit readback reproduced all hash and count fields")
    raw = pin(cmd.render / "EDITS_PRIVATE.jsonl", render["edits_sha256"])
    require(raw == pin(cmd.verify / "EDITS_PRIVATE.jsonl", verification["edits_sha256"]),
            "Independent label-free readback reproduced all edit bytes")
    add_write_arms(blocks, [json.loads(row) for row in raw.splitlines()], replay.codec)

    reserve = old.load(a.code / old.RESERVE, old.RESERVE_SHA)
    reference = old.load(a.code / old.HISTORIC, reserve.HISTORIC)
    evaluator = reference.module(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", reference.SCORE_SOURCE)
    old.support_empty_predictions(evaluator)
    adapter = reference.module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", reference.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    sys.path.append(str(a.shapely_site))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
            "Exact archived continuous geometry")
    census = json.loads(pin(a.census, reserve.CENSUS))
    dates = {p["paper_scene"]: (p["sensor"], p["calendar_date"]) for p in census["per_parent"]}
    parents = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    nested = {i for i, row in blocks.items() if row["group"] in parents}
    require(len(nested) == 42 and set(dates) == {r["group"] for r in blocks.values()},
            "Locked whole74 and nested42")
    print(json.dumps(dict(stage="CLOSED_ABLATION_PRELABEL_PARITY")), flush=True)
    gt, rekeyed = reference.coco_annotation_ids(reserve.select_labels(pin(a.author_train, reserve.TRAIN), blocks))
    require(rekeyed == historic["annotation_ids_rekeyed"], "Original canonical author annotation IDs")
    polygons = {v["id"]: shapely.Polygon(np.asarray(v["segmentation"][0]).reshape(-1, 2)) for v in gt["annotations"]}
    reserve.ARMS, reserve.PAIRS = ARMS, ORIGINAL_PAIRS + NEW_PAIRS
    views = {}
    for name, ids in (("full74", set(blocks)), ("nested42_sensor_date", nested)):
        view = score_view(old, reserve, evaluator, core, shapely, gt, ids, blocks, polygons, dates)
        original = historic["views"][name]
        require({arm: view["metrics_fraction"][arm] for arm in (N, F, I)} == original["metrics_fraction"]
                and same_tree(view, original), "Exact prior COCO and continuous N/F/I readback")
        view["directional_diagnostic"] = {}
        for left, right in NEW_PAIRS:
            key = left + " minus " + right
            ap = view["paired_contrasts_points"][key]["AP"]
            ci = view["geometry"]["paired_contrasts_points"][key][reserve.FIELDS[0]]["sensor_date_interval_95"]
            counts = view["changed_anchors"][key]
            coverage = counts["anchors"] >= 20 and counts["parents"] >= 8 and counts["sensor_date_blocks"] >= 8
            signed = "favourable" if ap >= .30 and ci[0] > 0 else "adverse" if ap <= -.30 and ci[1] < 0 else "inconclusive"
            view["directional_diagnostic"][key] = dict(coverage_met=coverage, result=signed if coverage else "inconclusive")
        views[name] = view
    result = dict(status=STATUS, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=cmd.protocol_sha256, prelock_commit=cmd.prelock_commit,
                  original_score_sha256=SCORE_SHA, original_score_terminal_sha256=SCORE_TERMINAL,
                  render_result_sha256=cmd.render_result_sha256, render_terminal_sha256=cmd.render_terminal_sha256,
                  verification_result_sha256=cmd.verify_result_sha256,
                  verification_terminal_sha256=cmd.verify_terminal_sha256, independent_edit_stream_readback_exact=True,
                  original_coco_all_metrics_exact=True, original_continuous_geometry_tolerance=1e-12,
                  historical_author_train_reserve=True, author_test_opened=False, detector_forwards=0,
                  optimizer_updates=0, seed=5217, counts=render["counts"], views=views,
                  elapsed_seconds=perf_counter()-started)
    cmd.output.mkdir()
    payload = replay.encoded(result)
    (cmd.output / "RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=STATUS, result_sha256=replay.digest(payload))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("previous", "protocol", "render", "render-terminal", "verify", "verify-terminal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol-sha256", "prelock-commit", "render-terminal-sha256", "render-result-sha256",
                 "render-worker-sha256", "verify-terminal-sha256", "verify-result-sha256"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

