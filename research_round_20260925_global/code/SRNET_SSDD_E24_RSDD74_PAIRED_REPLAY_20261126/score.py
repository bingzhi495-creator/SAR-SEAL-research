"""Apply the original author reserve scoring to the closed SRNet edit streams."""
import argparse
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


RESERVE = "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py"
RESERVE_SHA = "5ff671b2b83bb12e322b0c766897bf3cdacaedd233bf0d6edcdfcccf2c938df4"
HISTORIC = "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py"
STATUS = "SRNET_SSDD_E24_RSDD74_PAIRED_SCORE_COMPLETE"
PROTOCOL_SHA = "8cc6abb79faea6af12fa7d22aeec18d43e277323092dd5cdf91aba4c8d744826"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact digest: " + path.name)
    return raw


def load(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def audit_source(a, replay):
    source = replay.closed(a.source_terminal, a.source_terminal_sha, a.source_result,
                           a.source_result_sha, a.source_phase, a.source_worker_sha)
    require(source["status"] == "SRNET_FIT741_RSDD74_DECODED_RGB_EXPOSURE_COMPLETE"
            and source["source_images"] == source["source_distinct_consumed_ids"] == 741
            and source["target_images"] == source["target_intake_native_rgb_parity"] == 74
            and source["target_images_matching"] == 0 and source["no_exact_decoded_rgb_overlap"] is True
            and source["target_image_manifest_sha256"] == replay.MANIFEST_SHA
            and source["source_checkpoint_sha256"] == replay.SRNET_WEIGHTS_SHA
            and source["target_polygon_rows_opened"] == source["training_label_rows_opened"]
            == source["prediction_rows_opened"] == source["model_forwards"] == source["optimizer_updates"] == 0,
            "Closed whole FIT741/reserve74 decoded RGB exact screen")
    hrsid = replay.closed(a.hrsid_terminal, a.hrsid_terminal_sha, a.hrsid_result, a.hrsid_result_sha, "verify-capture")
    require(hrsid["status"] == "INSTANCE_RSDD_RESERVE74_SOURCE_IDENTITIES_VERIFIED_V1"
            and hrsid["passed"] is True and hrsid["reserve_manifest_sha256"] == replay.MANIFEST_SHA
            and hrsid["reserve_rgb_matches_hrsid_fit"] == hrsid["reserve_rgb_matches_historical_test"] == 0
            and hrsid["head_weight_bundle_sha256"] == replay.WEIGHTS_SHA, "Closed original source and RF head screen")


def blocks_from_stream(replay, stream, ledger):
    blocks = {}
    for row, curve in zip(stream, ledger, strict=True):
        image = row["image_id"]
        stock = replay.native_rows(row)
        ranks = list(range(1, len(stock) + 1))
        require(curve["image_id"] == image and image not in blocks, "Original image order")
        changes = {int(key): value for key, value in curve["changes"].items()}
        require(set(changes) <= set(ranks), "All and only original SRNet candidate keys")
        for rank, options in changes.items():
            require(set(options) == {replay.F, replay.I}
                    and options[replay.F]["swaps"] == options[replay.I]["swaps"] > 0,
                    "Frozen policies use identical positive counts and common sign support")
            s = stock[rank - 1]["segmentation"]
            shape = s["size"]
            area = int(replay.codec.area(s))
            require(32 <= area < 1024 and all(
                opt["rle"]["size"] == shape and int(replay.codec.area(opt["rle"])) == area
                for opt in options.values()), "Unchanged native canvas and area")
        blocks[image] = dict(stock=stock, ranks=ranks, file=row["source_image_name"], rgb=row["rgb_sha256"],
                             sensor=row["sensor"], group=row["acquisition_group"], changes=changes,
                             hw=row["original_hw"])
    return blocks


def read_closed_stream(a, replay):
    capture, index = replay.checked_capture(a)
    prepare = replay.closed(a.prepare / "observed_prepare-edits/TERMINAL.json", a.prepare_terminal_sha,
                            a.prepare / "prepare-edits/RESULT_SAFE.json", a.prepare_result_sha,
                            "prepare-edits", a.replay_worker_sha)
    verify = replay.closed(a.prepare / "observed_verify-edits/TERMINAL.json", a.verify_terminal_sha,
                           a.prepare / "verify-edits/RESULT_SAFE.json", a.verify_result_sha,
                           "verify-edits", a.replay_worker_sha)
    require(prepare["status"] == replay.PREPARED and verify["status"] == replay.VERIFIED
            and verify["exact_independent_replay"] is True
            and verify["prepare_result_sha256"] == a.prepare_result_sha
            and verify["prepare_terminal_sha256"] == a.prepare_terminal_sha,
            "Independent point edit replay closure")
    for proof in (prepare, verify):
        require(proof["protocol_sha256"] == a.protocol_sha and proof["prelock_commit"] == a.prelock_commit
                and proof["capture_result_sha256"] == a.capture_result_sha
                and proof["capture_terminal_sha256"] == a.capture_terminal_sha
                and proof["capture_index_sha256"] == capture["index_sha256"]
                and proof["manifest_sha256"] == replay.MANIFEST_SHA
                and proof["edits_sha256"] == prepare["edits_sha256"] and proof["counts"] == prepare["counts"]
                and proof["label_files_opened"] == proof["detector_forwards"] == proof["optimizer_updates"] == 0
                and proof["all_stock_threshold_exact"] is True and proof["same_per_candidate_signed_budget"] is True
                and proof["fit_weights_sha256"] == replay.WEIGHTS_SHA, "Matching frozen CPU edit authority")
    ledger = [json.loads(line) for line in pin(
        a.prepare / "prepare-edits/EDITS_PRIVATE.jsonl", verify["edits_sha256"]).splitlines()]
    blocks = blocks_from_stream(replay, index, ledger)
    require(len(blocks) == 74 and sum(len(r["stock"]) for r in blocks.values()) == verify["counts"]["native_candidates"],
            "Whole frozen candidate and image count")
    return verify, blocks


def score_view(reserve, evaluator, core, shapely, gt, ids, blocks, polygons, dates):
    native_anchors = {}
    untouched_score_one = evaluator.score_one

    def remember(*args, **kwargs):
        metrics, matches = untouched_score_one(*args, **kwargs)
        native_anchors.update(matches)
        return metrics, matches

    scorer = SimpleNamespace(score_one=remember)
    view = reserve.score_view(gt, ids, blocks, polygons, dates, scorer, core, shapely)
    changed, groups = 0, set()
    for gid, (image, rank, native) in native_anchors.items():
        poly = polygons[gid]
        area = int(core.codec.area(native["segmentation"]))
        if not poly.is_valid or poly.area <= 0 or not 32 <= area < 1024:
            continue
        options = blocks[image]["changes"].get(rank)
        if options and options[reserve.FIXED]["rle"] != options[reserve.MATCHED]["rle"]:
            changed += 1
            groups.add(blocks[image]["group"])
    original_counts = view["geometry"].pop("counts_by_parent")
    if original_counts:
        keys = next(iter(original_counts.values()))
        view["geometry"]["counts"] = {k: view["geometry"]["counts"].get(k, 0) for k in keys}
    view["geometry"]["actual_changed_eligible_native_matches"] = changed
    view["geometry"]["actual_changed_parent_groups"] = len(groups)
    view["geometry"]["actual_changed_sensor_date_blocks"] = len({dates[g] for g in groups})
    return view


def support_empty_predictions(evaluator):
    """COCOeval supports empty detections, but COCO.loadRes needs an existing row."""
    original = evaluator.COCO

    class EmptyResCOCO(original):
        def loadRes(self, rows):
            if rows:
                return super().loadRes(rows)
            result = original()
            result.dataset = {key: self.dataset[key] for key in ("images", "categories")}
            result.dataset["annotations"] = []
            result.createIndex()
            return result

    evaluator.COCO = EmptyResCOCO


def run(a, out, authority_sha):
    started = perf_counter()
    replay = load(a.prepare / "replay.py", a.replay_worker_sha)
    # Score author labels only after all lineage, native, and label-free replay checks have closed.
    audit_source(a, replay)
    edit, blocks = read_closed_stream(a, replay)
    reserve = load(a.code / RESERVE, RESERVE_SHA)
    old = load(a.code / HISTORIC, reserve.HISTORIC)
    evaluator = old.module(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    support_empty_predictions(evaluator)
    adapter = old.module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    sys.path.append(str(a.shapely_site))
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Archived geometry runtime")
    reserve.ARMS = (reserve.RAW, reserve.FIXED, reserve.MATCHED)
    reserve.PAIRS = ((reserve.MATCHED, reserve.FIXED), (reserve.MATCHED, reserve.RAW), (reserve.FIXED, reserve.RAW))
    census = json.loads(pin(a.census, reserve.CENSUS))
    dates = {p["paper_scene"]: (p["sensor"], p["calendar_date"]) for p in census["per_parent"]}
    parents = {row["group"] for row in blocks.values()}
    nested_parents = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    nested = {i for i, row in blocks.items() if row["group"] in nested_parents}
    require(len(nested) == 42 and set(dates) == parents and len(parents) == 13, "Original nested and whole group census")
    gt, rekeyed = old.coco_annotation_ids(reserve.select_labels(pin(a.author_train, reserve.TRAIN), blocks))
    polygons = {v["id"]: shapely.Polygon(np.asarray(v["segmentation"][0]).reshape(-1, 2)) for v in gt["annotations"]}
    views = {}
    for name, ids in (("full74", set(blocks)), ("nested42_sensor_date", nested)):
        view = score_view(reserve, evaluator, core, shapely, gt, ids, blocks, polygons, dates)
        view.pop("criteria")
        view.pop("within_recipe_mechanism_criterion")
        views[name] = view

    full = views["full74"]
    ap = full["paired_contrasts_points"][reserve.PRIMARY]["AP"]
    g = full["geometry"]
    iou = g["paired_contrasts_points"].get(reserve.PRIMARY, {}).get(reserve.FIELDS[0])
    interval = iou["sensor_date_interval_95"] if iou else [None, None]
    positive = ap is not None and ap >= .30 and interval[0] is not None and interval[0] > 0
    negative = ap is not None and ap <= -.30 and interval[1] is not None and interval[1] < 0
    coverage = (g["actual_changed_eligible_native_matches"] >= 20 and
                g["actual_changed_parent_groups"] >= 8 and g["actual_changed_sensor_date_blocks"] >= 8)
    decision = dict(full74_only=True, paired_changed_anchors_20_parents_8_sensor_date_8=coverage,
                    favourable_same_direction_effect=positive, adverse_same_direction_effect=negative,
                    result="favourable" if positive and coverage else "adverse" if negative and coverage else "inconclusive")
    result = dict(status=STATUS, protocol_sha256=a.protocol_sha, prelock_commit=a.prelock_commit,
                  authority_sha256=authority_sha, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  capture_result_sha256=a.capture_result_sha, source_result_sha256=a.source_result_sha,
                  source_terminal_sha256=a.source_terminal_sha, hrsid_source_result_sha256=a.hrsid_result_sha,
                  independent_edits_result_sha256=a.verify_result_sha, independent_edits_terminal_sha256=a.verify_terminal_sha,
                  edits_sha256=edit["edits_sha256"], counts=edit["counts"], fit_weights_sha256=replay.WEIGHTS_SHA,
                  census_sha256=reserve.CENSUS, author_train_sha256=reserve.TRAIN, author_test_opened=False,
                  geographic_acquisition_disjointness_verified=False, detector_forwards=0, optimizer_updates=0,
                  native_anchor_iou=.50, native_anchor_area_min=32, native_anchor_area_max_exclusive=1024,
                  bootstrap_draws=4096, bootstrap_seed=5217, seed=5217, views=views, primary_decision=decision, annotation_ids_rekeyed=rekeyed,
                  elapsed_seconds=perf_counter()-started)
    payload = replay.encoded(result)
    out.mkdir()
    (out / "RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=STATUS, result_sha256=replay.digest(payload))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    cmd = parser.parse_args()
    raw = json.loads(pin(cmd.authority, cmd.authority_sha))
    keys = {"prelock_commit", "source_phase"}
    paths = {"code", "protocol", "capture", "manifest", "prepare", "source_result", "source_terminal",
             "hrsid_result", "hrsid_terminal", "census", "author_train", "shapely_site"}
    hashes = {"protocol_sha", "capture_terminal_sha", "capture_result_sha", "capture_worker_sha",
              "prepare_terminal_sha", "prepare_result_sha", "verify_terminal_sha", "verify_result_sha",
              "replay_worker_sha", "source_terminal_sha", "source_result_sha", "source_worker_sha",
              "hrsid_terminal_sha", "hrsid_result_sha"}
    require(set(raw) == keys | paths | hashes and all(Path(raw[key]).is_absolute() for key in paths)
            and all(re.fullmatch(r"[0-9a-f]{64}", raw[key]) for key in hashes)
            and re.fullmatch(r"[0-9a-f]{40}", raw["prelock_commit"]) and sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and np.__version__ == "1.26.4"
            and version("pycocotools") == "2.0.11" and not cmd.output.exists(), "Fresh qualified score")
    a = SimpleNamespace(**{key: Path(value) if key in paths else value for key, value in raw.items()})
    require(a.protocol_sha == PROTOCOL_SHA, "Published SRNet paired evaluation prelock")
    pin(a.protocol, a.protocol_sha)
    run(a, cmd.output, cmd.authority_sha)


if __name__ == "__main__":
    main()

