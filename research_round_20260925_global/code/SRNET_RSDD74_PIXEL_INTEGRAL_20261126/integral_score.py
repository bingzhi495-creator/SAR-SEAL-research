"""Score the observed SRNet source pixel integral and native smoothing edits."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


PROTOCOL_SHA = "708c4183d61ffb36c2d3274a30eef5a429e4943d22fa98f8595bc610200dc026"
PRELOCK_COMMIT = "fbed6a496256e18640bb7c809877534b225dd9bc"
PRIOR_SCORE_SHA = "b4ca3fe0895c81035861e1c0b785b3faa5c549dbdf861582d29c5cb23ef3f7e8"
STATUS = "SRNET_RSDD74_PIXEL_INTEGRAL_SCORE_COMPLETE"
A, S, I = "source_pixel_integral", "native_smoothing", "free_matched_count"
CONTRASTS = ((A, I), (A, S), (S, I))


def run(c):
    from hashlib import sha256
    from types import ModuleType

    def require(ok, reason):
        if not ok:
            raise ValueError(reason)

    def pin(path, sha):
        data = path.read_bytes()
        require(sha256(data).hexdigest() == sha, "Exact digest " + path.name)
        return data

    def worker(path, sha):
        mod = ModuleType(path.stem)
        mod.__file__ = str(path)
        exec(compile(pin(path, sha), str(path), "exec"), mod.__dict__)
        return mod

    started = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
            and c.prelock_commit == PRELOCK_COMMIT and c.protocol_sha == PROTOCOL_SHA and not c.output.exists(),
            "Fresh qualified observed CPU score")
    pin(c.protocol, c.protocol_sha)
    p = worker(c.prior_score, PRIOR_SCORE_SHA)
    a, old, replay, edit, blocks, historic = p.load_original(c.previous)
    render = worker(c.render_worker, c.render_worker_sha)
    capture = render.checked_pregrid_capture(c, replay, a)
    prepare = replay.closed(c.prepare_terminal, c.prepare_terminal_sha, c.prepare / "RESULT_SAFE.json",
                            c.prepare_result_sha, "prepare-edits", c.render_worker_sha)
    verify = replay.closed(c.verify_terminal, c.verify_terminal_sha, c.verify / "RESULT_SAFE.json",
                           c.verify_result_sha, "verify-edits", c.render_worker_sha)
    require(prepare["status"] == render.PREPARED and verify["status"] == render.VERIFIED
            and verify["exact_independent_replay"] is True and prepare["exact_independent_replay"] is False
            and verify["prepare_result_sha256"] == c.prepare_result_sha
            and verify["prepare_terminal_sha256"] == c.prepare_terminal_sha, "Independent accepted label-free edits")
    for proof in (prepare, verify):
        require(proof["protocol_sha256"] == c.protocol_sha and proof["prelock_commit"] == c.prelock_commit
                and proof["original_edit_stream_sha256"] == edit["edits_sha256"]
                and proof["capture_result_sha256"] == c.capture_result_sha
                and proof["capture_terminal_sha256"] == c.capture_terminal_sha
                and proof["capture_readback_result_sha256"] == c.capture_readback_result_sha
                and proof["capture_readback_terminal_sha256"] == c.capture_readback_terminal_sha
                and proof["capture_index_sha256"] == capture["index_sha256"]
                and proof["integrator_sha256"] == c.integrator_sha
                and proof["fit_weights_sha256"] == replay.WEIGHTS_SHA
                and proof["original_fixed_and_full_edit_bytes_exact"] is True
                and proof["original_signed_k_and_support_exact"] is True and proof["residual_and_features_unchanged"] is True
                and proof["label_files_opened"] == proof["detector_forwards"] == proof["optimizer_updates"] == 0
                and proof["counts"] == prepare["counts"] and proof["edits_sha256"] == prepare["edits_sha256"]
                and proof["counts"]["images"] == 74 and proof["counts"]["candidates"] == 89
                and proof["counts"]["positive_candidates"] == edit["counts"]["positive_matched_budget_candidates"]
                and proof["counts"]["signed_swaps"] == edit["counts"]["exact_adds_per_policy"],
                "Binding original replay, unchanged residual and actual full denominators")
    rows = [json.loads(row) for row in pin(c.prepare / "EDITS_PRIVATE.jsonl", verify["edits_sha256"]).splitlines()]
    p.P, p.R = A, S
    p.add_write_arms(blocks, rows, replay.codec)

    reserve = old.load(a.code / old.RESERVE, old.RESERVE_SHA)
    official = old.load(a.code / old.HISTORIC, reserve.HISTORIC)
    evaluator = official.module(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", official.SCORE_SOURCE)
    old.support_empty_predictions(evaluator)
    adapter = official.module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", official.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    sys.path.append(str(a.shapely_site))
    import shapely

    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Frozen geometry backend")
    census = json.loads(pin(a.census, reserve.CENSUS))
    dates = {row["paper_scene"]:(row["sensor"], row["calendar_date"]) for row in census["per_parent"]}
    nested_groups = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    nested = {i for i, row in blocks.items() if row["group"] in nested_groups}
    require(len(nested) == 42 and set(dates) == {row["group"] for row in blocks.values()}, "Original nested view")
    print(json.dumps(dict(stage="INDEPENDENT_INTEGRAL_REPLAY_VERIFIED_BEFORE_LABELS")), flush=True)
    gt, ids_normalized = official.coco_annotation_ids(reserve.select_labels(pin(a.author_train, reserve.TRAIN), blocks))
    require(ids_normalized == historic["annotation_ids_rekeyed"], "Original author TRAIN annotation IDs")
    polygons = {item["id"]:shapely.Polygon(np.asarray(item["segmentation"][0]).reshape(-1,2)) for item in gt["annotations"]}
    n, f, i = p.N, p.F, p.I
    reserve.ARMS = (n, f, i, A, S)
    reserve.PAIRS = p.ORIGINAL_PAIRS + CONTRASTS
    p.NEW_PAIRS = CONTRASTS
    views = {}
    for name, ids in (("full74", set(blocks)), ("nested42_sensor_date", nested)):
        view = p.score_view(old, reserve, evaluator, core, shapely, gt, ids, blocks, polygons, dates)
        original = historic["views"][name]
        require({arm:view["metrics_fraction"][arm] for arm in (n,f,i)} == original["metrics_fraction"]
                and p.same_tree(view, original), "Exact original COCO and fixed-match N/F/I")
        view["directional_diagnostic"] = {}
        for left, right in CONTRASTS:
            key = left + " minus " + right
            ap = view["paired_contrasts_points"][key]["AP"]
            continuous = view["geometry"]["paired_contrasts_points"][key][reserve.FIELDS[0]]
            low, high = continuous["sensor_date_interval_95"]
            exposure = view["changed_anchors"][key]
            cover = exposure["anchors"] >= 20 and exposure["parents"] >= 8 and exposure["sensor_date_blocks"] >= 8
            direction = "favourable" if ap >= .30 and low > 0 else "adverse" if ap <= -.30 and high < 0 else "inconclusive"
            view["directional_diagnostic"][key] = dict(coverage_met=cover, result=direction if cover else "inconclusive")
        views[name] = view
    full_gates = views["full74"]["directional_diagnostic"]
    primary = [left + " minus " + right for left, right in CONTRASTS[:2]]
    projection_positive = all(full_gates[name]["result"] == "favourable" for name in primary)
    result = dict(status=STATUS, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=c.protocol_sha, prelock_commit=c.prelock_commit, frozen_score_sha256=p.SCORE_SHA,
                  capture_result_sha256=c.capture_result_sha, capture_terminal_sha256=c.capture_terminal_sha,
                  capture_readback_result_sha256=c.capture_readback_result_sha,
                  capture_readback_terminal_sha256=c.capture_readback_terminal_sha,
                  integrator_sha256=c.integrator_sha, accepted_replay_result_sha256=c.verify_result_sha,
                  accepted_replay_terminal_sha256=c.verify_terminal_sha, edits_sha256=verify["edits_sha256"],
                  old_n_f_i_ap_exact=True, old_n_f_i_geometry_tolerance=1e-12, original_author_train_only=True,
                  detector_forwards=0, optimizer_updates=0, author_test_opened=False, seed=5217,
                  bootstrap_draws=4096, counts=verify["counts"], prior_absolute_differences=verify["prior_absolute_differences"],
                  projection_specific_positive_full74=projection_positive,
                  primary_contrasts=primary, views=views, elapsed_seconds=perf_counter()-started)
    c.output.mkdir()
    raw = replay.encoded(result)
    (c.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=replay.digest(raw))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("protocol","previous","capture","prepare","prepare-terminal","verify","verify-terminal","prior-score",
                "render-worker","output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("protocol-sha","prelock-commit","capture-result-sha","capture-terminal-sha","capture-worker-sha",
                "capture-readback-result-sha","capture-readback-terminal-sha",
                "prepare-result-sha","prepare-terminal-sha","verify-result-sha","verify-terminal-sha",
                "render-worker-sha","integrator-sha"):
        p.add_argument("--" + key, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()

