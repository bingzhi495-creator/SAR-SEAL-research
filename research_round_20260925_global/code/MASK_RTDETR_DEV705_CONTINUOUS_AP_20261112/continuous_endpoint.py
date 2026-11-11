"""Re-score the closed author Mask RT-DETR DEV705 pair under fixed P374 overlap conventions."""

import argparse
from collections import defaultdict
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


ARMS = ("native", "center")
STATUS = "MASK_RTDETR_DEV705_E24_AUTHOR_PAIR_CONTINUOUS_SENSITIVITY_COMPLETE"
SCORE_STATUS = "MASK_RTDETR_DEV705_E24_AUTHOR_PAIR_SCORE_COMPLETE"
PROTOCOL = "437fa55f14644eab15156dc96092903bfde147da"
GT_SHA = "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628"
MANIFEST_SHA = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
OFFICIAL_ADAPTER = "DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py"
OFFICIAL_ADAPTER_SHA = "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748"
SCORE_SOURCE_SHA = "b64b9c4d9ceb90a53fa8b84a05fc95690ab60b67214eb7ba5ab2cc42af5a4c43"
DEV_ADAPTER_SHA = "1ce0eaa2106fe52452129fad534d0fc11bb8ad5fc4611f70858dbdf6f87a5ec3"
STREAM_SHA = "55e730a51432b4a2973e7ad75c45c31480b763b3b45b0b93b1b9aa340b7228e7"
MASK_ROWS_SHA = "1d55a7e4e79a6e3e62733c4c12547e27ec1ba713b1b92e5df0a183c0f1fcb1de"
GEOMETRY_SOURCES = {
    "geometry": ("HRSID_DEV705_CONTINUOUS_AP_20261111/full_continuous_ap.py",
                 "cba94ecd0df862b6a303117a56fa889bc76c0a8c017af50fff2cdb9859f94139"),
    "shift": ("HRSID_COMMON_DEV705_FIXED_SHIFT_AP_20261111/fixed_shift_ap.py",
              "68a7ec9e0811cc5b18d2b97a09c13b2471f3908a36b39c3dedd4a50b098f97f2"),
    "vector": ("HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110/vector_label_geometry.py",
               "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09"),
    "footprint": ("HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110/vector_population_math.py",
                  "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e"),
}
MIB = 1024**2


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def pinned(path, expected_sha, limit=MIB):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
         and 0 < path.stat().st_size <= limit and re.fullmatch("[0-9a-f]{64}", expected_sha) is not None,
         "Canonical bounded pinned input required: " + path.name)
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected_sha, "Original pinned digest differs: " + path.name)
    return raw


def load_module(path, expected_sha, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(pinned(path, expected_sha), str(path), "exec"), module.__dict__)
    return module


def geometry_modules(code_root):
    return SimpleNamespace(**{key: load_module(code_root / path, digest, "_paired_geometry_" + key)
                              for key, (path, digest) in GEOMETRY_SOURCES.items()})


def native_layout(rows):
    """Derive COCO's one-based detection ids and stable ranking directly from input."""
    by_image = defaultdict(list)
    for detection_id, row in enumerate(rows, 1):
        by_image[row["image_id"]].append((detection_id, row["score"]))
    return tuple((image_id, tuple(det_id for det_id, _ in order), tuple(score for _, score in order))
                 for image_id, entries in sorted(by_image.items())
                 for order in [sorted(entries, key=lambda entry: -entry[1])])


def compare(gt, streams, priors, adapter, modules, *, expected_support, required_dense_spots):
    geometry = modules.geometry
    raw, ranks = {}, {}
    for arm in ARMS:
        raw[arm], ranks[arm], _ = geometry.score(gt, streams[arm], adapter)
        expected = priors[arm]
        need(len(expected["all12"]) == 12
             and np.allclose(raw[arm]["all12"], expected["all12"], rtol=0, atol=1e-12)
             and abs(raw[arm]["AP90"] - expected["AP90"]) <= 1e-12,
             "All 12 original full-raster metrics and AP90 must reproduce before shared support: " + arm)
        need(ranks[arm] == native_layout(streams[arm]), "Original native score order or ties changed: " + arm)

    polygons, neutral, support = geometry.polygon_support(gt, modules.vector)
    need((len(polygons), len(neutral)) == expected_support, "Same frozen original polygon support required")
    results, audits = {}, {}
    for arm in ARMS:
        rows = streams[arm]
        raster, raster_rank, _ = geometry.score(gt, rows, adapter, neutral)
        continuous, continuous_rank, audit = geometry.score(gt, rows, adapter, neutral, polygons,
                                                           modules.shift.runs, modules.footprint.run_footprint_metrics)
        need(ranks[arm] == raster_rank == continuous_rank == native_layout(rows),
             "Native candidate ranks, ties, per-image layout and scores must remain fixed: " + arm)
        if required_dense_spots is not None:
            need((audit["dense_geometry_spots"], audit["pruned_dense_spots"]) == required_dense_spots,
                 "Exact P374 independent overlap/pruning checks required: " + arm)
        results[arm] = dict(original_full_raster=raw[arm], shared_support_raster=raster,
                            shared_support_continuous=continuous)
        audits[arm] = audit

    deltas = {mode: {metric: 100 * (results["center"][mode][metric] - results["native"][mode][metric])
                     for metric in geometry.METRICS} for mode in results["native"]}
    deltas["continuous_minus_raster_common_effect_points"] = {
        key: deltas["shared_support_continuous"][key] - deltas["shared_support_raster"][key]
        for key in geometry.METRICS
    }
    return dict(results=results, delta_points=deltas, support=support, supported_unique_annotations=len(polygons),
                neutral_unique_annotations=len(neutral), geometry_audits=audits, original_order_unchanged=True,
                raw_prediction_rows={arm: len(streams[arm]) for arm in ARMS}, verified_original_raster_scores=True)



def referenced_bytes(path, reference, limit=8 * MIB):
    need(Path(reference["path"]) == path, "Exact original artifact reference path required: " + path.name)
    data = pinned(path, reference["sha256"], limit)
    need(len(data) == reference["bytes"], "Exact original artifact reference length required: " + path.name)
    return data


def verify_score_receipt(score_dir, terminal_sha, score_sha, export_wait_sha):
    terminal = json.loads(pinned(score_dir / "ORIGINAL_SCORE_TERMINAL_PRIVATE.json", terminal_sha))
    need(terminal["original_os_wait"] is True and type(terminal["original_os_exit"]) is int
         and terminal["original_os_exit"] == 0
         and terminal["original_export_owner_wait_sha256"] == export_wait_sha,
         "Successful original independent CPU score closure must bind the original CUDA owner wait")
    stdout = referenced_bytes(score_dir / "ORIGINAL_SCORE.stdout", terminal["stdout_ref"])
    marker = json.loads(stdout.splitlines()[-1])
    need(marker == dict(status=SCORE_STATUS, result_sha256=score_sha),
         "Exact original independent CPU score footer must bind its result")
    result = json.loads(pinned(score_dir / "RESULT_SAFE.json", score_sha))
    need(result["status"] == SCORE_STATUS, "Exact original P392 score status required")
    return result


def verify_scope(result, pair_sha):
    expected = dict(protocol_commit=PROTOCOL, images=705, max_dets=[1, 10, 100], units="fraction",
                    original_gt_sha256=GT_SHA, image_manifest_sha256=MANIFEST_SHA, original_annotations=1782,
                    empty_sensitivity_exclusions=4, official_TEST_opened=False, official_TEST_AP=False,
                    paired_receipt_sha256=pair_sha)
    need(all(type(result[key]) is type(value) and result[key] == value for key, value in expected.items())
         and set(result["arms"]) == set(ARMS), "Exact previously scored P392 DEV705 paired population required")


def verify_exporter_closure(args, arm):
    closure_path = args.score_dir / (arm + "_ORIGINAL_CLOSED_PRIVATE.json")
    closure_sha = getattr(args, arm + "_closure_sha256")
    closure = json.loads(pinned(closure_path, closure_sha))
    terminal_path = args.export_root / arm / "CONTROL/OWNER_TERMINAL_PRIVATE.json"
    need(closure["arm"] == arm and closure["evidence"] == "ORIGINAL_WINDOWS_OS_WAIT_AND_EXTERNAL_OWNER_TOOL_WAIT"
         and closure["original_external_wait_sha256"] == args.export_wait_sha256,
         "Both exporter closures must bind the same original CUDA owner wait")
    terminal = json.loads(referenced_bytes(terminal_path, closure["original_child_terminal_ref"]))
    need(terminal["original_os_wait"] is True and type(terminal["original_os_exit"]) is int
         and terminal["original_os_exit"] == 0 and terminal["job_assigned"] is True
         and terminal["owner"] is not None and terminal["job_active_before_drain"] == 0
         and terminal["job_active_after_wait"] == 0 and terminal["stop_reason"] is None,
         "Exact original natural exporter closure required")
    original_stdout = referenced_bytes(terminal_path.parent / "OWNER.stdout", terminal["stdout_ref"])
    need(closure["exact_tool_wait"] == dict(exit_code=0, output=original_stdout.decode("utf-8")),
         "Original exporter stdout must match its independent score-owner receipt")
    return closure_path, closure_sha


def closed_arm(scorer, args, result, shapes, arm, closure):
    closure_path, closure_sha = closure
    report, rows, report_sha = scorer.read_closed_arm(arm, args.export_root / arm / "OUTPUT",
                                                        closure_path, closure_sha, shapes, MANIFEST_SHA)
    prior = result["arms"][arm]
    need(report["paired_receipt_sha256"] == args.paired_receipt_sha256
         and prior["export_report_sha256"] == report_sha
         and prior["checkpoint_sha256"] == report["checkpoint_sha256"]
         and prior["predictions_sha256"] == report["predictions_sha256"]
         and prior["native_rows"] == report["rows"] == len(rows) == 70500
         and prior["primary"]["annotations"] == 1782 and prior["sensitivity"]["annotations"] == 1778,
         "Original scorer, endpoint, stream and paired training digests must all agree")
    return rows, dict(closure_sha256=closure_sha, export_report_sha256=report_sha,
                      predictions_sha256=report["predictions_sha256"], checkpoint_sha256=report["checkpoint_sha256"])


def scoring_modules(code_root, p392_source):
    source = p392_source / "author_dev_score.py"
    pinned(code_root / "MASK_RTDETR_AUTHOR_MASK_ROWS_20261111/author_mask_rows.py", MASK_ROWS_SHA)
    sys.path.insert(0, str(code_root))
    sys.modules["author_stream"] = load_module(p392_source / "author_stream.py", STREAM_SHA, "author_stream")
    scorer = load_module(source, SCORE_SOURCE_SHA, "_original_mask_rtdetr_score")
    need(scorer.SOURCE_SHA == DEV_ADAPTER_SHA and scorer.PROTOCOL == PROTOCOL,
         "Exact P392 independent score source and DEV adapter binding required")
    common = scorer.evaluator()
    need(common.FILES["DEV_ORIGINAL_GT_PRIVATE.json"] == GT_SHA
         and common.FILES["DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"] == MANIFEST_SHA
         and common.OFFICIAL_SHA == OFFICIAL_ADAPTER_SHA, "Exact stock and DEV population source pins required")
    return scorer, common


def load_inputs(args):
    result = verify_score_receipt(args.score_dir, args.score_terminal_sha256, args.score_sha256,
                                  args.export_wait_sha256)
    verify_scope(result, args.paired_receipt_sha256)
    closures = {arm: verify_exporter_closure(args, arm) for arm in ARMS}
    scorer, common = scoring_modules(args.code_root, args.p392_source)
    manifest = common.qualified_manifest()
    shapes = {row["image_id"]: (row["height"], row["width"]) for row in manifest}
    need(len(shapes) == len(manifest) == 705 and set(shapes.values()) == {(800, 800)},
         "Original ordered DEV705 roster and native canvases required")
    streams, arm_provenance = {}, {}
    for arm in ARMS:
        streams[arm], arm_provenance[arm] = closed_arm(scorer, args, result, shapes, arm, closures[arm])
    need(len({row["checkpoint_sha256"] for row in arm_provenance.values()}) == 2,
         "Two independently scored endpoint checkpoints required")
    gt = common.dev_file("DEV_ORIGINAL_GT_PRIVATE.json")
    need(len(gt["annotations"]) == 1782 and [row["id"] for row in gt["images"]] == list(shapes)
         and [row["id"] for row in gt["categories"]] == [1],
         "Frozen original DEV705 COCO annotations, image order and category required")
    return gt, streams, {arm: result["arms"][arm]["primary"] for arm in ARMS}, arm_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "p392-source", "score-dir", "export-root", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("score-terminal-sha256", "score-sha256", "export-wait-sha256", "paired-receipt-sha256",
                 "native-closure-sha256", "center-closure-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    need(os.name == "nt" and sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == 1
         and sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
         and all(os.environ.get(name) == value for name, value in
                 (("CUDA_VISIBLE_DEVICES", "-1"), ("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")))
         and version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11",
         "Exact isolated one-thread CPU scoring runtime required")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink()
         and args.output.parent not in (args.score_dir, args.export_root)
         and not args.output.is_relative_to(args.export_root) and not args.output.is_relative_to(args.score_dir),
         "Fresh result path outside the original run and score directories required")
    need(all(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir()
             for path in (args.code_root, args.p392_source, args.score_dir, args.export_root)),
         "Canonical explicit source and closed directories required")
    need(all(re.fullmatch("[0-9a-f]{64}", getattr(args, name)) for name in
             ("score_terminal_sha256", "score_sha256", "export_wait_sha256", "paired_receipt_sha256",
              "native_closure_sha256", "center_closure_sha256")), "Original pinned SHA-256 values required")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
         "Exact previously used continuous geometry runtime required")

    def offline(event, _):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Closed offline CPU diagnostic only")

    sys.addaudithook(offline)
    started = perf_counter()
    modules = geometry_modules(args.code_root)
    adapter = load_module(args.code_root / OFFICIAL_ADAPTER, OFFICIAL_ADAPTER_SHA, "_p374_stock_adapter")
    gt, streams, priors, arms = load_inputs(args)
    values = compare(gt, streams, priors, adapter, modules, expected_support=(1761, 21),
                     required_dense_spots=(16, 8))
    report = dict(status=STATUS, primary_p392_raster_unchanged=True, policy=modules.geometry.POLICY,
                  training_protocol=PROTOCOL, images=705, original_unique_annotations=1782, DEV=True, FIT=False,
                  official_TEST_opened=False, official_TEST_AP=False, optimizer_updates=0, model_forward_calls=0,
                  original_score_sha256=args.score_sha256, original_score_terminal_sha256=args.score_terminal_sha256,
                  original_export_wait_sha256=args.export_wait_sha256, paired_receipt_sha256=args.paired_receipt_sha256,
                  original_gt_sha256=GT_SHA, image_manifest_sha256=MANIFEST_SHA,
                  provenance_by_arm=arms, elapsed_seconds=perf_counter() - started, **values)
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
