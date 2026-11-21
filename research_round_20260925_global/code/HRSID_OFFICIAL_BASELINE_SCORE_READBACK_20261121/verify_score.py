"""Recompute official HRSID native COCO scores from pinned predictions and ground truth after scorer exit."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


SCORE = "c3322233db5234bf89510ea370c88a255dc4c995d80fc483d9903ef43ef01628"
ADAPTER = "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748"
ENDPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
COHORTS = "1afb4b95632d3c301320eb07711bb48aa468d3a154f33ad84e7b28c1af110827"
GT = "eb611754afc74c34222fa52cc04a500d1b84f956782fcbcc0e91d9766c4ac937"
INPUT = "15aabe26b7580c55a5b5f3e4ae1558cac84294ae950f283c826de5b0720e53ee"
EXPORT_VERIFIER = "c232f532e16813383d19531ddbcf7aee8acaf5a28e9b0669f0d62c49d14d449d"
RUNTIME = dict(python="3.8.17", numpy="1.23.5", pycocotools="2.0.7")
SCORE_PHASE = "HRSID_OFFICIAL_PNG_TEST_NATIVE_BASELINE_SCORE_ONLY_V1"
SUCCESS = "HRSID_OFFICIAL_TEST_CENTER5202_E24_NATIVE_COCO_COMPLETE"
COUNTS = {"official": (1962, 5922), "empty_raster_sensitivity": (1962, 5918),
          "image_identity_sensitivity": (1959, 5891)}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def valid_sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def document(path, expected, limit):
    require(valid_sha(expected), "Expected SHA-256")
    with path.open("rb") as src:
        raw = src.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected, "Exact closed bytes of " + path.name)
    return json.loads(raw)


def check_terminal(terminal):
    require(terminal.get("status") == "HRSID_OFFICIAL_TEST_NATIVE_SCORE_TERMINAL"
            and terminal.get("actual_process_terminal") is True
            and type(terminal.get("actual_exit")) is int and terminal["actual_exit"] == 0
            and terminal.get("footer_status") == SUCCESS and terminal.get("score_source_sha256") == SCORE
            and all(valid_sha(terminal.get(k)) for k in ("output_sha256", "admission_sha256")),
            "Observed successful original CPU scoring process and final footer")
    pid, session = terminal.get("worker_pid"), terminal.get("tool_session_handle")
    require((type(pid) is int and pid > 0) or (type(session) in (int, str) and re.fullmatch("[1-9][0-9]*", str(session))),
            "Original score process identity")


def check_bindings(terminal, admission, result, closure):
    common = dict(phase=SCORE_PHASE, endpoint_sha256=ENDPOINT, cohorts_sha256=COHORTS,
                  gt_sha256=GT, adapter_sha256=ADAPTER, score_source_sha256=SCORE, runtime_versions=RUNTIME)
    require(all(admission.get(k) == result.get(k) == v for k, v in common.items())
            and admission.get("status") == "ROOT_ADMITTED_HRSID_OFFICIAL_TEST_NATIVE_BASELINE_SCORE_V1"
            and result.get("status") == SUCCESS and result.get("admission_sha256") == terminal["admission_sha256"]
            and valid_sha(admission.get("export_readback_sha256"))
            and result.get("export_readback_sha256") == admission["export_readback_sha256"], "Exact admitted scorer and result")
    require(closure.get("status") == "HRSID_OFFICIAL_TEST_NATIVE_EXPORT_INDEPENDENT_READBACK_ACCEPTED"
            and closure.get("independent_source_sha256") == EXPORT_VERIFIER
            and closure.get("resume_mode") is True and closure.get("original_exit_applies_to") == "resumed_export_worker_pid"
            and "original_interrupted_exit" in closure and closure["original_interrupted_exit"] is None
            and closure.get("original_process_terminal") is True and type(closure.get("actual_resume_exit")) is int
            and closure["actual_resume_exit"] == 0 and type(closure.get("original_exit")) is int and closure["original_exit"] == 0
            and closure.get("endpoint_sha256") == ENDPOINT and closure.get("input_readback_sha256") == INPUT
            and closure.get("images") == 1962 and closure.get("GT_opened") is False and closure.get("AP") is False,
            "Distinct successful export terminal and accepted independent full readback")
    require(all(admission.get(k) == result.get(k) == closure.get(k) for k in ("predictions_sha256", "mapped_rows", "prediction_bytes"))
            and valid_sha(closure.get("predictions_sha256")) and type(closure.get("mapped_rows")) is int
            and 0 <= closure["mapped_rows"] <= 196200 and type(closure.get("prediction_bytes")) is int
            and 0 < closure["prediction_bytes"] <= 1024**3, "Closed full predictions used by scorer")
    require(result.get("AP") is True and result.get("iou_type") == "segm" and result.get("max_dets") == [1, 10, 100]
            and result.get("historical_test_selection_exposure") is True
            and result.get("units") == "fraction; -1 means undefined in COCO"
            and all(type(result.get(k)) is int and result[k] == 0 for k in ("model_forward_calls", "optimizer_updates")),
            "Closed scientific score scope")


def check_populations(plan, gt):
    require(plan.get("status") == "HRSID_TEST_SCORING_COHORTS_FROZEN_BEFORE_PREDICTIONS"
            and plan.get("endpoint_sha256") == ENDPOINT and plan.get("independent_input_readback_sha256") == INPUT,
            "Predeclared cohort provenance")
    views = plan["views"]
    require(set(views) == set(COUNTS) and [c["id"] for c in gt["categories"]] == [1], "Original HRSID view/category schema")
    for name, (nimages, nannotations) in COUNTS.items():
        view = views[name]
        require(set(view) == {"images", "annotations"} and len(view["images"]) == len(set(view["images"])) == nimages
                and len(view["annotations"]) == len(set(view["annotations"])) == nannotations, "Exact cohort size and unique IDs")
    official, empty, identity = (views[k] for k in COUNTS)
    require(official["images"] == [x["id"] for x in gt["images"]]
            and official["annotations"] == [a["id"] for a in gt["annotations"]], "Original GT member IDs and order")
    kept = set(identity["images"])
    require(identity["images"] == [i for i in official["images"] if i in kept]
            and identity["annotations"] == [a["id"] for a in gt["annotations"] if a["image_id"] in kept],
            "All predictions and GT for retained identity-view images")
    removed = set(official["annotations"]) - set(empty["annotations"])
    require(empty["images"] == official["images"]
            and empty["annotations"] == [i for i in official["annotations"] if i not in removed] and len(removed) == 4,
            "Exactly four removed GT annotations across identical image populations")
    with redirect_stdout(StringIO()):
        original = COCO()
        original.dataset = gt
        original.createIndex()
        require(all(not original.annToMask(original.anns[i]).any() for i in removed)
                and len({original.anns[i]["image_id"] for i in removed}) == 4, "Independent four removed GT empty raster check")
    return views


def prediction_rows(path, official_ids, closure):
    expected = closure["prediction_bytes"]
    rows, hashes, file_hash = {}, {}, sha256()
    nbytes = mapped = 0
    with path.open("rb") as src:
        before = os.fstat(src.fileno())
        require(before.st_size == expected, "Closed prediction length")
        for image_id in official_ids:
            line = src.readline(expected - nbytes + 1)
            require(line.endswith(b"\n") and nbytes + len(line) <= expected, "Complete official prediction block")
            nbytes += len(line)
            file_hash.update(line)
            hashes[image_id] = sha256(line).digest()
            block = json.loads(line)
            require(block["image_id"] == image_id and all(r["image_id"] == image_id and r["category_id"] == 1 for r in block["coco"]),
                    "Official prediction and row identities")
            rows[image_id] = block["coco"]
            mapped += len(rows[image_id])
        require(src.read(1) == b"" and nbytes == expected and file_hash.hexdigest() == closure["predictions_sha256"]
                and mapped == closure["mapped_rows"], "Exact prediction SHA, coverage and mapped rows")
        after = os.fstat(src.fileno())
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Stable prediction descriptor")
    return rows, hashes


def coco_metrics(evaluator):
    p = evaluator.params
    require(p.iouType == "segm" and p.maxDets == [1, 10, 100] and p.catIds == [1]
            and np.allclose(p.iouThrs, np.arange(.5, .951, .05)) and len(p.recThrs) == 101, "Original COCO grid")

    def mean_valid(use_precision, area="all", limit=100, threshold=None):
        array = evaluator.eval["precision" if use_precision else "recall"]
        values = array[..., p.areaRngLbl.index(area), p.maxDets.index(limit)]
        if threshold is not None:
            values = values[np.flatnonzero(np.isclose(p.iouThrs, threshold))]
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.

    metrics = dict(AP=mean_valid(True), AP50=mean_valid(True, threshold=.5), AP75=mean_valid(True, threshold=.75),
                   AP90=mean_valid(True, threshold=.90))
    for area, key in zip(("small", "medium", "large"), ("s", "m", "l")):
        metrics["AP" + key] = mean_valid(True, area=area)
        metrics["AR" + key] = mean_valid(False, area=area)
    for n in (1, 10, 100):
        metrics["AR" + str(n)] = mean_valid(False, limit=n)
    return metrics


def evaluate_views(gt, views, rows, block_digests):
    images = {r["id"]:r for r in gt["images"]}
    annotations = {r["id"]:r for r in gt["annotations"]}
    out = {}
    for name, view in views.items():
        ids = view["images"]
        anns = [annotations[i] for i in view["annotations"]]
        flat = [r for i in ids for r in rows[i]]
        with redirect_stdout(StringIO()):
            ref = COCO()
            ref.dataset = dict(info=gt.get("info", {}), categories=deepcopy(gt["categories"]),
                               images=[images[i] for i in ids], annotations=deepcopy(anns))
            ref.createIndex()
            if flat:
                dt = ref.loadRes(deepcopy(flat))
            else:
                dt = COCO()
                dt.dataset = dict(categories=ref.dataset["categories"], images=ref.dataset["images"], annotations=[])
                dt.createIndex()
            ev = COCOeval(ref, dt, "segm")
            ev.params.imgIds = list(ids)
            ev.params.catIds = [1]
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
        annotated = {a["image_id"] for a in anns}
        out[name] = dict(images=len(ids), annotations=len(anns), mapped_predictions=len(flat),
                         images_without_annotations=len(set(ids) - annotated), images_without_predictions=sum(not rows[i] for i in ids),
                         ordered_prediction_block_digests_sha256=sha256(b"".join(block_digests[i] for i in ids)).hexdigest(),
                         metrics=coco_metrics(ev))
        del ev, ref, dt
    return out


def compare_views(actual, reported):
    require(set(actual) == set(reported) == set(COUNTS), "All three reported and recomputed views")
    for name in COUNTS:
        left, right = actual[name], reported[name]
        require(set(left) == set(right) and all(type(left[k]) is type(right[k]) and left[k] == right[k]
                for k in left if k != "metrics"), "Independent population, rows and block hashes for " + name)
        require(set(left["metrics"]) == set(right["metrics"]) and len(left["metrics"]) == 13
                and all(type(value) is float and value == left["metrics"][key]
                        for key, value in right["metrics"].items()), "Exact independently recomputed metrics for " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("terminal", "admission", "result", "export-readback", "cohorts", "gt", "predictions", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--terminal-sha256", required=True)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and dict(python=platform.python_version(), numpy=np.__version__, pycocotools=version("pycocotools")) == RUNTIME,
            "Original isolated scoring CPU environment")
    require(not args.output.exists() and args.output.parent.is_dir(), "Fresh readback receipt")
    terminal = document(args.terminal, args.terminal_sha256, 1024**2)
    check_terminal(terminal)
    admission = document(args.admission, terminal["admission_sha256"], 1024**2)
    result = document(args.result, terminal["output_sha256"], 1024**2)
    closure = document(args.export_readback, admission["export_readback_sha256"], 1024**2)
    check_bindings(terminal, admission, result, closure)
    plan = document(args.cohorts, COHORTS, 4 * 1024**2)
    gt = document(args.gt, GT, 16 * 1024**2)
    views = check_populations(plan, gt)
    rows, hashes = prediction_rows(args.predictions, views["official"]["images"], closure)
    actual = evaluate_views(gt, views, rows, hashes)
    compare_views(actual, result["scores"])
    official = actual["official"]
    require(official["ordered_prediction_block_digests_sha256"] == closure["ordered_prediction_block_digests_sha256"]
            and official["mapped_predictions"] == closure["mapped_rows"]
            and official["images_without_predictions"] == closure["images_without_predictions"]
            and official["images_without_annotations"] == 1, "Independent full export and GT population join")
    accepted = dict(status="HRSID_OFFICIAL_TEST_CENTER5202_E24_NATIVE_SCORE_INDEPENDENT_READBACK_ACCEPTED",
                    original_score_terminal_sha256=args.terminal_sha256, original_score_process_exit=terminal["actual_exit"],
                    original_score_worker_pid=terminal.get("worker_pid"), original_score_tool_session_handle=terminal.get("tool_session_handle"),
                    score_result_sha256=terminal["output_sha256"], admission_sha256=terminal["admission_sha256"],
                    score_source_sha256=SCORE, original_scorer_adapter_sha256=ADAPTER, endpoint_sha256=ENDPOINT,
                    export_readback_sha256=admission["export_readback_sha256"], predictions_sha256=closure["predictions_sha256"],
                    prediction_bytes=closure["prediction_bytes"], mapped_rows=closure["mapped_rows"], cohorts_sha256=COHORTS,
                    gt_sha256=GT, input_readback_sha256=INPUT, runtime_versions=RUNTIME, historical_test_selection_exposure=True,
                    checked_metrics=39, exact_metric_equality=True, model_forward_calls=0, optimizer_updates=0,
                    readback_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), scores=actual)
    raw = (json.dumps(accepted, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as dest:
        require(dest.write(raw) == len(raw), "Complete independent score readback")
        dest.flush()
        os.fsync(dest.fileno())
    print(json.dumps(dict(status=accepted["status"], readback_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
