"""Score one independently closed native center5202 E24 export on three frozen TEST cohorts."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import math
import os
from pathlib import Path
import platform
import re
from types import ModuleType


PHASE = "HRSID_OFFICIAL_PNG_TEST_NATIVE_BASELINE_SCORE_ONLY_V1"
ADMITTED = "ROOT_ADMITTED_HRSID_OFFICIAL_TEST_NATIVE_BASELINE_SCORE_V1"
EXPORT_ACCEPTED = "HRSID_OFFICIAL_TEST_NATIVE_EXPORT_INDEPENDENT_READBACK_ACCEPTED"
ENDPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
COHORT = "1afb4b95632d3c301320eb07711bb48aa468d3a154f33ad84e7b28c1af110827"
GT = "eb611754afc74c34222fa52cc04a500d1b84f956782fcbcc0e91d9766c4ac937"
INPUT_READBACK = "15aabe26b7580c55a5b5f3e4ae1558cac84294ae950f283c826de5b0720e53ee"
ADAPTER = "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748"
COUNTS = {"official": (1962, 5922), "empty_raster_sensitivity": (1962, 5918),
          "image_identity_sensitivity": (1959, 5891)}
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl",
           "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return sha256(value).hexdigest()


def valid_sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def pinned(path, expected, max_bytes):
    check(valid_sha(expected), "Expected pinned SHA-256")
    with path.open("rb") as stream:
        check(0 < os.fstat(stream.fileno()).st_size <= max_bytes, "Bounded input size")
        raw = stream.read(max_bytes + 1)
    check(0 < len(raw) <= max_bytes and digest(raw) == expected, "Pinned file bytes")
    return raw


def adapter_from(path):
    source = pinned(path, ADAPTER, 1024**2)
    module = ModuleType("_pinned_native_coco_adapter")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def runtime():
    import numpy
    return dict(python=platform.python_version(), numpy=numpy.__version__, pycocotools=version("pycocotools"))


def validate_admission(a):
    check(a.get("phase") == PHASE and a.get("status") == ADMITTED, "An explicit native baseline score admission is required")
    check(a.get("endpoint_sha256") == ENDPOINT and a.get("cohorts_sha256") == COHORT
          and a.get("gt_sha256") == GT and a.get("adapter_sha256") == ADAPTER, "Exact frozen baseline, cohort, GT and COCO adapter")
    check(a.get("score_source_sha256") == digest(Path(__file__).read_bytes())
          and a.get("runtime_versions") == runtime(), "Admitted scoring source and imported runtime")
    check(valid_sha(a.get("export_readback_sha256")) and valid_sha(a.get("predictions_sha256"))
          and type(a.get("mapped_rows")) is int and 0 <= a["mapped_rows"] <= 196200
          and type(a.get("prediction_bytes")) is int and a["prediction_bytes"] > 0,
          "Independently closed native prediction identity and size")


def validate_plan(plan):
    check(plan.get("status") == "HRSID_TEST_SCORING_COHORTS_FROZEN_BEFORE_PREDICTIONS"
          and plan.get("endpoint_sha256") == ENDPOINT and plan.get("independent_input_readback_sha256") == INPUT_READBACK,
          "Original independently closed cohort")
    views = plan["views"]
    check(set(views) == set(COUNTS), "Exactly three frozen cohorts")
    for name, (n_images, n_annotations) in COUNTS.items():
        view = views[name]
        check(set(view) == {"images", "annotations"}, "Frozen role schema")
        for key, count in (("images", n_images), ("annotations", n_annotations)):
            ids = view[key]
            check(type(ids) is list and len(ids) == len(set(ids)) == count and all(type(i) is int for i in ids),
                  "Distinct original identities in the frozen population")
    return views


def validate_export_closure(closure, admission):
    check(closure.get("status") == EXPORT_ACCEPTED and closure.get("GT_opened") is False and closure.get("AP") is False
          and closure.get("images") == 1962 and closure.get("original_process_terminal") is True
          and type(closure.get("original_exit")) is int and closure["original_exit"] == 0,
          "Independent full label-free native export readback and observed natural exit")
    check(all(closure.get(k) == admission[k] for k in ("endpoint_sha256", "predictions_sha256", "prediction_bytes", "mapped_rows")),
          "Scoring inputs joined to independent prediction readback")


def validate_rle(segmentation, height, width):
    check(type(segmentation) is dict and set(segmentation) == {"size", "counts"}
          and segmentation["size"] == [height, width], "Native original-canvas RLE")
    counts = segmentation["counts"]
    check(type(counts) is str and bool(counts), "COCO compressed ASCII runs")
    # COCO's variable-length signed integers use a two-run delta starting at run 4.
    # Check coverage before passing the externally supplied byte stream to the C extension.
    total = value = shift = n = 0
    older = recent = 0
    for char in counts:
        code = ord(char) - 48
        check(0 <= code < 64 and shift <= 25, "Valid compressed RLE group")
        value |= (code & 31) << shift
        shift += 5
        if code & 32:
            continue
        if code & 16:
            value -= 1 << shift
        count = value + (older if n >= 3 else 0)
        total += count
        check(0 <= count and total <= height * width, "Bounded native RLE runs")
        older, recent = recent, count
        n += 1
        value = shift = 0
    check(shift == 0 and total == height * width, "Complete native RLE canvas")


def parse_native(stream, image_ids, height, width, expected):
    file_digest, byte_count, mapped, rows, row_digests = sha256(), 0, 0, {}, {}
    for image_id in image_ids:
        line = stream.readline(expected["prediction_bytes"] - byte_count + 1)
        check(line.endswith(b"\n") and byte_count + len(line) <= expected["prediction_bytes"], "Complete native image block")
        byte_count += len(line)
        file_digest.update(line)
        block = json.loads(line)
        rank, predictions = block["native_ranks"], block["coco"]
        check(type(block["image_id"]) is int and block["image_id"] == image_id and block["raw_query_count"] == 164
              and block["native_count"] == 100 and type(block["unmapped_slot1_count"]) is int
              and 0 <= block["unmapped_slot1_count"] <= 100 and type(predictions) is list
              and block["unmapped_slot1_count"] + len(predictions) == 100
              and type(rank) is list and len(rank) == len(predictions) and all(type(r) is int and 1 <= r <= 100 for r in rank)
              and rank == sorted(set(rank)), "Exact ordered image and complete native slot accounting")
        last = 1.
        for row in predictions:
            check(set(row) == {"image_id","category_id","score","segmentation"} and type(row["image_id"]) is int
                  and row["image_id"] == image_id and type(row["category_id"]) is int and row["category_id"] == 1
                  and type(row["score"]) in (float, int) and math.isfinite(row["score"]) and 0 <= row["score"] <= last,
                  "Fixed native score and category order")
            last = row["score"]
            validate_rle(row["segmentation"], height, width)
        rows[image_id] = predictions
        row_digests[image_id] = sha256(line).digest()
        mapped += len(predictions)
    check(stream.read(1) == b"" and file_digest.hexdigest() == expected["predictions_sha256"]
          and mapped == expected["mapped_rows"] and byte_count == expected["prediction_bytes"],
          "Exact complete predictions bytes, row count and order")
    return rows, row_digests


def evaluate_views(gt, views, rows, row_digests, adapter):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    full = views["official"]
    check(full["images"] == [i["id"] for i in gt["images"]] == list(rows)
          and full["annotations"] == [a["id"] for a in gt["annotations"]] and [c["id"] for c in gt["categories"]] == [1],
          "Exact original primary image/annotation identities and order")
    empty, identity = views["empty_raster_sensitivity"], views["image_identity_sensitivity"]
    image_keep, ann_keep = set(identity["images"]), set(empty["annotations"])
    check(empty["images"] == full["images"] and empty["annotations"] == [i for i in full["annotations"] if i in ann_keep],
          "Empty-only sensitivity preserves every original image")
    check(identity["images"] == [i for i in full["images"] if i in image_keep]
          and identity["annotations"] == [a["id"] for a in gt["annotations"] if a["image_id"] in image_keep],
          "Image-only sensitivity retains all original annotations of every kept image")
    output = {}
    for name in ("official", "empty_raster_sensitivity", "image_identity_sensitivity"):
        view = views[name]
        ids, keep = set(view["images"]), set(view["annotations"])
        local_gt = dict(gt, images=[i for i in gt["images"] if i["id"] in ids],
                        annotations=[a for a in gt["annotations"] if a["id"] in keep])
        # COCOeval rewrites annotation segmentation and ignore fields in place.
        coco = COCO()
        coco.dataset = deepcopy(local_gt)
        flat = [r for i in view["images"] for r in rows[i]]
        captured = []

        def evaluator(*args):
            ev = COCOeval(*args)
            captured.append(ev)
            return ev

        with redirect_stdout(StringIO()):
            coco.createIndex()
            values = adapter.evaluate_population(dict(COCO=COCO, COCOeval=evaluator), coco, flat, view["images"])
        ev, = captured
        params = ev.params
        check(params.iouType == "segm" and params.catIds == [1] and params.maxDets == [1, 10, 100]
              and set(params.imgIds) == ids and len(params.imgIds) == len(ids)
              and np.allclose(params.iouThrs, np.arange(.5, .951, .05)) and len(params.recThrs) == 101,
              "Original full COCO segm evaluation parameters")
        t90, = np.flatnonzero(np.isclose(params.iouThrs, .90))
        precision = ev.eval["precision"][t90, :, :, params.areaRngLbl.index("all"), params.maxDets.index(100)]
        valid = precision[precision > -1]
        metrics = dict(zip(METRICS, values))
        metrics["AP90"] = float(valid.mean()) if valid.size else -1.
        group_digest = sha256()
        for i in view["images"]:
            group_digest.update(row_digests[i])
        annotated = {a["image_id"] for a in local_gt["annotations"]}
        output[name] = dict(images=len(ids), annotations=len(keep), images_without_annotations=len(ids - annotated),
                            mapped_predictions=len(flat), images_without_predictions=sum(not rows[i] for i in ids),
                            ordered_prediction_block_digests_sha256=group_digest.hexdigest(), metrics=metrics)
        del ev, captured, coco, flat
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("admission", "cohorts", "export-readback", "gt", "predictions", "adapter", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--admission-sha256", required=True)
    a = p.parse_args(argv)
    admission = json.loads(pinned(a.admission, a.admission_sha256, 1024**2))
    validate_admission(admission)
    views = validate_plan(json.loads(pinned(a.cohorts, COHORT, 4 * 1024**2)))
    closure = json.loads(pinned(a.export_readback, admission["export_readback_sha256"], 4 * 1024**2))
    validate_export_closure(closure, admission)
    check(not a.output.exists() and a.output.parent.is_dir(), "Fresh score output")
    adapter = adapter_from(a.adapter)
    with a.predictions.open("rb") as stream:
        check(os.fstat(stream.fileno()).st_size == admission["prediction_bytes"], "Exact closed prediction size")
        preflight = sha256()
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            preflight.update(chunk)
        check(preflight.hexdigest() == admission["predictions_sha256"], "Original prediction identity before parsing")
        stream.seek(0)
        rows, row_digests = parse_native(stream, views["official"]["images"], 800, 800, admission)
    gt = json.loads(pinned(a.gt, GT, 16 * 1024**2))
    scores = evaluate_views(gt, views, rows, row_digests, adapter)
    safe = dict(status="HRSID_OFFICIAL_TEST_CENTER5202_E24_NATIVE_COCO_COMPLETE", phase=PHASE, AP=True,
                model_forward_calls=0, optimizer_updates=0, units="fraction; -1 means undefined in COCO",
                iou_type="segm", max_dets=[1, 10, 100], scores=scores, endpoint_sha256=ENDPOINT, gt_sha256=GT,
                cohorts_sha256=COHORT, admission_sha256=a.admission_sha256, adapter_sha256=ADAPTER,
                historical_test_selection_exposure=True, **{k: admission[k] for k in
                ("runtime_versions", "export_readback_sha256", "predictions_sha256", "score_source_sha256", "prediction_bytes", "mapped_rows")})
    with a.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(safe, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status":safe["status"], "output_sha256":digest(a.output.read_bytes())}))


if __name__ == "__main__":
    main()
