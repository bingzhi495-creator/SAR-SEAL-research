"""Fixed-shortlist mask coverage and original-rank score donors on closed P392 DEV705."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

PROTOCOL_SHA = "df577adc7d23f3401eacecb9e3ec7e7c2fdfab7296155373e8022b78c2b12b58"
PINNED_SCORE_SOURCE = {
    "author_dev_score.py": "b64b9c4d9ceb90a53fa8b84a05fc95690ab60b67214eb7ba5ab2cc42af5a4c43",
    "author_stream.py": "55e730a51432b4a2973e7ad75c45c31480b763b3b45b0b93b1b9aa340b7228e7",
}
TEMPLATE_CODE = "HRSID_MASK_PHASE_REPEAT_DEV705_20261109"
PINNED_TEMPLATE_SOURCE = {
    "dev705_score_template.py": "65388e3f32388f8aea10b022e1a9c229848f10b19401fcecc292e410a48e5ee4",
    "dev705_paired_coverage.py": "2ac4abb1a0cb9e684f6bde6b2990419eb0ec209d469411b240dc7dd520e69d3d",
    "dev705_group_sensitivity.py": "f7a004ed3d128cffd2a7114618b892d964f1c00d946377044d0be09d46a871b9",
}
PRIMARY_STATUS = "MASK_RTDETR_DEV705_E24_AUTHOR_PAIR_SCORE_COMPLETE"
EXPORT_STATUS = "P392_MASK_RTDETR_DEV705_ORIGINAL_CUDA_EXPORTS_CLOSED_PENDING_SCORE"
RESULT_STATUS = "P392_HRSID_DEV705_EQUAL_NONEMPTY_COVERAGE_AND_SCORE_TEMPLATES_COMPLETE"
ARMS = ("native", "center")
AREAS = ("small", "medium", "large")
THRESHOLDS = (.50, .75, .90)
METRICS = ("AP", "AP50", "AP75", "AP90", "APs", "APm", "APl", "AR100")
MIB = 1024**2


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def pinned(path, digest, limit=MIB):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
         "Exact canonical closed input required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == digest, "Closed bytes differ: " + path.name)
    return raw


def document(path, digest, limit=MIB):
    return json.loads(pinned(path, digest, limit))


def load_original(path, digest, name):
    mod = ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(pinned(path, digest), str(path), "exec"), mod.__dict__)
    return mod


def admit_worker():
    from ctypes import POINTER, WinDLL, byref, c_size_t, wintypes
    need(os.name == "nt" and sys.version_info[:3] == (3, 10, 13)
         and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and all(os.environ.get(name) == "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
         "Exact isolated Windows CPU worker and thread limits")
    need({name: version(name) for name in ("numpy", "scipy", "pycocotools")} ==
         {"numpy": "1.26.4", "scipy": "1.15.3", "pycocotools": "2.0.11"}, "Exact numerical CPU packages")
    k = WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetPriorityClass.argtypes = (wintypes.HANDLE,)
    k.GetPriorityClass.restype = wintypes.DWORD
    k.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, POINTER(c_size_t), POINTER(c_size_t))
    k.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, c_size_t)
    k.SetProcessAffinityMask.restype = wintypes.BOOL
    # Current process pseudo-handle is -1; use an explicit handle-sized value.
    handle = k.GetCurrentProcess()
    k.GetProcessAffinityMask.restype = wintypes.BOOL
    a, system = c_size_t(), c_size_t()
    need(k.GetProcessAffinityMask(wintypes.HANDLE(handle), byref(a), byref(system)) and a.value,
         "Original Windows worker CPU affinity query failed")
    chosen = a.value & -a.value
    need(k.SetProcessAffinityMask(handle, chosen)
         and k.GetProcessAffinityMask(wintypes.HANDLE(handle), byref(a), byref(system)) and a.value == chosen
         and k.GetPriorityClass(handle) == 0x4000, "One logical CPU and below-normal process priority required")
    return dict(logical_cpus=1, priority="below-normal", thread_limits=1, CUDA=False)


def admit_closed(args, binding):
    score = args.score_root
    primary = document(score / "RESULT_SAFE.json", binding["primary_result_sha256"])
    end = document(score / "ORIGINAL_SCORE_TERMINAL_PRIVATE.json", binding["original_score_terminal_sha256"])
    score_wait = document(args.score_wait, binding["original_score_wait_sha256"])
    gpu_wait = document(args.export_wait, binding["original_export_wait_sha256"])
    marker = dict(status=PRIMARY_STATUS, result_sha256=binding["primary_result_sha256"])
    need(score_wait["original_score_result_sha256"] == binding["primary_result_sha256"]
         and score_wait["original_inner_score_terminal_sha256"] == binding["original_score_terminal_sha256"]
         and score_wait["original_cuda_export_owner_wait_sha256"] == binding["original_export_wait_sha256"]
         and score_wait["exact_tool_wait"]["exit_code"] == end["original_os_exit"] == 0
         and end["original_os_wait"] is True
         and end["original_export_owner_wait_sha256"] == binding["original_export_wait_sha256"]
         and json.loads(score_wait["exact_tool_wait"]["output"].splitlines()[-1]) == marker
         and Path(end["stdout_ref"]["path"]) == score / "ORIGINAL_SCORE.stdout"
         and json.loads(pinned(score / "ORIGINAL_SCORE.stdout", end["stdout_ref"]["sha256"]).splitlines()[-1]) == marker,
         "Exact original independent primary score closure required")
    gpu_events = [json.loads(line) for line in gpu_wait["combined_stdout"].splitlines() if line]
    need(len(gpu_events) == 5 and [row["stage"] for row in gpu_events[:4]] ==
         ["fit_smoke", "image_stage", "native", "center"] and gpu_wait["exact_tool_wait"]["exit_code"] == 0
         and gpu_events[-1]["status"] == EXPORT_STATUS and gpu_events[-1]["resource_closed"] is True,
         "Exact original CUDA owner natural close and all original export stages required")
    summary = document(args.export_root / "ORIGINAL_OWNER_SUMMARY_PRIVATE.json", gpu_events[-1]["owner_summary_sha256"])
    need(summary["status"] == EXPORT_STATUS and summary["resource_closed"] is True and summary["failure"] is None
         and summary["paired_receipt_sha256"] == primary["paired_receipt_sha256"] == binding["training_pair_sha256"]
         and primary["status"] == PRIMARY_STATUS and primary["images"] == 705
         and primary["original_annotations"] == 1782 and primary["empty_sensitivity_exclusions"] == 4
         and primary["original_gt_sha256"] == binding["original_gt_sha256"]
         and primary["image_manifest_sha256"] == binding["dev_manifest_sha256"]
         and primary["official_TEST_opened"] is primary["official_TEST_AP"] is False,
         "Exact same closed original DEV-only pair, GT and primary score")
    for arm, marker_index in (("native", 2), ("center", 3)):
        ref = document(score / (arm + "_ORIGINAL_CLOSED_PRIVATE.json"), binding["original_" + arm + "_closure_sha256"])
        stage = summary["stages"][arm]
        need(ref["arm"] == arm and ref["original_child_terminal_ref"] == stage["terminal"]
             and ref["original_external_wait_sha256"] == binding["original_export_wait_sha256"]
             and ref["exact_tool_wait"]["exit_code"] == 0
             and primary["arms"][arm]["native_rows"] == 70500
             and primary["arms"][arm]["export_report_sha256"] == stage["report"]["sha256"] ==
                 gpu_events[marker_index]["report_sha256"]
             and primary["arms"][arm]["predictions_sha256"] == binding[arm + "_stream_sha256"],
             "Exact original per-arm candidate and closure binding")
    return primary


def exact_maximum(iou, threshold):
    """Return GT membership of one maximum, without using candidate scores."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    covered = np.zeros(iou.shape[1], dtype=bool)
    if len(iou) and iou.shape[1]:
        edge = iou >= threshold
        match = maximum_bipartite_matching(csr_matrix(edge), perm_type="column")
        rows = np.flatnonzero(match >= 0)
        gt = match[rows]
        need(len(gt) == len(set(gt.tolist())) and edge[rows, gt].all(), "Exact threshold one-to-one matching")
        covered[gt] = True
    return covered


def coverage(gt, groups, counts, empty_ids):
    import numpy as np
    from pycocotools import mask
    from pycocotools.coco import COCO
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    empty, excluded, size, boundary = set(), Counter(crowd=0, ignore=0), Counter(), Counter()
    result = {arm: {f"{t:.2f}": Counter() for t in THRESHOLDS} for arm in ARMS}
    for i, image in enumerate(gt["images"]):
        objects, object_sizes = [], []
        for item in sorted(coco.imgToAnns[image["id"]], key=lambda x: x["id"]):
            rle = coco.annToRLE(item)
            if mask.area(rle) == 0:
                empty.add(item["id"])
                continue
            if item.get("iscrowd", 0) or item.get("ignore", 0):
                excluded.update(crowd=bool(item.get("iscrowd", 0)), ignore=bool(item.get("ignore", 0)))
                continue
            area = item["area"]
            need(type(area) in (float, int) and 0 <= area < float("inf"), "Exact finite COCO annotation area")
            bucket = "small" if area < 1024 else ("medium" if area < 9216 else "large")
            size[bucket] += 1
            if area in (1024, 9216):
                boundary[str(int(area))] += 1
            objects.append(rle)
            object_sizes.append(bucket)
        object_sizes = np.asarray(object_sizes, dtype="U6")
        for arm in ARMS:
            candidates = sorted((row["segmentation"] for row in groups[arm][i][:counts[i]]),
                                key=lambda rle: rle["counts"])
            matrix = (mask.iou(candidates, objects, [0] * len(objects)) if candidates and objects
                      else np.zeros((len(candidates), len(objects))))
            need(matrix.shape == (counts[i], len(objects)) and np.isfinite(matrix).all()
                 and (matrix >= 0).all() and (matrix <= 1).all(), "Exact original-raster IoU matrix")
            for threshold in THRESHOLDS:
                covered = exact_maximum(matrix, threshold)
                row = result[arm][f"{threshold:.2f}"]
                row.update(all=int(covered.sum()), **{area: int(covered[object_sizes == area].sum())
                                                       for area in AREAS})
    need(empty == empty_ids and len(empty) == 4 and sum(excluded.values()) == 0 and sum(size.values()) == 1778,
         "Exact only four prelocked empty-raster exclusions")
    denominators = dict(all=1778, **{area: size[area] for area in AREAS})
    for arm in ARMS:
        for row in result[arm].values():
            need(row["all"] == sum(row[area] for area in AREAS), "One global assignment must reconcile size slices")
    return dict(eligible_gt=1778, excluded_empty=len(empty), excluded_crowd=excluded["crowd"],
                excluded_ignore=excluded["ignore"], size_denominator=denominators,
                exact_area_boundaries={str(x): boundary[str(x)] for x in (1024, 9216)},
                counts=result,
                center_minus_native={threshold: {key: result["center"][threshold][key] -
                    result["native"][threshold][key] for key in denominators}
                    for threshold in result["native"]})


def score_cells(gt, selected, template):
    adapter = template.load_official_adapter()
    image_ids = [item["id"] for item in gt["images"]]
    cells, layouts = {}, {}
    for geometry in ARMS:
        cells[geometry] = {}
        reference = None
        for donor in ARMS:
            values = []

            def capture(backend, coco, predictions, ids):
                stats = adapter.evaluate_population(backend, coco, predictions, ids)
                values.append(stats)
                return stats

            rows = template.score_template(selected[geometry], selected[donor])
            metrics, matches, layout = template.evaluate_cell(gt, rows, image_ids,
                                                               SimpleNamespace(evaluate_population=capture))
            all12, = values
            metrics.update(APm=all12[4], APl=all12[5], AR100=all12[8], all12=all12)
            cells[geometry][donor] = metrics
            need(donor not in layouts or layouts[donor] == layout,
                 "Each exact original score donor must preserve COCO rank layout across both candidates")
            layouts[donor] = layout
            if reference is None:
                reference = matches
            else:
                template.require_same_matching(reference, matches)
                need(all(metrics[k] == cells[geometry]["native"][k] for k in ("recall75", "recall90", "AR100")),
                     "Within-image matches and recall invariant across same-geometry score donors")
    a, b = (cells[arm] for arm in ARMS)
    contrast_pairs = dict(
        score_on_native_masks=(a["native"], a["center"]),
        score_on_center_masks=(b["native"], b["center"]),
        center_masks_with_native_scores=(a["native"], b["native"]),
        center_masks_with_center_scores=(a["center"], b["center"]),
    )
    return dict(cells=cells, units="fraction", donor_rank_layout_exact=True, donor_within_image_matching_exact=True,
                contrasts_pp={name: {k: 100 * (right[k] - left[k]) if min(left[k], right[k]) >= 0 else None
                                       for k in METRICS} for name, (left, right) in contrast_pairs.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "score-source-dir", "score-root", "export-root", "score-wait", "export-wait", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    resource = admit_worker()
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink()
         and all(not args.output.is_relative_to(path) for path in (args.score_root, args.export_root)),
         "Fresh aggregate output outside original artifacts required")
    protocol = document(args.protocol, PROTOCOL_SHA)
    binding = protocol["input_binding"]
    primary = admit_closed(args, binding)
    for filename, digest in PINNED_SCORE_SOURCE.items():
        pinned(args.score_source_dir / filename, digest)
    template_root = args.code / TEMPLATE_CODE
    for filename, digest in PINNED_TEMPLATE_SOURCE.items():
        pinned(template_root / filename, digest)
    sys.path[:0] = [str(template_root), str(args.code)]
    template = import_module("dev705_score_template")
    score = load_original(args.score_source_dir / "author_dev_score.py",
                          PINNED_SCORE_SOURCE["author_dev_score.py"], "_closed_p392_original_score")
    evaluator = score.evaluator()
    need(evaluator.FILES["DEV_ORIGINAL_GT_PRIVATE.json"] == binding["original_gt_sha256"]
         and evaluator.FILES["DEV_EMPTY_RASTERS_PRIVATE.json"] == binding["prelocked_empty_raster_findings_sha256"]
         and evaluator.FILES["DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"] == binding["dev_manifest_sha256"],
         "Exact fixed original P392 DEV manifest/GT/raster findings")
    manifest = evaluator.qualified_manifest()
    shapes = {r["image_id"]: (r["height"], r["width"]) for r in manifest}
    need(len(shapes) == 705 and set(shapes.values()) == {(800, 800)}, "Exact original DEV canvas")
    predictions, natural = {}, {}
    for arm in ARMS:
        report, rows, report_sha = score.read_closed_arm(arm, args.export_root / arm / "OUTPUT",
            args.score_root / (arm + "_ORIGINAL_CLOSED_PRIVATE.json"),
            binding["original_" + arm + "_closure_sha256"], shapes, binding["dev_manifest_sha256"])
        need(len(rows) == 70500 and report["predictions_sha256"] == binding[arm + "_stream_sha256"]
             and report_sha == primary["arms"][arm]["export_report_sha256"], "Original ordered 70,500 native rows")
        predictions[arm] = rows
        natural[arm] = report["nonempty_masks"]
    gt = evaluator.dev_file("DEV_ORIGINAL_GT_PRIVATE.json")
    findings = evaluator.dev_file("DEV_EMPTY_RASTERS_PRIVATE.json")
    empty_ids = {row["annotation_id"] for row in findings if row["kind"] == "EMPTY_COCO_RASTER"}
    need(len(gt["annotations"]) == 1782 and len(empty_ids) == len(findings) == 4
         and [im["id"] for im in gt["images"]] == list(shapes)
         and [cat["id"] for cat in gt["categories"]] == [1], "Exact original GT population/order")
    groups = {arm: template.nonempty_by_image(predictions[arm], gt["images"]) for arm in ARMS}
    need({arm: sum(map(len, groups[arm])) for arm in ARMS} == natural, "Original native export nonempty census")
    selected, counts = template.prefix_rows(*(groups[arm] for arm in ARMS))
    ties = {}
    for arm in ARMS:
        tie_counts = Counter(row["score"] for row in selected[arm])
        ties[arm] = dict(within_image_equal_adjacent=sum(
            a["image_id"] == b["image_id"] and a["score"] == b["score"]
            for a, b in zip(selected[arm], selected[arm][1:])),
            score_values_shared_by_multiple_rows=sum(value > 1 for value in tie_counts.values()),
            rows_sharing_some_score=sum(value for value in tie_counts.values() if value > 1))
    result = dict(status=RESULT_STATUS, scientific_protocol_sha256=PROTOCOL_SHA, input_binding=binding,
                  scope="fixed equal-nonempty native score-selected shortlist; conditional comparisons only",
                  images=705, natural_rows_per_arm=70500, natural_nonempty=natural,
                  natural_empty={arm: 70500 - n for arm, n in natural.items()},
                  equal_nonempty_rows_per_arm=sum(counts), images_with_zero_shortlist=sum(k == 0 for k in counts),
                  maximum_shortlist_per_image=max(counts), donor_exact_score_ties=ties,
                  endpoints={arm: {key: primary["arms"][arm][key] for key in ("primary", "sensitivity")} for arm in ARMS},
                  coverage=coverage(gt, groups, counts, empty_ids), templates=score_cells(gt, selected, template),
                  runtime=resource, wall_seconds=perf_counter() - started, model_forward_calls=0,
                  optimizer_updates=0, official_TEST_opened=False, official_TEST_AP=False)
    need("torch" not in sys.modules and "paddle" not in sys.modules, "No model runtime loaded by CPU diagnostic")
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        need(stream.write(payload) == len(payload), "Complete fresh safe result required")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=RESULT_STATUS, result_sha256=sha256(payload).hexdigest()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

