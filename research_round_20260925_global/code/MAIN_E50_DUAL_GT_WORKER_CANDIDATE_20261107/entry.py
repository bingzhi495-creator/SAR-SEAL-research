"""Run the fixed E50 dual-GT comparison from admitted original CPU inputs."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_exact(reference):
    path = Path(reference["path"])
    need(
        set(reference) == {"path", "bytes", "sha256"} and path.is_absolute()
        and path.resolve(strict=True) == path and not path.is_symlink() and path.is_file()
        and type(reference["bytes"]) is int and 0 <= reference["bytes"] <= 64 * 1024**2,
        "Exact canonical input reference",
    )
    before = path.stat()
    need(before.st_size == reference["bytes"], "Fixed original file size")
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    need(len(raw) == reference["bytes"] and digest(raw) == reference["sha256"]
         and identity(before) == identity(after), "Stable exact original input bytes")
    return raw


def load_module(reference, alias):
    raw = read_exact(reference)
    module = types.ModuleType(alias)
    module.__file__ = reference["path"]
    sys.modules[alias] = module
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


class Control:
    def __init__(self, refs, psutil):
        self.refs = {}
        for r in refs:
            need(self.refs.get(r["path"], r) == r, "No ambiguous original role")
            self.refs[r["path"]] = r
        self.psutil = psutil
        self.process = psutil.Process()
        self.started = time.monotonic()
        self.peak_rss = 0
        self.min_available = None

    def check(self):
        rss = self.process.memory_info().rss
        available = self.psutil.virtual_memory().available
        self.peak_rss = max(self.peak_rss, rss)
        self.min_available = available if self.min_available is None else min(available, self.min_available)
        need(time.monotonic() - self.started < 290 and self.process.cpu_affinity() == [0, 1]
             and self.peak_rss <= 8 * 1024**3 and self.min_available >= 16 * 1024**3,
             "Declared CPU, elapsed, RSS and RAM guard")

    def closed(self, reference):
        self.check()
        need(self.refs.get(reference["path"]) == reference, "Explicit admitted source/input only")
        read_exact(reference)
        return Path(reference["path"])


def create_coco(loader, document):
    coco = loader.COCO()
    coco.dataset = deepcopy(document)
    coco.createIndex()
    return coco


def raster_audit(control, runtime, g, integer, floating):
    np = runtime["numpy"]
    ci, cf = (create_coco(runtime["coco_loader"], x) for x in (integer, floating))
    rows = []
    for aid, ai in ci.anns.items():
        control.check()
        af = cf.anns[aid]
        img = ci.imgs[ai["image_id"]]
        mi, mf = (runtime["mask_api"].decode(coco.annToRLE(ann)) != 0
                  for coco, ann in ((ci, ai), (cf, af)))
        need(mi.shape == mf.shape == (img["height"], img["width"]), "Original-grid rasterization")
        counts = g.raster_counts(int(np.count_nonzero(mi)), int(np.count_nonzero(mf)), int(np.count_nonzero(mi & mf)))
        need(counts["XOR_pixels"] == int(np.count_nonzero(mi != mf)), "Official mask XOR identity")
        rows.append(dict(counts, continuous_area_difference=af["area"] - ai["area"],
                         bbox_changed=af["bbox"] != ai["bbox"], polygon_changed=af["segmentation"] != ai["segmentation"]))
    return g.summarize_pixels(np, rows)


def run(spec_path, spec_sha):
    raw = spec_path.read_bytes()
    need(digest(raw) == spec_sha, "Exact frozen dual-GT spec bytes")
    spec = json.loads(raw)
    need(set(spec) == {"schema", "cpu_request", "sources", "candidate_float_ref",
         "accepted_integer_summary_ref", "expected_endpoint", "float_mapping_ref"}
         and spec["schema"] == "Root-E50-DEV187-dual-GT-executable-v1", "Exact locked scope")
    cpu = spec["cpu_request"]
    need(spec["sources"]["entry"]["path"] == str(Path(__file__)), "Exact this admitted entry")
    need(not any(n.split(".")[0] in {"numpy","pycocotools","torch"} for n in sys.modules), "Fresh CPU child")
    u70 = load_module(cpu["worker_source"], "_root_original_u70")
    u70.validate_request(cpu)
    psref = cpu["library_sources"]["psutil"]
    read_exact(psref)
    psutil = importlib.import_module("psutil")
    need(psutil.__file__ == psref["path"], "Exact original CPU resource module")
    refs = u70.direct_references(cpu) + list(spec["sources"].values())
    refs += [spec[k] for k in ("candidate_float_ref", "accepted_integer_summary_ref", "float_mapping_ref")]
    control = Control(refs, psutil)
    control.check()
    report = {"status": "READING_ORIGINAL_DEV_REFERENCES", "new_training": False, "TEST": False}
    output = spec_path.parent

    def save(value):
        tmp = output / "PROGRESS.next"
        with tmp.open("xb") as stream:
            stream.write(encoded(value))
        os.replace(tmp, output / "PROGRESS.json")

    save(report)
    def document(ref):
        return json.loads(u70.closed_bytes(control, ref))

    inputs = {k: document(ref) for k, ref in cpu["input_refs"].items()}
    shards = [document(ref) for ref in cpu["shard_refs"]]
    joined = u70.join_inputs(cpu, inputs, shards)
    original = document(spec["accepted_integer_summary_ref"])
    integer = inputs["DEV"]
    floating = document(spec["candidate_float_ref"])
    mapping = document(spec["float_mapping_ref"])
    need(original["endpoint"] == shards[0]["endpoint"] == spec["expected_endpoint"],
         "Same full original accepted E50 endpoint")
    need(len(integer["annotations"]) == len(floating["annotations"]) == 380 and len(joined["ids"]) == 187,
         "Complete two-GT population")
    g = load_module(spec["sources"]["gt"], "_root_original_gt_sensitivity")
    g.validate_GT_pair(integer, floating)
    g.validate_mapping_report(mapping, cpu["input_refs"]["DEV"], spec["candidate_float_ref"])
    candidate = load_module(spec["sources"]["eval"], "_root_frozen_e50_dual_gt")
    runtime = u70.import_runtime(control, cpu)
    np = runtime["numpy"]
    with u70.project_modules(control, cpu, runtime, {}) as (core, helpers):
        refs = {"native_official": joined["native"], "R": joined["strong"]}
        by_gt = {candidate.INTEGER: {}, candidate.FLOAT: {}}

        def controls(key, gt):
            for method, records in refs.items():
                control.check()
                coco = create_coco(runtime["coco_loader"], gt)
                result, by_iou, evaluator = helpers["eval_body"].official_metrics(
                    np, control, helpers["official"], runtime["cocoeval"].COCOeval, coco, records, joined["ids"])
                del evaluator
                candidate.check_metrics(result)
                if key == candidate.INTEGER:
                    need(result == joined["reference_COCO"][method] == original["results"][method]["COCO"],
                         "Whole original integer COCO12 reproduced for " + method)
                by_gt[key][method] = dict(COCO=result, per_IoU=by_iou)
                report["reference_cases"] = by_gt
                save(report)

        controls(candidate.INTEGER, integer)
        projected = {k: helpers[k] for k in ("population_contract", "eval_body", "official", "cocoeval", "population_metrics")}
        projected.update(population_core=core,coco_loader=runtime["coco_loader"],gt_sensitivity=g)
        paired = candidate.evaluate_dual_gt(np, control, shards=shards, native=joined["native"],
                    integer_gt=integer, floating_gt=floating, accepted_integer_summary=original,
                    modules=projected, report=report, save=save)
        controls(candidate.FLOAT, floating)
        report["GT_raster_sensitivity"] = raster_audit(control, runtime, g, integer, floating)
    control.check()
    need(u70.no_torch() and paired["integer_COCO12_reproduced"] is True, "Original CPU-only exact integer evaluation")
    report.update(status="EIGHT_CASES_COMPLETE_PENDING_ROOT_ACCEPTANCE", images=187, annotations=380,
                  primary_integer_unchanged=True, scope="Frozen candidates, integer and candidate float labels",
                  peak_sampled_worker_RSS_bytes=control.peak_rss, min_sampled_available_RAM_bytes=control.min_available)
    with (output/"RESULT.json").open("xb") as stream:
        stream.write(encoded(report))
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    need(len(sys.argv) == 3, "Original spec path and byte digest")
    run(Path(sys.argv[1]), sys.argv[2])

