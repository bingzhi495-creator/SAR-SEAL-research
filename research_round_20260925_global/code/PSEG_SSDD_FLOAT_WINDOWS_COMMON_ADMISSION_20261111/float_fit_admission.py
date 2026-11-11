"""Qualify the official float SSDD FIT layout through the real Windows common-support loader."""

from argparse import ArgumentParser
from hashlib import sha256
from importlib import import_module
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile

P330 = "SSDD_DEV187_ALIAS_AND_COMMON_FIT_20261111/p330_common_preflight.py"
P330_SHA = "5c1aa7441375692c0ca724fdab7b54e74659af481e42d82b268deb1dd50ceb51"
FLOAT_SHA = "97162d23a33b9387fe0b47a117f7e5b1f9dde7471033f244ab336cee1ca9039b"
ROSTER_SHA = "315f7cfad9a3782865989ced212a6eeea1052e31251a01f24dbe2758bbc4f0e9"

def need(condition, message):
    if not condition:
        raise ValueError(message)

def validate_polygon_only(integer, floating):
    need(integer.keys() == floating.keys()
         and all(integer[key] == floating[key] for key in integer if key != "annotations"),
         "Polygon overlay changed the image population, category, or metadata")
    first = integer["annotations"]
    second = floating["annotations"]
    need(len(first) == len(second), "Polygon overlay changed annotation count")
    for before, after in zip(first, second, strict=True):
        need(before.keys() == after.keys()
             and all(before[key] == after[key] for key in before if key != "segmentation"),
             "Polygon overlay changed ordered annotation identity or a non-polygon field")

def verify_float_layout(archive, integer_layout, float_layout, ids, checks):
    train = float_layout / "train"
    need(list(float_layout.iterdir()) == [train] and train.resolve(strict=True) == train,
         "The float layout must contain only its own canonical training directory")
    raw = (train / "_annotations.coco.json").read_bytes()
    need(sha256(raw).hexdigest() == FLOAT_SHA, "Exact admitted official float FIT index required")
    integer = json.loads((integer_layout / "train/_annotations.coco.json").read_bytes())
    floating = json.loads(raw)
    validate_polygon_only(integer, floating)
    need(sorted(image["id"] for image in floating["images"]) == ids
         and len(ids) == 741 and len(floating["annotations"]) == 1661,
         "Exact frozen FIT741 image and annotation census required")
    with ZipFile(archive) as bundle:
        manifest = json.loads(checks.read_member(bundle, "manifest.json", checks.MANIFEST_BYTES, checks.MANIFEST_SHA))
    rows = []
    for source, image in zip(manifest["rows"], floating["images"], strict=True):
        name, ref = image["file_name"], source["original_image_ref"]
        need(checks.JPEG_NAME.fullmatch(name) and source["original_basename"] == name
             and source["image_id"] == image["id"]
             and checks.file_sha256(train / name, ref["bytes"]) == ref["sha256"],
             "Floating FIT layout must retain every original JPEG byte")
        rows.append([image["id"], name, ref["bytes"], ref["sha256"]])
    need({path.name for path in train.iterdir()} == {"_annotations.coco.json", *(r[1] for r in rows)}
         and sha256(checks.compact(rows)).hexdigest() == ROSTER_SHA, "Exact frozen image roster required")

def load_p330(code):
    source = code / P330
    raw = source.read_bytes()
    need(sha256(raw).hexdigest() == P330_SHA, "Exact published P330 common replay required")
    spec = importlib.util.spec_from_file_location("_float_fit_p330", source)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(source), "exec"), module.__dict__)
    return module

def run(args):
    start = perf_counter()
    p330 = load_p330(args.code_root)
    entry = p330.pinned(args.code_root, "entry")
    lock = dict(operation="replay", code=str(args.code_root), overlay=str(args.overlay),
                fit_layout=str(args.integer_layout), fit_archive={"path": str(args.fit_archive)},
                source_manifest_sha256=entry.WORKER_MANIFEST_SHA)
    torch, init, _, deps = entry.admit_windows_runtime(lock)
    integer_layout, ids = entry.read_fit_population(lock)
    checks = import_module("ssdd_fit_stage_readback")
    verify_float_layout(args.fit_archive, integer_layout, args.float_layout, ids, checks)
    need(not torch.cuda.is_initialized(), "CPU replay must never initialize CUDA")
    config = init.load_api(init.CONFIG, init.CONFIG_SHA)
    init.pinned_bytes(init.RECIPE, entry.RECIPE_SHA)
    model, train = config._load_configs(init.RECIPE, args.float_layout)
    deps.e1.native_recipe(model, train, import_module("rfdetr.config"))
    need(train.seed == p330.SEED and train.num_workers == 0, "Exact stock FIT recipe")
    prefix = p330.prospective_positions()
    positions = (*prefix, *(i for i in range(len(ids)) if i not in prefix))
    p330.VISITS = len(positions)
    historical = import_module("ssdd_phase_fit_replay")
    helpers = {key: p330.pinned(args.code_root, key) for key in ("trace_resize", "common_mask_supervision")}
    with patch.dict(sys.modules, helpers):
        paired = p330.pinned(args.code_root, "paired_fit_transform")
        binding = p330.pinned(args.code_root, "common_fit_binding")
        with patch.dict(sys.modules, {"paired_fit_transform": paired}):
            arms = {}
            for mode in ("native", "center"):
                arms[mode] = p330.run_arm(mode=mode, positions=positions, ids=ids, layout=args.float_layout,
                    model=model, train=train, deps=deps, torch=torch, binding=binding, paired=paired, historical=historical)
                print(json.dumps({"event": "FLOAT_WINDOWS_CPU_ARM_RETURNED", "mode": mode, "visits": len(positions)}),
                      flush=True)
    a, b = arms["native"], arms["center"]
    p330.compare_common(a, b)
    rows = [json.loads(line) for line in a["ledger"].splitlines()]
    need(len(rows) == len(ids) and len({row["image_id"] for row in rows}) == len(ids)
         and not torch.cuda.is_initialized() and torch.get_num_threads() == 1, "Closed full FIT CPU pair required")
    counters = {key: sum(row[key] for row in rows) for key in ("raw_rows", "native_present", "center_present",
        "native_nonempty", "center_nonempty", "common_nonempty", "native_only_nonempty",
        "center_only_nonempty", "common_changed_masks")}
    need(counters["common_nonempty"] == a["stats"]["common_targets_total"], "Exact common row totals")
    result = dict(status="SSDD_FLOAT_WINDOWS_FIXED_COMMON_CPU_POPULATION_ACTUAL", fit=True, dev=False, test=False,
        ap=False, model_forward_calls=0, optimizer_updates=0, initial_seed=p330.SEED, visits_per_arm=len(positions),
        original_empty_guard_removals=0, common_empty_visits=a["stats"]["common_empty_visits"],
        ordered_common_ledger_exact=True, direct_visits=len(positions)-a["crop"], crop_visits=a["crop"],
        changed_direct_visits=a["changed_direct"], changed_crop_visits=a["changed_crop"], counters=counters,
        common_ledger_sha256=sha256(a["ledger"].encode()).hexdigest(),
        selection_sha256=sha256(entry.compact([ids[i] for i in positions])).hexdigest(),
        selected_mask_stream_sha256={mode: arm["chosen_sha"] for mode, arm in arms.items()},
        float_index_sha256=FLOAT_SHA, content_roster_sha256=ROSTER_SHA, p330_sha256=P330_SHA,
        wall_seconds=perf_counter()-start)
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"status": result["status"], "sha256": sha256(payload).hexdigest()}), flush=True)

def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    for name in ("code-root", "overlay", "integer-layout", "float-layout", "fit-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("code_root", "overlay", "integer_layout", "float_layout", "fit_archive"):
        path = getattr(args, name)
        need(path.is_absolute() and path.resolve(strict=True) == path, "Canonical closed input paths required")
    need(args.output.is_absolute() and not args.output.exists()
         and all(not args.output.is_relative_to(path) for path in (args.integer_layout, args.float_layout)),
         "Fresh output outside both FIT layouts required")
    run(args)

if __name__ == "__main__":
    main()