"""Independently read two closed Windows SSDD E24 archives on isolated CPU."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType


ENTRY_SHA = "e29edeb0a3aff7c871123deee7c0c198ef5913fd33e7c016434a671d186796d3"
FORMAL_SHA = "1794f4da4e58dfd077fd02662ea6c2611fec30fb76fb5304e94979f14b739592"
PLAN_STATUS = "SSDD_WINDOWS_Q100_E24_CLOSED_CPU_READBACK_PLAN_V1"
MODES = ("native", "center")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_exact(path, digest):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical source required")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    require(0 < len(raw) <= 65536 and sha256(raw).hexdigest() == digest, "Immutable SSDD source differs")
    module = ModuleType("_ssdd_cpu_e24_" + path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def read_plan(entry, plan_path, plan_sha):
    plan = entry.document(entry.pinned_source(plan_path, plan_sha))
    entry.exact(plan, dict(status=PLAN_STATUS, FIT=True, DEV=False, TEST=False, AP=False),
                ("training_protocol", "source_manifest_sha256", "qualification_pair", "windows_scope_lock", "arms"),
                reason="Exact closed FIT-only readback plan required")
    require(set(plan["arms"]) == set(MODES) and entry.hex_id(plan["training_protocol"], 40)
            and plan["source_manifest_sha256"] == entry.WORKER_MANIFEST_SHA,
            "Exact two original E24 arms and stock source scope required")
    scope = entry.document(entry.read_ref(plan["windows_scope_lock"], entry.MIB,
                                          basename="WINDOWS_SCOPE_LOCK_PRIVATE.json"))
    require(scope["protocol"] == plan["training_protocol"]
            and scope["source_manifest_sha256"] == plan["source_manifest_sha256"]
            and scope["operation"] == "short" and scope["mode"] == "native",
            "Exact original E24 scope reference required")
    entry.verify_replay_receipt(scope)
    return plan, scope


def run(args):
    entry = load_exact(args.windows_entry, ENTRY_SHA)
    formal = load_exact(args.e24_entry, FORMAL_SHA)
    plan, scope = read_plan(entry, args.plan, args.plan_sha256)
    source = Path(scope["code"]) / "SSDD_MASK_PHASE_TRANSFER_20261110"
    entry.pinned_source(source / "ssdd_phase_qualification.py", formal.QUALIFIER_SHA)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    api = load_exact(source / "ssdd_phase_e24_acceptance.py", formal.ACCEPTANCE_SHA)
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh isolated CPU receipt output required")
    torch, _, _, deps = entry.admit_windows_runtime(dict(scope, operation="replay"))
    require(not torch.cuda.is_initialized(), "Independent E24 readback is CPU only")
    layout, ids = entry.read_fit_population(scope)
    qualification = plan["qualification_pair"]
    qualification_bytes = entry.read_ref(qualification, 64 * 1024, basename="SHORT_PAIR_ACCEPTED_SAFE.json")
    kwargs = dict(protocol=plan["training_protocol"], source_manifest_sha256=plan["source_manifest_sha256"],
                  expected_fit_ids=ids, qualification_bytes=qualification_bytes, qualification_sha256=qualification["sha256"])
    _, _, short = api.verify_scope(**kwargs)
    _, _, model, initial = deps.initialize_q100(layout, args.output, Path(scope["coco"]["path"]))
    initial_classifier = model.model.class_embed.weight.detach().contiguous().clone()
    del model
    inspect = api.make_checkpoint_reader(torch=torch, native_inspector=deps.e100.inspect_archive,
                                         e1_api=deps.e1, boundary_api=deps.boundary, initial_classifier=initial_classifier,
                                         initialization=initial, classifier_initial_sha256=short["classifier_initial_sha256"])
    arms, visits = {}, {}
    for mode in MODES:
        record = plan["arms"][mode]
        entry.exact(record, {}, ("closure", "report", "epoch", "visit", "stdout", "checkpoint_path"),
                    reason="Exact closed E24 arm byte references required")
        report = entry.read_ref(record["report"], 64 * 1024, basename=api.REPORT_NAME)
        require(0 < entry.document(report)["peak_allocated_bytes"] <= 6 * entry.GIB, "Exact original GPU allocator bound")
        epoch = entry.read_ref(record["epoch"], 2 * entry.MIB, basename=api.EPOCH_NAME)
        visits[mode] = entry.read_ref(record["visit"], 2 * entry.MIB, basename=api.VISIT_NAME)
        stdout = entry.read_ref(record["stdout"], 8 * entry.MIB, basename="OWNER.stdout")
        evidence = api.ClosedE24Arm(record["closure"], report, epoch, visits[mode], stdout)
        arms[mode], _ = api.validate_arm(evidence, mode=mode, expected_checkpoint_path=Path(record["checkpoint_path"]),
                                         checkpoint_reader=inspect, **kwargs)
    pair, digest, stats = api.create_pair_receipt(*(arms[mode] for mode in MODES), *(visits[mode] for mode in MODES), **kwargs)
    require(not torch.cuda.is_initialized(), "Independent E24 readback must leave CUDA uninitialized")
    args.output.mkdir()
    for name, raw in (*((mode.upper() + "_E24_ACCEPTED_SAFE.json", arms[mode]) for mode in MODES),
                      ("E24_PAIR_ACCEPTED_SAFE.json", pair)):
        with (args.output / name).open("xb") as stream:
            stream.write(raw)
    print(json.dumps(dict(status=api.PAIR_STATUS, pair_sha256=digest, **stats), sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("windows-entry", "e24-entry", "plan", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
