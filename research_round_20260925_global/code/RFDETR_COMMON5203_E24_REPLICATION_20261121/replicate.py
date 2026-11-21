"""Own the single center/common-retention 5203 replication on the historical Windows runtime."""
import argparse
from hashlib import sha256
from io import BytesIO
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace


P238 = ("P238_ROOT_HRSID_COMMON_FIT_SHORT_PRIVATE_20261110.py",
        "e373975394c2602f13afb3f67c974833673312b7068aceec586e6c3cfeec208f")
P240 = ("P240_ROOT_HRSID_COMMON_FIT_E24_ACCEPT_PRIVATE_20261110.py",
        "41565bf16ec5aae27d3e52af86bfb92eaeb1659738d96947cfaedb5115e47e95")
WORKER = "b52f8dc7a8d5bd06c0bda3eb839f31cbed3a036befa709c8fc718c47706a959b"
PAIR = "2eaf88d006f0a6db54fe965230954fc2d33ff2117f473e754373b7568c365691"
LEDGER = "b35188724345316284ee9f4a03a7c1da8d8c4c4e9b4d7727c13baaea5c9a8143"
ACCEPT = "89b98e67a9b7596cae58ee14dc9aa028ac2c35109bdf3fd633915f83eec706b6"
SELF = Path(__file__).resolve()
PRELOCK = SELF.with_name("PRELOCK.json")
QUALIFIED = "SHORT_PAIR_ACCEPTED_SAFE.json"
COMPLETE = "CENTER_E24_ACCEPTED_PRIVATE.json"
STAGES = (("qualification", "native"), ("qualification", "center"), ("e24", "center"))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def pinned(path, expected, name):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Source pin differs: " + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def worker_bytes(original):
    old, new = b"TRAINING_SEED = 5202", b"TRAINING_SEED = 5203"
    need(digest(original) == WORKER and original.count(old) == 1, "Expected original common5202 trainer")
    return original.replace(old, new)


def context(work, *, needs_containment):
    root = work.parent
    short = pinned(root / P238[0], P238[1], "_common5203_inherited_owner")
    need(root == short.ROOT and work.name.startswith("RFDETR_COMMON5203_E24_")
         and work.resolve() == work, "Expected fresh dedicated original-runtime output root")
    base = pinned(short.REPLAY / "common_fit_train.py", WORKER, "_common5203_historical_worker")
    code = short.CODE
    locations = {base.NATIVE: root / "HRSID_FIT_COCO_ONLY_NATIVE_E24_EMPTY_MASK_FILTER_CANDIDATE_20261109_01" / base.NATIVE,
                 "phase_train.py": code / "HRSID_FIT_RESIZE_PHASE_TRAIN_20261109/phase_train.py",
                 "phase_patch.py": code / "HRSID_FIT_RESIZE_PHASE_TRAIN_20261109/phase_patch.py",
                 "phase_repeat_train.py": code / "HRSID_MASK_PHASE_REPEAT_TRAIN_20261109/phase_repeat_train.py"}
    paths = {name: locations.get(name, short.REPLAY / name) for name in base.SOURCES}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: digest(data) for name, data in raw.items()}
    need(digest(json.dumps(hashes, separators=(",", ":"), sort_keys=True).encode()) == short.MANIFEST,
         "Historical nine-source manifest differs")
    qualifier = (code / "HRSID_MASK_PHASE_REPEAT_QUALIFICATION_20261109/repeat_qualification.py").read_bytes()
    need(digest(qualifier) == "ae9a915b56aa27ec3376877c38a399cd5f722ee4e5d47f61aa74c32e39862e5f",
         "Historical independent qualification validator differs")
    owner = pinned(root / "P189_ROOT_HRSID_FILTERED_E24_ACCEPT_AFTER_CLOSURE_PRIVATE_20261109.py",
                   "0467c76ceee4a3976e5557a3e28b21ec148bee1f6f9fc095888aa709a609bdef", "_common5203_stable_io")
    runtime = None
    if needs_containment:
        owner = pinned(root / "P206_ROOT_HRSID_MASK_PHASE_DEV705_COORDINATOR_PRIVATE_20261109.py",
                       "55cb9a4a18a6b58fa5eac28a259422c058831523ae10ca0448818ee856ef0b39", "_common5203_windows_job")
        runtime = pinned(root / "RFDETR_FIT_TEACHER_BOUNDARY_CANDIDATE_20261108_01/root_fit_teacher_boundary.py",
                         "b5f03331136321715870e1f64b95d670e821ae5a16c90e5480ff1935b9af03c6", "_common5203_runtime")
        need(runtime.RUNTIME.resolve(strict=True) == base.PYTHON.resolve(strict=True), "Exact original GPU runtime")
    derived = worker_bytes(raw[base.SELF])
    worker_path = work / "SOURCE_PRIVATE" / base.SELF
    return SimpleNamespace(work=work, short=short, owner=owner, runtime=runtime, base=base,
                           qualifier=qualifier, paths=paths, raw=raw, derived=derived, worker_path=worker_path,
                           hashes={**hashes, base.SELF: digest(derived)})


def verify_git(release, commit):
    need(re.fullmatch("[0-9a-f]{40}", commit) is not None, "Full prospective Git commit required")
    found = subprocess.check_output(["git", "-C", str(release), "rev-parse", "--verify", commit + "^{commit}"],
                                    text=True).strip()
    need(found == commit, "Unavailable prospective protocol commit")
    blobs = set(subprocess.check_output(["git", "-C", str(release), "ls-tree", "-r", "--format=%(objectname)",
                                         commit], text=True).splitlines())
    for path in (SELF, PRELOCK):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        oid = subprocess.check_output(["git", "-C", str(release), "hash-object", "--stdin"], input=data)
        need(oid.decode().strip() in blobs, "Commit must contain exact " + path.name)


def header(ctx, commit):
    return dict(status="RFDETR_COMMON5203_CENTER_PREPARED", commit=commit, training_seed=5203,
                initialization_seed=5201, initial_expanded_sha256=ctx.base.INITIAL_EXPANDED_SHA,
                fit_index_sha256=ctx.base.FIT_INDEX_SHA, owner_sha256=digest(SELF.read_bytes()),
                prelock_sha256=digest(PRELOCK.read_bytes()), source_sha256s=ctx.hashes)


def prepare(ctx, commit):
    verify_git(ctx.short.ROOT / "github_release_20261011", commit)
    ctx.work.mkdir()
    ctx.worker_path.parent.mkdir()
    write(ctx.worker_path, ctx.derived)
    write(ctx.work / "PREPARED.json", canonical(header(ctx, commit)))


def bind(ctx):
    record = json.loads((ctx.work / "PREPARED.json").read_bytes())
    ctx.commit = record["commit"]
    need(record == header(ctx, ctx.commit), "Prepared sources or protocol changed")
    ctx.worker = pinned(ctx.worker_path, ctx.hashes[ctx.base.SELF], "common_fit_train")
    need(ctx.worker.TRAINING_SEED == 5203 and ctx.worker.INITIALIZATION_SEED == 5201, "Exact 5203/5201 split")
    ctx.refs = {name: ctx.owner.ref(ctx.worker_path if name == ctx.base.SELF else path)
                for name, path in ctx.paths.items()}
    need({k: r["sha256"] for k, r in ctx.refs.items()} == ctx.hashes, "Worker sources differ")
    replay = ctx.short.REPLAY
    ctx.pair = pinned(replay / "common_fit_pair_receipt.py", PAIR, "common_fit_pair_receipt")
    ctx.ledger = pinned(replay / "common_fit_e24_ledger.py", LEDGER, "common_fit_e24_ledger")
    ctx.acceptance = pinned(replay / "common_fit_e24_acceptance.py", ACCEPT, "common_fit_e24_acceptance")


def arm_paths(ctx, stage, mode):
    need((stage, mode) in STAGES, "Only the two short arms and the single center E24 are in scope")
    root = ctx.work / (stage + "_" + mode)
    return SimpleNamespace(root=root, output=root / "NATIVE_ACTUAL_PRIVATE", audit=root / "AUDIT_ACTUAL_PRIVATE",
                           control=root / "WORKER_CONTROL", outer=root / "OWNER_CONTROL", lock=root / "TRAINING_LOCK.json")


def lock_value(ctx, arm, stage, mode):
    qualified = None if stage == "qualification" else ctx.owner.ref(ctx.work / QUALIFIED)
    return dict(status=ctx.worker.LOCK_STATUS, training_protocol=ctx.commit, implementation_commit=ctx.commit,
                stage=stage, mode=mode, training_seed=5203, initialization_seed=5201,
                initial_expanded_sha256=ctx.worker.INITIAL_EXPANDED_SHA, fit_index_sha256=ctx.worker.FIT_INDEX_SHA,
                native_output=arm.output.as_posix(), audit_output=arm.audit.as_posix(), sources=ctx.refs,
                qualification_pair_ref=qualified)


def arguments(ctx, arm, stage, mode):
    need(arm.lock.read_bytes() == canonical(lock_value(ctx, arm, stage, mode)), "Per-arm training lock drift")
    return ["--native-source", ctx.refs[ctx.worker.NATIVE]["path"], "--output", str(arm.output),
            "--audit-output", str(arm.audit), "--training-lock", str(arm.lock),
            "--training-lock-sha256", ctx.owner.ref(arm.lock)["sha256"], "--protocol", ctx.commit,
            "--expected-initial-sha256", ctx.worker.INITIAL_EXPANDED_SHA, "--mode", mode, "--stage", stage,
            "--training-seed", "5203"]


def contained(ctx, script, argv, control, seconds, *, owner_process=False):
    owner, runtime = ctx.owner, ctx.runtime
    original = inspect.getsource(owner.wait_owner)
    need(original.count("> 7200:") == 1, "Pinned historical OS wait deadline")
    scope = dict(owner.__dict__, CONTROL=control, OWNER_PYTHON=owner.OWNER_PYTHON if owner_process else runtime.RUNTIME)
    child = None
    peak = 0
    def spawn(command, **kwargs):
        nonlocal child
        env = os.environ.copy() if owner_process else runtime.windows_runtime(control / "scratch")
        child = subprocess.Popen(command, env=env, **kwargs)
        return child
    scope["subprocess"] = SimpleNamespace(Popen=spawn, CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW,
                                          BELOW_NORMAL_PRIORITY_CLASS=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    exec(compile(original.replace("> 7200:", "> " + str(seconds) + ":"), str(owner.__file__), "exec"), scope)
    control.mkdir()
    (control / "scratch").mkdir()
    def resource_ok():
        nonlocal peak
        if child is not None and child.poll() is None:
            try:
                peak = max(peak, runtime.psutil.Process(child.pid).memory_info().rss)
            except runtime.psutil.NoSuchProcess:
                pass
        return runtime.psutil.virtual_memory().available >= runtime.RAM_FLOOR and peak <= runtime.RSS_CAP
    terminal = scope["wait_owner"](script, argv, resource_ok)
    write(control / "RESOURCE_GUARD.json", canonical(dict(peak_rss_bytes=peak, time_limit_seconds=seconds)))
    need(terminal["original_os_wait"] is True and terminal["original_os_exit"] == 0
         and terminal["stop_reason"] is None and terminal["job_assigned"] is True
         and terminal["job_active_before_drain"] == terminal["job_active_after_wait"] == 0,
         "Training process ended without a successful full OS wait; see " + str(control))


def p240(ctx):
    return pinned(ctx.work.parent / P240[0], P240[1], "_common5203_original_accept_io")


def original_closure(ctx, arm):
    helper = p240(ctx)
    io = pinned(helper.P220, helper.P220_SHA, "_common5203_checkpoint_io")
    records = []
    for control, expected_sha in ((arm.outer, digest(SELF.read_bytes())), (arm.control, ctx.hashes[ctx.base.SELF])):
        term_ref = ctx.owner.ref(control / "OWNER_TERMINAL_PRIVATE.json")
        birth = json.loads((control / "OWNER_BIRTH_PRIVATE.json").read_bytes())
        need(not ctx.owner.live(birth), "Original process not closed")
        terminal, stdout = helper.verified_terminal(control, term_ref, birth, expected_sha, io)
        records.append((terminal, term_ref))
    (outer, outer_ref), (inner, inner_ref) = records
    need(outer["owner"]["pid"] != inner["owner"]["pid"]
         and outer["owner"]["creation_time"] <= inner["owner"]["creation_time"], "Distinct parent and worker")
    closure = dict(original_os_wait=True, original_os_exit=0, worker_closed=True, resource_stop=False,
                   stdout_sha256=inner["stdout_ref"]["sha256"], terminal_sha256=inner_ref["sha256"])
    return closure, stdout, outer_ref, io, helper


def run_arm(ctx, stage, mode):
    arm = arm_paths(ctx, stage, mode)
    arm.root.mkdir()
    write(arm.lock, canonical(lock_value(ctx, arm, stage, mode)))
    seconds = 300 if stage == "qualification" else 36000
    contained(ctx, SELF, ["inner", "--work", str(ctx.work), "--stage", stage, "--mode", mode], arm.outer,
              seconds + 120, owner_process=True)
    original_closure(ctx, arm)


def qualify(ctx):
    for mode in ("native", "center"):
        run_arm(ctx, "qualification", mode)
    write(ctx.work / QUALIFIED, short_readback(ctx))


def short_readback(ctx):
    rows = {}
    for mode in ("native", "center"):
        arm = arm_paths(ctx, "qualification", mode)
        arguments(ctx, arm, "qualification", mode)
        closure, stdout, _, io, _ = original_closure(ctx, arm)
        stable = io.stable_read
        rows[mode] = ctx.pair.ClosedArm(closure, stable(arm.audit / "COMMON_FIT_TRAIN_SAFE.json", 65536),
            stable(arm.audit / "COMMON_FIT_VISITS_PRIVATE.jsonl", 256 * 1024),
            stable(arm.output / "QUALIFICATION_SAFE.json", 65536),
            stable(arm.output / "QUALIFICATION_VISITS_PRIVATE.json", 1024**2),
            stable(arm.output / "REPEAT_SEED_WITNESS_SAFE.json", 16384), stdout)
    value = ctx.pair.accept_short_pair(training_protocol=ctx.commit, implementation_commit=ctx.commit,
                source_sha256s=ctx.hashes, native_qualifier_source=ctx.qualifier, **rows)
    return ctx.pair.encode_receipt(value)


def single_ledger(api, safe, common, visits):
    """Cross-check each center/common target against its own original consumed native batches."""
    api.verify_safe_scope(safe, "center", 70560, 17640)
    common = api.Readback(common, "common", api.COMMON_LINE_LIMIT, api.COMMON_FILE_LIMIT)
    native = api.Readback(BytesIO(visits), "training", api.NATIVE_LINE_LIMIT, api.NATIVE_FILE_LIMIT)
    ordinal = targets = empty = changed = 0
    for epoch in range(1, api.EPOCHS + 1):
        for batch, (image_id, count) in enumerate(api.native_epoch(native.line(), epoch), 1):
            ordinal += 1
            row = api.common_visit(common.line(), epoch, batch, ordinal)
            need((row["image_id"], row["common_nonempty"]) == (image_id, count), "Wrong training row or count")
            targets += count
            empty += count == 0
            changed += row["common_changed_masks"] > 0
    need(ordinal == 70560 and changed > 0 and common.finish() == safe["consumed_ledger_sha256"]
         and common.bytes == safe["consumed_ledger_bytes"] and safe["ledger"] == dict(
             consumed_visits=ordinal, fetched_visits=ordinal, prefetched_unconsumed=0, unconsumed_prefetch_sha256=None,
             common_targets_total=targets, common_empty_visits=empty, changed_mask_visits=changed),
         "Original common ledger summary mismatch")
    return native.finish()


def accept(ctx):
    arm = arm_paths(ctx, "e24", "center")
    arguments(ctx, arm, "e24", "center")
    qualification = short_readback(ctx)
    need((ctx.work / QUALIFIED).read_bytes() == qualification, "Closed short pair receipt changed")
    closure, stdout, outer, io, helper = original_closure(ctx, arm)
    need(Path(sys.executable).resolve() == ctx.worker.PYTHON.resolve(), "Original CPU readback runtime required")
    io.check_runtime()
    check = ctx.acceptance
    identity = check.short_pair_identity(ctx.commit, ctx.commit, ctx.hashes, qualification, digest(qualification))
    legacy = check.scoped_repeat(ctx.commit, ctx.hashes[ctx.base.SELF])
    native, audit = arm.output, arm.audit
    stable = io.stable_read
    archives = {f"checkpoint_{epoch - 1}.ckpt" for epoch in (1, 8, 24)}
    need({p.name for p in native.glob("*.ckpt")} == archives, "Exact three fixed endpoint archives required")
    def read_archive(row):
        name = row["archive_name"]
        need(name in archives, "Unexpected archive filename")
        return legacy.read_closed_checkpoint(native / name, row)
    evidence = check.ClosedE24Arm(dict(closure, owner_closed=True), stable(audit / "COMMON_FIT_TRAIN_SAFE.json", 65536),
        stable(native / "NATIVE_HRSID_COCO_FIT_E24_SAFE.json", 65536), stable(native / "EPOCH_AUDIT_SAFE.jsonl", 1024**2),
        stable(native / "FIT_VISITS_PRIVATE.jsonl", 8 * 1024**2), stdout,
        stable(native / "PHASE_PROVENANCE_SAFE.json", 65536), stable(native / "REPEAT_SEED_WITNESS_SAFE.json", 8192),
        read_archive)
    safe = check.checked_arm_audit(evidence, "center", identity, legacy)
    with helper.common_stream(audit / "COMMON_FIT_VISITS_PRIVATE.jsonl", ctx.ledger.COMMON_FILE_LIMIT) as stream:
        visit_sha = single_ledger(ctx.ledger, safe, stream, evidence.visits)
    receipt = check.accept_arm(evidence, "center", safe, legacy, helper.fit_ids(io), visit_sha)
    need(receipt["checkpoint_sha256"] != "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11",
         "Returned original comparator checkpoint")
    write(ctx.work / COMPLETE, canonical(dict(status="COMMON5203_SINGLE_CENTER_E24_ACCEPTED", arm=receipt,
          original_outer_terminal=outer, source_sha256s=ctx.hashes, FIT=True, DEV=False, TEST=False, AP=False)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("action", choices=("prepare", "qualify", "train", "accept", "inner"))
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--stage", choices=("qualification", "e24"))
    parser.add_argument("--mode", choices=("native", "center"))
    args = parser.parse_args(argv)
    ctx = context(args.work, needs_containment=args.action in ("qualify", "train", "inner"))
    if args.action == "prepare":
        need(args.commit is not None and args.stage is args.mode is None, "Prepare requires the prospective commit")
        prepare(ctx, args.commit)
    else:
        need(args.commit is None, "Existing immutable preparation supplies the commit")
        bind(ctx)
        if args.action == "inner":
            arm = arm_paths(ctx, args.stage, args.mode)
            ack = arm.outer / "JOB_ATTACHED.txt"
            need(ack.read_text(encoding="ascii") == digest(SELF.read_bytes())
                 and ctx.owner.birth(os.getpid()) == json.loads((arm.outer / "OWNER_BIRTH_PRIVATE.json").read_bytes()),
                 "Only the original contained owner may launch a worker")
            contained(ctx, ctx.worker_path, arguments(ctx, arm, args.stage, args.mode), arm.control,
                      300 if args.stage == "qualification" else 36000)
        else:
            need(args.stage is args.mode is None, "Top-level actions have fixed scientific scope")
            if args.action == "qualify":
                qualify(ctx)
            elif args.action == "train":
                need(short_readback(ctx) == (ctx.work / QUALIFIED).read_bytes(), "Independent qualification must agree")
                run_arm(ctx, "e24", "center")
            else:
                accept(ctx)
    print(json.dumps({"action": args.action, "original_exit": 0, "work": str(ctx.work)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
