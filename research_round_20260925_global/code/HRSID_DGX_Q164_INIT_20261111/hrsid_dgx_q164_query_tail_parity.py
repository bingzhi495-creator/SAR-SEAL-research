"""Check exact HRSID Q164 parity after one verified position tensor and query-feature tail."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from unittest.mock import patch


P291_SHA = "ca1cfc94ec954fadfc48aa164d527d0914bba80224be589fe8a9d1d33e62a04b"
PRIOR_SHA = "f3344ae6447ba0936c75076ebb932666d0097e732db39bb163212a166c61310e"
PRIOR_BYTES = 5712
POSITION_ONLY_Q164 = "ce447ecc2fde1badedf1f80b5ce46e143e1d5f3d0f764c078370d86bd797646a"
POSITION_ONLY_QUERY = "0d5dfc78015f268f451265867a436cdaa17bee3396f4490a56284d52c68ec1fb"
POSITION_ONLY_TAIL = "2613593c488853eb1d11ef631d5aab9bf7fd7d6c536d2d4d46a99c6151fb5e4c"
QUERY_PREFIX = "385ec4d4b17543cf37d81693d69eba156bce489370a8418e95a6bbb8817be576"
QUERY_KEY = "query_feat.weight"
TAIL_LOCATION = "query_feat.weight[13 groups, rows 100:164]"
TAIL_SHAPE = (13, 64, 256)
TAIL_BYTES = 851968
TAIL_NAME = "QUERY_FEAT_TAIL_LE_F32_PRIVATE.bin"
TAIL_RECEIPT = "QUERY_FEAT_TAIL_REFERENCE_SAFE.json"
TAIL_RECEIPT_STATUS = "HRSID_Q164_WINDOWS_QUERY_FEAT_TAIL_REFERENCE_V1"
PRELOCK = "P293_HRSID_DGX_Q164_POSITION_AND_QUERY_TAIL_PARITY_PRELOCK_V1"
PARITY = "P293_HRSID_DGX_Q164_POSITION_AND_QUERY_TAIL_PARITY_V1"
MISMATCH = "P293_HRSID_DGX_Q164_POSITION_AND_QUERY_TAIL_MISMATCH_V1"
SCOPE = dict(FIT=False, DEV=False, TEST=False, AP=False, training=False, cuda_initialized=False,
             model_forward_calls=0, optimizer_updates=0)
LIMIT = 64 * 1024


class TailMismatch(ValueError):
    def __init__(self, location, audit):
        super().__init__("Exact registered query-feature tail parity did not pass")
        self.location = location
        self.audit = audit


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def matches(value, fixed):
    return (type(value) is dict
            and all(key in value and type(value[key]) is type(item) and value[key] == item for key, item in fixed.items()))


def is_digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def pinned_p291(path):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical exact P291 source required")
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    require(0 < len(raw) <= LIMIT and sha256(raw).hexdigest() == P291_SHA, "Exact immutable P291 source required")
    module = ModuleType("_p293_exact_p291")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def read_json(p290, p291, path, digest):
    return json.loads(p290.pinned(path, digest, LIMIT), object_pairs_hook=p291.unique)


def verify_prelock(p290, p291, lock, prior_sha, tail_receipt_sha):
    fixed = dict(status=PRELOCK, p290_source_sha256=p291.P290_SHA, p291_source_sha256=P291_SHA,
                 position_file_sha256=p291.POSITION_FILE_SHA, prior_position_only_result_sha256=prior_sha,
                 windows_query_tail_receipt_sha256=tail_receipt_sha,
                 expected_position_only_q164_sha256=POSITION_ONLY_Q164, expected_canonical_q100_sha256=p290.Q100,
                 expected_canonical_q164_sha256=p290.Q164, FIT=False, DEV=False, TEST=False, AP=False)
    require(matches(lock, fixed) and set(lock) == set(fixed) | {"p293_source_sha256", "query_tail_file_sha256"}
            and is_digest(lock["p293_source_sha256"]) and is_digest(lock["query_tail_file_sha256"]),
            "Exact preregistered P293 sources, preceding position-only result and Windows tail required")
    p290.pinned(Path(__file__).resolve(strict=True), lock["p293_source_sha256"], LIMIT)


def verify_position_only(p290, p291, value):
    fixed = dict(status=p291.MISMATCH, localization=[TAIL_LOCATION], original_expansion_completed=True,
                 reference_tensor_substitution=True, canonical_q100_matches=True, canonical_q164_matches=False,
                 observed_q100_sha256=p290.Q100, observed_q164_sha256=POSITION_ONLY_Q164,
                 original_query_prefixes_match=True, zero_refpoint_tail=True, **SCOPE)
    require(matches(value, fixed), "Exact reproduced P291 position-only result and sole Q164 query tail mismatch required")
    audit = value["position_audit"]
    require(matches(audit, dict(raw_q100_sha256=p290.P242_RAW_Q100, raw_position_sha256=p290.P242_RAW_POSITION,
                                shadow_q100_sha256=p290.Q100, canonical_q100_sha256=p290.Q100,
                                canonical_position_sha256=p290.WINDOWS_POSITION,
                                changed_q100_tensor_names=[p290.POSITION], unchanged_non_position_tensors=544,
                                reference_tensor_substitution=True))
            and is_digest(audit["non_position_before_sha256"])
            and audit["non_position_before_sha256"] == audit["non_position_after_sha256"]
            and is_digest(value["non_query_stock_sha256"])
            and value["non_query_stock_sha256"] == value["non_query_expanded_sha256"],
            "Exact 544 unchanged Q100 tensors and original Q164 non-query state required")
    diagnostic = value["diagnostic"]
    require(set(diagnostic) == {"q100", "q164"} and all(matches(row, dict(tensors=545, position_sha256=p290.WINDOWS_POSITION))
                                                       for row in diagnostic.values()),
            "Exact original Q100/Q164 tensor census and canonical position required")
    query = diagnostic["q164"]["queries"][QUERY_KEY]
    require(matches(query, dict(sha256=POSITION_ONLY_QUERY, group_tail_sha256=POSITION_ONLY_TAIL,
                                group_prefix_sha256=QUERY_PREFIX, shape=[13 * 164, 256])),
            "Exact original DGX Q164 query-feature tensor, prefix and tail fingerprints required")
    prefix = {key: diagnostic["q164"]["queries"][key]["group_prefix_sha256"] for key in p290.QUERIES}
    require(all(is_digest(sha) and sha == diagnostic["q100"]["queries"][key]["group_prefix_sha256"]
                for key, sha in prefix.items()), "Both original per-group query prefixes must remain exact")
    return prefix


def verify_tail_receipt(p290, p291, receipt, lock, prefix):
    fixed = dict(status=TAIL_RECEIPT_STATUS, q100_sha256=p290.Q100, q164_sha256=p290.Q164,
                 position_sha256=p290.WINDOWS_POSITION, query_prefix_sha256=prefix,
                 tail_sha256=lock["query_tail_file_sha256"], tail_bytes=TAIL_BYTES, tail_shape=list(TAIL_SHAPE),
                 tail_dtype="float32", tail_byte_order="little", initialization_seed=5201, expansion_seed=5201164,
                 coco_asset_sha256=p290.COCO_SHA, p290_source_sha256=p291.P290_SHA,
                 source_sha256={role: digest for role, (_, digest) in p290.SOURCE_PINS.items()},
                 reference_tensor_substitution=False, **SCOPE)
    require(matches(receipt, fixed) and is_digest(receipt["query_feat_tail_typed_sha256"])
            and receipt["query_feat_tail_typed_sha256"] != POSITION_ONLY_TAIL,
            "Exact independently verified Windows Q164 tail provenance and byte contract required")
    return receipt["query_feat_tail_typed_sha256"]


def scope_guard(p290, p291, coco, position, tail):
    original = p291.scope_guard(p290, coco, position)

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.path.abspath(os.fsdecode(args[0])))
            if path.suffix.lower() == ".bin":
                mode, flags = args[1:3]
                writing = (isinstance(mode, str) and any(char in mode for char in "wax+")
                           or isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                require(path == tail and path.resolve(strict=True) == tail and not writing,
                        "Only exact read-only admitted query-tail binary may be opened")
                return
        original(event, args)

    return guard


def load_tail(p290, path, file_sha, typed_sha, digest, torch):
    import numpy as np

    raw = p290.pinned(path, file_sha, TAIL_BYTES)
    require(len(raw) == TAIL_BYTES and sys.byteorder == "little", "Exact canonical little-endian query-tail bytes required")
    tensor = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(TAIL_SHAPE))
    require(tuple(tensor.shape) == TAIL_SHAPE and tensor.dtype == torch.float32 and tensor.device.type == "cpu"
            and tensor.is_contiguous() and bool(torch.isfinite(tensor).all())
            and digest({QUERY_KEY: tensor}) == typed_sha, "Exact finite canonical Windows query-tail tensor required")
    return tensor


def copy_query_tail(model, tail, torch, digest, original_sha, canonical_sha, prefix_sha, *, tensor_count=545):
    state = model.state_dict()
    require(len(state) == tensor_count and QUERY_KEY in state and digest(state) == original_sha,
            "Exact original position-only Q164 state required before tail substitution")
    original = state[QUERY_KEY]
    require(original.ndim == 2 and original.shape[0] == 13 * 164 and original.is_contiguous()
            and original.dtype == tail.dtype == torch.float32 and original.device.type == tail.device.type == "cpu"
            and tuple(tail.shape) == (13, 64, original.shape[1]) and tail.is_contiguous()
            and bool(torch.isfinite(tail).all()), "Exact grouped Q164 query and contiguous reference tail required")
    query = original.reshape(13, 164, -1)
    prefix = digest({QUERY_KEY: query[:, :100].contiguous()})
    require(prefix == prefix_sha, "Exact original 100-row per-group query-feature prefix required")
    before = {name: digest({name: tensor}) for name, tensor in state.items()}
    shadow = original.detach().clone()
    shadow.reshape(13, 164, -1)[:, 100:164].copy_(tail)
    shadow_sha = digest(dict(state, **{QUERY_KEY: shadow}))
    audit = dict(tail_copy_performed=False, position_only_q164_sha256=original_sha, shadow_q164_sha256=shadow_sha,
                 query_prefix_before_sha256=prefix, reference_tail_typed_sha256=digest({QUERY_KEY: tail}))
    if shadow_sha != canonical_sha:
        raise TailMismatch("canonical Q164 shadow full-state digest", audit)
    parameter = model.get_parameter(QUERY_KEY)
    require(parameter.data_ptr() == original.data_ptr(), "Exact registered Q164 query-feature parameter required")
    with torch.no_grad():
        parameter.view(13, 164, -1)[:, 100:164].copy_(tail)
    after_state = model.state_dict()
    after = {name: digest({name: tensor}) for name, tensor in after_state.items()}
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    selected = after_state[QUERY_KEY].reshape(13, 164, -1)
    prefix_after = digest({QUERY_KEY: selected[:, :100].contiguous()})
    tail_after = digest({QUERY_KEY: selected[:, 100:164].contiguous()})
    observed = digest(after_state)
    audit.update(tail_copy_performed=True, canonical_q164_sha256=observed, changed_tensor_names=changed,
                 unchanged_other_tensors=sum(before[name] == after.get(name) for name in before if name != QUERY_KEY),
                 query_prefix_after_sha256=prefix_after, query_tail_after_typed_sha256=tail_after)
    if (changed != [QUERY_KEY] or audit["unchanged_other_tensors"] != tensor_count - 1 or observed != canonical_sha
            or prefix_after != prefix_sha or tail_after != audit["reference_tail_typed_sha256"]):
        raise TailMismatch("registered Q164 query-feature parameter after bounded copy", audit)
    return audit


def replay_position_only(p290, p291, bundle, coco, position, sentinel, torch):
    observe = p290.observe
    expanded = []

    def capture(model, width, active_torch, digest):
        observation = observe(model, width, active_torch, digest)
        if width == 164:
            expanded.append((model, digest))
        return observation

    with patch.object(p290, "observe", capture):
        result = p291.run_position_check(p290, bundle, coco, position, sentinel, torch)
    require(len(expanded) == 2 and expanded[0][0] is not expanded[1][0],
            "Exact original expanded source and distinct strictly loaded Q164 model required")
    model, digest = expanded[-1]
    return result, model, digest


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("p290-source", "p291-source", "code-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--check-source", action="store_true")
    for key in ("coco", "position", "p291-result", "tail", "tail-receipt", "prelock", "output"):
        parser.add_argument("--" + key, type=Path)
    for key in ("position-sha256", "p291-result-sha256", "tail-receipt-sha256", "prelock-sha256"):
        parser.add_argument("--" + key)
    args = parser.parse_args(argv)
    p291 = pinned_p291(args.p291_source)
    p290 = p291.pinned_p290(args.p290_source)
    bundle = p291.source_bundle(p290, args.code_root)
    if args.check_source:
        if any(item is not None for key, item in vars(args).items() if key not in ("check_source", "p290_source", "p291_source", "code_root")):
            parser.error("--check-source takes no scientific inputs")
        for path, raw in bundle.values():
            if path.suffix == ".py":
                compile(raw, str(path), "exec")
        print("P293_HRSID_DGX_Q164_QUERY_TAIL_SOURCE_CHECK_PASSED", flush=True)
        return 0
    paths = (args.coco, args.position, args.p291_result, args.tail, args.tail_receipt, args.prelock, args.output)
    if (not all(path is not None and path.is_absolute() for path in paths) or len(set(paths)) != len(paths)
            or args.position_sha256 != p291.POSITION_FILE_SHA or args.p291_result_sha256 != PRIOR_SHA
            or not all(is_digest(value) for value in (args.tail_receipt_sha256, args.prelock_sha256))
            or args.coco.name != "rf-detr-seg-nano.pt" or args.position.name != "Q100_POSITION_REFERENCE_PRIVATE.pt"
            or args.tail.name != TAIL_NAME or args.tail_receipt.name != TAIL_RECEIPT or args.output.suffix != ".json"):
        parser.error("Exact isolated original COCO, P291 result, position, Windows tail receipt, prelock and fresh output required")
    require(all(path.resolve(strict=True) == path and path.is_file() for path in paths[:-1])
            and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Canonical separate inputs and fresh output required")
    lock = read_json(p290, p291, args.prelock, args.prelock_sha256)
    verify_prelock(p290, p291, lock, args.p291_result_sha256, args.tail_receipt_sha256)
    previous_raw = p290.pinned(args.p291_result, PRIOR_SHA, PRIOR_BYTES)
    require(len(previous_raw) == PRIOR_BYTES, "Exact independently closed P291 safe summary bytes required")
    previous = json.loads(previous_raw, object_pairs_hook=p291.unique)
    require(matches(previous, dict(p291_source_sha256=P291_SHA, p290_source_sha256=p291.P290_SHA,
                                   position_file_sha256=p291.POSITION_FILE_SHA)), "Exact original P291 result lineage required")
    expected_prefix = verify_position_only(p290, p291, previous)
    receipt = read_json(p290, p291, args.tail_receipt, args.tail_receipt_sha256)
    typed_tail = verify_tail_receipt(p290, p291, receipt, lock, expected_prefix)
    sentinel = args.output.parent / "UNUSED_HRSID_Q164_FIT_LAYOUT"
    require(not sentinel.exists() and not sentinel.is_symlink(), "FIT sentinel must not exist")
    sys.addaudithook(scope_guard(p290, p291, args.coco, args.position, args.tail))
    torch = p290.admit_runtime()
    current, model, digest = replay_position_only(p290, p291, bundle, args.coco, args.position, sentinel, torch)
    require(verify_position_only(p290, p291, current) == expected_prefix
            and all(current[field] == previous[field] for field in ("diagnostic", "position_audit", "non_query_stock_sha256",
                                                                   "non_query_expanded_sha256")),
            "Exact closed P291 tensor and source observations must reproduce before any tail data access")
    require(digest(model.state_dict()) == POSITION_ONLY_Q164, "Exact live P291 Q164 state required before tail data access")
    tail = load_tail(p290, args.tail, lock["query_tail_file_sha256"], typed_tail, digest, torch)
    try:
        audit = copy_query_tail(model, tail, torch, digest, POSITION_ONLY_Q164, p290.Q164, QUERY_PREFIX)
        status, location = PARITY, None
    except TailMismatch as error:
        status, location, audit = MISMATCH, error.location, error.audit
    require(not torch.cuda.is_initialized() and not sentinel.exists(), "Exact CPU-only scope and unused FIT sentinel required")
    value = dict(status=status, mismatch_location=location, initialization_seed=5201, expansion_seed=5201164,
                 input_resolution=576, query_width=164, query_groups=13, expected_canonical_q164_sha256=p290.Q164,
                 canonical_q100_sha256=p290.Q100, query_tail_bytes=TAIL_BYTES, query_tail_shape=list(TAIL_SHAPE),
                 query_tail_file_sha256=lock["query_tail_file_sha256"], windows_query_tail_receipt_sha256=args.tail_receipt_sha256,
                 prior_position_only_result_sha256=PRIOR_SHA, position_file_sha256=p291.POSITION_FILE_SHA,
                 p290_source_sha256=p291.P290_SHA, p291_source_sha256=P291_SHA, p293_source_sha256=lock["p293_source_sha256"],
                 prelock_sha256=args.prelock_sha256, exact_p291_reproduced=True, position_audit=current["position_audit"],
                 query_tail_audit=audit, **SCOPE)
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(raw.decode("utf-8").rstrip("\n"), flush=True)
    return 0 if status == PARITY else 4


if __name__ == "__main__":
    raise SystemExit(main())

