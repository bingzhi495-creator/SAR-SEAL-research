"""Check DGX HRSID Q164 parity after one previously verified Q100 position substitution."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from unittest.mock import patch


P290_SHA = "22e12203a8ad09a5f24911c316983b29ae5ce3d75bfde6065ecf3859c5970f5d"
POSITION_FILE_SHA = "f8c2c8d83b4efa917a1e731a312fc0adaaafba89dda61ef845b61ced0916f9ee"
POSITION_FILE_BYTES = 3542121
LOADER_SHA = "e0281838d992d8491c34b727100754d3937b07fa6df1710a55dbf51beb53e99d"
INIT_SHA = "444e91009bce7a56e65a9bd2110c3479a86609d29b5ccdc925b06ec013b7894a"
PRELOCK_STATUS = "P291_HRSID_DGX_Q164_POSITION_PARITY_PRELOCK_V1"
PARITY = "P291_HRSID_DGX_Q100_POSITION_ONLY_Q164_PARITY_V1"
MISMATCH = "P291_HRSID_DGX_Q100_POSITION_ONLY_Q164_MISMATCH_V1"
SCOPE = dict(FIT=False, DEV=False, TEST=False, AP=False, training=False, cuda_initialized=False,
             model_forward_calls=0, optimizer_updates=0)
LIMIT = 64 * 1024


class NumericMismatch(ValueError):
    def __init__(self, locations, details):
        super().__init__("Exact position-only initialization parity did not pass")
        self.locations = locations
        self.details = details


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest_id(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def pinned_p290(path):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical P290 source required")
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    require(0 < len(raw) <= LIMIT and sha256(raw).hexdigest() == P290_SHA, "Exact immutable uncorrected P290 source required")
    module = ModuleType("_p291_exact_p290")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def source_bundle(p290, code):
    bundle = p290.source_bundle(code)
    paths = (("p242_init", "ssdd_phase_init.py", INIT_SHA),
             ("p242_position", "ssdd_phase_init_dgx_canonical.py", LOADER_SHA))
    for role, name, digest in paths:
        path = code / "SSDD_MASK_PHASE_TRANSFER_20261110" / name
        bundle[role] = path, p290.pinned(path, digest)
    return bundle


def unique(pairs):
    value = dict(pairs)
    require(len(value) == len(pairs), "Duplicate prelock fields forbidden")
    return value


def prelock_value(p290, path, digest):
    raw = p290.pinned(path, digest, LIMIT)
    value = json.loads(raw, object_pairs_hook=unique)
    expected = dict(status=PRELOCK_STATUS, p290_source_sha256=P290_SHA, position_loader_sha256=LOADER_SHA,
                    position_file_sha256=POSITION_FILE_SHA, expected_raw_q100_sha256=p290.P242_RAW_Q100,
                    expected_canonical_q100_sha256=p290.Q100, expected_canonical_q164_sha256=p290.Q164,
                    FIT=False, DEV=False, TEST=False, AP=False)
    require(type(value) is dict and set(value) == set(expected) | {"p291_source_sha256", "prior_uncorrected_result_sha256"}
            and all(type(value[key]) is type(item) and value[key] == item for key, item in expected.items())
            and digest_id(value["p291_source_sha256"]) and digest_id(value["prior_uncorrected_result_sha256"]),
            "Exact preregistered position-only parity prelock required")
    p290.pinned(Path(__file__).resolve(strict=True), value["p291_source_sha256"], LIMIT)
    return value


def scope_guard(p290, coco, position):
    original = p290.scope_guard(coco)

    def guard(event, args):
        if (event == "open" and isinstance(args[0], (str, bytes))
                and Path(os.path.abspath(os.fsdecode(args[0]))) == position):
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(char in mode for char in "wax+")
                       or isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            require(not writing and position.resolve(strict=True) == position, "Exact read-only canonical position file required")
            return
        original(event, args)

    return guard


def tensor_pins(state, digest):
    return {key: digest({key: tensor}) for key, tensor in state.items()}


def pins_digest(pins, exclude):
    raw = json.dumps({key: value for key, value in pins.items() if key not in exclude},
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def canonicalize(model, load_position, torch, digest, p290, *, tensor_count=545):
    key = p290.POSITION
    state = model.state_dict()
    require(len(state) == tensor_count and key in state, "Exact original stock state tensor census required")
    before = tensor_pins(state, digest)
    before_full = digest(state)
    details = dict(raw_q100_sha256=before_full, raw_position_sha256=before[key], reference_tensor_substitution=False)
    if before_full != p290.P242_RAW_Q100 or before[key] != p290.P242_RAW_POSITION:
        raise NumericMismatch(["uncorrected Q100 or position differs from known P242 fingerprint"], details)

    reference = load_position()
    target_sha = digest({key: reference})
    require(target_sha == p290.WINDOWS_POSITION, "Exact verified reference position tensor required")
    shadow = digest(dict(state, **{key: reference}))
    details.update(reference_position_sha256=target_sha, shadow_q100_sha256=shadow)
    if shadow != p290.Q100:
        raise NumericMismatch(["Q100 still differs in the position-only shadow state"], details)

    parameter = model.get_parameter(key)
    require(parameter.data_ptr() == state[key].data_ptr(), "Exact registered Q100 position parameter required")
    with torch.no_grad():
        parameter.copy_(reference)
    after_state = model.state_dict()
    after = tensor_pins(after_state, digest)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    details.update(reference_tensor_substitution=True, changed_q100_tensor_names=changed,
                   canonical_q100_sha256=digest(after_state), canonical_position_sha256=after.get(key),
                   unchanged_non_position_tensors=sum(before[name] == after.get(name) for name in before if name != key),
                   non_position_before_sha256=pins_digest(before, {key}), non_position_after_sha256=pins_digest(after, {key}))
    if changed != [key] or details["canonical_q100_sha256"] != p290.Q100 or after.get(key) != p290.WINDOWS_POSITION:
        raise NumericMismatch(changed or ["canonical Q100 full state"], details)
    return details, after


def localize_expansion(stock, final, queries, prefixes_same, zero_refpoint_tail, matches):
    locations = sorted(name for name in (stock.keys() | final.keys()) - set(queries) if stock.get(name) != final.get(name))
    if not prefixes_same:
        locations.append("original per-group query prefixes")
    if not zero_refpoint_tail:
        locations.append("refpoint_embed.weight[13 groups, rows 100:164]")
    if not locations and not matches:
        locations.append("query_feat.weight[13 groups, rows 100:164]")
    return locations


def run_position_check(p290, bundle, coco, position, sentinel, torch):
    p290.load(bundle, "p242_init", "ssdd_phase_init")
    loader = p290.load(bundle, "p242_position", "_p291_exact_p242_position")
    require(loader.POSITION_KEY == p290.POSITION and loader.POSITION_FILE_BYTES == POSITION_FILE_BYTES
            and loader.NATIVE_DGX_SHA256 == p290.P242_RAW_Q100
            and loader.NATIVE_DGX_POSITION_SHA256 == p290.P242_RAW_POSITION
            and loader.CANONICAL_Q100_SHA256 == p290.Q100
            and loader.CANONICAL_POSITION_SHA256 == p290.WINDOWS_POSITION,
            "Exact original P242 position loader identity required")
    original_load = p290.load
    native = stock = digest = final_module = None

    def observe_source(current_bundle, role, alias):
        nonlocal native, digest
        result = original_load(current_bundle, role, alias)
        if role == "state":
            digest = result.tensor_dict_sha256
        if role == "native":
            native = result
            initialize = native.initialize

            def initialize_position(base, output, module_type):
                def construct(config, train, *, load_encoder_weights):
                    nonlocal stock, final_module
                    model = module_type(config, train, load_encoder_weights=load_encoder_weights)
                    if config.pretrain_weights is not None:
                        require(stock is None and final_module is None and Path(config.pretrain_weights) == coco
                                and config.num_queries == 100 and config.resolution == 576,
                                "Only the exact first stock official COCO Q100 factory can be corrected")
                        load_ref = lambda: loader._load_position(position, POSITION_FILE_SHA, torch, digest)
                        stock = canonicalize(model.model, load_ref, torch, digest, p290)
                    else:
                        require(stock is not None and final_module is None and config.num_queries == 164
                                and config.resolution == 576, "Exact fresh original Q164 module factory required")
                        final_module = model
                    return model

                return initialize(base, output, construct)

            result.initialize = initialize_position
        return result

    try:
        with patch.object(p290, "load", observe_source):
            value = p290.original_initialize(bundle, coco, sentinel, torch)
    except NumericMismatch as error:
        return dict(status=MISMATCH, localization=error.locations, position_audit=error.details,
                    original_expansion_completed=False, reference_tensor_substitution=error.details["reference_tensor_substitution"],
                    **SCOPE)

    require(stock is not None and final_module is not None and digest is not None, "Both original initialization factories required")
    audit, stock_pins = stock
    final_state = final_module.model.state_dict()
    final = tensor_pins(final_state, digest)
    refpoint = final_state[p290.QUERIES[0]].detach().reshape(13, 164, -1)[:, 100:]
    zero_refpoint_tail = bool(torch.count_nonzero(refpoint).item() == 0)
    diagnostics = value["diagnostic"]
    prefixes_same = all(diagnostics["q100"]["queries"][key]["group_prefix_sha256"]
                        == diagnostics["q164"]["queries"][key]["group_prefix_sha256"] for key in p290.QUERIES)
    location = localize_expansion(stock_pins, final, p290.QUERIES, prefixes_same, zero_refpoint_tail,
                                  value["observed_q164_sha256"] == p290.Q164)
    matched = value["status"] == p290.PARITY and not location
    value.update(status=PARITY if matched else MISMATCH, position_audit=audit,
                 reference_tensor_substitution=True, original_expansion_completed=True, localization=location,
                 canonical_q100_matches=(value["observed_q100_sha256"] == p290.Q100),
                 canonical_q164_matches=(value["observed_q164_sha256"] == p290.Q164),
                 original_query_prefixes_match=prefixes_same, zero_refpoint_tail=zero_refpoint_tail,
                 non_query_stock_sha256=pins_digest(stock_pins, set(p290.QUERIES)),
                 non_query_expanded_sha256=pins_digest(final, set(p290.QUERIES)), **SCOPE)
    return value


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--p290-source", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--check-source", action="store_true")
    for key in ("coco", "position", "prelock", "output"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--position-sha256")
    parser.add_argument("--prelock-sha256")
    args = parser.parse_args(argv)
    p290 = pinned_p290(args.p290_source)
    bundle = source_bundle(p290, args.code_root)
    if args.check_source:
        if any(getattr(args, key) is not None for key in ("coco", "position", "prelock", "output", "position_sha256", "prelock_sha256")):
            parser.error("--check-source takes no scientific inputs")
        for path, raw in bundle.values():
            if path.suffix == ".py":
                compile(raw, str(path), "exec")
        print("P291_HRSID_DGX_Q164_POSITION_SOURCE_CHECK_PASSED", flush=True)
        return 0
    coco, position, prelock, output = (getattr(args, key) for key in ("coco", "position", "prelock", "output"))
    if (not all(path is not None and path.is_absolute() for path in (coco, position, prelock, output))
            or args.position_sha256 != POSITION_FILE_SHA or not digest_id(args.prelock_sha256)
            or coco.name != "rf-detr-seg-nano.pt" or position.name != "Q100_POSITION_REFERENCE_PRIVATE.pt"
            or output.suffix != ".json"):
        parser.error("Exact local COCO, P242 position artifact, prelock and fresh safe JSON required")
    require(len({coco, position, prelock, output}) == 4 and all(path.resolve(strict=True) == path and path.is_file()
            for path in (coco, position, prelock)) and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Canonical separate inputs and fresh output required")
    lock = prelock_value(p290, prelock, args.prelock_sha256)
    sentinel = output.parent / "UNUSED_HRSID_Q164_FIT_LAYOUT"
    require(not sentinel.exists() and not sentinel.is_symlink(), "FIT sentinel must not exist")
    sys.addaudithook(scope_guard(p290, coco, position))
    torch = p290.admit_runtime()
    value = run_position_check(p290, bundle, coco, position, sentinel, torch)
    require(not torch.cuda.is_initialized() and not sentinel.exists(), "CPU-only scope and unused FIT sentinel required")
    value.update(position_file_sha256=POSITION_FILE_SHA, p242_position_loader_sha256=LOADER_SHA, p242_initializer_sha256=INIT_SHA,
                 prelock_sha256=args.prelock_sha256, p291_source_sha256=lock["p291_source_sha256"], p290_source_sha256=P290_SHA,
                 prior_uncorrected_result_sha256=lock["prior_uncorrected_result_sha256"],
                 expected_canonical_q100_sha256=p290.Q100, expected_canonical_q164_sha256=p290.Q164,
                 original_source_sha256={role: digest for role, (_, digest) in p290.SOURCE_PINS.items()},
                 runtime=p290.PACKAGES, cpu_affinity=[0, 1], cpu_threads=2)
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(raw.decode("utf-8").rstrip("\n"), flush=True)
    return 0 if value["status"] == PARITY else 4


if __name__ == "__main__":
    raise SystemExit(main())

