"""Replay a closed native matcher capture and summarize paired FIT-only attribution."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from itertools import chain
import json
import os
from pathlib import Path
import re
import sys
import time
import types

import numpy as np

CAPTURE_SHA = "7a39df88f849d36ff139195d6001ccf7a5ea258144d9bcd4a3440e7a1a572703"
MATH_SHA = "1d48816237ddbbc81ec23798c5bb7b3cada5a8e20f7480b35bcfcfc3aad8bfed"
FIT_MANIFEST_SHA = "6dd6dd35f8418285696a224eab0ada6538224a0603c9eeb14c5ae98ea9fbcf95"
DRAWS = 8
DRAW_SEED = 5201
BOOTSTRAPS = 2048
STATUSES = ("STOCK_TRACE_EXACT", "ZERO_TARGET_AUGMENTED", "INVALID_TARGET_ROWS")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def sha_string(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical(path, *, directory=False):
    need(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
         and (path.is_dir() if directory else path.is_file()), "Expected canonical input path")
    return path


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def checked_bytes(path, expected, limit=2**20):
    canonical(path)
    before = path.stat()
    need(0 < before.st_size <= limit, "Bounded pinned input size")
    raw = path.read_bytes()
    need(digest(raw) == expected and identity(before) == identity(path.stat()), "Exact stable pinned bytes")
    return raw


def module(path, expected, name):
    raw = checked_bytes(path, expected)
    result = types.ModuleType(name)
    result.__file__ = str(path)
    sys.modules[name] = result
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def validate_closed(report, receipt, capture_protocol, report_sha, cap):
    closure = dict(status="NATIVE_MATCHER_CAPTURE_CLOSED_ACCEPTED", protocol=capture_protocol,
                   capture_report_sha256=report_sha, capture_source_sha256=CAPTURE_SHA,
                   original_exit_code=0, resource_stop=False)
    need(all(receipt.get(key) == value for key, value in closure.items()), "Owner-confirmed original capture closure")
    need(report["status"] == "NATIVE_SSDD_FIT_FINAL_MATCHER_CAPTURE_COMPLETE"
         and report["protocol"] == capture_protocol and report["FIT_only"] is True
         and all(report[k] is False for k in ("DEV", "TEST", "AP", "optimizer_created", "backward"))
         and report["updates"] == 0
         and (report["resolution"], report["groups"], report["width"], report["stock_points"]) == (576, 13, 100, 1296)
         and report["native_versions"] == cap.VERSIONS, "Exact original FIT train-style capture scope")
    references = dict(original_archive_sha256=cap.ARCHIVE_SHA, raw_model_sha256=cap.RAW_SHA,
                      fit_index_sha256=FIT_MANIFEST_SHA, fit_bundle_sha256=cap.FIT_BUNDLE_SHA,
                      fit_gt_sha256=cap.FIT_GT_SHA)
    need(all(report[key] == value for key, value in references.items()), "Exact native archive and FIT scope pins")
    rows = report["rows"]
    trace_count = sum(row["status"] == "STOCK_TRACE_EXACT" for row in rows)
    need(report["visited_views"] == len(rows) and report["exact_stock_traces"] == trace_count
         and all(row["ordinal"] == ordinal and row["status"] in STATUSES
                 for ordinal, row in enumerate(rows)), "Complete consecutive capture row ledger")
    tranche = report["tranche"]
    need((tranche == "smoke" and trace_count == 1 and 1 <= len(rows) <= 24
          and rows[-1]["status"] == "STOCK_TRACE_EXACT" and report["predeclared_mechanism_trace_count_met"] is False)
         or (tranche == "cohort32" and len(rows) == 32 and trace_count >= 24
             and report["predeclared_mechanism_trace_count_met"] is True), "Distinct technical and mechanism scope")
    return rows


def load_packet(torch, capture_dir, row, plan, protocol, cap):
    ordinal = row["ordinal"]
    leaf = f"VIEW_{ordinal:02d}_PRIVATE.pt"
    need(row["private_packet_name"] == leaf and sha_string(row["private_packet_sha256"]),
         "Exact private packet role and digest")
    path = canonical(capture_dir / leaf)
    with path.open("rb") as stream:
        before = identity(os.fstat(stream.fileno()))
        need(0 < before[2] == row["private_packet_bytes"] <= 160 * 1024**2
             and cap.file_digest(stream) == row["private_packet_sha256"], "Exact closed packet bytes")
        stream.seek(0)
        packet = torch.load(stream, map_location="cpu", weights_only=True)
        need(identity(os.fstat(stream.fileno())) == before and identity(path.stat()) == before,
             "No packet replacement during load")
    item, repeat = plan[ordinal]
    need((packet["protocol"], packet["fit_index_sha256"], packet["raw_model_sha256"],
          packet["ordinal"], packet["dataset_index"], packet["image_id"], packet["augmentation_seed"])
         == (protocol, FIT_MANIFEST_SHA, cap.RAW_SHA, ordinal, item["dataset_index"], item["image_id"],
             5301000 + 2 * item["dataset_index"] + repeat)
         and sha_string(packet["input_sha256"]), "Exact fixed FIT view identity")
    targets = row["target_rows"]
    need(type(targets) is int and 0 < targets <= 100 and packet["width"] == 100 and packet["groups"] == 13
         and tuple(packet["mask_weights"]) == (5., 5.), "Locked matcher weights and group sizes")
    for key in ("nonmask", "native_total", "native_mask_ce", "native_mask_dice"):
        tensor = packet[key]
        need(tuple(tensor.shape) == (1300, targets) and tensor.dtype == torch.float32
             and bool(torch.isfinite(tensor).all()), "Finite captured cost matrix")
    masks = packet["target_masks"]
    area = packet["target_area_pixels"]
    small = packet["is_small"]
    need(masks.dtype == small.dtype == torch.bool and tuple(masks.shape) == (targets, 576, 576)
         and tuple(area.shape) == tuple(small.shape) == (targets,)
         and torch.equal(masks.flatten(1).sum(1), area) and bool((area > 0).all())
         and torch.equal(small, area < 1024) and int(small.sum()) == row["small_targets"],
         "True augmented GT areas and small-instance scope")
    return packet


def sampler_and_anchor(torch, packet, cap, math, device):
    sample = cap.make_sampler(packet, device)
    native_terms = cap.make_sampler(packet, device, native_cost=True)
    recorded_dtypes = packet["native_cost_dtypes"]
    allowed = {str(x): x for x in (torch.float32, torch.bfloat16, torch.float16)}
    need(set(recorded_dtypes) == {"nonmask", "ce", "dice"}
         and all(x in allowed for x in recorded_dtypes.values()), "Known native matcher arithmetic dtype")
    w0, w1 = packet["mask_weights"]
    nonmask = packet["nonmask"].to(device=device, dtype=allowed[recorded_dtypes["nonmask"]])

    def recompose(ce, dice):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ce = torch.as_tensor(ce, device=device, dtype=allowed[recorded_dtypes["ce"]])
            dice = torch.as_tensor(dice, device=device, dtype=allowed[recorded_dtypes["dice"]])
            return ((nonmask + w0 * ce) + w1 * dice).float().cpu().numpy()

    def total(points):
        return recompose(*native_terms(points))

    draw = packet["uniform_draw"].numpy()
    math.paired_points(draw)
    pred, labels = sample(draw)
    ce, dice = native_terms(draw)
    observed = packet["native_total"].numpy()
    need(np.array_equal(pred, packet["native_pred_at_draw"].numpy())
         and np.array_equal(labels, packet["native_target_at_draw"].numpy())
         and np.array_equal(ce, packet["native_mask_ce"].numpy())
         and np.array_equal(dice, packet["native_mask_dice"].numpy())
         and np.array_equal(recompose(ce, dice), observed), "Exact native stock logits, labels, terms and total")
    reference = math.assignment_from_native_pairs(packet["native_src"].numpy(), packet["native_dst"].numpy(),
                                                  targets=observed.shape[1])
    replay = math.solve_groups(observed)
    delta = math.take_assignments(observed, reference).sum(axis=1, dtype=np.float64) - (
        math.take_assignments(observed, replay).sum(axis=1, dtype=np.float64))
    need(np.all(np.abs(delta) <= 1e-12), "Common solver agrees on native stock objective per group")
    return sample, total, bool(np.array_equal(replay, reference))


def image_pair(pairs):
    if not pairs:
        return dict(images=0, iid=None, jitter=None, iid_minus_jitter=None, percentile95=None,
                    positive_images=0, negative_images=0, tied_images=0)
    values = np.asarray(pairs, dtype=np.float64)
    difference = values[:, 0] - values[:, 1]
    rng = np.random.default_rng(DRAW_SEED)
    index = rng.integers(len(values), size=(BOOTSTRAPS, len(values)))
    return dict(images=len(values), iid=float(values[:, 0].mean()), jitter=float(values[:, 1].mean()),
                iid_minus_jitter=float(difference.mean()),
                percentile95=np.quantile(difference[index].mean(axis=1), (.025, .975)).tolist(),
                positive_images=int((difference > 0).sum()), negative_images=int((difference < 0).sum()),
                tied_images=int((difference == 0).sum()))


def aggregate(views, math):
    out = {}
    for tag in math.THRESHOLDS:
        out[tag] = {}
        for scope in ("small", "other", "all"):
            references = Counter()
            sampled = {arm: Counter() for arm in ("iid", "jitter")}
            paired, gains = [], []
            for view in views:
                ref = view["reference"][tag][scope]
                references.update(ref)
                counters = {arm: Counter() for arm in sampled}
                need(len(view["draws"]) == DRAWS, "Exact predeclared repetitions per image")
                for draw in view["draws"]:
                    for arm in sampled:
                        counts = draw[arm][tag][scope]
                        need(counts["stable_reference_good"] == ref["stable_dense_good"],
                             "Fixed stable good opportunities in every paired draw")
                        counters[arm].update(counts)
                        sampled[arm].update(counts)
                support = DRAWS * ref["stable_dense_good"]
                if support:
                    paired.append(tuple(counters[a]["stable_reference_good_lost"] / support for a in sampled))
                bad = DRAWS * (ref["targets"] - ref["stable_dense_good"] - ref["dense_quality_unstable"])
                if bad:
                    gains.append(tuple(counters[a]["stable_reference_bad_gained"] / bad for a in sampled))
            out[tag][scope] = dict(reference=dict(references), sampled={a: dict(c) for a, c in sampled.items()},
                                   image_paired_lost_rate=image_pair(paired), image_paired_gained_rate=image_pair(gains))
    regrets = [
        tuple(float(np.mean([draw[arm]["dense_cost_regret"] for draw in view["draws"]]))
              for arm in ("iid", "jitter"))
        for view in views
    ]
    return dict(thresholds=out, image_paired_dense_cost_regret=image_pair(regrets))


def run(args):
    need(sys.version_info[:3] in ((3, 10, 13), (3, 12, 14))
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Approved Windows/DGX runtime, one offline GPU")
    for value in (args.protocol, args.capture_protocol):
        need(re.fullmatch(r"[0-9a-f]{40}", value), "Exact preregistered commit")
    need(sha_string(args.capture_report_sha256) and sha_string(args.capture_closure_sha256)
         and re.fullmatch(r"[0-9]+[.][0-9]+[.][0-9]+", args.scipy_version), "Exact expected hashes and SciPy version")
    canonical(args.capture_dir, directory=True)
    need(args.output.is_absolute() and canonical(args.output.parent, directory=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink()
         and not args.output.is_relative_to(args.capture_dir), "Fresh output outside closed capture")
    started = time.perf_counter()
    cap = module(args.capture_source, CAPTURE_SHA, "_pinned_capture_matcher_cost")
    math = module(Path(__file__).resolve().with_name("matcher_cost_math.py"), MATH_SHA, "_pinned_matcher_cost_math")
    report = json.loads(checked_bytes(args.capture_dir / "CAPTURE_SAFE.json", args.capture_report_sha256))
    receipt = json.loads(checked_bytes(args.capture_closure, args.capture_closure_sha256))
    rows = validate_closed(report, receipt, args.capture_protocol, args.capture_report_sha256, cap)
    items = json.loads(checked_bytes(args.fit_index, FIT_MANIFEST_SHA))["items"]
    need(len(items) == 741 and [item["dataset_index"] for item in items] == list(range(741)), "Exact original FIT census")
    plan = cap.view_plan(items, report["tranche"])
    need(len(rows) <= len(plan)
         and (report["tranche"] == "smoke" or len({item["image_sha256"] for item, _ in plan}) == 32),
         "Exact original-content image grouping")
    sys.addaudithook(cap.offline)
    import torch
    need(version("scipy") == args.scipy_version
         and all(version(name) == value for name, value in cap.VERSIONS.items())
         and all(digest(Path(import_module(name).__file__).read_bytes()) == value
                 for name, value in cap.NATIVE_SOURCE.items()), "Exact capture runtime and pinned RF-DETR native source")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One locked native CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(min(.35, cap.CAP / torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    traced = [row for row in rows if row["status"] == "STOCK_TRACE_EXACT"]
    identical = 0
    # Complete native anchor acceptance on every packet before starting mechanism work.
    for row in traced:
        packet = load_packet(torch, args.capture_dir, row, plan, args.capture_protocol, cap)
        _, _, pairs_exact = sampler_and_anchor(torch, packet, cap, math, device)
        identical += int(pairs_exact)
    del packet, _
    result = dict(
        status="NATIVE_MATCHER_TECHNICAL_LABEL_ANCHOR_ONLY_COMPLETE",
        protocol=args.protocol, capture_protocol=args.capture_protocol, tranche=report["tranche"],
        FIT_only=True, DEV=False, TEST=False, AP=False, training=False, updated_parameters=0,
        visited_statuses=dict(Counter(row["status"] for row in rows)),
        exact_native_anchors=len(traced), exact_native_solver_pair_images=identical,
        sources=dict(capture_sha256=CAPTURE_SHA, capture_report_sha256=args.capture_report_sha256,
                     capture_closure_sha256=args.capture_closure_sha256, fit_index_sha256=FIT_MANIFEST_SHA,
                     matcher_math_sha256=MATH_SHA, runtime_sha256=digest(Path(__file__).resolve().read_bytes())),
        native_versions=cap.VERSIONS, scipy_version=args.scipy_version,
    )
    if report["tranche"] == "cohort32":
        views = []
        for row in traced:
            packet = load_packet(torch, args.capture_dir, row, plan, args.capture_protocol, cap)
            sample, total, _ = sampler_and_anchor(torch, packet, cap, math, device)
            seed = np.random.SeedSequence((DRAW_SEED, row["ordinal"]))
            draws = chain((packet["uniform_draw"].numpy(),), math.uniform_points(seed, DRAWS - 1))
            view = math.diagnose_view(packet["nonmask"].numpy(), sample, packet["is_small"].numpy(),
                                      packet["mask_weights"], draws, native_sampled_total=total)
            need(view["point_budget"] == 1296 and view["sampled_cost_source"] == "native_total",
                 "Identical sampled budget and native arithmetic")
            views.append(view)
            print(json.dumps(dict(event="MATCHER_ATTRIBUTION_VIEW_COMPLETE", ordinal=row["ordinal"])), flush=True)
        result.update(status="NATIVE_MATCHER_FIT_COHORT32_ATTRIBUTION_COMPLETE", draws_per_image=DRAWS,
                      seed=DRAW_SEED, bootstraps=BOOTSTRAPS, dense_side=math.DENSE_SIDE,
                      dense_phases=[list(p) for p in math.DENSE_PHASES], dense_dice_smoothing=1. / math.K,
                      attribution=aggregate(views, math))
    result["seconds"] = time.perf_counter() - started
    args.output.mkdir()
    with (args.output / "RESULT.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "capture-source", "capture-closure", "fit-index", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol", "capture-protocol", "capture-report-sha256", "capture-closure-sha256", "scipy-version"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
