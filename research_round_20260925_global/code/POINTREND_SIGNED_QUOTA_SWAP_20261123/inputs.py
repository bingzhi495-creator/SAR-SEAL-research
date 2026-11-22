"""Load closed Instance-RSDD native and PointRend ledgers using image-only provenance."""

from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
PORTABLE_740 = "1269ecbc18820594bcaa1996d96d78e7a659e5adada3f0310a131751e6bb0ffa"
POINTREND_HEAD = "7401de08367ef31b909ea9c92daa27be02d4efb04460acbf96eab33648fdd7cc"
# Final result and external observer digests. For 740 the independently verified
# ledgers live in the original runs; the portable capture has identical native stream bytes.
CAPTURE = {
    74: ("f43fb04035045b1e30b7ae06ced778e66fcc4cd3e34077f83ebff6096dacc71d",
         "af977d49d1a9c06399a145aaff5e19c7e2d36d0bc8311d88014f66079bbbdaf5"),
    740: ("8fed919f68d7cd8d4b1ba1aaa949f796a3b5b260d340199f8b64b180abef68c7",
          "836826714d978469c225994ff8d77adde73d01cb357df4ce6acb118e2d966576"),
}
OCCUPANCY = {
    74: ("b31d4d1c718aa6cac9b73c61778aafb94404a897233928d8b78fa678d560f083",
         "628334c39074358eb9aa4ff0a8436c9a4a33c5eeb03d56d106e9b307395c918e"),
    740: ("13581737cc2f7358acad5b00cd5a6d37ad0075f8fd9fc457e4c20ff4edb27fad",
          "bded198793fa9b99a6a88bac85de917be4195be33e40bc8e91e48aacd1998375"),
}
POINT = {
    74: ("dbc6e2dbce37cdd8fea34ccefbce16af0ed1ce58db2495de682e278921626fc8",
         "62b9eecb1411169db4e3620822ff957e555a66981b5af8ee570582044ce04131"),
    740: ("ed6e7d48b5f5ec0ba9a98f37327ed4cd075a95c5ffc3857167a79da53f6a0cbd",
          "2dcbecab56ac7dfab07c91f47302ee4f4c2dd5e70bd99ca3069272123cc5b848"),
}
SUPPORT = ("7146d6e6e9b184d81ab2a8fdcd86f7f508d0c2a82f7bc4a6f4fb6d759737cdf0",
           "6cdbb672dee8ceeacc064f49f6ebd05bd5ab0e18f5df999b9db958fd2dd798a7")
POINT_ARMS = {"coarse_paste", "point_paste", "point_equal", "learned_common"}


def require(value, detail):
    if not value:
        raise ValueError(detail)


def pin(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Closed input digest mismatch: " + path.name)
    return raw


def module(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def closed(root, output_dir, observer_dir, expected, phase):
    result = json.loads(pin(root / output_dir / "RESULT_SAFE.json", expected[0]))
    terminal = json.loads(pin(root / observer_dir / "TERMINAL.json", expected[1]))
    require(type(terminal["actual_exit"]) is int and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True and terminal["phase"] == phase
            and terminal["result_sha256"] == expected[0] and terminal["worker_sha256"] == result["worker_sha256"],
            "Successful closed worker and verified external footer required")
    return result


def checked_lines(path, expected):
    seal = sha256()
    with path.open("rb") as stream:
        for line in stream:
            seal.update(line)
            yield json.loads(line)
    require(seal.hexdigest() == expected, "Closed ledger digest mismatch: " + path.name)


def add_input_arguments(parser):
    parser.add_argument("--cohort", required=True, type=int, choices=(74, 740))
    for name in ("code", "stock-root", "point-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--support-root", "--h10-root", dest="support_root", type=Path)


def closed_captures(args, cohort, portable):
    reserve = cohort == 74
    stock = args.stock_root
    source = SimpleNamespace(code=args.code, capture=stock / "capture", intake=stock / ("conversion" if reserve else "intake"),
                             roster=stock / "reserve.csv")
    cap = closed(stock, "capture", "observed_capture", CAPTURE[cohort], "prepare-edits" if reserve else "capture")
    native = portable.module(args.code / portable.NATIVE / "capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    if reserve:
        items, manifest_sha = portable.load_intake(source, native)
        readback, predictions = portable.read_capture(source, items, manifest_sha)
        require(readback == cap and cap["roster_sha256"] == portable.ROSTER, "Closed reserve carrier required")
    else:
        full = module(args.code / "SAR_B5_MAIN_TABLES_20261123/portable_rsdd740.py", PORTABLE_740)
        items = full.load_items(source, portable, native)
        readback, cap_sha = full.capture_info(source, portable, items)
        require(readback == cap and cap_sha == CAPTURE[cohort][0], "Closed 740-image carrier required")
        predictions = full.capture_rows(source, portable, items, cap)
    require(len(items) == cap["images"] == cap["model_forward_calls"] == cohort
            and cap["target_labels_opened"] is False and cap["optimizer_updates"] == 0
            and cap["checkpoint_sha256"] == portable.CHECKPOINT, "Original label-free capture lineage required")
    return source, items, predictions, cap


def closed_edits(args, cohort, cap, portable):
    reserve = cohort == 74
    occupancy_dir = "replay" if reserve else "replay5217"
    old = closed(args.stock_root, occupancy_dir, "observed_" + occupancy_dir, OCCUPANCY[cohort], "verify-edits")
    point_dir = "replay" if reserve else "verify-edits"
    point = closed(args.point_root, point_dir, "observed_" + point_dir, POINT[cohort], point_dir)
    require(old["status"] == ("B5_RESERVE74_REPLAY_COMPLETE" if reserve else "B5_RSDD740_REPLAY_COMPLETE")
            and old["point_head_sha256"] == portable.POINT_HEAD
            and old["target_labels_opened"] is False and old["optimizer_updates"] == 0
            and all(old[k] == cap[k] for k in ("predictions_sha256", "audit_sha256", "manifest_sha256", "checkpoint_sha256", "images")),
            "Closed occupancy ledger must refer to the portable stock")
    require(point["GT_opened"] is False and point["fit_weight_sha256"] == POINTREND_HEAD
            and point["model_forwards"] == point["optimizer_updates"] == 0 and point["images"] == cohort
            and point["predictions_sha256"] == cap["predictions_sha256"], "Frozen PointRend lineage required")
    if reserve:
        require(point["status"] == "INSTANCE_RSDD_RESERVE74_POINTREND_REPLAY_ACCEPTED_V1"
                and point["exact_independent_render"] is point["same_endpoints_exact_realized_adds_drops"] is True
                and point["capture_result_sha256"] == CAPTURE[cohort][0]
                and all(point[k] == cap[k] for k in ("manifest_sha256", "roster_sha256", "audit_sha256", "checkpoint_sha256"))
                and old["independent_decisions_equal"] is True and old["edits_sha256"] == point["occupancy_edits_sha256"],
                "Reserve original renderer, carrier and occupancy do not match")
    else:
        require(point["status"] == "INSTANCE_RSDD_POINTREND_REPLAY_ACCEPTED_V1" and point["author_TEST"] is True
                and point["exact_replayed_policy_choices"] is point["exact_support_xor_area"] is True
                and point["candidate_audit_sha256"] == cap["audit_sha256"] and old["capture_sha256"] == CAPTURE[cohort][0]
                and old["exact_replayed_policy_choices"] is old["exact_support_xor_area"] is True
                and old["edits_sha256"] == point["old_edits_sha256"], "740-image original and portable streams differ")
    support = None
    if args.support_root is not None:
        require(not reserve, "Archived support-factor ledger exists only for the original 740-image cohort")
        support = closed(args.support_root, "verify-edits", "observed_verify-edits", SUPPORT, "verify-edits")
        require(support["status"] == "INSTANCE_RSDD_POINTREND_SUPPORT_FACTOR_ACCEPTED_V1" and support["GT_opened"] is False
                and support["exact_replayed_policy_choices"] is support["exact_support_xor_area"] is True
                and support["images"] == cohort and support["model_forwards"] == support["optimizer_updates"] == 0
                and support["source_replay_receipt_sha256"] == POINT[cohort][0]
                and support["source_replay_terminal_sha256"] == POINT[cohort][1] and support["source_edits_sha256"] == point["edits_sha256"]
                and all(support[k] == point[k] for k in ("capture_receipt_sha256", "candidate_audit_sha256", "predictions_sha256")),
                "Historical support-factor ledger must bind the exact 740-image PointRend run")
    directory = "replay" if reserve else "prepare-edits"
    files = dict(occupancy=args.stock_root / occupancy_dir / portable.EDITS, point=args.point_root / directory / portable.EDITS)
    if support:
        files["support"] = args.support_root / "prepare-edits" / portable.EDITS
    return old, point, support, files


def validate_row(item, audit, pred, point, old, support):
    image = item["image_id"]
    require(image == audit["image_id"] == pred["image_id"] == point["image_id"] == old["image_id"]
            and (support is None or support["image_id"] == image) and not audit["posterior_overrides"]
            and audit["original_hw"] == [512, 512], "Order, original grid or capture changed")
    native = pred["coco"]
    for ledger, arms in ((point, POINT_ARMS), (support, {"p_on_support", "p_off_support"})):
        if ledger is None:
            continue
        for key, changes in ledger["changes"].items():
            require(key.isdecimal() and str(int(key)) == key and 1 <= int(key) <= len(native)
                    and changes and set(changes) <= arms, "Unknown historical rank or mask policy")
    natural = {k: v["free_natural"] for k, v in old["changes"].items() if "free_natural" in v}
    learned = {k: v["learned_common"] for k, v in point["changes"].items() if "learned_common" in v}
    require(natural == learned, "PointRend comparison must retain the original seed-5217 natural masks and quotas")
    ranks = audit["native_ranks"]
    require(pred["native_ranks"] == ranks == sorted(set(ranks)) and all(1 <= k <= 100 for k in ranks),
            "Native top-100 rank order differs")
    for pos, (r, q, c, det) in enumerate(zip(ranks, audit["query_ids"], audit["candidates"], native, strict=True), 1):
        require((pos, r, q, det["score"]) == (c["position"], c["native_rank"], c["query_id"], c["score"])
                and det["image_id"] == image and det["category_id"] == 1 and det["segmentation"]["size"] == [512, 512],
                "Original native candidate identity, area grid or score differs")
        opts = point["changes"].get(str(pos), {})
        require(opts.get("point_equal", {}).get("swaps", 0) == opts.get("learned_common", {}).get("swaps", 0),
                "Historical balanced controls do not share the per-candidate intervention")


def load_population(args):
    """Return [(item,audit,prediction,point_edits,historical_support_or_None)], and aggregate provenance."""
    cohort = int(args.cohort)
    require(cohort in CAPTURE, "Unknown cohort")
    portable = module(args.code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", PORTABLE)
    source, items, predictions, cap = closed_captures(args, cohort, portable)
    old, point, support, files = closed_edits(args, cohort, cap, portable)
    priors = checked_lines(files["occupancy"], old["edits_sha256"])
    points = checked_lines(files["point"], point["edits_sha256"])
    history = checked_lines(files["support"], support["edits_sha256"]) if support else [None] * cohort
    result = []
    for item, (audit, pred), change, previous, h10 in zip(items, predictions, points, priors, history, strict=True):
        validate_row(item, audit, pred, change, previous, h10)
        result.append((item, audit, pred, change, h10))
    observed = sum(len(pred["coco"]) for _, _, pred, _, _ in result)
    eligible = sum(len(audit["logit_ranks"]) for _, audit, _, _, _ in result)
    require(len(result) == cohort and observed == cap["counts"]["mapped"] == point["counts"]["mapped_candidates"]
            and eligible == cap["counts"]["eligible"] == point["counts"]["area_eligible"], "Original full image and mask census required")
    files.update(manifest=source.intake / portable.MANIFEST, predictions=source.capture / portable.PREDICTIONS,
                 audit=source.capture / portable.AUDIT)
    lineage = dict(cohort=cohort, images=len(result), mapped_candidates=observed, eligible=eligible,
                   checkpoint_sha256=cap["checkpoint_sha256"], manifest_sha256=cap["manifest_sha256"],
                   predictions_sha256=cap["predictions_sha256"], audit_sha256=cap["audit_sha256"],
                   capture_result_sha256=CAPTURE[cohort][0], capture_terminal_sha256=CAPTURE[cohort][1],
                   occupancy_result_sha256=OCCUPANCY[cohort][0], occupancy_terminal_sha256=OCCUPANCY[cohort][1],
                   occupancy_edits_sha256=old["edits_sha256"], point_result_sha256=POINT[cohort][0],
                   point_terminal_sha256=POINT[cohort][1], point_edits_sha256=point["edits_sha256"],
                   support_result_sha256=SUPPORT[0] if support else None, support_terminal_sha256=SUPPORT[1] if support else None,
                   support_edits_sha256=support["edits_sha256"] if support else None,
                   stream_bytes={key: path.stat().st_size for key, path in files.items()})
    return result, lineage
