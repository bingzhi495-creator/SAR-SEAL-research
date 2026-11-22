"""Decompose static-attraction versus learned editing on the closed UEIKAP hull panel."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import shapely


ARMS = ("SS", "LS", "SL", "LL", "DD", "native")
OLD = {"SS": "attraction_static_at_dynamic_count", "LL": "learned_at_attraction_count",
       "DD": "attraction_dynamic", "native": "raw"}
SIDES = {"addition": "LS", "deletion": "SL"}
IOU_PAIRS = (("LS", "SS"), ("LS", "LL"), ("SL", "SS"), ("SL", "LL"), ("LL", "SS"),
             ("LS", "SL"), ("SS", "DD"), ("LL", "DD"), ("LS", "DD"), ("SL", "DD"),
             ("SS", "native"), ("LL", "native"), ("DD", "native"))
OLD_SCORER = "b50f3035c1575fc417ced6b67a8ffeeb04a9cbe933f5c531b9b310c7c4792711"
MASK_CHECKER = "548258c6041187918c564c87e45d1744b3763d1895782bf0b26f6834d45ff89c"
PROTOCOL = "c0053d7cdf70372b1a2b8e41093a2cac3800982e7f05b3c3772618fabe7fd15a"
OLD_SCORE = "9b17a62bf9804ea481ef1c4eee909fb27e52f7620b14cd60789bb9a8dc1d5f5a"
OLD_SCORE_TERMINAL = "350d64ca74e1a81b28d1c75f02010b73f3bfb867457771b537518dd5b293cfd7"
FIELDS = ("continuous_iou_points", "intersection_per_hull_area_points")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Digest mismatch for " + path.name)
    return raw


def module(path, stamp):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pinned(path, stamp), str(path), "exec"), obj.__dict__)
    return obj


def old_population(args, old, core, rf, reference):
    binds = reference["inputs"]
    opts = SimpleNamespace(protocol=args.code.parent / "UEIKAP_ADDITIONAL_FAMILIES_PRELOCK_20261123.md",
                           intake=args.root / "intake", intake_terminal=args.root / "observed_intake/TERMINAL.json",
                           roster=args.root / "roster/ROSTER_PRIVATE.json", capture=args.root / "capture",
                           prepared=args.root / "prepare-edits", readback=args.root / "verify-edits/RESULT_SAFE.json",
                           readback_terminal=args.root / "observed_verify-edits/TERMINAL.json",
                           intake_terminal_sha=binds["intake_terminal_sha256"], intake_result_sha=binds["intake_result_sha256"],
                           readback_sha=binds["accepted_edit_receipt_sha256"], readback_terminal_sha=binds["accepted_edit_terminal_sha256"])
    return old.admitted_rows(opts, core, rf)


def summary(values, draws, groups, quantiles=(2.5, 97.5)):
    values = np.asarray(values, dtype=np.float64)
    ci = np.percentile(values[draws].mean(axis=1), quantiles).tolist()
    leave_out = (values.sum() - values) / (len(values)-1)
    return dict(mean_points=float(values.mean()), interval_points=ci, percentile_limits=list(quantiles),
                groups_improved=int((values > 0).sum()), groups_equal=int((values == 0).sum()),
                groups_worse=int((values < 0).sum()), by_family_points=dict(zip(groups, values.tolist(), strict=True)),
                leave_one_family_out_points=dict(zip(groups, leave_out.tolist(), strict=True)))


def summarize(groups, exposure):
    names = sorted(groups)
    means = np.stack([np.mean(groups[n], axis=0) for n in names])
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(names), size=(4096, len(names)))
    iou, normalized = means[:, :, 0], means[:, :, 1]
    i = ARMS.index
    increments = {side: normalized[:, i(arm)] - normalized[:, i("SS")] for side,arm in SIDES.items()}
    primary = {}
    for side, values in increments.items():
        stats = summary(values, draws, names, (1.25, 98.75))
        changed = exposure[side]
        covered = sum(changed.values()) >= 20 and len(changed) >= 8
        lo, hi = stats["interval_points"]
        primary[side] = dict(**stats, changed_original_matches=sum(changed.values()),
                             changed_families=len(changed), coverage_20_matches_8_families=covered,
                             beneficial_with_coverage=covered and lo > 0, adverse_with_coverage=covered and hi < 0)
    dominance = summary(increments["addition"]-increments["deletion"], draws, names)
    lower, upper = dominance["interval_points"]
    smaller = "addition" if upper < 0 else ("deletion" if lower > 0 else None)
    both = all(p["coverage_20_matches_8_families"] for p in primary.values())
    dominated = smaller if both and smaller and primary[smaller]["interval_points"][1] < 0 else None
    dominance.update(both_sides_covered=both, more_adverse_side=dominated, criterion_met=dominated is not None)
    total = normalized[:, i("LL")] - normalized[:, i("SS")]
    decomposition_residual = float(np.max(np.abs(total-increments["addition"]-increments["deletion"])))
    return dict(source_groups=len(names), hulls=sum(len(v) for v in groups.values()),
                arm_means_points={a:dict(zip(FIELDS, means[:, j].mean(axis=0).tolist(), strict=True))
                                  for j,a in enumerate(ARMS)}, primary_side_effects=primary,
                addition_minus_deletion=dominance,
                learned_minus_static_normalized_intersection=summary(total, draws, names),
                max_family_decomposition_residual_points=decomposition_residual,
                iou_contrasts={a+"_minus_"+b:summary(iou[:, i(a)]-iou[:, i(b)], draws, names) for a,b in IOU_PAIRS})


def geometry(core, checked, labels, intake):
    all_groups, matched_groups = defaultdict(list), defaultdict(list)
    exposure = {side:Counter() for side in SIDES}
    count = Counter()
    max_residual = 0.
    for (native, variants), label in zip(checked, labels, strict=True):
        family = label["acquisition_group"]
        truth, types = core.parse_yolo(core.label_path(intake, label["label_ref"]), core.HEIGHT, core.WIDTH, shapely)
        require(bool(truth) and types == label["count_by_class"], "Released hull/wake row census differs")
        matches = core.match_native(native, [raster for _,raster in truth])
        count.update(labeled_hulls=len(truth), native_matched=len(matches), native_unmatched=len(truth)-len(matches),
                     excluded_wakes=types.get("1",0))
        for ordinal, (polygon, _) in enumerate(truth):
            if ordinal not in matches:
                all_groups[family].append(np.zeros((len(ARMS),len(FIELDS))))
                continue
            pos = matches[ordinal]
            stock = native[pos-1]
            masks = variants.get(pos, dict.fromkeys(ARMS, stock))
            area = int(core.codec.area(stock))
            values, cache = {}, {}
            for arm, rle in masks.items():
                key = rle["counts"]
                if key not in cache:
                    iou = core.geometric_iou(rle, polygon, shapely)
                    intersection = iou*(area+polygon.area)/(1+iou)
                    cache[key] = (100*iou, 100*intersection/polygon.area, intersection)
                values[arm] = cache[key]
            residual = abs(values["LL"][2]+values["SS"][2]-values["LS"][2]-values["SL"][2])
            require(residual <= 1e-7, "Signed polygon-intersection additivity failed")
            max_residual = max(max_residual,residual)
            for side, arm in SIDES.items():
                if masks[arm]["counts"] != masks["SS"]["counts"]:
                    exposure[side][family] += 1
            row = np.asarray([values[a][:2] for a in ARMS])
            all_groups[family].append(row)
            matched_groups[family].append(row)
    require((count["labeled_hulls"], count["native_matched"], count["native_unmatched"], count["excluded_wakes"],
             len(all_groups), len(matched_groups)) == (528,121,407,396,13,10), "Historical match census differs")
    return dict(counts=dict(count), max_intersection_interaction_px2=max_residual, tolerance_px2=1e-7,
                primary_all_labeled_hulls=summarize(all_groups, exposure),
                originally_matched_hulls_sensitivity=summarize(matched_groups, exposure))


def check_parity(result, historical):
    for pop in ("primary_all_labeled_hulls", "originally_matched_hulls_sensitivity"):
        current, prior = result[pop], historical[pop]
        require(current["source_groups"] == prior["source_groups"], "Historical grouping differs")
        for new, old in OLD.items():
            require(abs(current["arm_means_points"][new][FIELDS[0]]-prior["arm_mean_iou_points"][old]) < 1e-9,
                    "Historical arm mean differs for " + new)
        for right in ("SS","DD"):
            new = current["iou_contrasts"]["LL_minus_"+right]
            old = prior["contrasts"][OLD["LL"]+" minus "+OLD[right]]
            require(abs(new["mean_points"]-old["mean_points"]) < 1e-9 and
                    np.allclose(new["interval_points"], old["paired_95_interval_points"], rtol=0, atol=1e-9),
                    "Historical paired contrast or bootstrap differs")


def read_replay(args):
    terminal_bytes = args.replay_terminal.read_bytes()
    event = json.loads(terminal_bytes)
    require(type(event["actual_exit"]) is int and event["actual_exit"] == 0 and event["output_footer_verified"] is True
            and event["phase"] == "replay", "Successful independent replay and verified footer required")
    raw = pinned(args.replay/"RESULT_SAFE.json", event["result_sha256"])
    receipt = json.loads(raw)
    require(receipt["status"] == "ATTRACTION_SIGNED_SIDES_UEIKAP342_REPLAY_COMPLETE"
            and receipt["phase"] == "replay" and receipt["worker_sha256"] == event["worker_sha256"]
            and receipt["protocol_sha256"] == args.protocol_sha256 and tuple(receipt["arms"]) == ARMS[:4]
            and receipt["images"] == 342 and receipt["groups"] == 13
            and all(receipt[k] is True for k in ("original_arms_exact", "per_candidate_signed_counts_exact",
                                                 "cross_context_signed_xor_exact"))
            and receipt["target_labels_opened"] is receipt["GT_opened"] is False
            and receipt["model_forwards"] == receipt["optimizer_updates"] == 0, "Unqualified label-free replay")
    edits = pinned(args.replay/"EDITS_PRIVATE.jsonl", receipt["edits_sha256"])
    return receipt, [json.loads(line) for line in edits.splitlines()], digest(terminal_bytes), digest(raw)


def checked_variants(rows, population, core, mask_checker, census):
    output, tally = [], Counter()
    for image, (row, (native, historical)) in enumerate(zip(rows, population, strict=True), 1):
        require(set(row) == {"image_id","changes"} and row["image_id"] == image, "Replay row identity differs")
        expected = {str(p) for p, choices in historical.items() if choices.get(OLD["DD"], (0,))[0] > 0}
        require(set(row["changes"]) == expected, "Historical positive dynamic-budget positions differ")
        checked = {}
        tally.update(mapped_candidates=len(native), positive_budget_candidates=len(expected))
        for key, variants in row["changes"].items():
            pos = int(key)
            prior = historical[pos]
            k, dynamic = prior[OLD["DD"]]
            require(set(variants) == set(ARMS[:4]) and 0 < k <= 8, "Crossed arm grid differs")
            masks = {}
            for arm, choice in variants.items():
                require(set(choice) == {"rle","adds","drops"} and type(choice["adds"]) is type(choice["drops"]) is int
                        and choice["adds"] == choice["drops"] == k, "Dynamic realized budget differs")
                masks[arm] = core.checked_rle(choice["rle"],core.HEIGHT,core.WIDTH)
            require(all(prior[OLD[a]] == (k, masks[a]) for a in ("SS","LL")), "Original pure-mask RLE mismatch")
            mask_checker(core.codec, dict(zip(("FF","IF","FI","II"), [masks[a] for a in ARMS[:4]], strict=True)),
                         native[pos-1], k)
            add, drop = masks["LS"] != masks["SS"], masks["SL"] != masks["SS"]
            tally.update(swaps_per_arm=k, adds_differ_candidates=int(add), drops_differ_candidates=int(drop),
                         both_sides_differ_candidates=int(add and drop))
            masks["DD"] = dynamic
            masks["native"] = native[pos-1]
            checked[pos] = masks
        output.append((native,checked))
    require(all(tally[k] == census[k] for k in tally) and len(rows) == 342, "Replay-wide candidate census differs")
    return output


def run(args):
    require(digest(args.protocol.read_bytes()) == args.protocol_sha256 == PROTOCOL, "Frozen protocol digest differs")
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2"
            and np.__version__ == "1.26.4", "Use the original pinned geometry runtime")
    receipt, rows, terminal_sha, receipt_sha = read_replay(args)
    old = module(args.code/"UEIKAP_ADDITIONAL_FAMILIES_20261123/score_additional.py", OLD_SCORER)
    historical = json.loads(pinned(args.root/"score/RESULT_SAFE.json", OLD_SCORE))
    old_event = old.observed(args.root/"observed_score/TERMINAL.json", OLD_SCORE_TERMINAL, OLD_SCORE, "score")
    require(old_event["worker_sha256"] == historical["inputs"]["worker_sha256"] == OLD_SCORER,
            "Historical scoring worker differs")
    inp, bind = receipt["input_lineage"], historical["inputs"]
    for key, source in (("predictions_sha256","predictions_sha256"),("audit_sha256","candidate_audit_sha256"),
                        ("old_edits_sha256","edits_sha256"),("manifest_sha256","images_roster_sha256"),
                        ("old_replay_result_sha256","accepted_edit_receipt_sha256"),
                        ("old_replay_terminal_sha256","accepted_edit_terminal_sha256"),
                        ("intake_result_sha256","intake_result_sha256"),("intake_terminal_sha256","intake_terminal_sha256")):
        require(inp[key] == bind[source], "Replay source differs for "+key)
    rf, core = old.methods(args.code)
    population, labels, _, _ = old_population(args, old, core, rf, historical)
    crossing = module(args.code/"INDEPENDENT_SIDE_FACTORIAL_20261123/score_sides.py", MASK_CHECKER)
    checked = checked_variants(rows, population, core, crossing.check_mask_sets, receipt["counts"])
    result = geometry(core, checked, labels, args.root/"intake")
    check_parity(result,historical)
    result.update(status="ATTRACTION_SIGNED_SIDES_UEIKAP342_SCORE_COMPLETE", images=342, arms=list(ARMS),
                  phase="score", primary_outcome="100 times the signed intersection change divided by each hull area",
                  exploratory_target=True, whole_scene_AP=False, geographic_holdout=False,
                  bootstrap_draws=4096, bootstrap_seed=5217, bootstrap_unit="acquisition family",
                  original_SS_LL_DD_exact=True, historical_IoU_parity_tolerance_points=1e-9,
                  protocol_sha256=args.protocol_sha256, worker_sha256=digest(Path(__file__).read_bytes()),
                  historical_score_sha256=OLD_SCORE, historical_score_terminal_sha256=OLD_SCORE_TERMINAL,
                  replay_sha256=receipt_sha, replay_terminal_sha256=terminal_sha, edits_sha256=receipt["edits_sha256"],
                  input_lineage=inp, model_forwards=0, optimizer_updates=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","root","protocol","replay","replay-terminal","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh score output directory")
    result = run(args)
    raw = (json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    args.output.mkdir()
    with (args.output/"RESULT_SAFE.json").open("xb") as output:
        output.write(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(raw))),flush=True)


if __name__ == "__main__":
    main()
