"""Measure a bounded, symmetric same-query-neighbor FIT proposal ceiling."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
from scipy import ndimage as ndi

PLAN_SHA = "83626640b6619526c9e1e6e933ac2499f2e69586c05eb6bc08027484662348f2"
ENVELOPE_SOURCE_SHA = "1f3f49bf4f9c7afff5934f36080666255ab8c10023dfd92533c57a3302d955f2"
ENVELOPE_RESULT_SHA = "2dfc9674d59cad6a322210ae054268f56fb7a9896448edf8d3669d4fc2fe63d5"
ENVELOPE_TOOL_SHA = "ea4dc547d4ac77658f08b69dc8779c70e112b036a14e871826e21fa26cb36586"
ENVELOPE_WAIT_SHA = "a5c644b0e7474ad1260b71ed2a48dc893893d998c4e5455861174a153c04ceeb"
CONNECTIVITY = np.ones((3, 3), dtype=bool)
MENUS = ("peer", "contour")


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def read_exact(path, expected, maximum):
    need(path == path.resolve(strict=True), "Expected a canonical source")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    need(0 < len(raw) <= maximum and sha256(raw).hexdigest() == expected, "Wrong pinned source: " + path.name)
    return raw


def closest(eligible, depth, k):
    ids = np.flatnonzero(eligible)
    return ids[np.argsort(depth.ravel()[ids], kind="stable")[:k]]


def match_depth(eligible, depth, selected):
    flat = depth.ravel()
    ids = np.flatnonzero(eligible)
    return np.concatenate([ids[flat[ids] == level][:np.count_nonzero(flat[selected] == level)]
                           for level in (1, 2)])


def proposal(anchor, donor, editable):
    depth_add = ndi.distance_transform_cdt(np.pad(~anchor, 1, constant_values=True), metric="chessboard")[1:-1, 1:-1]
    depth_trim = ndi.distance_transform_cdt(np.pad(anchor, 1), metric="chessboard")[1:-1, 1:-1]
    add = editable & ~anchor & (depth_add >= 1) & (depth_add <= 2)
    trim = editable & anchor & (depth_trim >= 1) & (depth_trim <= 2)
    k = min(int(anchor.sum()) // 20, int((add & donor).sum()), int((trim & ~donor).sum()))
    selected = {"add": closest(add & donor, depth_add, k), "trim": closest(trim & ~donor, depth_trim, k)}
    controls = {"add": match_depth(add, depth_add, selected["add"]),
                "trim": match_depth(trim, depth_trim, selected["trim"])}
    need(all(len(p) == k for menu in (selected, controls) for p in menu.values()), "Equal signed edit budgets")
    return {"k": k, "peer": selected, "contour": controls}


def permitted(original, candidate):
    return (candidate["core_recall"] + 0.0025 + 1e-12 >= original["core_recall"]
            and candidate["new_bridge_count"] == candidate["splits"] == 0 and candidate["max_fraction"] < .01)


def oracle(original, add, trim, constrain):
    choices = {"anchor": original, "add": add, "trim": trim}
    return max((name for name, case in choices.items() if not constrain or permitted(original, case)),
               key=lambda name: choices[name]["iou"])


def load_previous(root):
    prior_draft = root / "P383_HRSID_FIT_LARGE_EDIT_ENVELOPE_DRAFT_PRIVATE_20261111_01"
    prior_result = root / "P383_HRSID_FIT_LARGE_EDIT_ENVELOPE_ACTUAL_PRIVATE_20261111_01"
    tool = json.loads(read_exact(prior_draft / "TOOL_TERMINAL_PRIVATE.json", ENVELOPE_TOOL_SHA, 16384))
    wait = json.loads(read_exact(prior_result / "ORIGINAL_WAIT_PRIVATE.json", ENVELOPE_WAIT_SHA, 16384))
    footer = json.loads(tool["original_exec"]["output"].splitlines()[-1])
    report = json.loads(read_exact(prior_result / "RESULT_SAFE.json", ENVELOPE_RESULT_SHA, 1 << 20))
    need(tool["terminal_observed"] and tool["original_exec"]["exit_code"] == 0 and wait["original_os_wait"]
         and wait["original_os_exit"] == 0 and wait["process_group_closed"] and wait["stop_reason"] is None
         and footer["wait_sha256"] == ENVELOPE_WAIT_SHA and footer["result_sha256"] == ENVELOPE_RESULT_SHA
         and report["FIT"] and report["DEV"] is report["TEST"] is report["AP"] is False,
         "Exact closed original P383 FIT custody")
    src = prior_draft / "envelope.py"
    envelope = ModuleType("_original_closed_envelope")
    envelope.__file__ = str(src)
    exec(compile(read_exact(src, ENVELOPE_SOURCE_SHA, 65536), str(src), "exec"), envelope.__dict__)
    original, rows, previous, phase = envelope.setup()
    return envelope, original, rows, previous, phase


def split_count(original_labels, original_count, candidate):
    after, _ = ndi.label(candidate, CONNECTIVITY)
    representatives = np.unique(after[candidate], return_index=True)[1]
    counts = np.bincount(original_labels[candidate][representatives], minlength=original_count + 1)[1:]
    return int(np.count_nonzero(counts > 1))


def case_metrics(previous, candidate, truth, core, union_all, anchor, neighbors, labels, label_count):
    result = previous.evaluate(candidate, truth, core, union_all, anchor, neighbors)
    need(result["core_recall"] is not None, "Original large core is required")
    total = int(truth.sum())
    result.update(splits=split_count(labels, label_count, candidate),
                  fp_gt=(result["union_pixels"] - total) / total,
                  fn_gt=(total - result["intersection_pixels"]) / total)
    return result


def collect_target(previous, anchor, donor, truth, core, union_all, neighbors, phase, prior):
    yy, xx = np.nonzero(anchor)
    roi = np.zeros_like(anchor)
    roi[max(0, int(yy.min()) - 16):int(yy.max()) + 17, max(0, int(xx.min()) - 16):int(xx.max()) + 17] = True
    labels, label_count = ndi.label(anchor, CONNECTIVITY)
    args = (truth, core, union_all, anchor, neighbors, labels, label_count)
    original = case_metrics(previous, anchor, *args)
    out = []
    for fit, fold in enumerate(prior["folds"]):
        old = fold["values"]["anchor"]
        for field in ("intersection_pixels", "union_pixels", "core_true_pixels", "core_covered_pixels"):
            need(original[field] == old[field], "Exact original per-fold raster and core counts")
        editable = roi & (phase == 1-fit)
        held = phase == 1-fit
        need(int((anchor & core & held).sum()) == old["held_core_covered_pixels"]
             and int((core & held).sum()) == old["held_core_pixels"], "Original held core counts")
        plans = proposal(anchor, donor, editable) if donor is not None else None
        k = plans["k"] if plans is not None else 0
        values = {"anchor": original}
        choices = {}
        for key in MENUS:
            for action in ("add", "trim"):
                if k == 0:
                    values[key + "_" + action] = original
                    continue
                candidate = anchor.copy()
                candidate.ravel()[plans[key][action]] = action == "add"
                values[key + "_" + action] = case_metrics(previous, candidate, *args)
            add, trim = values[key + "_add"], values[key + "_trim"]
            for constrained in (False, True):
                label = key + ("_constrained" if constrained else "_oracle")
                choice = oracle(original, add, trim, constrained)
                choices[label] = choice
                values[label] = original if choice == "anchor" else values[key + "_" + choice]
        out.append(dict(fit=fit, edits_per_active_action=k, values=values, choices=choices))
    return out


def collect(envelope, previous, phase, reference):
    streams = previous.load_p378()
    roster = json.loads(envelope.pinned(previous.P378_DRAFT / "IMAGE_ONLY_MANIFEST_PRIVATE.json",
                                        previous.MANIFEST_SHA, 65536))
    need(roster["FIT"] and roster["DEV"] is roster["TEST"] is False, "Exact FIT-only roster")
    wanted = {row["image_id"]: row["native_basename"].split("_", 1)[0] for row in roster["images"]}
    source = json.loads(envelope.pinned(previous.INDEX, previous.INDEX_SHA, 32 * 1024**2))
    images = {row["id"]: row for row in source["images"] if row["id"] in wanted}
    by_image = defaultdict(list)
    for ann in source["annotations"]:
        if ann["image_id"] in wanted:
            by_image[ann["image_id"]].append(ann)
    need(len(wanted) == len(images) == 28 and len(set(wanted.values())) == 4
         and len(source["images"]) == 2937 and len(source["annotations"]) == 9260, "Original frozen FIT population")
    result = {}
    for arm in previous.SIDES:
        prior = {row["annotation_id"]: row for row in reference[arm]}
        rows = []
        for block in streams[arm][0]:
            iid = block["image_id"]
            anns = by_image[iid]
            pool, _, _ = previous.ranked_candidates(block)
            matches, rles, _ = previous.freeze_coco(images[iid], anns, pool)
            indexed = {item["rank"]: item for item in pool}
            targets = {key: mask.decode(rle).astype(bool) for key, rle in rles.items()}
            foreground = [ann["id"] for ann in anns if not ann.get("iscrowd", 0)]
            union_all = np.logical_or.reduce([targets[key] for key in foreground])
            for ann in anns:
                if float(ann["area"]) < 9216:
                    continue
                aid = ann["id"]
                row = prior[aid]
                need(row["matched"] and matches.get(aid) == row["anchor_rank"] and row["group"] == wanted[iid],
                     "Exact original anchor and fold identity")
                item = indexed[row["anchor_rank"]]
                anchor = mask.decode(item["rle"]).astype(bool)
                truth = targets[aid]
                core = ndi.binary_erosion(truth, previous.CORE, border_value=0)
                exclusive = [targets[key] & ~truth for key in foreground if key != aid]
                neighbors = [(n, ndi.binary_dilation(n, CONNECTIVITY)) for n in exclusive if n.any()]
                donor_item, _, _ = previous.first_donor(item, anchor, pool)
                donor = mask.decode(donor_item["rle"]).astype(bool) if donor_item is not None else None
                folds = collect_target(previous, anchor, donor, truth, core, union_all, neighbors, phase, row)
                rows.append(dict(annotation_id=aid, group=wanted[iid], has_donor=donor is not None, folds=folds))
        need(len(rows) == len({row["annotation_id"] for row in rows}) == 50, "Exactly 50 original large targets")
        result[arm] = rows
        print(json.dumps(dict(event="EXACT_ANCHORS_AND_BOUNDARY_OPTIONS_READY", arm=arm, targets=len(rows))), flush=True)
    return result, sorted(set(wanted.values()))


def aggregate(rows, names):
    keys = ("iou", "core_recall", "fp_gt", "fn_gt")
    grouped = {group: [row for row in rows if row["group"] == group] for group in names}
    metrics = {}
    for method in rows[0]["folds"][0]["values"]:
        values = {name: [row["folds"] for row in group] for name, group in grouped.items()}
        group_metrics = {str(i): {key: mean(mean(f["values"][method][key] for f in row) for row in values[name])
                                  for key in keys} for i, name in enumerate(names, 1)}
        folds = [f for row in rows for f in row["folds"]]
        chosen = [f["values"][method] for f in folds]
        metrics[method] = dict(equal_prefix={key: mean(a[key] for a in group_metrics.values()) for key in keys},
                               groups=group_metrics, strict=int(sum(f["pass_90"] for f in chosen)),
                               splits=int(sum(f["splits"] for f in chosen)),
                               new_bridges=int(sum(f["new_bridge_count"] for f in chosen)),
                               max_neighbor_invasion=max(f["max_fraction"] for f in chosen))
        if method.endswith(("_oracle", "_constrained")):
            edited = [r for r in rows if any(f["choices"][method] != "anchor" for f in r["folds"])]
            metrics[method].update(edited_targets=len(edited), edited_groups=len({r["group"] for r in edited}),
                                   selected={k: sum(f["choices"][method] == k for f in folds)
                                             for k in ("anchor", "add", "trim")})
    return dict(metrics=metrics, group_target_counts=[len(grouped[n]) for n in names],
                donor_targets=sum(r["has_donor"] for r in rows),
                eligible_folds=sum(f["edits_per_active_action"] > 0 for r in rows for f in r["folds"]),
                matched_area_total=sum(f["edits_per_active_action"] for r in rows for f in r["folds"]))


def write_new(path, document):
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256(raw).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path)
    args = parser.parse_args()
    local = Path(__file__).resolve().parent
    root = (args.prior_root or local.parent).resolve(strict=True)
    need(os.name == "nt" and sys.version_info[:3] == (3,10,13)
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Pinned Windows CPU runtime")
    read_exact(local / "PROTOCOL.md", PLAN_SHA, 16384)
    start = perf_counter()
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Closed local FIT only")
    sys.addaudithook(offline)
    envelope, old, reference, previous, phase = load_previous(root)
    rows, groups = collect(envelope, previous, phase, reference)
    arms = {arm: aggregate(cases, groups) for arm, cases in rows.items()}
    for arm in previous.SIDES:
        observed = arms[arm]["metrics"]["anchor"]["equal_prefix"]
        expected = old["arms"][arm]["equal_prefix"]["anchor"]
        need(abs(observed["iou"] - expected["mean_iou"]) < 1e-14
             and abs(observed["core_recall"] - expected["mean_core_recall"]) < 1e-14,
             "Exact original equal-prefix FIT anchor")
    center = arms["center"]["metrics"]
    active = center["peer_constrained"]
    original = center["anchor"]
    delta = 100 * (active["equal_prefix"]["iou"] - original["equal_prefix"]["iou"])
    contrast = 100 * (active["equal_prefix"]["iou"] - center["contour_constrained"]["equal_prefix"]["iou"])
    checks = dict(iou_vs_anchor=delta >= .50, iou_vs_contour=contrast >= .25,
                  targets=active["edited_targets"] >= 10, groups=active["edited_groups"] >= 3,
                  strict=active["strict"] >= original["strict"])
    row_sha = write_new(args.output.with_name("ROW_AUDIT_PRIVATE.json"), rows)
    result = dict(status="HRSID_FIT_LARGE_PAIRED_BOUNDARY_MENU_ACTUAL", FIT=True, DEV=False, TEST=False,
                  AP=False, model_forward_calls=0, optimizer_updates=0, images=28, large_targets=50,
                  source_groups=4, arms=arms, protocol_sha256=PLAN_SHA, prior_result_sha256=ENVELOPE_RESULT_SHA,
                  private_row_audit_sha256=row_sha, elapsed_seconds=perf_counter()-start,
                  gate=dict(advance=all(checks.values()), checks=checks,
                            peer_minus_anchor_iou_points=delta, peer_minus_contour_iou_points=contrast))
    digest = write_new(args.output, result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest)), flush=True)


if __name__ == "__main__":
    main()

