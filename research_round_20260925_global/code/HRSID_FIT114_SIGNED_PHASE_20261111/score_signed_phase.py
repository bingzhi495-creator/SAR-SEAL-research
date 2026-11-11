"""FIT-only assessment of signed one-pixel edits on frozen HRSID candidates."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec
from scipy.optimize import minimize
from scipy.special import expit


SIZE = 800
MIB = 1024**2
BUNDLE_BYTES = 151_320_868
BUNDLE_SHA = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
INDEX_SHA = "6f25f80ed0e0b9c3f0c57ab6cc3b25e947a102353757a33bca4d3823df3d547d"
ROLE_SHA = "ce308cf07c8b0d1e0db09f051f1a3a6699fc5a06b383029acda966bacfdbbfb6"
SALT = "HRSID-FIT114-P317-SIGNED-PHASE-v1"
HEADS = ("ordered", "phase", "permuted")
CONTROLS = ("abs_count", "sham_count", "ordered_count", "permuted_count")
ARMS = ("raw", *HEADS, *CONTROLS)
NUMERIC = ("edited_candidates", "edited_unmatched", "unmatched_pixels", "unmatched_add", "unmatched_delete", "add", "delete",
           "matched_correct_add", "matched_correct_delete", "margin_gain",
           "recover_75", "lose_75", "recover_90", "lose_90")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key_hash(*parts):
    return sha256(json.dumps([SALT, *parts], separators=(",", ":")).encode("ascii")).digest()


def split_roles(cohort):
    """Keep source prefixes and duplicate image encodings together."""
    require(len(cohort) == 114 and sum(r["core"] for r in cohort) == 109
            and sum(r["stress"] for r in cohort) == 5
            and all(type(r["core"]) is type(r["stress"]) is bool and r["core"] != r["stress"] for r in cohort)
            and len({r["image_id"] for r in cohort}) == 114, "Exact disjoint FIT114 core/stress images required")
    parents = {row["group"]: row["group"] for row in cohort}

    def find(group):
        while parents[group] != group:
            group = parents[group]
        return group

    first = {}
    for row in cohort:
        a = find(row["group"])
        b = find(first.setdefault(row["encoded_sha256"], row["group"]))
        parents[max(a, b)] = min(a, b)
    by_image = {row["image_id"]: find(row["group"]) for row in cohort}
    core = {by_image[row["image_id"]] for row in cohort if row["core"]}
    stress = {by_image[row["image_id"]] for row in cohort if row["stress"]}
    eligible = sorted(core - stress, key=lambda g: (key_hash("split", g), g))
    require(len(eligible) >= len(core)//2 > 0, "Too few independent core fitting groups")
    fitting = set(eligible[:len(core)//2])
    roles = {row["image_id"]: ("stress" if row["stress"] else
                              "fit" if by_image[row["image_id"]] in fitting else "held") for row in cohort}
    return roles, by_image


def split_digest(roles, groups):
    by_role = {role: sorted({groups[i] for i in roles if roles[i] == role}) for role in ("fit", "held", "stress")}
    return sha256(json.dumps(by_role, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def features(xy, inside, logits, candidate_key):
    """Use original-grid logits and the known 144-to-800 bilinear phase."""
    base = np.column_stack((logits, inside))
    phase = ((18 * xy - 41) % 100) / 100. - .5
    permuted = phase.copy()
    for side in (False, True):
        indices = np.flatnonzero(inside == side)
        seed = int.from_bytes(key_hash("phase_permutation", candidate_key, side)[:8], "big")
        permuted[indices] = phase[np.random.default_rng(seed).permutation(indices)]
    contrast = np.column_stack((logits[:, 5] - logits[:, 4], logits[:, 7] - logits[:, 2]))
    return dict(ordered=base, phase=np.column_stack((base, phase, phase*contrast)),
                permuted=np.column_stack((base, permuted, permuted*contrast)))


def choose_points(scores, inside, pixels, area, *, counts=None):
    """Rank each side under an abstaining cap or under fixed control counts."""
    cap = min(16, area//10)//2
    selected = []
    for side in (True, False):
        options = np.flatnonzero(inside == side)
        if counts is None:
            threshold = 10/19 if side else 9/19
            options = options[scores[options] > threshold]
            quota = cap
        else:
            quota = counts[side]
            require(0 <= quota <= min(cap, len(options)), "Count-matched side exceeds candidate support")
        order = np.lexsort((pixels[options], -scores[options]))
        selected.extend(options[order[:quota]])
    return np.asarray(selected, dtype=np.intp)


def score_edit(intersection, union, inside, truth):
    added_tp = int((~inside & truth).sum())
    added_fp = int((~inside & ~truth).sum())
    removed_tp = int((inside & truth).sum())
    removed_fp = int((inside & ~truth).sum())
    return intersection + added_tp - removed_tp, union + added_fp - removed_fp


def sha_file(stream):
    digest = sha256()
    for block in iter(lambda: stream.read(MIB), b""):
        digest.update(block)
    return digest.hexdigest()


def load_scope(path):
    with path.open("rb") as stream:
        require(os.fstat(stream.fileno()).st_size == BUNDLE_BYTES and sha_file(stream) == BUNDLE_SHA,
                "Exact sealed FIT114 archive required")
        stream.seek(0)
        with ZipFile(stream) as bundle:
            out = []
            for name, expected, limit in (
                ("COHORT_PRIVATE.json", COHORT_SHA, MIB), ("FIT114_COCO_PRIVATE.json", INDEX_SHA, 4*MIB)
            ):
                require(bundle.namelist().count(name) == 1 and bundle.getinfo(name).file_size <= limit,
                        "Exact bounded FIT index member required")
                raw = bundle.read(name)
                require(sha256(raw).hexdigest() == expected, "FIT member digest differs")
                out.append(json.loads(raw))
    cohort, index = out
    require(cohort["fit_only"] is True and len(index["images"]) == 114, "Only the original FIT114 scope is allowed")
    return cohort["images"], index


def stream_capture(stream, cohort, expected):
    stream.seek(0)
    hashed = sha256()
    for item in cohort:
        raw = stream.readline(4*MIB + 1)
        require(0 < len(raw) <= 4*MIB and raw.endswith(b"\n"), "Bounded complete capture image row required")
        hashed.update(raw)
        row = json.loads(raw)
        fields = ("image_id", "group", "encoded_sha256", "core", "stress")
        require(row.keys() == {*fields, "candidates"} and all(row[key] == item[key] for key in fields),
                "Capture image provenance/order differs from sealed FIT114")
        yield row
    require(not stream.read(1) and hashed.hexdigest() == expected, "Capture stream digest or image count differs")


def rle(segmentation):
    require(segmentation.keys() == {"size", "counts"} and segmentation["size"] == [SIZE, SIZE]
            and isinstance(segmentation["counts"], str) and segmentation["counts"].isascii(),
            "Exact original-grid compressed prediction mask required")
    return dict(size=segmentation["size"], counts=segmentation["counts"].encode("ascii"))


def ground_truth(index, cohort):
    images = {item["id"]: item for item in index["images"]}
    require(len(images) == 114 and all((images[r["image_id"]]["height"], images[r["image_id"]]["width"],
                images[r["image_id"]]["file_name"]) == (SIZE, SIZE, r["native_basename"]) for r in cohort),
            "FIT image IDs, basenames or sizes differ")
    by_image = {key: [] for key in images}
    for item in sorted(index["annotations"], key=lambda r: r["id"]):
        require(item["image_id"] in by_image and item["category_id"] == 1, "Only sealed FIT ship labels required")
        if item.get("iscrowd", 0) or item.get("ignore", 0):
            continue
        raw = item["segmentation"]
        mask = rle(raw) if isinstance(raw, dict) else codec.merge(codec.frPyObjects(raw, SIZE, SIZE))
        if codec.area(mask):
            by_image[item["image_id"]].append(mask)
    return by_image


def image_masks(rows):
    require(1 <= len(rows) <= 100, "Native mapped candidate count differs")
    ranks = [row["native_rank"] for row in rows]
    queries = [row["query_id"] for row in rows]
    require(ranks == sorted(set(ranks)) and all(type(v) is int and 1 <= v <= 100 for v in ranks)
            and len(set(queries)) == len(queries) and all(type(v) is int and 0 <= v < 164 for v in queries)
            and all(row.keys() == {"native_rank", "query_id", "score", "category_id", "segmentation", "points"}
                    and row["category_id"] == 1 and 0 <= row["score"] <= 1 for row in rows),
            "Native ranks, IDs, class or score differ")
    predictions = [rle(row["segmentation"]) for row in rows]
    return predictions, codec.area(predictions)


def original_matches(predictions, truth):
    if not truth:
        return {}
    ious = codec.iou(predictions, truth, [0]*len(truth))
    free = np.ones(len(truth), dtype=bool)
    matches = {}
    for i, values in enumerate(ious):
        values = np.where(free, values, -1)
        best = int(values.argmax())
        if values[best] >= .50:
            free[best] = False
            matches[i] = truth[best]
    return matches


def point_bank(row, area, key):
    points = row["points"]
    require(type(points) is list and len(points) <= 64 and (area > 0 or not points)
            and all(p.keys() == {"x", "y", "inside", "logits"} and type(p["inside"]) is bool
                    and type(p["x"]) is type(p["y"]) is int and len(p["logits"]) == 9 for p in points),
            "Exact bounded prediction-only sparse bank required")
    xy = np.asarray([(p["x"], p["y"]) for p in points], dtype=np.intp).reshape(-1, 2)
    inside = np.asarray([p["inside"] for p in points], dtype=bool)
    logits = np.asarray([p["logits"] for p in points], dtype=np.float64).reshape(-1, 9)
    pixels = xy[:, 1]*SIZE + xy[:, 0]
    # The binary band sees outside-canvas background. Logit features clamp
    # outside samples to the nearest canvas pixel.
    border = np.any((xy == 0) | (xy == SIZE-1), axis=1)
    has_opposite = np.any((logits[:, 1:] > 0) != inside[:, None], axis=1)
    require(len(np.unique(pixels)) == len(pixels) and np.all((xy >= 0) & (xy < SIZE))
            and np.isfinite(logits).all() and np.array_equal(logits[:, 0] > 0, inside)
            and (inside.sum() <= 32) and ((~inside).sum() <= 32)
            and np.all(has_opposite | (inside & border)),
            "Sparse point uniqueness, bounds, sign or boundary band differs")
    return xy, inside, pixels, logits, features(xy, inside, logits, key)


def truth_bits(gt, xy):
    return codec.decode(gt)[xy[:, 1], xy[:, 0]].astype(bool)


def fit_heads(stream, digest, cohort, roles, groups, truth):
    blocks = defaultdict(list)
    for image in stream_capture(stream, cohort, digest):
        image_id = image["image_id"]
        if roles[image_id] != "fit":
            continue
        rows = image["candidates"]
        predictions, areas = image_masks(rows)
        for i, gt in original_matches(predictions, truth[image_id]).items():
            row = rows[i]
            xy, inside, _, _, blocks_by_head = point_bank(row, areas[i], f"{image_id}/{row['query_id']}")
            if len(xy):
                require(np.array_equal(truth_bits(predictions[i], xy), inside), "Matched native RLE signs differ")
                label = truth_bits(gt, xy) != inside
                weights = np.zeros(len(xy))
                sides = [inside == s for s in (False, True) if np.any(inside == s)]
                for mask in sides:
                    weights[mask] = 1 / (len(sides) * mask.sum())
                blocks[groups[image_id]].append((blocks_by_head, label, weights))
    require(len(blocks) >= 2, "At least two independent fitted source groups with fixed matches required")
    fit_rows = [(data, target, weight/len(group_rows)) for group_rows in blocks.values()
                for data, target, weight in group_rows]
    label = np.concatenate([row[1] for row in fit_rows]).astype(np.float64)
    weight = np.concatenate([row[2] for row in fit_rows])
    weight /= weight.sum()
    require(label.min() == 0 and label.max() == 1, "Both edit labels required in the fitted groups")
    models = {}
    for name in HEADS:
        data = np.concatenate([row[0][name] for row in fit_rows])
        center = np.average(data, weights=weight, axis=0)
        scale = np.sqrt(np.average((data-center)**2, weights=weight, axis=0))
        scale[scale < 1e-8] = 1
        data = np.column_stack(((data-center)/scale, np.ones(len(data))))

        def objective(params):
            z = data @ params
            loss = weight @ (np.logaddexp(0., z) - label*z) + .005*(params[:-1] @ params[:-1])
            grad = data.T @ (weight*(expit(z)-label))
            grad[:-1] += .01*params[:-1]
            return loss, grad

        fit = minimize(objective, np.zeros(data.shape[1]), method="L-BFGS-B", jac=True,
                       options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-7})
        require(fit.success and np.isfinite(fit.x).all(), "Fixed logistic fit failed")
        models[name] = (center, scale, fit.x)
    return models, dict(groups_with_points=len(blocks), matched_candidates=len(fit_rows),
                        sampled_points=len(label), correct_edit_points=int(label.sum()))


def select_edits(xy, inside, pixels, logits, data, models, area, key):
    probabilities = {}
    for name, (center, scale, coef) in models.items():
        probabilities[name] = expit(((data[name]-center)/scale) @ coef[:-1] + coef[-1])
    edits = {"raw": np.empty(0, dtype=np.intp)}
    for name in HEADS:
        edits[name] = choose_points(probabilities[name], inside, pixels, area)
    phase = inside[edits["phase"]]
    counts = {True: int(phase.sum()), False: int((~phase).sum())}
    if not len(phase):
        edits.update({name: edits["raw"] for name in CONTROLS})
        return edits
    seed = int.from_bytes(key_hash("count_matched_sham", key)[:8], "big")
    scores = dict(abs_count=-np.abs(logits[:, 0]), sham_count=np.random.default_rng(seed).random(len(xy)),
                  ordered_count=probabilities["ordered"], permuted_count=probabilities["permuted"])
    for name, values in scores.items():
        edits[name] = choose_points(values, inside, pixels, area, counts=counts)
    return edits


def score_candidate(counter, selected, inside, truth, intersection, union):
    side = inside[selected]
    adds, deletes = int((~side).sum()), int(side.sum())
    counter["edited_candidates"] += bool(len(selected))
    counter["add"] += adds
    counter["delete"] += deletes
    if truth is None:
        counter["edited_unmatched"] += bool(len(selected))
        counter["unmatched_pixels"] += len(selected)
        counter["unmatched_add"] += adds
        counter["unmatched_delete"] += deletes
        return
    target = truth[selected]
    counter["matched_correct_add"] += int((~side & target).sum())
    counter["matched_correct_delete"] += int((side & ~target).sum())
    new_i, new_u = score_edit(intersection, union, side, target)
    require(0 <= new_i <= new_u and new_u > 0, "Edited exact raster counts are inconsistent")
    counter["margin_gain"] += 10*(new_i-intersection) - 9*(new_u-union)
    counter["fixed_iou_sum"] += new_i/new_u
    for threshold in (75, 90):
        before = 100*intersection >= threshold*union
        after = 100*new_i >= threshold*new_u
        counter[f"recover_{threshold}"] += int(not before and after)
        counter[f"lose_{threshold}"] += int(before and not after)


def evaluate_roles(stream, digest, cohort, roles, groups, truth, models):
    census = {"held": Counter(), "stress": Counter()}
    totals = {role: {arm: Counter() for arm in ARMS} for role in census}
    held_groups = defaultdict(Counter)
    for image in stream_capture(stream, cohort, digest):
        image_id = image["image_id"]
        role = roles[image_id]
        if role == "fit":
            continue
        rows = image["candidates"]
        predictions, areas = image_masks(rows)
        matches = original_matches(predictions, truth[image_id])
        observed = census[role]
        observed.update(images=1, candidates=len(rows), fixed_matches=len(matches),
                        eligible_gt=len(truth[image_id]), unmatched_candidates=len(rows)-len(matches))
        group = held_groups[groups[image_id]] if role == "held" else Counter()
        group["matches"] += len(matches)
        for i, (row, area) in enumerate(zip(rows, areas, strict=True)):
            key = f"{image_id}/{row['query_id']}"
            xy, inside, pixels, logits, data = point_bank(row, area, key)
            edits = select_edits(xy, inside, pixels, logits, data, models, area, key)
            gt = matches.get(i)
            gt_bits = None
            intersection = union = 0
            if gt is not None:
                require(np.array_equal(truth_bits(predictions[i], xy), inside), "Held native RLE signs differ")
                gt_bits = truth_bits(gt, xy)
                intersection = int(codec.area(codec.merge([predictions[i], gt], intersect=True)))
                union = int(area) + int(codec.area(gt)) - intersection
                require(2*intersection >= union > 0, "Original exact fixed match is below .50")
                observed["original_75"] += int(100*intersection >= 75*union)
                observed["original_90"] += int(100*intersection >= 90*union)
            for arm, selected in edits.items():
                values = Counter()
                score_candidate(values, selected, inside, gt_bits, intersection, union)
                totals[role][arm].update(values)
                group[arm] += values["margin_gain"]
    return census, totals, held_groups


def group_interval(groups, control):
    differences = [(v["phase"]-v[control])/v["matches"] for _, v in sorted(groups.items()) if v["matches"]]
    if not differences:
        return dict(groups=0, mean=None, lower=None, upper=None, positive_groups=0)
    values = np.asarray(differences)
    draws = np.random.default_rng(317114).integers(0, len(values), size=(4096, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return dict(groups=len(values), mean=float(values.mean()), lower=float(low), upper=float(high),
                positive_groups=int((values > 0).sum()))


def summarize(roles, groups, census, totals, held_groups):
    out = {}
    for role, observed in census.items():
        arms = {}
        for arm, values in totals[role].items():
            arms[arm] = {field: int(values[field]) for field in NUMERIC}
            for action in ("add", "delete"):
                arms[arm][f"matched_wrong_{action}"] = (
                    values[action] - values[f"unmatched_{action}"] - values[f"matched_correct_{action}"])
            arms[arm]["net_90"] = int(values["recover_90"]-values["lose_90"])
            arms[arm]["mean_fixed_iou"] = values["fixed_iou_sum"]/observed["fixed_matches"] if observed["fixed_matches"] else None
        out[role] = dict(census=dict(observed), arms=arms)
    split_groups = {role: {groups[i] for i in roles if roles[i] == role} for role in ("fit", "held", "stress")}
    held = out["held"]["arms"]
    comparisons = {arm: group_interval(held_groups, arm) for arm in ("raw", *CONTROLS)}
    strict = census["held"]["original_90"]
    gate = dict(
        beats_raw_and_same_count_control_margins=all(
            values["lower"] is not None and values["lower"] > 0 for values in comparisons.values()),
        positive_strict_net_over_same_count_controls=held["phase"]["net_90"] > max(
            held[arm]["net_90"] for arm in ("raw", *CONTROLS)),
        original_strict_losses_at_most_5_percent=bool(strict) and 20*held["phase"]["lose_90"] <= strict,
    )
    return dict(splits={role: dict(images=sum(r == role for r in roles.values()), groups=len(group))
                        for role, group in split_groups.items()},
                held_stress_shared_groups=len(split_groups["held"] & split_groups["stress"]),
                held_group_bootstrap_phase_margin_minus_control=comparisons,
                result=out, phase_gate=gate, advance=all(gate.values()))


def run(capture, digest, bundle):
    require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "Exact closed capture SHA required")
    cohort, index = load_scope(bundle)
    roles, groups = split_roles(cohort)
    role_digest = split_digest(roles, groups)
    require(role_digest == ROLE_SHA, "Independent pre-capture source role digest differs")
    with capture.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= 256*MIB and sha_file(stream) == digest, "Capture stream identity differs")
        gt = ground_truth(index, cohort)
        models, fitting = fit_heads(stream, digest, cohort, roles, groups, gt)
        scores = evaluate_roles(stream, digest, cohort, roles, groups, gt, models)
        after = os.fstat(stream.fileno())
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "Closed capture changed during scoring")
    return dict(status="HRSID_FIT114_SIGNED_PHASE_POINT_SCORE_COMPLETE_V1",
                capture_sha256=digest, fit_bundle_sha256=BUNDLE_SHA, fit_cohort_sha256=COHORT_SHA,
                fit_index_sha256=INDEX_SHA, split_sha256=role_digest, fitting=fitting,
                scope=dict(FIT=True, DEV=False, TEST=False, AP=False, model_forward_calls=0,
                           detector_optimizer_updates=0, prediction_only_edit_selection=True,
                           unmatched_edit_correctness_not_evaluated=True, role_groups_are_source_proxies=True),
                **summarize(roles, groups, *scores))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "bundle", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--capture-sha256", required=True)
    args = cli.parse_args()
    for path in (args.capture, args.bundle):
        require(path.is_absolute() and path.resolve(strict=True) == path, "Canonical input file required")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh canonical score output required")
    result = run(args.capture, args.capture_sha256, args.bundle)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
