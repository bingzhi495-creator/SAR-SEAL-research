"""Test only the supervision target of the frozen 1,321-parameter point head."""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO


SEEDS = ((5217, 20261112), (5317, 20261212), (5417, 20261312))
TARGETS = ("fractional", "threshold50", "coco_raster")
HISTORICAL = ("raw", "fixed", "free_natural", "free_matched_count")
CLASSICAL = "acb7e00e7dfd2c60e5afc4397e073a6fabcb6a4ed977f1c8eaf89115724b52ed"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_exact(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact source bytes: " + path.name)
    return raw


def module(path, digest):
    result = ModuleType(path.stem)
    result.__file__ = str(path)
    exec(compile(read_exact(path, digest), str(path), "exec"), result.__dict__)
    return result


def mode_for(init_seed, target):
    return "free_matched_count" if init_seed == SEEDS[0][0] and target == "fractional" else f"{target}_{init_seed}"


def target_populations(slots, gt):
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    ann = {r["id"]: r for r in gt["annotations"]}
    threshold, raster = [], []
    counts = Counter()
    for slot in slots:
        q = slot["truth"]
        if q is None:
            threshold.append(slot)
            raster.append(slot)
            continue
        iid, aid = slot["row"]["image_id"], slot["gt_id"]
        coords = slot["data"]["coordinates"]
        record = ann[aid]
        rle = coco.annToRLE(record)
        need(record["image_id"] == iid and slot["role"] == "valid"
             and tuple(rle["size"]) == tuple(slot["row"]["segmentation"]["size"]) and coords.shape == q.shape,
             "Both labels attach to the original matched annotation and identical cells")
        binary = codec.decode(rle).ravel(order="C")[coords].astype(np.float32)
        cutoff = (q >= 0.50).astype(np.float32)
        threshold.append(dict(slot, truth=cutoff))
        raster.append(dict(slot, truth=binary))
        counts.update(matches=1, pairs=len(q), cells=q.size, exact_zero=int(np.count_nonzero(q == 0)),
                      exact_half=int(np.count_nonzero(q == .50)), exact_one=int(np.count_nonzero(q == 1)),
                      intermediate=int(np.count_nonzero((q > 0) & (q < 1))),
                      threshold_positive=int(cutoff.sum()), raster_positive=int(binary.sum()),
                      threshold_raster_disagreements=int(np.count_nonzero(cutoff != binary)),
                      raster_positive_below_half=int(np.count_nonzero((binary == 1) & (cutoff == 0))),
                      raster_negative_at_least_half=int(np.count_nonzero((binary == 0) & (cutoff == 1))))
    return dict(fractional=slots, threshold50=threshold, coco_raster=raster), dict(counts)


def seed_trainer(a, learner, source_digest, init_seed, sample_seed):
    if (init_seed, sample_seed) == SEEDS[0]:
        return learner.train
    text = read_exact(a.training, source_digest).decode("utf-8").replace("\r\n", "\n")
    body = text[text.index("def train("):text.index("\n\ndef gains(")]
    seams = {
        "    own_t=torch.from_numpy(standard)\n": "",
        '        partner=np.concatenate([s["data"]["contexts"][arm] for s in train_slots])\n'
        '        partner_t=torch.from_numpy(np.asarray((partner-center)/scale,dtype=np.float32))':
        "        own_t=torch.from_numpy(standard)\n        partner_t=own_t",
        "np.random.PCG64(20261112+fold)": f"np.random.PCG64({sample_seed}+fold)",
        "torch.manual_seed(5217+fold)": f"torch.manual_seed({init_seed}+fold)",
    }
    for old, new in seams.items():
        need(body.count(old) == 1, "Exact frozen training seam")
        body = body.replace(old, new, 1)
    env = dict(learner.__dict__, HEADS=("full",))
    exec(compile(body, str(a.training), "exec"), env)
    return env["train"]


def paired_head(a, m, prior, slots, populations, actions, base_training, init_seed, sample_seed):
    train = seed_trainer(a, m.learner, prior.REFERENCES["training"][1], init_seed, sample_seed)
    common_count = [len(picks) for picks in actions["fixed"]]
    audits = {}
    for target in TARGETS:
        name = mode_for(init_seed, target)
        if name == "free_matched_count":
            audits[target] = base_training
            continue
        actions[name] = [[] for _ in slots]
        audits[target] = []
        for fold in range(5):
            models, center, scale, audit = train(populations[target], fold)
            audits[target].append(dict(fold=fold, **audit))
            for i, slot in enumerate(slots):
                if slot["fold"] != fold or not common_count[i]:
                    continue
                f = slot["data"]["features"]
                p = m.learner.probabilities(models["full"], f, f, center, scale)
                cap = min(8, slot["area"] // 10, len(p))
                actions[name][i] = m.free.choose_free(p, cap, fixed_count=common_count[i])
            print(json.dumps(dict(event="FIT109_TARGET_HEAD_FOLD", seed=init_seed, fold=fold, target=target)), flush=True)
        need(all(len(picks) == k for picks, k in zip(actions[name], common_count, strict=True)),
             "Per-candidate exact replay budget on original endpoint sets")
    for fold in range(5):
        a = [audits[target][fold] for target in TARGETS]
        fields = ("steps", "parameters", "observed_pair_examples", "observed_point_examples",
                  "initial_tensor_sha256", "batch_sha256")
        structure = [tuple(d["heads"]["full"][key] for key in fields) for d in a]
        need(len(set(structure)) == 1 and structure[0][:4] == (200, 1321, 12800, 25600)
             and len({(d["groups"], d["unique_train_pairs"]) for d in a}) == 1,
             "Same exact paired initialization, input groups, batches and update count")
    return audits


def group_differences(slots, actions, delta):
    values = defaultdict(list)
    names = [(mode_for(seed, "fractional"), mode_for(seed, control))
             for seed, _ in SEEDS for control in TARGETS[1:]]
    for i, slot in enumerate(slots):
        if slot["raw"] is None:
            continue
        target_area, raw = slot["raw"]
        candidate_area = slot["area"]

        def iou(name):
            picks = actions[name][i]
            intersection = raw + (float(delta(slot["truth"], picks).sum()) if picks else 0)
            bound = min(target_area, candidate_area)
            need(-1e-8 <= intersection <= bound + 1e-8, "Exact continuous intersection bounds")
            intersection = float(np.clip(intersection, 0, bound))
            return 100 * intersection / (candidate_area + target_area - intersection)

        values[slot["group"]].append([iou(x) - iou(y) for x, y in names])
    grouped = np.array([np.mean(values[key], axis=0) for key in sorted(values)])
    need(grouped.shape == (93, 6), "Original 93 groups and six contrasts")
    return grouped.reshape(93, 3, 2)


def paired_contrasts(slots, actions, coco, delta):
    grouped = group_differences(slots, actions, delta)
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(grouped), size=(4096, len(grouped)))

    def means_and_interval(a):
        return dict(delta_points=float(a.mean()),
                    interval95_points=np.percentile(a[draws].mean(axis=1), [2.5, 97.5]).tolist())

    contrasts = {}
    for column, target in enumerate(TARGETS[1:]):
        per_seed = []
        for k, (seed, sampling) in enumerate(SEEDS):
            first, other = mode_for(seed, "fractional"), mode_for(seed, target)
            per_seed.append(dict(seed=seed, sampler=sampling, AP_points=coco[first]["AP"]-coco[other]["AP"],
                                 AP75_points=coco[first]["AP75"]-coco[other]["AP75"],
                                 continuous_iou=means_and_interval(grouped[:, k, column])))
        contrasts[target] = dict(seeds=per_seed, mean_AP_points=float(np.mean([r["AP_points"] for r in per_seed])),
                                 seed_averaged_continuous_iou=means_and_interval(grouped[:, :, column].mean(axis=1)))
    return contrasts


def run(a, m, prior, classical, modes):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    historic = json.loads(prior.pinned(a.historical, prior.INPUTS["historical"][1]))
    previous = json.loads(prior.pinned(a.previous, prior.INPUTS["previous"][1]))
    _, blocks = m.base.read_core(a)
    receipt, planned = m.pair.sealed_points(a, m.base, m.math, blocks)
    need(receipt["points_manifest_sha256"] == prior.INPUTS["points_manifest"][1], "Exact frozen point manifest")
    a.output.mkdir()
    prelabel = m.base.seal(a.output / "PRELABEL_SAFE.json", m.base.encoded(dict(
        status="FIT109_FRACTIONAL_TARGETS_PRELABEL", gt_opened=False, protocol_sha256=a.protocol_sha,
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), seeds=SEEDS,
        inputs={key: value[1] for key, value in prior.INPUTS.items()})))
    gt, annotations = m.pair.read_labels(a, m.base, blocks)
    slots, scope = m.pair.fixed_slots(a, m.base, blocks, planned, gt, annotations)
    heads, counts = target_populations(slots, gt)
    actions, first, usage = m.free.fit(slots, m.learner, m.math, previous)
    need(scope == historic["match_scope"] and first == historic["training"] and usage == historic["usage"],
         "Exact first fractional initialization, fitting, and readout records")
    budget = [len(picks) for picks in actions["fixed"]]
    need(sum(budget) == 20809 and len(budget) == 10900 and
         all(len(picks) == k for picks, k in zip(actions["free_matched_count"], budget, strict=True)),
         "Exact original matched-count intervention")
    training = {str(seed): paired_head(a, m, prior, slots, heads, actions, first, seed, sampler)
                for seed, sampler in SEEDS}
    need(set(actions) == set(modes), "All nine heads and original controls represented")
    with classical.verify_materialization(m.base) as verified:
        coco, budgets = m.gray.materialize(slots, actions, m.base, m.official, gt)
    geometry = m.pair.fixed_geometry(slots, actions, m.math)
    del geometry["free_natural_minus"]
    transitions, _ = m.common.threshold(slots, actions)
    m.common.validate_transitions(transitions, geometry)
    for arm in HISTORICAL:
        need((coco[arm], budgets[arm], geometry["metrics"][arm], transitions[arm]) ==
             (historic["cohorts"][arm], historic["budgets"][arm], historic["geometry"]["metrics"][arm],
              historic["strict90_transitions"][arm]), "Exact historical metric and geometry replay: " + arm)
    need(all(budgets[mode_for(seed, target)] == budgets["free_matched_count"]
             for seed, _ in SEEDS for target in TARGETS), "Exact all-arm edit counts and roles")
    contrasts = paired_contrasts(slots, actions, coco, m.free.endpoint_deltas)
    data = dict(status="FIT109_FRACTIONAL_TARGETS_MATCHED_COUNT_ACTUAL", FIT=True, DEV=False, TEST=False,
                historical_parity=True, prelabel_sha256=prelabel, detector_updates=0, detector_forwards=0,
                point_optimizer_updates=9000, target_counts=counts, training=training, verified_xor=verified,
                cohort_scores=coco, budgets=budgets, geometry=geometry, strict90_inclusive=True,
                strict90_transitions=transitions, fractional_minus_controls=contrasts)
    digest = m.base.seal(a.output / "RESULT_SAFE.json", m.base.encoded(data))
    print(json.dumps(dict(status=data["status"], result_sha256=digest)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run"))
    for key in ("pilot", "classical", "helper", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("helper-sha256", "protocol-sha"):
        parser.add_argument("--" + key, required=True)
    a = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not a.output.exists(), "Fresh CPU experiment")
    read_exact(a.protocol, a.protocol_sha)
    classical = module(a.classical, CLASSICAL)
    prior = classical.import_checked(a.pilot, classical.PILOT_SHA)
    need(all(version(package) == expected for package, expected in prior.VERSIONS.items()), "Original package versions")
    a.private_root, a.code = prior.PRIVATE, prior.CODE
    m = prior.sources(a)
    modes = (*HISTORICAL, *(mode_for(seed, target) for seed, _ in SEEDS for target in TARGETS
                            if mode_for(seed, target) != "free_matched_count"))
    m.common.MODES = m.gray.MODES = m.pair.MODES = modes
    m.pair.GEOMETRY = (*modes, "oracle")
    if a.mode == "run":
        run(a, m, prior, classical, modes)
    else:
        for seed, sampling in SEEDS:
            seed_trainer(a, m.learner, prior.REFERENCES["training"][1], seed, sampling)
        print(json.dumps(dict(status="FIT109_FRACTIONAL_TARGETS_IMPORTS_ACCEPTED", closed_inputs_opened=0)))


if __name__ == "__main__":
    main()
