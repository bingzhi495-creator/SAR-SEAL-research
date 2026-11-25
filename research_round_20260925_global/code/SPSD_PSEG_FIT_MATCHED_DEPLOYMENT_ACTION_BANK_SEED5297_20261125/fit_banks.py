"""Compare only the source of the same-count signed FIT action proposals."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import torch


PROTOCOL = "282fc190309aeccdbf32eab879fa5e202a41dafb9f4b522d99afc2df33426f19"
CURRENT_CONTROL = "c0f7d0f84e242d5e6549cb38d91e28cdf44fee0f379605d9b7fdcfe456425d93"
ARCHIVED_CONTROL = "5d50b31dc9f7733bd2aa4a5c8b078a97d4def679afb57e81a32426fbf9293822"
CONTROL_REPEATS = (
    ("parity_probe", "e8b62b3c6b207f2fc111f1820b980754813176e8c8f93d51abacafff39eb7481",
     "67ed73f3a236780b2dda91482512686c640f14313d52698afbF45f75d1c46dda".lower()),
    ("parity_probe_repeat", "3aa7d210e3cb7f1f24ff002808a89301ac50b045abd43987aca5872e08a293b0",
     "a14603fdddfdf687f3f20e82c0deba3fdd59fb504512b6c5da9ce82edb3f0137"),
)
ORIGINAL = "011914954482cd505343a6421296ae754b1ea3098f5ad3fe37cdaa14e3c05fd3"
ORIGINAL_TERMINAL = "b0d837ca9fdeb4b21bb6c4bb83ecda130b86696d751b4cad309a41379887991b"
ORIGINAL_RESULT = "ae1b165a111a6b88939eb37b5741421259f0f39531c859a7c7663ad9116d1b05"
SELECTOR = "7c7a5415dad9c70804a0cf4dd07d0be46f8cbc5b3c0d1666e3cd067d25846205"
STATUS = "SPSD_PSEG_MATCHED_DEPLOYMENT_ACTION_BANK_FIT_COMPLETE"
INPUT_STATUS = "FIT_MATCHED_ACTION_BANK_PACKETS_ADMITTED"
SEED = 5297
FOLDS = 5


def need(check, reason):
    if not check:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Changed input " + path.name)
    return raw


def source(path, expected):
    loaded = ModuleType(path.stem)
    loaded.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), loaded.__dict__)
    return loaded


def fixed_folds(groups):
    ordered = sorted(set(groups), key=lambda g: (digest(("OOF-5297-v1|" + g).encode("ascii")), g))
    need(len(ordered) >= FOLDS, "Need five nonempty group folds")
    group_fold = {group: index % FOLDS for index, group in enumerate(ordered)}
    return np.asarray([group_fold[group] for group in groups], dtype=np.int8)


def signed_weights(plan, static, endpoint_count):
    result = np.zeros((endpoint_count, 2), dtype=np.int8)
    for side, sign in enumerate((1, -1)):
        selected = np.asarray(plan[side], dtype=np.intp)
        incumbent = np.asarray(static[side], dtype=np.intp)
        need(len(selected) == len(set(selected)) == len(incumbent)
             and np.all((selected >= 0) & (selected < endpoint_count)),
             "Distinct same-count write on the supplied endpoints")
        result[selected, side] += sign
        result[incumbent, side] -= sign
    need(np.all(result.sum(axis=0) == 0), "Signed write preserves each count")
    return result


def deployment_bank(estimates, static, previous, selector):
    n = len(estimates)
    k = len(static[0])
    need(estimates.shape == (n, 2, 2) and previous.shape[1:] == (n, 2)
         and 0 < k < n and len(static[1]) == k, "Exact selector support and incumbent")
    posterior = np.asarray(estimates, dtype=np.float64)
    add, drop = posterior[:, 0, :], posterior[:, 1, :]
    verified = selector.select(add, drop, *static, k=k)
    add_sets = selector._candidate_sets(add, static[0], k, True)
    drop_sets = selector._candidate_sets(drop, static[1], k, False)
    add_sum = add[np.asarray(add_sets, dtype=np.intp)].sum(axis=1)
    drop_sum = drop[np.asarray(drop_sets, dtype=np.intp)].sum(axis=1)
    relative = add_sum[:, None, :] - drop_sum[None, :, :] - (add_sum[0] - drop_sum[0])
    values = relative.min(axis=-1)
    order = np.argsort(-values.ravel(), kind="stable")
    first = int(order[0])
    first_add, first_drop = divmod(first, len(drop_sets))
    need(verified["static"] == static and verified["robust"] == (add_sets[first_add], drop_sets[first_drop]),
         "Exact deployed selector winner and tie order")
    chosen = []
    used = set()
    for index in order:
        i, j = divmod(int(index), len(drop_sets))
        if i == j == 0:
            continue
        proposal = signed_weights((add_sets[i], drop_sets[j]), static, n)
        raw = proposal.tobytes()
        need(raw not in used, "Unique nonstatic writes in finite selector menu")
        chosen.append(proposal)
        used.add(raw)
        if len(chosen) == len(previous):
            break
    from_menu = len(chosen)
    old = {row.tobytes() for row in previous}
    need(len(old) == len(previous), "Distinct previous action tensors")
    for action in previous:
        if len(chosen) == len(previous):
            break
        raw = action.tobytes()
        if raw not in used:
            chosen.append(action)
            used.add(raw)
    need(len(chosen) == len(previous) == len(used), "Exact original per-candidate action count")
    return np.stack(chosen), dict(menu_available=values.size - 1, menu_used=from_menu,
                                  shared_actions=len(old & used),
                                  robust_nonstatic=verified["robust"] != static)


def original_packets(args, old, action_core, attraction, original_result):
    previous = source(args.prior_fit, old.PRIOR_FIT)
    action_bank = old.action_bank
    static_rows = []

    def record(coords, feature, stock, quota, original, engine, ident):
        weights = action_bank(coords, feature, stock, quota, original, engine, ident)
        canonical = old.canonical(original)
        static_rows.append((ident, canonical))
        return weights

    old.action_bank = record
    packets, counts, packet_sha = old.fit_packets(args, previous, action_core, attraction)
    need(counts == original_result["counts"] and packet_sha == original_result["fit_packet_sha256"]
         and len(static_rows) == len({iden for iden, _ in static_rows}) == len(packets) == 848,
         "Exact closed original FIT packet content, association and order")
    return packets, counts, packet_sha, static_rows


def crossfit_occupancy(packets, action_core, occupancy_core, source_core):
    groups = [packet.group_id for packet in packets]
    fold_ids = fixed_folds(groups)
    fold_sha = digest(fold_ids.tobytes())
    predictions = [None] * len(packets)
    audits = []
    action_core.MODES = ("bce",)
    for fold in range(FOLDS):
        train = np.flatnonzero(fold_ids != fold)
        held = np.flatnonzero(fold_ids == fold)
        train_group = {groups[i] for i in train}
        held_group = {groups[i] for i in held}
        train_images = {packets[i].image_id for i in train}
        held_images = {packets[i].image_id for i in held}
        need(train_group and held_group and not train_group & held_group and not train_images & held_images,
             "No exact decoded or image overlap between cross-fit folds")
        model, center, scale, audited = action_core.train_matched_action(
            [packets[i] for i in train], source_core=source_core, seed=SEED)
        stats = audited["heads"]["bce"]
        need(stats["parameters"] == 1346 and stats["optimizer_updates"] == 1000
             and stats["observed_candidates"] == 64000, "Exact complement BCE-only update schedule")
        for i in held:
            need(predictions[i] is None, "Each eligible candidate predicted once")
            predictions[i] = occupancy_core.predict(model["bce"], packets[i].features, center, scale)
        audits.append(dict(fold=fold, train_groups=len(train_group), held_groups=len(held_group),
                           train_images=len(train_images), held_images=len(held_images),
                           train_candidates=len(train), held_candidates=len(held), overlap_groups=0,
                           overlap_images=0, train_head=stats))
    predicted = sha256()
    for row, packet in zip(predictions, packets, strict=True):
        need(row is not None and row.shape == packet.targets.shape and np.isfinite(row).all()
             and np.all((0 <= row) & (row <= 1)), "Complete ordered out-of-fold probabilities")
        predicted.update(row.astype("<f4", copy=False).tobytes())
    return predictions, audits, fold_sha, predicted.hexdigest()


def aligned_packets(packets, predictions, static_rows, selector):
    counts = Counter()
    by_candidate = Counter()
    aligned = []
    packet_digest = sha256()
    for packet, prediction, (key, static) in zip(packets, predictions, static_rows, strict=True):
        old = packet.action_weights
        new, audit = deployment_bank(prediction, static, old, selector)
        aligned.append(replace(packet, action_weights=new))
        packet_digest.update((key + "\n").encode("ascii"))
        packet_digest.update(new.tobytes())
        cardinality = len(old)
        counts.update(audit)
        counts.update(candidates=1, actions=cardinality, fallback=cardinality-audit["menu_used"])
        by_candidate["any_fallback"] += int(audit["menu_used"] != cardinality)
        by_candidate["any_new_action"] += int(audit["shared_actions"] != cardinality)
        by_candidate["no_nonstatic_menu"] += int(audit["menu_available"] == 0)
        by_candidate["same_bank_set"] += int(audit["shared_actions"] == cardinality)
    need(counts["actions"] == 5042 and counts["candidates"] == 848
         and counts["menu_used"] + counts["fallback"] == 5042, "Exact matched action bank sizes")
    return aligned, dict(actions=dict(counts), candidates=dict(by_candidate)), packet_digest.hexdigest()


def final_fits(packets, aligned, action_core, source_core, original):
    action_core.MODES = ("bce_signed_action",)
    models = {}
    heads = {}
    reference_audit = original["train_audit"]
    fit_stats = None
    moments = None
    compared_keys = ("parameters", "optimizer_updates", "observed_candidates", "observed_endpoints", "observed_actions",
                     "initial_tensor_sha256", "batch_sha256")
    for name, data in (("previous_bank", packets), ("deployment_bank", aligned)):
        fitted, center, scale, audit = action_core.train_matched_action(data, source_core=source_core, seed=SEED)
        entry = audit.pop("heads")["bce_signed_action"]
        need(all(entry[k] == reference_audit["heads"]["bce_signed_action"][k] for k in compared_keys)
             and all(audit[k] == reference_audit[k] for k in audit), "Identical final budget and fit roster")
        if name == "previous_bank":
            need(entry["final_tensor_sha256"] == CURRENT_CONTROL,
                 "Previous bank final tensors reproduce both current ordinary-worker control fits")
            fit_stats = audit
            moments = center, scale
        else:
            need(audit == fit_stats and all(np.array_equal(current, earlier)
                 for current, earlier in zip((center, scale), moments, strict=True)),
                 "Identical draw schedule and scaling in the final paired arms")
        models[name] = fitted["bce_signed_action"].state_dict()
        heads[name] = entry
    need(heads["previous_bank"]["final_tensor_sha256"] != heads["deployment_bank"]["final_tensor_sha256"],
         "A realized nonidentity action-bank intervention")
    return models, moments, dict(fit_stats, heads=heads)


def current_control(root, archived):
    check_fields = ("initial_tensor_sha256", "batch_sha256", "parameters", "optimizer_updates",
                    "observed_actions", "observed_endpoints", "observed_candidates")
    archived_head = archived["train_audit"]["heads"]["bce_signed_action"]
    need(archived_head["final_tensor_sha256"] == ARCHIVED_CONTROL != CURRENT_CONTROL,
         "Keep the separate archived sensitivity anchor")
    for name, result_sha, terminal_sha in CONTROL_REPEATS:
        stage = root / name
        end = json.loads(pin(stage / "observed/TERMINAL.json", terminal_sha))
        doc = json.loads(pin(stage / "actual/RESULT_SAFE.json", result_sha))
        head = doc["train_audit"]["heads"]["bce_signed_action"]
        need(end["phase"] == "prepare-edits" and end["actual_exit"] == 0
             and end["worker_sha256"] == ORIGINAL and end["output_footer_verified"] is True
             and end["result_sha256"] == result_sha and doc["worker_sha256"] == ORIGINAL
             and doc["source_gate_labels_opened"] is doc["historical_test_opened"] is False
             and doc["fit_packet_sha256"] == archived["fit_packet_sha256"]
             and head["final_tensor_sha256"] == CURRENT_CONTROL
             and all(head[key] == archived_head[key] for key in check_fields),
             "Two independent current-runtime reference fits with exact closed parity")
    return [result for _, result, _ in CONTROL_REPEATS]


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("protocol", "capture", "points", "labels", "prior-fit", "previous", "action-core",
                 "source-core", "attraction", "selector", "out", "probe-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and all(os.environ.get(key) == "1" for key in
         ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")) and torch.get_num_threads() == 1,
         "Exact repeatable single-thread CPU fit runtime")
    need(not args.out.exists(), "Use a fresh output location")
    pin(args.protocol, PROTOCOL)
    original = source(args.previous / "fit_action.py", ORIGINAL)
    old_exit = json.loads(pin(args.previous / "observed_fit/TERMINAL.json", ORIGINAL_TERMINAL))
    old_result = json.loads(pin(args.previous / "actual/RESULT_SAFE.json", ORIGINAL_RESULT))
    need(old_exit["actual_exit"] == 0 and old_exit["output_footer_verified"]
         and old_exit["worker_sha256"] == ORIGINAL and old_exit["result_sha256"] == ORIGINAL_RESULT
         and old_result["status"] == original.STATUS and old_result["historical_test_opened"] is False
         and old_result["source_gate_labels_opened"] is False, "Actual prior fit closed and verified")
    current_controls = current_control(args.probe_root, old_result)
    action = source(args.action_core, original.ACTION_CORE)
    occupancy = source(args.source_core, original.OCCUPANCY_CORE)
    attraction = source(args.attraction, original.ATTRACTION)
    selector = source(args.selector, SELECTOR)
    packets, counts, packet_sha, static = original_packets(args, original, action, attraction, old_result)
    args.out.mkdir()
    input_path = args.out / "INPUT_SAFE.json"
    admitted = original.write(input_path, dict(status=INPUT_STATUS, protocol_sha256=PROTOCOL, counts=counts,
              packet_tensor_sha256=packet_sha, previous_fit_result_sha256=ORIGINAL_RESULT, fit_only=True,
              source_gate_labels_opened=False, source_gate_points_opened=False, historical_test_opened=False))
    print(json.dumps(dict(status=INPUT_STATUS, input_receipt_sha256=admitted)), flush=True)
    predictions, folds, fold_sha, pred_sha = crossfit_occupancy(packets, action, occupancy, args.source_core)
    aligned, bank_audit, aligned_sha = aligned_packets(packets, predictions, static, selector)
    models, (center, scale), audit = final_fits(packets, aligned, action, args.source_core, old_result)
    path = args.out / "FITTED_WEIGHTS_PRIVATE.pt"
    with path.open("xb") as stream:
        torch.save(dict(models=models, center=torch.from_numpy(center), scale=torch.from_numpy(scale), seed=SEED), stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(status=STATUS, protocol_sha256=PROTOCOL, worker_sha256=digest(Path(__file__).read_bytes()),
                  action_core_sha256=original.ACTION_CORE, training_core_sha256=original.OCCUPANCY_CORE,
                  selector_sha256=SELECTOR, previous_fit_result_sha256=ORIGINAL_RESULT, input_receipt_sha256=admitted,
                  fit_packet_sha256=packet_sha, counts=counts, out_of_fold_group_sha256=fold_sha,
                  out_of_fold_probabilities_sha256=pred_sha, out_of_fold_audit=folds, bank_audit=bank_audit,
                  aligned_action_packet_sha256=aligned_sha, paired_previous_fit_tensor_parity=True,
                  archived_previous_fit_tensor_parity=False, archived_previous_fit_tensor_sha256=ARCHIVED_CONTROL,
                  current_runtime_control_tensor_sha256=CURRENT_CONTROL, current_runtime_control_result_sha256s=current_controls,
                  runtime=dict(torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                               numpy_version=np.__version__, omp=1, mkl=1, openblas=1),
                  source_gate_labels_opened=False, source_gate_points_opened=False, historical_test_opened=False,
                  detector_forwards=0, total_optimizer_updates=2000, oof_optimizer_updates=5000,
                  output_checkpoint_sha256=digest(path.read_bytes()), train_audit=audit,
                  elapsed_seconds=perf_counter() - started)
    checksum = original.write(args.out / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=STATUS, result_sha256=checksum)), flush=True)


if __name__ == "__main__":
    main()
