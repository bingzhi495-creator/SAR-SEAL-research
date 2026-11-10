"""Verify two independently closed, fixed-B one-pass FIT qualification arms."""

from hashlib import sha256

from common_fit_pair_receipt import (
    ClosedArm, SCOPE, UPDATES, VISITS, canonical, compact, consumed_visits, digest, encode_receipt, exact,
    is_hex, load_verifier, require, verify_summary,
)
from common_fit_train import (
    FIT_INDEX_SHA, INITIAL_EXPANDED_SHA, INITIALIZATION_SEED, NATIVE, ORIGINAL_PROTOCOLS,
    SOURCE_PINS as ORIGINAL_SOURCE_PINS, TRAINING_SEED,
)


SELF = "one_pass_common_qualification.py"
POLICY = "one_pass_on_stepwise_common_nonempty_rows_v1"
AUDIT_STATUS = "HRSID_ONE_PASS_COMMON_FIT_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1"
PAIR_STATUS = "HRSID_ONE_PASS_COMMON_FIT_SHORT_PAIR_ACCEPTED_V1"
MODES = ("stepwise_center", "one_pass")
SELECTED_MASK = dict(stepwise_center="native_common_masks_sha256", one_pass="center_common_masks_sha256")
SOURCE_PINS = {
    **ORIGINAL_SOURCE_PINS,
    SELF: "4b2bb3bbe00aaf28c748f66a540182993589039304ea1d966d1dde7e7c1980d8",
    "common_fit_train.py": "b52f8dc7a8d5bd06c0bda3eb839f31cbed3a036befa709c8fc718c47706a959b",
    "fit_query_export.py": "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5",
    "reference_resize.py": "3ca1d4a6400324c13c777cb4d5f7627713bc625580195d8e05363950872cc8df",
    "replay_fit.py": "4f95f1b5b84fcf89b005f999cfb8c5fd647a7b262aedd2622860cf8785a14157",
    "one_pass_fit_transform.py": "05cad566f9b5cc761e91b5283585a5392a7450c42c3133349e6dc21e8bf7496d",
    "one_pass_common_pair_transform.py": "5db1d9024a7ffa03608488b07669e5692d378f166f8773b1a5475089519d08b7",
    "common_fit_binding.py": "2bfc4d6039b28a632693131126120cc2502c144b7cf6564de5dc4d765a9e4fba",
    "one_pass_common_binding.py": "987f95263c5dd685403afc8c7c96150ee2f3e1f911ce7ba6b2f43734632f028f",
}
OLD_PROTOCOLS = (*ORIGINAL_PROTOCOLS, "04308c52bd2b4730f1474335c6316f6b8a11762e",
                 "3ee0dc2f5f6508c238b760b9b234edc016e89f54", "91728d38a359e4a207e3f741490f09060c46663f")
SELECTED_STATS = {
    "verified_selected_mask_visits", "verified_selected_mask_stream_sha256", "consumed_direct_visits",
    "consumed_crop_visits", "changed_crop_visits",
}


def check_shared_rows(rows):
    for row in rows:
        require(row["native_present"] == row["center_present"], "Both fixed-support arms must retain the same B crop roster")
        if not row["crop"]:
            require(row["common_changed_masks"] == 0
                    and row["native_nonempty"] == row["center_nonempty"] == row["common_nonempty"],
                    "Actual direct visits must have identical common masks and support")
    direct = sum(not row["crop"] for row in rows)
    changed_crop = sum(row["crop"] and row["common_changed_masks"] > 0 for row in rows)
    require(any(not row["crop"] and row["common_nonempty"] > 0 for row in rows) and changed_crop > 0,
            "The closed short pair must exhibit nonempty direct equality and actual common crop mask changes")
    return dict(consumed_direct_visits=direct, consumed_crop_visits=VISITS - direct, changed_crop_visits=changed_crop)


def check_actual_selected(rows, native_rows, mode):
    selected = sha256()
    for row, native in zip(rows, native_rows, strict=True):
        mask = native["postfilter_masks_sha256"]
        require((native["image_id"], native["labels"], native["postfilter_labels"]) ==
                (row["image_id"], row["common_nonempty"], row["common_nonempty"]),
                "Actual native batch image and supervision count must match the consumed row")
        require(mask == row[SELECTED_MASK[mode]], "Actual selected native postfilter mask digest differs from the admitted arm")
        item = {key: row[key] for key in ("ordinal", "image_id", "epoch", "batch")}
        selected.update(compact(dict(item, mask_sha256=mask)) + b"\n")
    return selected.hexdigest()


def verify_arm(arm, mode, identity, verifier):
    require(type(arm) is ClosedArm and all(type(value) is bytes for value in arm[1:]),
            "Exact independently acquired original closed arm bytes required")
    exact(arm.closure, dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True),
          "Original independently closed short worker required", extra=("stdout_sha256", "terminal_sha256"))
    require(is_hex(arm.closure["terminal_sha256"]) and type(arm.stdout) is bytes and 0 < len(arm.stdout) <= 2 * 1024**2
            and arm.closure["stdout_sha256"] == digest(arm.stdout), "Exact original stdout and terminal digests required")
    common = canonical(arm.common_receipt, verifier, dict, 64 * 1024)
    exact(common, dict(status=AUDIT_STATUS, optimizer_updates=UPDATES, ema_updates=UPDATES, completed_epochs=0,
                       consumed_ledger_sha256=digest(arm.consumed_ledger), consumed_ledger_bytes=len(arm.consumed_ledger),
                       **SCOPE), "Exact new short-only common FIT arm report required",
          extra=("common_support", "phase", "native_result_sha256", "repeat_seed_witness_sha256", "ledger"))
    exact(common["common_support"], dict(**identity, policy=POLICY, stage="qualification", mode=mode,
                                         qualification_pair_sha256=None), "Exact fixed-B one-pass short policy and lineage required")
    phase = dict(protocol=identity["training_protocol"], mask_resize_mode=mode, stage="qualification", seed=TRAINING_SEED,
                 phase_trainer_sha256=SOURCE_PINS[SELF], phase_patch_sha256=SOURCE_PINS["phase_patch.py"],
                 native_trainer_sha256=SOURCE_PINS[NATIVE], initial_expanded_sha256=INITIAL_EXPANDED_SHA)
    exact(common["phase"], phase, "Exact new prospective short arm phase required")

    native = canonical(arm.native_report, verifier, dict, 64 * 1024)
    witness = canonical(arm.seed_witness, verifier, dict, 16 * 1024)
    canonical(arm.native_visits, verifier, list, 1024**2)
    require(common["native_result_sha256"] == digest(arm.native_report)
            and common["repeat_seed_witness_sha256"] == digest(arm.seed_witness),
            "Closed short report must bind original native report and effective seed witness")
    exact(witness, dict(status="HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1", protocol=phase["protocol"],
                        mask_resize_mode=mode, stage="qualification", initialization_seed=INITIALIZATION_SEED,
                        training_seed=TRAINING_SEED, global_rank=0, train_config_seed=TRAINING_SEED,
                        torch_cpu_initial_seed=TRAINING_SEED, torch_cuda_initial_seed=TRAINING_SEED,
                        lightning_global_seed=TRAINING_SEED, lightning_seed_workers=True),
          "Exact actual one-pass short runtime seed witness required")
    native_rows = verifier.validate_report(native, arm.native_visits, protocol=phase["protocol"], seed=TRAINING_SEED,
                                           mode=mode, trainer_sha=SOURCE_PINS[SELF])
    verifier.validate_stdout(arm.stdout, native, witness, digest(arm.seed_witness), native_rows)
    rows = consumed_visits(arm.consumed_ledger, verifier)
    branches = check_shared_rows(rows)
    selected = check_actual_selected(rows, native_rows, mode)
    summary = common["ledger"]
    require(type(summary) is dict, "Exact closed fixed-support ledger summary required")
    base = {key: value for key, value in summary.items() if key not in SELECTED_STATS}
    old_stats = verify_summary(base, rows)
    exact(summary, dict(base, verified_selected_mask_visits=VISITS, verified_selected_mask_stream_sha256=selected, **branches),
          "Closed selected-mask stream or direct/crop accounting differs from independent native batch readback")
    require(native["removed_empty_rows"] == 0 and native["targets_after_filter"] == old_stats["common_targets_total"],
            "Fixed common supervision must reach the model without losing rows")
    return rows, native, native_rows, selected, branches


def accept_short_pair(*, training_protocol, implementation_commit, source_sha256s, stepwise_center, one_pass,
                      native_qualifier_source=None):
    """Accept the short-only pair from original bytes already closed by the owner."""
    require(all(is_hex(value, 40) and value not in OLD_PROTOCOLS for value in (training_protocol, implementation_commit)),
            "Exact separate prospective one-pass training and implementation commit identities required")
    require(type(source_sha256s) is dict and source_sha256s == SOURCE_PINS,
            "Exact transitive one-pass qualification source pins required")
    identity = dict(training_protocol=training_protocol, implementation_commit=implementation_commit,
                    training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,
                    initial_expanded_sha256=INITIAL_EXPANDED_SHA, fit_index_sha256=FIT_INDEX_SHA,
                    source_manifest_sha256=digest(compact(source_sha256s)))
    verifier = load_verifier(native_qualifier_source)
    a, b = (verify_arm(arm, mode, identity, verifier) for arm, mode in zip((stepwise_center, one_pass), MODES, strict=True))
    rows, a_report, a_visits, a_selected, branches = a
    _, b_report, b_visits, b_selected, _ = b
    require(stepwise_center.closure["terminal_sha256"] != one_pass.closure["terminal_sha256"],
            "Distinct original OS-closed fixed-support arms required")
    require(stepwise_center.consumed_ledger == one_pass.consumed_ledger and a_selected != b_selected,
            "Exact ordered shared consumed ledger and distinct verified selected streams required")
    require(all(a_report[key] == b_report[key] for key in verifier.LR_FIELDS), "Same paired AdamW LR schedule required")
    fields = ("image_id", "image_sha256", "cpu_rng_sha256", "labels_sha256", "postfilter_boxes_sha256")
    require(all(all(left[key] == right[key] for key in fields)
                and (left["postfilter_masks_sha256"] != right["postfilter_masks_sha256"]) == (row["common_changed_masks"] > 0)
                for row, left, right in zip(rows, a_visits, b_visits, strict=True)),
            "Actual native batches must agree in image, CPU RNG, nonmask supervision and exact mask change pattern")
    return dict(status=PAIR_STATUS, stage="qualification", policy=POLICY, **identity,
                ordered_common_fit_visits_match=True, visits_per_arm=VISITS, optimizer_updates_per_arm=UPDATES,
                ema_updates_per_arm=UPDATES, verified_selected_mask_visits_per_arm=VISITS,
                changed_mask_visits=branches["changed_crop_visits"], **branches,
                stepwise_center_arm_receipt_sha256=digest(stepwise_center.common_receipt),
                one_pass_arm_receipt_sha256=digest(one_pass.common_receipt),
                stepwise_center_terminal_sha256=stepwise_center.closure["terminal_sha256"],
                one_pass_terminal_sha256=one_pass.closure["terminal_sha256"],
                stepwise_center_selected_mask_stream_sha256=a_selected, one_pass_selected_mask_stream_sha256=b_selected,
                shared_consumed_ledger_sha256=digest(stepwise_center.consumed_ledger), **SCOPE)
