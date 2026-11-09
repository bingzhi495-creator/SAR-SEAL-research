"""Compare independently closed E24 common-support and native FIT streams.

The caller binds already-closed streams to their original files and verifies
worker closure, training markers, and archives. Only aggregate counts and
computed byte fingerprints leave this verifier.
"""

from hashlib import sha256
import json

from common_fit_pair_receipt import COUNTS, DIGESTS, SCOPE, compact, exact, is_hex, require
from common_fit_train import AUDIT_STATUS


EPOCHS = 24
VISITS_PER_EPOCH = 2940
UPDATES_PER_EPOCH = 735
MAX_TARGETS = 164
COMMON_LINE_LIMIT = 4096
COMMON_FILE_LIMIT = 128 * 1024**2
NATIVE_LINE_LIMIT = 1024**2
NATIVE_FILE_LIMIT = 8 * 1024**2


class Readback:
    def __init__(self, stream, label, line_limit, file_limit):
        self.stream = stream
        self.label = label
        self.line_limit = line_limit
        self.file_limit = file_limit
        self.bytes = 0
        self.digest = sha256()

    def line(self):
        raw = self.stream.readline(self.line_limit + 1)
        require(type(raw) is bytes and len(raw) <= self.line_limit and raw.endswith(b"\n"),
                "Missing, oversized, or unterminated " + self.label + " line")
        self.bytes += len(raw)
        require(self.bytes <= self.file_limit, "Exceeded closed " + self.label + " byte limit")
        self.digest.update(raw)
        return raw

    def finish(self):
        require(self.stream.readline(1) == b"", "Unexpected trailing bytes in " + self.label)
        return self.digest.hexdigest()


def decode(raw, label):
    try:
        value = json.loads(raw)
        canonical = compact(value) + b"\n"
    except (ValueError, RecursionError) as error:
        raise ValueError("Invalid closed " + label + " JSONL") from error
    require(type(value) is dict and raw == canonical, "Expected canonical " + label + " JSONL")
    return value


def native_epoch(raw, epoch):
    record = decode(raw, "native FIT visit")
    exact(record, dict(epoch=epoch), "Exact original native FIT epoch cursor required", extra=("visits",))
    visits = record["visits"]
    require(type(visits) is list and len(visits) == VISITS_PER_EPOCH
            and all(type(pair) is list and len(pair) == 2
                    and type(pair[0]) is int and pair[0] >= 0
                    and type(pair[1]) is int and 0 <= pair[1] <= MAX_TARGETS for pair in visits),
            "Exact original native FIT image and retained target counts required")
    return visits


def common_visit(raw, epoch, batch, ordinal):
    row = decode(raw, "common FIT consumed")
    exact(row, dict(epoch=epoch, batch=batch, ordinal=ordinal), "Exact E24 common FIT visit cursor required",
          extra=("image_id", "crop", "flipped", *COUNTS, *DIGESTS))
    require(all(type(row[key]) is int and row[key] >= 0 for key in ("image_id", *COUNTS))
            and all(type(row[key]) is bool for key in ("crop", "flipped"))
            and all(is_hex(row[key]) for key in DIGESTS),
            "Exact E24 common FIT visit identity, trajectory, RNG and mask fingerprints required")
    common = row["common_nonempty"]
    for mode in ("native", "center"):
        require(common <= row[f"{mode}_nonempty"] <= row[f"{mode}_present"] <= row["raw_rows"]
                and row[f"{mode}_only_nonempty"] == row[f"{mode}_nonempty"] - common,
                "Invalid common support in E24 consumed FIT row")
    require(row["native_nonempty"] + row["center_nonempty"] - common <= row["raw_rows"]
            and row["common_changed_masks"] <= common
            and (row["common_changed_masks"] > 0) ==
            (row["native_common_masks_sha256"] != row["center_common_masks_sha256"]),
            "Invalid common support or mask change in E24 consumed FIT row")
    return row


def verify_safe_scope(safe, mode, visits, updates):
    exact(safe, dict(status=AUDIT_STATUS, completed_epochs=EPOCHS,
                     optimizer_updates=updates, ema_updates=updates, **SCOPE),
          "Exact original E24 common FIT safe receipt scope required",
          extra=("common_support", "phase", "native_result_sha256", "repeat_seed_witness_sha256",
                 "consumed_ledger_sha256", "consumed_ledger_bytes", "ledger"))
    for key, mode_field in (("common_support", "mode"), ("phase", "mask_resize_mode")):
        marker = safe[key]
        require(type(marker) is dict and marker.get("stage") == "e24" and marker.get(mode_field) == mode,
                "Exact E24 common FIT arm and stage required")
    require(type(safe["consumed_ledger_bytes"]) is int and 0 < safe["consumed_ledger_bytes"] <= COMMON_FILE_LIMIT
            and all(is_hex(safe[key]) for key in ("native_result_sha256", "repeat_seed_witness_sha256",
                                                 "consumed_ledger_sha256")),
            "Exact closed E24 common FIT byte references required")
    exact(safe["ledger"], dict(fetched_visits=visits, consumed_visits=visits, prefetched_unconsumed=0,
                              unconsumed_prefetch_sha256=None),
          "E24 requires all visits consumed and zero unconsumed lookahead",
          extra=("common_targets_total", "common_empty_visits", "changed_mask_visits"))


def verify_e24_ledgers(*, native_common, center_common, native_visits, center_visits,
                       native_common_safe, center_common_safe):
    """Validate four already-closed binary streams from their beginning."""
    visits = EPOCHS * VISITS_PER_EPOCH
    updates = EPOCHS * UPDATES_PER_EPOCH
    receipts = (native_common_safe, center_common_safe)
    for mode, safe in zip(("native", "center"), receipts, strict=True):
        verify_safe_scope(safe, mode, visits, updates)
    common_a = Readback(native_common, "native common FIT", COMMON_LINE_LIMIT, COMMON_FILE_LIMIT)
    common_b = Readback(center_common, "center common FIT", COMMON_LINE_LIMIT, COMMON_FILE_LIMIT)
    native_a = Readback(native_visits, "native original FIT", NATIVE_LINE_LIMIT, NATIVE_FILE_LIMIT)
    native_b = Readback(center_visits, "center original FIT", NATIVE_LINE_LIMIT, NATIVE_FILE_LIMIT)

    ordinal = targets = empty = changed = changed_masks = native_only = center_only = disagreement = 0
    for epoch in range(1, EPOCHS + 1):
        native_line = native_a.line()
        require(native_line == native_b.line(), "Original native FIT visit streams differ")
        for batch, (image_id, retained) in enumerate(native_epoch(native_line, epoch), 1):
            ordinal += 1
            line = common_a.line()
            require(line == common_b.line(), "Exact ordered E24 common FIT consumed streams must match")
            row = common_visit(line, epoch, batch, ordinal)
            require((row["image_id"], row["common_nonempty"]) == (image_id, retained),
                    "E24 consumed image order and retained targets must match both original native FIT ledgers")
            targets += retained
            empty += retained == 0
            changed += row["common_changed_masks"] > 0
            changed_masks += row["common_changed_masks"]
            native_only += row["native_only_nonempty"]
            center_only += row["center_only_nonempty"]
            disagreement += bool(row["native_only_nonempty"] or row["center_only_nonempty"])

    common_a_sha, common_b_sha = common_a.finish(), common_b.finish()
    native_a_sha, native_b_sha = native_a.finish(), native_b.finish()
    require(changed > 0, "At least one actual E24 common FIT visit must change masks")
    stats = dict(fetched_visits=visits, consumed_visits=visits, prefetched_unconsumed=0,
                 unconsumed_prefetch_sha256=None, common_targets_total=targets,
                 common_empty_visits=empty, changed_mask_visits=changed)
    for safe, readback, digest in zip(receipts, (common_a, common_b), (common_a_sha, common_b_sha), strict=True):
        require(safe["consumed_ledger_sha256"] == digest and safe["consumed_ledger_bytes"] == readback.bytes,
                "Closed E24 common FIT receipt digest or byte count does not match its stream")
        exact(safe["ledger"], stats, "Closed E24 common FIT summary disagrees with consumed stream")

    return dict(ordered_common_fit_visits_match=True, completed_epochs=EPOCHS, visits_per_arm=visits,
                optimizer_updates_per_arm=updates, ema_updates_per_arm=updates,
                shared_consumed_ledger_sha256=common_a_sha, shared_consumed_ledger_bytes=common_a.bytes,
                native_visit_ledger_sha256=native_a_sha, center_visit_ledger_sha256=native_b_sha,
                ledger=stats, native_only_nonempty_total=native_only, center_only_nonempty_total=center_only,
                common_changed_masks_total=changed_masks, support_disagreement_visits=disagreement)

