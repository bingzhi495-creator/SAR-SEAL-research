"""Bind a prospective common-row augmentation audit to consumed FIT visits."""

from collections import deque
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
from unittest.mock import patch


ROW_COUNTS = (
    "raw_rows", "native_present", "center_present", "native_nonempty", "center_nonempty",
    "common_nonempty", "native_only_nonempty", "center_only_nonempty", "common_changed_masks",
)
DIGEST_FIELDS = (
    "image_sha256", "trace_sha256", "rng_before_sha256", "rng_after_sha256",
    "common_row_ids_sha256", "common_nonmask_sha256",
    "native_common_masks_sha256", "center_common_masks_sha256",
)
AUDIT_FIELDS = {"image_id", "crop", "flipped", *ROW_COUNTS, *DIGEST_FIELDS}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def encode(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


class ConsumedFitLedger:
    """Track ordered batch consumption separately from one loader lookahead."""

    def __init__(self, stream):
        self.stream = stream
        self.pending = deque()
        self.fetched_visits = 0
        self.consumed_visits = 0
        self.common_targets_total = 0
        self.common_empty_visits = 0
        self.changed_mask_visits = 0
        self.cursor = (1, 0)
        self.closed = False

    def fetched(self, record):
        require(not self.closed and len(self.pending) < 2,
                "Only a serial FIT visit and at most one lookahead are supported")
        require(
            type(record) is dict and set(record) == AUDIT_FIELDS
            and all(type(record[key]) is int and record[key] >= 0 for key in ("image_id", *ROW_COUNTS))
            and all(type(record[key]) is bool for key in ("crop", "flipped"))
            and all(type(record[key]) is str and re.fullmatch(r"[0-9a-f]{64}", record[key])
                    for key in DIGEST_FIELDS),
            "Exact common FIT visit audit required",
        )
        common = record["common_nonempty"]
        for mode in ("native", "center"):
            require(common <= record[f"{mode}_nonempty"] <= record[f"{mode}_present"] <= record["raw_rows"]
                    and record[f"{mode}_only_nonempty"] == record[f"{mode}_nonempty"] - common,
                    "Common FIT support must be contained in both natural supports")
        require(record["common_changed_masks"] <= common, "Changed masks must be common rows")
        self.pending.append(record.copy())
        self.fetched_visits += 1

    def require_next(self, image_id, common_count):
        require(not self.closed and self.pending
                and self.pending[0]["image_id"] == image_id
                and self.pending[0]["common_nonempty"] == common_count,
                "Actual FIT batch does not match the next prepared visit")

    def consume(self, *, image_id, common_count, epoch, batch):
        self.require_next(image_id, common_count)
        prior_epoch, prior_batch = self.cursor
        require(
            type(epoch) is type(batch) is int
            and ((epoch, batch) == (prior_epoch, prior_batch + 1)
                 or (prior_batch > 0 and (epoch, batch) == (prior_epoch + 1, 1))),
            "Actual FIT batches must be recorded in order",
        )
        row = self.pending[0]
        line = encode(dict(row, ordinal=self.consumed_visits + 1, epoch=epoch, batch=batch))
        require(self.stream.write(line) == len(line), "Incomplete common FIT visit audit write")
        self.pending.popleft()
        self.consumed_visits += 1
        self.cursor = epoch, batch
        self.common_targets_total += common_count
        self.common_empty_visits += int(common_count == 0)
        self.changed_mask_visits += int(row["common_changed_masks"] > 0)

    def finish(self, expected_visits):
        require(
            not self.closed and type(expected_visits) is int and expected_visits > 0
            and self.consumed_visits == expected_visits and len(self.pending) <= 1
            and self.fetched_visits == expected_visits + len(self.pending),
            "Exact consumed FIT visits with bounded unconsumed lookahead required",
        )
        self.stream.flush()
        self.closed = True
        prefetch = sha256(encode(self.pending[0]).encode("utf-8")).hexdigest() if self.pending else None
        return dict(
            fetched_visits=self.fetched_visits, consumed_visits=self.consumed_visits,
            prefetched_unconsumed=len(self.pending), unconsumed_prefetch_sha256=prefetch,
            common_targets_total=self.common_targets_total, common_empty_visits=self.common_empty_visits,
            changed_mask_visits=self.changed_mask_visits,
        )

@contextmanager
def common_fit_dataset(module_data, detection_type, fit_root, fit_count, native_transforms, mode, ledger):
    """Wrap the one stock FIT dataset and restore its original transform."""
    original_build = module_data.build_dataset
    dataset = None
    original_transform = None

    def build(image_set, *args, **kwargs):
        nonlocal dataset, original_transform
        require(image_set == "train" and dataset is None, "Only one original FIT dataset is allowed")
        candidate = original_build(image_set, *args, **kwargs)
        require(
            type(candidate) is detection_type and Path(candidate.root).resolve(strict=True) == fit_root
            and len(candidate) == len(candidate.ids) == len(set(candidate.ids)) == fit_count
            and candidate.include_masks is True and candidate.include_keypoints is False
            and callable(candidate._transforms),
            "Expected the exact stock FIT mask dataset",
        )
        from paired_fit_transform import PairedFitTransform

        original_transform = candidate._transforms
        adapter = PairedFitTransform(native_transforms, original_transform, mode, audit=ledger.fetched)
        candidate._transforms = adapter
        dataset = candidate
        return dataset

    try:
        with patch.object(module_data, "build_dataset", build):
            yield
    finally:
        if dataset is not None:
            dataset._transforms = original_transform


def bind_common_observer(observer, ledger):
    """Bind prepared observations to the native trainer's real batch callback."""
    original = observer.on_train_batch_start

    def on_batch_start(active, model, batch, batch_idx):
        _, targets = batch
        target, = targets
        image_id = int(target["image_id"].item())
        count = target["labels"].numel()
        ledger.require_next(image_id, count)
        original(active, model, batch, batch_idx)
        require(target["labels"].numel() == count, "The common FIT masks must already be nonempty")
        ledger.consume(image_id=image_id, common_count=count, epoch=active.current_epoch + 1,
                       batch=batch_idx + 1)

    observer.on_train_batch_start = on_batch_start
