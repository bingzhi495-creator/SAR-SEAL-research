"""Bind fixed-support one-pass labels to the selected actual FIT batches."""

from contextlib import contextmanager
from hashlib import sha256
from unittest.mock import patch

from torch import bool as torch_bool

import common_fit_binding as shared
from one_pass_common_pair_transform import OnePassCommonPairTransform
import paired_fit_transform


MASK_DIGEST_FIELDS = {
    "stepwise_center": "native_common_masks_sha256",
    "one_pass": "center_common_masks_sha256",
}


class ConsumedFitLedger(shared.ConsumedFitLedger):
    """Keep the paired stream common and bind a separate selected-mask digest."""

    def __init__(self, stream):
        super().__init__(stream)
        self._mask_digest_field = None
        self._selected_stream = sha256()
        self._verified_visits = 0
        self._crop_visits = 0
        self._changed_crop_visits = 0

    def bind_mode(self, mode):
        shared.require(type(mode) is str and mode in MASK_DIGEST_FIELDS and self._mask_digest_field is None
                       and self.fetched_visits == 0 and not self.closed,
                       "A fresh ledger must bind one explicit fixed-support arm")
        self._mask_digest_field = MASK_DIGEST_FIELDS[mode]

    def fetched(self, record):
        shared.require(self._mask_digest_field is not None, "Bind the selected arm before fetching FIT targets")
        if type(record) is dict and set(record) == shared.AUDIT_FIELDS:
            changed = record["common_changed_masks"] != 0
            unequal = record["native_common_masks_sha256"] != record["center_common_masks_sha256"]
            shared.require(record["native_present"] == record["center_present"] and changed == unequal,
                           "Fixed B crop rows and exact paired mask-change accounting required")
            shared.require(record["crop"] or (
                not changed and record["native_nonempty"] == record["center_nonempty"] == record["common_nonempty"]
            ), "Direct FIT visits must have identical masks and support")
        super().fetched(record)

    def require_selected(self, image_id, common_count, mask_sha256):
        self.require_next(image_id, common_count)
        shared.require(mask_sha256 == self.pending[0][self._mask_digest_field],
                       "Actual selected FIT mask digest does not match the queued arm")

    def consume_selected(self, *, image_id, common_count, epoch, batch, mask_sha256):
        self.require_selected(image_id, common_count, mask_sha256)
        row = self.pending[0]
        self.consume(image_id=image_id, common_count=common_count, epoch=epoch, batch=batch)
        self._selected_stream.update(shared.encode(dict(
            ordinal=self.consumed_visits, image_id=image_id, epoch=epoch, batch=batch, mask_sha256=mask_sha256,
        )).encode("utf-8"))
        self._verified_visits += 1
        self._crop_visits += int(row["crop"])
        self._changed_crop_visits += int(row["crop"] and row["common_changed_masks"] > 0)

    def finish(self, expected_visits):
        shared.require(self._verified_visits == self.consumed_visits
                       and self._changed_crop_visits == self.changed_mask_visits,
                       "Every consumed FIT visit needs a selected digest and every change needs a crop")
        stats = super().finish(expected_visits)
        return dict(stats, verified_selected_mask_visits=self._verified_visits,
                    verified_selected_mask_stream_sha256=self._selected_stream.hexdigest(),
                    consumed_direct_visits=self._verified_visits - self._crop_visits,
                    consumed_crop_visits=self._crop_visits, changed_crop_visits=self._changed_crop_visits)


@contextmanager
def common_fit_dataset(module_data, detection_type, fit_root, fit_count, native_transforms, mode, ledger):
    """Reuse the stock dataset admission with the fixed-B adapter for this process."""
    ledger.bind_mode(mode)
    # The shared builder imports the adapter when constructing the stock dataset.
    with (patch.object(paired_fit_transform, "PairedFitTransform", OnePassCommonPairTransform),
          shared.common_fit_dataset(module_data, detection_type, fit_root, fit_count, native_transforms, mode, ledger)):
        yield


def _selected_mask(batch):
    _, targets = batch
    target, = targets
    image_id = int(target["image_id"].item())
    count = target["labels"].numel()
    masks = target["masks"]
    shared.require(masks.dtype == torch_bool and masks.ndim == 3 and masks.shape[0] == count,
                   "Actual FIT batch must contain one Boolean mask per selected row")
    return image_id, count, paired_fit_transform._training_mask_digest(masks.detach().cpu())


def bind_common_observer(observer, ledger):
    """Verify selected mask bytes before and after the native empty-row guard."""
    original = observer.on_train_batch_start

    def on_batch_start(active, model, batch, batch_idx):
        selected = _selected_mask(batch)
        ledger.require_selected(*selected)
        original(active, model, batch, batch_idx)
        image_id, count, mask_sha256 = _selected_mask(batch)
        shared.require((image_id, count, mask_sha256) == selected,
                       "Native FIT guard changed the selected target masks")
        ledger.consume_selected(image_id=image_id, common_count=count, epoch=active.current_epoch + 1,
                                batch=batch_idx + 1, mask_sha256=mask_sha256)

    observer.on_train_batch_start = on_batch_start
