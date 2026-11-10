from types import SimpleNamespace
import unittest

import numpy as np
from paired_short import ARM_STATUS, EmptyRows, ROW_KEY, Witness


def center(masks, size):
    self = masks[:, 1:2, 1:2]
    assert self.shape[-2:] == size
    return self.copy()


class Stock:
    interp = 0
    def apply(self, sample, context=None):
        sample["gt_segm"] = sample["gt_segm"][:, :1, :1].copy()
        sample["im_shape"] = np.array([1, 1], dtype=np.float32)
        return sample


def sample(mask):
    return {
        "im_id": np.array([8]), ROW_KEY: np.array([[3]]),
        "gt_segm": np.array([mask], dtype=np.uint8),
        "gt_bbox": np.array([[0, 0, 1, 1]], dtype=np.float32),
        "gt_class": np.array([[0]], dtype=np.int32),
    }


class PairingTest(unittest.TestCase):
    def test_center_is_only_model_mask_replacement_and_keeps_empty_witness(self):
        outputs = []
        audits = []
        for arm in ("native", "center"):
            witness = Witness(arm, np, SimpleNamespace(), center)
            output = witness.resize_type(Stock)().apply(sample([[1, 1], [0, 0]]))
            self.assertNotIn(ROW_KEY, output)
            self.assertEqual(output["gt_bbox"].tolist(), [[0, 0, 1, 1]])
            self.assertEqual(output["gt_class"].tolist(), [[0]])
            outputs.append(output["gt_segm"].item())
            audits.append(witness.pending.popleft())
        self.assertEqual(outputs, [1, 0])
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(audits[0]["rows"], [3])
        self.assertEqual(audits[0]["nonempty_native"], [True])
        self.assertEqual(audits[0]["nonempty_center"], [False])

    def observe(self, arm, *, empty=False):
        def both_nonempty(masks, size):
            return np.array([[[0, 1], [0, 0]]], dtype=np.uint8) if not empty else np.zeros((1, 2, 2), dtype=np.uint8)

        def tensor(value):
            return SimpleNamespace(numpy=lambda: value, __len__=lambda: len(value))

        class Tensor:
            def __init__(self, value):
                self.value = value
            def numpy(self):
                return self.value
            def __len__(self):
                return len(self.value)

        class Resize2:
            interp = 1
            def apply(self, row, context=None):
                row["im_shape"] = np.array([2, 2], dtype=np.float32)
                row["gt_segm"] = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)
                return row

        paddle = SimpleNamespace(get_rng_state=lambda **_: ["-1 " + "1 " * 314 + "0"])
        witness = Witness(arm, np, paddle, both_nonempty)
        rows = []
        for iid in range(4):
            row = sample([[1, 1], [0, 0]])
            row["gt_segm"] = np.pad(row["gt_segm"], ((0, 0), (0, 1), (0, 1)))
            row["im_id"] = np.array([iid])
            rows.append(witness.resize_type(Resize2)().apply(row))
        data = {field: [Tensor(row[field]) for row in rows] for field in ("gt_segm", "gt_bbox", "gt_class")}
        data.update(im_id=tensor(np.array([row["im_id"] for row in rows])),
                    image=tensor(np.zeros((4, 3, 2, 2), dtype=np.float32)), epoch_id=0,
                    im_shape=tensor(np.ones((4, 2), dtype=np.float32)),
                    scale_factor=tensor(np.ones((4, 2), dtype=np.float32)))
        layer = SimpleNamespace(state_dict=lambda: {str(i): np.array([i]) for i in range(646)})
        if empty:
            with self.assertRaises(EmptyRows):
                witness.pre_forward(layer, (data,))
        else:
            witness.pre_forward(layer, (data,))
        return witness.report(ARM_STATUS, {})

    def test_actual_batch_keeps_common_ledger_and_selected_stream_differs(self):
        a = self.observe("native")
        b = self.observe("center")
        self.assertEqual(a["ordered_image_ids"], [0, 1, 2, 3])
        self.assertEqual(a["initial_model_sha256"], b["initial_model_sha256"])
        self.assertEqual(a["common_ledger_sha256"], b["common_ledger_sha256"])
        self.assertNotEqual(a["selected_mask_stream_sha256"], b["selected_mask_stream_sha256"])
        self.assertEqual(a["changed_masks"], 4)

    def test_actual_batch_rejects_and_records_differential_empty(self):
        failed = self.observe("center", empty=True)
        self.assertEqual((failed["empty_native"], failed["empty_center"], failed["differential_empty"]), (0, 4, 4))
        self.assertEqual(failed["ordered_image_ids"], [0, 1, 2, 3])
