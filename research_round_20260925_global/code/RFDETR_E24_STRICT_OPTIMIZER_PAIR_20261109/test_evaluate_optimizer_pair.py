"""Synthetic DEV187 evaluator contracts. No protected inputs."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import evaluate_optimizer_pair as evaluator


class ClosedPairTest(unittest.TestCase):
    def test_requires_both_closed_endpoints_in_original_arm_paths(self):
        base = evaluator.load_base()
        with TemporaryDirectory() as tmp:
            pair = Path(tmp).resolve()
            protocol = "3" * 40
            refs = {}
            for arm in ("F", "S"):
                p = pair / arm / "ENDPOINT.pt"
                p.parent.mkdir()
                raw = arm.encode("ascii")
                p.write_bytes(raw)
                refs[arm] = dict(path=p.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())
            receipt = dict(schema="RFDETR_E24_OPTIMIZER_STATE_PAIR_CLOSED_V1", protocol_commit=protocol,
                           arms={arm: dict(closed=True, checkpoint_ref=ref) for arm, ref in refs.items()})
            self.assertEqual(evaluator.validate_acceptance(base, pair, receipt), (protocol, refs))
            receipt["arms"]["S"]["closed"] = False
            with self.assertRaisesRegex(ValueError, "closed"):
                evaluator.validate_acceptance(base, pair, receipt)
            receipt["arms"]["S"].update(closed=True, checkpoint_ref=refs["F"])
            with self.assertRaisesRegex(ValueError, "path"):
                evaluator.validate_acceptance(base, pair, receipt)


    def test_endpoint_requires_matching_optimizer_and_ema_clocks(self):
        base = evaluator.load_base()
        protocol = "3" * 40
        weights = {"model.weight": object()}
        meta = dict(arm="S", seed=5201, epochs=2, updates=372, microbatches=1488,
                    ema_updates=372, ema_absolute_updates=4836, starting_ema_clock=4464,
                    archive_start_sha256=base.E24_SHA, protocol_commit=protocol,
                    adam_step_counts={"4836": 543}, DEV=False, TEST=False)
        checkpoint = dict(meta=meta, ema_state=weights)
        self.assertIs(evaluator.endpoint_state(base, checkpoint, "S", protocol), weights)
        meta["adam_step_counts"] = {"372": 543}
        with self.assertRaisesRegex(ValueError, "AdamW"):
            evaluator.endpoint_state(base, checkpoint, "S", protocol)
        meta.update(adam_step_counts={"4836": 543}, ema_absolute_updates=372)
        with self.assertRaisesRegex(ValueError, "meta"):
            evaluator.endpoint_state(base, checkpoint, "S", protocol)


    def test_geometry_includes_exact_point_nine_and_limits_one_to_one_matches(self):
        import numpy as np
        from pycocotools import mask
        base = evaluator.load_base()
        geometry = evaluator.load_geometry(base)
        truth = np.ones((2, 5), dtype=np.uint8, order="F")
        candidate = truth.copy(order="F")
        candidate[0, 0] = 0

        def rle(arr):
            value = mask.encode(arr)
            return dict(size=value["size"], counts=value["counts"].decode("ascii"))

        annotation = dict(image_id=1, area=10, category_id=1, iscrowd=0, segmentation=rle(truth))
        dataset = dict(images=[dict(id=1, height=2, width=5)], annotations=[annotation, annotation.copy()])
        prediction = dict(image_id=1, category_id=1, score=0.01, segmentation=rle(candidate))
        rows = [("E24", [prediction]), ("F", []), ("S", [prediction, prediction])]
        result = evaluator.summarize_retained(base, geometry, dataset, rows, mask, np)
        self.assertEqual(result["coverage"]["E24"]["0.90"]["per_gt_max"], 2)
        self.assertEqual(result["coverage"]["E24"]["0.90"]["max_injective"], 1)
        self.assertEqual(result["coverage"]["S"]["0.90"]["max_injective"], 2)
        self.assertEqual(result["S_vs_F_per_GT"]["0.90"]["small"]["gained"], 2)


if __name__ == "__main__":
    unittest.main()
