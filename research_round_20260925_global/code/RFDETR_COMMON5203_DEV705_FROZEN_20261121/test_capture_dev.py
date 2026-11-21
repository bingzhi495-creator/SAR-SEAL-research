"""Generated CPU fixtures only, exercising real historical modules without pixels, weights or training outputs."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zlib

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


dev = load(HERE / "capture_dev.py", "capture_dev")
CODE = dev.release_code()
verify = load(CODE / "HRSID_OFFICIAL_FROZEN_REFINEMENT_CAPTURE_READBACK_20261121/readback.py", "official_independent")
generated = load(CODE / "HRSID_OFFICIAL_FROZEN_REFINEMENT_CAPTURE_READBACK_20261121/test_readback.py", "generated_witness")


def receipt():
    common = dict(policy="ordered_common_nonempty_prepared_rows_v1", mode="center", stage="e24", training_seed=5203,
                  initialization_seed=5201, initial_expanded_sha256=dev.INIT, fit_index_sha256=dev.FIT_INDEX,
                  qualification_pair_sha256="a"*64, training_protocol="b"*40)
    phase = dict(protocol="b"*40, seed=5203, mask_resize_mode="center", stage="e24")
    arm = dict(status="HRSID_MASK_COMMON_FIT_E24_ARM_ACCEPTED_V1", original_os_wait=True, original_os_exit=0,
               resource_stop=False, completed_epochs=24, optimizer_updates=17640, ema_updates=17640,
               filtered_empty_rows_total=0, checkpoint_sha256="c"*64, common_support=common, phase=phase)
    return dict(status="COMMON5203_SINGLE_CENTER_E24_ACCEPTED", FIT=True, TEST=False, DEV=False, AP=False, arm=arm)


class TestCaptureDev(unittest.TestCase):
    def test_historical_loader_accepts_synthetic_common5203_and_rejects_wrong_seed(self):
        arm = dev.endpoint(receipt())
        full = dev.module(CODE / "RFDETR_CENTER5202_FULL_CAPTURE_20261120/capture.py", dev.FULL, "_fixture_full")
        core = dev.load_core(CODE, full)
        self.assertEqual(core.CODE.resolve(), CODE.resolve())
        self.assertTrue(core.source("phase")[0].resolve().is_relative_to(CODE.resolve()))
        base = dev.bind_loader(core, CODE, arm)
        filt = core.load("filtered")
        phase = core.load("phase")
        common = core.load("admission")
        protocol = arm["phase"]["protocol"]
        marker = dict(protocol=protocol, epoch=24, actual_updates=17640, native_index_sha256=dev.FIT_INDEX,
                      coco_sha256=base.COCO_SHA, original_recipe_sha256=base.RECIPE_SHA, initial_expanded_sha256=dev.INIT,
                      FIT_only=True, exact_restart_trajectory_verified=False, rng_state={},
                      target_filter_sha256=filt.FILTER_SHA, filtered_empty_rows_total=0)
        archive = {"args":{"seed":5203}, common.COMMON_MARKER:arm["common_support"], phase.PHASE_MARKER:arm["phase"],
                   "pytorch-lightning_version":"2.6.6", "epoch":23, "global_step":17640, base.ENDPOINT_KEY:marker}
        base.check_endpoint(archive, protocol)
        with self.assertRaises(ValueError):
            base.check_endpoint(dict(archive, args={"seed":5202}), protocol)
        rejected = receipt()
        rejected["arm"]["checkpoint_sha256"] = dev.OLD_CKPT
        with self.assertRaises(ValueError):
            dev.endpoint(rejected)

    def test_raw_identity_catches_unmapped_query_change(self):
        raw = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in dev.SHAPES.items()}
        _, first = dev.raw_hashes(raw, torch, 9001)
        altered = dict(raw, pred_masks=raw["pred_masks"].clone())
        altered["pred_masks"][0, 0, 0, 0] += 1
        _, second = dev.raw_hashes(altered, torch, 9001)
        first, second = json.loads(first), json.loads(second)
        self.assertEqual(list(first), ["image_id","pred_logits_sha256","pred_boxes_sha256","pred_masks_sha256"])
        self.assertNotEqual(first.pop("pred_masks_sha256"), second.pop("pred_masks_sha256"))
        self.assertEqual(first, second)

    def test_packed_generated_point_witness_reconstructs_all_twelve_streams(self):
        line, block = generated.fixture()
        native = json.loads(line)
        witness = block["witness"]
        timings = block["timings"]
        predictions = {int(seed): dict(N=native, **arms) for seed, arms in block["predictions"].items()}
        packed = dev.pack(0, line, predictions, witness, timings)
        tally = Counter()
        verify.validate_image(json.loads(zlib.decompress(packed)), 0, line, tally)
        self.assertEqual((tally["mapped"], tally["images"], tally["supported"]), (2, 1, 1))
        with TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            rows, output = root/"IMAGE_ROWS_PRIVATE", root/"accepted"
            rows.mkdir()
            output.mkdir()
            (rows/"00000.json.zlib").write_bytes(packed)
            stock = root/"DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl"
            stock.write_bytes(line)
            baseline = dict(images=1, mapped_rows=2, prediction_bytes=len(line))
            receipt = dict(image_row_bytes=len(packed), image_rows_sha256=dev.hash_bytes(dev.sha256(packed).digest()),
                           phase_seconds=timings)
            refs, names, totals, _ = verify.replay(rows, stock, receipt, baseline, output, native_sha=dev.hash_bytes(line))
            self.assertEqual(totals["supported"], 1)
            for seed in dev.SEEDS:
                for arm in dev.ARMS:
                    saved = (output / names[str(seed)][arm]).read_bytes()
                    self.assertEqual(refs[str(seed)][arm]["sha256"], dev.hash_bytes(saved))
                    self.assertEqual(saved, dev.encode(predictions[seed][arm]))

    def test_conditional_render_requires_exact_sign_and_preserves_raw_evidence(self):
        raw = torch.full((1, 1, 144, 144), -1.)
        device_plane = raw.clone()
        device_plane[:, :, 70:74,70:74] = 1.
        actual = torch.nn.functional.interpolate(device_plane, (800,800), mode="bilinear", align_corners=False)[0,0]
        stock = (actual > 0).numpy()
        def disagreement(*args):
            raise ValueError("Exact native stock parity")
        with TemporaryDirectory(dir=HERE) as tmp:
            rows = []
            render = dev.render_with_backend(disagreement, raw, device_plane, [0], [3], rows, torch, np, 10, Path(tmp))
            prob = render(torch, np, None, raw[0,0].numpy(), stock)
            np.testing.assert_array_equal(prob, actual.sigmoid().numpy())
            row, = rows
            self.assertEqual((row["image_id"],row["query_id"],row["native_rank"]), (10,0,3))
            np.testing.assert_array_equal(np.load(Path(tmp)/row["plane_file"],allow_pickle=False), raw[0,0].numpy())
            np.testing.assert_array_equal(np.load(Path(tmp)/row["probability_file"],allow_pickle=False), prob)
            bad = stock.copy()
            bad[0,0] = True
            with self.assertRaisesRegex(ValueError,"same-device|Same-device"):
                render(torch, np, None, raw[0,0].numpy(), bad)


if __name__ == "__main__":
    unittest.main()
