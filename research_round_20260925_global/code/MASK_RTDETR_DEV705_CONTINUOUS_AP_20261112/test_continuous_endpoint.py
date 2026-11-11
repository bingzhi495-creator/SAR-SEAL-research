from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from pycocotools import mask

import continuous_endpoint as app


def rectangle(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def row(image_id, pixels, score):
    rle = mask.encode(np.asfortranarray(pixels, dtype=np.uint8))
    return dict(image_id=image_id, category_id=1, score=score,
                segmentation=dict(size=rle["size"], counts=rle["counts"].decode("ascii")))


def write_fixture(path, data):
    raw = data if isinstance(data, bytes) else json.dumps(data).encode()
    path.write_bytes(raw)
    return dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())


class ContinuousEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = app.geometry_modules(cls.code_root)
        cls.adapter = app.load_module(cls.code_root / app.OFFICIAL_ADAPTER, app.OFFICIAL_ADAPTER_SHA, "_fixture_stock")

    def fixture(self):
        gt = dict(images=[dict(id=7, height=12, width=12)], categories=[dict(id=1, name="ship")],
                  annotations=[dict(id=1, image_id=7, category_id=1, iscrowd=0, area=12,
                                    bbox=[2.24, 2, 4, 3], segmentation=[rectangle(2.24, 2, 6.24, 5)]),
                               dict(id=2, image_id=7, category_id=1, iscrowd=1, area=4,
                                    bbox=[8, 8, 2, 2], segmentation=[rectangle(8, 8, 10, 10)])])
        exact_raster = np.zeros((12, 12), dtype=np.uint8)
        exact_raster[2:5, 2:6] = 1
        shifted = np.zeros((12, 12), dtype=np.uint8)
        shifted[2:5, 3:7] = 1
        empty = np.zeros((12, 12), dtype=np.uint8)
        streams = dict(native=[row(7, exact_raster, .8), row(7, empty, .8)],
                       center=[row(7, shifted, .8), row(7, empty, .8)])
        prior = {arm: self.modules.geometry.score(gt, rows, self.adapter)[0] for arm, rows in streams.items()}
        return gt, streams, prior

    def test_real_evaluator_keeps_rank_and_changes_only_supported_overlap(self):
        gt, streams, prior = self.fixture()
        results = app.compare(gt, streams, prior, self.adapter, self.modules,
                              expected_support=(1, 1), required_dense_spots=None)
        self.assertEqual(results["supported_unique_annotations"], 1)
        self.assertEqual(results["neutral_unique_annotations"], 1)
        self.assertEqual(results["raw_prediction_rows"], {"native": 2, "center": 2})
        self.assertEqual(results["results"]["native"]["original_full_raster"]["AP90"], prior["native"]["AP90"])
        self.assertAlmostEqual(results["results"]["native"]["shared_support_continuous"]["AP90"], 0.)
        self.assertGreater(results["geometry_audits"]["native"]["dense_geometry_spots"], 0)
        self.assertEqual(results["geometry_audits"]["native"]["neutral_original_overlap_pairs"], 2)
        self.assertTrue(results["original_order_unchanged"])

    def test_score_terminal_and_exact_footer_must_both_bind_result(self):
        with TemporaryDirectory() as path:
            score_dir = Path(path).resolve()
            result = {"status": app.SCORE_STATUS}
            result_raw = json.dumps(result).encode()
            result_sha = sha256(result_raw).hexdigest()
            (score_dir / "RESULT_SAFE.json").write_bytes(result_raw)
            stdout_path = score_dir / "ORIGINAL_SCORE.stdout"
            stdout = json.dumps({"status": app.SCORE_STATUS, "result_sha256": result_sha}).encode()
            stdout_path.write_bytes(stdout)
            terminal = dict(original_os_wait=True, original_os_exit=0, original_export_owner_wait_sha256="1" * 64,
                            stdout_ref=dict(path=stdout_path.as_posix(), bytes=len(stdout),
                                            sha256=sha256(stdout).hexdigest()))
            terminal_raw = json.dumps(terminal).encode()
            (score_dir / "ORIGINAL_SCORE_TERMINAL_PRIVATE.json").write_bytes(terminal_raw)
            terminal_sha = sha256(terminal_raw).hexdigest()
            self.assertEqual(app.verify_score_receipt(score_dir, terminal_sha, result_sha, "1" * 64), result)
            with self.assertRaisesRegex(ValueError, "closure"):
                app.verify_score_receipt(score_dir, terminal_sha, result_sha, "2" * 64)
            stdout_path.write_bytes(stdout + b" ")
            with self.assertRaisesRegex(ValueError, "digest"):
                app.verify_score_receipt(score_dir, terminal_sha, result_sha, "1" * 64)

    def test_incorrect_sealed_ap90_blocks_any_shared_support_call(self):
        gt, streams, prior = self.fixture()
        prior["center"]["AP90"] += .01
        with self.assertRaisesRegex(ValueError, "before shared support: center"):
            app.compare(gt, streams, prior, self.adapter, self.modules,
                        expected_support=(99, 99), required_dense_spots=None)

    def test_original_export_job_requires_natural_zero_and_empty_tree(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            score_dir = root / "score"
            score_dir.mkdir()
            control = root / "export/native/CONTROL"
            control.mkdir(parents=True)
            stdout = write_fixture(control / "OWNER.stdout", b"test original stdout\n")
            terminal = dict(original_os_wait=True, original_os_exit=0, job_assigned=True, owner={"pid": 8},
                            job_active_before_drain=0, job_active_after_wait=0, stop_reason=None, stdout_ref=stdout)
            terminal_path = control / "OWNER_TERMINAL_PRIVATE.json"
            closure_path = score_dir / "native_ORIGINAL_CLOSED_PRIVATE.json"
            closure = dict(arm="native", evidence="ORIGINAL_WINDOWS_OS_WAIT_AND_EXTERNAL_OWNER_TOOL_WAIT",
                           original_external_wait_sha256="1" * 64,
                           exact_tool_wait=dict(exit_code=0, output="test original stdout\n"),
                           original_child_terminal_ref=write_fixture(terminal_path, terminal))
            args = SimpleNamespace(score_dir=score_dir, export_root=root / "export", export_wait_sha256="1" * 64,
                                   native_closure_sha256=write_fixture(closure_path, closure)["sha256"])
            self.assertEqual(app.verify_exporter_closure(args, "native"), (closure_path, args.native_closure_sha256))
            terminal["job_active_after_wait"] = 1
            closure["original_child_terminal_ref"] = write_fixture(terminal_path, terminal)
            args.native_closure_sha256 = write_fixture(closure_path, closure)["sha256"]
            with self.assertRaisesRegex(ValueError, "natural exporter closure"):
                app.verify_exporter_closure(args, "native")

    def test_pinned_p392_sources_load_without_opening_any_data(self):
        scorer, dev = app.scoring_modules(self.code_root, self.p392_source)
        self.assertEqual(scorer.CANDIDATES, 100)
        self.assertEqual(dev.FILES["DEV_ORIGINAL_GT_PRIVATE.json"], app.GT_SHA)


if __name__ == "__main__":
    ContinuousEndpoint.code_root = Path(sys.argv[1]).resolve(strict=True)
    ContinuousEndpoint.p392_source = Path(sys.argv[2]).resolve(strict=True)
    unittest.main(argv=[sys.argv[0]])
