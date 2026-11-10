"""Synthetic role, closure and scoring dispatch checks for the private P284 worker."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssdd_dev187_stage_worker as worker


def ref(path, digest="a" * 64, size=1):
    return dict(path=path.as_posix(), bytes=size, sha256=digest)


def common(root, stage):
    return dict(status=worker.REQUEST_STATUS, stage=stage, training_protocol="a" * 40,
                evaluation_protocol="b" * 40, training_pair_ref=ref(root / "accepted_pair.json"),
                native_arm_ref=ref(root / "accepted_native.json"),
                center_arm_ref=ref(root / "accepted_center.json"),
                image_manifest_ref=ref(root / "dev_manifest.json", worker.IMAGE_MANIFEST_SHA),
                output_directory=(root / "new").as_posix())


class RequestTest(unittest.TestCase):
    def test_export_request_is_pinned_and_rejects_a_label_role(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "REQUEST.json"
            request = dict(common(root, "EXPORT_NATIVE"), archive_ref=ref(root / "checkpoint_23.ckpt"),
                           dev_root=(root / "images").as_posix())

            def write(document):
                raw = json.dumps(document).encode("utf-8")
                path.write_bytes(raw)
                return sha256(raw).hexdigest()

            expected = write(request)
            self.assertEqual(worker.read_request(path, expected, "EXPORT_NATIVE"), request)
            with self.assertRaisesRegex(ValueError, "request SHA-256"):
                worker.read_request(path, "0" * 64, "EXPORT_NATIVE")
            with self.assertRaisesRegex(ValueError, "exact stage"):
                worker.read_request(path, expected, "EXPORT_CENTER")
            test_role = dict(request, image_manifest_ref=ref(root / "test" / "dev_manifest.json", worker.IMAGE_MANIFEST_SHA))
            with self.assertRaisesRegex(ValueError, "Reserved TEST paths"):
                worker.read_request(path, write(test_role), "EXPORT_NATIVE")
            bad = dict(request, integer_gt_ref=ref(root / "not_admitted.json"))
            with self.assertRaisesRegex(ValueError, "exact stage"):
                worker.read_request(path, write(bad), "EXPORT_NATIVE")


def closed_arm(root, letter, terminal):
    report = ref(root / worker.EXPORT_REPORT, letter * 64)
    return dict(closure=dict(worker.CLOSURE, terminal_sha256=terminal * 64, report_sha256=report["sha256"]),
                report_ref=report, predictions_ref=ref(root / "ARTIFACTS/PREDICTIONS.json"),
                export_manifest_ref=ref(root / "ARTIFACTS/IMAGE_MANIFEST.json"),
                checkpoint_ref=ref(root / "checkpoint/checkpoint_23.ckpt", letter * 64))


class RuntimeContractTest(unittest.TestCase):
    def test_score_request_rejects_shared_or_unwaited_export_closure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "REQUEST.json"
            request = dict(common(root, "SCORE"),
                           native=closed_arm(root / "a", "c", "e"),
                           center=closed_arm(root / "b", "d", "f"),
                           integer_gt_ref=ref(root / "integer.json", worker.INTEGER_GT_SHA, worker.INTEGER_GT_BYTES))

            def read():
                raw = json.dumps(request).encode("utf-8")
                path.write_bytes(raw)
                return worker.read_request(path, sha256(raw).hexdigest(), "SCORE")

            self.assertEqual(read(), request)
            request["center"]["closure"]["original_os_wait"] = False
            with self.assertRaisesRegex(ValueError, "Original SSDD export OS wait"):
                read()
            request["center"]["closure"]["original_os_wait"] = True
            request["center"]["closure"]["terminal_sha256"] = request["native"]["closure"]["terminal_sha256"]
            with self.assertRaisesRegex(ValueError, "Distinct closed export processes"):
                read()

    def test_published_windows_runtime_loads_exact_config_without_constructing_model(self):
        code = Path(__file__).resolve().parents[1]
        modules = worker.load_published(code, scoring=True)
        api, init, score = (
            modules["ssdd_phase_dev_admission"], modules["ssdd_phase_init"], modules["ssdd_phase_dev_score"])
        self.assertEqual(score.INTEGER_GT_SHA, worker.INTEGER_GT_SHA)
        self.assertEqual(api.IMAGE_MANIFEST_SHA, worker.IMAGE_MANIFEST_SHA)
        with TemporaryDirectory() as temporary:
            model, train = worker.load_recipe(init, Path(temporary).resolve())
        self.assertEqual((model.num_queries, model.num_select, model.group_detr, model.resolution), (100, 100, 13, 576))
        self.assertEqual((train.seed, train.epochs, train.run_test), (5201, 100, False))

    def test_stage_scope_allows_exact_jpeg_but_blocks_test_and_weight(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            allowed = root / "valid.jpg"
            allowed.write_bytes(b"fixture bytes")
            guard, seen = worker.scope_guard(allowed=(allowed,), images=(allowed,), data_root=root)
            guard("open", (str(allowed), "r", 0))
            self.assertEqual(seen, {allowed})
            for denied in (root / "test" / "other.jpg", root / "label.json", root.parent / "foreign.ckpt"):
                with self.assertRaisesRegex(PermissionError, "outside its exact read-only stage roles"):
                    guard("open", (str(denied), "r", 0))
            with self.assertRaises(PermissionError):
                guard("open", (str(allowed), "w", 0))

    def test_unaccepted_pair_never_reaches_gpu_setup_or_score_artifact_loading(self):
        api = SimpleNamespace(read_pair=lambda *a, **k: (_ for _ in ()).throw(ValueError("not independently accepted")))
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            export = common(root, "EXPORT_NATIVE")
            with patch.object(worker, "configure_export", side_effect=AssertionError("early CUDA")), \
                    patch.object(worker, "read_artifact", side_effect=AssertionError("early score bytes")):
                with self.assertRaisesRegex(ValueError, "not independently accepted"):
                    worker.run_export(export, api, object(), event_sink=None)
                with self.assertRaisesRegex(ValueError, "not independently accepted"):
                    worker.run_score(common(root, "SCORE"), api, object())


if __name__ == "__main__":
    unittest.main()
