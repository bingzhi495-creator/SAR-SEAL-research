from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


v = SimpleNamespace(**run_path(str(Path(__file__).with_name("verify_score.py"))))


class ScoreProvenance(unittest.TestCase):
    def setUp(self):
        self.terminal = dict(status="HRSID_OFFICIAL_TEST_NATIVE_SCORE_TERMINAL", actual_process_terminal=True,
                             actual_exit=0, footer_status=v.SUCCESS, worker_pid=102,
                             output_sha256="a"*64, admission_sha256="b"*64, score_source_sha256=v.SCORE)
        common = dict(phase=v.SCORE_PHASE, endpoint_sha256=v.ENDPOINT, cohorts_sha256=v.COHORTS, gt_sha256=v.GT,
                      adapter_sha256=v.ADAPTER, score_source_sha256=v.SCORE, runtime_versions=v.RUNTIME,
                      export_readback_sha256="c"*64, predictions_sha256="d"*64, mapped_rows=4, prediction_bytes=108)
        self.admission = dict(common, status="ROOT_ADMITTED_HRSID_OFFICIAL_TEST_NATIVE_BASELINE_SCORE_V1")
        self.result = dict(common, status=v.SUCCESS, admission_sha256=self.terminal["admission_sha256"], AP=True,
                           iou_type="segm", max_dets=[1,10,100], historical_test_selection_exposure=True,
                           units="fraction; -1 means undefined in COCO", model_forward_calls=0, optimizer_updates=0)
        self.closure = dict(status="HRSID_OFFICIAL_TEST_NATIVE_EXPORT_INDEPENDENT_READBACK_ACCEPTED",
                            independent_source_sha256=v.EXPORT_VERIFIER, resume_mode=True, original_process_terminal=True,
                            original_exit_applies_to="resumed_export_worker_pid", original_interrupted_exit=None, original_exit=0,
                            actual_resume_exit=0, endpoint_sha256=v.ENDPOINT, input_readback_sha256=v.INPUT,
                            images=1962, GT_opened=False, AP=False, predictions_sha256="d"*64, mapped_rows=4, prediction_bytes=108)

    def test_terminal_identity(self):
        v.check_terminal(self.terminal)
        other = dict(self.terminal)
        del other["worker_pid"]
        v.check_terminal(dict(other, tool_session_handle="360"))
        for key, value in (("actual_exit", None), ("actual_exit", False), ("footer_status", "wrong"),
                           ("actual_process_terminal", False), ("worker_pid", 0), ("admission_sha256", "not a digest")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                v.check_terminal(dict(self.terminal, **{key:value}))

    def test_result_admission_and_resumed_export(self):
        v.check_bindings(self.terminal, self.admission, self.result, self.closure)
        for change in (dict(original_interrupted_exit=0), dict(original_exit_applies_to="interrupted"),
                       dict(mapped_rows=7), dict(actual_resume_exit=False), dict(independent_source_sha256="e"*64)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                v.check_bindings(self.terminal, self.admission, self.result, dict(self.closure, **change))
        for change in (dict(runtime_versions={}), dict(historical_test_selection_exposure=False), dict(score_source_sha256="e"*64)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                v.check_bindings(self.terminal, self.admission, dict(self.result, **change), self.closure)

    def test_complete_prediction_byte_join(self):
        lines = [(json.dumps(dict(image_id=i, coco=[]))+"\n").encode() for i in (3, 1)]
        data = b"".join(lines)
        closed = dict(mapped_rows=0, predictions_sha256=sha256(data).hexdigest(), prediction_bytes=len(data))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.jsonl"
            path.write_bytes(data)
            rows, hashes = v.prediction_rows(path, [3, 1], closed)
            self.assertEqual(rows, {3:[], 1:[]})
            self.assertEqual(hashes[1], sha256(lines[1]).digest())
            with self.assertRaises(ValueError):
                v.prediction_rows(path, [1, 3], closed)
            for change in (dict(mapped_rows=1), dict(prediction_bytes=len(data)-1), dict(predictions_sha256="0"*64)):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    v.prediction_rows(path, [3, 1], dict(closed, **change))


if __name__ == "__main__":
    unittest.main()
