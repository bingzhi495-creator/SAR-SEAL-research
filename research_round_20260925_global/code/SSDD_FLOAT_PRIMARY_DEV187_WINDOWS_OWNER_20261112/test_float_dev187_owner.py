from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import float_dev187_owner as app


class FloatOwner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = app.load_source(cls.stock_source, app.STOCK_SHA, "_fixture_stock_owner")
        cls.worker = app.load_source(cls.worker_source, app.WORKER_SHA, "_fixture_float_worker")

    def test_original_wait_binds_same_session_role_request_terminal_and_score(self):
        birth = {"pid": 218, "creation_time": 18900.5}
        record = dict(owner_birth=birth, terminal_ref={"sha256": "1" * 64}, report_ref={"sha256": "2" * 64},
                      request_ref={"sha256": "3" * 64})
        event = app.closed_event(self.worker, "SCORE", record)
        wait = dict(status=app.SCORE_WAIT_STATUS, original_session_id=42,
                    exact_tool_wait=dict(exit_code=0, output=json.dumps(event)),
                    **{key: event[key] for key in app.WAIT_KEYS})
        self.assertEqual(app.verify_original_wait(self.stock, self.worker, "SCORE", record, wait), 42)
        mutated = deepcopy(wait)
        mutated["exact_tool_wait"]["output"] = json.dumps(dict(event, score_report_sha256="4" * 64))
        with self.assertRaisesRegex(ValueError, "original outer"):
            app.verify_original_wait(self.stock, self.worker, "SCORE", record, mutated)
        mutated = deepcopy(wait)
        mutated["exact_tool_wait"]["exit_code"] = 5
        with self.assertRaisesRegex(ValueError, "original outer"):
            app.verify_original_wait(self.stock, self.worker, "SCORE", record, mutated)

    def test_real_source_preflight_uses_existing_stock_limits_without_data(self):
        ctx = app.context(self.stock_source, self.worker_source, self.code_root, self.protocol)
        self.assertEqual(ctx.stock.SECONDS, 1200)
        self.assertEqual(ctx.stock.GPU_FREE_MIB, 6144)
        self.assertEqual(ctx.sources.runtime.RAM_FLOOR, 16 * 1024**3)
        self.assertEqual(ctx.sources.runtime.RSS_CAP, 10 * 1024**3)
        self.assertEqual(ctx.image_attestation["images"], 187)
        self.assertEqual(ctx.image_attestation["distinct_jpeg_sha256"], 183)
        env = app.runtime_for(ctx.sources.runtime, "SCORE").windows_runtime(self.protocol.parent)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")

    def test_real_windows_jobobject_and_stock_readback_on_synthetic_score(self):
        ctx = app.context(self.stock_source, self.worker_source, self.code_root, self.protocol)
        with TemporaryDirectory() as path:
            root = Path(path).resolve(strict=True)
            fixture = root / "synthetic_worker.py"
            fixture.write_text(
                "from argparse import ArgumentParser\n"
                "from hashlib import sha256\n"
                "import json, os, time\n"
                "from pathlib import Path\n"
                "p = ArgumentParser()\n"
                "for arg in ('code-root','protocol','request','request-sha256','stage'):\n"
                "    p.add_argument('--' + arg, required=True)\n"
                "a = p.parse_args()\n"
                "assert a.stage == 'SCORE' and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'\n"
                "assert os.environ['OMP_NUM_THREADS'] == os.environ['OPENBLAS_NUM_THREADS'] == '1'\n"
                "time.sleep(0.9)\n"
                "raw = b'{\"synthetic\": true}\\n'\n"
                "out = Path.cwd().parent / 'ACTUAL'\n"
                "out.mkdir()\n"
                "(out / 'SCORE_SAFE.json').write_bytes(raw)\n"
                "print(json.dumps(dict(status=" + repr(ctx.worker.REPORT_EVENT) + ", stage=a.stage,"
                " request_sha256=a.request_sha256, report_sha256=sha256(raw).hexdigest())), flush=True)\n",
                encoding="utf-8")
            ctx.worker_path = fixture
            ctx.worker_ref = ctx.worker.file_ref(fixture, sha256(fixture.read_bytes()).hexdigest())
            legacy = ctx.sources.legacy
            legacy.STATE = root / "SIMULATED_RESERVED_STATE.json"
            reserved = b'{"synthetic_reserved":true}\n'
            legacy.STATE.write_bytes(reserved)
            area = root / "SCORE"
            (area / "CONTROL/scratch").mkdir(parents=True)
            request = area / "STAGE_REQUEST_PRIVATE.json"
            legacy.write_json(request, {"synthetic": True})
            request_ref = legacy.ref(request)
            coordinator = legacy.birth(os.getpid())
            terminal, guard, ref = app.stage_owner(ctx, area, request_ref, "SCORE", reserved, coordinator)
            report, closure = app.stage_readback(ctx, "SCORE", area, request_ref, terminal, guard, ref, coordinator)
            self.assertEqual(report, {"synthetic": True})
            self.assertEqual(terminal["job_active_after_wait"], 0)
            self.assertEqual(terminal["original_os_exit"], 0)
            self.assertTrue(closure["closure"]["worker_closed"])
            self.assertEqual(closure["terminal_ref"], ref)
            self.assertEqual(legacy.STATE.read_bytes(), reserved)

    def test_independent_score_recomputes_all_cell_deltas(self):
        scores_a = self.worker.score_record(dict(AP=.70, AP50=.71, AP75=.64, AP90=.30, APs=.61, AR100=.82,
                                                all12=[.70,.71,.64,.61,.63,.57,.22,.78,.82,.81,.82,.60]))
        scores_b = dict(scores_a, AP90=.33)
        cell = dict(native=scores_a, center=scores_b,
                    center_minus_native_points=self.worker.delta(scores_a, scores_b))
        app.valid_cell(self.worker, cell)
        cell["center_minus_native_points"]["AP90"] = 1.
        with self.assertRaisesRegex(ValueError, "paired score arithmetic"):
            app.valid_cell(self.worker, cell)


if __name__ == "__main__":
    FloatOwner.stock_source, FloatOwner.worker_source, FloatOwner.code_root, FloatOwner.protocol = (
        Path(name).resolve(strict=True) for name in sys.argv[1:5])
    unittest.main(argv=[sys.argv[0]])
