import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import hrsid_dgx_bc_short as worker


class PlatformFixtures(unittest.TestCase):
    def test_wrong_stage_cannot_share_or_resume(self):
        with TemporaryDirectory() as root:
            work = Path(root).resolve()
            for mode in worker.MODES:
                (work / mode).mkdir()
            cfg = SimpleNamespace(mode="one_pass", training_lock=work / "LOCK.json",
                                  training_lock_sha256="a" * 64)
            platform = {"work_root": work.as_posix(), "training_protocol": "b" * 40}
            args = worker.arm_args(cfg, platform)
            self.assertEqual((args.mode, args.stage, args.training_seed), ("one_pass", "qualification", 5202))
            self.assertEqual(args.output, work / "one_pass" / "ACTUAL")
            self.assertEqual(args.audit_output, work / "one_pass" / "AUDIT")
            self.assertEqual(args.expected_initial_sha256, worker.Q164)
            args.audit_output.mkdir()
            with self.assertRaisesRegex(ValueError, "Fresh"):
                worker.arm_args(cfg, platform)

    def test_fit_index_exact_roster_and_no_split(self):
        with TemporaryDirectory() as root:
            layout = Path(root).resolve()
            train = layout / "train"
            train.mkdir()
            index = train / "_annotations.coco.json"
            index.write_text(json.dumps({"images":[{"id":7,"width":800,"height":800,"file_name":"a.png"}],
                                         "annotations":[{"image_id":7}]}), encoding="utf-8")
            (train / "a.png").touch()
            ref = worker.closed_ref(index)
            self.assertEqual(worker.fit_ids(layout, index_ref=ref, counts=(1,1)), {7})
            (train / "unexpected.png").touch()
            with self.assertRaisesRegex(ValueError, "PNG roster"):
                worker.fit_ids(layout, index_ref=ref, counts=(1,1))
            (layout / "test").mkdir()
            with self.assertRaisesRegex(ValueError, "FIT-only"):
                worker.fit_ids(layout, index_ref=ref, counts=(1,1))

    def test_scope_allows_only_exact_science_inputs(self):
        with TemporaryDirectory() as root:
            layout = Path(root).resolve()
            train = layout / "train"
            train.mkdir()
            coco = layout / "original.pt"
            coco.touch()
            (train / "a.png").touch()
            allowed = worker.scope_guard(train, {coco})
            allowed("open", (str(coco), "r", 0))
            allowed("open", (str(train / "a.png"), "r", 0))
            with self.assertRaises(PermissionError):
                allowed("open", (str(coco), "w", 1))
            with self.assertRaises(PermissionError):
                allowed("open", (str(layout / "dev" / "test.png"), "r", 0))
            with self.assertRaises(PermissionError):
                allowed("socket.connect", ())

    def test_original_per_arm_lock_accepts_only_matching_fresh_mode(self):
        import os
        code = Path(os.environ["P297_ORDINARY_CODE"]).resolve()
        _, engine, plan = worker.source_plan(code)
        with TemporaryDirectory() as root:
            work = Path(root).resolve()
            for mode in worker.MODES:
                (work / mode).mkdir()
            cfg = SimpleNamespace(mode="one_pass", training_lock=work / "LOCK.json")
            platform = {"work_root": work.as_posix(), "training_protocol": "b" * 40}
            cfg.training_lock_sha256 = "0" * 64
            args = worker.arm_args(cfg, platform)
            locked = dict(status=engine.LOCK_STATUS, training_protocol=args.protocol, stage="qualification",
                          mode=args.mode, training_seed=5202, initialization_seed=5201,
                          initial_expanded_sha256=worker.Q164, fit_index_sha256=worker.INDEX,
                          native_output=args.output.as_posix(), audit_output=args.audit_output.as_posix(),
                          implementation_commit="c" * 40, sources=plan["scientific_sources"], qualification_pair_ref=None)
            args.training_lock.write_text(json.dumps(locked), encoding="utf-8")
            args.training_lock_sha256 = worker.closed_ref(args.training_lock)["sha256"]
            self.assertEqual(engine.read_training_lock(args), locked)
            self.assertEqual(engine.SELF, worker.QUALIFIER)
            self.assertEqual(engine.manifest_digest(locked), engine.manifest_digest(engine.read_training_lock(args)))
            for field, bad in (("mode", "stepwise_center"), ("qualification_pair_ref", {})):
                with self.subTest(field=field):
                    modified = dict(locked, **{field:bad})
                    args.training_lock.write_text(json.dumps(modified), encoding="utf-8")
                    args.training_lock_sha256 = worker.closed_ref(args.training_lock)["sha256"]
                    with self.assertRaises(RuntimeError):
                        engine.read_training_lock(args)

    def test_initialization_substitution_is_local_and_bound_before_witness(self):
        from unittest.mock import Mock
        with TemporaryDirectory() as root:
            audit = Path(root).resolve()
            base = object()
            state_api = SimpleNamespace(tensor_dict_sha256=lambda data: data)
            p290 = SimpleNamespace(Q100="q100", QUERIES=("query", "refpoint"), observe=Mock(return_value={
                "queries": {"query":{"sha256":"prior", "group_tail_sha256":"prior-tail", "group_prefix_sha256":"prefix"},
                            "refpoint":{"group_prefix_sha256":"rp-prefix"}}}))
            p291 = SimpleNamespace(canonicalize=Mock(return_value=({"position":"exact"}, {})))
            p293 = SimpleNamespace(QUERY_KEY="query", QUERY_PREFIX="prefix", POSITION_ONLY_Q164="q164-before",
                                   POSITION_ONLY_QUERY="prior", POSITION_ONLY_TAIL="prior-tail",
                                   verify_tail_receipt=Mock(return_value="typed"), load_tail=Mock(return_value="tail"))
            module = SimpleNamespace(model=SimpleNamespace(state_dict=Mock(return_value="q164-before")))

            def tail_update(*args):
                module.model.state_dict.return_value = worker.Q164
                return {"tail":"exact"}

            p293.copy_query_tail = Mock(side_effect=tail_update)
            phase = SimpleNamespace(exclusive_json=Mock())
            source = {"position_audit":{"position":"exact"}, "query_tail_audit":{"tail":"exact"}}
            receipt = {"query_prefix_sha256":{"query":"prefix", "refpoint":"rp-prefix"}}

            def original(active_base, output, build):
                self.assertIs(active_base, base)
                self.assertIs(build(SimpleNamespace(num_queries=100), None), module)
                self.assertIs(build(SimpleNamespace(num_queries=164), None), module)
                return None, None, module, state_api, dict(width100_loaded_sha256="q100", expanded_online_sha256="q164-before")

            native = SimpleNamespace(initialize=original)
            worker.install_initialization(native, base, state_api, p290, p291, p293, Mock(),
                                          {"position":None, "query_tail":None}, source, receipt, audit, phase, object(), provenance={"mode":"one_pass"})
            _, _, _, _, info = native.initialize(base, audit, lambda *a: module)
            self.assertEqual(info["expanded_online_sha256"], worker.Q164)
            p291.canonicalize.assert_called_once()
            p293.load_tail.assert_called_once()
            p293.copy_query_tail.assert_called_once()
            output_path, witness = phase.exclusive_json.call_args.args
            self.assertEqual(output_path, audit / "DGX_INITIALIZATION_WITNESS_SAFE.json")
            self.assertEqual((witness["q164_sha256"], witness["optimizer_updates"]), (worker.Q164, 0))
            module.model.state_dict.return_value = "unexpected"
            p293.load_tail.reset_mock()
            phase.exclusive_json.reset_mock()
            with self.assertRaisesRegex(ValueError, "intermediate"):
                native.initialize(base, audit, lambda *a: module)
            p293.load_tail.assert_not_called()
            phase.exclusive_json.assert_not_called()
