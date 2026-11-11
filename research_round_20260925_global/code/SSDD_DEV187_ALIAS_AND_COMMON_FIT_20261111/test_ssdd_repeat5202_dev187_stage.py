"""Regression for the changed seed bindings using the existing exact scorer fixture."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest


CODE = Path(sys.argv.pop(1)).resolve(strict=True)
subject_path = Path(__file__).with_name("ssdd_repeat5202_dev187_stage.py")
spec = importlib.util.spec_from_file_location("repeat_dev", subject_path)
repeat_dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repeat_dev)
FIXTURE = CODE / "SSDD_MASK_PHASE_TRANSFER_20261110/test_ssdd_phase_dev_score.py"


def scoring_fixture(score):
    fixture = ModuleType("_original_score_fixture")
    fixture.__file__ = str(FIXTURE)
    name = "ssdd_phase_dev_score"
    previous = sys.modules.get(name)
    sys.modules[name] = score
    try:
        exec(compile(FIXTURE.read_bytes(), str(FIXTURE), "exec"), fixture.__dict__)
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return fixture


class Repeat5202DevStageTest(unittest.TestCase):
    def setUp(self):
        self.stage = repeat_dev.load_stage(CODE)
        self.repeat = repeat_dev.configure_repeat_modules(
            self.stage.load_published, self.stage.PINS, CODE, scoring=True
        )

    def test_fixed_cohort_scoring_preserves_every_metric(self):
        stock = self.stage.load_published(CODE, scoring=True)["ssdd_phase_dev_score"]
        fixture = scoring_fixture(stock)
        original, _ = fixture.fixture()
        expected = fixture.evaluate(original)
        score = self.repeat["ssdd_phase_dev_score"]
        repeat_fixture = scoring_fixture(score)
        args, _ = repeat_fixture.fixture()
        with self.assertRaisesRegex(ValueError, "accepted SSDD E24"):
            repeat_fixture.evaluate(args)
        pair = json.loads(args["training_pair"].raw)
        pair["seed"] = 5202
        args["training_pair"] = repeat_fixture.artifact(args["training_pair"].ref["path"], pair)
        for mode in ("native", "center"):
            arm = args[mode]
            report = json.loads(arm.report.raw)
            report["training_seed"] = 5202
            report["training_pair_sha256"] = args["training_pair"].ref["sha256"]
            args[mode] = repeat_fixture.rebind(arm, report=report)
        measured = repeat_fixture.evaluate(args)
        self.assertEqual(measured["images"], 187)
        self.assertEqual(measured["annotations"], 380)
        self.assertEqual(measured["primary"], expected["primary"])
        self.assertEqual(measured["arms"]["center"]["zero_prediction_images"], 1)

        report = json.loads(args["center"].report.raw)
        report["training_seed"] = 5201
        with self.assertRaisesRegex(ValueError, "original pair, arm, mode and image manifest"):
            repeat_fixture.evaluate(dict(args, center=repeat_fixture.rebind(args["center"], report=report)))

    def test_archive_keeps_init_seed_and_rejects_wrong_training_seed(self):
        stock = self.stage.load_published(CODE, scoring=False)["ssdd_phase_dev_admission"]
        api = self.repeat["ssdd_phase_dev_admission"]
        original = stock.phase_marker("native", "1"*40, "2"*64, "3"*64)
        phase = api.phase_marker("native", "1"*40, "2"*64, "3"*64)
        self.assertEqual(original["seed"], 5201)
        self.assertEqual(phase, dict(original, seed=5202))
        calls = []
        native = SimpleNamespace(
            BOUNDARY_KEY="boundary",
            endpoint_marker=lambda epoch, recipe: {"epoch": epoch, "recipe": recipe},
            validate_archive_progress=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        check = api.bind_archive_validator({"phase": phase}, long_worker=native)
        ckpt = {
            "epoch": 23,
            "global_step": 4464,
            api.PHASE_MARKER: phase,
            "args": {"seed": 5202},
            "model_config": {
                "num_queries": 100, "num_select": 100, "group_detr": 13,
                "resolution": 576, "num_classes": 1, "segmentation_head": True,
            },
            "boundary": native.endpoint_marker(24, api.RECIPE_SHA),
        }
        check(ckpt, 24, boundary_api=None, recipe_sha256=api.RECIPE_SHA)
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(ValueError, "training seed 5202"):
            check(dict(ckpt, args={"seed": 5201}), 24, boundary_api=None, recipe_sha256=api.RECIPE_SHA)
        with self.assertRaisesRegex(ValueError, "phase marker"):
            check(
                dict(ckpt, **{api.PHASE_MARKER: dict(phase, initialization_seed=5202)}),
                24, boundary_api=None, recipe_sha256=api.RECIPE_SHA,
            )


if __name__ == "__main__":
    unittest.main()

