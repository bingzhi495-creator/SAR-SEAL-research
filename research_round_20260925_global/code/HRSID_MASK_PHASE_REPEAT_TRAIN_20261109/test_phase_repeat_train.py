from contextlib import redirect_stderr
from io import StringIO
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import phase_repeat_train


class RepeatTrainingTest(unittest.TestCase):
    def test_seed_controls_loader_and_survives_the_model_fit_hook(self):
        train = SimpleNamespace(seed=5201)
        model = SimpleNamespace(train_config=train)
        data = SimpleNamespace(train_config=train)
        config = SimpleNamespace(training_seed=5202, protocol="a" * 40, mode="center", stage="qualification")
        order = []
        witnesses = []
        runtime = SimpleNamespace(cpu=0, cuda=0)

        def seed_everything(seed, *, workers, verbose):
            runtime.cpu = runtime.cuda = seed
            os.environ["PL_GLOBAL_SEED"] = str(seed)
            os.environ["PL_SEED_WORKERS"] = str(int(workers))

        observer = SimpleNamespace(on_train_start=lambda *_: order.append("observed"))
        torch = SimpleNamespace(
            initial_seed=lambda: runtime.cpu,
            cuda=SimpleNamespace(initial_seed=lambda: runtime.cuda),
        )
        trainer = SimpleNamespace(global_rank=0)

        def stock_fit(module, *, datamodule, ckpt_path):
            order.append(("loader", runtime.cpu))
            seed_everything(module.train_config.seed + trainer.global_rank, workers=True, verbose=False)
            observer.on_train_start(trainer, module)

        trainer.fit = stock_fit
        with patch.dict(os.environ, {}):
            phase_repeat_train.bind_repeat_training(trainer, observer, train, config, seed_everything, torch, witnesses.append)
            seed_everything(5201, workers=True, verbose=False)
            trainer.fit(model, datamodule=data, ckpt_path=None)

        self.assertEqual(order, [("loader", 5202), "observed"])
        self.assertEqual(witnesses, [dict(
            status="HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1",
            protocol="a" * 40, mask_resize_mode="center", stage="qualification",
            initialization_seed=5201, training_seed=5202, global_rank=0, train_config_seed=5202,
            torch_cpu_initial_seed=5202, torch_cuda_initial_seed=5202, lightning_global_seed=5202,
            lightning_seed_workers=True,
        )])


    def test_archive_metadata_rejects_the_initialization_seed(self):
        train = SimpleNamespace(seed=5201)
        model = SimpleNamespace(train_config=train)
        data = SimpleNamespace(train_config=train)
        config = SimpleNamespace(training_seed=5203, protocol="a" * 40, mode="native", stage="e24")
        observer = SimpleNamespace(on_train_start=lambda *_: None)

        def stock_save(checkpoint):
            checkpoint["args"] = {"seed": 5201}

        model.on_save_checkpoint = stock_save
        trainer = SimpleNamespace(fit=lambda module, **_: module.on_save_checkpoint({}))
        phase_repeat_train.bind_repeat_training(trainer, observer, train, config, lambda *a, **k: None, None, lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "checkpoint.*training seed"):
            trainer.fit(model, datamodule=data, ckpt_path=None)


    def test_cli_rejects_the_original_seed_before_touching_training_inputs(self):
        arguments = [
            "--native-source", "missing-native.py", "--output", "missing-output",
            "--protocol", "a" * 40, "--expected-initial-sha256", "b" * 64,
            "--mode", "native", "--stage", "e24", "--training-seed", "5201",
        ]
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as failure:
            phase_repeat_train.main(arguments)
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("invalid choice: 5201", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
