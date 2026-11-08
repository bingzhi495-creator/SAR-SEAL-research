import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hrsid_coco_native_e24 import archive_due, verify_loop, optimizer_state


class NativeE24Tests(unittest.TestCase):
    def test_archives_only_at_complete_fixed_endpoints(self):
        self.assertEqual([n for n in range(1, 25) if archive_due(n)], [1, 8, 24])
        self.assertFalse(archive_due(0))

    def test_epoch_archive_preserves_distinct_current_and_total_cursors(self):
        epoch = 8
        steps, visits = 735, 2940
        def counts(n):
            return dict(ready=n, started=n, processed=n, completed=n)

        epochs = dict(counts(epoch), completed=epoch - 1)
        fit = {
            "epoch_progress": dict(current=epochs.copy(), total=epochs.copy()),
            "epoch_loop.batch_progress": dict(current=counts(visits), total=counts(visits * epoch), is_last_batch=True),
            "epoch_loop.automatic_optimization.optim_progress": {
                "optimizer": {"step": dict(current=counts(steps), total=counts(steps * epoch))}
            },
        }
        record = dict(epoch=epoch - 1, global_step=steps * epoch, loops={"fit_loop": fit})
        record["pytorch-lightning_version"] = "2.6.6"
        verify_loop(record, epoch)
        fit["epoch_loop.batch_progress"]["total"]["completed"] -= 1
        with self.assertRaisesRegex(RuntimeError, "batch or step cursor"):
            verify_loop(record, epoch)

    def test_e24_cosine_clock_is_sized_for_e100(self):
        step = 17640
        factor = .05 + .95 * .5 * (1 + math.cos(math.pi * (step - 2205) / (73500 - 2205)))
        lr = 1e-4 * factor
        adam = dict(state={0: dict(step=step, exp_avg=None, exp_avg_sq=None)}, param_groups=[{"lr": lr}])
        scheduler = dict(last_epoch=step, base_lrs=[1e-4], _last_lr=[lr])
        self.assertEqual(optimizer_state(adam, scheduler, step)["adam_parameter_clocks"], {step: 1})
        with self.assertRaisesRegex(RuntimeError, "scheduler cursor"):
            optimizer_state(adam, dict(scheduler, last_epoch=step + 1), step)


if __name__ == "__main__":
    unittest.main()
