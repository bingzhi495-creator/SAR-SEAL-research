"""Synthetic hooks for the native E2 boundary. No training dependencies."""
from types import SimpleNamespace
import unittest

from e2_worker import make_first_update_stop


class FakeAdamW:
    def register_step_post_hook(self, hook):
        self.post_hook = hook
        return SimpleNamespace(remove=lambda: None)

    def return_from_step(self):
        self.post_hook(self, (), {})


class SizedLoader:
    num_workers = 0

    def __len__(self):
        return 744


class StopTests(unittest.TestCase):
    def setUp(self):
        self.adam = FakeAdamW()
        self.observer = SimpleNamespace(checkpoint_seen=True, first_new_update_observed=False)
        self.trainer = SimpleNamespace(
            min_epochs=0, min_steps=0, enable_validation=False, should_stop=False,
            current_epoch=1, global_step=186, optimizers=[self.adam],
            train_dataloader=SizedLoader(),
        )
        self.guard = make_first_update_stop(object, observer=self.observer, optimizer_type=FakeAdamW)
        self.guard.on_train_start(self.trainer, None)

    def window(self, *, real_adam_step):
        for index in range(4):
            self.assertFalse(self.trainer.should_stop)
            self.guard.on_train_batch_start(self.trainer, None, None, index)
            if index == 3:
                if real_adam_step:
                    self.adam.return_from_step()
                self.trainer.global_step = 187
                self.observer.first_new_update_observed = True
            self.guard.on_train_batch_end(self.trainer, None, None, None, index)

    def test_stops_after_four_completed_batches_and_one_adam_return(self):
        self.window(real_adam_step=True)
        self.guard.verify_completion(self.trainer)
        self.assertTrue(self.trainer.should_stop)
        with self.assertRaisesRegex(RuntimeError, 'unexpected resumed batch start'):
            self.guard.on_train_batch_start(self.trainer, None, None, 4)


    def test_rejects_cursor_progress_without_adam_post_hook(self):
        with self.assertRaisesRegex(RuntimeError, 'actual AdamW return'):
            self.window(real_adam_step=False)
        self.assertFalse(self.trainer.should_stop)

    def test_refuses_to_save_validate_or_test(self):
        for hook, extra in ((self.guard.on_save_checkpoint, ({},)),
                            (self.guard.on_validation_start, ()),
                            (self.guard.on_test_start, ())):
            with self.subTest(hook=hook):
                with self.assertRaisesRegex(RuntimeError, 'E2 probe refuses'):
                    hook(self.trainer, None, *extra)


if __name__ == '__main__':
    unittest.main()
