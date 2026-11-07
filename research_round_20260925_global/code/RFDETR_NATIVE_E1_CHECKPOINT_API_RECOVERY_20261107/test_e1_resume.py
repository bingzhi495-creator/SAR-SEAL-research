"""Synthetic Lightning 2.6.6 callback contract tests. No training libraries."""
from types import SimpleNamespace
import unittest

from e1_resume import make_e1_boundary_callback


class E1CheckpointCallbackTests(unittest.TestCase):
    def setUp(self):
        self.boundary = make_e1_boundary_callback(
            object, recipe_sha256='a' * 64, capture_rng=lambda: None,
        )
        self.module = SimpleNamespace(automatic_optimization=True)
        self.trainer = SimpleNamespace(
            precision='bf16-mixed', scaler=None, world_size=1, max_epochs=100,
            max_steps=-1, num_training_batches=744, estimated_stepping_batches=18600,
            accumulate_grad_batches=4, current_epoch=0, global_step=0,
            min_epochs=0, min_steps=0, should_stop=False,
            checkpoint_callbacks=[SimpleNamespace(save_weights_only=False)],
        )

    def test_e1_can_start_with_full_checkpoint_callback(self):
        self.boundary.on_train_start(self.trainer, self.module)


    def test_e1_rejects_absent_or_weights_only_checkpoint_callbacks(self):
        for callbacks in ([], [SimpleNamespace(save_weights_only=True)], [SimpleNamespace()]):
            with self.subTest(callbacks=callbacks):
                self.trainer.checkpoint_callbacks = callbacks
                with self.assertRaisesRegex(ValueError, 'full checkpointing'):
                    self.boundary.on_train_start(self.trainer, self.module)


if __name__ == '__main__':
    unittest.main()
