"""Synthetic native checkpoint gate tests. No training dependencies."""
from types import SimpleNamespace
import unittest

from e100_worker import ArchiveEndpoints, BOUNDARY_KEY, schedule_factor, validate_schedule


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.archive = SimpleNamespace(save_top_k=-1)
        self.trainer = SimpleNamespace(should_stop=False)
        self.digest = 'a' * 64

    def test_e24_horizon_stops_after_only_the_first_endpoint_archive(self):
        gate = ArchiveEndpoints(self.archive, stop_epoch=24, recipe_sha256=self.digest)
        self.assertEqual(self.archive.save_top_k, 0)
        for epoch in range(1, 25):
            gate.complete_epoch(self.trainer, epoch)
            self.assertEqual(self.archive.save_top_k, -1 if epoch == 24 else 0)
            self.assertEqual(self.trainer.should_stop, epoch == 24)
        payload = {'epoch': 23, 'global_step': 4464}
        gate.mark_archive(payload)
        self.assertEqual(payload[BOUNDARY_KEY]['recipe_sha256'], self.digest)
        self.assertEqual(gate.verify_completion(), (24,))
        self.assertGreater(schedule_factor(4464), 0.05)


    def test_e100_writes_only_native_e24_e50_e100_and_rejects_duplicate(self):
        gate = ArchiveEndpoints(self.archive, stop_epoch=100, recipe_sha256=self.digest)
        for epoch in range(1, 101):
            gate.complete_epoch(self.trainer, epoch)
            selected = epoch in (24, 50, 100)
            self.assertEqual(self.archive.save_top_k, -1 if selected else 0)
            self.assertFalse(self.trainer.should_stop)
            payload = {'epoch': epoch - 1, 'global_step': epoch * 186}
            if selected:
                gate.mark_archive(payload)
                self.assertEqual(payload[BOUNDARY_KEY]['microbatches'], epoch * 744)
            with self.assertRaisesRegex(RuntimeError, 'Unexpected or duplicate'):
                gate.mark_archive(payload)
        self.assertEqual(gate.verify_completion(), (24, 50, 100))

    def test_e24_uses_e100_lr_and_rejects_a_shortened_schedule(self):
        self.assertEqual(schedule_factor(558), 1.0)
        self.assertAlmostEqual(schedule_factor(18600), 0.05)
        actual = 0.001 * schedule_factor(4464)
        optimizer = {'param_groups': [{'lr': actual}]}
        scheduler = {'last_epoch': 4464, 'base_lrs': [0.001], '_last_lr': [actual]}
        validate_schedule(optimizer, scheduler, 4464)
        scheduler['_last_lr'] = [0.001 * 0.05]
        with self.assertRaisesRegex(RuntimeError, 'Native E100 learning rate'):
            validate_schedule(optimizer, scheduler, 4464)


if __name__ == '__main__':
    unittest.main()
