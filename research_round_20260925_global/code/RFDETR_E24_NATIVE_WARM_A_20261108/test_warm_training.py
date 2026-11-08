import unittest

from warm_training import training_tape


class WarmTapeTest(unittest.TestCase):
    def test_all_native_aligned_positions_are_consumed_in_both_epochs(self):
        tape=training_tape(5201,744)
        self.assertEqual(len(tape),2)
        for e,epoch in enumerate(tape):
            self.assertEqual(len(epoch),744)
            self.assertEqual(sorted(i for i,seed in epoch),list(range(744)))
            self.assertEqual([seed for _,seed in epoch],[520100000+1000*e+i for i in range(744)])
        self.assertEqual(tape,training_tape(5201,744))
        self.assertNotEqual(tape,training_tape(5203,744))

    def test_partial_optimizer_window_fails_admission(self):
        with self.assertRaises(ValueError):
            training_tape(5201,741)


if __name__=="__main__":
    unittest.main()
