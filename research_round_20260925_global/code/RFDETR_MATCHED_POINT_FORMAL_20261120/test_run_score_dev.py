import unittest
from types import SimpleNamespace
from run_score_dev import SOURCE_READOUT, STOCK, verify_readout_pair

class ScoreProvenanceTest(unittest.TestCase):
    def test_accepts_exact_readout_and_rejects_source_stock_or_arm_drift(self):
        reader = SimpleNamespace(SEEDS=(5217, 5218, 5219),
                                 ARMS=("N", "SF", "SU=F", "SU", "LF=S", "LU=S", "LW=S", "NP=S"))
        approved = {"readout_source_sha256": SOURCE_READOUT, "stock_predictions_sha256": STOCK}
        stage = {"worker_sha256": SOURCE_READOUT, "stock_predictions_sha256": STOCK,
                 "seeds": list(reader.SEEDS), "arms": list(reader.ARMS)}
        verify_readout_pair(approved, stage, reader)
        for changed_approved, changed_stage in (
            ({**approved, "readout_source_sha256": "0" * 64}, stage),
            (approved, {**stage, "worker_sha256": "0" * 64}),
            (approved, {**stage, "stock_predictions_sha256": "0" * 64}),
            (approved, {**stage, "seeds": [5217, 5218, 5220]}),
            (approved, {**stage, "arms": list(reversed(reader.ARMS))}),
        ):
            with self.assertRaises(ValueError):
                verify_readout_pair(changed_approved, changed_stage, reader)

if __name__ == "__main__":
    unittest.main()
