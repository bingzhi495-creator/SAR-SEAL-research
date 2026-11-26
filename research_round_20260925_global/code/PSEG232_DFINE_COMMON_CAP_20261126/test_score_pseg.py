"""Generated-data tests for pre-label closure and equal signed geometry."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from pycocotools import mask

from score_pseg import SPARSE, decision, module, selected_geometry, validate_qualification


class ScoreGuardTests(unittest.TestCase):
    def test_label_free_receipt_requires_independent_exact_same_write(self):
        contract = SimpleNamespace(protocol_sha="a" * 64, prelock_commit="b" * 40,
                                   render_sha="c" * 64, render_terminal_sha="d" * 64)
        from score_pseg import QUALIFIED_SOURCE, QUALIFIED_COUNTS
        receipts = dict(**QUALIFIED_SOURCE, counts=dict(QUALIFIED_COUNTS, other=1), solver_totals=dict(feasible=2),
                        protocol_sha256=contract.protocol_sha,
                        prelock_commit=contract.prelock_commit, edits_sha256="e" * 64,
                        eof_complete=True, exact_archived_five_arm_bytes=True, original_f_i_reconstructed=True,
                        exact_signed_cap_readback=True, label_files_opened=0, detector_forwards=0,
                        optimizer_updates=0)
        render = dict(receipts, status="PSEG232_DFINE_COMMON_CAP_RENDER_COMPLETE", independent_replay=False,
                      render_result_sha256=None, render_terminal_sha256=None)
        verified = dict(receipts, status="PSEG232_DFINE_COMMON_CAP_VERIFY_COMPLETE", independent_replay=True,
                        render_result_sha256=contract.render_sha, render_terminal_sha256=contract.render_terminal_sha)
        validate_qualification(contract, render, verified)
        for corrupted in (dict(verified, edits_sha256="1" * 64),
                          dict(verified, render_terminal_sha256="2" * 64),
                          dict(verified, label_files_opened=1),
                          dict(verified, counts=dict(QUALIFIED_COUNTS, swaps=1))):
            with self.assertRaises(ValueError):
                validate_qualification(contract, render, corrupted)

    def test_reconstructs_equal_edits_and_rejects_pair_above_common_cap(self):
        sparse = module(Path(os.environ["SAR_SEAL_PUBLISHED_SPARSE"]), SPARSE, "_pseg_sparse_test")
        native = np.zeros((3, 5), np.uint8, order="F")
        native[0, 0] = native[1, 4] = 1
        original = mask.encode(native)
        original["counts"] = original["counts"].decode("ascii")
        item = dict(coordinates=[[1, 0], [8, 9]], fixed_pairs=[[0, 0]],
                    bounded_pairs=[[1, 1]], independent_pairs=[[0, 1]], cap_squared=1)
        for arm, changes in (("fixed", [1, 0]), ("bounded", [8, 9]), ("free_matched_count", [1, 9])):
            item[arm] = dict(swaps=1, rle=sparse.xor_rle(original, changes))
        self.assertEqual(selected_geometry(sparse, original, item), 1)
        changed = deepcopy(item)
        changed["bounded_pairs"] = changed["independent_pairs"]
        changed["bounded"] = changed["free_matched_count"]
        with self.assertRaisesRegex(ValueError, "within F's cap"):
            selected_geometry(sparse, original, changed)

    def test_signed_prerequisite_thresholds(self):
        self.assertEqual(decision(.30, [.01, .20], 20, 8), "positive")
        self.assertEqual(decision(-.30, [-.20, -.01], 20, 8), "negative")
        for coverage, interval, effect in ((19, [.01, .2], .30), (20, [0, .2], .30), (20, [.1, .2], .299)):
            self.assertEqual(decision(effect, interval, coverage, 8), "inconclusive")


if __name__ == "__main__":
    unittest.main()
