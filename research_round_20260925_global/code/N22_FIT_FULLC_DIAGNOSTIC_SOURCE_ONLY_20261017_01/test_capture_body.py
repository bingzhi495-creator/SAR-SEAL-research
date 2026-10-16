# encoding: utf-8
"""UNIT_FIXTURE only; no scientific libraries, files or models."""
import importlib.util
from pathlib import Path
import unittest
P = Path(__file__).with_name("capture_body.py")
spec = importlib.util.spec_from_file_location("n22_unbound_unit", P)
body = importlib.util.module_from_spec(spec)
spec.loader.exec_module(body)

class CaptureScopeTests(unittest.TestCase):
    def test_all100_includes_nonempty_C_when_K_zero(self):
        calls = []
        def collect(q):
            calls.append(q)
            return dict(query_id=q, C_count=3, K=0 if q==7 else 1)
        rows = body.collect_all100(collect)
        self.assertEqual(calls, list(range(100)))
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[7], dict(query_id=7, C_count=3, K=0))
        with self.assertRaises(RuntimeError):
            body.collect_all100(lambda q: dict(query_id=99-q, C_count=3, K=0))

    def test_locked32_identity_and_refs_fail_closed(self):
        rows = []
        for i in range(32):
            identity = dict(image_id=1000+i)
            rows.append(dict(dataset_index=i, old_row=dict(dataset_index=i,
                image_id=1000+i, identity=identity, matching_loss_seed=7000+i),
                packet_ref=dict(path=f"E:/FAKE/packet{i}.pth", bytes=1, sha256="a"*64),
                native_L_ref=dict(path=f"E:/FAKE/L{i}.pth", bytes=1, sha256="b"*64)))
        spec = dict(selected_fit_rows=rows, expected_selected_dataset_indices=list(range(32)),
                    expected_selected_image_ids=list(range(1000,1032)))
        self.assertEqual(body.validate_selected_rows(spec), rows)
        for mutate in (lambda s:s["selected_fit_rows"].pop(),
                       lambda s:s["selected_fit_rows"][1].update(dataset_index=0),
                       lambda s:s["expected_selected_image_ids"].reverse()):
            import copy
            changed = copy.deepcopy(spec)
            mutate(changed)
            with self.assertRaises(RuntimeError):
                body.validate_selected_rows(changed)

if __name__ == "__main__":
    unittest.main()
