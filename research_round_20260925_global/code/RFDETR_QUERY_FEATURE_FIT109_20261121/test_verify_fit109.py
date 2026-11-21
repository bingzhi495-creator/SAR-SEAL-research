"""Synthetic closure, old budget, fold prediction and result-tamper tests."""
import json
from hashlib import sha256
from io import BytesIO
import unittest

import numpy as np

from verify_fit109 import ARMS, STATUS, accept_footer, budget_check, fold_predictions, exact


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


class ReadbackTests(unittest.TestCase):
    def test_terminal_binds_sha_session_and_footer(self):
        digest = "a"*64
        footer = dict(status=STATUS, result_sha256=digest, advances=False)
        receipt = dict(launch={"session_id":50833}, terminal={"exit_code":0,"output":json.dumps(footer)},
                       observed_footer=footer, result_sha256=digest)
        self.assertFalse(accept_footer(receipt,digest))
        for key, bad in (("launch",{"session_id":500}), ("terminal",{"exit_code":2}),
                         ("terminal",{"exit_code":0,"session_id":1}), ("result_sha256","c"*64),
                         ("terminal",{"exit_code":0,"output":"{}"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                accept_footer(dict(receipt,**{key:bad}),digest)

    def test_old_replay_not_inferred_from_new_predictions(self):
        key = (3,2,4)
        q = np.array([[.3,.2],[.4,.6]],np.float32)
        old = {key: dict(k=1,cap=2,fold=1,group="A",old_q=q,inputs_sha256="b"*64)}
        line = dict(key=list(key),k=1,cap=2,fold=1,old_q_sha256=sha256(q.tobytes()).hexdigest(),
                    inputs_sha256="b"*64)
        counts = budget_check(canonical(line),old,canonical)
        self.assertEqual(counts, (1,1))
        with self.assertRaises(ValueError):
            budget_check(canonical(dict(line,k=2)),old,canonical)
        old[key]["k"]=2
        with self.assertRaises(ValueError):
            budget_check(canonical(dict(line,k=2)),old,canonical)

    def test_restore_exact_held_zero_rows_and_fail_misalignment(self):
        ordered = [(1,5,7),(2,9,8)]
        budgets = {key: dict(k=k,coordinates=np.arange(2*n).reshape(n,2))
                   for key,k,n in zip(ordered,(0,1),(0,2))}
        arrays = dict(keys=np.asarray(ordered,dtype=np.int64), offsets=np.array([0,0,2],np.int64),
                      k=np.array([0,1],np.int8))
        arrays.update({a:np.array([[.1,.2],[.9,.8]],dtype=np.float32) for a in ARMS})
        def raw(changes):
            f=BytesIO()
            np.savez(f,**dict(arrays,**changes))
            return f.getvalue()
        observed=fold_predictions(raw({}),ordered,budgets)
        self.assertEqual(observed[ordered[0]]["Q"].shape,(0,2))
        self.assertEqual(observed[ordered[1]]["F"].shape,(2,2))
        native32=fold_predictions(raw(dict(offsets=np.array([0,0,2],np.int32))),ordered,budgets)
        np.testing.assert_array_equal(native32[ordered[1]]["Q"],observed[ordered[1]]["Q"])
        with self.assertRaises(ValueError):
            fold_predictions(raw(dict(offsets=np.array([0,0,2],np.uint32))),ordered,budgets)
        for change in (dict(keys=arrays["keys"][::-1]),dict(k=np.array([1,0],np.int8)),
                       dict(offsets=np.array([0,1,2])),dict(Z=np.ones((2,2),np.float64)),
                       dict(QP=np.array([[.3,1.1],[.4,.8]],np.float32)),dict(extra=np.zeros(1))):
            with self.subTest(change=list(change)),self.assertRaises(ValueError):
                fold_predictions(raw(change),ordered,budgets)
        with self.assertRaises(ValueError):
            exact({"gate":False},{"gate":0},canonical,"Must retain JSON type")


if __name__ == "__main__":
    unittest.main()
