"""Synthetic acceptance of the evaluator's actual final stdout envelope."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import verify_optimizer_eval_cpu as verifier


class EvaluationReadbackTest(unittest.TestCase):
    def test_worker_terminal_envelope_names_and_binds_the_exact_result(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            result = Path(tmp).resolve() / "RESULT.json"
            raw = b'{"status":"example"}\n'
            result.write_bytes(raw)
            receipt = dict(path=result.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())
            self.assertEqual(verifier.read_worker_result(json.dumps(dict(result_ref=receipt)).encode(), result),
                             {"status": "example"})
            with self.assertRaisesRegex(ValueError, "terminal envelope"):
                verifier.read_worker_result(json.dumps(receipt).encode(), result)


if __name__ == "__main__":
    unittest.main()
