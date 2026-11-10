import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import p304_bc_cpu_init as probe


class ClosedCPUInitTests(unittest.TestCase):
    def test_lock_rejects_extra_fit_input_before_worker_or_data_reads(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock = root / "lock.json"
            cfg = {name: str(root / name) for name in probe.PATH_FIELDS}
            cfg.update({name: {} for name in probe.REF_FIELDS}, status=probe.LOCK_STATUS)
            cfg["fit_layout"] = str(root / "forbidden")
            raw = json.dumps(cfg).encode()
            lock.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "CPU init lock"):
                probe.read_lock(lock, sha256(raw).hexdigest())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["lock.json"])

    def test_scope_refuses_even_a_train_like_image_or_annotation_name(self):
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            observations = []
            worker = SimpleNamespace(scope_guard=lambda *_: lambda *args: observations.append(args))
            sentinel = root / "UNUSED_FIT_LAYOUT"
            coco = root / "rf-detr-seg-nano.pt"
            guard = probe.cpu_scope(worker, sentinel, {coco})
            for name in ("image.png", "_annotations.coco.json"):
                with self.subTest(name=name), self.assertRaises(PermissionError):
                    guard("open", (str(sentinel / "train" / name), "r", 0))
            guard("open", (str(coco), "rb", 0))
            self.assertEqual(len(observations), 1)
