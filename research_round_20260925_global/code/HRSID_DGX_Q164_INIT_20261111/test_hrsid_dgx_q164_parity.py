"""Safe result contract for a CPU-only Q164 parity check."""
from unittest import TestCase

import hrsid_dgx_q164_parity as subject


class ParityTests(TestCase):
    def test_only_both_full_digests_pass_and_known_q100_pattern_is_bounded(self):
        info = dict(native_original_width=100, query_width=164, query_groups=13, class_head_slots=2,
                    coco_asset_sha256=subject.COCO_SHA, coco_asset_bytes=134545398, initialized_from_SSDD=False,
                    width100_loaded_sha256=subject.Q100, expanded_online_sha256=subject.Q164)
        observations = {
            "q100": dict(tensors=545, position_sha256=subject.WINDOWS_POSITION),
            "q164": dict(tensors=545, position_sha256=subject.WINDOWS_POSITION),
        }
        ready = subject.result_from(info, observations)
        self.assertEqual(ready["status"], subject.PARITY)
        self.assertFalse(ready["FIT"])
        self.assertFalse(ready["cuda_initialized"])
        self.assertEqual(ready["optimizer_updates"], 0)
        only_q100 = subject.result_from(dict(info, expanded_online_sha256="a" * 64), observations)
        self.assertEqual(only_q100["status"], subject.MISMATCH)
        raw = subject.result_from(dict(info, width100_loaded_sha256=subject.P242_RAW_Q100, expanded_online_sha256="b" * 64),
                                  dict(observations, q100=dict(tensors=545, position_sha256=subject.P242_RAW_POSITION)))
        self.assertTrue(raw["matches_prior_p242_uncorrected_q100_fingerprint"])
        self.assertEqual(raw["status"], subject.MISMATCH)



    def test_scope_guard_allows_only_read_only_admitted_model_asset(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            parent = Path(tmp).resolve(strict=True)
            approved = parent / "rf-detr-seg-nano.pt"
            approved.write_bytes(b"synthetic-only")
            guard = subject.scope_guard(approved)
            guard("open", (str(approved), "rb", 0))
            for path, mode in ((approved, "wb"), (parent / "other.pt", "rb"),
                               (parent / "FIT.png", "rb"), (parent / "_annotations.coco.json", "rb")):
                with self.subTest(path=path, mode=mode), self.assertRaisesRegex(ValueError, "read-only official COCO"):
                    guard("open", (str(path), mode, 0))
            for event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
                with self.subTest(event=event), self.assertRaisesRegex(ValueError, "Offline"):
                    guard(event, ())

