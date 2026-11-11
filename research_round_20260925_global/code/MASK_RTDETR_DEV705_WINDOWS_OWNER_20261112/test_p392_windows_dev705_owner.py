import unittest

import p392_windows_dev705_owner as subject


class OriginalCudaOwnerTest(unittest.TestCase):
    def test_stage_order_and_original_job_gate(self):
        self.assertEqual(subject.STAGES, ("fit_smoke", "image_stage", "native", "center"))
        terminal = dict(original_os_wait=True, original_os_exit=0, owner={"pid": 5, "creation_time": 2.0},
                        job_assigned=True, job_active_before_drain=0, job_active_after_wait=0, stop_reason=None)
        subject.require_clean_job(terminal, lambda _: False)
        for corrupted in (dict(terminal, original_os_exit=False), dict(terminal, job_active_after_wait=1),
                          dict(terminal, stop_reason="guard")):
            with self.assertRaisesRegex(ValueError, "original.*JobObject"):
                subject.require_clean_job(corrupted, lambda _: False)
        with self.assertRaisesRegex(ValueError, "original.*JobObject"):
            subject.require_clean_job(terminal, lambda _: True)

    def test_export_arms_must_share_stage_pair_and_bind_distinct_receipts(self):
        stage, pair = "a"*64, "b"*64
        expected = {arm: dict(path=None, checkpoint_sha256=code*64, training_report_sha256="f"*64,
                             training_independent_receipt_sha256=stamp*64)
                    for arm, code, stamp in (("native", "c", "d"), ("center", "e", "f"))}
        reports = {arm: {k: v for k, v in ref.items() if k != "path"} | dict(
            arm=arm, image_stage_report_sha256=stage, paired_receipt_sha256=pair,
            status="MASK_RTDETR_DEV705_E24_AUTHOR_EXPORT_COMPLETE", protocol_commit=subject.PROTOCOL,
            GPU=True, DEV=True, GT_opened=False, official_TEST_opened=False, AP=False,
            rows=70500, images=705, candidates_per_image=100, device="gpu:0", CUDA="11.8",
            attention_calls={"author_paddle_grid_sample": 4230}, image_manifest_sha256=subject.DEV_MANIFEST_SHA)
                   for arm, ref in expected.items()}
        subject.verify_export_pair(reports, expected, stage, pair)
        reports["center"]["training_independent_receipt_sha256"] = reports["native"]["training_independent_receipt_sha256"]
        with self.assertRaisesRegex(ValueError, "Exact.*arm"):
            subject.verify_export_pair(reports, expected, stage, pair)


if __name__ == "__main__":
    unittest.main()
