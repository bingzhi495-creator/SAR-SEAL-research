import unittest

import launch_center_after_native as center


class SerialCenterGateTest(unittest.TestCase):
    def inputs(self):
        phase = dict(stage="e24",mask_resize_mode="native",seed=5201,initialization_seed=5201,
                     training_protocol=center.PROTOCOL,source_manifest_sha256=center.MANIFEST)
        arm = dict(status=center.ARM_STATUS,qualification_pair_sha256=center.SHORT_PAIR,
                   completed_epochs=24,visits=17856,optimizer_updates=4464,ema_updates=4464,
                   original_os_wait=True,original_os_exit=0,resource_stop=False,worker_closed=True,
                   phase=phase,FIT=True,DEV=False,TEST=False,AP=False,terminal_sha256="a"*64,launch_sha256="b"*64)
        authority=dict(mode="native",resource_closed=True,active_attempt=None,failure=None,
                       terminal_ref={"sha256":"a"*64},launch_ref={"sha256":"b"*64})
        return arm,authority

    def test_accepts_exact_closed_native_and_rejects_active_or_other_launch(self):
        arm,authority=self.inputs()
        center.verify_independent_native(arm,authority)
        for changed in (dict(authority,active_attempt="native"),dict(authority,terminal_ref={"sha256":"c"*64})):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError,"terminal"):
                    center.verify_independent_native(arm,changed)

    def test_rejects_wrong_qualification_or_truncated_arm(self):
        arm,authority=self.inputs()
        for changed in (dict(arm,qualification_pair_sha256="d"*64),dict(arm,visits=32)):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError,"receipt"):
                    center.verify_independent_native(changed,authority)


if __name__=="__main__":
    unittest.main()
