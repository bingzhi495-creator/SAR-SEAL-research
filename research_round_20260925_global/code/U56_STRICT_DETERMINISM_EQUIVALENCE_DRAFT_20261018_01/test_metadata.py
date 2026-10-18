"""Tiny strict-backend metadata guards only; no ML import or numerical evidence."""
from copy import deepcopy
import unittest
import equivalence_body as body


class StrictBackend(unittest.TestCase):
    def test_new_prelock_keeps_original_comparison_but_rejects_warn_or_workspace_fallback(self):
        pre=body.equivalence_contract()
        expected=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic_algorithms_enabled=True,
            deterministic_warn_only=False,cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False,
            cudnn_benchmark=False,cudnn_deterministic=True)
        self.assertEqual(pre['schema'],'U56-shared-encoder-equivalence-prelock-v1')
        self.assertEqual(pre['backend_policy'],expected)
        self.assertEqual(pre['baseline_control'],'bitexact_all_grad_model_Adam_BN_RNG')
        self.assertEqual(pre['maximum_disposable_updates'],14)
        body.validate_prelock(pre)
        for key,value in (('deterministic_warn_only',True),('CUBLAS_WORKSPACE_CONFIG',':16:8')):
            changed=deepcopy(pre);changed['backend_policy'][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):body.validate_prelock(changed)

    def test_caller_determinism_both_flags_must_be_restored_even_if_caller_used_warning_mode(self):
        import entry
        policy=body.strict_backend_policy()
        caller=dict(deterministic_algorithms_enabled=False,deterministic_warn_only=True)
        report=dict(CUBLAS_WORKSPACE_CONFIG_before_Torch_import=':4096:8',
            strict_backend_settings_before_body=deepcopy(policy),strict_backend_settings_after_body=deepcopy(policy),
            outside_determinism_before=deepcopy(caller),outside_determinism_after=deepcopy(caller),
            flags=dict(caller_determinism_controls_restored=True))
        entry.completed_strict_controls(report,policy)
        wrong=deepcopy(report);wrong['outside_determinism_after']['deterministic_warn_only']=False
        with self.assertRaises(ValueError):entry.completed_strict_controls(wrong,policy)
        wrong=deepcopy(report);wrong['strict_backend_settings_before_body']['deterministic_algorithms_enabled']=False
        with self.assertRaises(ValueError):entry.completed_strict_controls(wrong,policy)


if __name__=='__main__':unittest.main()
