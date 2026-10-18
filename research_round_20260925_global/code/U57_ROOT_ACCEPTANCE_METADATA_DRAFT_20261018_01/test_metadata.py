"""Tiny pure dictionaries only; no actual terminal, files, model or qualification."""
import unittest
import acceptance


class MetadataAcceptance(unittest.TestCase):
    def test_formal_credit_is_delta_per_arm_and_aggregate_not_cursor(self):
        self.assertEqual(acceptance.budget_credit('FORMAL',7,9),
                         {'updates_per_arm':2,'aggregate_optimizer_updates':4,
                          'formal_credit_per_arm':2,'formal_credit_aggregate':4})
        self.assertEqual(acceptance.budget_credit('SMOKE',1,2)['formal_credit_aggregate'],0)
        for args in (('FORMAL',False,2),('SMOKE',0,2),('FORMAL',9,7)):
            with self.assertRaises(ValueError):acceptance.budget_credit(*args)

    def test_root_terminal_needs_actual_zero_reap_both_births_and_empty_group(self):
        terminal=dict(root_observed_tool_exit_code=0,SSH_returncode=0,worker_returncode=0,
            reaped=True,owner_original_birth_absent=True,child_original_birth_absent=True,
            remaining_owned_group=[],all_owned_processes_absent=True)
        acceptance.validate_terminal(terminal)
        for key,value in (('SSH_returncode',False),('worker_returncode',1),
                          ('owner_original_birth_absent',False),('remaining_owned_group',[{'pid':1}])):
            bad=dict(terminal);bad[key]=value
            with self.assertRaises(ValueError):acceptance.validate_terminal(bad)

    def test_missing_root_byte_verification_never_creates_acceptance(self):
        checks={'source_and_logs_byte_verified':True,'metadata_records_byte_verified':True,
                'outputs_byte_verified':True}
        acceptance.validate_byte_evidence(checks)
        for key in checks:
            bad=dict(checks);bad.pop(key)
            with self.assertRaises(ValueError):acceptance.validate_byte_evidence(bad)
        with self.assertRaises(ValueError):acceptance.accept_training({}, {}, {}, {}, {})


if __name__=='__main__':unittest.main()
