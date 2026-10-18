"""Tiny prelock/context checks only. No tensors, ML imports, or actual paths opened."""
from copy import deepcopy
from collections import OrderedDict
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('_U52_BODY_TEST', Path(__file__).with_name('equivalence_body.py'))
body = importlib.util.module_from_spec(spec)
spec.loader.exec_module(body)


class MetadataTests(unittest.TestCase):
    def test_ordered_state_mapping_dispatch(self):
        # A type sentinel only: no fake tensor arithmetic or numerical restore.
        class TensorSentinel: pass
        class TypeNamespace: Tensor=TensorSentinel
        class NoTensorControl:
            def check(self):raise AssertionError('No tensor branch is exercised')
        original=OrderedDict([('z',OrderedDict([('later',2),('earlier',1)])),('a',[None,True])])
        result=body._cpu_tree(TypeNamespace,original,NoTensorControl())
        self.assertIs(type(result),dict)
        self.assertEqual(list(result),['z','a'])
        self.assertIs(type(result['z']),dict)
        self.assertEqual(list(result['z']),['later','earlier'])
        self.assertEqual(result,original)
        self.assertIsNot(result,original)

    def test_fixed_warm_control_and_carried_sequence(self):
        pre = body.equivalence_contract()
        body.validate_prelock(pre)
        self.assertEqual(pre['warmup'], {'dataset_index':54, 'G':1, 'step':1, 'implementation':'original'})
        self.assertEqual(pre['comparison_indices'], [710,54])
        self.assertEqual(pre['comparison_steps'], [2,3])
        self.assertEqual(pre['maximum_disposable_updates'], 14)
        for key, value in (('comparison_steps',[1,2]), ('baseline_control','approximate'),
                           ('main_training',True), ('maximum_disposable_updates',12)):
            bad=deepcopy(pre);bad[key]=value
            with self.assertRaises(ValueError):body.validate_prelock(bad)
        bad=deepcopy(pre);bad['tolerances']['gradients']['rtol']=1.
        with self.assertRaises(ValueError):body.validate_prelock(bad)

    def test_two_private_packages_without_hidden_inputs(self):
        ref=lambda path:{'path':path,'bytes':1,'sha256':'a'*64}
        ctx={'portable_refs':{'54':ref('/inputs/a.pt'),'710':ref('/inputs/b.pt')},'scratch_directory':'/outputs/u52'}
        body.validate_context_metadata(ctx)
        for bad in (dict(ctx, hidden_GT=ref('/gt.json')), dict(ctx,scratch_directory='/inputs'),
                    dict(ctx,portable_refs={'54':ctx['portable_refs']['54']})):
            with self.assertRaises(ValueError):body.validate_context_metadata(bad)

    def test_numerical_result_does_not_hide_baseline_failure(self):
        self.assertEqual(body.comparison_verdict(False,True),'BASELINE_CONTROL_MISMATCH')
        self.assertEqual(body.comparison_verdict(True,False),'CANDIDATE_OUTSIDE_PRELOCKED_TOLERANCES')
        self.assertEqual(body.comparison_verdict(True,True),'WITHIN_PRELOCKED_TOLERANCES')


if __name__=='__main__':unittest.main()
