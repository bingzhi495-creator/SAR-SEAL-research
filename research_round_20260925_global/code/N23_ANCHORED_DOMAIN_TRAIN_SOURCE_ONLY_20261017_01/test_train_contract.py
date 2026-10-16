"""Synthetic metadata only: fix the complete added stage and primary decision."""
import copy
import importlib.util
from pathlib import Path
import unittest


def load_body():
    p=Path(__file__).with_name('train_body.py')
    s=importlib.util.spec_from_file_location('n23_train_contract_unit',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture():
    pre=dict(arms=['A_COMMON','A_ACTIVE','B_C'],epochs=24,stage_windows=[[1,8],[9,16],[17,24]],
             anchor_points_per_GT=1024,anchor_loss_coefficient=1.,extension_weight=.25,
             updates_per_epoch_per_arm=1661,AdamW_lr=.0003,AdamW_weight_decay=.0001,
             AdamW_betas=[.9,.999],AdamW_eps=1e-8,training_seed=202610172401,
             main_AP_endpoint=24,selection='Omega',score='original',
             no_DEV_during_training=True,no_TEST=True,reference_recomputed_same_entry=True)
    return pre,dict(first_epoch=1,last_epoch=8,resume=None)


class TrainingContractTests(unittest.TestCase):
    def test_added_stage_and_original_anchor_are_fixed(self):
        m=load_body();p,s=fixture();saved=copy.deepcopy((p,s));m.validate_recipe(p,s)
        self.assertEqual((p,s),saved)

    def test_loss_budget_or_endpoint_cannot_silently_change(self):
        m=load_body()
        for key,value in [('epochs',8),('anchor_loss_coefficient',.5),('extension_weight',1.),
            ('anchor_points_per_GT',768),('updates_per_epoch_per_arm',1650),('AdamW_lr',.001),
            ('main_AP_endpoint',8),('selection','full_C'),('score','reranked'),
            ('reference_recomputed_same_entry',False),('no_DEV_during_training',False)]:
            p,s=fixture();p[key]=value
            with self.assertRaises(AssertionError):m.validate_recipe(p,s)

    def test_only_prelocked_stage_windows_and_explicit_resume(self):
        m=load_body()
        for start,end,resume in [(1,4,None),(1,24,None),(9,16,None),(1,8,{'checkpoint':1})]:
            p,s=fixture();s.update(first_epoch=start,last_epoch=end,resume=resume)
            with self.assertRaises(AssertionError):m.validate_recipe(p,s)
        for start,end in [(9,16),(17,24)]:
            p,s=fixture();s.update(first_epoch=start,last_epoch=end,resume={'checkpoint':1})
            m.validate_recipe(p,s)


if __name__=='__main__':unittest.main()
