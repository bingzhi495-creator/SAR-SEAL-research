"""Synthetic selector tests using the archived pure ranking helpers."""
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np

from transverse_quota import NEW_ARMS, select, within_width_cap
from render_transverse import OLD_ARMS, adapted, verify_old


def archive_writer():
    path=Path(os.environ["SAR_SEAL_VARIABLE_WRITER"])
    writer=ModuleType("test_archived_writer")
    exec(compile(path.read_bytes(),str(path),"exec"),writer.__dict__)
    return writer


def example():
    native=np.zeros((12,21),dtype=bool)
    native[5:7,5:15]=True
    add=[(4,8),(7,10),(5,15),(6,4)]
    drop=[(5,7),(6,11),(5,5),(6,14)]
    def flat(values):
        return np.ravel_multi_index(np.asarray(values).T,native.shape)
    return native,np.column_stack((flat(add),flat(drop)))


class TestTransverseQuota(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.writer=archive_writer()

    def test_original_arms_and_same_menu_width_and_budget(self):
        native,coords=example()
        posterior=np.array([[.90,.89],[.87,.86],[.58,.61],[.60,.62]])
        posterior=np.stack((posterior,np.full_like(posterior,.92)),axis=1)
        old=self.writer.select(posterior,((2,),(0,)),1,int(native.sum()))
        decisions,audit=select(self.writer,posterior,((2,),(0,)),1,native,coords)
        for arm in ("fixed_iou","variable_iou","near_iou","variable_raw_min","variable_mean_iou","menu_actions"):
            self.assertEqual(decisions[arm],old[arm])
        self.assertTrue(audit["eligible"])
        self.assertGreater(audit["feasible_quotas"],1)
        self.assertEqual(len(set(audit["admitted_counts"].values())),1)
        for arm in NEW_ARMS:
            self.assertIn(1,audit["admitted_quotas"][arm])
            self.assertEqual(sum(map(len,decisions[arm]["action"])),2)
        self.assertLessEqual(audit["width_by_arm"]["transverse_iou"],audit["width_by_arm"]["fixed_iou"])


    def test_selected_variances_are_centered_at_each_action_centroid(self):
        native,coords=example()
        rng=np.random.default_rng(839)
        posterior=rng.uniform(.02,.98,(len(coords),2,2))
        decisions,audit=select(self.writer,posterior,((1,2),(1,2)),2,native,coords)
        points=np.column_stack(native.nonzero())
        _,axes=np.linalg.eigh(np.cov(points.T,bias=True))
        self.assertTrue(audit["eligible"])
        for arm,measured in audit["width_by_arm"].items():
            adds,drops=decisions[arm]["action"]
            edited=native.copy()
            edited.flat[coords[list(adds),0]]=True
            edited.flat[coords[list(drops),1]]=False
            expected=np.var(np.column_stack(edited.nonzero())@axes,axis=0)
            self.assertAlmostEqual(measured,expected[0],places=13)
            self.assertAlmostEqual(audit["longitudinal_variance_by_arm"][arm],expected[1],places=13)
        self.assertTrue(within_width_cap(2.0+1e-12,2.0))
        self.assertFalse(within_width_cap(2.0+3e-12,2.0))

    def test_disconnected_or_round_native_masks_keep_original_free(self):
        native,coords=example()
        disconnected=native.copy()
        disconnected[9,14]=True
        square=np.zeros_like(native)
        square[5:8,5:8]=True
        square_add=[(4,5),(4,6),(4,7),(8,7)]
        square_drop=[(5,5),(5,6),(5,7),(7,7)]
        square_coords=np.column_stack([
            np.ravel_multi_index(np.asarray(points).T,square.shape)
            for points in (square_add,square_drop)
        ])
        p=np.random.default_rng(5).uniform(.05,.95,(4,2,2))
        for mask,indices,reason in (
            (disconnected,coords,"disconnected"), (square,square_coords,"aspect")
        ):
            decisions,audit=select(self.writer,p,((1,),(1,)),1,mask,indices)
            self.assertFalse(audit["eligible"])
            self.assertEqual(audit["disposition"],reason)
            for arm in NEW_ARMS:
                self.assertEqual(decisions[arm],decisions["variable_iou"])

    def test_only_feasible_balanced_allocation_keeps_every_arm_equal(self):
        native,coords=example()
        p=np.full((len(coords),2,2),.5)
        incumbent=(tuple(range(len(coords))),)*2
        decisions,audit=select(self.writer,p,incumbent,len(coords),native,coords)
        self.assertEqual(audit["feasible_quotas"],1)
        for arm in NEW_ARMS:
            self.assertEqual(decisions[arm],decisions["fixed_iou"])
            self.assertEqual(audit["admitted_quotas"][arm],[len(coords)])

    def test_native_one_pixel_wide_diagonal_is_eligible(self):
        native=np.zeros((14,14),dtype=bool)
        native[np.arange(4,10),np.arange(4,10)]=True
        adds=np.ravel_multi_index(([3,5,8,10],[4,6,9,9]),native.shape)
        drops=np.ravel_multi_index(([4,5,6,7],[4,5,6,7]),native.shape)
        p=np.full((4,2,2),.5)
        _,audit=select(self.writer,p,((0,),(0,)),1,native,np.column_stack((adds,drops)))
        self.assertTrue(audit["eligible"])

    def test_render_rejects_archived_mask_change_even_with_same_counts(self):
        previous={arm:{"rle":"old","counts":[1,1]} for arm in OLD_ARMS}
        original={"image_id":7,"rgb_sha256":"synthetic","edits":{
            "2":{"k":1,"coordinates":[[14,3]],"arms":previous}}}
        edited=dict(previous,**{arm:{"rle":"new","counts":[1,1]} for arm in NEW_ARMS})
        output=dict(original,edits={"2":dict(original["edits"]["2"],arms=edited)})
        verify_old(original,output)
        edited["fixed_iou"]={"rle":"different","counts":[1,1]}
        with self.assertRaises(ValueError):
            verify_old(original,output)

    def test_archive_source_seams_compile_without_loading_scientific_inputs(self):
        path=Path(os.environ["SAR_SEAL_ARCHIVED_TRANSFER_WORKER"])
        worker=adapted(path.read_text(encoding="utf-8"))
        code=compile(worker,str(path),"exec")
        self.assertIsNotNone(code)


if __name__=="__main__":
    unittest.main()

