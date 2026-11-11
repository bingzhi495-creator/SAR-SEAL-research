import unittest
from full_fit_crop_paste import summarize

ARMS=("interior_full","center_full","interior_contracted","center_contracted")

class SummaryTest(unittest.TestCase):
    def test_empty_masks_and_paired_denominator_are_preserved(self):
        one=dict(iou=1.,strict_075=True,strict_090=True,intersection=2,union=2,fp=0,fn=0)
        none=dict(iou=None,strict_075=False,strict_090=False,intersection=0,union=0,fp=0,fn=0)
        bad=dict(iou=0.,strict_075=False,strict_090=False,intersection=0,union=2,fp=0,fn=2)
        rows=[dict(metrics={a:one.copy() for a in ARMS},truth_area=2),
              dict(metrics={a:none.copy() for a in ARMS},truth_area=0),
              dict(metrics={a:(one.copy() if a=="center_full" else bad.copy()) for a in ARMS},truth_area=2)]
        result=summarize(rows)
        self.assertEqual(result["targets"],3)
        self.assertEqual(result["empty_truth"],1)
        self.assertEqual(result["arms"]["interior_full"]["valid_iou"],2)
        self.assertEqual(result["arms"]["interior_full"]["empty_prediction"],2)
        self.assertEqual(result["arms"]["interior_full"]["mean_iou"],.5)
        paired=result["paired_differences"]["center_full_minus_interior_full"]
        self.assertEqual((paired["valid_pairs"],paired["undefined_pairs"],paired["wins"],paired["ties"],paired["losses"]),(2,1,1,1,0))
        self.assertEqual(paired["mean"],.5)



class ReplayTest(unittest.TestCase):
    def test_prior_replay_rejects_changed_arm_and_boundary_bins_are_fixed(self):
        from full_fit_crop_paste import group_names, prior_replay
        metric=dict(iou=1.,strict_075=True,strict_090=True,intersection=2,union=2,fp=0,fn=0)
        rows=[dict(metrics={a:metric.copy() for a in ARMS},truth_area=2)]
        summarized=summarize(rows)
        prior=dict(arms={a:{k:summarized["arms"][a][k] for k in ("mean_iou","at_75","at_90")} for a in ARMS},
                   paired_differences={k:{j:v[j] for j in ("mean","wins","ties","losses")}
                                       for k,v in summarized["paired_differences"].items()
                                       if k in ("center_full_minus_interior_full","interior_contracted_minus_interior_full",
                                                "center_full_minus_interior_contracted","center_full_minus_center_contracted")},
                   max_abs_coordinate_error=0.)
        self.assertTrue(prior_replay(rows,prior,0.))
        prior["arms"]["center_full"]["mean_iou"]=.9
        with self.assertRaises(ValueError):
            prior_replay(rows,prior,0.)
        self.assertIn("size_mid",group_names(1024.,[0.,0.,9.,3.,0.],False,False,False))
        self.assertIn("size_large",group_names(9216.,[0.,0.,2.,2.,0.],True,True,False))
        self.assertIn("thin",group_names(1023.,[0.,0.,9.,3.,0.],False,False,False))

if __name__=="__main__":
    unittest.main()
