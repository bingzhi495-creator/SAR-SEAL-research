from pathlib import Path
from runpy import run_path
from collections import Counter
import unittest

ROOT = Path(__file__).parent
P = run_path(str(ROOT/"policy.py"))


class SelectivePolicyTest(unittest.TestCase):
    def test_matched_adoption_counts_and_quota_strata(self):
        ranks=[2, 3, 5, 8, 9, 10]
        quotas=[4, 4, 8, 8, 8, 8]
        learned=P["choose"](ranks,quotas,[9.,8.,2.,.1,.2,.3])
        margin=P["choose"](ranks,quotas,[8.,9.,0.,1.,1.,1.])
        self.assertEqual(learned,{2,5})
        self.assertEqual(margin,{3,8})
        k=dict(zip(ranks,quotas,strict=True))
        self.assertEqual(Counter(k[r] for r in learned),Counter(k[r] for r in margin))

    def test_sparse_images_and_fixed_ties(self):
        self.assertEqual(P["choose"]([],[],[]),set())
        self.assertEqual(P["choose"]([1,2],[8,8],[100.,200.]),set())
        self.assertEqual(P["choose"]([8,3,1],[2,8,3],[1.,50.,60.]),{8})
        self.assertEqual(P["choose"]([8,3,1],[8,8,8],[1.,1.,1.]),{1})

    def test_reference_guards_are_required_for_each_primary_anchor(self):
        def effect(iou=.16,limit=.01,ap=-.05):
            return dict(iou_points=iou,percentile95_points=[limit,.4],ap_points=ap)
        names=["bce_robust","margin_ranked","bce_mean","static"]
        refs={k:{"selective_minus":dict((arm,effect()) for arm in names)}
              for k in ("pseg","spsd")}
        exposure={"bce_robust":(20,8),"margin_ranked":(20,8)}
        self.assertTrue(P["gate"](refs,exposure)["advances"])
        refs["spsd"]["selective_minus"]["margin_ranked"]=effect(limit=-.01)
        self.assertFalse(P["gate"](refs,exposure)["advances"])


if __name__=="__main__":
    unittest.main()
