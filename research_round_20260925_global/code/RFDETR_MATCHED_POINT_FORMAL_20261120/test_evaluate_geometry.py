import copy
import unittest
import numpy as np
from evaluate_geometry import candidate_geometry, strict_transition


def record(image_id=1, group='a', intersection=6., unsupported=False):
    return dict(image_id=image_id,rank=7,query_id=99,group=group,role='valid',matched_gt_id=image_id+100,
                raw_continuous=dict(predicted_area=64,truth_area=8.,intersection=intersection),
                coordinates=np.empty((0,2),dtype=np.int32) if unsupported else np.array([[100,1],[101,2]],dtype=np.int32),
                truth=np.empty((0,2),dtype=np.float64) if unsupported else np.array([[.75,.25],[.5,.1]],dtype=np.float64))


def edit(indices=(),adds=0,drops=0):
    return dict(flipped=np.array(indices,dtype=np.int32),edits=len(indices),adds=adds,drops=drops,area_delta=adds-drops)


class GeometryTests(unittest.TestCase):
    def test_locked_gate_requires_lw_and_preserves_undefined(self):
        from evaluate_geometry import locked_gate, ARMS, SEEDS
        aps={'N':.50,'SF':.505,'SU=F':.507,'SU':.51,'LF=S':.506,
             'LU=S':.512,'LW=S':.506,'NP=S':.502}
        coco={seed:{arm:dict(AP=aps[arm],AP75=.6) for arm in ARMS} for seed in SEEDS}
        geometry=dict(seeds=list(SEEDS),arms=list(ARMS),bootstrap_draws=4096,bootstrap_seed=5217,
                      source_groups=2,comparisons={'SU_minus_'+arm:dict(delta_points=.3,interval95_points=[.1,.5])
                                                 for arm in ARMS if arm!='SU'})
        result=locked_gate(coco,geometry)
        self.assertTrue(result['advance'])
        self.assertLess(result['AP_contrasts_points']['SU_minus_LU=S'],0)
        bad=copy.deepcopy(coco)
        for seed in SEEDS:
            bad[seed]['LW=S']['AP']=.509
        failed=locked_gate(bad,geometry)
        self.assertFalse(failed['advance'])
        self.assertFalse(failed['gates']['AP_vs_LW=S'])
        self.assertTrue(failed['gates']['AP_vs_LF=S'])
        bad=copy.deepcopy(coco)
        bad[5217]['N']['AP']=-1
        undefined=locked_gate(bad,geometry)
        self.assertFalse(undefined['advance'])
        self.assertIsNone(undefined['mean_COCO']['N']['AP'])
        self.assertIsNone(undefined['AP_contrasts_points']['SU_minus_N'])
        failed_geometry=copy.deepcopy(geometry)
        failed_geometry['comparisons']['SU_minus_LW=S']['interval95_points'][0]=0
        self.assertFalse(locked_gate(coco,failed_geometry)['gates']['geometry_vs_LW=S'])

    def test_three_seed_source_equal_bootstrap_keeps_unsupported_denominator(self):
        from evaluate_geometry import evaluate_geometry, ARMS, SEEDS
        rows=[record(1,'a'),record(2,'a',unsupported=True),record(3,'b')]
        rows[2]['truth']=np.array([[.25,.75],[.5,.1]],dtype=np.float64)
        by_seed={}
        for si,seed in enumerate(SEEDS):
            by_seed[seed]={}
            for ri,row in enumerate(rows):
                chosen=(ri==0 and si<2) or (ri==2 and si==2)
                same=edit([100,1],1,1) if chosen else edit()
                by_seed[seed][(row['image_id'],row['rank'],row['query_id'])]={
                    arm:copy.deepcopy(same if arm in ('SU','LF=S','LU=S','LW=S','NP=S') else edit())
                    for arm in ARMS}
        result=evaluate_geometry(rows,by_seed)
        summary=result['summary']
        self.assertEqual(summary['counts']['valid'],3)
        self.assertEqual(summary['counts']['valid_unsupported_noop'],1)
        self.assertEqual(summary['source_groups'],2)
        ga=((6.5/65.5-6/66)*2/3)/2*100
        gb=((5.5/66.5-6/66)/3)*100
        effect=summary['comparisons']['SU_minus_N']
        self.assertAlmostEqual(effect['delta_points'],(ga+gb)/2)
        self.assertTrue(np.allclose(effect['interval95_points'],[min(ga,gb),max(ga,gb)],atol=1e-12))
        self.assertEqual(summary['comparisons']['SU_minus_LW=S']['delta_points'],0.)
        self.assertEqual(summary['bootstrap_draws'],4096)
        self.assertEqual(summary['bootstrap_seed'],5217)
        self.assertEqual(len(result['private_witness']),9)
        for seed in SEEDS:
            transition=summary['per_seed'][seed]['strict90']['SU']
            self.assertEqual(sum(transition.values()),3)
            self.assertEqual(transition['stable_miss'],3)
        bad=copy.deepcopy(by_seed)
        del bad[SEEDS[0]][(1,7,99)]['LW=S']
        with self.assertRaises(ValueError):
            evaluate_geometry(rows,bad)

    def test_fractional_intersection_and_changing_area_and_strict_ties(self):
        row=record()
        old=copy.deepcopy(row)
        swap=candidate_geometry(row,edit([100,1],1,1))
        self.assertEqual((swap['I'],swap['P'],swap['T']),(6.5,64.,8.))
        self.assertAlmostEqual(swap['iou'],6.5/(64+8-6.5))
        lw=candidate_geometry(row,edit([100,101],2,0))
        self.assertEqual((lw['I'],lw['P'],lw['T']),(7.25,66.,8.))
        self.assertAlmostEqual(lw['iou'],7.25/(66+8-7.25))
        self.assertEqual(row['raw_continuous'],old['raw_continuous'])
        self.assertTrue(np.array_equal(row['truth'],old['truth']))
        self.assertEqual(strict_transition(.75,.7500001,1,.75),'new')
        self.assertEqual(strict_transition(.91,.9,2,.9),'lost')
        self.assertEqual(strict_transition(.91,.91,0,.9),'clean_retained')
        self.assertEqual(strict_transition(.91,.91,2,.9),'edited_retained')
        self.assertEqual(strict_transition(.9,.9,0,.9),'stable_miss')
        with self.assertRaises(ValueError):
            candidate_geometry(row,edit([100,100],2,0))


if __name__ == '__main__':
    unittest.main()
