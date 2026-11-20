"""Small exhaustive native-cell witnesses; no NumPy, Torch or input files."""
from itertools import combinations, product
import math
import unittest

from oracle_geometry import candidate_oracles


def exhaustive(stock, truth, edits, equal_area):
    original=[bool(v) for v in stock]
    choices=[]
    for selected in combinations(range(len(stock)),edits):
        adds=sum(not original[j] for j in selected)
        drops=len(selected)-adds
        if equal_area and adds!=drops:
            continue
        changed=original.copy()
        for j in selected:
            changed[j]=not changed[j]
        I=math.fsum(t for t,on in zip(truth,changed) if on)
        P=sum(changed)
        T=math.fsum(truth)
        J=I/(P+T-I)
        choices.append((J,selected,adds,drops,I,P))
    return min(choices,key=lambda r:(-r[0],r[1]))


class OracleTests(unittest.TestCase):
    def test_scorer_roundoff_is_preserved_and_invalid_values_rejected(self):
        for I in (-5e-9,32+5e-9):
            with self.subTest(raw_I=I):
                value=candidate_oracles([],[],I,32,32,0,add_indices=[],drop_indices=[])
                self.assertEqual(value['original']['I'],I)
                self.assertEqual(value['original']['iou'],I/(64-I))
                self.assertEqual(value['oracles']['O_U']['I'],I)
                self.assertEqual(value['losses']['O_F_minus_O_U'],0.)
        q=1+5e-13
        for add,drop in ((q,.5),(.5,q)):
            with self.subTest(add=add,drop=drop):
                value=candidate_oracles([add],[drop],2,32,5,2,add_indices=[33],drop_indices=[1])
                row=value['oracles']['O_U']
                self.assertEqual(row['added_intersection'],add)
                self.assertEqual(row['removed_intersection'],drop)
                self.assertEqual(row['I'],math.fsum((2,add,-drop)))
        for I in (-1.01e-8,32+1.01e-8):
            with self.subTest(invalid_I=I),self.assertRaises(ValueError):
                candidate_oracles([],[],I,32,32,0,add_indices=[],drop_indices=[])
        for q in (-1e-15,1+2e-12):
            for add,drop in ((q,.5),(.5,q)):
                with self.subTest(invalid_occupancy=q),self.assertRaises(ValueError):
                    candidate_oracles([add],[drop],2,32,5,2,add_indices=[33],drop_indices=[1])
        for I in (1e-9,2e-9):
            with self.subTest(nonpositive_original_union=I),self.assertRaisesRegex(ValueError,'positive union'):
                candidate_oracles([],[],I,0,1e-9,0,add_indices=[],drop_indices=[])

    def test_fixed_negative_gain_area_bias_and_unsupported_zero_budget(self):
        exact=candidate_oracles([0,0],[1,1],2,2,2,2,add_indices=[2,3],drop_indices=[0,1])
        self.assertEqual(exact['oracles']['O_U']['iou'],1/3)
        self.assertEqual(exact['oracles']['O_F']['iou'],.5)
        self.assertTrue(all(row['gain_over_original']<0 and row['edits']==2
                            for row in exact['oracles'].values()))
        oversized=candidate_oracles([0,0],[0,0],2,4,2,2,add_indices=[4,5],drop_indices=[2,3])
        undersized=candidate_oracles([1,1],[1,1],2,2,4,2,add_indices=[2,3],drop_indices=[0,1])
        for value,delta in ((oversized,-2),(undersized,2)):
            self.assertEqual(value['oracles']['O_U']['iou'],.5)
            self.assertEqual(value['oracles']['O_F']['iou'],1.)
            self.assertEqual(value['oracles']['O_F']['area_delta'],delta)
            self.assertEqual(value['losses']['O_F_minus_O_U'],.5)
        for add,indices in (([],[]),([.5],[5])):
            zero=candidate_oracles(add,[],1,2,3,0,add_indices=indices,drop_indices=[])
            self.assertTrue(zero['budget']['k_zero'])
            self.assertTrue(zero['budget']['balanced_support_unavailable'])
            self.assertEqual(zero['budget']['add_support_count'],len(add))
            self.assertEqual(zero['budget']['drop_support_count'],0)
            self.assertEqual(zero['losses']['O_F_minus_O_U'],0.)
            for row in zero['oracles'].values():
                self.assertEqual((row['I'],row['P'],row['T'],row['iou']),(1.,2.,3.,.25))
                self.assertEqual(row['flipped'],[])

    def test_native_index_ties_full_precision_and_invalid_support_rejected(self):
        first=candidate_oracles([0,0],[0,0],0,2,1,2,add_indices=[9,3],drop_indices=[7,1])
        shuffled=candidate_oracles([0,0],[0,0],0,2,1,2,add_indices=[3,9],drop_indices=[1,7])
        self.assertEqual(first,shuffled)
        self.assertEqual(first['oracles']['O_U']['flipped'],[1,3])
        self.assertEqual(first['oracles']['O_F']['flipped'],[1,3])
        # Adjacent unequal binary64 occupancies must not be collapsed by rounding.
        higher=math.nextafter(.5,1.)
        near=candidate_oracles([.5,higher],[0,0],0,2,2,2,add_indices=[1,9],drop_indices=[2,3])
        self.assertEqual(near['oracles']['O_U']['add_indices'],[9])
        base=dict(t_add=[.5],t_drop=[.5],I=.5,P=1,T=1,edits=2,add_indices=[1],drop_indices=[0])
        for patch in (dict(edits=1),dict(edits=4),dict(edits=True),dict(P=1.5),dict(T=0),
                      dict(t_add=[float('nan')]),dict(t_drop=[1.01]),dict(I=.1),
                      dict(add_indices=[0]),dict(add_indices=[True]),dict(t_add=[.5,.5],add_indices=[1,1])):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                candidate_oracles(**dict(base,**patch))

    def test_exhaustive_fractional_four_cell_masks_and_fixed_count(self):
        stock=[True,True,False,False]
        for truth in product((0.,.25,.5,1.),repeat=4):
            if not sum(truth):
                continue
            for edits in (0,2,4):
                value=candidate_oracles(truth[2:],truth[:2],sum(truth[:2]),2,sum(truth),edits,
                                        add_indices=[2,3],drop_indices=[0,1])
                for label,equal in (('O_U',True),('O_F',False)):
                    expected=exhaustive(stock,truth,edits,equal)
                    row=value['oracles'][label]
                    self.assertEqual(row['iou'],expected[0])
                    self.assertEqual(tuple(row['flipped']),expected[1])
                    self.assertEqual((row['adds'],row['drops'],row['I'],row['P']),expected[2:])
                    self.assertEqual(row['edits'],edits)
                self.assertGreaterEqual(value['losses']['O_F_minus_O_U'],0.)


if __name__=='__main__':
    unittest.main()
