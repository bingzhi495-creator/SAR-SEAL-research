"""Synthetic COCO-shaped lists only; no real masks, NN, NumPy or COCO execution."""
import copy
import unittest

import geometry_diagnostic as diagnostic


def fixture():
    area=[0,1e10]
    row=dict(image_id=7,category_id=1,aRng=area,maxDet=100,
             gtIds=[101,102,103],gtIgnore=[0,1,0],gtMatches=[[10.,11.,0.],[0.,0.,12.]],
             dtIds=[10,11,12])
    annotations={10:dict(id=10,image_id=7,category_id=1,query_id=42),
                 11:dict(id=11,image_id=7,category_id=1,query_id=3),
                 12:dict(id=12,image_id=7,category_id=1,query_id=91)}
    return area,row,annotations


class NativePairMetadataTest(unittest.TestCase):
    def test_selects_only_first_threshold_nonignored_original_match(self):
        area,row,annotations=fixture()
        other=copy.deepcopy(row);other['aRng']=[0,1024]
        small_budget=copy.deepcopy(row);small_budget['maxDet']=10
        entries=[None,other,small_budget,row]
        before=copy.deepcopy(entries)
        actual=diagnostic.select_native_pairs(entries,annotations,area)
        self.assertEqual(actual,[dict(image_id=7,GT_annotation_id=101,dt_id=10,query_id=42)])
        self.assertEqual(entries,before)

    def test_rejects_duplicate_or_ambiguous_original_pair_metadata(self):
        area,row,annotations=fixture()
        cases=[]
        bad=copy.deepcopy(row);bad['gtIgnore']=[0,0,0];bad['gtMatches'][0]=[10.,10.,0.]
        cases.append(([bad],annotations))
        bad=copy.deepcopy(row);bad['gtIds']=[101,101,103]
        cases.append(([bad],annotations))
        bad=copy.deepcopy(row);bad['dtIds']=[11,12]
        cases.append(([bad],annotations))
        bad=copy.deepcopy(row);bad['gtIgnore']=[0]
        cases.append(([bad],annotations))
        bad=copy.deepcopy(row);bad['gtMatches'][0][0]=10.5
        cases.append(([bad],annotations))
        bad_annotations=copy.deepcopy(annotations);bad_annotations[10]['image_id']=99
        cases.append(([row],bad_annotations))
        bad_annotations=copy.deepcopy(annotations);bad_annotations[10]['query_id']=100
        cases.append(([row],bad_annotations))
        cases.append(([row,copy.deepcopy(row)],annotations))
        for entries,mapping in cases:
            with self.subTest(entries=entries):
                with self.assertRaises(ValueError):
                    diagnostic.select_native_pairs(entries,mapping,area)


if __name__=='__main__':
    unittest.main()
