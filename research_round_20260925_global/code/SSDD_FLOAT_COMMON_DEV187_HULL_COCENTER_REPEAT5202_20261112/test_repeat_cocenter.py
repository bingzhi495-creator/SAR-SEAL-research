import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask
from pycocotools.cocoeval import COCOeval

from repeat_cocenter import stock_matches


class ReplicateMatchTest(unittest.TestCase):
    def test_stock_match_preserves_order_and_skips_ap(self):
        raster = np.zeros((8,8),dtype=np.uint8,order="F")
        raster[1:3,1:3] = 1
        rle = mask.encode(raster)
        rle["counts"] = rle["counts"].decode("ascii")
        gt = dict(info={},images=[dict(id=1,height=8,width=8)],
                  categories=[dict(id=1,name="ship")],
                  annotations=[dict(id=9,image_id=1,category_id=1,segmentation=rle,
                                    area=4,bbox=[1,1,2,2],iscrowd=0)])
        preds = [dict(image_id=1,category_id=1,score=.8,segmentation=rle,bbox=[0,0,8,8]),
                 dict(image_id=1,category_id=1,score=.8,segmentation=rle,bbox=[1,1,2,2])]
        def resolve(records, rows, identities):
            self.assertIs(rows,preds)
            self.assertEqual(identities,{9:1})
            cell, = records
            self.assertEqual(cell["dtIds"],[1,2])
            self.assertEqual(cell["dtMatches"][0].tolist(),[9,0])
            return {9:rows[0]["segmentation"]}
        with patch.object(COCOeval,"accumulate",side_effect=AssertionError("No AP")):
            self.assertEqual(stock_matches(gt,preds,resolve),{9:rle})


if __name__=="__main__":
    unittest.main()

