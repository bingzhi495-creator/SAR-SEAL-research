"""COCO scoring must preserve the source polygons for continuous geometry."""
import unittest

from score_endpoint import official


class GeometryIsolationTest(unittest.TestCase):
    def test_original_polygon_survives_raster_scoring(self):
        polygon=[[0.,0.,2.,0.,2.,2.,0.,2.]]
        gt={"images":[{"id":1}],"categories":[{"id":1}],"annotations":[{"id":1,"image_id":1,"category_id":1,"segmentation":polygon}]}

        class Adapter:
            @staticmethod
            def evaluate_population(backend, g, rows, ids):
                import numpy as np
                evaluator=backend["COCOeval"](g,g,"segm")
                evaluator.eval={"precision":np.ones((10,1,1,1,1))}
                g.dataset["annotations"][0]["segmentation"]={"size":[4,4],"counts":"0"}
                rows[0]['bbox']=[0,0,2,2]
                return [1.] * 12

        rows=[{"image_id":1,"segmentation":{"size":[4,4],"counts":"0"}}]
        self.assertEqual(official(gt,rows,Adapter)["AP"],1.)
        self.assertNotIn("bbox",rows[0])
        self.assertEqual(gt["annotations"][0]["segmentation"],polygon)


if __name__=="__main__":
    unittest.main()
