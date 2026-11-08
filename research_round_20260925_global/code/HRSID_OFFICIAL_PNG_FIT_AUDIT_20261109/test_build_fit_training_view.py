from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_fit_training_view import training_view


class FitViewTest(unittest.TestCase):
    def setUp(self):
        self.source=dict(images=[{"id":1},{"id":2}],
                         annotations=[{"id":3,"image_id":1,"segmentation":[[1,2,3,4,5,6]]},
                                      {"id":4,"image_id":1},{"id":5,"image_id":2}],
                         categories=[{"id":1}])

    def test_retains_frozen_records_and_image_scope(self):
        result=training_view(self.source,{1},{4})
        self.assertEqual(result["images"],[self.source["images"][0]])
        self.assertEqual(result["annotations"],[self.source["annotations"][0]])
        self.assertEqual(len(self.source["annotations"]),3)

    def test_rejects_exclusion_outside_fit_or_empty_image(self):
        for excluded in ({5},{3,4}):
            with self.assertRaises(ValueError):
                training_view(self.source,{1},excluded)


if __name__=="__main__":
    unittest.main()

