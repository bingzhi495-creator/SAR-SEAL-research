from unittest import TestCase
from fit_selector_materialize import validate_gt_population

class FitAnnotationBoundaryTests(TestCase):
    def test_annotation_image_outside_exact_fit_roster_is_rejected(self):
        rows=[dict(image_id=3,original_hw=[4,5])]
        fit=dict(images=[dict(id=3,height=4,width=5)],categories=[dict(id=1)],
                 annotations=[dict(id=9,image_id=3,category_id=1)])
        validate_gt_population(fit, rows)
        with self.assertRaises(ValueError):
            validate_gt_population(dict(fit,annotations=[dict(id=9,image_id=8,category_id=1)]),rows)
