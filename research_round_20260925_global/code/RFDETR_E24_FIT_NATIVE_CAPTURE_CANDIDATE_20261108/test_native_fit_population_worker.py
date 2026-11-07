from unittest import TestCase
from native_fit_population_worker import population_items

class PopulationContractTest(TestCase):
    def test_full_actual_scope_refuses_a_partial_roster(self):
        doc=dict(schema='RFDETR-E24-FIT741-image-only-manifest-v1',population=741,items=[{'dataset_index':0}])
        with self.assertRaises(ValueError):
            population_items(doc)
