"""SYNTHETIC_ONLY native mapping tuple/list regression; no actual artifacts."""
import importlib.util
from pathlib import Path
import unittest

root=Path(__file__).parent
path=root/'test_N16_ALL100_FIT_CACHE_CPU_QUALIFIER_SYNTHETIC_ONLY_20261014_02.py'
spec=importlib.util.spec_from_file_location('_n16_mapping_fixture',path)
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
source=root/'N16_ALL100_FIT_CACHE_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261014_03.py'
if not source.exists():
    source=root/'N16_ALL100_FIT_CACHE_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261014_02.py'
body_spec=importlib.util.spec_from_file_location('_n16_mapping_body',source)
body=importlib.util.module_from_spec(body_spec)
body_spec.loader.exec_module(body)

def validate(mapping):
    args=fixture.fixture()
    args[0]['raw_target_mapping']=mapping
    return body.validate_packet(fixture.torch,fixture.np,fixture.cv2,*args)

class MappingTests(unittest.TestCase):
    def test_native_tuple_and_valid_list_pass(self):
        for mapping in ((0,1),[0,1]):
            with self.subTest(mapping=mapping):
                self.assertTrue(validate(mapping)['labels_exact'])

    def test_nonpermutation_bool_float_wrong_length_and_other_types_reject(self):
        for mapping in ([False,1],(0,True),[0.,1],[0,0],(0,2),(0,),(),{0,1},'01'):
            with self.subTest(mapping=mapping):
                with self.assertRaisesRegex(RuntimeError,'raw-GT mapping/ann_ids required'):
                    validate(mapping)

if __name__=='__main__':
    unittest.main(verbosity=2)
