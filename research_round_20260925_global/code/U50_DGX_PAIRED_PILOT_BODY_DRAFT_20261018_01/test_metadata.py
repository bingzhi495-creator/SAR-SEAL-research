"""Tiny recipe/order checks only; no model, tensor, data, or runtime mock."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).with_name('pilot_body.py')
spec=importlib.util.spec_from_file_location('_U50_METADATA',path)
body=importlib.util.module_from_spec(spec)
spec.loader.exec_module(body)


class PilotMetadata(unittest.TestCase):
    def test_exact_two_image_sequence_and_full_query_recipe(self):
        pre=body.pilot_contract()
        self.assertEqual(body.validate_prelock(pre),[54,710,54])
        for field,value in (('image_order',[54,54,710]),('query_chunk',4),('all_matched_GT',False)):
            bad=deepcopy(pre);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):body.validate_prelock(bad)

    def test_new_experiment_cannot_resume_old_runtime_or_change_optimizer(self):
        pre=body.pilot_contract()
        for section,field,value in ((None,'AP',True),(None,'AMP',True),(None,'compile',True),
            (None,'resume_U40_checkpoint',True),('optimizer','lr',1e-3),
            ('environment','torch','2.0.1+cu118')):
            bad=deepcopy(pre)
            if section is None:bad[field]=value
            else:bad[section][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):body.validate_prelock(bad)


if __name__=='__main__':unittest.main()
