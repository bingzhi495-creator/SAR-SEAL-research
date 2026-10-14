"""SYNTHETIC_ONLY: literal closed parser ABI and training Q interface."""
import copy
import importlib.util
from pathlib import Path
import unittest

root=Path(__file__).parent
path=root/'N16_ALL100_FIT_CACHE_CPU_ROOT_UNBOUND_DRAFT_20261014_05.py'
if not path.exists():
    path=root/'N16_ALL100_FIT_CACHE_CPU_ROOT_UNBOUND_DRAFT_20261014_04.py'
spec=importlib.util.spec_from_file_location('_n16_mapping_q_contract',path)
wrapper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)

def packet():
    return dict(raw_target_mapping=(0,1),raw_GT_ids=[101,102],
        identity=dict(ann_ids=[101,102],parser_rows=[
            dict(annotation_id=101,native_parser_retained=True),
            dict(annotation_id=999,native_parser_retained=False),
            dict(annotation_id=102,native_parser_retained=True)]),
        processed_target_proof=dict(ann_ids=[101,102],complete_all_raw_annotations=True),
        actual_raw_processed_GT_correspondence=dict(calls=1,correspondence_validator_passed=True))

class WrapperContractTests(unittest.TestCase):
    def test_canonical_parser_tuple_and_list_mapping_pass(self):
        self.assertNotIn('raw_target_mapping',wrapper.EXTERNAL_FIELDS)
        self.assertEqual(len(wrapper.EXTERNAL_FIELDS),5)
        for mapping in ((0,1),[0,1]):
            value=packet()
            value['raw_target_mapping']=mapping
            self.assertEqual(wrapper.validate_raw_target_mapping(value,[101,102]),(0,1))

    def test_parser_processed_canonical_or_correspondence_mismatch_rejects(self):
        changes=[lambda p:p['identity']['parser_rows'][0].update(annotation_id=102),
            lambda p:p['processed_target_proof'].update(ann_ids=[102,101]),
            lambda p:p['processed_target_proof'].update(complete_all_raw_annotations=False),
            lambda p:p['actual_raw_processed_GT_correspondence'].update(calls=2),
            lambda p:p['actual_raw_processed_GT_correspondence'].update(calls=True),
            lambda p:p['actual_raw_processed_GT_correspondence'].update(correspondence_validator_passed=False),
            lambda p:p.update(raw_target_mapping=[False,1]),lambda p:p.update(raw_target_mapping=[1,0])]
        for change in changes:
            value=packet()
            change(value)
            with self.assertRaises(RuntimeError):
                wrapper.validate_raw_target_mapping(value,[101,102])
        with self.assertRaises(RuntimeError):
            wrapper.validate_raw_target_mapping(packet(),[102,101])

    def test_nonempty_all_true_training_checks_do_not_claim_actual_terminal(self):
        q=dict(qualified=False,ROOT_actual_CPU0_verified=False)
        for key in ('all100_C_K_private_sampling_importance_exact','actual_rawGT_targets_both_label_classes_exact',
            'two_same_actual_GPU_input_PR_formulas_absolute1e4_strictsign','ROOT_external_proofs_checked','original_PR8_unchanged'):
            q[key]=True
        q['checks']=wrapper.qualification_checks(q)
        self.assertTrue(q['checks'] and all(v is True for v in q['checks'].values()))
        self.assertFalse(q['qualified'])
        self.assertFalse(q['ROOT_actual_CPU0_verified'])
        for bad in (False,1):
            value=copy.deepcopy(q)
            value['original_PR8_unchanged']=bad
            with self.assertRaises(RuntimeError):
                wrapper.qualification_checks(value)

if __name__=='__main__':
    unittest.main(verbosity=2)
