"""Tiny metadata only; no complete order fixtures, ML, checkpoint or training execution."""
from copy import deepcopy
import unittest
import training_body as body


class MetadataContracts(unittest.TestCase):
    def test_python38_producer_is_explicit_and_does_not_equal_consumer_version(self):
        value=dict(python_implementation='CPython',python_version='3.8.17',python_major_minor=[3,8])
        body.validate_order_producer(value)
        for key,replacement in (('python_version','3.12.14'),('python_major_minor',[3,12]),('python_implementation','PyPy')):
            wrong=deepcopy(value);wrong[key]=replacement
            with self.subTest(key=key):
                with self.assertRaises(ValueError):body.validate_order_producer(wrong)

    def test_smoke_is_exactly_two_continuations_and_cannot_be_formal_budget(self):
        smoke=body.training_contract('SMOKE','one-smoke')
        self.assertEqual(smoke['smoke_indices'],[54,710])
        self.assertEqual(body.validate_span(dict(start=0,stop=1),smoke),(0,1))
        self.assertEqual(body.validate_span(dict(start=1,stop=2),smoke),(1,2))
        for span in (dict(start=0,stop=2),dict(start=2,stop=3),dict(start=True,stop=2)):
            with self.assertRaises(ValueError):body.validate_span(span,smoke)
        formal=body.training_contract('FORMAL','independent-formal')
        self.assertIsNone(formal['smoke_indices'])
        self.assertEqual(formal['recipe']['updates'],37050)
        self.assertIs(formal['backend_policy']['deterministic_algorithms_enabled'],False)

    def test_parent_requires_same_new_dgx_profile_and_all_paired_refs(self):
        pre=body.training_contract('SMOKE','one-smoke')
        parent=dict(pair_commit_ref='pair',checkpoint_refs=dict(PLAIN='plain',FULL='full'),
            trace_refs=dict(PLAIN='plain-trace',FULL='full-trace'))
        value=dict(schema='Root-U57-DGX-original-update-training-accepted-v1',root_accepted=True,
            paired_prefix_advanced=True,mode='SMOKE',run_id='one-smoke',result=dict(parent,
                order_ref='order',paired_checkpoint_prefix=body.cursor(1)))
        body.validate_parent_accepted(value,parent,pre,1,'order')
        for key,replacement in (('schema','ROOT-U40-continuous-training-segment-qualified-v1'),
                                ('mode','FORMAL'),('run_id','other'),('root_accepted',False)):
            wrong=deepcopy(value);wrong[key]=replacement
            with self.subTest(key=key):
                with self.assertRaises(ValueError):body.validate_parent_accepted(wrong,parent,pre,1,'order')


if __name__=='__main__':unittest.main()
