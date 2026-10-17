"""Invented endpoint metadata only; no checkpoint deserialization or NN probe."""
import copy
import unittest
import eval_body


def fixture():
    ref=dict(path='E:/invented/E24.pth',bytes=0,sha256='a'*64)
    source=dict(path='E:/invented/factory.py',bytes=0,sha256='b'*64)
    spec=dict(seed=202610172801,checkpoint=ref,factory_source=source,
              train_core_source=source,training_reader_source=source)
    cp=dict(schema='N28-dense-routing-paired-train-state-v1',seed=spec['seed'],completed_epoch=24,
            modes=['SELECTIVE','WHOLE_NODE'],factory_source=source,train_core_source=source,
            reader_source=source,cumulative_updates_per_arm=17784,history=[{'epoch':i} for i in range(1,25)],
            models={'SELECTIVE':{},'WHOLE_NODE':{}})
    accepted=dict(schema='ROOT-N28-dense-routing-paired-train-qualified-v1',qualified=True,
                  seed=spec['seed'],end_epoch=24,full_24_epochs_complete=True,history=copy.deepcopy(cp['history']),
                  factory_source=source,train_core_source=source,reader_source=source,
                  result=dict(checkpoint_ref=ref,seed=spec['seed'],end_epoch=24,cumulative_updates_per_arm=17784,
                              new_training=True,AP=False,DEV=False,TEST=False))
    return spec,accepted,cp


class FixedEndpointTests(unittest.TestCase):
    def test_only_requested_seed_original_fixed24_is_eligible(self):
        spec,accepted,cp=fixture()
        self.assertEqual(eval_body.validate_endpoint(spec,accepted,cp),202610172801)
        for changed,key,value in ((cp,'completed_epoch',8),(cp,'seed',202610172802),
                                  (accepted,'qualified',False),(accepted,'full_24_epochs_complete',False),
                                  (cp,'history',[])):
            bad=copy.deepcopy(changed);bad[key]=value
            with self.assertRaises(RuntimeError):
                eval_body.validate_endpoint(spec,bad if changed is accepted else accepted,
                                             bad if changed is cp else cp)


if __name__=='__main__':unittest.main()
