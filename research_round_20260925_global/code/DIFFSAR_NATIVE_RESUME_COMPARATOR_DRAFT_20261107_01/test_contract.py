"""Metadata behavior only: no Torch imports, checkpoint IO or fake tensor results."""
from copy import deepcopy
import unittest
from worker import validate_pair

def fixture():
    refs={k:dict(path='/private/'+k+'.pt',bytes=100,sha256='a'*64) for k in ('C4','C8','S8')}
    digests={k:'b'*64 for k in ('model','gradients','optimizer','scheduler','rng','loader')}
    def step(i):
        return dict(global_update=i,dataset_index=i,image_id=i+1,mapped_image_shape=[3,400,512],mapped_instances=1,
            augmentation_and_target_sha256=('%064x'%i),losses={'loss_masks':1.0},total_loss=1.0,
            used_learning_rates=[3.5e-5],post_scheduler_learning_rates=[3.5e-5])
    c=dict(worker_exit_code=0,all_requested_updates_completed=True,optimizer_steps_completed=8,new_optimizer_updates=8,
        steps=[step(i) for i in range(1,9)],checkpoints=[dict(updates=i,ref=refs['C'+str(i)],component_digests=digests) for i in (4,8)])
    s=dict(worker_exit_code=0,all_requested_updates_completed=True,optimizer_steps_completed=8,new_optimizer_updates=4,
        steps=[step(i) for i in range(5,9)],checkpoints=[dict(updates=8,ref=refs['S8'],component_digests=digests)],
        resume_start_exact=True,resume_start_component_digests=digests)
    return c,s,refs

class ContractTests(unittest.TestCase):
    def test_missing_exact_restored_start_is_rejected(self):
        c,s,refs=fixture();s.pop('resume_start_exact')
        with self.assertRaises(ValueError):validate_pair(c,s,refs)
    def test_changed_actual_input_digest_is_rejected(self):
        c,s,refs=fixture();self.assertTrue(validate_pair(c,s,refs))
        s['steps'][0]['augmentation_and_target_sha256']='f'*64
        with self.assertRaises(ValueError):validate_pair(c,s,refs)
    def test_unperformed_resume_updates_are_rejected(self):
        c,s,refs=fixture();s['new_optimizer_updates']=0
        with self.assertRaises(ValueError):validate_pair(c,s,refs)

if __name__=='__main__':unittest.main()
