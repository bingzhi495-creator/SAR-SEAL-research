"""One tiny stdlib terminal/environment contract test, never a process or NN run."""
import unittest
import linux_owner as owner


class TrainingOwnerMetadata(unittest.TestCase):
    def test_only_complete_paired_training_and_real_reap_can_complete(self):
        env=owner.child_environment({'CUDA_VISIBLE_DEVICES':'','KEEP':'value'})
        self.assertEqual((env['CUDA_VISIBLE_DEVICES'],env['CUBLAS_WORKSPACE_CONFIG']),('0',':4096:8'))
        self.assertEqual(env['KEEP'],'value')
        args=[0,True,True,[],[],'SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE']
        self.assertTrue(owner.terminal_success(*args))
        for index,value in ((0,1),(0,False),(1,False),(2,False),(3,[{'pid':1}]),(4,['RSS']),(5,'FAILED')):
            bad=list(args);bad[index]=value
            self.assertFalse(owner.terminal_success(*bad))


if __name__=='__main__':unittest.main()
