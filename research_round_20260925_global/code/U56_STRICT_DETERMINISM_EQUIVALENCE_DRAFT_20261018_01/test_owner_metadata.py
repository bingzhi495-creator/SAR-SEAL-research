"""One tiny stdlib child-environment and terminal contract test; no process/ML."""
import unittest
import linux_owner as owner


class OwnerMetadata(unittest.TestCase):
    def test_workspace_is_preimport_child_policy_and_success_requires_real_reap(self):
        outside={'CUBLAS_WORKSPACE_CONFIG':':16:8','KEEP':'unchanged'}
        actual=owner.child_environment(outside)
        self.assertEqual(actual['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        self.assertEqual(actual['CUDA_VISIBLE_DEVICES'],'0')
        self.assertEqual(actual['KEEP'],'unchanged')
        self.assertEqual(outside,{'CUBLAS_WORKSPACE_CONFIG':':16:8','KEEP':'unchanged'})
        args=[0,True,True,[],[],'COMPARISON_COMPLETED_PENDING_ROOT']
        self.assertTrue(owner.terminal_success(*args))
        for index,value in ((0,1),(0,False),(1,False),(2,False),(3,[{'pid':1}]),(4,['deadline']),(5,'FAILED')):
            bad=list(args);bad[index]=value
            self.assertFalse(owner.terminal_success(*bad))


if __name__=='__main__':unittest.main()
