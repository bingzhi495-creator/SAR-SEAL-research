"""One tiny standard-library terminal-acceptance test; no process/file/ML execution."""
import unittest
import linux_owner as owner


class TerminalMetadata(unittest.TestCase):
    def test_actual_reap_and_complete_CPU_report_are_all_required(self):
        args=[0,True,True,[],[],'COMPLETE_PENDING_ROOT']
        self.assertTrue(owner.terminal_success(*args))
        for index,value in ((0,1),(0,False),(1,False),(2,False),(3,[{'pid':1}]),(4,['deadline']),(5,'FAILED')):
            bad=list(args);bad[index]=value
            self.assertFalse(owner.terminal_success(*bad))


if __name__=='__main__':unittest.main()
