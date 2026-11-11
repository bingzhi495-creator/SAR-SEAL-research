import unittest
from srnet_author_batch4_train import VisitAudit


class VisitAuditTest(unittest.TestCase):
    def test_fallback_is_audited_against_consumed_batch(self):
        audit=VisitAudit(set(range(5)),batches_per_epoch=2)
        for key,ok in [(0,True),(1,False),(4,True),(2,True),(3,True)]:
            audit.attempt(key,ok)
        first=audit.consume([0,4,2,3])
        self.assertEqual(first["scheduled_ids"],[0,1,2,3])
        self.assertEqual(first["consumed_ids"],[0,4,2,3])
        self.assertEqual(first["rejected_attempt_ids"],[1])
        for key in [4,4,3,2]:
            audit.attempt(key,True)
        audit.consume([4,4,3,2])
        row=audit.close_epoch()
        self.assertEqual((row["scheduled_visits"],row["consumed_visits"],row["scheduled_unique"],
                          row["scheduled_padding_visits"],row["consumed_unique"],row["rejected_attempts"]),
                         (8,8,5,3,4,1))

    def test_consumed_identity_mismatch_fails(self):
        audit=VisitAudit(set(range(5)),batches_per_epoch=2)
        for key in range(4):
            audit.attempt(key,True)
        with self.assertRaises(ValueError):
            audit.consume([1,0,2,3])


if __name__=="__main__":
    unittest.main()
