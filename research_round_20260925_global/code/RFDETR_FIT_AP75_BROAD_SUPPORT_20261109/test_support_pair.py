from collections import Counter
from hashlib import sha256
import unittest

from support_pair import fold_map, rescue_case


class SupportPairTest(unittest.TestCase):
    def test_original_folds_preserved_and_added_folds_balanced(self):
        old={f"old-{i}" for i in range(288)}
        extra={f"new-{i}" for i in range(191)}
        groups=sorted(old,key=lambda s:sha256(("P175_AP75_GROUP_OOF_V1:"+s).encode()).digest())
        plan=fold_map(old,extra)
        self.assertEqual({k:plan[k] for k in old},{s:i%5 for i,s in enumerate(groups)})
        self.assertEqual(sorted(Counter(plan[k] for k in extra).values()),[38,38,38,38,39])
        with self.assertRaises(ValueError):
            fold_map(old,extra|{next(iter(old))})

    def test_rescue_availability_and_paired_hit_outcomes(self):
        self.assertIsNone(rescue_case(0,0,8,2,7))
        self.assertEqual(rescue_case(-22,1,7,6,7),(8,"unreachable","neither"))
        self.assertEqual(rescue_case(-1,1,6,4,5),(5,"choice_matters","large_only"))
        self.assertEqual(rescue_case(-1,5,7,5,7),(5,"all_actions_pass","both"))
        with self.assertRaises(ValueError):
            rescue_case(-1,1,5,2,6)


if __name__=="__main__":
    unittest.main()

