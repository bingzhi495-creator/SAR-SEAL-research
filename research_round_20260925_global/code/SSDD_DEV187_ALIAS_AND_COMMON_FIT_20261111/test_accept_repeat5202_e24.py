import os
from pathlib import Path
import unittest

from accept_repeat5202_e24 import TRAINING_PROTOCOL, SOURCE_MANIFEST_SHA, SHORT_SHA, load_acceptance, verify_wait


class WaitTest(unittest.TestCase):
    def test_refuses_live_owner_and_unordered_child_starts(self):
        birth=dict(pid=21,creation_time=5.)
        request="1"*64
        summary="2"*64
        import json
        rows=[
            dict(status="P350_WINDOWS_REPEAT5202_E24_OWNER_RESERVED",owner_birth=birth,request_sha256=request),
            dict(status="P350_WINDOWS_REPEAT5202_E24_ORIGINAL_ARM_START",mode="native",request_sha256=request),
            dict(status="P350_WINDOWS_REPEAT5202_E24_ORIGINAL_ARM_START",mode="center",request_sha256=request),
            dict(status="P350_WINDOWS_REPEAT5202_E24_ORIGINALS_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE",
                 request_sha256=request,summary_sha256=summary)]
        wait=dict(status="P350_WINDOWS_REPEAT5202_E24_EXTERNAL_ORIGINAL_WAIT_V1",owner_birth=birth,
                  request_sha256=request,summary_sha256=summary,original_os_wait=True,original_os_exit=0,
                  combined_stdout="\n".join(json.dumps(row) for row in rows))
        verify_wait(wait,birth,request,summary,lambda _:False)
        with self.assertRaises(ValueError):
            verify_wait(wait,birth,request,summary,lambda _:True)
        wait["combined_stdout"]="\n".join(json.dumps(rows[i]) for i in (0,2,1,3))
        with self.assertRaises(ValueError):
            verify_wait(wait,birth,request,summary,lambda _:False)


class SourceBindingTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("P351_TEST_CODE"),"ordinary released source fixture required")
    def test_expected_phase_retains_distinct_init_and_training_seeds(self):
        api=load_acceptance(Path(os.environ["P351_TEST_CODE"]))
        result=api.expected_phase(dict(seed=5202), "native")
        self.assertEqual((result["seed"],result["initialization_seed"]),(5202,5201))
        import json
        raw=json.loads(Path(os.environ["P351_TEST_SHORT"]).read_text())
        witness=dict(online_tensor_sha256=raw["initial_online_sha256"],state_tensors=545,
                     num_queries=100,num_select=100,query_groups=13,training_queries_per_image=1300,
                     class_head_slots=2,initialization_seed=5201,coco_asset_sha256=raw["coco_asset_sha256"],
                     coco_asset_bytes=134545398,recipe_sha256=raw["recipe_sha256"])
        api.validate_initialization(witness)
        witness["initialization_seed"]=5202
        with self.assertRaises(ValueError):
            api.validate_initialization(witness)


if __name__=="__main__":
    unittest.main()
