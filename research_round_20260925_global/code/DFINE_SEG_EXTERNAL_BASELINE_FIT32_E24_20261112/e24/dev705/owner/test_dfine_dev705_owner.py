from pathlib import Path
import json
import subprocess
import sys
import unittest

from dfine_dev705_owner import clean_job, load_job, reservation_contract, run_job, validate_attestation


class JobPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = cls.scratch.resolve(strict=True)
        cls.job = load_job(cls.job_path.resolve(strict=True))

    def test_natural_exit_and_drain(self):
        cmd = [sys.executable,"-I","-B","-c","print('synthetic worker natural close')"]
        terminal = run_job(self.job,cmd,self.scratch/"natural",lambda:None,timeout=10.)
        self.assertTrue(clean_job(terminal))
        self.assertEqual(terminal["original_os_exit"],0)
        self.assertEqual(terminal["job_active_before_drain"],0)

    def test_leaked_descendant_gets_drained_and_rejected(self):
        spawned = "import subprocess,sys; subprocess.Popen([sys.executable,'-I','-B','-c','import time; time.sleep(30)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
        cmd = [sys.executable,"-I","-B","-c",spawned]
        terminal = run_job(self.job,cmd,self.scratch/"leaked",lambda:None,timeout=10.)
        self.assertFalse(clean_job(terminal))
        self.assertEqual(terminal["original_os_exit"],0)
        self.assertGreater(terminal["job_active_before_drain"],0)
        self.assertEqual(terminal["job_active_after_wait"],0)

    def test_reservation_requires_one_evaluator_and_no_training(self):
        row = dict(active_attempt="export", resource_closed=False, DEV=True, FIT=False, TEST=False, AP=False)
        state = dict(GPU_owner_or_worker_active=True, gpu_training_active=False, export=row)
        reservation_contract(state,"export")
        for invalid in (dict(state,gpu_training_active=True),dict(state,another={"active_attempt":"another"})):
            with self.assertRaisesRegex(ValueError,"exclusive evaluator reservation"):
                reservation_contract(invalid,"export")

    def test_outer_wait_must_bind_exact_child_and_result(self):
        from dfine_dev705_owner import CLOSURE_COMPLETE, EVAL_PROTOCOL, WORKER_SHA
        h = {k:str(i)*64 for i,k in enumerate(("closure","result","terminal","tool"),1)}
        result = dict(status="HRSID_DFINE_E24_DEV705_EXPORT_COMPLETE_V1",protocol_sha256=EVAL_PROTOCOL,
                      export_source_sha256=WORKER_SHA,FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False,images=705)
        terminal = dict(original_os_wait=True,original_os_exit=0,job_assigned=True,job_active_before_drain=0,
                        job_active_after_wait=0,stop_reason=None)
        closure = dict(status=CLOSURE_COMPLETE,protocol_sha256=EVAL_PROTOCOL,result_sha256=h["result"],
                       child_terminal_sha256=h["terminal"],resource_closed=True,worker_source_sha256=WORKER_SHA)
        event = dict(status=CLOSURE_COMPLETE,closure_sha256=h["closure"],result_sha256=h["result"])
        wait = dict(launch=dict(session_id=701),terminal=dict(exit_code=0,output=json.dumps(event)))
        result_ = validate_attestation(closure,terminal,wait,result,h)
        self.assertTrue(result_["job_drained"])
        for bogus in (dict(wait,terminal=dict(exit_code=1,output=wait["terminal"]["output"])),
                      dict(wait,terminal=dict(exit_code=0,output=wait["terminal"]["output"].replace(h["result"],"f"*64)))):
            with self.assertRaisesRegex(ValueError,"original outer tool wait"):
                validate_attestation(closure,terminal,bogus,result,h)


if __name__=="__main__":
    JobPreflight.scratch = Path(sys.argv[-1])
    JobPreflight.job_path = Path(sys.argv[-2])
    unittest.main(argv=sys.argv[:1])


