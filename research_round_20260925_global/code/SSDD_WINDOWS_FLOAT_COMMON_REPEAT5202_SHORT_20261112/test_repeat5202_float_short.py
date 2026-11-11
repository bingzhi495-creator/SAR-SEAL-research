import unittest
from collections import namedtuple
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

bridge = SimpleNamespace(**run_path(str(Path(__file__).with_name("repeat5202_float_short.py"))))


class RepeatShortBindingTest(unittest.TestCase):
    def test_existing_init_called_before_downstream_seed_binding(self):
        calls=[]
        deps_type=namedtuple("Deps", "initialize_q100 other")
        def stock_init(*args):
            train=SimpleNamespace(seed=5201)
            calls.append(train.seed)
            return "model",train,SimpleNamespace(train_config=train),{"initialization_seed":5201}
        def repeat_initializer(stock):
            def wrapped(*args):
                model,train,module,witness=stock(*args)
                train.seed=5202
                return model,train,module,witness
            return wrapped
        admitted=("torch","source",SimpleNamespace(__name__="run_fit_arm"),deps_type(stock_init,"pinned"))
        programs=SimpleNamespace(worker=SimpleNamespace(run_fit_arm="repeat worker"))
        adapter=SimpleNamespace(repeat_initializer=repeat_initializer)
        torch,source,worker,deps=bridge.bind_repeat_runtime(admitted,programs,adapter)
        _,train,module,witness=deps.initialize_q100("fit","out","coco")
        self.assertEqual((torch,source,worker,deps.other),("torch","source","repeat worker","pinned"))
        self.assertEqual((calls,train.seed,module.train_config.seed,witness["initialization_seed"]),([5201],5202,5202,5201))

    def test_gate_accepts_only_prespecified_single_repeat(self):
        score={"seed":5201,"FIT":False,"DEV":True,"TEST":False,"AP":True,"first_seed_gate":{
            "status":"AT_MOST_ONE_PRELOCKED_REPLICATION","threshold_points":1.0,
            "paired_center_minus_native_AP_points":3.233436705382531,
            "controlling_margin_AP_points":0.22034536085387968,
            "center_minus_integer_trained_native_AP_points":0.22034536085387968}}
        bridge.check_first_seed_gate(score)
        for changed in (dict(score,TEST=True),dict(score,seed=5202),dict(score,first_seed_gate={})):
            with self.assertRaisesRegex(ValueError,"one prospectively"):
                bridge.check_first_seed_gate(changed)


if __name__ == "__main__":
    unittest.main()
