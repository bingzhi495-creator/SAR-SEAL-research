"""Small injected metadata behavior only; no fake NN or numerical qualification."""
import importlib.util
from pathlib import Path
import types
import unittest


def api():
    spec=importlib.util.spec_from_file_location('U47_METADATA_ONLY',str(Path(__file__).with_name('teacher_restore.py')))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Poison:
    def __getattribute__(self,name):raise AssertionError('Image/GT/cache must not be accessed by construction metadata')


class Metadata(unittest.TestCase):
    def test_original_one_argument_lineage_and_no_data_access(self):
        m=api();context=dict(training_source={'original':'source'},checkpointmap={'native0':'CP'},
            qualificationmap={'native0':'Q'},FIT=Poison(),image_root=Poison(),unused_cache=Poison())
        teacher=dict(training_spec={'initializers':'unchanged','backbone_with_cp':True},qualification={'original':'Q'})
        calls=[]
        def qualified_training(value):
            self.assertIs(value,context);calls.append(value);return {'native0':teacher}
        legacy=types.SimpleNamespace(qualified_training=qualified_training)
        native=types.SimpleNamespace(ORDER_SEED=12,LOSS_SEED_BASE=34)
        result=m.qualified_teacher_inputs(context,legacy,native)
        self.assertIs(result['teacher'],teacher);self.assertEqual(calls,[context])
        self.assertEqual(result['construction'],dict(initializers='unchanged',backbone_with_cp=True,
            training_source=context['training_source'],checkpointmap=context['checkpointmap'],
            qualificationmap=context['qualificationmap'],order_seed=12,phase_loss_seed_base=34))
        self.assertEqual(teacher['training_spec'],{'initializers':'unchanged','backbone_with_cp':True})

    def test_closed_helper_routes_and_restores_even_when_loading_fails(self):
        m=api();first=lambda ref:('old-first',ref);second=lambda ref:('old-second',ref)
        modules=dict(a=types.SimpleNamespace(bound=first),b=types.SimpleNamespace(bound=second))
        closed=lambda ref:('new-closed',ref)
        with self.assertRaisesRegex(RuntimeError,'retained loader failure'):
            with m.closed_helpers(modules,closed):
                self.assertEqual(modules['a'].bound('CP'),('new-closed','CP'))
                self.assertEqual(modules['b'].bound('Q'),('new-closed','Q'))
                raise RuntimeError('retained loader failure')
        self.assertIs(modules['a'].bound,first);self.assertIs(modules['b'].bound,second)


if __name__=='__main__':unittest.main()
