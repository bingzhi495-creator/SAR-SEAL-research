"""Pure string-routing behavior only; no Torch, arrays, network or real data."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
source = Path(__file__).with_name('factory.py')
spec = importlib.util.spec_from_file_location('N28_PURE_ROUTING_ONLY', str(source))
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)


class RoutingMetadataTests(unittest.TestCase):
    def test_selective_gates_cross_once_and_retains_original_memory(self):
        trace = []
        raw = 'D_2_1'

        def gate(value):
            trace.append('gate(' + value + ')')
            return 'G(' + value + ')'

        cross, memory = factory.route_node(raw, gate, 'SELECTIVE')
        self.assertEqual(cross, 'G(D_2_1)')
        self.assertIs(memory, raw)
        self.assertEqual(trace, ['gate(D_2_1)'])

    def test_whole_node_reuses_one_gated_object_for_both_outputs(self):
        trace = []

        def gate(value):
            trace.append('gate(' + value + ')')
            return 'G(' + value + ')'

        cross, memory = factory.route_node('D_1_2', gate, 'WHOLE_NODE')
        self.assertEqual(cross, 'G(D_1_2)')
        self.assertIs(memory, cross)
        self.assertEqual(trace, ['gate(D_1_2)'])

    def test_bad_mode_is_rejected_before_gate_call(self):
        for mode in ('', 'selective', 'OTHER', None, True, []):
            with self.subTest(mode=mode):
                calls = []
                with self.assertRaises(ValueError):
                    factory.route_node('D', lambda value: calls.append(value), mode)
                self.assertEqual(calls, [])

    def test_factory_bad_mode_is_rejected_before_torch_access(self):
        with self.assertRaises(ValueError):
            factory.make_head(None, 'NOT_A_MODE')


if __name__ == '__main__':
    assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    unittest.main()
