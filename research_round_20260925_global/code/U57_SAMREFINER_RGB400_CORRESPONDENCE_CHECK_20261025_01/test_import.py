import importlib
import sys
import unittest


class ImportContractTest(unittest.TestCase):
    def test_candidate_import_does_not_load_scientific_libraries_or_run_main(self):
        importlib.import_module('rgb400_check')
        self.assertFalse([name for name in sys.modules
                          if name.split('.')[0] in ('torch', 'numpy')])


if __name__ == '__main__':
    unittest.main()
