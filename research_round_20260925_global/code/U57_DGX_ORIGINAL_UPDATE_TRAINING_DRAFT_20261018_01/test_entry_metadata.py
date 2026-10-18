"""Tiny standard-library cursor/parent-presence boundary; no files or ML."""
import unittest
import entry


class CursorMetadata(unittest.TestCase):
    def test_only_new_zero_cursor_has_no_accepted_parent(self):
        self.assertEqual(entry.validate_cursor_bounds(0,2,None),(0,2))
        self.assertEqual(entry.validate_cursor_bounds(2,4,{'reference':'checked separately'}),(2,4))
        for args in ((0,2,{}),(2,4,None),(False,2,None),(0,0,None),(0,37051,None)):
            with self.assertRaises(ValueError):entry.validate_cursor_bounds(*args)


if __name__=='__main__':unittest.main()
