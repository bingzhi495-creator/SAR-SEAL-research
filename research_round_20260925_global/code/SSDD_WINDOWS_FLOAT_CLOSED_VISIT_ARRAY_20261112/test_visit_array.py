import json
import unittest

from visit_array import decode_stock_visits


def unique(items):
    value = dict(items)
    if len(value) != len(items):
        raise ValueError("duplicate")
    return value


def invalid_constant(value):
    raise ValueError("nonfinite: " + value)


class VisitArrayTest(unittest.TestCase):
    def parse(self, raw):
        return decode_stock_visits(raw, unique, invalid_constant)

    def test_exact_array_decodes_in_original_order(self):
        rows = [{"image_id": n, "batch": n+1} for n in range(32)]
        self.assertEqual(self.parse(json.dumps(rows)), rows)

    def test_reject_wrong_top_level_length_and_row_type(self):
        for value in ({}, [{}]*31, [None]*32):
            with self.subTest(value=type(value)):
                with self.assertRaisesRegex(ValueError, "32-row"):
                    self.parse(json.dumps(value))

    def test_retain_locked_parser_duplicate_and_nonfinite_rejection(self):
        for value, message in (('{"x":1,"x":2}', "duplicate"), ('{"x":NaN}', "nonfinite")):
            with self.assertRaisesRegex(ValueError, message):
                self.parse("[" + ",".join([value]+["{}"]*31) + "]")


if __name__ == "__main__":
    unittest.main()
