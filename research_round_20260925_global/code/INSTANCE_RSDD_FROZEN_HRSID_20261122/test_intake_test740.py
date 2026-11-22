"""Exercise bounded ZIP range extraction against a synthetic archive."""
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import unittest
import zipfile

spec = spec_from_file_location("image_intake", Path(__file__).with_name("intake_test740.py"))
intake = module_from_spec(spec)
spec.loader.exec_module(intake)


class ZipIntake(unittest.TestCase):
    def test_member_validates_local_name_and_crc(self):
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            dst.writestr("RSDD/JPEGImages/7_1_2.jpg", b"raw jpeg fixture")
        original = stream.getvalue()
        with zipfile.ZipFile(stream) as src:
            member, = src.infolist()
            span = original[:src.start_dir]
        self.assertEqual(intake.extract_member(member, span), b"raw jpeg fixture")
        corrupt = span.replace(b"7_1_2.jpg", b"6_1_2.jpg", 1)
        with self.assertRaisesRegex(ValueError, "member name"):
            intake.extract_member(member, corrupt)

    def test_span_merging_is_bounded(self):
        a, b, c = (0, 11, 1), (14, 19, 2), (24, 31, 3)
        self.assertEqual(intake.coalesce_ranges([a,b,c], gap=5,max_size=20), [(0,19,[a,b]),(24,31,[c])])


if __name__ == "__main__":
    unittest.main()
