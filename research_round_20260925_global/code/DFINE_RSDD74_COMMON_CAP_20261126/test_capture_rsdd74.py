"""Synthetic intake controls for the frozen Instance-RSDD adapter."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_rsdd74 import ordered_roster


def synthetic():
    items = []
    for index in range(74):
        filename = f"chip_{index:03d}.jpg"
        items.append(dict(image_id=111+index, source_image_name=filename, original_hw=[512, 512],
                          acquisition_group=index % 13, sensor="GF-3" if index < 42 else "TerraSAR-X",
                          rgb_sha256=f"{index:064x}", image_ref=dict(path=f"images/{filename}", bytes=10,
                                                                    sha256=f"{100+index:064x}")))
    return dict(expected_ids=[i["image_id"] for i in items], items=items)


class RosterTest(unittest.TestCase):
    def test_exact_intake_order_and_native_paths(self):
        manifest = synthetic()
        rows, _ = ordered_roster(manifest)
        self.assertEqual([row["image_id"] for row in rows], manifest["expected_ids"])
        self.assertEqual(rows[0]["image_ref"]["path"], "chip_000.jpg")
        self.assertTrue(all(row["original_hw"] == [512, 512] for row in rows))


    def test_rejects_reordered_duplicate_or_resized_inputs(self):
        for field in ("reordered", "duplicate", "resized"):
            with self.subTest(field=field):
                manifest = synthetic()
                if field == "reordered":
                    manifest["items"][0], manifest["items"][1] = manifest["items"][1], manifest["items"][0]
                elif field == "duplicate":
                    manifest["items"][0]["rgb_sha256"] = manifest["items"][1]["rgb_sha256"]
                else:
                    manifest["items"][0]["original_hw"] = [640, 640]
                with self.assertRaises(ValueError):
                    ordered_roster(manifest)

if __name__ == "__main__":
    unittest.main()

