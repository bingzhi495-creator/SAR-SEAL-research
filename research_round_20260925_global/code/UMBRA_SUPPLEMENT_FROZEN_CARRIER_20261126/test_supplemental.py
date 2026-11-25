from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from supplemental import load_items, roster_image_ids, validate_native_lineage


class SupplementalRosterTest(unittest.TestCase):
    def roster(self, ids):
        return {
            "schema": "sar-hull-roster-v1",
            "images": [dict(image_id=f"UMB-{n:03d}", height=1024, width=1024,
                            png_sha256=f"{n:064x}") for n in ids],
        }

    def test_original_ids_survive_frozen_withholding(self):
        ids = [n for n in range(101, 121) if n not in {103, 109}]
        self.assertEqual(roster_image_ids(self.roster(ids), 20), ids)
        for bad_ids in (ids[::-1], ids[:-1], [12, *ids[1:]], [*ids[:-1], 123], [ids[0], *ids[:-1]]):
            with self.subTest(bad_ids=bad_ids), self.assertRaises(ValueError):
                roster_image_ids(self.roster(bad_ids), 20)


class SupplementalPixelsTest(unittest.TestCase):
    def test_preserve_native_grayscale_bytes_with_id_gaps(self):
        from PIL import Image

        ids = [101, 103]
        rows = []
        with TemporaryDirectory() as scratch:
            intake = Path(scratch).resolve()
            (intake / "images").mkdir()
            for row_id in ids:
                name = f"UMB-{row_id:03d}"
                path = intake / "images" / (name + ".png")
                Image.new("L", (1024, 1024), row_id).save(path)
                rows.append(dict(image_id=name, png_sha256=sha256(path.read_bytes()).hexdigest()))
            items = load_items(intake, {"images": rows}, ids)
            self.assertEqual([row["image_id"] for row in items], ids)
            self.assertEqual([row["rgb_sha256"] for row in items],
                             [sha256(bytes([i]) * (1024 * 1024 * 3)).hexdigest() for i in ids])
            rows[1]["png_sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                load_items(intake, {"images": rows}, ids)


class SupplementalLineageTest(unittest.TestCase):
    def test_partial_requires_independent_observed_raw_tile_readback(self):
        n, total = 18, 20
        keys = ("intake_result", "intake_terminal", "readback_result", "manifest_sha256",
                "metadata_result_sha256", "metadata_terminal_sha256")
        binding = {key: str(i) * 64 for i, key in enumerate(keys, 1)}
        shared = {key: binding[key] for key in keys[3:]}
        shared.update(requested_windows=total, failed_window_count=total - n, native_window_shape_yx=[1024, 1024])
        intake = dict(status="UMBRA_SUPPLEMENT_NATIVE_WINDOWS_PARTIAL", worker_sha256="a" * 64,
                      verified_native_windows=n, decoded_unique_windows=n, annotation_files=0,
                      detector_forwards=0, optimizer_updates=0, **shared)
        readback = dict(status="UMBRA_SUPPLEMENT_NATIVE_READBACK_COMPLETE", worker_sha256="b" * 64,
                        eligible_for_detector=True, extraction_result_sha256=binding["intake_result"],
                        extraction_terminal_sha256=binding["intake_terminal"], extraction_actual_exit=1,
                        readback_windows=n, unique_pixel_arrays=n, matching_uncompressed_ranges=4 * n, **shared)
        intake_term = dict(phase="native-window-intake", actual_exit=1, output_footer_verified=True,
                           result_sha256=binding["intake_result"], worker_sha256=intake["worker_sha256"])
        check_term = dict(phase="native-window-readback", actual_exit=0, output_footer_verified=True,
                          result_sha256=binding["readback_result"], worker_sha256=readback["worker_sha256"])
        self.assertEqual(validate_native_lineage(intake, intake_term, readback, check_term, binding, n), total)
        for invalid in ({"matching_uncompressed_ranges": 4 * n - 1}, {"eligible_for_detector": False},
                        {"extraction_terminal_sha256": "f" * 64}, {"metadata_result_sha256": "f" * 64}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_native_lineage(intake, intake_term, {**readback, **invalid}, check_term, binding, n)
        with self.assertRaises(ValueError):
            validate_native_lineage(intake, {**intake_term, "output_footer_verified": False}, readback,
                                    check_term, binding, n)


if __name__ == "__main__":
    unittest.main()


