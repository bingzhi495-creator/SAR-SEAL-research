"""Synthetic checks for the FIT-only large-label replay and summary."""

from pathlib import Path
import os
import unittest

from large_fit_footprint import build_population, load_runtime, measure_view, prepare_image, summarize_rows, view_seeds


class PopulationTests(unittest.TestCase):
    def test_original_cohort_order_and_full_annotations_are_preserved(self):
        images = [{"id": 17, "file_name": "P0001_a.png", "width": 800, "height": 800},
                  {"id": 13, "file_name": "P0001_b.png", "width": 800, "height": 800},
                  {"id": 3, "file_name": "P0002_a.png", "width": 800, "height": 800}]
        annotations = [
            {"id": 25, "image_id": 13, "area": 9215.999, "category_id": 1, "iscrowd": 0},
            {"id": 26, "image_id": 3, "area": 9216.0, "category_id": 1, "iscrowd": 0},
            {"id": 27, "image_id": 13, "area": 9216.0, "category_id": 1, "iscrowd": 0},
            {"id": 28, "image_id": 17, "area": 2.0, "category_id": 1, "iscrowd": 0},
        ]
        layout = [{"image_id": im["id"], "native_basename": im["file_name"]} for im in reversed(images)]
        population = build_population(dict(images=images, annotations=annotations), layout)
        self.assertEqual([row["image"]["id"] for row in population], [13, 3])
        self.assertEqual([row["ordinal"] for row in population], [1, 2])
        self.assertEqual([ann["id"] for ann in population[0]["annotations"]], [25, 27])
        self.assertEqual(population[0]["large_indices"], [1])
        self.assertEqual(population[1]["large_indices"], [0])
        self.assertEqual(view_seeds(population[0]["ordinal"]), tuple(range(520198004, 520198008)))

    def test_prefix_equal_mean_preserves_separate_view_and_source_denominators(self):
        def row(prefix, image, ann, signed, fn, fp, iou):
            return dict(prefix=prefix, image_id=image, annotation_id=ann, visible=100.,
                        native=dict(signed_area=0., fn=10., fp=10., iou=.8),
                        center=dict(signed_area=signed, fn=fn, fp=fp, iou=iou))
        rows = [
            row("P1", 1, 1, -10., 15., 5., .7),
            row("P1", 1, 1, -10., 15., 5., .7),
            row("P2", 2, 2, 5., 5., 10., .85),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["valid_rows"], 3)
        self.assertEqual(summary["unique_annotations"], 2)
        self.assertEqual(summary["unique_images"], 2)
        self.assertEqual(summary["prefix_groups"], 2)
        self.assertAlmostEqual(summary["delta_prefix_equal"]["signed_area_fraction"], -.025)
        self.assertAlmostEqual(summary["delta_row_equal"]["signed_area_fraction"], -.05)
        self.assertEqual(summary["delta_area_total_source_px2"]["fn"], 5.)



    def test_synthetic_image_uses_exact_native_replay_and_rational_footprint(self):
        from PIL import Image

        runtime = load_runtime(Path(os.environ["P288_CODE_ROOT"]))
        source = Image.new("RGB", (800, 800), (8, 12, 25))
        annotations = [
            dict(id=101, image_id=1, category_id=1, iscrowd=0, area=90000.,
                 bbox=[250., 250., 300., 300.],
                 segmentation=[[250., 250., 550., 250., 550., 550., 250., 550.]]),
            dict(id=102, image_id=1, category_id=1, iscrowd=0, area=100.,
                 bbox=[435., 503., 10., 10.],
                 segmentation=[[435., 503., 445., 503., 445., 513., 435., 513.]]),
        ]
        item = dict(image=dict(id=1), prefix="P001", annotations=annotations, large_indices=[0])
        image, target = prepare_image(runtime, source, item)
        self.assertEqual(len(target["labels"]), 2)
        torch = runtime.source.torch
        seed_by_path = {}
        for seed in range(520198000, 520198008):
            branch = "direct" if torch.rand((), generator=torch.Generator().manual_seed(seed)) < .5 else "crop"
            seed_by_path.setdefault(branch, seed)
        self.assertEqual(set(seed_by_path), {"direct", "crop"})
        for expected_path, seed in seed_by_path.items():
            path, support, rows = measure_view(runtime, item, image, target, seed)
            self.assertEqual(path, expected_path)
            self.assertEqual(support["common_nonempty"], 1)
            self.assertEqual([row["annotation_id"] for row in rows], [101])
            self.assertGreater(rows[0]["visible"], 0.)
            for name in ("native", "center"):
                values = rows[0][name]
                self.assertAlmostEqual(values["signed_area"], values["fp"] - values["fn"], places=10)


if __name__ == "__main__":
    unittest.main()
