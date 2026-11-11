from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import dgx_float_common_short as subject


def documents():
    rows = [
        dict(id=8, image_id=12, category_id=1, bbox=[0, 0, 3, 3], area=9, iscrowd=0,
             segmentation=[[0.0, 0.0, 3.0, 0.0, 3.0, 3.0]]),
        dict(id=5, image_id=12, category_id=1, bbox=[4, 4, 3, 3], area=9, iscrowd=0,
             segmentation=[[4, 4, 7, 4, 7, 7]]),
    ]
    integer = dict(images=[dict(id=12), dict(id=17)], categories=[dict(id=1)], annotations=rows)
    floating = deepcopy(integer)
    floating["annotations"][0]["segmentation"][0][0] = .375
    return integer, floating


class NativeFitDataset:
    def __init__(self, root, rows):
        self.root = root
        self.ids = [12, 17]
        self.rows = rows

    def _load_target(self, image_id):
        return [row for row in self.rows if row["image_id"] == image_id]


class FloatCommonShortTest(unittest.TestCase):
    def test_original_dataset_loads_float_rows_and_restores_both_hooks(self):
        integer, floating = documents()
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            dataset = NativeFitDataset(root, integer["annotations"])

            def build(split):
                return dataset

            module = SimpleNamespace(build_dataset=build)
            overlay = subject.paired_annotations(integer, floating, [12, 17], rows=2)
            with subject.float_dataset_overlay(module, NativeFitDataset, root, [12, 17], overlay):
                with self.assertRaisesRegex(ValueError, "one original FIT"):
                    module.build_dataset("test")
                self.assertIs(module.build_dataset("train"), dataset)
                actual = dataset._load_target(12)
                self.assertEqual(actual, floating["annotations"])
                self.assertEqual([row["id"] for row in actual], [8, 5])
                self.assertEqual(dataset._load_target(17), [])
                actual[0]["segmentation"][0][0] = 100
                self.assertEqual(dataset._load_target(12), floating["annotations"])
                with self.assertRaisesRegex(ValueError, "one original FIT"):
                    module.build_dataset("train")
            self.assertIs(module.build_dataset, build)
            self.assertNotIn("_load_target", vars(dataset))
            self.assertEqual(dataset._load_target(12), integer["annotations"])


    def test_polygon_only_admission_keeps_nonmask_metadata_and_row_order_strict(self):
        integer, floating = documents()
        floating["annotations"][0]["bbox"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "ordered annotation metadata"):
            subject.paired_annotations(integer, floating, [12, 17], rows=2)
        integer, floating = documents()
        floating["annotations"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered annotation metadata"):
            subject.paired_annotations(integer, floating, [12, 17], rows=2)
        integer, floating = documents()
        floating["annotations"][0]["segmentation"][0][0] = 1.1
        with self.assertRaisesRegex(ValueError, "vertex integer truncation"):
            subject.paired_annotations(integer, floating, [12, 17], rows=2)

    def test_original_wait_must_belong_to_float_owner(self):
        owner = dict(pid=300, created=2.5, boot_id="fixture")
        request, summary = "a" * 64, "b" * 64
        events = [subject.arm_start(mode, request) for mode in subject.MODES]
        events.append(subject.pending_event(request, summary))
        captured = dict(status=subject.WAIT_STATUS, owner=owner, request_sha256=request,
                        original_os_wait=True, original_os_exit=0, process_group_closed=True,
                        combined_stdout=b"".join(map(subject.encode, events)).decode("ascii"))
        subject.verify_external_wait(captured, owner, request, summary, lambda _: False)
        with self.assertRaisesRegex(ValueError, "original owner wait"):
            subject.verify_external_wait(dict(captured, status="P340_DGX_ORIGINAL_OWNER_WAIT_V1"),
                                         owner, request, summary, lambda _: False)
        with self.assertRaisesRegex(ValueError, "still live"):
            subject.verify_external_wait(captured, owner, request, summary, lambda _: True)
        with self.assertRaisesRegex(ValueError, "ordered"):
            subject.verify_external_wait(dict(captured, combined_stdout=b"".join(
                map(subject.encode, events[::-1])).decode("ascii")), owner, request, summary, lambda _: False)


if __name__ == "__main__":
    unittest.main()

