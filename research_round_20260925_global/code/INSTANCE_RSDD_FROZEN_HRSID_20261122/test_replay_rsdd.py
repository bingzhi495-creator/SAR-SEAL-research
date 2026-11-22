"""Check that frozen replay retains author image identities, including empty rows."""
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


spec = spec_from_file_location("rsdd_replay_test", Path(__file__).with_name("replay_rsdd.py"))
module = module_from_spec(spec)
spec.loader.exec_module(module)


def worker_probe(size):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return torch.ones(size).sum().item()


class OfficialRoster(unittest.TestCase):
    def setUp(self):
        self.roster = {"images": [
            {"id": 0, "file_name": "11_2_3.jpg", "height": 512, "width": 512},
            {"id": 7, "file_name": "12_2_3.jpg", "height": 512, "width": 512}]}
        self.audit = [dict(image_id=row["id"], source_image_name=row["file_name"], original_hw=[512, 512])
                      for row in self.roster["images"]]
        self.predictions = [{"image_id": 0, "coco": []}, {"image_id": 7, "coco": []}]

    def test_original_and_empty_image_order_is_preserved(self):
        rows = module.author_rows(self.roster, self.predictions, self.audit, count=2)
        self.assertEqual([image["image_id"] for _, image in rows], [0, 7])

    def test_shuffled_or_missing_prediction_cannot_enter_replay(self):
        with self.assertRaisesRegex(ValueError, "Author image identity and order"):
            module.author_rows(self.roster, list(reversed(self.predictions)), self.audit, count=2)
        with self.assertRaisesRegex(ValueError, "Complete author TEST roster"):
            module.author_rows(self.roster, self.predictions[:1], self.audit, count=2)

    def test_windows_workers_load_torch_and_preserve_order(self):
        with ProcessPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(worker_probe, [8, 1])), [8, 1])


if __name__ == "__main__":
    unittest.main()
