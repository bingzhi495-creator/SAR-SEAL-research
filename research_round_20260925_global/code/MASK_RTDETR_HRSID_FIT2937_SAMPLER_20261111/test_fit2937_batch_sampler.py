from collections import Counter
from itertools import chain
import unittest

import paddle

from fit2937_batch_sampler import Fit2937BatchSampler


class IndexedDataset(paddle.io.Dataset):
    def __len__(self):
        return 2937

    def __getitem__(self, index):
        return index


class Fit2937BatchSamplerTest(unittest.TestCase):
    def test_all_24_epochs_keep_the_native_permutation_and_finish_batch(self):
        dataset = IndexedDataset()
        author = paddle.io.DistributedBatchSampler(dataset, batch_size=4, shuffle=True, drop_last=False)
        first_arm = Fit2937BatchSampler(dataset, batch_size=4, shuffle=True, drop_last=True)
        second_arm = Fit2937BatchSampler(dataset, batch_size=4, shuffle=True, drop_last=True)
        observed = set()
        for epoch in range(24):
            for sampler in (author, first_arm, second_arm):
                sampler.set_epoch(epoch)
            batches = list(first_arm)
            self.assertEqual(batches, list(second_arm))
            native = list(author)
            self.assertEqual(batches[:-1], native[:-1])
            self.assertEqual(batches[-1], native[-1] + native[0][:3])
            self.assertEqual(len(first_arm), len(batches), 735)
            self.assertEqual({len(batch) for batch in batches}, {4})
            visits = list(chain.from_iterable(batches))
            counts = Counter(visits)
            self.assertEqual(len(visits), 2940)
            self.assertEqual(len(counts), 2937)
            self.assertEqual({i for i, count in counts.items() if count == 2}, set(visits[:3]))
            observed.add(tuple(visits))
        self.assertEqual(len(observed), 24)

    def test_real_paddle_dataloader_yields_all_batches(self):
        paddle.set_device("cpu")
        dataset = IndexedDataset()
        sampler = Fit2937BatchSampler(dataset, batch_size=4, shuffle=True, drop_last=True)
        sampler.set_epoch(0)
        expected = list(sampler)
        sampler.set_epoch(0)
        loader = paddle.io.DataLoader(dataset, batch_sampler=sampler, num_workers=0, return_list=True)
        actual = [batch.numpy().tolist() for batch in loader]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
