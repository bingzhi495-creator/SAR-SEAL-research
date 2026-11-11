"""Keep the stock one-rank Paddle order and complete the HRSID FIT2937 tail batch."""

from paddle.io import DistributedBatchSampler


class Fit2937BatchSampler(DistributedBatchSampler):
    def __init__(self, dataset, batch_size, shuffle, drop_last):
        if len(dataset) != 2937 or batch_size != 4 or shuffle is not True or drop_last is not True:
            raise ValueError("Expected the parsed 2,937-image author training roster and batch size four")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        if self.nranks != 1 or self.local_rank != 0:
            raise ValueError("The fixed FIT2937 sampler requires one training device")

    def __iter__(self):
        native_batches = super().__iter__()
        first_batch = next(native_batches)
        padding_indices = first_batch[:3]
        yield first_batch

        for batch in native_batches:
            if len(batch) < self.batch_size:
                batch = batch + padding_indices
            yield batch
