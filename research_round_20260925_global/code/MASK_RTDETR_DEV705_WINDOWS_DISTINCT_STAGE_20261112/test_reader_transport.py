import unittest

import p392_windows_dev705_owner as subject


class Place:
    def __init__(self, text):
        self.text = text

    def is_cpu_place(self):
        return self.text == "Place(cpu)"

    def is_cuda_pinned_place(self):
        return self.text == "Place(gpu_pinned)"

    def is_gpu_place(self):
        return self.text == "Place(gpu:0)"

    def __str__(self):
        return self.text


class ByteView:
    def __init__(self, value):
        self.value = value

    def tobytes(self):
        return self.value


class Tensor:
    def __init__(self, value, shape, dtype, place="Place(cpu)", mutate=False):
        self.value, self.shape, self.dtype = value, shape, dtype
        self.place, self.mutate = Place(place), mutate

    def cuda(self, *, blocking):
        assert blocking
        return Tensor(self.value + b"x" if self.mutate else self.value, self.shape, self.dtype, "Place(gpu:0)")

    def numpy(self):
        return ByteView(self.value)


class Paddle:
    Tensor = Tensor
    float32 = "paddle.float32"


def reader_batch():
    return {"image": Tensor(b"input", (1, 3, 640, 640), Paddle.float32),
            "im_shape": Tensor(b"shape", (1, 2), Paddle.float32),
            "scale_factor": Tensor(b"scale", (1, 2), Paddle.float32),
            "im_id": Tensor(b"identity", (1, 1), "paddle.int64")}


class TransportTest(unittest.TestCase):
    def test_fit_transfers_each_tensor_and_records_exact_before_after(self):
        batch, stats = reader_batch(), {}
        got = subject.transport_cuda_batch(batch, Paddle, stats, verify_bytes=True)
        self.assertEqual(str(batch["image"].place), "Place(cpu)")
        self.assertTrue(all(str(item.place) == "Place(gpu:0)" for item in got.values()))
        witness = stats["batch_transport"]
        self.assertEqual(witness["fit_before"]["image_place"], "Place(cpu)")
        self.assertEqual(witness["fit_after"]["image_place"], "Place(gpu:0)")
        self.assertEqual(witness["fit_transfer_byte_checks"], 4)

    def test_fit_accepts_stock_pinned_host_reader(self):
        batch = reader_batch()
        batch["image"].place = Place("Place(gpu_pinned)")
        stats = {}
        subject.transport_cuda_batch(batch, Paddle, stats, verify_bytes=True)
        self.assertTrue(stats["batch_transport"]["fit_before_image_host"])
        self.assertEqual(stats["batch_transport"]["fit_before"]["image_place"], "Place(gpu_pinned)")

    def test_fit_rejects_nonexact_copy(self):
        batch = reader_batch()
        batch["image"].mutate = True
        with self.assertRaisesRegex(ValueError, "Bitwise FIT"):
            subject.transport_cuda_batch(batch, Paddle, {}, verify_bytes=True)

    def test_reader_still_rejects_bad_shape_or_ground_truth(self):
        batch = reader_batch()
        batch["gt_bbox"] = object()
        with self.assertRaisesRegex(ValueError, "Unexpected pretransport"):
            subject.transport_cuda_batch(batch, Paddle, {}, verify_bytes=False)


if __name__ == "__main__":
    unittest.main()

