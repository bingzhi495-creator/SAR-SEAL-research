import copy
import unittest

from verify_strict_optimizer_pair import FIRST_GRADIENT_SHA256, verify_backward_control


def route(calls, unchanged_key):
    return dict(routed_forward_calls=calls, cpu_backward_calls=calls,
                observed_shapes=[dict(routed_forward_calls=calls, cpu_backward_calls=calls)],
                **{unchanged_key: 0})


class BackwardControlContract(unittest.TestCase):
    def test_rejects_missing_cpu_backward_calls_even_when_aggregate_claims_match(self):
        row = dict(strict_calls=1488, strict_seconds=320.0, first_counts=[4, 36, 8],
                   first_gradient_sha256=FIRST_GRADIENT_SHA256, cublas_workspace_override=None,
                   grid_sample=route(13392, "unchanged_differentiable_cuda_bilinear_calls"),
                   bilinear_interpolate=route(2976, "unchanged_cuda_bilinear_antialias_calls"))
        self.assertEqual(verify_backward_control(row)["grid_sample_cpu_backward_calls"], 13392)
        mutated = copy.deepcopy(row)
        mutated["grid_sample"]["observed_shapes"][0]["cpu_backward_calls"] -= 1
        with self.assertRaisesRegex(ValueError, "grid_sample"):
            verify_backward_control(mutated)


if __name__ == "__main__":
    unittest.main()

