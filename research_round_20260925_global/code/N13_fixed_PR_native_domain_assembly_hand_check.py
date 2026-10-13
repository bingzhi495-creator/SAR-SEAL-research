"""Portable literal-label callback check; not an author-function numerical test."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys


def run_check(assembly_path):
    if sys.flags.optimize:
        raise RuntimeError("this check requires normal Python mode")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(name + " must be explicitly 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    import psutil
    process = psutil.Process()
    if not {0, 1}.issubset(set(process.cpu_affinity())):
        raise RuntimeError("CPU 0/1 must both be available")
    process.cpu_affinity([0, 1])
    specification = importlib.util.spec_from_file_location("explicit_unbound_domain_assembly", assembly_path)
    module = importlib.util.module_from_spec(specification)
    torch_preloaded = "torch" in sys.modules
    specification.loader.exec_module(module)
    ordinary_import_kept_torch_unloaded = not torch_preloaded and "torch" not in sys.modules
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checks = 0

    def require(condition, description):
        nonlocal checks
        checks += 1
        if not bool(condition):
            raise RuntimeError(description)

    native_L = torch.tensor([[-3., 9., -2., 9., 0., 9., 2., 9.],
                             [3., 9., -1., 9., 1., 9., -4., 9.]], dtype=torch.float32)[:, ::2]
    original_L = native_L.clone()
    literal_labels = torch.tensor([[[1, 0, 1, 2], [0, 0, 1, 0]]], dtype=torch.int64)
    callback_masks = []

    def literal_callback(mask, *, boundary_width):
        require(boundary_width == 2, "author boundary width must be exactly 2")
        require(mask.device.type == "cpu", "author callback must receive CPU input")
        require(mask.dtype == torch.float32, "author callback must receive FP32 input")
        require(mask.is_contiguous(), "author callback input must be contiguous")
        require(tuple(mask.shape) == (1, 2, 4), "author callback input shape")
        require(torch.equal(mask, torch.tensor([[[0., 0., 0., 1.], [1., 0., 1., 0.]]])),
                "strict L>0 must define the callback mask")
        callback_masks.append(mask)
        mask.zero_()  # A destructive callback must not change L or the returned native_P.
        return literal_labels

    result = module.assemble_fixed_pr_domains(torch, literal_callback, native_L)
    expected = {
        "native_P": [[False, False, False, True], [True, False, True, False]],
        "old_LOG": [[False, True, True, True], [False, True, True, False]],
        "native_width2": [[True, False, True, False], [False, False, True, False]],
        "new_halo": [[True, False, False, False], [False, False, False, False]],
        "Omega": [[True, True, True, True], [False, True, True, False]],
    }
    for key, value in expected.items():
        require(torch.equal(result[key], torch.tensor(value, dtype=torch.bool)), key + " literal result")
        require(result[key].device == native_L.device and result[key].dtype == torch.bool,
                key + " must be bool on L device")
    require(result["logical_count"] == {"native_P": 3, "old_LOG": 5, "native_width2": 3,
                                        "new_halo": 1, "Omega": 6, "outside": 2},
            "logical region counts")
    require(torch.equal(native_L, original_L), "destructive callback must leave L unchanged")
    require(result["scope"]["author_function_identity_verified"] is False,
            "explicit callback must not authenticate its identity")
    require(result["scope"]["scientific_qualification"] is False, "no scientific qualification")
    require(ordinary_import_kept_torch_unloaded, "ordinary assembly import must not load Torch")
    require(not native_L.is_contiguous(), "main fixture must exercise a strided native raster")
    main_counts = dict(result["logical_count"])

    # Labels/callback input and each returned domain have independent storage.
    saved_width2 = result["native_width2"].clone()
    literal_labels.fill_(2)
    callback_masks[0].fill_(1.)
    require(torch.equal(result["native_width2"], saved_width2), "returned width2 must not alias callback labels")
    require(torch.equal(result["native_P"], torch.tensor(expected["native_P"])),
            "returned native_P must not alias the callback argument")
    saved_masks = {key: result[key].clone() for key in expected}
    result["old_LOG"].zero_()
    for key in ("native_P", "native_width2", "new_halo", "Omega"):
        require(torch.equal(result[key], saved_masks[key]), "output domains must not alias each other: " + key)
    require(torch.equal(native_L, original_L), "output mutations must not change native_L")

    no_halo_L = torch.tensor([[-3., 0., 3.]], dtype=torch.float32)

    def no_halo_callback(mask, *, boundary_width):
        require(boundary_width == 2 and mask.device.type == "cpu" and mask.dtype == torch.float32,
                "no-halo callback contract")
        require(tuple(mask.shape) == (1, 1, 3) and mask.is_contiguous(), "no-halo callback raster")
        return torch.tensor([[[0, 1, 2]]], dtype=torch.int64)

    no_halo = module.assemble_fixed_pr_domains(torch, no_halo_callback, no_halo_L)
    require(no_halo["logical_count"] == {"native_P": 1, "old_LOG": 1, "native_width2": 1,
                                         "new_halo": 0, "Omega": 1, "outside": 2},
            "empty halo is valid and has literal counts")
    require(torch.equal(no_halo["Omega"], torch.tensor([[False, True, False]])),
            "empty halo must leave Omega exactly old_LOG")

    rejection_count = 0

    def rejected(callback, L, exception_type, message):
        nonlocal rejection_count
        try:
            module.assemble_fixed_pr_domains(torch, callback, L)
        except exception_type as error:
            require(message in str(error), "invalid input must be explicitly rejected: " + message)
            rejection_count += 1
        else:
            raise RuntimeError("invalid input unexpectedly accepted: " + message)

    def returning(value):
        return lambda mask, *, boundary_width: value

    rejected(returning(torch.zeros((2, 4), dtype=torch.int64)), native_L, ValueError, "shape [1,H,W]")
    rejected(returning(torch.zeros((1, 2, 4), dtype=torch.float32)), native_L, TypeError, "int64 dtype")
    rejected(returning(torch.empty((1, 2, 4), dtype=torch.int64, device="meta")), native_L, ValueError, "on CPU")
    rejected(returning(torch.full((1, 2, 4), 3, dtype=torch.int64)), native_L, ValueError, "labels must be 0, 1, or 2")
    rejected(returning(torch.full((1, 2, 4), -1, dtype=torch.int64)), native_L, ValueError, "labels must be 0, 1, or 2")
    rejected(returning([[[0] * 4] * 2]), native_L, TypeError, "Torch tensor")
    rejected(None, native_L, TypeError, "must be callable")
    bad_input_calls = []

    def forbidden_callback(mask, *, boundary_width):
        bad_input_calls.append(True)
        raise RuntimeError("invalid native_L must be rejected before the callback")

    for value in (float("nan"), float("inf"), -float("inf")):
        rejected(forbidden_callback, torch.tensor([[value]], dtype=torch.float32), ValueError, "finite")
    rejected(forbidden_callback, torch.zeros((2, 4), dtype=torch.float64), TypeError, "FP32")
    rejected(forbidden_callback, torch.zeros((2, 4, 1), dtype=torch.float32), ValueError, "two-dimensional")
    rejected(forbidden_callback, torch.zeros((0, 4), dtype=torch.float32), ValueError, "nonempty")
    require(not bad_input_calls, "invalid native input must not reach the callback")
    require(torch.equal(native_L, original_L), "all checks must leave the supplied native L unchanged")
    require(not torch.cuda.is_initialized(), "CUDA must remain uninitialized")
    return {"status": "PASS_SYNTHETIC_ONLY", "checks": checks, "main_counts": main_counts,
            "no_halo_counts": no_halo["logical_count"], "explicit_rejections": rejection_count,
            "python_version": sys.version.split()[0], "torch_version": torch.__version__,
            "numpy_version": importlib.metadata.version("numpy"),
            "cpu_affinity": process.cpu_affinity(), "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "CUDA_initialized": torch.cuda.is_initialized(),
            "author_function_numerical_validation": False, "scientific_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembly", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_check(arguments.assembly), sort_keys=True, allow_nan=False))
