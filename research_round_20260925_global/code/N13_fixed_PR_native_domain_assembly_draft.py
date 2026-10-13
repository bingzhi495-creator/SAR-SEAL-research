"""UNBOUND_DRAFT / SYNTHETIC_ONLY: explicitly supplied native-domain assembly.

Normal import uses no Torch, model, data, author module, or source loader. The
caller supplies Torch and the boundary callback. This module cannot authenticate
that callback, establish scientific qualification, or substitute a distance band
for the callback's label-1 region. Logical counts are not runtime/cost measures.
"""


def assemble_fixed_pr_domains(torch_module, author_boundary_function, native_L):
    """Return independent bool masks on L's device from an explicit FP32 raster.

    The callback receives a cloned contiguous CPU FP32 [1,H,W] native mask and
    boundary_width=2. Its result must be CPU int64 [1,H,W] labels from {0,1,2}.
    Only label 1 defines native_width2; no author formula is implemented here.
    """
    t = torch_module
    if not callable(author_boundary_function):
        raise TypeError("author_boundary_function must be callable")
    if not isinstance(native_L, t.Tensor):
        raise TypeError("native_L must be an explicit Torch tensor")
    if native_L.dtype != t.float32:
        raise TypeError("native_L must be FP32")
    if native_L.layout != t.strided:
        raise ValueError("native_L must have strided layout")
    if native_L.device.type not in ("cpu", "cuda"):
        raise ValueError("native_L must be on CPU or CUDA")
    if native_L.ndim != 2 or native_L.numel() == 0:
        raise ValueError("native_L must be a nonempty two-dimensional raster")
    if not bool(t.isfinite(native_L).all().item()):
        raise ValueError("native_L must be finite")

    native_P = (native_L > 0).contiguous().clone()
    old_LOG = (native_L.abs() <= 2).contiguous().clone()
    callback_mask = native_P.detach().to(device="cpu", dtype=t.float32)
    callback_mask = callback_mask.unsqueeze(0).contiguous().clone()
    block = author_boundary_function(callback_mask, boundary_width=2)
    if not isinstance(block, t.Tensor):
        raise TypeError("author boundary result must be a Torch tensor")
    if block.device.type != "cpu":
        raise ValueError("author boundary result must be on CPU")
    if block.dtype != t.int64:
        raise TypeError("author boundary result must have int64 dtype")
    if block.layout != t.strided:
        raise ValueError("author boundary result must have strided layout")
    if tuple(block.shape) != (1, *native_L.shape):
        raise ValueError("author boundary result must have shape [1,H,W]")
    if not bool(((block == 0) | (block == 1) | (block == 2)).all().item()):
        raise ValueError("author boundary labels must be 0, 1, or 2")
    native_width2 = (block[0] == 1).to(device=native_L.device, dtype=t.bool).contiguous().clone()
    new_halo = (native_width2 & ~old_LOG).contiguous().clone()
    Omega = (old_LOG | new_halo).contiguous().clone()
    masks = {"native_P": native_P, "old_LOG": old_LOG, "native_width2": native_width2,
             "new_halo": new_halo, "Omega": Omega}
    counts = {name: int(mask.sum().item()) for name, mask in masks.items()}
    counts["outside"] = int(native_L.numel()) - counts["Omega"]
    return {**masks, "logical_count": counts,
            "scope": {"status": "UNBOUND_DRAFT", "synthetic_validation_only": True,
                      "author_function_identity_verified": False, "scientific_qualification": False,
                      "native_L_identity_verified": False, "actual_SAR_domains_verified": False,
                      "real_runtime_resource_measurement": False,
                      "author_function_numerical_validation": False,
                      "logical_count_is_runtime_or_compute_cost": False}}
