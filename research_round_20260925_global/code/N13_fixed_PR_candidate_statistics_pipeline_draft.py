"""UNBOUND_DRAFT / SYNTHETIC_ONLY: one explicit fixed-PR matched pair.

Caller supplies Torch, adapter, statistics, oracle, eight weights, exact native
L, F_original and disjoint old_LOG/new_halo. No loader or scientific binding is
provided. old_LOG is checked exactly as abs(L)<=2; native A is L>0. The adapter
gets no G, selected set or K. It computes every Omega candidate with the fixed
strict_gt_zero rule. Only old_LOG is written to reconstruct the full original
LOG binary; exact equality to the caller's mandatory prediction reference is a
gate before statistics sees G. S=old_LOG and K=count(old_LOG) are fixed here.

The reference is an original prediction, not GT. Passing equality certifies
only equality of the supplied binary rasters: weights, L/F/reference identity,
renderer, candidate independence, real pairing, runtime and scientific source
remain unqualified. Caller declarations cannot authenticate these properties.
No width2 construction, ordering implementation, AP/COCO, SOURCE registration,
file access, data/weight loading, or scientific execution occurs. Normal import
uses only the standard library. The companion runner uses one hand-built case.
"""
import json


STATUS = "UNBOUND_DRAFT_SYNTHETIC_ONLY"
BINARY_RULE = "strict_gt_zero"


def _raster(torch_module, value, name, shape, dtype, device):
    if not isinstance(value, torch_module.Tensor):
        raise TypeError(name + " must be an explicit Torch tensor")
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(name + " must have the explicit original-raster shape/dtype")
    if value.layout != torch_module.strided or value.device != device:
        raise ValueError(name + " must have strided layout and the native_L device")


def diagnose_fixed_pr_pair(torch_module, adapter, statistics, oracle, *,
                           weight_tensors, F_original, native_L, old_LOG,
                           new_halo, G, image_id, gt_id, query_id,
                           expected_original_LOG_binary, metadata, distance_bands=None):
    """Return Torch candidate/reference reconstruction and JSON-safe records.

    Raster/weights are explicit Torch tensors. G is the explicit NumPy Boolean
    raster required by supplied statistics; it is used only in that call after
    reconstruction succeeds. metadata and all three actual identities are
    mandatory and checked by statistics. No final S, K or comparator is exposed.
    Optional distance_bands are forwarded unchanged after reconstruction passes;
    no distance formula or source identity is constructed or certified here.
    """
    if not isinstance(native_L, torch_module.Tensor) or native_L.ndim != 2:
        raise ValueError("native_L must have shape [H,W]")
    shape = tuple(native_L.shape)
    if min(shape) <= 0 or native_L.device.type not in ("cpu", "cuda"):
        raise ValueError("native_L must have a nonempty raster on CPU/CUDA")
    _raster(torch_module, native_L, "native_L", shape, torch_module.float32, native_L.device)
    for name, value in (("old_LOG", old_LOG), ("new_halo", new_halo),
                        ("expected_original_LOG_binary", expected_original_LOG_binary)):
        _raster(torch_module, value, name, shape, torch_module.bool, native_L.device)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an explicit JSON dictionary")
    if not callable(getattr(adapter, "fixed_pr_candidates", None)):
        raise TypeError("caller must supply the fixed PR adapter API")
    if not callable(getattr(statistics, "diagnose_instance", None)):
        raise TypeError("caller must supply the diagnostic statistics API")

    with torch_module.no_grad():
        if not bool(torch_module.isfinite(native_L).all().item()):
            raise ValueError("native_L must contain finite exact supplied FP32 values")
        if not torch_module.equal(old_LOG, native_L.abs() <= 2):
            raise ValueError("old_LOG must equal abs(native_L)<=2 exactly")
        if bool((old_LOG & new_halo).any().item()):
            raise ValueError("old_LOG and new_halo must be disjoint")
        candidates = adapter.fixed_pr_candidates(
            torch_module, weight_tensors, F_original, native_L, old_LOG, new_halo,
            refined_binary_rule=BINARY_RULE, return_merged=True)
        C = candidates["merged_C"]
        _raster(torch_module, C, "adapter merged_C", shape, torch_module.bool, native_L.device)
        A = native_L > 0
        Omega = old_LOG | new_halo
        if not torch_module.equal(candidates["native_support_L_gt_zero"], A):
            raise ValueError("adapter native support must equal native_L>0")
        if not torch_module.equal(C[~Omega], A[~Omega]):
            raise ValueError("candidate must preserve native support outside Omega")
        reconstructed = A.contiguous().clone()
        reconstructed[old_LOG] = C[old_LOG]
        if not torch_module.equal(reconstructed, expected_original_LOG_binary):
            raise ValueError("original LOG reconstruction differs from the mandatory prediction reference; statistics refused")
        K = int(old_LOG.sum().item())
        omega_count = int(Omega.sum().item())

    # Copies avoid sharing native/candidate/domain tensor storage with statistics.
    def cpu_boolean_copy(value):
        return value.detach().cpu().numpy().copy()

    diagnostic = statistics.diagnose_instance(
        oracle, A=cpu_boolean_copy(A), C=cpu_boolean_copy(C), G=G,
        old_LOG=cpu_boolean_copy(old_LOG), new_halo=cpu_boolean_copy(new_halo),
        S=cpu_boolean_copy(old_LOG), K=K, image_id=image_id, gt_id=gt_id,
        query_id=query_id, candidate_independent_of_selection=True, metadata=metadata,
        distance_bands=distance_bands)
    diagnostic = json.loads(json.dumps(diagnostic, allow_nan=False))
    return {
        "status": STATUS, "candidate": C, "candidate_partitions": candidates,
        "original_LOG_binary": reconstructed, "diagnostic": diagnostic,
        "reconstruction_check": {
            "passed": True, "comparison": "exact_full_raster_boolean_equality",
            "binary_rule": BINARY_RULE, "shape": list(shape), "compared_pixels": int(native_L.numel()),
            "old_LOG_exact_abs_L_le_2": True, "native_support_L_gt_zero": True,
            "selected_count": K, "written_count": K,
            "outside_old_LOG_preserved_as_native": True, "G_used": False,
            "reference_kind": "caller_supplied_original_LOG_prediction_binary",
            "float_logit_equality_checked": False, "class_score_order_empty_export_checked": False,
        },
        "scope": {
            "evidence": "SYNTHETIC_ONLY", "scientific_execution_qualified": False,
            "actual_input_reference_and_pair_identity_certified": False,
            "candidate_independence_certified": False, "source_registered": False,
            "logical_candidate_points": omega_count, "actual_end_to_end_cost_measured": False,
            "ap_or_deployable_gain_claim": False,
        },
    }
