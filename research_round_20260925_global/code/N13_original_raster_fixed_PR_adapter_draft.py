"""UNBOUND DRAFT: original-raster fixed PR candidates, 2026-10-14.

Evidence scope: SYNTHETIC_ONLY. This file has no checkpoint loader, renderer,
selector, GT access, source/admission logic, scientific qualification, or upload.
It must not be adopted as a qualified scientific source without ROOT's review
of the actual immutable call chain after the relevant owner has terminated.

The only authoring references are ordinary release Markdown:
  TRAIN_ABI.md
  N13_FIXED_DEV_EVALUATION_INTERFACE_CONTRACT_20261013.md

Documented mathematics:
  h = non-affine LayerNorm(F256, eps=1e-5)
  c = clamp(native_L, -8, 8) / 8
  h = ReLU(Linear(concat(h, c))) three times, width 91
  u = Linear(concat(h, c)), width 1
  refined_L = native_L + u at supplied candidate points
No old |L|<=2 head-internal gate is applied. The documented e and zero patch49
are residual-independent ABI inputs; this mathematical API omits them. A private
synthetic wrapper checks only their shapes/zero requirement, not real identities.

Eight ordered FP32 tensors are required; WEIGHT_ROLES are architecture labels,
NOT verified checkpoint keys. ROOT/caller must establish the actual mapping,
native query identity, exact native L, original renderer/provenance, domain
definitions, and the actual refined exporter binary rule. Neither strict >0 nor
inclusive >=0 is silently selected for refined candidates. Native support is
explicitly native_L > 0; optional merged C preserves that outside the two domains.

F_original is already supplied on the original raster [256,H,W]. There is no
x/y coordinate, crop/ROI transform, interpolation, or automatic rendering.
originalLOG and addedHalo are disjoint, explicit bool [H,W] domains. Their
ascending raster indices are evaluated in SEPARATE fixed 2048-point partitions.
Changing addedHalo therefore cannot repartition originalLOG's point batches.
There is no GT, K, final selected-set, or selection-budget argument. All supplied
candidate points are evaluated. The caller can select among the returned
candidates only outside this API.

Importing this file imports standard library modules only. Torch is an explicit
argument to mathematical functions. Only the --self-test CLI imports host Torch;
it uses CPU, one intra/inter-op thread, and only synthetic in-memory tensors.
The documented scientific library is Torch 2.0.1+cu118; host self-tests do not
qualify that runtime, a renderer, an endpoint, a source mapping, or a cohort.
"""
import argparse
import inspect
import json
import os
import time


STATUS = "UNBOUND_DRAFT_SYNTHETIC_ONLY"
POINT_CHUNK_SIZE = 2048
PARAMETER_SCALARS = 40497
WEIGHT_ROLES = (
    "hidden1.weight", "hidden1.bias",
    "hidden2.weight", "hidden2.bias",
    "hidden3.weight", "hidden3.bias",
    "final.weight", "final.bias",
)
WEIGHT_SHAPES = (
    (91, 257), (91,),
    (91, 92), (91,),
    (91, 92), (91,),
    (1, 92), (1,),
)
BINARY_RULES = ("strict_gt_zero", "inclusive_ge_zero")
DOCUMENTED_SCIENCE_TORCH = "2.0.1+cu118"


def _tensor_check(torch_module, value, name, shape, dtype, device=None):
    if not isinstance(value, torch_module.Tensor):
        raise TypeError(name + " must be a Torch tensor")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(name + " has an invalid shape")
    if value.dtype != dtype:
        raise TypeError(name + " has an invalid dtype; FP32/bool is explicit")
    if value.layout != torch_module.strided:
        raise TypeError(name + " must have strided layout")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError(name + " must be on a material CPU or CUDA device")
    if device is not None and value.device != device:
        raise ValueError(name + " has a different device")
    if dtype != torch_module.bool:
        if not bool(torch_module.isfinite(value).all().item()):
            raise ValueError(name + " contains nonfinite values")
    return value


def _weights_check(torch_module, weight_tensors, device):
    if not isinstance(weight_tensors, (tuple, list)):
        raise TypeError("weight_tensors must be an explicit ordered tuple/list")
    if len(weight_tensors) != 8:
        raise ValueError("exactly eight ordered weight tensors are required")
    for role, shape, tensor in zip(WEIGHT_ROLES, WEIGHT_SHAPES, weight_tensors):
        _tensor_check(torch_module, tensor, role, shape, torch_module.float32,
                      device)
    if sum(int(tensor.numel()) for tensor in weight_tensors) != PARAMETER_SCALARS:
        raise ValueError("weight scalar count is not 40497")


def raw_pr_residual(torch_module, weight_tensors, F_points, native_logit_points):
    """Return ungated FP32 u[N] for explicit F_points[N,256], native L[N].

    This is inference arithmetic only: no backward, optimizer, input mutation,
    .grad mutation, or implicit library/device/dtype conversion. All tensors,
    including the eight weights, must be finite FP32 on one material device.
    The result is detached. Input requires_grad and existing .grad are preserved.
    """
    if not isinstance(F_points, torch_module.Tensor) or F_points.ndim != 2:
        raise ValueError("F_points must have shape [N,256]")
    point_count = int(F_points.shape[0])
    _tensor_check(torch_module, F_points, "F_points", (point_count, 256),
                  torch_module.float32)
    _tensor_check(torch_module, native_logit_points, "native_logit_points",
                  (point_count,), torch_module.float32, F_points.device)
    _weights_check(torch_module, weight_tensors, F_points.device)
    if point_count == 0:
        return native_logit_points.detach().clone()

    # Independent documented LN / linear / ReLU arithmetic, without head calls.
    with torch_module.no_grad():
        coarse = native_logit_points.clamp(-8.0, 8.0).unsqueeze(1) / 8.0
        hidden = torch_module.nn.functional.layer_norm(
            F_points, (256,), weight=None, bias=None, eps=1e-5)
        hidden = torch_module.nn.functional.linear(
            torch_module.cat((hidden, coarse), dim=1),
            weight_tensors[0], weight_tensors[1])
        hidden = torch_module.nn.functional.relu(hidden)
        hidden = torch_module.nn.functional.linear(
            torch_module.cat((hidden, coarse), dim=1),
            weight_tensors[2], weight_tensors[3])
        hidden = torch_module.nn.functional.relu(hidden)
        hidden = torch_module.nn.functional.linear(
            torch_module.cat((hidden, coarse), dim=1),
            weight_tensors[4], weight_tensors[5])
        hidden = torch_module.nn.functional.relu(hidden)
        residual = torch_module.nn.functional.linear(
            torch_module.cat((hidden, coarse), dim=1),
            weight_tensors[6], weight_tensors[7]).squeeze(1)
        if not bool(torch_module.isfinite(residual).all().item()):
            raise ValueError("PR arithmetic produced a nonfinite raw residual")
    return residual


def _refined_binary(refined_logits, refined_binary_rule):
    if refined_binary_rule == "strict_gt_zero":
        return refined_logits > 0
    if refined_binary_rule == "inclusive_ge_zero":
        return refined_logits >= 0
    raise ValueError("an explicit supported refined_binary_rule is required")


def _partition(torch_module, weight_tensors, F_raster_points, L_flat, domain,
               refined_binary_rule):
    # nonzero returns row-major order; sort makes the ascending contract explicit.
    indices = torch_module.sort(
        torch_module.nonzero(domain.reshape(-1), as_tuple=False).reshape(-1)
    ).values
    point_count = int(indices.numel())
    residual_chunks = []
    refined_chunks = []
    binary_chunks = []
    for start in range(0, point_count, POINT_CHUNK_SIZE):
        chunk_indices = indices[start:start + POINT_CHUNK_SIZE]
        coarse = L_flat[chunk_indices]
        residual = raw_pr_residual(
            torch_module, weight_tensors,
            F_raster_points[chunk_indices], coarse)
        refined = coarse + residual
        if not bool(torch_module.isfinite(refined).all().item()):
            raise ValueError("PR arithmetic produced a nonfinite refined logit")
        residual_chunks.append(residual)
        refined_chunks.append(refined)
        binary_chunks.append(_refined_binary(refined, refined_binary_rule))
    if point_count:
        residuals = torch_module.cat(residual_chunks)
        refined_logits = torch_module.cat(refined_chunks)
        binary_candidates = torch_module.cat(binary_chunks)
    else:
        residuals = L_flat.new_empty((0,))
        refined_logits = L_flat.new_empty((0,))
        binary_candidates = torch_module.empty(
            (0,), dtype=torch_module.bool, device=L_flat.device)
    return {
        "flat_indices": indices,
        "raw_residual": residuals,
        "refined_logit": refined_logits,
        "binary_candidate": binary_candidates,
        "chunk_count": (point_count + POINT_CHUNK_SIZE - 1) // POINT_CHUNK_SIZE,
    }


def fixed_pr_candidates(torch_module, weight_tensors, F_original, native_L,
                        originalLOG_domain, addedHalo_domain, *,
                        refined_binary_rule, return_merged=False):
    """Evaluate both independent candidate partitions on the original raster.

    This accepts supplied domains, not a selection rule. It neither validates
    that originalLOG is |L|<=2 nor constructs the halo. Those and true source
    identity remain caller/ROOT obligations. Chunk size is fixed, not adjustable.

    Returns originalLOG/addedHalo dictionaries, each with ascending int64
    flat_indices, FP32 raw_residual/refined_logit, bool binary_candidate and
    chunk_count; also bool native_support_L_gt_zero. If return_merged=True,
    merged_C is native support with all supplied candidate points substituted.
    merged_C is a complete candidate mask, not a final budget-selected mask.
    """
    if refined_binary_rule not in BINARY_RULES:
        raise ValueError("refined_binary_rule must explicitly be one of " +
                         repr(BINARY_RULES))
    if type(return_merged) is not bool:
        raise TypeError("return_merged must be bool")
    if not isinstance(native_L, torch_module.Tensor) or native_L.ndim != 2:
        raise ValueError("native_L must have shape [H,W]")
    height, width = (int(native_L.shape[0]), int(native_L.shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("original raster H and W must be positive")
    _tensor_check(torch_module, native_L, "native_L", (height, width),
                  torch_module.float32)
    _tensor_check(torch_module, F_original, "F_original",
                  (256, height, width), torch_module.float32, native_L.device)
    _weights_check(torch_module, weight_tensors, native_L.device)
    _tensor_check(torch_module, originalLOG_domain, "originalLOG_domain",
                  (height, width), torch_module.bool, native_L.device)
    _tensor_check(torch_module, addedHalo_domain, "addedHalo_domain",
                  (height, width), torch_module.bool, native_L.device)
    if bool((originalLOG_domain & addedHalo_domain).any().item()):
        raise ValueError("originalLOG and addedHalo domains must be disjoint")

    with torch_module.no_grad():
        raster_points = F_original.reshape(256, height * width).transpose(0, 1)
        L_flat = native_L.reshape(-1)
        original = _partition(torch_module, weight_tensors, raster_points,
                              L_flat, originalLOG_domain, refined_binary_rule)
        added = _partition(torch_module, weight_tensors, raster_points,
                           L_flat, addedHalo_domain, refined_binary_rule)
        native_support = native_L > 0
        result = {
            "status": STATUS,
            "refined_binary_rule": refined_binary_rule,
            "point_chunk_size": POINT_CHUNK_SIZE,
            "originalLOG": original,
            "addedHalo": added,
            "native_support_L_gt_zero": native_support,
        }
        if return_merged:
            # Force a view-safe raster for updates even when native_L is strided.
            merged = native_support.contiguous().clone()
            merged_flat = merged.reshape(-1)
            merged_flat[original["flat_indices"]] = original["binary_candidate"]
            merged_flat[added["flat_indices"]] = added["binary_candidate"]
            result["merged_C"] = merged
    return result


def _synthetic_abi_wrapper(torch_module, weight_tensors, F_points, L_points,
                           e_points, zero_patch49):
    """Synthetic shape-only ABI illustration; qualifies no original e identity."""
    point_count = int(F_points.shape[0])
    _tensor_check(torch_module, e_points, "synthetic e_points",
                  (point_count, 256), torch_module.float32, F_points.device)
    _tensor_check(torch_module, zero_patch49, "synthetic zero_patch49",
                  (point_count, 49), torch_module.float32, F_points.device)
    if bool((zero_patch49 != 0).any().item()):
        raise ValueError("synthetic ABI illustration requires a zero patch49")
    return raw_pr_residual(torch_module, weight_tensors, F_points, L_points)


def _check(condition, message):
    if not bool(condition):
        raise RuntimeError("synthetic check failed: " + message)


def _expect_rejected(operation, message):
    try:
        operation()
    except (TypeError, ValueError):
        return
    raise RuntimeError("synthetic rejection check failed: " + message)


def _self_test(torch_module):
    if not __debug__:
        raise RuntimeError("optimized self-tests are rejected; omit -O/-OO")
    started = time.perf_counter()
    checks = []
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(2026101401)

    weights = tuple(torch_module.randn(
        shape, generator=generator, device="cpu", dtype=torch_module.float32
    ) * 0.04 for shape in WEIGHT_SHAPES)
    _check(len(weights) == 8 and sum(w.numel() for w in weights) == 40497,
           "eight weights / 40497 scalar parameters")
    _check(set(inspect.signature(fixed_pr_candidates).parameters) == {
        "torch_module", "weight_tensors", "F_original", "native_L",
        "originalLOG_domain", "addedHalo_domain", "refined_binary_rule",
        "return_merged"}, "API contains only the explicit mathematical inputs")
    checks.append("8_tensors_40497_parameters_and_no_GT_K_selection_API")

    # Construct nn.Module independently: real LayerNorm and four Linear layers.
    class SyntheticModuleReference(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch_module.nn.LayerNorm(
                256, eps=1e-5, elementwise_affine=False)
            self.first = torch_module.nn.Linear(257, 91)
            self.second = torch_module.nn.Linear(92, 91)
            self.third = torch_module.nn.Linear(92, 91)
            self.last = torch_module.nn.Linear(92, 1)
            with torch_module.no_grad():
                for layer, offset in ((self.first, 0), (self.second, 2),
                                      (self.third, 4), (self.last, 6)):
                    layer.weight.copy_(weights[offset])
                    layer.bias.copy_(weights[offset + 1])

        def forward(self, features, logits):
            c = (logits.clamp(min=-8.0, max=8.0) / 8.0).reshape(-1, 1)
            h = self.norm(features)
            h = torch_module.relu(self.first(torch_module.cat((h, c), 1)))
            h = torch_module.relu(self.second(torch_module.cat((h, c), 1)))
            h = torch_module.relu(self.third(torch_module.cat((h, c), 1)))
            return self.last(torch_module.cat((h, c), 1)).reshape(-1)

    features = torch_module.randn((19, 256), generator=generator)
    logits = torch_module.linspace(-12.0, 12.0, 19)
    reference = SyntheticModuleReference().eval()
    with torch_module.no_grad():
        module_u = reference(features, logits)
    functional_u = raw_pr_residual(torch_module, weights, features, logits)
    _check(torch_module.allclose(functional_u, module_u, rtol=1e-6, atol=1e-7),
           "raw formula equals independently constructed Module/LayerNorm")
    _check(sum(p.numel() for p in reference.parameters()) == 40497,
           "independent reference has the documented parameter count")
    e1 = torch_module.randn((19, 256), generator=generator)
    e2 = torch_module.randn((19, 256), generator=generator)
    patch = torch_module.zeros((19, 49))
    _check(torch_module.equal(
        _synthetic_abi_wrapper(torch_module, weights, features, logits, e1, patch),
        _synthetic_abi_wrapper(torch_module, weights, features, logits, e2, patch)),
        "synthetic e values have no residual dependence")
    _expect_rejected(lambda: _synthetic_abi_wrapper(
        torch_module, weights, features, logits, e1, patch + 1),
        "nonzero synthetic patch49")
    checks.append("independent_Module_reference_and_synthetic_ABI_only")

    zero_final = list(w.clone() for w in weights)
    zero_final[6].zero_()
    zero_final[7].zero_()
    small_F = torch_module.randn((256, 2, 4), generator=generator)
    small_L = torch_module.tensor(
        [[-3.0, 0.0, 1.0, 0.0], [0.0, -0.5, 2.0, 0.0]],
        dtype=torch_module.float32)
    old_domain = torch_module.tensor(
        [[True, True, True, False], [False, False, False, False]])
    add_domain = torch_module.tensor(
        [[False, False, False, True], [True, False, False, False]])
    for rule in BINARY_RULES:
        result = fixed_pr_candidates(
            torch_module, zero_final, small_F, small_L, old_domain, add_domain,
            refined_binary_rule=rule, return_merged=True)
        for key, domain in (("originalLOG", old_domain), ("addedHalo", add_domain)):
            part = result[key]
            native_at_points = small_L.reshape(-1)[part["flat_indices"]]
            _check(torch_module.equal(part["raw_residual"],
                                      torch_module.zeros_like(native_at_points)),
                   "zero final layer yields zero raw residual")
            _check(torch_module.equal(part["refined_logit"], native_at_points),
                   "zero final layer preserves native logits")
            expected = native_at_points > 0 if rule == "strict_gt_zero" \
                else native_at_points >= 0
            _check(torch_module.equal(part["binary_candidate"], expected),
                   "explicit refined threshold including exact L=0")
            _check(torch_module.equal(
                part["flat_indices"], torch_module.nonzero(
                    domain.reshape(-1), as_tuple=False).reshape(-1)),
                "disjoint partition identities")
        outside = ~(old_domain | add_domain)
        _check(torch_module.equal(
            result["native_support_L_gt_zero"], small_L > 0),
            "native support always uses documented strict L>0")
        _check(torch_module.equal(
            result["merged_C"][outside], (small_L > 0)[outside]),
            "merged C preserves native outside domains including L=0")
        _check(not result["merged_C"][1, 3].item(),
               "inclusive refined rule must not include an outside L=0")
    empty = torch_module.zeros_like(old_domain)
    empty_result = fixed_pr_candidates(
        torch_module, zero_final, small_F, small_L, empty, empty,
        refined_binary_rule="strict_gt_zero", return_merged=True)
    _check(all(empty_result[k]["chunk_count"] == 0 and
               empty_result[k]["flat_indices"].numel() == 0
               for k in ("originalLOG", "addedHalo")), "empty domains")
    _check(torch_module.equal(empty_result["merged_C"], small_L > 0),
           "empty domains preserve complete native support")
    noncontiguous = fixed_pr_candidates(
        torch_module, zero_final, small_F.transpose(1, 2), small_L.transpose(0, 1),
        old_domain.transpose(0, 1), add_domain.transpose(0, 1),
        refined_binary_rule="inclusive_ge_zero", return_merged=True)
    expected_noncontiguous = torch_module.where(
        (old_domain | add_domain).transpose(0, 1),
        small_L.transpose(0, 1) >= 0, small_L.transpose(0, 1) > 0)
    _check(torch_module.equal(noncontiguous["merged_C"], expected_noncontiguous),
           "strided input raster receives merged candidate updates")
    checks.append(
        "zero_final_strict_inclusive_L_zero_native_outside_empty_and_strided")

    constant_u = list(w.clone() for w in zero_final)
    constant_u[7].fill_(0.25)
    outside_band_L = torch_module.tensor([[-3.0, 3.0]])
    outside_band_F = torch_module.randn((256, 1, 2), generator=generator)
    ungated = fixed_pr_candidates(
        torch_module, constant_u, outside_band_F, outside_band_L,
        torch_module.tensor([[False, False]]),
        torch_module.tensor([[True, True]]),
        refined_binary_rule="strict_gt_zero")
    _check(torch_module.equal(
        ungated["addedHalo"]["refined_logit"], outside_band_L.reshape(-1) + 0.25),
        "added points outside old |L|<=2 band receive raw residual")
    checks.append("no_old_head_internal_band_gate")

    # >2048 original points exercises two unchanged original batches.
    large_F = torch_module.randn((256, 5, 870), generator=generator)
    large_L = torch_module.randn((5, 870), generator=generator)
    flat_ids = torch_module.arange(4350, dtype=torch_module.int64)
    old_large = (flat_ids % 2 == 0).reshape(5, 870)
    add_small = ((flat_ids % 2 == 1) & (flat_ids < 20)).reshape(5, 870)
    add_large = (flat_ids % 2 == 1).reshape(5, 870)
    guarded = list(weights) + [large_F, large_L]
    for tensor in guarded:
        tensor.requires_grad_(True)
        tensor.grad = torch_module.ones_like(tensor)
    before_values = [tensor.detach().clone() for tensor in guarded]
    before_grads = [tensor.grad.clone() for tensor in guarded]
    before_domains = [mask.clone() for mask in (old_large, add_small, add_large)]
    unchanged_old = []
    for add in (add_small, add_large):
        result = fixed_pr_candidates(
            torch_module, weights, large_F, large_L, old_large, add,
            refined_binary_rule="strict_gt_zero", return_merged=True)
        unchanged_old.append(result["originalLOG"])
        _check(result["originalLOG"]["chunk_count"] == 2,
               "original points retain fixed 2048 chunk boundaries")
        for key, mask in (("originalLOG", old_large), ("addedHalo", add)):
            part = result[key]
            _check(torch_module.equal(
                part["flat_indices"], flat_ids[mask.reshape(-1)]),
                "partition indices exactly equal ascending raster domain")
            _check(not part["raw_residual"].requires_grad and
                   not part["refined_logit"].requires_grad,
                   "inference outputs do not construct gradients")
        all_ids = torch_module.cat((
            result["originalLOG"]["flat_indices"],
            result["addedHalo"]["flat_indices"]))
        _check(torch_module.unique(all_ids).numel() == all_ids.numel(),
               "partitions are disjoint")
        _check(torch_module.equal(torch_module.sort(all_ids).values,
                                  flat_ids[(old_large | add).reshape(-1)]),
               "partition union exactly matches supplied candidate union")
    for key in ("flat_indices", "raw_residual", "refined_logit",
                "binary_candidate"):
        _check(torch_module.equal(unchanged_old[0][key], unchanged_old[1][key]),
               "original candidate values invariant to added-domain widening")
    _check(unchanged_old[0]["chunk_count"] == unchanged_old[1]["chunk_count"],
           "original chunk count invariant")
    for tensor, value_before, grad_before in zip(
            guarded, before_values, before_grads):
        _check(torch_module.equal(tensor.detach(), value_before),
               "input values are not modified")
        _check(torch_module.equal(tensor.grad, grad_before),
               "existing input/weight gradients are not modified")
    for mask, before in zip((old_large, add_small, add_large), before_domains):
        _check(torch_module.equal(mask, before), "domain inputs are not modified")
    checks.append("original_batch_invariance_disjoint_union_and_no_input_grad_mutation")

    def small_call(**overrides):
        arguments = {
            "torch_module": torch_module, "weight_tensors": zero_final,
            "F_original": small_F, "native_L": small_L,
            "originalLOG_domain": old_domain, "addedHalo_domain": add_domain,
            "refined_binary_rule": "strict_gt_zero",
        }
        arguments.update(overrides)
        return fixed_pr_candidates(**arguments)

    _expect_rejected(lambda: small_call(weight_tensors=zero_final[:7]),
                     "wrong weight count")
    wrong_shape = list(zero_final)
    wrong_shape[7] = torch_module.zeros(2)
    _expect_rejected(lambda: small_call(weight_tensors=wrong_shape),
                     "wrong weight shape")
    _expect_rejected(lambda: small_call(weight_tensors={
        role: weight for role, weight in zip(WEIGHT_ROLES, zero_final)}),
        "automatic checkpoint-key mapping")
    _expect_rejected(lambda: small_call(F_original=small_F[:255]),
                     "wrong F shape")
    _expect_rejected(lambda: small_call(F_original=small_F.double()),
                     "wrong F dtype")
    _expect_rejected(lambda: small_call(native_L=small_L.double()),
                     "wrong L dtype")
    _expect_rejected(lambda: small_call(originalLOG_domain=old_domain.float()),
                     "non-bool domain")
    _expect_rejected(lambda: small_call(addedHalo_domain=old_domain),
                     "overlapping domains")
    bad_L = small_L.clone()
    bad_L[0, 0] = float("inf")
    _expect_rejected(lambda: small_call(native_L=bad_L), "nonfinite L")
    bad_F = small_F.clone()
    bad_F[0, 0, 0] = float("nan")
    _expect_rejected(lambda: small_call(F_original=bad_F), "nonfinite F")
    bad_weights = list(w.clone() for w in zero_final)
    bad_weights[0][0, 0] = float("nan")
    _expect_rejected(lambda: small_call(weight_tensors=bad_weights),
                     "nonfinite weight")
    wrong_device_L = torch_module.empty((2, 4), device="meta")
    _expect_rejected(lambda: small_call(native_L=wrong_device_L),
                     "unmaterialized or mismatched device")
    _expect_rejected(lambda: small_call(refined_binary_rule="automatic"),
                     "implicit/invalid binary rule")
    _expect_rejected(lambda: small_call(return_merged=1), "non-bool option")
    _expect_rejected(lambda: fixed_pr_candidates(
        torch_module, zero_final, small_F, small_L, old_domain, add_domain),
        "missing explicit refined binary rule")
    for forbidden in ("GT", "K", "final_selected_set", "x", "y", "ROI"):
        _expect_rejected(lambda forbidden=forbidden: small_call(**{
            forbidden: object()}), "forbidden selection/coordinate argument")
    overflow_weights = list(w.clone() for w in zero_final)
    overflow_weights[6].fill_(torch_module.finfo(torch_module.float32).max)
    _expect_rejected(lambda: small_call(weight_tensors=overflow_weights),
                     "nonfinite arithmetic result")
    checks.append("finite_dtype_device_shape_overlap_rule_and_API_rejections")

    return {
        "status": "PASS_SYNTHETIC_ONLY",
        "draft_status": STATUS,
        "synthetic_cpu_seconds": time.perf_counter() - started,
        "checks": checks,
        "weight_tensor_count": 8,
        "parameter_scalars": 40497,
        "point_chunk_size": POINT_CHUNK_SIZE,
        "science_qualified": False,
        "actual_checkpoint_key_mapping_verified": False,
        "original_renderer_or_provenance_verified": False,
        "actual_exporter_binary_rule_verified": False,
        "native_query_embedding_or_cohort_identity_verified": False,
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="run CPU-only synthetic checks; no scientific files")
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("this unbound draft only has a --self-test CLI")
    if not __debug__:
        raise RuntimeError("optimized self-tests are rejected; omit -O/-OO")
    if not __import__("sys").dont_write_bytecode:
        raise RuntimeError("synthetic CLI requires python -B to disable bytecode")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Explicit CLI import only; importing this file does not import Torch.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = _self_test(torch)
    host_version = str(torch.__version__)
    differs = host_version != DOCUMENTED_SCIENCE_TORCH
    report.update({
        "host_torch_version": host_version,
        "documented_science_torch_version_not_verified_here":
            DOCUMENTED_SCIENCE_TORCH,
        "host_differs_from_documented_science_runtime": differs,
        "runtime_comparison": (
            "HOST_TORCH_DIFFERS_FROM_DOCUMENTED_SCIENCE_RUNTIME; "
            "synthetic host checks do not qualify Torch 2.0.1+cu118"
            if differs else
            "VERSION_STRING_MATCH_ONLY; scientific runtime not qualified"),
        "device": "cpu",
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "bytecode_disabled": True,
    })
    _check(not report["cuda_initialized"], "CLI did not initialize CUDA")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
