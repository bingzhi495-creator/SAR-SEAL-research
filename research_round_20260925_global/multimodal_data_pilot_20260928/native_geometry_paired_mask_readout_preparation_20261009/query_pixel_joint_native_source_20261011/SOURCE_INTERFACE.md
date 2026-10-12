# Joint native tensor interface

This is SOURCE preparation, not scientific admission or qualified end-to-end
integration. The old readout, frozen adapter and all SCI-used files remain
immutable. The caller supplies the unchanged `QueryPixelPosteriorReadout`
with eight parameter tensors / 9441 scalars; this adapter does not construct it.

`project_live_native_fields(readout, F256, e0, M200, actual_img_shape,
original_shape, renderer)` requires ambient autograd and finite FP32 tensors on
one device. Shapes are F256 `[1,256,200,200]`, e0 `[1,100,256]`, and M200
`[1,100,200,200]`. All three inputs must require gradients. No input is detached,
modified, or copied to another device.

The caller-bound `renderer(masks, identity)` must apply the original bilinear
200 -> 800, actual image crop, bilinear original raster sequence with
`align_corners=False`, returning `[C,H,W]`. It must preserve autograd for both
the projected feature field and native logits. Shape, finite-value and
`requires_grad` checks do not prove that a custom renderer or supplied native
tensors have the claimed upstream provenance.

The exact four-field packet is:

- `projected_pixel`: live `[1,16,H,W]`, projection of F256 followed by renderer.
- `projected_query`: live `[1,100,16]`, projection of e0 in original query order.
- `native_logits`: live `[1,100,H,W]`, renderer of M200.
- `contrast`: live `[1,100,H,W]`, sigmoid(native logits) minus the full-query
  3x3 average, padding 1, `count_include_pad=False`. Compute before point gather.

`matched_joint_point_loss(readout, fields, query_ids, gt_masks, point_indices)`
requires all four fields live and all eight readout parameter tensors trainable.
Unique int64 query IDs `[G]` are actual ordinals 0..99, aligned with original
bool GT `[G,H,W]` and int64 original flat indices `[G,12544]` on the same device.
Repeated point indices retain their multiplicity. It uses the readout's
`posterior_logits = L+u` with stable BCEWithLogits, pixel mean then GT mean.
It does not train the bounded tanh logits, sample points, create matches, add
unmatched-query loss, apply accumulation scaling, or add native-stage loss.

Empty G returns an exact finite zero connected through empty slices of all four
fields and all eight readout parameters. This avoids full-raster reductions
and their possible overflow. Gradient connectivity is required; a gradient's
numerical value may be zero. At cold final zero, direct F/e0 projection paths
and upstream head gradients are zero, while the L+u loss already supervises
native L. F/e0 can also receive gradients through an actual native M=e0.F graph.
Independent F/e0/M fixture leaves do not establish that native graph identity.

The future entry must bind one live native forward, original native30 losses,
current final-stage dynamic Hungarian assignment and raw annotation ordering.
The existing detached teacher wrapper, stripped GT samples, cached assignments
and cached scalar targets are not an end-to-end training entry. All100 query
and class identities remain the caller's native identities.

The caller must separately preserve the intended typed BatchNorm policy and
native paramwise optimizer groups, add/read back the eight readout parameters,
and prelock loss weighting, LR schedule, private RNG and global accumulation.
Sum native and auxiliary losses before one backward through their shared graph.
Backbone checkpointing does not by itself bound the added original-raster graph;
actual native forward/backward/update memory and gradients require qualification.
The readout's dense API remains inference-only and is not a training shortcut.

The sibling native-point objective is the same-sampled native-L BCE control.
Any observed gain over native30 alone could be due to added supervision, so it
does not establish the residual conditioning mechanism. Simple unfreezing is
not innovation; no calibration, improvement, convergence, GPU-fit or publication
claim follows from these interfaces or a synthetic fixture.
