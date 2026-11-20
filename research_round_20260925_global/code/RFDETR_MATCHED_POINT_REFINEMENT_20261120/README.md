# RF-DETR native posterior and exact-count policy increment

Runnable CPU synthetic first increment; no model loader, dataset access, training,
DEV/TEST evaluation or execution binding. Original sources and archives unchanged.

`adapter.native_probability(torch, logits, original_hw, stock=None)` uses actual
PyTorch FP32 CPU `interpolate` on logits, bilinear/align_corners=False. Stock is
strict `rendered_logits > 0`; posterior is sigmoid afterwards. Optional stock
must match exactly. Thus exact-zero ties and positive logits whose sigmoid rounds
to 0.5 do not silently redefine the native mask. Shape is generic for small
witnesses; the future RF caller must require native144 logits/800 canvas/Q164.

`readouts(stock, pairs, q_small, q_large, q_native, cap)` takes a flat bool native
mask, frozen disjoint integer (add,drop) pixel pairs and aligned posteriors. It
returns N/SF/SU=F/SU/LF=S/LU=S/LW=S/NP=S with actual masks, flipped indices,
add/drop counts and area change. Caller supplies the previously locked cap,
including original area/10. Fixed and independent readouts retain the original
D-FINE pair ordinal tie rule. LW combines both signs, uses signed q−0.5/0.5−q,
ties by native flat pixel index, and forces exactly twice SU's swap count even
for nonpositive gains. LW nonpositive-flip count is reported; it is an ordinary
matched-total pointwise baseline, not PointRend replication.

`replace_segmentation(record, segmentation)` changes only segmentation on a
copy; scores, ranks, query/category identities and other metadata stay intact.
Encoding/raster dimensions remain the caller's explicit job.

Tests were written before each implementation and run with real CPU Torch.
The first RED was missing adapter; the second missing policy functions. A
nonmonotone paired-drop fixture then exposed a changed independent tie rule;
the final implementation preserves the published original pair ordinal.
Run `python -m unittest -v test_adapter.py`. Tests use tiny synthetic tensors
and masks, not mock tensors or actual carriers. The third-stage real CORE36
interface, 14-feature construction, scalar training heads and artifact/identity
admission are deliberately still unimplemented; this increment does not claim
actual native capture or candidate population parity.
