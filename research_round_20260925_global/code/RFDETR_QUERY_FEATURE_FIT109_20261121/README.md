# FIT109 query feature comparison

See the [preregistered experiment](../../RFDETR_QUERY_CHANNEL_RESIDUAL_FIT109_PRELOCK_20261120.md) and [verified outcome](../../RFDETR_QUERY_CHANNEL_FIT109_ACTUAL_20261121.md).

Run the worker scripts with the pinned CPU runtime (Python 3.10.13, PyTorch 2.10.0+cu130 and NumPy 1.26.4), CUDA visibility disabled, and interpreter flags -I -B -X utf8. Each script's --help describes its inputs. The ordinary code root is this repository's research code directory. The original HRSID SAR inputs, model weights, identities, native masks and point predictions come from the earlier fixed and independently accepted runs.

1. run_core4.py and verify_core4.py qualify the fixed first four original FIT images.
2. run_capture109.py reuses those four and observes the remaining 105. verify_capture109.py replays old identities and feature bytes after capture exits.
3. run_fit109_query.py first seals the original OOF per-candidate edit budgets. It then trains four equally sized heads over five source-held-out folds and evaluates every native candidate.
4. verify_fit109.py recomputes the complete result from the closed four-arm OOF files, with zero new head updates or detector forwards.

Closed measurements cover FIT only. HRSID DATA_ACCESS and the earlier benchmark protocol describe dataset intake and restrictions. Raw captures, labels, old weights and tool receipts remain private.
