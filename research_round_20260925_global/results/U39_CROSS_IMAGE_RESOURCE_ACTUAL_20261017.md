# U39 disposable cross-image resource result

Actual original single GPU leaf and Root terminal/source/report metadata intake. Neural, gradient, optimizer and CPU-transfer checks remain worker evidence; no Root tensor or pixel replay.

- Full metadata population: 741 images / 1661 matched instances. Actual tensor population: 2 images.
- Three updates per arm, six returned disposable updates total; formal budget credit 0.
- Matched-GT visits: 29 per arm, 58 total including the repeated small image; these are visits, not independent GT counts.
- Common states: 104 = 68 parameter tensors + 36 BN buffers; initialization seed 202610173101.
- FP32 RGB400 / canvas800 / query chunk1 / canonical pinned CPU saved activations / fixed AdamW.
- Original leaf elapsed seconds: 40.610000; probe seconds: 35.076812.
- Maximum actual stage/final-interval allocated CUDA peak bytes: 2047209472.
- Worker peak RSS bytes: 13980127232; worker minimum available RAM bytes: 21861564416; original leaf minimum available RAM bytes: 21831278592.
- Both arm and final allocated/reserved CUDA releases: 0 / 0.

| Arm | Step | G | Original shape | CUDA allocated peak bytes | Post-image RSS bytes | Read seconds | Train/check seconds | Image-scope seconds | CPU pack event bytes | Restore event bytes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| PLAIN | 1 | 1 | 190x303 | 1294577664 | 7202529280 | 0.202728 | 2.223030 | 2.429338 | 4098603776 | 4098603776 |
| PLAIN | 2 | 27 | 327x489 | 1889022976 | 7207161856 | 0.261954 | 8.991595 | 9.259421 | 4098603776 | 110662301952 |
| PLAIN | 3 | 1 | 190x303 | 1674953728 | 7207321600 | 0.116940 | 0.781286 | 0.900216 | 4098603776 | 4098603776 |
| FULL | 1 | 1 | 190x303 | 1375999488 | 13824610304 | 0.116656 | 2.892338 | 3.012517 | 7638842304 | 6807355584 |
| FULL | 2 | 27 | 327x489 | 2047209472 | 13830332416 | 0.263853 | 15.890451 | 16.160266 | 7638842304 | 183798600768 |
| FULL | 3 | 1 | 190x303 | 1807920640 | 13830688768 | 0.117759 | 1.352035 | 1.471805 | 7638842304 | 6807355584 |

CPU save/restore counters over all six updates: {"cpu_saved_tensor_count": 3576, "cuda_restore_calls": 32219, "cuda_source_pack_calls": 3576, "pack_calls": 3576, "packed_bytes_total": 35212338240, "pinned_nonempty_pack_calls": 3348, "restored_bytes_total": 316272821440, "unpack_calls": 32219}.
Cumulative saved/transfer events include duplicates and are not unique activations, resident pinned memory, RSS peaks or speedup evidence. The short resource probe establishes neither long-run leak absence nor a complete 50-epoch budget. No main-training eligibility, AP/DEV/TEST, utility or innovation conclusion is granted. No checkpoints or weights were saved.
