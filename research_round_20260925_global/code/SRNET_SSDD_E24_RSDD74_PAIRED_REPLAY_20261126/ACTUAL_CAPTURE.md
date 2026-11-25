# The complete 74-image SRNet native capture

The closed project PSeg-SSDD FIT741 E24 model produced the candidate stream for all 74 original Instance-RSDD reserve images. The pinned author le90 implementation used the 608-pixel input pipeline, 0.30 score threshold and 0.10 rotated NMS threshold. Every retained candidate used its original rank, score and native oriented box. Capture was image-only and used no reference annotations, model selection, optimization or mask editing. [Registered design](PRELOCK.md) · [Earlier projection pilot](../SRNET_SSDD_E24_RSDD74_NATIVE_PARITY_20261126/ACTUAL.md)

| Sensor | Images | Original candidates | Images without candidates | Exact probability-to-mask checks |
| --- | ---: | ---: | ---: | ---: |
| Gaofen-3 | 42 | 41 | 3 | 41 |
| TerraSAR-X | 32 | 48 | 0 | 48 |
| Total | 74 | 89 | 3 | 89 |

Thresholding the 512 × 512 projected float32 probabilities at (p>0.50) exactly reproduced every original COCO RLE. The capture retained all three rows without candidates. A separate closed-stream readback verified all 74 compressed-array digests, 89 candidate masks and 23,330,816 candidate-grid positions. The observed capture exited with status 0 after 126.60 seconds, using four CPU processes with one numerical thread each. These observations establish input and geometry parity for the registered comparisons. [Capture summary](CAPTURE_RESULT_SAFE.json) · [Observed termination](CAPTURE_TERMINAL.json) · [Reproduction](REPRODUCE_CAPTURE.md)
