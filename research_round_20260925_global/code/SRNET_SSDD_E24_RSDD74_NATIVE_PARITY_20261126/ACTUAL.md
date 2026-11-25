# Native SRNet probability projection check

We ran the project’s source-trained PSeg-SSDD FIT741 E24 SRNet on eight prespecified native Instance-RSDD chips, one from each of eight author parents. The unmodified author inference used the SSDD le90 configuration at 608 × 608 and returned 512 × 512 native masks. The instrumented repeat independently recreated the mapping from the 28 × 28 pre-paste probabilities to the native grid. [Frozen design](PRELOCK.md) · [Original author implementation](https://github.com/XIDIAN212Lab/SRNet/tree/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6)

| Sensor | Chips | Native candidates | Nonempty masks | Independently reproduced masks |
| --- | ---: | ---: | ---: | ---: |
| Gaofen-3 | 4 | 4 | 4 | 4 |
| TerraSAR-X | 4 | 4 | 4 | 4 |
| Total | 8 | 8 | 8 | 8 |

The two passes yielded identical boxes, scores, order and native COCO RLE. Thresholding the independently projected probabilities at (p>0.50) reproduced every mask across 2,097,152 candidate-grid positions. The synthetic fixture separately checks equality at the half-probability boundary. The observed worker exited naturally after 89.96 seconds. The pilot used zero reference files, optimizer updates, pixel edits or threshold and model selection decisions. It establishes coordinate parity for the tested source-trained model, with accuracy unmeasured. [Aggregate result](RESULT_SAFE.json) · [Observed termination](TERMINAL.json) · [Reproduction commands](REPRODUCE.md)
