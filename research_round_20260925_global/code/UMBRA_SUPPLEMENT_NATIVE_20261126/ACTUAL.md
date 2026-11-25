# Supplemental Umbra original-window acquisition

All **24** prospectively locked Umbra GEC windows passed native extraction and independent byte readback. The readback recovered **96** original uncompressed TIFF tile hashes from the saved PNG arrays. The 24 decoded arrays were distinct. No window failed and no network request reported a transport error. The original 24 positions are available for the separately locked blind review and frozen detector workflow.

The direct-range reader transferred or reserved 26,738,688 response bytes against its 64-MiB limit. Each scene supplied the same predetermined 1,024 × 1,024 closest-to-center native window. Projected native pixel steps ranged from 0.191 to 1.419 m. The acquisition and independently observed readback exited naturally with code 0 after 213.0 s and 0.62 s. Neither phase accessed annotations or ran a detector.

| Phase | Aggregate receipt | Observed terminal SHA-256 |
|---|---|---|
| Native extraction | [RESULT_SAFE.json](intake/RESULT_SAFE.json) | `b8e9e548789cbb7fe945d7ae3d58e14025b815cbdf4751ceed9c5db9cb2bfaea` |
| Independent readback | [RESULT_SAFE.json](readback/RESULT_SAFE.json) | `4357b47afa3683a82762f3301701fc2fcd1b2c1cf80688f344aa70226b91125f` |

The exact extraction and readback result hashes are `5d7bdee2634960afc54889c08172d96d2b7fe070887f4ab5e3473b960890827f` and `f4b2f907c97170515836f0be1c2373e0988bc4a90f722f5f1bdee07bbf3837c1`. The [metadata cohort lock](../UMBRA_SUPPLEMENT_METADATA_20261126/LOCK.md) was verified at commit `8c877983a1994765404cfccec7d8eb6871e819d1` before native acquisition began. The [technical protocol](PRELOCK.md) and [reproduction instructions](README.md) bind the exact ranged reader and independent verifier.

These results establish that the prospectively selected images are available at their original grid. Hull availability, reviewer agreement, edit coverage, and transfer performance require the separate reference and frozen evaluation stages.
