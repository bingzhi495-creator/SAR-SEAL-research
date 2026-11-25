# Decoded image exposure for the frozen SRNet carrier

The original SRNet E24 acceptance binds checkpoint `8998d0f6…` to SSDD FIT741. Its closed ledger covers 24 epochs, 4,464 optimizer updates, 17,856 consumed visits and every one of the 741 logical FIT rows. Each FIT JPEG was reread from the pinned archive and compared byte for byte with the original training layout. Both embedded and external FIT manifests agreed with the original independent layout acceptance. The 74 Instance-RSDD images were reread in the original accepted order after binding the image-only intake manifest and natural exit. Every reserve JPEG and canonical RGB digest matched the intake.

| Quantity | Count |
| --- | ---: |
| Logical FIT images / unique decoded FIT images | 741 / 730 |
| Reserve images / unique decoded reserve images | 74 / 74 |
| Reserve author parents | 13 |
| Cross-set matching decoded identities / logical image pairs | 0 / 0 |

Pillow applies EXIF orientation before converting to native RGB. Whole-frame equality requires equal dimensions and SHA-256 of interleaved RGB bytes. The zero crossings qualify absence of exact decoded whole-frame duplicates. Overlapping geographic regions, physically shared acquisition products and uncertainty from different crops or processing remain outside the verified scope. The worker parsed neither label set, opened no predictions or checkpoint payload, and computed no scores or model outputs. No per-image records are published.

The observer recorded natural exit 0 and verified the worker footer and output digest. The complete aggregate is in [RESULT_SAFE.json](RESULT_SAFE.json) and the observed terminal in [TERMINAL_SAFE.json](TERMINAL_SAFE.json). The pinned inference and evaluation plan is separate from this input audit.

To reproduce with the original accepted local artifacts:

```powershell
python -B -m unittest -q test_rgb_audit
python -B audit_rgb.py --acceptance <original-fit-acceptance.json> --training <original-SRNet-training-root> --fit-stage <closed-FIT-image-layout-root> --fit-manifest <original-portable-FIT-manifest.json> --fit-archive <original-FIT-archive.zip> --intake <closed-reserve74-intake-root> --protocol PRELOCK.md --protocol-sha256 767be6156dca683b556a0dbadccca838faf89a7f32a70961c69a2bfcc12c16e8 --output <new-empty-output>
```

