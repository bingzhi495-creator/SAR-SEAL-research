# SAR-SSP author val frozen native capture

Admit the 232 500 × 500 JPEGs with twelve-digit names in `datasets/SAR-SSP/val2017` from the exact 866,821,756-byte author archive, SHA-256 `23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5`. Process all admitted images in file name order, without target annotations, image-quality exclusions, or source-product metadata. Bind the image-only manifest SHA-256 `16c8e6b7655e51f208c3a9708acd419a60126e55a9075cdea7d4b9e4edbb9425` and verify each native RGB decode before inference. Author image identifiers are numeric file stems.

Reuse the HRSID center5202 E24 RF-DETR checkpoint SHA-256 `011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11`, the accepted B5 loader and val576 preprocessing, the 164-query joint top-100 order, class-zero ship mapping, scores, and native mask postprocessing. Native-canvas logits follow the original bilinear half-pixel convention and exact stock raster check. Preserve every mapped output and the original area rule [32, 1024) for exportable probability planes.

Observe the frozen capture exit and independently read back the stock mask and candidate geometry on CPU. This phase establishes inference outputs and internal parity. Source-product independence, target scoring and point editing are outside the capture phase.
