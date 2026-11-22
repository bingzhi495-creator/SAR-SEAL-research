# Umbra window sources locked before image intake

Metadata-only intake actually exited 0. The [predeclared rule](UMBRA_PROSPECTIVE_GEC_ANNOTATION_PRELOCK_20261122.md) sampled 64 of 999 catalog tasks. Sixty unique records passed metadata and target-IPR checks. Twelve acquisition windows were selected, six in each target-IPR band. The closest selected center pair is 176.185 km apart, and the smallest separation from any of the 120 archived ten-port footprint centroids is 221.758 km. The selected footprint pairs have zero intersections. These distances describe metadata, with native pixel spacing measured during image intake.

The exact ordered coordinator-only manifest (`UMB-001` through `UMB-012`) was sealed at SHA-256 `24d08f3bf1ecb08212f78e5a3cfa8c709ef9a2194e9d345f5cb04a1f9f48c29e`. Each image object is bound by its original author URL, ETag and length. The ordered public-task-list digest is `d2d343bd49e0cfc88c61f43fd7bbf923846e7f4148ad5ca854289e74ad034552`. This selection has zero pixel reads, zero labels and zero detector forwards. [Verified selection aggregate](results/UMBRA_PROSPECTIVE_GEC_METADATA_SAFE_20261122.json)

The first local worker stopped before network access because a validated footprint centroid crossed a strict native-number conversion. The corrected worker kept the same selection rules and used a fresh destination.
