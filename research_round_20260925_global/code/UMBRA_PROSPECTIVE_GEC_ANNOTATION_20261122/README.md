# Umbra prospective annotation feasibility

[Prospective protocol](../../UMBRA_PROSPECTIVE_GEC_ANNOTATION_PRELOCK_20261122.md) · [Frozen cohort decision](../../UMBRA_PROSPECTIVE_GEC_COHORT_LOCK_20261122.md)

Use `select_cohort.py` for metadata qualification under the frozen historical ten-port and pilot exclusions. The coordinator manifest pins the public GEC object list, ETags and object lengths. `native_windows.py` retrieves a single exact native window from each pinned object. `observe_native.py` verifies its process exit and result footer. `verify_native_windows.py` independently checks the closed image arrays and geospatial composition. Read each script's `--help` for the file roles. Keep the source mapping separate from readers.

`build_packets.py` reads the observed successful intake, checks the anonymous native PNGs, and joins a frozen roster into the offline browser application. It creates a ZIP for each reader. `REVIEWER_GUIDE.md` covers the reviewer steps. `annotate.html` is the template. The builder embeds the verified roster and reviewer identity in the final files.

`validate_reviews.py validate --roster ROSTER.json --annotation SAR_ship_review_A.json` checks a single return. `validate_reviews.py agreement --roster ROSTER.json --a SAR_ship_review_A.json --b SAR_ship_review_B.json` compares the two separate returns. It reports aggregate counts and anonymous pairing IDs for reconciliation. Consensus requires a separate adjudication step.

The qualified path uses tifffile, NumPy, Pillow, Shapely and SciPy. The editor uses modern Edge or Chrome and local file input. Test fixtures are generated separately from author imagery.
