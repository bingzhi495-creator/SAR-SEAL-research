Dear Editor-in-Chief,

Please consider our manuscript “NORA: Native-Grid Occupancy Ranking and Allocation for SAR Ship Instance Segmentation” for IEEE Transactions on Geoscience and Remote Sensing.

A small shift of a SAR ship outline can change a large fraction of a narrow mask. We study which cells to add and remove on the native image grid after detection. A compact occupancy head uses fractional polygon–cell coverage as supervision. With detector candidates, confidence scores, editable cells, and realized addition and removal counts held fixed, we compare distance-first pairing, exact re-pairing within the original maximum displacement, and independent ranking of the two sides. We also examine how redistributing a fixed total number of flips affects overlap when two author references differ.

On HRSID DEV705, bounded re-pairing improves mask average precision by 0.475 percentage points over distance-first pairing, and independent ranking adds 0.431 points over bounded re-pairing. On the 74-chip historical Instance-RSDD reserve, independent ranking improves both mask average precision and continuous intersection over union relative to bounded re-pairing in the frozen RF-DETR and D-FINE pipelines. The pipelines share the same target cohort. The paired intervals leave the additional effect of independent ranking on one-pixel Boundary IoU uncertain. Varying the signed allocation improves both SSDD development references, but frozen transfer to SAR-SSP increases excess foreground and lowers mask average precision. The article uses these results to identify where a native-grid edit rule helps and where the observed gains stop.

Earlier review versions of this work, evaluation code, and aggregate results are available in the [SAR-SEAL research repository](https://github.com/bingzhi495-creator/SAR-SEAL-research). [AUTHOR_INPUT_NEEDED: confirm the prior posting and publication history and add the applicable disclosure and exact version links.] [AUTHOR_INPUT_NEEDED: confirm all authors’ approval and the statement on exclusive consideration. Resolve any related publications, conflicts, or required permissions before submission.]

Thank you for your consideration.

Sincerely,

[Name of corresponding author]  
[Affiliation]  
[Correspondence email]  
[On behalf of all authors]
