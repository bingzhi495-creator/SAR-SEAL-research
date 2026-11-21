# HRSID official-split evaluation of frozen boundary refinement

This experiment measures whether observations from DEV705 recur when the unchanged boundary-edit policies are applied to the official HRSID PNG test allocation. The project previously used this split for model selection. This is a historical comparison on the official allocation, with a separate diagnostic scope from the original DEV705 advancement decision. Previous HRSID test scores only covered the native detector.

| Item | Frozen choice |
| --- | --- |
| Detector | RF-DETR 1.12.0, center-aligned target masks, seed 5202, E24, CPU float32 |
| Point heads | All-FIT109 refits 5217, 5218 and 5219, original 14 features and 200 optimizer steps per head |
| Policies | Native masks (N), same-head distance pairing (SF), independent selection at SF count (SU=F), natural-count independent selection (SU), large-head area-free writer at SU count (LW=S) |
| Main population | Original full test allocation, 1,962 images and 5,922 annotations |
| Sensitivities | 1,962 images/5,918 annotations after removing empty raster targets; 1,959 images/5,891 annotations after removing three exact-identity images, using the already frozen cohorts |
| Identity control | Exact native per-image replay, all mapped candidate IDs and native rank, score and class preserved, only segmentation bits change |
| Metrics | COCO segmentation AP, AP50, AP75, AP90, area-stratified AP and AR, maxDets 100; all values by head seed, followed by mean effects |
| Runtime | CPU forward, native mapping and refinement times measured separately; refinement timing covers the emitted diagnostic arms and is distinct from optimized serving time |

The main historical comparisons are SU minus N (within-carrier utility), SU=F minus SF (same head, same per-candidate edit count, both preserve area) and SU minus LW=S (same per-candidate total edit count). SU and LW=S differ in predictor and edit policy, as in the published DEV705 comparison. Report all seeds and signs, as well as all three population summaries. Record actual counts and area changes, including no-ops and worse seeds. The test labels stay closed until all inference rows finish and the emitted masks pass an independent identity, XOR and budget check. The original three sensitivity views are reused without any changes.

This protocol allows no target refit, threshold adjustment, favorable-seed selection, or removal of hard cases. It leaves the existing DEV705 gates and their decisions intact. Prior HRSID data exposure bounds the interpretation. A truly independent site or dataset with instance annotations and comparable strong alternatives remains necessary for a generalization claim.
