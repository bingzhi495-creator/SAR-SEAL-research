# D-FINE FIT109 direct swap-utility supervision

Three identical 1,321-parameter independent point heads were fitted on the same frozen D-FINE FIT109 native point pairs, with source-disjoint evaluation folds, original batch draws and 200 AdamW updates of 64 pairs per arm and fold. The heads used per-point BCE, per-point Brier or mean squared direct pair utility error. If e is predicted minus exact continuous cell occupancy, the direct objective is 0.5(e_add−e_drop)^2, or Brier minus e_add·e_drop. All heads ranked identical action pairs by predicted p_add−p_drop and selected the original positive-only capped swaps. Original BCE AP, geometry, budgets, training and strict transitions replayed exactly.

| Arm | Pooled mask AP | AP75 | AP90 | AR100 | Source-equal continuous IoU | Fixed strict .90 | Natural swaps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 49.3251 | 47.3883 | 1.5337 | 53.1250 | 80.0492 | 7 | 0 |
| BCE | 50.0428 | 48.5535 | 1.5501 | 53.7097 | 80.5044 | 6 | 20,809 |
| Brier | 50.0724 | 48.5535 | 1.5501 | 53.7500 | 80.4866 | 6 | 20,702 |
| Direct | 50.1396 | 48.8093 | 1.5501 | 53.7500 | 80.5388 | 8 | 26,056 |

Direct versus raw gained +0.8145 AP and +0.4896 IoU with paired 95% interval [0.3580, 0.6290] across 412 valid original stock-.50 matches and 93 source groups. Against BCE and Brier, AP differences were +0.0969 and +0.0672, and IoU differences were +0.0344 [−0.0274, 0.0935] and +0.0522 [−0.0005, 0.1053]. Direct retained five of the seven original strict .90 identities and gained three new ones. The two original losses fail the zero-loss gate even though the aggregate strict count increased.

The secondary common-min comparison ran 17,908 swaps per trained arm on the same 4,015 candidates. Pooled AP was 50.0759 for BCE, 50.0824 for Brier and 50.0284 for direct. Direct source-equal IoU gains over BCE and Brier were +0.0304 [−0.0124, 0.0746] and +0.0371 [−0.0013, 0.0783]. Strict identity counts and transitions were identical to the corresponding natural arms.

Across all 13,076 identical available valid OOF action pairs, direct utility RMSE was 0.4172, versus 0.4226 for BCE and 0.4237 for Brier. On the common-count valid actions, direct selected 207 negative exact-gain actions among 1,728 versus 228 for BCE and 226 for Brier. These descriptive readouts and the natural gain over raw do not establish a meaningful advantage over the two trained controls. The locked advance gate stopped.

All 10,900 candidates retained original score, rank and area. The three arms performed 3,000 point-optimizer updates in total. Detector training included this FIT cohort. No detector forwards or updates, DEV, official TEST or RD were involved.

The direct loss keeps the pointwise Brier diagonal terms and changes the interaction and curvature. In the mean-and-contrast basis, its contrast penalty doubles and the common-mode penalty vanishes. The practical comparison therefore identifies the complete loss change. Equal-area swap utility is additive in exact intersection. Its conditional expectation is also the difference of two conditional point occupancies when they share the same conditioning information. Action-wise changes in segmentation IoU have prior uses in interactive or sequential segmentation. The current experiment isolates supervised native-cell swaps on frozen SAR instance candidates. [SegAgent](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_SegAgent_Exploring_Pixel_Understanding_Capabilities_in_MLLMs_by_Imitating_Human_CVPR_2025_paper.html) · [Sequential IoU reward](https://pmc.ncbi.nlm.nih.gov/articles/PMC8037138/) · [PointRend](https://arxiv.org/abs/1912.08193)
