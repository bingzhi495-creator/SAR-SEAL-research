# Probability quality follow-up — SOURCE / public evidence only

Read 2026-10-11 while the original frozen-readout GPU worker is live. No experiment payload, checkpoint or numerical scientific program was accessed for this note.

Primary source: [Mur-Labadia et al., arXiv:2605.03614v1](https://arxiv.org/html/2605.03614v1), submitted 2026-05-05. The author manuscript names Pattern Recognition; publisher acceptance and final version have not been independently verified here.

Section 4.1, equations 9–14, evaluates probabilistic masks against ground truth using class probability and foreground/background logarithmic penalties, then Hungarian assignment. This is an evaluation metric requiring ground truth, not a deployable confidence estimator. Section 4.2 evaluates calibration using ECE and sparsification/Brier measures. The paper's Bayesian sample/ensemble construction differs from our single deterministic residual head.

Implication for our study: probability-derived quality, separate foreground/background penalties and calibration evaluation already have direct precedents. PMQ is not algebraically identical to our soft Jaccard ratio, and no identity is claimed. Our ratio is also subject to the previously recorded Dice/Jaccard equivalence and ratio-of-expectations caveat. A single sigmoid output does not by itself establish a Bayesian posterior or separate epistemic/aleatoric uncertainty.

Keep the current prelocked A/B joint long-training recipe. Do not cancel or alter it based on literature terminology. Its first question is whether a residual readout improves masks under the same native supervision and additional matched point loss. An AP gain cannot establish calibration, isolated query conditioning or new probability theory. If the long comparison supports efficacy, distinguish geometry and ranking gains, retain a feature/capacity control, and separately test score-quality error and detection calibration using fixed, explicitly defined targets and all-query populations. No new metric selection or superiority claim is made in this note.

O2Former and DACF-Net were already in the local literature record; today's search does not count them as new papers. A sea-state guided panoptic title was found only through an aggregator; without a primary source it remains an unverified lead and is excluded from technical conclusions.

