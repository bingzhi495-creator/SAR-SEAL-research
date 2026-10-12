# TGRS revision plan — 2026-09-05

## Scope and evidence boundary
Revise the complete manuscript against the supplied PDF, saved implementation, and the preceding audit. Preserve all reported numeric results; do not fabricate experiments, significance, novelty, latency, author information, or independent validation. This is a substantive working revision, not a declaration of submission readiness. Original files remain intact in the parent directory.

## Technical narrative
Proposed title: **Cluster-Support-Gated Mask Fusion and Boundary Refinement for SAR Ship Instance Segmentation**.
The paper studies how mask-space fusion, local boundary correction, and score calibration act on different errors of fixed, already trained segmentation carriers. Both use COCO initialization: HRSID uses auxiliary mask-quality supervision, while PSeg-SSDD uses the standard head. They are frozen during postprocessing. Soup and TTA are adopted candidate-generation techniques, not new methods. Candidate cardinality is not distinct-view support. Separate fields do not imply commuting operators or guaranteed AP gains.

## Structure
1. Abstract / introduction: operational SAR motivation; precise method; two bounded technical contributions and an empirical characterization; strongest reported within-protocol changes; disclose exploratory HRSID selection.
2. Related work: SAR instance segmentation; mask-space merging and voting (MNC, FCIS); boundary refinement and mask-quality scoring. Explain the limited combination/conditional-gate contribution without priority claims.
3. Method: true carrier and auxiliary training; soup and view inversion; score-ordered greedy seed clustering; cardinality gate and foreground tie rule; two-channel BPR and exact morphological guard; rerank features, target, blend and matching dependence.
4. Experiments: data and actual selection protocol before results; distinguish literature-reported comparisons from local reimplementations; preserve numeric tables but correct captions, cost attribution and significance; interpret diagnostic controls conservatively.
5. Discussion/conclusion: transfer scope, conditional bootstrap inference, split/selection limitations and compute; evidence-supported conclusions.

## Figures
Use the requested built-in imagegen skill for new conceptual architecture/mechanism illustrations. Keep recorded scientific masks, SAR imagery, PR curves and tables unchanged and traceable. The new conceptual overview must depict auxiliary training, frozen carrier, soup/TTA, gated fusion, BPR, reranking, and distinct train/inference paths. Do not fabricate measured heatmaps, SAR observations or performance curves.

## Main claims and evidence
| Claim | Evidence | Status / allowed wording |
|---|---|---|
| A cardinality-gated mask-space merge balances seed retention and voting in the evaluated pool | HRSID fusion ablation and exact code | Exploratory empirical observation; no general optimality or view-agreement interpretation |
| Local learned correction improves the reported high-IoU segmentation results | BPR ladder, PR curves and seed rows | Conditional on reported carrier and selection; boundary guard is a constraint, not a proven cause of gains |
| Score adjustment changes ranking without altering masks | Reranking implementation and progressive table | Algebraic field preservation; no commutativity or guaranteed AP improvement |
| Transfer is promising on PSeg-SSDD | Frozen settings and reported transfer rows | Dataset-specific; sparse inshore CIs cross zero for some stages; no universal/SOTA claim |
| Carrier architecture preservation is a general law | Existing modified carrier and confounded controls | Unsupported: remove |
| All results are independent held-out confirmation | Test-set sweeps in original manuscript | Unsupported: explicitly exploratory; independent confirmation remains required |

## Acceptance
Full IEEE LaTeX compilation with resolved references; complete visual page review; 150–250 word abstract; no invented author affiliations; generated images inspected and copied locally; all manuscript numbers preserved with provenance scope. Provide a separate confirmatory experiment plan and revision/change report in Chinese. No submission or upload is authorized by this task.
