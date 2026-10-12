I have everything needed. The audit is overwhelmingly clean — the key finding for a pre-submission report is that the paper body and tracking files are internally consistent, with only one genuine numeric error (the -13.7 vs -13.76→-13.8 delta), several "seed-specific number presented as if it were the mean" framing risks, and two stale-artifact / unverifiable items.

Here is the consolidated report.

---

# SEAL Pre-Submission Numerical Audit — Consolidated Report

**Scope:** 40 claims re-derived (logs, `seal_metrics.json`, independent pycocotools/COCOeval recomputes, bootstrap reruns). Headline result is sound: **A0R-refine = +2.6 AP75, two-seed robust, both seeds positive.** One genuine numeric fix, three framing fixes, two unverifiable/stale items.

---

## (A) CONFIRMED — matches, no action

**PSeg AP75 main line (triple-checked: log ↔ recompute):**
- A0 seed0 0.615, seed1 0.623 → mean **0.619** ✓
- A0R-refine seed0 0.643, seed1 0.646 → mean **0.645** ✓
- Δ **+2.6 pp, seed-robust** (seed0 +2.84 / seed1 +2.30, both positive) ✓
- Inshore AP75 A0 0.341 → A0R 0.387, **+4.6** ✓
- A0R seed0 overall AP 0.526, AP_s 0.506 ✓

**M4 inshore bootstrap:** point +4.6, 95% CI **[-0.5,+9.5]**, P(Δ>0)=**0.96** — exact deterministic match (B=2000, seed 0). CI straddles 0; headlining overall +2.6 is the correct call. ✓

**HRSID cross-dataset (M1) — every cell matches log/json/recompute:**
- A0 AP/AP50/AP75/AP_s/AP_m/AP_l = 0.348 / 0.625 / 0.378 / 0.339 / 0.466 / 0.126 ✓
- A0R = 0.351 / 0.633 / 0.383 / 0.344 / 0.448 / 0.124 ✓
- Deltas AP75 +0.5, AP_l −0.2, AP_m −1.8 ✓

**Large-instance counts (these underpin the "−13.7 is noise" argument):** HRSID **76** large GT, PSeg **2** large GT — both confirmed two ways (area field AND RLE recompute). ✓

**Detection seeding:** AR@100 **+1.85** (s0 +1.8/s1 +1.9), AR small **+2.1** (s0 +1.9/s1 +2.3) ✓. Seed/source provenance (s0=`seed_pred_gate`, s1=`multiseed`) confirmed via checkpoint-load lines. ✓

**Protocol:** HRSID leakage 92.7% any-overlap, 10.8% ≥50%, 1 full duplicate ✓. BIoU degeneracy HRSID 91.5% ✓. **G1 negatives** HRSID +0.0019 / PSeg −0.0022 (both < +0.03 gate) ✓.

**Model-selection integrity:** best ckpt = iter_8352 (final iter) in all 4 runs, selected on overall segm_mAP not AP75 — **no AP75 cherry-picking / no leakage.** ✓

---

## (B) DISCREPANCIES — fix before submission

### B1. [GENUINE NUMERIC ERROR] PSeg AP_l delta is **−13.8**, not −13.7
- **Correct value:** A0 0.7134 → A0R 0.5757 = **−13.76 pp → rounds to −13.8** (endpoints 0.713/0.576 are correct; only the delta's last digit is wrong — claim appears truncated, not rounded).
- **Appears wrong (−13.7) in — these all need updating:**
  - `paper/seal.tex` line **46** (Intro body) — *the submission text*
  - `paper/seal.tex` line **100** (comment)
  - `PROGRESS.md` lines 7, 73; `STATE.md` lines 16, 30, 49
  - `runs/refine_gate/细化头_裁决.md` (lines 14, 26), `runs/refine_gate/细化头机制分析.md` (line 30)
  - `runs/a0r_refine_hrsid/HRSID细化头裁决.md` (lines 21, 44)
  - `docs/paper_draft_v1.md` (lines 37, 57, 328, 352…), `docs/experiment_gaps.md` (26, 41), `docs/grounding_positioning.md` (61, 89), `POSITIONING.md` line 33
- **Action:** global replace −13.7 → −13.8 for *this specific delta*. Low risk to the argument (it's being dismissed as noise either way), but a reviewer will diff 0.713−0.576 and catch −13.7. **Note:** reviewer file `paper/review_summary.md` (M18) recommends leading with the cheaper **within-PSeg sign-flip** evidence (seed0 −13.8, seed1 +2.7, n=2) — adopt that and the exact magnitude matters even less.

### B2. [FRAMING] mAP detection delta "−0.1" is the **seed1-only** number, not the mean
- **Correct:** seed1 = −0.1 (0.634−0.635); **seed0 = +0.4** (0.642−0.638); 2-seed mean = **+0.15**.
- **Appears as bare "mAP−0.1" in:** `PROGRESS.md` line 8; `STATE.md` line 18 ("mAP平/mAP flat"). `seal.tex` line 48 already says "no mAP cost" (safe).
- **Action:** keep the qualitative "mAP flat" (honest, both interpretations agree), but where a number is given say "flat (seed1 −0.1, seed0 +0.4)" or "+0.15 mean." Do not present −0.1 as the headline mean.

### B3. [FRAMING] Precision "0.913 → 0.931" is a **single-seed (seed1) working point**, threshold-shift-confounded
- **Correct:** these are seed1-only P values (Table shows `--` for DINO-s0/Seed-s1). Direction is 2-seed-robust (seed0 also rises 0.943→0.955), but the cited pair is one seed and confounded by the conf-threshold shift (0.34→0.42) the verdict itself flags.
- **Appears in:** `PROGRESS.md` line 8; `runs/multiseed/检测播种多seed裁决.md` lines 9, 16. `seal.tex` does **not** cite the raw pair (line 48 says "no mAP cost" + random-seed control) — good, keep it out of the body.
- **Action:** this is already an open reviewer item — `paper/review_summary.md` line 89 flags the "more discriminative not noisier" precision claim as causally unsupported / single-seed. Label the pair as single-seed + working-point in the verdict md, and do not let it migrate into Abstract/body as causal.

### B4. [STALE ARTIFACT — minor] PSeg BIoU degeneracy: paper says **67%**, but on-disk json still stores only the stale **89%**
- **Correct:** 67% at per-image standard_d (recompute 66.3% over all 546 anns; seal_eval's preds-subset gives ~67% — within rounding). The 89% (0.8901) is reproducibly the **fixed-d=23 artifact** (wrong: d=23 is HRSID's value, PSeg diag→d≈12).
- **State:** `seal.tex` line 50 and line 109 correctly say 91.5%/67%. The correction lives in recompute logic + `runs/a0_pseg_seed0/A0完整指标_G4裁决.md`. **But** `runs/a0_pseg_seed0/seal_metrics.json` still has only `boundary_degeneracy_frac['23']=0.8901` and **no `_std`/per-image key** — anyone reading that json sees 89%.
- **Action:** re-dump `seal_metrics.json` with a `boundary_degeneracy_frac_std` (per-image d) key = 0.66–0.67 so the artifact (0.8901) and the corrected value coexist explicitly. Paper text is already correct; this is artifact hygiene to prevent a reproducer landing on 89%.

---

## (C) UNVERIFIABLE — needs an artifact that does not exist yet

### C1. HRSID A0R-refine in-training validation has **no copypaste log line** (validation crashed mid-run)
- The final in-training val **OOM-crashed** at val 1300/1962 (`runs/a0r_refine_hrsid/.../20260617_234003.log` truncated; no copypaste ever logged). All reported HRSID A0R numbers (0.351/0.383/0.124 etc.) come from a **separate offline `test.py` eval** (`preds.pkl` dated 2026-06-18 13:45, after the 12:48 crash).
- **Status:** my independent COCOeval recompute from `preds.pkl` **confirms every value**, so the numbers are sound — but there is **no canonical in-training log line** a reviewer/co-author could grep, unlike the other 3 runs. The provenance is offline-eval-only.
- **Action (optional, robustness):** either (a) note in the artifact trail that HRSID A0R uses offline eval from iter_100848 (`--resume` lost only ~24 iters), or (b) re-run the single final validation in isolation (HRSID-no-concurrency rule already in PROGRESS line 12) to produce a copypaste line. Not blocking — value is independently reproduced — but it is the one main-table number lacking a log-line anchor.

### C2. Within-PSeg seed1 AP_l = **+2.7** (the sign-flip that best proves "noise") is not re-derived in this audit
- The audit confirms the **seed0** PSeg AP_l (0.713→0.576) from preds.pkl, and confirms HRSID's 76-instance −0.2. The **seed1 +2.7** (0.625→0.652) is cited from the verdict md but was **not independently recomputed here** (no seed1 PSeg preds.pkl in the audit inputs).
- **Why it matters:** `review_summary.md` M18 wants the within-PSeg sign-flip (seed0 −13.8 / seed1 +2.7) promoted to the *primary* noise argument. If you adopt that framing, the +2.7 becomes load-bearing and should be recomputed from the seed1 PSeg predictions before it goes in the paper.
- **Action:** locate/regenerate `a0_pseg_seed1` + `a0r_refine_seed1` AP_l (large-stratum) preds and confirm +2.7 independently, the same way seed0 was confirmed.

---

**Bottom line:** Headline (+2.6 AP75, two-seed-robust) and all protocol/leakage/BIoU/G1 numbers are confirmed and submission-ready. Required edits: **one true fix (−13.7→−13.8, ~12 files incl. `seal.tex:46`)**, **three framing labels** (mAP −0.1, P 0.913→0.931, both = seed-specific not mean), **one json re-dump** (PSeg BIoU 67%). Two items are reproduced-but-unanchored (HRSID A0R offline provenance; seed1 +2.7 not yet recomputed) — neither blocks submission.