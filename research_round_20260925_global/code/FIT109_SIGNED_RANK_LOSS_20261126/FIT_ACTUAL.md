# Actual FIT109 matched scalar training

Two scalar PointHeads were fitted for each of the five original prefix folds and each of the seeds 5217, 5218 and 5219. Fractional BCE is the control. The other arm adds the registered 0.25-weighted within-candidate, within-side ranking loss. The two arms used the same 1,321-parameter architecture, exact starting tensors, fold scaler, first-batch components, ordered sampled row packets and optimizer. Each update used 64 paired rows. The rank loss averaged 192 unordered comparisons, with fractional label gaps as weights and zero contribution from ties.

The observed job completed all 30 planned fits and 6,000 optimizer updates. The five fold training sets used 73–75 valid source prefixes and 259–343 supervised candidates. The original feature scaler was computed on the training folds only. Each fitted file was read back and its restored tensor digest checked. Each arm and seed then emitted one out-of-fold prediction for each of the 203,903 candidate pair rows, retaining canonical order. No all-source checkpoint was fitted. This stage opened the admitted FIT-only source data and made zero detector forwards. COCO evaluation is a separate stage.

The witnessed child ran 20.177 seconds from start to natural exit, including runtime startup. The worker measured 16.759 seconds internally. The cumulative training scopes measured 6.990 seconds for BCE and 3.866 seconds for BCE plus ranking in fixed sequential order. These times include asymmetric startup effects and are recorded for execution provenance.

[Registration](PRELOCK.md) · [Training instructions](REPRODUCE_FIT.md) · [Safe training data](FIT_SAFE.json) · [Witnessed closure](FIT_TERMINAL_SAFE.json)

