# Repeat validation of fixed-total variable signed edits

This record covers two independently initialized heads, seeds 5298 and 5299, on the same 183-image source-development cohort. Both repeats passed the prespecified comparison against their same-head fixed-split and BCE robust writers. The aggregate 5297–5299 comparison with the same-objective fixed split was +0.971 fixed-match IoU points on PSeg and +0.573 on SPSD, with shared group-resampling intervals of [0.740, 1.207] and [0.319, 0.840]. The references annotate the same images.

- [Actual repeat report](ACTUAL.md)
- [Prespecified evaluation protocol](PRELOCK.md)
- [Seed 5298 render aggregate](seed5298/render/RESULT_SAFE.json)
- [Seed 5299 render aggregate](seed5299/render/RESULT_SAFE.json)
- [Combined scoring aggregate](score/RESULT_SAFE.json)

## Using the source

The renderer and scorer expose all paths as command-line arguments. Run each with `--help` for its complete argument list. The seed-pooling test runs locally with Python and NumPy.

The render entry requires the group protocol, closed repeat FIT result and readback, shared point packets, original source and action archives, and the frozen selector, attraction and [variable allocation writer](../SPSD_PSEG_GATE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/variable_writer.py). The score entry additionally requires the exact closed repeat render artifacts and authorized gate references, the [historical scorer and mechanism source](../SPSD_PSEG_SCORE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/score_gate.py), the original closed 5297 render and score, and the [archive admission worker](../SPSD_PSEG_GATE_SELECTIVE_BCE_ADOPTION_SEED5297_20261125/render_gate.py). Receipt digests and seed-specific tensor identities identify the precise inputs. Consult the [data-access guide](../../../docs/DATA_ACCESS.md) for official sources. The repository's aggregate data and these files are sufficient to check the reported comparison tables. Full rescore requires the separately authorized annotated evaluation data and private replay input artifacts.
