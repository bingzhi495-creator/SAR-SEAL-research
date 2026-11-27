# Observed examples for the NORA method figures

The Instance-RSDD panels follow two previously reviewed D-FINE detections from the historical 74-chip reserve. These are examples close to the positive and negative medians of the continuous-IoU change from bounded re-pairing to independent selection. On the positive case, all three writers use four exchanges. The loss case uses eight. The two-exchange diagrams above them are separately constructed examples. Case-level scores use the original polygon and native cell geometry.

| Figure | Source and interpretation |
| --- | --- |
| Scene to mask | Observed native and independent masks on the positive reserve case, with four additions and four removals. The public 8-by-8 GT and prediction coverages are rounded to quarter levels for display. |
| Sampling | The observed source mask accompanies an illustration of the two rings, eligible signs and ranking. The public lower inset shows the measured native mask pooled into an 8-by-8 grid. |
| Scalar readout | A measured hull footprint sets the context for an illustrative half-covered cell and interpolation at the cell center. The public reference footprint is pooled into an 8-by-8 grid and rounded to quarter levels. |
| Placement, gain and loss pages | The upper panels enumerate the same two-cell illustrative example. The lower panels use the mean coverage per bin already released in the [Instance-RSDD case summary](../rsdd74_common_cap_cases_safe.json). The coverage color scale is continuous. |
| Signed allocation | The four-flip example is followed by observed signed counts under the equal and variable policies at the first fit on 183 SSDD development images. The counts sum across the full candidate stream. They are available in the [manuscript](../../boundary_refinement_concise.tex) and in the same-image signed-allocation section of the [supplement](../../boundary_refinement_supplement.tex). |
| Frozen Phoenix comparison | The four experimental arms are illustrated in the upper matrix. The lower row displays the two whole-grid arms for two PSeg-SSDD instances. Public footprints reproduce the displayed 8-by-8, quarter-level vectors of the [published case figure](../journal_20261126/pseg_budget_cases_binned.pdf). The measured counts and IoUs are specific to those instances. |

The scalar and dual ROH layer diagram describes learned parameter dimensions and supervision. The overlap-accounting diagram illustrates an algebraic identity. Both retain their established source figures.

To regenerate the public PDF/SVG pairs and seven editable Draw.io pages from the manuscript’s released figures, run the following from this directory.

```sh
python build_observed_methods.py --figures .. --output rebuilt
```

The optional `--author-root` argument accepts an existing admitted local manuscript directory with its approved case PDFs. It writes the native variants to a separate output folder. The generator reads only the named manuscript figure sources and their authorized summary records. The public outputs use aggregated shapes and measured counts. Preserve the existing figure-source directories when using this generator. Scientific inputs, software dependencies and experimental scope are described in the original source sections linked above. The primary rendered references are the PDF/SVG pairs. The editable companion contains the public diagrams.
