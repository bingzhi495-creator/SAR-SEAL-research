# Native-grid method figures

These five diagrams explain the manuscript's existing method. All cell maps and score values are worked examples. No experimental result, reference footprint or native SAR crop enters these outputs.

| Stem | IEEE width | Canvas (pt) | Figure content |
| --- | --- | --- | --- |
| `matched_budget_pipeline` | Two columns | 514.8 × 171 | Frozen candidate, sampled support, occupancy and signed write. Independent training-only and frozen-refiner lanes. |
| `method_native_occupancy` | One column | 247.9 × 308 | Native cell grid, boundary support, endpoint readout, visibly separate fractional polygon supervision, and residual head. |
| `method_bounded_selection` | Two columns | 514.8 × 176 | Original signed-count example with two additions and two removals per writer. The paired and bounded panels show selected edges. The independent panel shows endpoint sets. |
| `method_signed_allocation` | Two columns | 514.8 × 157 | Equal-total comparison on four flips. Balanced allocation preserves 15 foreground cells. Variable allocation produces 17. |
| `method_phoenix_controls` | One column | 247.9 × 262 | One frozen response, a support-by-writer table, and a separate matched-total allocation control. |

All SVG files contain editable type, vector cells and connectors. Their companion PDF files retain vector geometry. The layout uses one font family and the same add/remove signs throughout. The paper should retain the explanatory figure captions.

Rebuild with Python 3.10.13, Matplotlib 3.10.9 and NumPy 1.26.4. Run `python draw_method_figures_cvpr.py --fixtures-dir ORIGINAL_FIGURES --output OUTPUT`. The default fixture location beside the redraw script supports placing it in the existing manuscript figure directory. Originals supplying the worked examples are `draw_matched_budget_pipeline.py`, `draw_selection_controls.py`, `draw_signed_allocation.py` and `draw_supporting_stages_redesign.py` with their existing dependencies.

The original algorithms are re-executed when drawing. The generator checks the edited cell counts, total and signed budgets, mask areas and inherited cap. It checks text against the final canvas. The staged previews were inspected at their final column proportions. PNG previews are optional and do not enter the vector manuscript.

