# Label-free SAR-SSP transverse quota replay

The original historical SAR-SSP validation images and D-FINE candidate streams were replayed in a standalone CPU phase. The published prelock and original fit remained fixed. The observer recorded exit zero and verified the result footer after 120.35 s. All archived positive-budget ranks, signed endpoints, actions and four archived arms' native RLEs were reproduced exactly. For the complete 232 images and 23,200 candidate slots, there are 10,393 positive-budget candidates. Each arm writes 147,118 native cells. The label-free phase records no ground-truth-dependent metric.

Among positive-budget candidates, 4,024 meet the single-component and native aspect requirements. The transverse rule changes 2,266 candidate writes relative to the unrestricted quota rule. These were exactly the eligible writes that exceeded the balanced short-axis cap. The same-menu-width area and longitudinal controls change 2,092 and 2,177 writes, respectively.

| Writer | Additions | Removals | Changes from variable |
| --- | ---: | ---: | ---: |
| Original balanced | 73,559 | 73,559 | — |
| Original variable | 80,493 | 66,625 | — |
| Transverse cap | 73,215 | 73,903 | 2,266 |
| Area control | 80,748 | 66,370 | 2,092 |
| Longitudinal control | 77,214 | 69,904 | 2,177 |

The controls fix total edits and eligible menu size per candidate. They allow different realized addition and removal counts because selecting that split is the treatment. The transverse policy produces a large change in aggregate signed allocation. Its comparison can evaluate the specified selection rule under the matched total, with limits on interpreting any resulting difference as independent of foreground volume.

Seven synthetic tests cover selection parity, relative floating-point tolerance, centered variance, shape eligibility, degenerate allocations, archive-source compatibility and rejection of altered original RLE. [Protocol](PRELOCK.md) · [Replay instructions](REPRODUCE_RENDER.md) · [Safe render result](RENDER_SAFE.json) · [Observed terminal](RENDER_TERMINAL_SAFE.json)

