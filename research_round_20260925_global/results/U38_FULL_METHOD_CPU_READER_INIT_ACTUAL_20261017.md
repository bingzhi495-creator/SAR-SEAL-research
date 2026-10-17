# U38 CPU reader and common initialization result

Scope: actual single CPU leaf; Root terminal/source/report metadata intake. Worker checked inputs and fresh CPU initialization; Root did not replay pixels or neural operations.

- FIT images / matched instances: 741 / 1661, complete original population.
- Common states: 104 = 68 parameter tensors + 36 BN buffers; seed 202610173101.
- Branch / canvas: 400 / 800; native-logit baseline.
- Original shape families: 572; content shape families: 262; G range: 1 to 29; original pixel-area range: 57570 to 306257.
- Proposed small / high-burden / small coverage: 2 distinct images, 2 original shapes, 2 G values; GPU not executed.
- Qualification / reader / initialization seconds: 167.781881 / 166.826471 / 0.864321.
- Whole original leaf elapsed seconds: 289.953000.
- Worker observed peak RSS bytes: 824074240; minimum available RAM bytes: 32907644928; CPU affinity 0,1; threads 2.

CUDA stayed uninitialized; no model forward, backward, optimizer, training, DEV, TEST or AP. No GPU resource or full-method training eligibility is granted. Boundary RSS samples do not prove absence of leaks.
