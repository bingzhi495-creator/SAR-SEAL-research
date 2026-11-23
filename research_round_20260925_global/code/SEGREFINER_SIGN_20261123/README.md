# SegRefiner-LR signed component diagnostic

This CPU-only postprocessing reproduces the closed stock author output and native score rows before measuring add-only and delete-only controls. Follow the [signed-lock protocol](../../SEGREFINER_SIGN_PRELOCK_20261123.md), [stock baseline](../SEGREFINER_STOCK_20261123/README.md) and [qualified stock output](../../SEGREFINER_STOCK_ACTUAL_20261123.md). The analysis uses immutable validation, inference and score receipts and the original benchmark split.

Use the observed-termination script from [the released Phoenix comparison](../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_20261123/observe.py), phase `score`, with separate observer and result directories. Check the actual exit and result digest before reporting. The protocol SHA-256 is `3e5918d16dad9f6cf0642e4c3222d16673f53a56835045de3b2687ab9776a7b1`.

The Instance-RSDD worker takes the qualified stock-run directory, frozen native roots, published float annotation path, and signed protocol. The PSeg worker takes the original stock scorer inputs and also its verified source, score directory and observed score terminal. Both scripts expose the full CLI in `--help`. They use the existing B5 CPU geometry environment and store only aggregate safe statistics.

Four masks are native, union, intersection and stock. Union and intersection isolate the writes of each sign while fixing every detection identity and score. No label is used to construct any of the four masks. The primary deletion gate uses both IoU contexts on the eligible fixed geometry subset. Full-mask AP includes the independent rematching of all candidates.
