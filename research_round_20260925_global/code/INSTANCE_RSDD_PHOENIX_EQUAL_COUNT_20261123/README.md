# Phoenix signed-order comparison on Instance-RSDD 74

The [locked protocol](../../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_PRELOCK_20261123.md) compares Phoenix sigmoid ordering and three source-trained occupancy heads at the exact archived dynamic classical edit counts. The [74-chip classical source study](../../INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_ACTUAL_20261123.md) supplies the fixed counts, edited policies and replay acceptance. The admitted RF-DETR detector population contains 7,346 original candidates, and 4,107 candidates have positive counts.

The [official pinned author source](https://github.com/naver-ai/Phoenix/tree/13ee3996ef3d0292fab144316cc8e03b69c0ecfa) and [XL1 instance weights](https://huggingface.co/naver-iv/phoenix-weights/blob/8a0eaf4f3396ee6c4bce9197d07b64efff23a174/phoenix_efficientvit_xl1.pt) are required at inference. The offline CPU worker uses the [previously qualified FastGeodis environment](../../results/PHOENIX_HRSID_FIT114_CPU_PILOT_ACTUAL_SAFE_20261111.json) and the pinned source manifest. The hosted COCO/Cityscapes evaluator skips predictions at or below one percent of image area. LVIS uses a zero area fraction, which still skips empty masks. In this study the matched-count comparator bypasses the COCO/Cityscapes rule and evaluates its final probability order on a fixed signed support. There is no whole-mask Phoenix transport comparison.

Always keep the original images, official annotations, native estimates, checkpoints and edge-probability ledger in local or authorized research storage. Files required from [the B5 reproduction](../INSTANCE_RSDD_B5_REPRO_20261123/README.md) and [the label-free PointRend archive](../INSTANCE_RSDD_RESERVE_POINTREND_20261123/README.md) are read through their existing cohort reader. Only the stage summaries and actual score aggregates are retained in the public source tree.

## Local request construction

Set `py` to the qualified isolated B5 interpreter, `code` to the published code directory, and `protocol` to the exact prelocked file. Run the following worker under `observe.py --phase request` with the request stage's result path.

```
python -I -B -X utf8 build_request.py --code CODE --stock-root B5 --point-root POINTREND --cfa-verify CFA/verify-edits --cfa-terminal CFA/observed_verify-edits/TERMINAL.json --protocol LOCK --protocol-sha256 SHA --output RUN/request --workers 4
```

After the actual zero exit, transfer the one private request JSON and 74 bound RGB PNGs to the remote compute root. Transfer the inference worker, observer and lock without changing any bytes. Check the request readback hash.

## DGX author inference

Use the exact previous DGX CPU interpreter with FastGeodis 1.0.5 and Torch 2.10.0+cu130. Set CUDA visibility to `-1`. Run in a fresh remote root with `observe.py --phase inference`. Provide the trusted author's actual source directory, its official checkpoint and its `SOURCE_ACQUISITION_SAFE.json` manifest. The worker loads the official XL1 architecture strictly.

```
python -I -B -X utf8 infer.py --source-root AUTHOR --request RUN/request/REQUEST_PRIVATE.json --request-sha256 REQUEST_SHA --protocol LOCK --protocol-sha256 SHA --output RUN/inference --workers 4 --threads 2
```

Return just the closed probability ledger and aggregate result and terminal for local scoring. Every image is handled as one complete task. Author decoder batches contain at most two candidates, and whole 512×512 native charts are given to the author refiner. Every candidate retains its original identity when only endpoint scores return.

## Local edits and score

Use `score.py edits` with the original source cohort paths, `--request RUN/request --request-terminal RUN/observed_request/TERMINAL.json --inference RUN/inference --inference-terminal RUN/observed_inference/TERMINAL.json --endpoints RUN/inference/ENDPOINT_PROBABILITIES_PRIVATE.jsonl`, the exact lock, and fresh output `RUN/edits`. Observe actual closure with phase `edits`. The edit worker independently checks both sorting implementations, the complete original candidate population and exact signed pixel sets and counts. Use `score.py score` with the same arguments (fresh score output), `--edits RUN/edits --edits-terminal RUN/observed_edits/TERMINAL.json --cfa-score CFA/score --cfa-score-terminal CFA/observed_score/TERMINAL.json --author-labels AUTHOR_ANNOTATIONS/train.json`. Run the score under the observer with phase `score`. Use `--help` for all argument roles. The scorer requires the archived candidate and metric parity before evaluating the main and secondary contrast decisions.

[Observed experiment](../../INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_ACTUAL_20261123.md) · [Aggregate score and stage receipts](../../results/INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_ACTUAL_SAFE_20261123.json)
