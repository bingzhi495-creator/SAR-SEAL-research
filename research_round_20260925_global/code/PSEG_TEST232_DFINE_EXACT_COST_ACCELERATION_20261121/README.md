# Reproducing the D-FINE point probability timing

`sparse_sampling.py` contains the only replacement used in the method. The model, source head and all downstream policy functions are loaded through the original closed replay. `replacement()` wraps the native-size adapter on fresh objects and leaves the original module available as the paired reference.

The timing tools receive the following paths as environment variables.

| Variable | Contents |
| --- | --- |
| `DFINE_BASELINE_SOURCE` | Independently timed baseline worker containing `read_closed_inputs`, `natural`, `timed`, and `summary` |
| `DFINE_BASELINE_RUN` | Directory of the prior accepted CPU benchmark in its source/owner layout |
| `DFINE_CLOSED_INPUT_ROOT` | Root of the closed PSeg input and published research-code layout used by the original replay |
| `DFINE_CAPTURE_OWNER` | Original PSeg232 owner directory with the accepted manifest and input JPEGs |
| `DFINE_CLOSED_REPLAY` | Directory of the original verified natural RLE edit ledger |
| `DFINE_BASELINE_SOURCE_SHA` | Optional, independently pinned source digest for a relocated compatible baseline. The original exact digest is the default. |

Set `CUDA_VISIBLE_DEVICES=-1`. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 1 before starting the isolated Python interpreter. Use a fresh directory containing the protocol and scripts for each attempt. The tools create result files exclusively in their own directory.

```text
python -I -B -X utf8 test_sampling.py -v
python -I -B -X utf8 profile_stage.py
python -I -B -X utf8 compare.py
python -I -B -X utf8 verify_results.py --result-sha OBSERVED_FOOTER_SHA --session OBSERVED_SESSION --exit OBSERVED_EXIT --chunk FINAL_OBSERVED_CHUNK
```

The profile covers 32 predetermined image ordinals. The exact-output and timing comparison covers every image. `PROFILE_SAFE.json` records disjoint component times. `RESULT_SAFE.json` records the original and optimized latency and per-repetition totals. The warm optimized execution independently compares every sampled binary16 value before all four timed outputs are checked against the closed reference. `verify_results.py` reads the paired timing observations and checks the reported latency summaries using the Python standard library.
