# FIT109 fractional-target comparison

[Frozen protocol](PROTOCOL.md)

Use the original CPU environment and closed FIT109 artifacts required by the [classical replay](../FIT109_CLASSICAL_ATTRACTION_20261121/README.md). Set `SEAL_FIT109_SOURCE_ROOT` to the local source tree and `CUDA_VISIBLE_DEVICES=-1`. Run the command below from this directory. The check phase validates only sources and runtime.

```text
python -I -B -X utf8 compare_targets.py check --pilot ../FIT109_BOUNDARY_SIMPLE_POINT_PILOT_20261121/pilot.py --classical ../FIT109_CLASSICAL_ATTRACTION_20261121/compare.py --helper ../FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py --helper-sha256 <helper SHA256> --protocol PROTOCOL.md --protocol-sha 52a6d25e2789906ec7f57b4d5cc6264dbf4211db3286e44e41c24b1e013bf30f --output <fresh directory>
```

After prelock review, replace `check` with `run`. Retain the actual child exit and final digest. The result records source label counts, all paired schedules, exact original score replay, and each outcome. Run `test_targets.py` with the same `--pilot`, `--classical`, `--helper` and `--helper-sha256` for generated native-label and trainer tests.
