# FIT109 native rival-mask pilot

[Frozen protocol](PROTOCOL.md)

The original FIT109 environment and qualified source artifacts are also required by the [classical replay](../FIT109_CLASSICAL_ATTRACTION_20261121/README.md). Use the prescribed Python `-I -B -X utf8` environment. Set `SEAL_FIT109_SOURCE_ROOT` to the local source root and `CUDA_VISIBLE_DEVICES=-1`. Specify a fresh output path. The check command resolves dependencies and verifies ordinary source hashes without opening prediction or annotation inputs.

```text
python -I -B -X utf8 run_rival_pilot.py check --pilot ../FIT109_BOUNDARY_SIMPLE_POINT_PILOT_20261121/pilot.py --classical ../FIT109_CLASSICAL_ATTRACTION_20261121/compare.py --helper rival.py --helper-sha256 a6f927db39800f2a251eee265032e6cb8a4b8c5318c1b77ef6c3417432a4a4d1 --prelock PROTOCOL.md --prelock-sha256 79e63614e20e2563b9a4f68e9aa2d0aac2e2e87989afb3f6f0225d81b1e95341 --output <fresh local directory>
```

Use `run` with the same arguments to execute the published prelock. The output contains the pre-label penalty digest, historical parity, aggregate scores and near-instance contrast. Run the generated checks with `python -I -B -X utf8 test_rival.py`.
