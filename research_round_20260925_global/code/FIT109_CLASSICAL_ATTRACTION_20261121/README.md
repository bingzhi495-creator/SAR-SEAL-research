# FIT109 spatial-attraction control

[Protocol](PROTOCOL.md) · [Original FIT109 scoring entry](../FIT109_BOUNDARY_SIMPLE_POINT_PILOT_20261121/README.md)

Use the Python environment and historical FIT109 inputs for the original scoring entry. Set `CUDA_VISIBLE_DEVICES=-1` and `SEAL_FIT109_SOURCE_ROOT` to the local directory containing the historical FIT109 artifacts and the independent release copy. The original scorer pins the retained model-output receipts, point support, scorer sources, and annotation scope.

```text
python -I -B -X utf8 compare.py check --pilot ../FIT109_BOUNDARY_SIMPLE_POINT_PILOT_20261121/pilot.py --helper spatial_attraction.py --helper-sha256 <helper SHA256> --prelock PROTOCOL.md --prelock-sha256 <protocol SHA256> --output <fresh local directory>
```

Use the same arguments with `run` after the import check. The output includes the realized count distribution, exact native write-back checks, original score parity, three matched-count contrasts and source-group intervals. The run is CPU-only and uses the same five 200-step source-held-out occupancy fits. Data and original predictions follow the repository's data access and release rules.
