# D-FINE DEV705 rival-mask transport

[Frozen protocol](PROTOCOL.md)

The privately supplied input map gives paths for the original HRSID PNG archive, DEV audit, probability sidecar and acceptance, frozen FIT head and acceptance, historical edit stream and acceptance, original geometry runtime, and the closed FIT result. The worker pins the original replay, detector outputs, point head and FIT pilot code. The legacy readout completes both the original arms and the two transports from each single exact probability reconstruction.

Run `transport.py` in the exact original CPU Python environment with `CUDA_VISIBLE_DEVICES=-1`, using `-I -B -X utf8`. For each phase, pass `--code` for the research code directory, `--rival` and `--fit-worker` for the immutable FIT109 experiment sources, and `--protocol PROTOCOL.md --protocol-sha` for the frozen text. Use `check` without data first. Then use `capture --inputs <map> --output <new directory>` and observe the actual exit 0 and result SHA.

Next run `readback` with the same inputs, a fresh output, the finished `--capture` directory, `--capture-sha` and `--capture-terminal`. A terminal record has fields `original_exit` and `result_sha256` from the actual process. The separately observed readback terminal closes the label-free phase.

For `score`, also supply the readback directory, SHA and terminal by `--readback`, `--readback-sha` and `--readback-terminal`. Supply `--score-source score.py --score-source-sha` and `--reference` pointing to the historical independently replayed score, `DFINE_E24_DEV705_FREE_ENDPOINT_20261120_SCORE_REPLAY_SAFE.json`. Use a new score directory. The source accepts original annotation access only after both prior phase checks pass.

Use `python -I -B -X utf8 test_transport.py --code <research code> --rival <FIT helper> --fit-worker <FIT worker>` for the generated old/transport RLE and exact drop check.
