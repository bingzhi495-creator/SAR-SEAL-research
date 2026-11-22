# Reproduce the frozen 74-image Instance-RSDD result

This entry covers the parent reserve and all of its original evaluation views. It rebuilds the official input roster, converts the 74 images, exercises a short real-source fit, and evaluates the original frozen models. It accepts local paths and fresh output directories. [Evaluation protocol](../../INSTANCE_RSDD_PARENT_RESERVE_PRELOCK_20261123.md) · [Frozen reference](../../results/INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_SAFE_20261123.json)

## Inputs and environment

- Download the author Instance-RSDD `annotation_coco` directory, the original RSDD-SAR ZIP and the original paper PDF with Appendix Table 1. The public input and extraction links are in the [roster instructions](../INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/README.md).
- For the short training pass, supply the original HRSID training PNG directory and author `train2017.json`. The program selects four fixed source images by filename. [HRSID author data](https://github.com/chaozhong2010/HRSID)
- Supply the original RF-DETR HRSID center5202 E24 checkpoint and the FIT109 seed-5217 head bundle. The SHA-256 values are `011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11` and `3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1`. The bundle includes the normalization parameters. These project assets are supplied separately.
- The verified runtime is Windows x64 CPython 3.10.13 with the 77 hash-pinned wheels in the supplied lock file. It includes RF-DETR 1.12.0, PyTorch 2.10.0+cu130, NumPy 1.26.4, pycocotools 2.0.11 and Shapely 2.0.2 with GEOS 3.11.2. The loader uses CUDA for detector capture and CPU for point training, replay and scoring. The dataset roster builder calls `pdftotext`. [PyTorch installation sources](https://pytorch.org/get-started/locally/)

For a local wheel directory, create a new environment with `uv venv --python <python-3.10.13> <venv>`. Install the lock with `uv pip sync --python <venv>/Scripts/python.exe --no-index --find-links <wheel-directory> --require-hashes requirements.win-cp310.lock.txt`. The wheels are obtained from the package sources and checked by the lock. If the Conda Windows virtual-environment launcher fails to initialize PyTorch, run `prepare_win_host.ps1 -BasePythonRoot <conda-root> -EnvironmentRoot <venv> -Output <new-host>` and use the resulting `python.exe` with the same newly installed packages.

## Run from this directory

Replace the illustrative paths below and create the parent `runs` directory. Use the new environment's Python for every command.

```powershell
python ../INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/build_reserve_roster.py --author-dir data/annotation_coco --paper-pdf data/rsdd-paper.pdf --image-archive data/RSDD-SAR.zip --output runs/reserve.csv
python convert_reserve.py --roster runs/reserve.csv --archive data/RSDD-SAR.zip --output runs/intake
python portable_fit.py prepare --images data/HRSID/PNG --hrsid-annotations data/HRSID/train2017.json --checkpoint weights/checkpoint_23.ckpt --output runs/samples.npz
python portable_fit.py train --samples runs/samples.npz --steps 8 --output runs/training
$env:CUDA_VISIBLE_DEVICES = '0'
python portable_eval.py capture --intake runs/intake --roster runs/reserve.csv --checkpoint weights/checkpoint_23.ckpt --output runs/capture
$env:CUDA_VISIBLE_DEVICES = '-1'
python portable_eval.py replay --intake runs/intake --roster runs/reserve.csv --capture runs/capture --point-head weights/5217_all_fit.npz --output runs/replay
python portable_eval.py score --intake runs/intake --roster runs/reserve.csv --capture runs/capture --replay runs/replay --author-train data/annotation_coco/train.json --output runs/score
python verify_reproduction.py --reference ../../results/INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_SAFE_20261123.json --conversion runs/intake/RESULT_SAFE.json --training runs/training/FIT_REPORT.json --capture runs/capture/RESULT_SAFE.json --replay runs/replay/RESULT_SAFE.json --score runs/score/RESULT_SAFE.json --output runs/verification.json
```

In the observed run, the dataset step used official HTTPS byte ranges. The roster builder accepts the original central-directory bytes via `--zip-directory`, and the converter accepts the official central directory, archive tail and Drive download form via `--public-range-cache`. This gives the same 74 CRC-checked JPEGs and the same canonical manifest. The generated labels, local JPEGs, native predictions, edit rows and weights stay local.

The minimum training run rebuilds source samples from original PNGs and executes eight optimizer steps. It produces a separate test head. The complete frozen evaluation compares four arms over 74 images, both sensor views, the fixed 42-image date-separated view and the deduplicated view, plus per-parent results. The checker compares the regenerated data, candidates and edit digests and applies the predeclared tolerance to all reported numeric fields. [Tolerance and frozen scope](../../INSTANCE_RSDD_B5_REPRODUCTION_PRELOCK_20261123.md)
