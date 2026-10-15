"""Supported sequential MKL threading avoids duplicate OpenMP runtimes on this PC.

First launch exited during dataset setup before any epoch. Training parameters,
seed and initialization remain those in train_yolo11n.py. KMP_DUPLICATE_LIB_OK is
not enabled; shared environment DLLs are not modified.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
from train_yolo11n import main

if __name__ == "__main__":
    main()
