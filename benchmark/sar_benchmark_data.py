# -*- coding: utf-8 -*-
"""Authoritative, verbatim record of the user-provided SAR detection/recognition
benchmark (single RTX 3090 Ti, from scratch, no pretrained). DO NOT fabricate or
extrapolate. Values stored as the EXACT strings given (preserve precision);
None -> N/A. Source: metrics.json/results.csv, Mamba output_dir/sar/*/results.csv,
MSCK log.txt, results_comparison.csv, BENCHMARK_REPORT.md.
"""

# Horizontal-box detection. Per row: [mAP50, mAP50_95, P, R_Pd, FA_1mP]; None=N/A.
DET = {
    "SSDD": {"split": "928/232", "cls": 1, "target": "ship", "rows": [
        ("YOLOv8n",      ["0.9724", "0.6967", "0.9714", "0.9029", "0.0286"]),
        ("Mamba-YOLO-T", ["0.9769", "0.6865", "0.9419", "0.9267", "0.0581"]),
        ("YOLOv12n",     ["0.9579", "0.6703", "0.9277", "0.8718", "0.0723"]),
        ("YOLO26n",      ["0.9497", "0.6455", "0.9039", "0.8790", "0.0961"]),
        ("MSCK-Net",     ["0.936",  "0.7146", None,     None,     None]),
    ]},
    "HRSID": {"split": "3642/1962", "cls": 1, "target": "ship", "rows": [
        ("YOLOv8n",      ["0.8947", "0.6314", "0.9180", "0.8015", "0.0820"]),
        ("YOLO26n",      ["0.8750", "0.6045", "0.8925", "0.7909", "0.1075"]),
        ("Mamba-YOLO-T", ["0.8713", "0.6227", "0.8969", "0.7916", "0.1031"]),
        ("YOLOv12n",     ["0.8519", "0.5929", "0.8799", "0.7567", "0.1201"]),
        ("MSCK-Net",     ["0.812",  "0.635",  None,     None,     None]),
    ]},
    "SADD": {"split": "2373/593", "cls": 1, "target": "aircraft (many negatives)", "rows": [
        ("YOLOv8n",      ["0.9854", "0.5956", "0.9669", "0.9568", "0.0331"]),
        ("Mamba-YOLO-T", ["0.9821", "0.6297", "0.9535", "0.9654", "0.0465"]),
        ("YOLOv12n",     ["0.9796", "0.5926", "0.9554", "0.9375", "0.0446"]),
        ("MSCK-Net",     ["0.971",  "0.683",  None,     None,     None]),
        ("YOLO26n",      ["0.9679", "0.5630", "0.9192", "0.9258", "0.0808"]),
    ]},
    "SAR-AIRcraft-1.0": {"split": "3489/879", "cls": 7, "target": "aircraft (7-class)", "rows": [
        ("Mamba-YOLO-T", ["0.8623", "0.6048", "0.8515", "0.8167", "0.1485"]),
        ("YOLOv8n",      ["0.8482", "0.5509", "0.8564", "0.7742", "0.1436"]),
        ("MSCK-Net",     ["0.826",  "0.610",  None,     None,     None]),
        ("YOLO26n",      ["0.8041", "0.5038", "0.8125", "0.7127", "0.1875"]),
        ("YOLOv12n",     ["0.7843", "0.4753", "0.7273", "0.7361", "0.2727"]),
    ]},
}

# Oriented-box detection (RSAR). Only YOLO26n-OBB has an OBB head.
OBB = {
    "RSAR": {"split": "78837/8538", "cls": 6, "target": "ship/aircraft/car/tank/bridge/harbor",
             "rows": [("YOLO26n-OBB", ["0.6681", "0.4049", "0.7619", "0.6123", "0.2381"])],
             "incompatible": ["YOLOv8n", "YOLOv12n", "Mamba-YOLO-T", "MSCK-Net"]},
}

# Classification: [Top1, Top5].
CLS = {
    "MSTAR (SOC, 10-class)": {"split": "2747/2425", "rows": [("YOLO26n-cls", ["0.9542", "0.9988"])],
                              "incompatible": "all detection/OBB models"},
    "FUSAR-Ship (8-class)":  {"split": "4106/1020", "rows": [("YOLO26n-cls", ["0.7412", "0.9833"])],
                              "incompatible": "all detection/OBB models"},
}

META = {
    "hardware": "single RTX 3090 Ti (24 GB), from scratch, no pretrained weights; same train/val split per dataset across models",
    "input": "640x640 (RSAR native 256x256); batch 16 (RSAR/cls batch 64); 100 epochs; MSCK-Net 160 epochs (its official recipe)",
    "opt": {"YOLOv8n/YOLOv12n/YOLO26n": "AdamW lr=2e-3 mom=0.9",
            "Mamba-YOLO-T": "SGD lr=1e-2 mom=0.937", "MSCK-Net": "AdamW lr=8e-4 flat-cosine + EMA"},
    "metrics": "COCO mAP@0.5 and mAP@0.5:0.95; P/R(=Pd) at max-F1 confidence (ultralytics default); FA=1-Precision; Top-1/Top-5",
    "caveats": [
        "MSCK-Net P/R/FA are N/A: its DEIM evaluator outputs only COCO AP, not P/R at a fixed confidence.",
        "MSCK-Net mAP50 is from the FINAL epoch, mAP50-95 from the BEST epoch -- inconsistent checkpoint pair; re-extract a consistent pair from log.txt if one checkpoint is required.",
        "MSCK-Net trains 160 epochs vs 100 for others; Mamba-YOLO-T uses SGD not AdamW; MSCK-Net's native domain is INFRARED ship detection (domain shift to SAR). Declare these; do not assert one model is simply 'better'.",
        "RSAR reported on the official test set (serving as val); SADD/FUSAR-Ship use self-built fixed-seed-0 splits (no official split).",
        "All models are nano/lightweight configs -- a LIGHTWEIGHT cross-comparison, not comparable to large-model COCO numbers in the original papers.",
        "HGTFN is EXCLUDED: hyperspectral per-pixel classification, mismatched in task/modality/label to all 7 SAR datasets.",
    ],
}
