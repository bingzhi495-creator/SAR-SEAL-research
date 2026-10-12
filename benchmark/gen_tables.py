# -*- coding: utf-8 -*-
"""Generate the three SAR-benchmark LaTeX tables from sar_benchmark_data.py.
No 'winner' bolding (cross-model comparison is confounded by epochs/optimizer/
domain -- see captions). N/A for missing values. Output: benchmark/tables.tex."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sar_benchmark_data import DET, OBB, CLS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.tex")


def cell(v):
    return "N/A" if v is None else v


def det_table():
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Horizontal-box detection benchmark on four SAR datasets (COCO protocol). "
         r"All models are nano/lightweight, trained \emph{from scratch} (no pretrained weights) on one RTX~3090\,Ti. "
         r"$P$/$R$ are taken at the max-$F_1$ confidence; recall $R$ is the detection probability $P_d$; "
         r"false-alarm rate $\mathrm{FA}=1-P$. Comparisons are \emph{confounded} and must not be read as a clean ranking: "
         r"MSCK-Net trains 160 epochs (vs.\ 100), its evaluator reports only COCO AP (so $P$/$R$/FA are N/A) "
         r"and its native domain is infrared (domain shift to SAR); Mamba-YOLO-T uses SGD rather than AdamW. "
         r"Each dataset's rows are ordered by mAP$_{50}$.}",
         r"\label{tab:det}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Dataset & Model & mAP$_{50}$ & mAP$_{50:95}$ & $P$ & $R$\,($P_d$) & FA \\", r"\midrule"]
    for ds, d in DET.items():
        hdr = f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{ds}}}\\ \\footnotesize ({d['target']}, {d['cls']} cls, train/val {d['split']})}} \\\\"
        L.append(hdr)
        for name, m in d["rows"]:
            L.append(f"\\quad {name} & & {m[0]} & {m[1]} & {cell(m[2])} & {cell(m[3])} & {cell(m[4])} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def obb_table():
    d = OBB["RSAR"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Oriented-box (OBB) detection on RSAR (6 classes, DOTA$\to$YOLO-OBB, official "
         r"test set as val). Only YOLO26n provides an OBB head; the remaining four detectors are "
         r"\textsc{task-incompatible} (no rotated-box head) and are excluded from this task, not scored as zero.}",
         r"\label{tab:obb}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Model & mAP$_{50}$ & mAP$_{50:95}$ & $P$ & $R$\,($P_d$) & FA \\", r"\midrule"]
    for name, m in d["rows"]:
        L.append(f"{name} & {m[0]} & {m[1]} & {m[2]} & {m[3]} & {m[4]} \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{6}{@{}p{0.92\columnwidth}@{}}{\footnotesize "
             + ", ".join(d["incompatible"]) + r": \textsc{task-incompatible} (no OBB head).} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def cls_table():
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Chip classification (Top-1/Top-5 accuracy). MSTAR uses the official SOC protocol "
         r"(17$^\circ$ train / 15$^\circ$ test); FUSAR-Ship uses a stratified fixed-seed-0 split over the "
         r"8 classes with $\ge\!50$ samples. Only YOLO26n provides a classification head; all detection/OBB "
         r"models are \textsc{task-incompatible} here.}",
         r"\label{tab:cls}", r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
         r"Dataset & Model & Top-1 & Top-5 \\", r"\midrule"]
    for ds, d in CLS.items():
        for name, m in d["rows"]:
            L.append(f"{ds} \\footnotesize({d['split']}) & {name} & {m[0]} & {m[1]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    txt = "\n\n".join([det_table(), obb_table(), cls_table()])
    open(OUT, "w", encoding="utf-8").write(txt)
    print("wrote", OUT, "-", len(txt), "chars")
    print(txt[:600])
