# -*- coding: utf-8 -*-
"""Generate the segmentation-baseline LaTeX tables from seg_baselines.json
(D:/SAR/UNet++ project, from-scratch). Two SEPARATE protocols, never merged with
SEAL's COCO instance AP on Mask2Former-R50: (1) semantic Dice/mIoU at budget-10
tiny configs; (2) UNet++ connected-component instance AP. Output: seg_tables.tex.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "seg_baselines.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "seg_tables.tex")


def f4(v):
    return "N/A" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def sem_table():
    sem = D["semantic"]
    rows = {}
    for key, m in sem.items():
        ds, model = key.split("/")
        rows.setdefault(model, {})[ds] = m
    name = {"deeplabv3plus": "DeepLabV3+", "segformer": "SegFormer", "mask2former": "Mask2Former (tiny)"}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Semantic segmentation baselines on SAR ship sets, \emph{from scratch}, single RTX~3090\,Ti, "
         r"budget-10 lightweight configs (binary ship/background, Dice/mIoU at threshold $0.5$). This is a "
         r"\emph{separate protocol} from SEAL's COCO instance mask AP on Mask2Former-R50 and is \emph{not} merged "
         r"with it; numbers are a same-hardware semantic landscape, claims limited to this protocol.}",
         r"\label{tab:seg-semantic}", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Model & Dataset & Dice & mIoU & $P$ & $R$ \\", r"\midrule"]
    for model in ["deeplabv3plus", "segformer", "mask2former"]:
        for ds in ["hrsid", "ssdd"]:
            m = rows.get(model, {}).get(ds)
            if m:
                L.append(f"{name[model]} & {ds.upper()} & {f4(m['dice'])} & {f4(m['miou'])} & {f4(m['precision'])} & {f4(m['recall'])} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def inst_table():
    ui = D["unetpp_instance"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{UNet++ instance segmentation baselines (connected-component instance protocol), from scratch. "
         r"A \emph{third, distinct} protocol from both Table~\ref{tab:seg-semantic} and SEAL's COCO instance AP; "
         r"reported separately, claims limited to this protocol.}",
         r"\label{tab:seg-instance}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Variant & Dataset & Dice & AP$_{50}$ & AP$_{75}$ \\", r"\midrule"]
    for variant in ["clean", "boundary"]:
        for ds in ["hrsid", "ssdd"]:
            m = ui[variant][ds]
            L.append(f"UNet++ ({variant}) & {ds.upper()} & {f4(m['dice'])} & {f4(m['ap50'])} & {f4(m['ap75'])} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    txt = sem_table() + "\n\n" + inst_table()
    open(OUT, "w", encoding="utf-8").write(txt)
    print("wrote", OUT)
    print(txt[:500])
