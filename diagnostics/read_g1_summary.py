# -*- coding: utf-8 -*-
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("method          all       inshore   offshore")
for m in d:
    if m.startswith("_"):
        continue
    row = d[m]
    a = row.get("all", {}).get("mean", float("nan"))
    ins = row.get("inshore", {}).get("mean", float("nan"))
    off = row.get("offshore", {}).get("mean", float("nan"))
    # HRSID has no inshore/offshore split (subset='all'); guard for nan
    def f(x):
        return f"{x:.4f}" if x == x else "  -   "
    print(f"{m:<14}  {f(a)}    {f(ins)}    {f(off)}")
g = d.get("_gate", {})
print("\n--- GATE (verdict-1) ---")
print("mne_E - box_z      :", round(g.get("mne_E_minus_box_z", float('nan')), 4), "  (pass >= +0.03)")
print("mne_E - strongest  :", round(g.get("mne_E_minus_strongest", float('nan')), 4))
