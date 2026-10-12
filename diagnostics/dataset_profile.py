# -*- coding: utf-8 -*-
"""Profile the per-dataset target-size / density / resolution distribution, to
MOTIVATE an adaptive (parameterized) refinement mechanism: different SAR
datasets differ in ship pixel-size, scene density and image resolution, so the
mask-refinement scale/gate should self-configure from these statistics.

Reports, per dataset: image size; instances-per-image (density); the
sqrt(area) (~ship pixel extent) distribution (quartiles); and the COCO
small/medium/large mix. Pure CPU on the GT json.
"""
import json, os, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Profile the TRAIN split: the AMR auto-config protocol derives operating points
# from leakage-free statistics, so profiling must use train (never test) GT.
DSETS = {
    "PSeg-SSDD": os.path.join(BASE, "data", "pseg_ssdd", "annotations", "train.json"),
    "HRSID-official": os.path.join(BASE, "data", "hrsid", "annotations", "official_train.json"),
    "HRSID-scene": os.path.join(BASE, "data", "hrsid", "annotations", "scene_train.json"),
}


def profile(name, path):
    d = json.load(open(path, encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    per_img = {}
    areas, sqrt_areas = [], []
    for a in d["annotations"]:
        per_img.setdefault(a["image_id"], 0)
        per_img[a["image_id"]] += 1
        ar = a.get("area") or (a["bbox"][2] * a["bbox"][3])
        areas.append(ar); sqrt_areas.append(ar ** 0.5)
    areas = np.array(areas); sqrt_areas = np.array(sqrt_areas)
    dens = np.array([per_img.get(i, 0) for i in imgs])
    whs = np.array([[im["width"], im["height"]] for im in imgs.values()])
    diag = np.sqrt((whs ** 2).sum(1))
    sm = (areas < 1024).mean(); md = ((areas >= 1024) & (areas < 9216)).mean(); lg = (areas >= 9216).mean()
    print(f"\n===== {name} =====")
    print(f"  images={len(imgs)}  instances={len(areas)}  img_size~{int(np.median(whs[:,0]))}x{int(np.median(whs[:,1]))} (diag {int(np.median(diag))}px)")
    print(f"  density (inst/img): mean {dens.mean():.1f}  median {int(np.median(dens))}  p90 {int(np.percentile(dens,90))}  max {dens.max()}")
    print(f"  ship extent sqrt(area) px: p25 {np.percentile(sqrt_areas,25):.0f}  median {np.median(sqrt_areas):.0f}  p75 {np.percentile(sqrt_areas,75):.0f}  p90 {np.percentile(sqrt_areas,90):.0f}")
    print(f"  ship extent / image-diag: median {np.median(sqrt_areas)/np.median(diag)*100:.1f}%  (relative target size)")
    print(f"  COCO size mix: small {sm*100:.0f}% / medium {md*100:.0f}% / large {lg*100:.0f}%")
    print(f"  large instances (area>96^2): {int((areas>=9216).sum())}")
    # crude inter-instance crowding proxy: density / image-area, higher = more bleeding risk
    crowd = dens.mean() / (np.median(whs[:,0]) * np.median(whs[:,1]) / 1e6)  # inst per megapixel
    print(f"  crowding (inst/megapixel): {crowd:.1f}  <- higher => more adjacent-mask bleeding risk under upsampling")
    return dict(name=name, density=float(dens.mean()), median_extent=float(np.median(sqrt_areas)),
                small_frac=float(sm), crowd=float(crowd), n_large=int((areas >= 9216).sum()))


if __name__ == "__main__":
    rows = [profile(n, p) for n, p in DSETS.items() if os.path.exists(p)]
    print("\n=== contrast (motivates adaptive scale/gate) ===")
    for r in rows:
        print(f"  {r['name']:16s} density={r['density']:.1f} median_extent={r['median_extent']:.0f}px small={r['small_frac']*100:.0f}% crowd={r['crowd']:.1f}")
