"""Illustrate median positive/negative GF-3 cases on the closed detector-5203 replay."""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import ModuleType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import shapely


SCORE_SHA = "fd0d832de7b518dd96da11aedcca8a62ca99bbb91cf782d815b5369b9c93a039"
WORKER = "06e5da31b3f480f32600cda3af924fdc5ecd5c14bec2fe31a08892d53927a3c3"
ARMS = ("raw", "dynamic", "5217/learned")
BLUE, TEAL, RED, INK = "#315981", "#087960", "#ab4232", "#1b2735"


def need(test, detail):
    if not test:
        raise ValueError(detail)


def read(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Digest mismatch for " + path.name)
    return raw


def trusted_module(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(read(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def population(args):
    data = args.run
    end = json.loads((data / "observed_score/TERMINAL.json").read_bytes())
    need(end["phase"] == "score" and end["output_footer_verified"] is True and
         end["actual_exit"] == 0 and end["result_sha256"] == SCORE_SHA and
         end["worker_sha256"] == WORKER, "Required historical score closure")
    report = json.loads(read(data / "score/RESULT_SAFE.json", SCORE_SHA))
    need(report["detector_seed"] == 5203 and report["signed_budgets_verified"] is True and
         report["worker_sha256"] == WORKER, "Wrong frozen evaluation")
    for phase in ("capture", "replay"):
        t = json.loads(read(data / ("observed_" + phase) / "TERMINAL.json", report[phase + "_terminal_sha256"]))
        s = json.loads(read(data / phase / "RESULT_SAFE.json", report[phase + "_result_sha256"]))
        need(t["actual_exit"] == 0 and t["output_footer_verified"] is True and t["phase"] == phase
             and t["result_sha256"] == report[phase + "_result_sha256"] and t["worker_sha256"] == s["worker_sha256"]
             and s["target_labels_opened"] is False, "Unaccepted frozen inputs")
    score = trusted_module(args.code / "INSTANCE_RSDD_CARRIER5203_20261123/score.py", WORKER)
    old = score.load(args.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", score.OLD_SCORE)
    evaluator = old.module(args.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(args.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(args.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    def stream(path, key):
        return map(json.loads, read(data / path, report[key]).splitlines())
    streams = (stream("capture/PREDICTIONS_PRIVATE.jsonl", "predictions_sha256"),
               stream("capture/CANDIDATE_AUDIT_PRIVATE.jsonl", "audit_sha256"),
               stream("replay/EDITS_PRIVATE.jsonl", "edits_sha256"))
    blocks = {}
    for pred, aud, ledger in zip(*streams, strict=True):
        i = pred["image_id"]
        need(i == aud["image_id"] == ledger["image_id"], "Misregistered inputs")
        if aud["sensor"] != "GF-3":
            continue
        ranks = [x["position"] for x in aud["candidates"]]
        need(ranks == list(range(1, len(pred["coco"])+1)), "Invalid compact positions")
        blocks[i] = dict(stock=pred["coco"], ranks=ranks, changes={}, group=aud["acquisition_group"],
                         rgb=aud["rgb_sha256"], file=aud["source_image_name"],
                         edits={int(pos): {k: v for k, v in options.items() if k in ARMS}
                                for pos, options in ledger["changes"].items()})
    gt = json.loads(read(args.author_labels, report["annotation_sha256"]))
    gt, _ = old.coco_annotation_ids({"info": {}, **gt})
    gt = dict(gt, annotations=[a for a in gt["annotations"] if a["image_id"] in blocks],
              images=[im for im in gt["images"] if im["id"] in blocks])
    polygons = {a["id"]: shapely.Polygon(np.array(a["segmentation"][0]).reshape(-1, 2)) for a in gt["annotations"]}
    metrics, anchors = evaluator.score_one(gt, blocks, "raw", anchors=True)
    published = report["views"]["gf3"]
    need(len(blocks) == 583 and len(polygons) == 631 and len(anchors) == 628 and
         abs(metrics["AP"] - published["metrics_fraction"]["raw"]["AP"]) < 1e-12,
         "Historical raw population not reproduced")
    rows, grouped = [], defaultdict(list)
    for gid, (i, pos, pred) in anchors.items():
        raw = pred["segmentation"]
        area = int(core.codec.area(raw))
        if not 32 <= area < 1024:
            continue
        opts = blocks[i]["edits"].get(pos, {})
        k = opts.get("dynamic", {}).get("swaps", 0)
        need(opts.get(ARMS[2], {}).get("swaps", 0) == k, "Count mismatch on fixed raw match")
        masks = [opts.get(arm, {}).get("rle", raw) for arm in ARMS]
        poly = polygons[gid]
        need(poly.is_valid and poly.area > 0, "Unexpected invalid Gaofen-3 truth")
        ious = [core.geometric_iou(m, poly, shapely) for m in masks]
        grouped[blocks[i]["group"]].append(ious)
        rows.append(dict(i=i, pos=pos, gid=gid, k=k, poly=poly, rles=masks, ious=ious,
                         delta=ious[2]-ious[1]))
    actual = np.mean([np.mean(v, axis=0) for v in grouped.values()],axis=0)*100
    for j, arm in enumerate(ARMS):
        expected = published["geometry"]["group_equal"][arm]["continuous_iou_points"]
        need(abs(actual[j]-expected) < 1e-9, "Published fixed geometry parity failed for " + arm)
    need(len(rows) == 425, "Expected all 425 eligible anchors")
    groups = {"gain": [r for r in rows if r["delta"] > 1e-12],
              "loss": [r for r in rows if r["delta"] < -1e-12]}
    selected, used, medians = [], set(), {}
    for name, pool in groups.items():
        medians[name] = float(np.median([r["delta"] for r in pool]))
        ranked = sorted(pool, key=lambda r: (abs(r["delta"]-medians[name]), r["i"], r["pos"], r["gid"]))
        n = 0
        for r in ranked:
            if r["i"] in used:
                continue
            selected.append(r)
            used.add(r["i"])
            n += 1
            if n == 2:
                break
        need(n == 2, "Two distinct near-median images per sign required")
    counts = {name: len(v) for name,v in groups.items()}
    counts["unchanged"] = len(rows)-sum(counts.values())
    return core, blocks, selected, counts, medians, score.MANIFEST[740]


def case_image(row, block, item, intake, core):
    i = row["i"]
    need(i == item["image_id"] and item["source_image_name"] == block["file"] and
         item["rgb_sha256"] == block["rgb"], "Original image mismatch")
    compressed = read(intake / item["image_ref"]["path"], item["image_ref"]["sha256"])
    rgb = np.asarray(Image.open(BytesIO(compressed)).convert("RGB"))
    need(sha256(rgb.tobytes()).hexdigest() == block["rgb"], "Decoded pixel mismatch")
    masks = [core.codec.decode(rle).astype(bool) for rle in row["rles"]]
    for m in masks[1:]:
        need(np.count_nonzero(m & ~masks[0]) == np.count_nonzero(masks[0] & ~m) == row["k"],
             "Per-case signed mask-count mismatch")
    yy, xx = np.where(np.logical_or.reduce(masks))
    poly = row["poly"]
    x0,y0,x1,y1 = poly.bounds
    x0,y0 = min(xx.min(), x0), min(yy.min(), y0)
    x1,y1 = max(xx.max()+1, x1), max(yy.max()+1, y1)
    # At least four native samples per public map tile.
    side = min(512, 8*int(np.ceil(max(32, max(x1-x0,y1-y0)+8)/8)))
    left = int(np.clip(np.floor((x0+x1-side)/2),0,512-side))
    top = int(np.clip(np.floor((y0+y1-side)/2),0,512-side))
    ys,xs = slice(top,top+side),slice(left,left+side)
    grid_y,grid_x = np.mgrid[ys,xs]
    cells = shapely.box(grid_x, grid_y, grid_x+1, grid_y+1)
    fraction = shapely.area(shapely.intersection(cells, poly))
    cropped = np.array([m[ys,xs] for m in masks])
    need(abs(float(fraction.sum())-poly.area) < 1e-7 and
         all(m.sum()==p.sum() for m,p in zip(cropped,masks)), "Crop omitted relevant geometry")
    inter = np.sum(cropped*fraction, axis=(1,2))
    derived = inter/(cropped.sum(axis=(1,2))+fraction.sum()-inter)
    need(np.allclose(derived, row["ious"],rtol=0,atol=1e-12), "Plot overlay and scorer disagree")
    return dict(rgb=rgb[ys,xs], masks=cropped, truth=fraction, contour=np.array(poly.exterior.coords)-[left,top],
                ious=np.array(row["ious"])*100, k=row["k"], side=side, inter=inter)


def layout():
    fig, axes = plt.subplots(4,4,figsize=(7.15,7.2))
    fig.subplots_adjust(left=.102,right=.992,bottom=.102,top=.943,wspace=.105,hspace=.30)
    return fig, axes


def decorations(fig, axes, cases, public):
    titles = ("Native mask", "Dynamic attraction", "Learned occupancy", "Overlap change")
    for ax,t in zip(axes[0],titles,strict=True):
        ax.set_title(t,fontsize=9,fontweight="semibold",pad=5,color=INK)
    for i,(row,c) in enumerate(zip(axes,cases,strict=True)):
        for j, ax in enumerate(row):
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(.45)
                spine.set_edgecolor("#a6afb8")
            v = c["ious"]
            text = f"IoU  {v[j]:.2f}" if j<3 else f"ΔIoU  {v[2]-v[1]:+.2f} pt"
            ax.set_xlabel(text,fontsize=8,labelpad=2.5,color=INK if j<3 else (TEAL if v[2]>v[1] else RED))
        bounds = row[0].get_position()
        fig.text(.012,bounds.y0+bounds.height*.53,f"({'abcd'[i]})\n{'Gain' if i<2 else 'Loss'}\n$k$ = {c['k']}",
                 fontsize=8,ha="left",va="center",linespacing=1.48,color=INK)
    if public:
        legend = [Patch(facecolor=BLUE,label="Mask fraction"), Line2D([],[],color=INK,ls="--",label="50% reference contour"),
                  Line2D([],[],color=TEAL,marker="+",ls="none",markeredgewidth=1.2,label="Added"),
                  Line2D([],[],color=RED,marker="x",ls="none",markeredgewidth=1.1,label="Removed")]
        note = "Each square pools native pixels. Right column shows change in true intersection, learned minus dynamic."
    else:
        legend = [Line2D([],[],color="#40c9ff",label="Native / edited mask"), Line2D([],[],color="#f2c654",ls="--",label="Author hull"),
                  Patch(facecolor="#4de6ba",label="Added"),Patch(facecolor="#f0708b",label="Removed")]
        note = "Right column shows the change in true intersection on the registered native cells."
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.51,.038),ncol=4,frameon=False,
               fontsize=7.6,handlelength=1.6,columnspacing=1.25)
    fig.text(.51,.026,note,ha="center",fontsize=7.2,color=INK)
    fig.text(.51,.010,"Teal increases intersection; rust decreases it. IoU is reported in points (0–100).",
             ha="center",fontsize=7.2,color=INK)


def pooled(x):
    width = x.shape[-1]//8
    return x.reshape(8,width,8,width).sum(axis=(1,3))


def sign_colors(z,limit):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("signed", [RED,"#fafaf7",TEAL])
    return cmap,matplotlib.colors.TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit)


def public_figure(cases):
    fig, aa = layout()
    shades = ListedColormap(["#f7f8f7","#dbe5eb","#a7c1d3","#7599b4",BLUE])
    contrasts = [pooled((m["masks"][2].astype(float)-m["masks"][1])*m["truth"]) for m in cases]
    limit = max(float(np.abs(x).max()) for x in contrasts)
    cmap, norm = sign_colors(None,limit)
    for c,row,z in zip(cases,aa,contrasts,strict=True):
        n = (c["side"]//8)**2
        truth = pooled(c["truth"])/n
        for j,ax in enumerate(row):
            if j<3:
                foreground = pooled(c["masks"][j])/n
                ax.pcolormesh(np.arange(9),np.arange(9),np.rint(4*foreground),cmap=shades,vmin=0,vmax=4)
                if j:
                    raw,m = c["masks"][0],c["masks"][j]
                    for bits,marker,clr,off in [(m&~raw,"+",TEAL,-.18),(raw&~m,"x",RED,.18)]:
                        y,x=np.nonzero(pooled(bits))
                        ax.scatter(x+.5+off,y+.5,marker=marker,s=20,color=clr,linewidths=1.25,zorder=4)
            else:
                ax.pcolormesh(np.arange(9),np.arange(9),z,cmap=cmap,norm=norm,edgecolors="#e4e5e3",linewidth=.25)
                delta_inter=float(c["inter"][2]-c["inter"][1])
                ax.text(.04,.06,f"{delta_inter:+.2f} px²",transform=ax.transAxes,fontsize=7.1,color=INK,
                        bbox=dict(facecolor="white",edgecolor="none",alpha=.86,pad=1.3))
            ax.contour(np.arange(8)+.5,np.arange(8)+.5,truth,levels=[.5],colors=INK,linestyles="--",linewidths=.85)
            ax.set(xlim=(0,8),ylim=(8,0))
    decorations(fig,aa,cases,True)
    return fig


def edges(mask):
    y,x=np.nonzero(mask)
    segments = []
    for dy,dx,p,q in [(-1,0,(0,0),(1,0)),(1,0,(0,1),(1,1)),
                      (0,-1,(0,0),(0,1)),(0,1,(1,0),(1,1))]:
        ok=(y+dy>=0)&(y+dy<mask.shape[0])&(x+dx>=0)&(x+dx<mask.shape[1])
        exposed=np.ones(len(y),bool)
        exposed[ok] = ~mask[y[ok]+dy,x[ok]+dx]
        points=np.stack([x[exposed],y[exposed]],axis=1)
        segments.extend(np.stack([points+p,points+q],axis=1))
    return segments


def detailed_figure(cases):
    fig, aa=layout()
    contrasts=[(c["masks"][2].astype(float)-c["masks"][1])*c["truth"] for c in cases]
    cmap,norm=sign_colors(None,1.)
    for c,row,z in zip(cases,aa,contrasts,strict=True):
        side=c["side"]
        extent=(0,side,side,0)
        for j,ax in enumerate(row):
            if j<3:
                ax.imshow(c["rgb"],extent=extent,interpolation="nearest")
                ax.add_collection(LineCollection(edges(c["masks"][j]),colors="#40c9ff",linewidths=.8))
                if j:
                    raw,m=c["masks"][0],c["masks"][j]
                    colors=np.zeros((*raw.shape,4))
                    colors[m&~raw]=matplotlib.colors.to_rgba("#4de6ba",.91)
                    colors[raw&~m]=matplotlib.colors.to_rgba("#f0708b",.91)
                    ax.imshow(colors,extent=extent,interpolation="nearest")
                line="#f2c654"
            else:
                ax.imshow(z,cmap=cmap,norm=norm,extent=extent,interpolation="nearest")
                line=INK
                ax.text(.04,.055,f"{c['inter'][2]-c['inter'][1]:+.2f} px²",transform=ax.transAxes,
                        fontsize=7.1,bbox=dict(facecolor="white",edgecolor="none",pad=1.4))
            ax.plot(*c["contour"].T,color=line,lw=.75,ls=(0,(2.4,1.6)))
            ax.set(xlim=(0,side),ylim=(side,0))
    decorations(fig,aa,cases,False)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("code","run","intake","author-labels","public-output","private-output"):
        p.add_argument("--"+key,required=True,type=Path)
    args=p.parse_args()
    public=args.public_output.resolve()
    private=args.private_output.resolve()
    need(public!=private and public not in private.parents, "Detailed panels require a separate private directory")
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","svg.hashsalt":"gf3-cases-5203",
                         "pdf.fonttype":42,"figure.dpi":120,"savefig.dpi":240,"axes.unicode_minus":True})
    core,blocks,rows,counts,medians,manifest_sha=population(args)
    manifest=json.loads(read(args.intake/"IMAGE_MANIFEST_PRIVATE.json",manifest_sha))
    by_id={r["image_id"]:r for r in manifest["items"]}
    cases=[case_image(r,blocks[r["i"]],by_id[r["i"]],args.intake,core) for r in rows]
    private.mkdir(parents=True,exist_ok=True)
    public.mkdir(parents=True,exist_ok=True)
    for target,tag,fig in ((public,"gf3_matched_cases_binned",public_figure(cases)),
                           (private,"gf3_matched_cases_SAR_PRIVATE",detailed_figure(cases))):
        fig.savefig(target/(tag+".svg"),metadata={"Date":None})
        fig.savefig(target/(tag+".pdf"),metadata={"CreationDate":None,"ModDate":None})
        if target==private:
            fig.savefig(target/(tag+".png"),dpi=300)
        plt.close(fig)
    sel=[{"image_id":r["i"],"gt_rekeyed":r["gid"],"compact_position":r["pos"],"swaps":r["k"],
          "ious_points":c["ious"].tolist(),"delta_intersection_pixels2":float(c["inter"][2]-c["inter"][1])}
         for r,c in zip(rows,cases,strict=True)]
    (private/"case_selection_PRIVATE.json").write_text(json.dumps(sel,indent=2),encoding="utf8")
    summary=dict(eligible=425, counts=counts, median_delta_points={k:100*v for k,v in medians.items()},
                 cases=[{k:v for k,v in row.items() if k in ("swaps","ious_points","delta_intersection_pixels2")} for row in sel])
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
