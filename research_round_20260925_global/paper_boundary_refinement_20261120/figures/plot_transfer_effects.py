"""Regenerate the editable transfer-effects figure from the public aggregate results.

Run from the published release with Python 3.10+ and matplotlib:
    python plot_transfer_effects.py
Use --results to point at a separate copy of the release's results directory.
"""
import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SOURCES = {
    "df": ("DFINE_E24_DEV705_FREE_ENDPOINT_20261120_SCORE_REPLAY_SAFE.json",
           "e32f2e6f45fb2e1a777765a57b021877e0803588285c6868aa2a915c5c6542e8"),
    "rf2": ("RFDETR_CENTER5202_DEV705_MATCHED_SCORE_PUBLIC_SAFE_20261120.json",
            "c0be05b24ca1bb23283ac19043b2ad96bbf2db39bc0ef525e1ea41a3dc9316e9"),
    "rf3": ("RFDETR_COMMON5203_DEV705_FROZEN_SCORE_ACTUAL_20261122.json",
            "16c973bf4acf950e5213de2567cb98ea138a697e75e0a8aeab3eae0758d121c2"),
    "official": ("HRSID_OFFICIAL_FROZEN_REFINEMENT_SCORE_ACTUAL_20261122.json",
                 "90ef9abb8387eb1ede4d6cd7979f08d4c57e75997347df448ab34af2bd33411f"),
    "gf17": ("INSTANCE_RSDD_FROZEN_HRSID_5217_SAFE_20261122.json",
             "59936bb4ceb70db1af4801f8d31d1ba84f223adb688759c795d8a71e58f7da4b"),
    "gf18": ("INSTANCE_RSDD_FROZEN_HRSID_5218_SAFE_20261122.json",
             "a8cf7537d8a42c6c17cc08b11d4699354b3eedc3fb26cb2ee24150eef7b1d90a"),
    "factorial": ("INDEPENDENT_SIDE_FACTORIAL_ACTUAL_SAFE_20261123.json",
                  "098191b4448f02dd7bf2230998493417a9dfb79c3cd4a137b80fc0962d33dd75"),
    "external3": ("INSTANCE_RSDD_CARRIER5203_ACTUAL_SAFE_20261123.json",
                  "16cd6d1ab2d688b98c7b18e02199803bc7c5003f5be8643a8ced278cec05f6d7"),
    "pseg": ("PSEG_TEST232_CLASSICAL_ATTRACTION_MATCHED_ACTUAL_20261122.json",
             "3efc20ab1218d2f36b21e5967a9f178488af32d0a3e06c78e64a1ca8062b7365"),
    "ueikap": ("UEIKAP_ADDITIONAL_FAMILIES_ACTUAL_SAFE_20261123.json",
               "962c7e600bb1c6dd4bf3c97bbea4b3dde0e9e2f9f616121a51cf3a844ed287a1"),
}


@dataclass
class Row:
    label: str
    population: str
    ap: tuple
    iou: tuple  # (estimate, lower, upper) or empty when unavailable


def read(results, key):
    name, expected = SOURCES[key]
    raw = (results/name).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Unexpected public result digest for "+name)
    return json.loads(raw)


def ap_difference(metrics, first, second):
    return 100*(metrics[first]["AP"]-metrics[second]["AP"])


def interval(record, value, limits):
    return (record[value],*record[limits])


def rows(results):
    df = read(results,"df")
    geom = df["geometry_points"]
    first = Row("HRSID DEV705 · D-FINE", f'{geom["counts"]["eligible"]:,} matches · {geom["source_groups"]} groups',
                (ap_difference(df["coco_fraction"],"free_matched_count","fixed"),),
                interval(geom["comparisons"]["free_matched_count_minus_fixed"],"delta_points","interval95_points"))
    rf2 = read(results,"rf2")
    rf3 = read(results,"rf3")
    seeds = ("5217","5218","5219")
    def dev_effects(coco):
        return tuple(100*(coco[s]["SU=F"]["primary"]["AP"]-coco[s]["SF"]["primary"]["AP"]) for s in seeds)
    supplement=rf2["independent_audit"]["position_only_supplement"]
    second = Row("HRSID DEV705 · RF-DETR 5202",
                 f'{rf2["geometry_summary"]["counts"]["valid"]:,} matches · {rf2["geometry_summary"]["source_groups"]} groups',
                 dev_effects(rf2["coco"]),interval(supplement,"IoU_points","source_interval95_points"))
    third = Row("HRSID DEV705 · RF-DETR 5203",f'{rf3["eligible_valid_matches"]:,} matches · {rf3["source_prefixes"]} groups',
                dev_effects(rf3["per_seed_COCO"]),
                interval(rf3["paired_contrasts"]["SU=F minus SF"]["mean_fixed_iou_points"],"5203","5203_prefix_interval95"))
    official = read(results,"official")
    ap = tuple(100*(official["treatments"][s]["SU=F"]["official"]["metrics"]["AP"]
                    -official["treatments"][s]["SF"]["official"]["metrics"]["AP"]) for s in seeds)
    fourth = Row("HRSID official · RF-DETR 5202", f'{official["native"]["official"]["images"]:,} test images',ap,())
    datasets = [read(results,k)["views"]["gf3"] for k in ("gf17","gf18")]
    factorial = read(results,"factorial")["cohorts"]["740"]["score"]["views"]["gf3"]["geometry"]
    ap = tuple(ap_difference(x["metrics_fraction"],"free_matched_count","fixed") for x in datasets)
    # Both sides independent (II) minus both fixed (FF) is precisely the original matched-count comparison.
    # Use its paired two-head bootstrap. Separate single-head interval endpoints cannot be averaged.
    pooled = factorial["pooled"]["II_minus_FF"]["continuous_iou_points"]
    single = [x["geometry"]["stats"]["paired_contrasts_points"]["free_matched_count minus fixed"]
              ["continuous_iou_points"]["delta"] for x in datasets]
    if abs(mean(single)-pooled["delta"])>1e-10:
        raise ValueError("The pooled geometry population differs from the two prior head scores")
    fifth = Row("RSDD Gaofen-3 · RF-DETR 5202",
                f'{factorial["counts"]["eligible_matches"]} matches · {factorial["represented_groups"]} groups',
                ap,interval(pooled,"delta","interval_95"))
    gf3 = read(results,"external3")["scope"]["author_test740"]["views"]["gf3"]
    m,g = gf3["metrics_fraction"],gf3["geometry"]
    sixth = Row("RSDD Gaofen-3 · RF-DETR 5203",
                f'{g["counts"]["eligible_matches"]} matches · {g["represented_groups"]} groups',
                tuple(ap_difference(m,s+"/free_matched_count",s+"/fixed") for s in seeds),
                interval(g["effects"]["independent_minus_fixed"]["head_mean"]["continuous_iou_points"],"delta","interval_95"))
    key = "learned_at_attraction_count minus attraction_dynamic"
    pseg = read(results,"pseg")
    view = "exclude_exact_fit_overlap"
    geom = pseg["geometry"][view]
    seventh = Row("PSeg-SSDD · D-FINE", f'{geom["eligible_pairs"]} matches · {geom["decoded_image_groups_with_pairs"]} image groups',
                  (ap_difference(pseg["views"][view]["metrics_fraction"],"learned_at_attraction_count","attraction_dynamic"),),
                  interval(geom["pairwise"][key]["continuous_iou_points"],"delta","percentile95"))
    ue = read(results,"ueikap")["phases"]["score"]
    hulls = ue["labeled_hulls_and_coverage"]["labeled_hulls"]
    eighth = Row("UEIKAP · RF-DETR 5202",f'{hulls} labeled hulls · {ue["groups"]} groups',(),
                 interval(ue["primary_all_labeled_hulls"]["contrasts"][key],"mean_points","paired_95_interval_points"))
    return [first,second,third,fourth,fifth,sixth,seventh,eighth]


NAVY,TEAL,ORANGE = "#193954","#116c69","#a54a19"
INK,MUTED,GRID = "#162c39","#455a66","#d5dcdf"


def draw(data, out):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.1,"text.color":INK,
                         "axes.labelcolor":INK,"xtick.color":MUTED,"svg.fonttype":"none",
                         "pdf.fonttype":42,"svg.hashsalt":"sar-seal-transfer-effects-v1"})
    fig=plt.figure(figsize=(7.16,4.58),facecolor="white")
    # Fixed grid in physical page units keeps all row labels directly aligned with both outcome panels.
    left,bottom,height=.018,.145,.729
    ax_ap=fig.add_axes([.376,bottom,.245,height])
    ax_io=fig.add_axes([.704,bottom,.245,height])
    labels=fig.add_axes([left,bottom,.353,height])
    limits=(-.75,8.5)
    ys=[7.55,6.59,5.63,4.67,3.71,2.75,.86,-.10]
    for ax,range_,ticks in ((ax_ap,(-.12,1.47),(0,.4,.8,1.2)),(ax_io,(-.54,.85),(-.4,0,.4,.8))):
        ax.set(xlim=range_,ylim=limits,yticks=[],xticks=ticks,axisbelow=True)
        ax.grid(axis="x",color=GRID,linewidth=.55)
        ax.axvline(0,linewidth=.85,color="#617783",zorder=1)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x",labelsize=8.0,length=2.5,pad=2)
    labels.set(xlim=(0,1),ylim=limits)
    labels.axis("off")
    for i,(row,y) in enumerate(zip(data,ys)):
        top=i<6
        labels.text(0,y+.11,row.label,fontsize=8.05,va="center")
        labels.text(0,y-.23,row.population,fontsize=8.0,va="center",color=MUTED)
        color=NAVY if top else ORANGE
        if row.ap:
            if len(row.ap)>1:
                ax_ap.hlines(y,min(row.ap),max(row.ap),color=color,linewidth=.8,zorder=2)
                for j,x in enumerate(row.ap):
                    ax_ap.scatter(x,y+.13+.055*(j-(len(row.ap)-1)/2),s=13,facecolor="white",
                                  edgecolor=color,linewidth=.75,zorder=3)
            ax_ap.scatter(mean(row.ap),y,s=18,marker="D",color=color,zorder=4)
        else:
            ax_ap.text(.28,y,"Not evaluated",fontsize=8,color=MUTED,va="center")
        if row.iou:
            val,lo,hi=row.iou
            color=TEAL if top else ORANGE
            ax_io.plot([lo,hi],[y,y],color=color,lw=1.2)
            ax_io.vlines([lo,hi],y-.10,y+.10,color=color,lw=.9)
            ax_io.scatter(val,y,s=22,color=color,marker="s",zorder=4)
        else:
            ax_io.text(.025,y,"Not evaluated",fontsize=8,color=MUTED,va="center")
    # Put contrast headings in the wide gaps between cohorts, outside both plotting axes.
    def fy(y):
        return bottom+height*(y-limits[0])/(limits[1]-limits[0])
    for y,text,color in ((8.32,"A   Independent − fixed endpoint pairing",NAVY),
                         (1.67,"B   Learned ranking − dynamic spatial attraction",ORANGE)):
        fig.text(left,fy(y),text,weight="bold",color=color,fontsize=8.6,va="center",
                 bbox=dict(facecolor="white",edgecolor="none",pad=1))
    fig.text(left,.934,"Evaluation population",weight="bold",fontsize=9.0)
    fig.text(.376,.934,"Mask AP difference",weight="bold",fontsize=9.0)
    fig.text(.704,.934,"Continuous IoU difference",weight="bold",fontsize=9.0)
    fig.text(.376,.897,"percentage points",fontsize=8.0,color=MUTED)
    fig.text(.704,.897,"percentage points",fontsize=8.0,color=MUTED)
    y=.044
    fig.add_artist(Line2D([.020,.053],[y,y],transform=fig.transFigure,color=NAVY,linewidth=.8))
    fig.add_artist(Line2D([.020,.053],[y,y],transform=fig.transFigure,color=NAVY,marker="o",mfc="white",
                          markersize=3.4,linewidth=0))
    fig.text(.061,y,"Individual head seeds and range",va="center",fontsize=8.0)
    fig.add_artist(Line2D([.400],[y],transform=fig.transFigure,color=NAVY,marker="D",markersize=3.9))
    fig.text(.416,y,"Mean AP",va="center",fontsize=8.0)
    fig.add_artist(Line2D([.567,.607],[y,y],transform=fig.transFigure,color=TEAL,lw=1.1,marker="|",markersize=5))
    fig.add_artist(Line2D([.587],[y],transform=fig.transFigure,color=TEAL,marker="s",markersize=4.1))
    fig.text(.616,y,"Paired estimate and 95% group interval",va="center",fontsize=8.0)
    fig.savefig(out/"transfer_effects.pdf",metadata={"Title":"Matched edit effects across SAR cohorts",
                "Creator":"SAR-SEAL research","CreationDate":None,"ModDate":None})
    fig.savefig(out/"transfer_effects.svg",metadata={"Title":"Matched edit effects across SAR cohorts",
                "Creator":"SAR-SEAL research","Date":None})
    plt.close(fig)


if __name__=="__main__":
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=here.parent.parent/"results")
    parser.add_argument("--output",type=Path,default=here)
    args=parser.parse_args()
    draw(rows(args.results),args.output)
