"""Draw NORA method diagrams from the verified schematic implementations.

The public case strip uses the released pooled 8-by-8 evaluation record. The
optional local mode clips the previously generated SAR review artwork. It does
not open predictions, author polygons, training files, or other raw inputs.
"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import runpy
from types import SimpleNamespace

import fitz
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from lxml import etree
import numpy as np


INK = "#382B26"
ORANGE = "#AB5B38"
GREEN = "#356B5D"
WINE = "#8F4958"
SAND = "#F8F3EB"
LINE = "#D7CBBE"
MUTED = "#6F635C"
FILL = "#B8A797"
WHITE = "#FFFFFF"
COOL_FILL = "#DCE9E1"
RED_FILL = "#F0DFE3"
W = 516
FONT = "DejaVu Sans"
WHITE_GREEN = LinearSegmentedColormap.from_list("occupancy", (SAND, GREEN))
WHITE_ORANGE = LinearSegmentedColormap.from_list("posterior", (SAND, "#C59C81", ORANGE, INK))
STEMS = ("nora_pairing_paths", "nora_signed_budget", "nora_frozen_writer")
SVG = "{http://www.w3.org/2000/svg}"


def require(test, reason):
    if not test:
        raise ValueError(reason)


def frame(w, h):
    fig = plt.figure(figsize=(w/72, h/72), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, w), ylim=(h, 0), frame_on=False)
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, label, size=8.1, color=INK, *, bold=False, anchor="left"):
    ax.text(x, y, label, color=color, fontsize=size, weight="bold" if bold else "normal",
            ha=anchor, va="center", clip_on=False)


def rule(ax, p1, p2, color=LINE, width=0.6, dashed=False, zorder=2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=width,
            linestyle=(0, (3.3, 2.4)) if dashed else "-", zorder=zorder)


def arrow(ax, a, b, color=ORANGE, scale=8, width=0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=color, lw=width,
                                 mutation_scale=scale, shrinkA=0, shrinkB=0, zorder=4))


def cell_center(origin, step, colrow):
    col, row = colrow
    return (origin[0] + (col+0.5)*step, origin[1] + (row+0.5)*step)


def square(ax, x, y, size, color, edge=LINE, lw=0.42, height=None):
    ax.add_patch(Rectangle((x,y), size, size if height is None else height,
                 facecolor=color, edgecolor=edge, linewidth=lw, zorder=1))


def chips(ax, origin, step, mask, add=(), drop=(), endpoints=None, edges=()):
    plus, minus = set(add), set(drop)
    for row, col in np.ndindex(mask.shape):
        key = (col,row)
        backing = COOL_FILL if key in plus else RED_FILL if key in minus else FILL if mask[row,col] else SAND
        square(ax, origin[0]+step*col, origin[1]+step*row, step, backing)
    if endpoints is not None:
        for side, tint in zip(endpoints, (GREEN, WINE), strict=True):
            for coord in side:
                x, y = cell_center(origin, step, coord)
                ax.add_patch(Circle((x,y), step*0.18, facecolor=WHITE, edgecolor=tint,
                                    lw=0.75, zorder=3))
    for a,d in edges:
        first = cell_center(origin, step, a)
        second = cell_center(origin, step, d)
        rule(ax, first, second, color=ORANGE, width=1.05, zorder=4)
    for side, glyph, tint in ((add,"+",GREEN),(drop,"−",WINE)):
        for coord in side:
            x,y = cell_center(origin, step, coord)
            ax.add_patch(Circle((x,y), step*0.31, color=tint, lw=0, zorder=5))
            ax.text(x, y, glyph, color=WHITE, va="center", ha="center",
                    fontsize=step*0.56, weight="bold", zorder=6)


def pooled(ax, grid, x, y, step):
    require(grid.shape == (8,8), "Observed public display uses the verified 8×8 grid")
    for row,col in np.ndindex(grid.shape):
        square(ax,x+step*col,y+step*row,step,WHITE_GREEN(grid[row,col]),edge=WHITE,lw=0.28)


def markers(ax, x, y, step=8.9):
    for offset, glyph, tint, name in ((0,"+",GREEN,"Added"),(51,"−",WINE,"Removed")):
        ax.add_patch(Circle((x+offset,y),step/2,fc=tint,ec="none"))
        text(ax,x+offset,y,glyph,size=step*0.62,color=WHITE,bold=True,anchor="center")
        text(ax,x+offset+8,y,name,size=7.1,color=MUTED)


def action_example(figures):
    locked = SimpleNamespace(**runpy.run_path(str(figures/"draw_selection_controls.py")))
    result = locked.actions()
    base = np.asarray(locked.example.BASE_MASK)
    endpoints = (tuple(locked.ADD), tuple(locked.example.DROP))
    for mask,adds,removes,gain in result:
        require(len(adds)==len(removes)==2 and int(mask.sum())==int(base.sum()),
                "Each schematic writer swaps two cells and preserves area")
    require(np.allclose([row[3] for row in result], (.89,.99,1.44)),
            "Exact released schematic scores")
    cap = max(sum((u-v)**2 for u,v in zip(a,d))
              for a,d in zip(result[0][1],result[0][2]))
    require(cap == 2 and all(sum((u-v)**2 for u,v in zip(a,d))<=cap
                            for a,d in zip(result[1][1],result[1][2])),
            "Only the two paired writers obey the matched cap")
    return base, endpoints, result


def paired_paths(figures, case, local):
    base, endpoints, choices = action_example(figures)
    fig,ax = frame(W,258)
    text(ax,10,12,"Shared endpoints and signed counts",size=11,bold=True)
    text(ax,506,12,"Worked example · 2 add / 2 remove",size=8,color=MUTED,anchor="right")
    rule(ax,(10,23),(506,23))
    y0=47
    chips(ax,(16,y0),10,base,endpoints=endpoints)
    text(ax,51,37,"Native mask",bold=True,anchor="center")
    text(ax,51,126,r"$S_+$ and $S_-$",size=7.6,color=MUTED,anchor="center")
    rule(ax,(97,81),(111,81),ORANGE,.8)
    rule(ax,(111,81),(111,28),ORANGE,.8)
    rule(ax,(111,28),(428,28),ORANGE,.8)
    for x in (183,305,428):
        arrow(ax,(x,28),(x,32),ORANGE,scale=5,width=.8)

    x_positions=(148,270,393)
    desc=("Distance-first","Bounded re-pairing","Independent ranking")
    limits=(r"Sets $\rho=\sqrt{2}$",r"Uses F limit $\rho=\sqrt{2}$","No pair edges")
    for j,(start,title,cap,result) in enumerate(zip(x_positions,desc,limits,choices,strict=True)):
        mask,plus,minus,gain=result
        edges=tuple(zip(plus,minus)) if j<2 else ()
        text(ax,start+35,37,f"{'FLI'[j]}  {title}",size=8.4,bold=True,anchor="center")
        chips(ax,(start,y0),10,mask,plus,minus,endpoints,edges)
        text(ax,start+35,126,rf"$\widehat g={gain:.2f}$",size=8.4,color=GREEN,anchor="center")
        text(ax,start+35,138,cap,size=7.05,color=MUTED,anchor="center")
    markers(ax,22,144)
    rule(ax,(10,156),(506,156))
    text(ax,10,168,"Observed fixed match",size=9.3,bold=True)
    text(ax,506,168,f"Instance-RSDD · D-FINE · {case['swaps']} add / {case['swaps']} remove",
         size=8,color=MUTED,anchor="right")
    centers=(51,183,305,428)
    details=("GT hull","F","L","I")
    if not local:
        grids=np.asarray(case["pooled_fraction"])
        for center,grid in zip(centers,grids[[0,2,3,4]],strict=True):
            pooled(ax,grid,center-25,184,6.25)
        text(ax,51,247,"8×8 pooled",size=7,color=MUTED,anchor="center")
    for center,title in zip(centers,details,strict=True):
        text(ax,center,179,title,size=7.3,bold=True,anchor="center")
    for center,key in zip(centers[1:],"FLI",strict=True):
        text(ax,center,245,"IoU "+f"{case['continuous_iou_points'][key]:.2f}",
             size=8,color=INK,anchor="center")
    return fig, [(c-25,184,c+25,234) for c in centers]


def area_band(ax,x,y,new_area,reference):
    for j in range(max(17,new_area)):
        mark=FILL if j<min(new_area,reference) else GREEN if j<new_area else SAND
        square(ax,x+j*4.7,y,4.7,mark,edge=WHITE,lw=.25,height=5.2)
    rule(ax,(x+4.7*reference,y-1.8),(x+4.7*reference,y+6.5),ORANGE,.85)


def signed_menu(figures, local):
    verified=SimpleNamespace(**runpy.run_path(str(figures/"draw_signed_allocation.py")))
    balanced, variable=verified.allocations()
    fig,ax = frame(W,186 if local else 174)
    text(ax,10,11,"Signed allocation at a shared total",size=11,bold=True)
    text(ax,506,11,r"Worked example · $2k_s=4$",size=8,color=MUTED,anchor="right")
    rule(ax,(10,22),(506,22))
    text(ax,53,36,"Native mask",bold=True,anchor="center")
    chips(ax,(19,46),9.55,verified.MASK,endpoints=(verified.ADD,verified.DROP))
    text(ax,53,111,"Area 15 cells",size=8,anchor="center")
    arrow(ax,(90,73),(138,73))
    text(ax,211,36,"Addition count",bold=True,anchor="center")
    bead=(145,179,213,247,281)
    rule(ax,(bead[0],73),(bead[-1],73),LINE,width=1.1)
    for num,x in enumerate(bead):
        tint=ORANGE if num==2 else GREEN if num==3 else MUTED
        ax.add_patch(Circle((x,73),3.1,fc=tint,ec="none",zorder=3))
        text(ax,x,62,str(num),size=8.8,color=tint,bold=num in (2,3),anchor="center")
        text(ax,x,86,str(4-num),size=7.7,color=MUTED,anchor="center")
    text(ax,211,100,"Removal count",size=7.6,color=MUTED,anchor="center")
    text(ax,183,118,r"Maximize the smaller $\widehat{\Delta\mathrm{IoU}}_r$",
         size=7.75,color=MUTED,anchor="center")
    rule(ax,(213,70),(226,56),ORANGE,0.9)
    rule(ax,(226,56),(226,44),ORANGE,0.9)
    rule(ax,(226,44),(322,44),ORANGE,0.9)
    arrow(ax,(322,44),(333,52),ORANGE)
    rule(ax,(247,77),(259,85),GREEN,0.9)
    rule(ax,(259,85),(259,130),GREEN,0.9)
    rule(ax,(259,130),(391,130),GREEN,0.9)
    arrow(ax,(391,130),(414,108),GREEN)

    for ordinal,(x,title,result,color) in enumerate(((336,"Balanced",balanced,ORANGE),
                                                    (424,"Variable",variable,GREEN))):
        _,adds,drops=result
        plus=tuple(verified.ADD[i] for i in adds)
        minus=tuple(verified.DROP[i] for i in drops)
        mask=verified.MASK.copy()
        for col,row in plus:
            mask[row,col]=True
        for col,row in minus:
            mask[row,col]=False
        area=int(mask.sum())
        require((len(plus),len(minus),area)==((2,2,15),(3,1,17))[ordinal],
                "Retain the verified areas and signed split")
        require(np.count_nonzero(mask!=verified.MASK)==4,"Every schematic writes four flips")
        text(ax,x+33,36,title,size=8.7,bold=True,anchor="center",color=color)
        chips(ax,(x,46),9.4,mask,plus,minus)
        text(ax,x+33,114,f"+{len(plus)} / −{len(minus)}",size=8.6,color=color,anchor="center")
        area_band(ax,x-6,140,area,15)
        text(ax,x+33,155,f"Area {area} cells",size=8.2,color=color,anchor="center")
    if local:
        text(ax,10,176,"Observed SAR context only",size=7.1,color=MUTED)
        coords=[(10,124,53,167)]
    else:
        text(ax,10,138,r"$P'=P+|A|-|D|$",size=9,color=INK)
        markers(ax,16,158)
        coords=[]
    if local:
        text(ax,61,145,r"$P'=P+|A|-|D|$",size=8.4,color=INK)
    return fig,coords


def probability_grid(ax, origin, step, field):
    for row,col in np.ndindex(field.shape):
        square(ax,origin[0]+step*col,origin[1]+step*row,step,
               WHITE_ORANGE(float(field[row,col])),edge=WHITE,lw=.35)


def phoenix(figures, local):
    verified=SimpleNamespace(**runpy.run_path(str(figures/"draw_supporting_stages_redesign.py")))
    native,prob,variants=verified.phoenix_example()
    width,height=246,324 if local else 255
    fig,ax=frame(width,height)
    text(ax,9,12,"Frozen writer controls",size=10.6,bold=True)
    text(ax,237,12,"Schematic",size=7.1,color=MUTED,anchor="right")
    rule(ax,(9,22),(237,22))
    chips(ax,(13,35),5.4,native)
    text(ax,35,30,"Native mask",size=7.3,bold=True,anchor="center")
    arrow(ax,(63,52),(80,52))
    text(ax,116,44,"Phoenix XL1",size=8.45,bold=True,anchor="center")
    text(ax,116,58,"5 iterations",size=7.1,color=MUTED,anchor="center")
    arrow(ax,(154,52),(172,52))
    probability_grid(ax,(186,35),5.4,prob)
    text(ax,209,30,r"Fixed $r_i$",size=7.6,bold=True,anchor="center")
    text(ax,103,77,"Same candidate and posterior",size=7.05,color=MUTED,anchor="center")
    rule(ax,(9,85),(237,85))
    text(ax,108,95,"Phoenix threshold",size=7.55,bold=True,anchor="center")
    text(ax,205,95,"Exact total",size=7.55,bold=True,anchor="center")
    text(ax,205,105,r"$2k_{\rm att}=4$",size=7.1,color=MUTED,anchor="center")
    rule(ax,(155,105),(155,229))
    rule(ax,(9,171),(237,171))
    for scope,y,description in (("narrow",114,r"$S_+\cup S_-$"),
                                ("whole",178,r"$\Omega$")):
        text(ax,10,y+13,"Sampled" if scope=="narrow" else "Full grid",size=7.8,bold=True)
        text(ax,10,y+26,description,size=8.1,color=MUTED)
        for writer,x in (("threshold",86),("count",183)):
            plus,minus=variants[scope,writer]
            written=native.copy()
            for col,row in plus:
                written[row,col]=True
            for col,row in minus:
                written[row,col]=False
            count=len(plus)+len(minus)
            if writer=="count":
                require(count==4,"Each exact-total schematic case writes four flips")
            chips(ax,(x,y),5.55,written,plus,minus)
            text(ax,x+22,y+49,f"{count} flips",size=7.6,color=MUTED,anchor="center")
    markers(ax,13,242)
    if local:
        rule(ax,(9,252),(237,252))
        text(ax,9,260,"Observed · PSeg-SSDD whole grid",size=7.7,bold=True)
        coords=[(49,275,86,312),(159,275,196,312)]
        text(ax,67.5,269,"Phoenix threshold",size=7.05,anchor="center")
        text(ax,177.5,269,"Exact count",size=7.05,anchor="center")
        text(ax,67.5,318,"38 flips  ·  IoU 56.40",size=6.95,anchor="center")
        text(ax,177.5,318,"16 flips  ·  IoU 62.26",size=6.95,anchor="center")
    else:
        coords=[]
    return fig,coords


def source_rects(doc, indices):
    page=doc[0]
    boxes=[]
    for item in page.get_image_info():
        box=fitz.Rect(item["bbox"])
        if all(any(abs(a-b)>1e-3 for a,b in zip(box,old)) for old in boxes):
            boxes.append(box)
    return [boxes[i] for i in indices]


def image_digests(page):
    return {sha256(page.parent.xref_stream_raw(img[0])).hexdigest()
            for img in page.get_images(full=True)}


def save(fig, out, stem, native=None, coords=(), indices=()):
    dest=out/(stem+".pdf")
    editable=out/(stem+".svg")
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for item in fig.axes[0].texts:
        box=item.get_window_extent(renderer)
        require(fig.bbox.contains(*box.min) and fig.bbox.contains(*box.max),
                "Outer text fits in "+stem)
    if native is None:
        fig.savefig(dest,metadata={"CreationDate":None,"ModDate":None})
        fig.savefig(editable,metadata={"Date":None})
        root=etree.parse(str(editable)).getroot()
        require(not root.findall(".//"+SVG+"image"),"Public method geometry stays vector-only")
    else:
        temp=out/("tmp_"+stem+".pdf")
        fig.savefig(temp,metadata={"CreationDate":None,"ModDate":None})
        with fitz.open(temp) as doc, fitz.open(native) as source:
            before=image_digests(source[0])
            source_labels=source[0].get_text()
            required={
                "nora_pairing_paths":("4 add / 4 remove","51.52","51.73","51.92"),
                "nora_signed_budget":("SAR chip","Author mask"),
                "nora_frozen_writer":("38 flips","16 flips","56.40","62.26"),
            }[stem]
            require(all(value in source_labels for value in required),
                    "The archived review figure contains the observed inset labels")
            boxes=source_rects(source,indices)
            for box,to in zip(boxes,coords,strict=True):
                doc[0].show_pdf_page(fitz.Rect(to),source,0,clip=box,keep_proportion=True)
            require(before==image_digests(doc[0]),"All original image bytes survive the native inset")
            doc.save(dest,garbage=4,deflate=False)
            editable.write_text(doc[0].get_svg_image(text_as_path=False),encoding="utf-8")
        temp.unlink()
    plt.close(fig)
    with fitz.open(dest) as pdf:
        require(len(pdf)==1 and bool(pdf[0].get_images())==(native is not None),
                "Only the optional author-reviewed files embed SAR pixels")
        require(all(s["bbox"][3]<pdf[0].rect.height for s in pdf[0].get_text("dict")["blocks"] if "lines" in s),
                "PDF text remains inside the artwork")
        return {"file":dest.name,"native":native is not None,"page_pt":[pdf[0].rect.width,pdf[0].rect.height]}


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--figures",type=Path,required=True,
                     help="Released folder containing the original verified schematic generators")
    cli.add_argument("--native-figures",type=Path,
                     help="Optional existing approved native figures PDF directory")
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    plt.rcParams.update({"font.family":FONT,"pdf.fonttype":42,"svg.fonttype":"none",
                         "svg.hashsalt":"NORA-method-paths-20261126","mathtext.fontset":"dejavusans"})
    case_data=json.loads((args.figures/"rsdd74_common_cap_cases_safe.json").read_text(encoding="utf-8"))
    case=case_data["cases"][0]
    require(case["swaps"]==4 and abs(case["independent_minus_bounded_iou_points"]-.18389090676661368)<1e-12,
            "Observed I minus L median-gain case matches its release")
    expected=("51.52","51.73","51.92")
    require(tuple(f"{case['continuous_iou_points'][arm]:.2f}" for arm in "FLI")==expected,
            "Original full-precision observations round to the verified case labels")
    source=("rsdd74_common_cap_cases.pdf","sarssp_private_cases.pdf","pseg_budget_cases_exact.pdf")
    indices=((0,2,3,4),(0,),(1,2))

    audit=[]
    for native in (False,True):
        if native and args.native_figures is None:
            continue
        out=args.out/("local" if native else "public")
        out.mkdir(parents=True,exist_ok=False)
        views=(paired_paths(args.figures,case,native),signed_menu(args.figures,native),
               phoenix(args.figures,native))
        for (fig,coords),stem,path,ids in zip(views,STEMS,source,indices,strict=True):
            approved=args.native_figures/path if native else None
            audit.append(save(fig,out,stem,approved,coords,ids))
    print(json.dumps(audit,indent=2))
    (args.out/"VALIDATION.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()

