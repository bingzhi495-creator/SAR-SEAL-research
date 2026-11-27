"""Build NORA observed method panels from the admitted manuscript figures and released summaries."""
import argparse
from pathlib import Path
import json
import runpy
import shutil
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pymupdf as fitz

BROWN, GREEN, WINE, RULE, CREAM = "#463830", "#477163", "#A14A44", "#DED3C8", "#F8F3EB"
SHADES = ["#f8f3eb","#ded5ca","#bdab96","#8f7562","#6f4d39"]


def text(ax,x,y,label,size=8,color=BROWN,bold=False,align="left"):
    ax.text(x,y,label,fontsize=size,color=color,fontweight="bold" if bold else "normal",
            ha=align,va="center",zorder=20)


def tiles(ax,grid,x,y,size,colors=False):
    grid=np.asarray(grid)
    assert grid.shape[:2]==(8,8)
    step=size/8
    for row,col in np.ndindex(8,8):
        color=grid[row,col] if colors else SHADES[int(np.rint(4*grid[row,col]))]
        ax.add_patch(Rectangle((x+col*step,y+row*step),step,step,facecolor=color,
                               edgecolor=RULE,linewidth=.24,zorder=7))


def extend(fig,height):
    width=fig.get_figwidth()*72
    fig.set_size_inches(width/72,height/72)
    ax=fig.axes[0];ax.set_ylim(height,0)
    return ax


def save(fig,out,name,slots=()):
    out.mkdir(parents=True,exist_ok=True)
    dest=out/(name+".pdf")
    fig.savefig(dest,metadata={"CreationDate":None,"ModDate":None})
    if slots:
        base=dest.read_bytes()
        with fitz.open(stream=base,filetype="pdf") as combined:
            for source,clip,target in slots:
                with fitz.open(source) as material:
                    combined[0].show_pdf_page(fitz.Rect(target),material,0,clip=fitz.Rect(clip))
            combined.save(dest,garbage=4,deflate=True)
            (out/(name+".svg")).write_text(combined[0].get_svg_image(text_as_path=False),encoding="utf-8")
    else:
        fig.savefig(out/(name+".svg"),metadata={"Date":None})
    plt.close(fig)
    with fitz.open(dest) as made:
        made[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(out/(name+".png"))


def capture_core(core,func):
    fn=core[func];ns=fn.__globals__;saved=ns["save"];figures=[]
    ns["save"]=lambda fig,name: figures.append(fig)
    try:fn()
    finally:ns["save"]=saved
    assert len(figures)==1
    return figures[0]


def transfer(source,out,name):
    for ext in (".pdf",".svg"):
        shutil.copy2(source.with_suffix(ext),out/(name+ext))
    with fitz.open(out/(name+".pdf")) as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(out/(name+".png"))


def public_scene(out,case):
    fig=plt.figure(figsize=(516/72,202/72))
    ax=fig.add_axes((0,0,1,1),xlim=(0,516),ylim=(202,0));ax.axis("off")
    text(ax,8,13,"Observed boundary edit",10,bold=True)
    text(ax,509,13,"Instance-RSDD reserve · D-FINE · 8 × 8 display bins",8,align="right")
    mask=np.asarray(case["pooled_fraction"])
    for cx,index,label,value in [(63,0,"GT footprint","Polygon coverage"),(194,1,"Native mask","IoU 51.71"),(447,4,"Independent output","IoU 51.92")]:
        text(ax,cx,35,label,8.4,bold=True,align="center")
        tiles(ax,mask[index],cx-41,46,82)
        text(ax,cx,142,value,8,align="center")
    ax.annotate("",xy=(273,85),xytext=(241,85),arrowprops=dict(arrowstyle="-|>",color=GREEN,lw=.85))
    ax.annotate("",xy=(402,85),xytext=(365,85),arrowprops=dict(arrowstyle="-|>",color=GREEN,lw=.85))
    text(ax,321,67,"Scalar ROH",8,bold=True,align="center")
    text(ax,321,85,"Independent ranks",7.8,align="center")
    text(ax,321,102,"+4 / −4 cells",8,GREEN,True,"center")
    ax.plot([8,508],[158,158],c=RULE,lw=.8)
    text(ax,8,176,"Fitting only",8,bold=True)
    text(ax,96,176,"Source polygon → native cell area fraction → scalar ROH",8)
    save(fig,out,"observed_scene")


def public_sampling(figs,out,case):
    fn=runpy.run_path(str(figs/"method_atlas_20261126/draw_nora_scene_atlas.py"))["sampling"]
    fig,_,_=fn(None)
    ax=extend(fig,329)
    ax.plot([10,238],[259,259],c=RULE,lw=.8)
    text(ax,10,270,"Observed native mask",8,bold=True)
    tiles(ax,case["pooled_fraction"][1],12,279,44)
    text(ax,69,285,"Instance-RSDD reserve · D-FINE",7.3)
    text(ax,69,300,"8 × 8 display bins",7.3)
    text(ax,69,315,"Native mask IoU 51.71",7.3)
    save(fig,out,"observed_sampling")


def public_readout(figs,out,case):
    fn=runpy.run_path(str(figs/"method_atlas_20261126/build_roh_atlas.py"))["readout"]
    ns=fn.__globals__;orig=ns["save"];capt=[]
    ns["save"]=lambda fig,path,name:capt.append(fig)
    try:fn(out)
    finally:ns["save"]=orig
    fig=capt[0];ax=fig.axes[0]
    for artist in list(ax.texts)+list(ax.patches)+list(ax.lines):
        if hasattr(artist,"get_position"):
            x=artist.get_position()[0]
        elif hasattr(artist,"get_xdata"):
            x=min(artist.get_xdata())
        else:
            x=artist.get_extents().transformed(ax.transData.inverted()).x0
        if x<150:artist.remove()
    text(ax,12,13,"Observed GT footprint",10,bold=True)
    tiles(ax,case["pooled_fraction"][0],18,31,90)
    text(ax,12,139,"Instance-RSDD · 8 × 8 bins",8.2)
    save(fig,out,"observed_readout")


def placement(figs,out,local,case,gain):
    atlas=figs/"method_atlas_20261126"
    module=runpy.run_path(str(atlas/"draw_nora_writer_figures.py"));fn=module["paired_paths"]
    fig,slots=fn(figs,case,bool(local));ax=fig.axes[0]
    for label in ax.texts:
        if label.get_text()=="GT hull":label.set_text("GT outline" if local else "GT coverage")
        if label.get_text()=="Observed fixed match":label.set_text("Observed gain case" if gain else "Observed loss case")
    clips=[]
    if local:
        source=local/"local_figures/rsdd74_common_cap_cases.pdf"
        # Rectangles on the final manuscript plate. Both signs use the GT,
        # F, L and I panes on the same source crop, in that order.
        left = 22.5 if gain else 274.5
        with fitz.open(source) as d:
            expected = ["51.52","51.73","51.92"] if gain else ["67.73","68.00","67.85"]
            assert all(s in d[0].get_text() for s in expected)
        boxes=[(left,76),(left+152,76),(left,175),(left+76,175)]
        for (x,y),target in zip(boxes,slots):
            clips.append((source,(x,y,x+67,y+67),target))
    save(fig,out,"observed_placement" if gain else "observed_placement_loss",clips)


def signed(core,out):
    fig=capture_core(core,"allocation");ax=extend(fig,241)
    ax.plot([8,508],[174,174],c=RULE,lw=.8)
    text(ax,8,186,"Observed counts",8.5,bold=True)
    text(ax,508,186,"183 SSDD images · first dual ROH fit · 115,384 flips",7.8,align="right")
    for y,tag,added,removed in [(205,"Equal",57692,57692),(226,"Variable",67777,47607)]:
        text(ax,8,y,tag,8,bold=True)
        left=83; total=420;w=total*added/(added+removed)
        ax.add_patch(Rectangle((left,y-7.3),w,14.6,fc=GREEN,ec="none"))
        ax.add_patch(Rectangle((left+w,y-7.3),total-w,14.6,fc=WINE,ec="none"))
        text(ax,left+w/2,y,f"{added:,} added",7.7,"white",True,"center")
        text(ax,left+(total+w)/2,y,f"{removed:,} removed",7.7,"white",True,"center")
    save(fig,out,"observed_allocation")


def published_bins(page,left,top):
    cells={}
    for drawing in page.get_drawings():
        bb=drawing["rect"]
        if abs(bb.width-12)>.02 or abs(bb.height-12)>.02 or drawing["fill"] is None:continue
        col=round((bb.x0-left)/12);row=round((bb.y0-top)/12)
        if 0<=col<8 and 0<=row<8 and abs(left+12*col-bb.x0)<.02 and abs(top+12*row-bb.y0)<.02:
            cells[row,col]=drawing["fill"]
    if len(cells)!=64:raise ValueError("Expected all 64 released binned coverage patches")
    return np.array([[cells[i,j] for j in range(8)] for i in range(8)])


def phoenix(core,figs,out,local):
    fig=capture_core(core,"phoenix");ax=extend(fig,300)
    ax.plot([8,508],[163,163],c=RULE,lw=.8)
    text(ax,8,176,"Observed whole-grid writes",8.7,bold=True)
    text(ax,508,176,"PSeg-SSDD · fixed Phoenix response",7.9,align="right")
    slots=[]
    centers=[67,182,327,442];titles=["Threshold","Exact total","Threshold","Exact total"]
    with fitz.open(figs/"journal_20261126/pseg_budget_cases_binned.pdf") as pub:
        for i,(cx,title,stats) in enumerate(zip(centers,titles,["38 flips  ·  56.40","16 flips  ·  62.26","75 flips  ·  58.75","16 flips  ·  56.64"])):
            text(ax,cx,202,title,7.6,bold=True,align="center")
            target=(cx-30,209,cx+30,269)
            if local:
                x=[31.5,136.5,283.5,388.5][i]
                slots.append((local/"local_figures/pseg_budget_cases_exact.pdf",(x,204,x+96,300),target))
            else:
                x,y=[(136.5,76),(31.5,204),(388.5,76),(283.5,204)][i]
                tiles(ax,published_bins(pub[0],x,y),target[0],target[1],60,colors=True)
            text(ax,cx,280,stats,7.9,align="center")
    text(ax,124,189,"Gain case",8,GREEN,True,"center");text(ax,385,189,"Loss case",8,WINE,True,"center")
    text(ax,258,293,"Flip count and continuous IoU (points) for the same instance within each pair",7.4,align="center")
    save(fig,out,"observed_phoenix",slots)


def editable_book(figs,output):
    atlas=runpy.run_path(str(figs/"method_atlas_20261126/build_NORA_method_atlas.py"))
    names=["observed_scene","observed_sampling","observed_readout","observed_placement",
           "observed_placement_loss","observed_allocation","observed_phoenix"]
    root=ET.Element("mxfile",host="app.diagrams.net",pages=str(len(names)),compressed="false")
    for name in names:
        svg=ET.parse(output/(name+".svg")).getroot()
        if svg.findall(".//{*}image"):raise ValueError("Public diagram contains an image object")
        _,_,width,height=map(float,svg.get("viewBox").split())
        page=atlas["Page"](ET.SubElement(root,"diagram",id=name,name=name.replace("observed_","").replace("_"," ").title()),width,height)
        page.subclips={}
        for clip in svg.findall(".//{*}clipPath"):
            rect=clip.find("{*}rect")
            if rect is None:raise ValueError("Unsupported nonrectangular clipping path")
            x,y,w,h=[float(rect.get(k,0)) for k in ("x","y","width","height")]
            if (x,y,w,h)!=(0,0,width,height):page.subclips["url(#"+clip.get("id")+")"]=(x,y,x+w,y+h)
        page.visit(svg)
        assert page.count>20
    ET.ElementTree(root).write(output/"observed_methods.drawio",encoding="utf-8",xml_declaration=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures",type=Path,required=True,help="Published manuscript figures directory")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--author-root",type=Path,help="Optional admitted paper_author directory")
    args=parser.parse_args();figs=args.figures.resolve()
    data=json.loads((figs/"rsdd74_common_cap_cases_safe.json").read_text(encoding="utf-8"))
    gain,loss=data["cases"]
    assert gain["swaps"]==4 and loss["swaps"]==8
    with TemporaryDirectory() as tmp:
        old=sys.argv;sys.argv=[str(figs/"revision_20261126_02/draw_core_revision.py"),"--output",tmp]
        try:core=runpy.run_path(sys.argv[0])
        finally:sys.argv=old
        for kind,local in [("public",None),("author",args.author_root)]:
            if kind=="author" and not local:continue
            dest=args.output/kind;dest.mkdir(parents=True,exist_ok=True)
            if local:
                for root,name in [("nora_scene_to_mask","observed_scene"),("nora_native_sampling","observed_sampling"),("nora_fractional_readout","observed_readout")]:
                    transfer(local/"local_method_figures"/(root+".pdf"),dest,name)
            else:
                public_scene(dest,gain);public_sampling(figs,dest,gain);public_readout(figs,dest,gain)
            placement(figs,dest,local,gain,True);placement(figs,dest,local,loss,False)
            signed(core,dest);phoenix(core,figs,dest,local)
            if not local:editable_book(figs,dest)


if __name__=="__main__":
    main()
