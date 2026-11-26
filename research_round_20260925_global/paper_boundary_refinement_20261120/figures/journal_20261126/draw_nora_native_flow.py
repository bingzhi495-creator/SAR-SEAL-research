"""Editable vector diagrams for NORA. Both figures use a checked worked example."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np
from scipy.optimize import linear_sum_assignment


NAVY, BLUE, TEAL, RUST, OCHRE, GRAY = ("#473830", "#A95B36", "#477163", "#A14A44", "#B07B3E", "#756C65")
WHITE, LINE, PALE, SKY, MINT, CLAY, SAND = ("#FFFFFF", "#DED3C8", "#F5EEE4", "#F2E4DA", "#E5EEE7", "#F7E8E4", "#F7EBDC")
mpl.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],
                     "mathtext.fontset":"dejavusans", "svg.fonttype":"none", "pdf.fonttype":42,
                     "svg.hashsalt":"nora-native-flow-20261126"})


def paper(width,height):
    fig=plt.figure(figsize=(width/72,height/72),facecolor=WHITE)
    ax=fig.add_axes((0,0,1,1),xlim=(0,width),ylim=(height,0))
    ax.axis("off")
    return fig,ax


def text(ax,x,y,s,pt=8,*,color=NAVY,bold=False,align="left"):
    return ax.text(x,y,s,fontsize=pt,color=color,fontweight="bold" if bold else "normal",
                   ha=align,va="center",zorder=10)


def center(ax,x,y,s,pt=8,**kw):
    return text(ax,x,y,s,pt,align="center",**kw)


def box(ax,x,y,w,h,fill=WHITE,edge=LINE,lw=.75,dash=False,z=1):
    artist=Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,linewidth=lw,
                     linestyle=(0,(3,2)) if dash else "-",zorder=z)
    ax.add_patch(artist)
    return artist


def edge(ax,start,end,color=GRAY,width=.75,dash=False,z=5):
    x,y=start
    xx,yy=end
    ax.plot((x,xx),(y,yy),color=color,lw=width,ls=(0,(3,2)) if dash else "-",zorder=z)


def arrow(ax,start,end,color=BLUE,dash=False,z=6):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=7,color=color,lw=.9,
                                 linestyle=(0,(4,2)) if dash else "-",shrinkA=0,shrinkB=0,zorder=z))


def title(ax,x,y,letter,heading):
    text(ax,x,y,f"({letter})",8.6,color=BLUE,bold=True)
    text(ax,x+16,y,heading,9.7,bold=True)


def neighborhood(mask,dilation):
    padded=np.pad(mask,1)
    height,width=mask.shape
    offsets=[padded[y:y+height,x:x+width] for y in range(3) for x in range(3)]
    return np.logical_or.reduce(offsets) if dilation else np.logical_and.reduce(offsets)


def example():
    native=np.zeros((8,12),dtype=bool)
    for row,left,right in ((1,5,8),(2,3,9),(3,2,10),(4,2,11),(5,3,10),(6,5,8)):
        native[row,left:right]=True
    inside_box=np.zeros_like(native)
    inside_box[:,1:11]=True
    outside=neighborhood(neighborhood(native,True),True)&~native&inside_box
    inside=native&~neighborhood(neighborhood(native,False),False)&inside_box
    n=min(32,int(outside.sum()),int(inside.sum()))
    assert 32<=native.sum()<1024 and n>=4

    # A smooth native boundary response with deterministic row-major ties.
    yy,xx=np.indices(native.shape)
    p=np.where(native,.66,.34)
    bias=(7*xx+3*yy)%11/100
    p=np.where(inside,.51+bias,p)
    p=np.where(outside,.49-bias,p)

    def select(support):
        rows,cols=np.nonzero(support)
        order=np.lexsort((rows*native.shape[1]+cols,abs(p[rows,cols]-.5)))[:n]
        row_major=np.argsort(rows[order]*native.shape[1]+cols[order],kind="stable")
        return np.column_stack((cols[order[row_major]],rows[order[row_major]]))

    additions,removals=select(outside),select(inside)
    distances=((additions[:,None,:]-removals[None,:,:])**2).sum(axis=2)
    ai,di=linear_sum_assignment(distances)
    assert list(ai)==list(range(n))
    q_add=np.full(n,.20)
    q_drop=np.full(n,.78)
    positions=(1,n//3,n//2,n-2)
    q_add[list(positions)]=(.90,.84,.77,.02)
    q_drop[di[list(positions)]]=(.10,.24,.78,.04)
    q_diff=q_add-q_drop[di]
    limit=min(8,int(native.sum())//10,n)
    k=min(limit,int((q_diff>0).sum()))
    ai=np.lexsort((np.arange(n),-q_add))[:k]
    di_sorted=di[np.lexsort((np.arange(n),q_drop[di]))[:k]]
    add=[tuple(a) for a in additions[ai]]
    drop=[tuple(d) for d in removals[di_sorted]]
    after=native.copy()
    for x,y in add:
        assert not native[y,x]
        after[y,x]=True
    for x,y in drop:
        assert native[y,x]
        after[y,x]=False
    assert k==2 and after.sum()==native.sum() and np.count_nonzero(after!=native)==2*k
    return dict(native=native,inside=inside,outside=outside,addition_sites=additions,removal_sites=removals,
                q_add=q_add,q_drop=q_drop,addition=add,removal=drop,after=after,k=k)


def glyph(ax,x,y,sign,pt=8,strong=False):
    center(ax,x,y,sign,pt,color=WHITE if strong else (TEAL if sign=="+" else RUST),bold=True)


def tiles(ax,x,y,step,mask,*,out=None,inner=None,add=(),drop=(),subset=None):
    ox,oy,w,h=subset if subset else (0,0,mask.shape[1],mask.shape[0])
    add,drop=set(add),set(drop)
    for py in range(oy,oy+h):
        for px in range(ox,ox+w):
            cell=(px,py)
            fill=NAVY if mask[py,px] else WHITE
            symbol=None
            if out is not None and out[py,px]:
                fill=MINT
            if inner is not None and inner[py,px]:
                fill=CLAY
            if cell in add:
                fill,symbol=TEAL,"+"
            if cell in drop:
                fill,symbol=RUST,"−"
            left,top=x+(px-ox)*step,y+(py-oy)*step
            box(ax,left,top,step,step,fill,LINE,.36,z=2)
            if symbol:
                glyph(ax,left+step/2,top+step/2,symbol,min(10,max(6.1,.93*step)),strong=True)


def mini_coverage(ax,x,y,step):
    # The shaded trapezoid inside the unit cell has area (0.9 + 0.4)/2 = 0.65.
    yleft,yright=.9,.4
    assert abs((yleft+yright)/2-.65)<1e-12
    tile=box(ax,x,y,step,step,WHITE,NAVY,.7,z=2)
    line=(yleft+yright)/2
    shape=np.array([(-.23,0), (1.18,0), (1.18,yright-.09), (1,yright),
                    (0,yleft), (-.23,yleft+.115)])*step+(x,y)
    poly=Polygon(shape,closed=True,facecolor=MINT,edgecolor="none",zorder=3)
    ax.add_patch(poly)
    poly.set_clip_path(tile)
    edge(ax,(x-.19*step,y+(yleft+.095)*step),(x+1.13*step,y+(yright-.065)*step),TEAL,1.05,z=4)
    box(ax,x,y,step,step,"none",NAVY,.7,z=5)
    return line


def overview(demo):
    fig,ax=paper(514.8,206)
    widths=(114,116,125,112)
    xs=(9,133,259,393)
    heads=("Retain detection","Sample endpoints","Estimate occupancy","Write the mask")
    for index,(left,width,head) in enumerate(zip(xs,widths,heads,strict=True)):
        box(ax,left,10,width,119,WHITE,LINE)
        box(ax,left,10,width,24,SKY if index in (0,3) else PALE,LINE)
        title(ax,left+7,22,"abcd"[index],head)
    for p,q in zip((123,249,384),(133,259,393)):
        arrow(ax,(p+1,77),(q-1,77))

    # Frozen detector interfaces are shown as native data, not a new feature backbone.
    center(ax,66,44,"Frozen detector",7.9,color=BLUE,bold=True)
    tiles(ax,30,55,6,demo["native"])
    box(ax,36,55,60,48,"none",BLUE,.75,dash=True,z=6)
    center(ax,66,113,r"$M$ · $B$ · $p_i$",9,bold=True)
    center(ax,66,122,r"Fixed score $s$",7.2,color=GRAY)

    tiles(ax,155,55,6,demo["native"],out=demo["outside"],inner=demo["inside"],
          add=map(tuple,demo["addition_sites"][:2]),drop=map(tuple,demo["removal_sites"][:2]))
    center(ax,191,44,"Two rings within box",7.7,bold=True)
    center(ax,191,113,r"$n$ sites on each side",8.1,bold=True)
    center(ax,191,122,r"Rank by $|p_i-0.5|$",7.2,color=GRAY)

    center(ax,321.5,44,"Scalar ROH",7.7,bold=True)
    for xc,lab in ((278,"14"),(299,"28"),(320,"24"),(341,"24"),(362,"1")):
        box(ax,xc-8,58,16,18,MINT,LINE)
        center(ax,xc,67,lab,7.8,bold=True)
    for a in (278,299,320,341):
        arrow(ax,(a+8.3,67),(a+12.6,67),TEAL)
    center(ax,321.5,89,r"$\sigma[\mathrm{logit}(p_i^\epsilon)+f_\theta]$",8.3,color=TEAL)
    edge(ax,(273,97),(370,97),LINE,.65)
    center(ax,321.5,110,r"Predicted occupancy $\widehat q_i$",8.3,color=TEAL,bold=True)

    center(ax,449,44,"Independent endpoints",7.75,bold=True)
    tiles(ax,413,55,6,demo["after"],add=demo["addition"],drop=demo["removal"])
    center(ax,449,113,r"Add $k$ and remove $k$",8.2,bold=True)

    box(ax,9,140,496,25,SAND,LINE)
    text(ax,16,152.5,"SOURCE TRAINING",7.7,color=OCHRE,bold=True)
    text(ax,104,152.5,"Author polygon",7.7)
    arrow(ax,(161,152.5),(178,152.5),OCHRE)
    text(ax,186,152.5,r"Fractional cell coverage $q_i$",7.7)
    arrow(ax,(278,152.5),(293,152.5),OCHRE)
    text(ax,299,152.5,"BCE fits ROH",7.7,bold=True)
    arrow(ax,(350,140),(350,130),OCHRE,dash=True)

    box(ax,9,173,496,24,PALE,LINE)
    text(ax,16,185,"FROZEN REFINER CONTROL",7.5,bold=True,color=GRAY)
    text(ax,143,185,"Image + mask",7.5)
    arrow(ax,(213,185),(228,185),GRAY)
    text(ax,235,185,"One Phoenix response",7.5)
    arrow(ax,(313,185),(329,185),GRAY)
    text(ax,335,185,"Vary support or write rule",7.5)
    text(ax,497,185,"Worked example",7.1,color=GRAY,align="right")
    return fig


def occupancy(demo):
    fig,ax=paper(247.9,323)
    box(ax,8,8,232,99,WHITE,LINE)
    box(ax,8,8,232,20,SKY,LINE)
    title(ax,14,18,"a","Native support and endpoint sampling")

    gx,gy,step=16,36,6.7
    tiles(ax,gx,gy,step,demo["native"],out=demo["outside"],inner=demo["inside"])
    box(ax,gx+step,gy,10*step,8*step,"none",BLUE,.75,dash=True,z=4)
    text(ax,101,40,r"$B$",8,color=BLUE)
    window=(3,1,4,4)
    ox,oy,w,h=window
    zx,zy,zstep=138,34,12.6
    edge(ax,(gx+(ox+w)*step,gy+oy*step),(zx,zy),BLUE,.65,True,z=2)
    edge(ax,(gx+(ox+w)*step,gy+(oy+h)*step),(zx,zy+h*zstep),BLUE,.65,True,z=2)
    box(ax,gx+ox*step,gy+oy*step,w*step,h*step,"none",BLUE,1.2,z=6)
    # Show one of the selected points per side in the inset.
    shown=[]
    for candidates in (demo["addition_sites"],demo["removal_sites"]):
        example_sites=[tuple(v) for v in candidates if ox<=v[0]<ox+w and oy<=v[1]<oy+h]
        assert example_sites
        shown.append(example_sites[:1])
    tiles(ax,zx,zy,zstep,demo["native"],out=demo["outside"],inner=demo["inside"],
          add=shown[0],drop=shown[1],subset=window)
    box(ax,zx,zy,w*zstep,h*zstep,"none",BLUE,.8,z=6)
    glyph(ax,198,44,"+")
    glyph(ax,198,61,"−")
    text(ax,206,44,r"$S_+$",8,color=TEAL,bold=True)
    text(ax,206,61,r"$S_-$",8,color=RUST,bold=True)
    text(ax,17,99,r"Order by $|p_i-0.5|$, then row-major index",7.7,color=GRAY)

    box(ax,8,117,232,91,WHITE,LINE)
    box(ax,8,117,232,20,SKY,LINE)
    title(ax,14,127,"b","Native readout and target coverage")
    box(ax,15,143,103,57,PALE,LINE)
    box(ax,129,143,104,57,SAND,LINE)
    center(ax,66.5,151,"Decoder readout",7.6,bold=True)
    center(ax,181,151,"SOURCE TRAINING",7.35,color=OCHRE,bold=True)
    x,y,span=46,160,23
    box(ax,x,y,span,span,WHITE,LINE,z=2)
    c=(x+.62*span,y+.56*span)
    for dx,dy in ((0,0),(0,span),(span,0),(span,span)):
        edge(ax,(x+dx,y+dy),c,BLUE,.50,True,z=3)
        ax.add_patch(Circle((x+dx,y+dy),1.65,facecolor=GRAY,edgecolor=WHITE,lw=.3,zorder=5))
    ax.add_patch(Circle(c,2.1,facecolor=BLUE,edgecolor=WHITE,lw=.4,zorder=6))
    text(ax,76,171,r"$c_i$",8,color=BLUE)
    center(ax,66.5,193,"Bilinear at cell center",7.5)
    fraction=mini_coverage(ax,168,159,24)
    text(ax,199,170,r"$G$",7.8,color=TEAL)
    text(ax,152,168,r"$C_i$",7.8,color=GRAY)
    center(ax,181,193,rf"$q_i=|C_i\cap G|/|C_i|={fraction:.2f}$",7.65,color=NAVY)

    box(ax,8,218,232,97,WHITE,LINE)
    box(ax,8,218,232,20,SKY,LINE)
    title(ax,14,228,"c","Residual occupancy head (ROH)")
    marks=((18,40,r"$v_i$ (14)"),(72,58,r"$[z_i,z_i]$ (28)"),(143,23,"24"),(174,23,"24"),(205,24,"1"))
    for x,w,label in marks:
        box(ax,x,249,w,17,MINT if w<25 else PALE,LINE)
        center(ax,x+w/2,257.5,label,7.75,bold=True)
    for prev,nxt in zip(marks,marks[1:]):
        arrow(ax,(prev[0]+prev[1]+1,257.5),(nxt[0]-1,257.5),TEAL)
    text(ax,17,279,r"$\mathrm{logit}(p_i^\epsilon)$",8.15)
    arrow(ax,(84,279),(98,279),TEAL)
    ax.add_patch(Circle((105,279),5.4,facecolor=WHITE,edgecolor=TEAL,lw=.8,zorder=6))
    center(ax,105,278.7,"+",8.5,color=TEAL,bold=True)
    edge(ax,(217,266),(217,271),TEAL,.8)
    edge(ax,(217,271),(106,271),TEAL,.8)
    arrow(ax,(106,271),(106,273.2),TEAL)
    arrow(ax,(112,279),(130,279),TEAL)
    box(ax,132,272,29,15,MINT,LINE)
    center(ax,146.5,279.5,r"$\sigma$",8.2,bold=True)
    arrow(ax,(161,279),(176,279),TEAL)
    text(ax,181,279,r"$\widehat q_i$",9.6,color=TEAL,bold=True)
    edge(ax,(15,293),(233,293),LINE,.7)
    text(ax,15,305,"Fractional BCE · 1,321 parameters",7.5,color=GRAY)
    text(ax,233,305,"Schematic",7.1,color=GRAY,align="right")
    return fig


def emit(fig,output,stem,preview):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for label in fig.axes[0].texts:
        extent=label.get_window_extent(renderer)
        if not all(fig.bbox.contains(x,y) for x,y in extent.get_points()):
            raise ValueError(f"{stem} has clipped text: {label.get_text()}")
    for ext,meta in (("svg",{"Date":None}),("pdf",{"CreationDate":None,"ModDate":None})):
        fig.savefig(output/f"{stem}.{ext}",facecolor=WHITE,metadata=meta)
    if preview:
        fig.savefig(output/f"{stem}.png",dpi=175,facecolor=WHITE)
    print(stem,*np.round(fig.get_size_inches()*72,1))
    plt.close(fig)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--output",required=True,type=Path)
    cli.add_argument("--preview",action="store_true")
    args=cli.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    demo=example()
    for stem,figure in (("matched_budget_pipeline",overview(demo)),("method_native_occupancy",occupancy(demo))):
        emit(figure,args.output,stem,args.preview)


if __name__=="__main__":
    main()

