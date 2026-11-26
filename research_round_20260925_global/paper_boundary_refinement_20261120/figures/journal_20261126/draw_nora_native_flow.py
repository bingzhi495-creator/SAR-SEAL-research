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


def layer_card(ax,x,y,w,h,label,depth=1,pt=7.6):
    """A perspective-free stack denotes the feature width of one MLP stage."""
    for back in range(depth-1,0,-1):
        box(ax,x+1.18*back,y-1.23*back,w,h,"#DDE7DE",TEAL,.52)
    box(ax,x,y,w,h,MINT if depth>1 else PALE,TEAL,.62)
    center(ax,x+w/2,y+h/2,label,pt,bold=True)


def rail_pixel_icon(ax,x,y,pattern,colors,step=2.7):
    for row,col in np.ndindex(pattern.shape):
        box(ax,x+col*step,y+row*step,step,step,colors[pattern[row,col]],WHITE,.24,z=2)
    box(ax,x,y,pattern.shape[1]*step,pattern.shape[0]*step,"none",LINE,.52,z=5)


def graphic_rails(ax,native):
    header_ink=OCHRE
    box(ax,9,143,496,37,SAND,LINE)
    text(ax,16,160,"SOURCE TRAINING",7.4,color=header_ink,bold=True)
    # A contour, native cells intersecting a fragment, and the fitting objective.
    shape=np.array(((115,158),(122,150),(136,149),(143,155),(135,164),(122,165)))
    ax.add_patch(Polygon(shape,closed=True,facecolor="#F1DEC2",edgecolor=OCHRE,lw=.9,zorder=4))
    center(ax,130,173,r"Polygon $G$",7.15)
    arrow(ax,(155,157),(187,157),OCHRE)
    gx,gy,gg=197,148,6
    for yy in range(3):
        for xx in range(4):
            box(ax,gx+xx*gg,gy+yy*gg,gg,gg,WHITE,LINE,.44,z=2)
    clipped=Polygon(((198,151),(221,151),(215,160),(201,165),(198,164)),
                    closed=True,facecolor=MINT,edgecolor=TEAL,lw=.8,zorder=3)
    ax.add_patch(clipped)
    for yy in (154,160):
        edge(ax,(197,yy),(221,yy),LINE,.4,z=4)
    for xx in (203,209,215):
        edge(ax,(xx,148),(xx,166),LINE,.4,z=4)
    center(ax,209,173,r"Cell fraction $q_i$",7.15)
    arrow(ax,(233,157),(267,157),OCHRE)
    box(ax,279,148,58,18,"#F9EEDC",OCHRE,.55)
    center(ax,308,157,r"$\mathcal{L}_{\mathrm{BCE}}$",9.1,bold=True)
    center(ax,308,173,"Fit ROH",7.15)
    arrow(ax,(319,143),(319,136),OCHRE,True)

    box(ax,9,186,496,37,PALE,LINE)
    text(ax,16,203,"FROZEN REFINER",7.4,bold=True,color=GRAY)
    binary=native[::2,::2].astype(int)
    texture=np.array(((0,0,1,1,0,1),(1,2,2,3,1,0),(0,1,3,2,3,1),(0,1,0,2,1,0)))
    heat=np.array(((0,0,1,1,0,0),(0,1,2,3,2,0),(0,1,3,3,3,1),(0,0,1,2,1,0)))
    rail_pixel_icon(ax,113,193,texture,("#423731","#6C6159","#B8AEA4","#E4D5C5"),3.35)
    center(ax,141,200.5,"+",8.2,color=GRAY,bold=True)
    rail_pixel_icon(ax,152,193,binary,(WHITE,NAVY),3.35)
    center(ax,142.5,215.5,"Image + mask",7.15)
    arrow(ax,(186,201),(216,201),GRAY)
    rail_pixel_icon(ax,228,191,heat,("#F5EEE4","#D9B993","#B9824F","#85553B"),3.75)
    text(ax,256,200.5,r"$r_i$",8.4,color=GRAY,bold=True)
    center(ax,246.5,215.5,"Phoenix response",7.15)
    arrow(ax,(276,201),(307,201),GRAY)
    narrow=np.array(((0,1,1,1,1,0),(1,2,2,2,2,1),(1,2,2,2,2,1),(0,1,1,1,1,0)))
    whole=np.where(binary,2,1)
    rail_pixel_icon(ax,324,193,narrow,(WHITE,"#D9C4A5",NAVY),3.35)
    rail_pixel_icon(ax,386,193,whole,(WHITE,"#D9C4A5",NAVY),3.35)
    center(ax,334,215.5,"Narrow",7.05)
    center(ax,396,215.5,"Whole",7.05)


def overview(demo):
    fig,ax=paper(514.8,230)
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
    for xc,lab,depth in ((278,"14",1),(299,"28",3),(320,"24",2),(341,"24",2),(362,"1",1)):
        layer_card(ax,xc-8,58,16,18,lab,depth)
    center(ax,299,82,r"$[z_i,z_i]$",6.8,color=GRAY)
    for a in (278,299,320,341):
        arrow(ax,(a+8.3,67),(a+12.6,67),TEAL)
    center(ax,321.5,94,r"$\sigma[\mathrm{logit}(p_i^\epsilon)+f_\theta]$",8.3,color=TEAL)
    edge(ax,(273,102),(370,102),LINE,.65)
    center(ax,321.5,114,r"Predicted occupancy $\widehat q_i$",8.3,color=TEAL,bold=True)

    center(ax,449,44,"Independent endpoints",7.75,bold=True)
    tiles(ax,413,55,6,demo["after"],add=demo["addition"],drop=demo["removal"])
    center(ax,449,113,r"Add $k$ and remove $k$",8.2,bold=True)

    graphic_rails(ax,demo['native'])
    return fig


def occupancy(demo):
    fig,ax=paper(247.9,323)
    box(ax,8,8,232,99,WHITE,LINE)
    box(ax,8,8,232,20,SKY,LINE)
    title(ax,14,18,"a","Sample native cells")

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
    title(ax,14,127,"b","Readout and target")
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
    center(ax,66.5,193,r"Bilinear at $c_i$",7.5)
    fraction=mini_coverage(ax,168,159,24)
    text(ax,199,170,r"$G$",7.8,color=TEAL)
    text(ax,152,168,r"$C_i$",7.8,color=GRAY)
    center(ax,181,193,rf"$q_i=|C_i\cap G|/|C_i|={fraction:.2f}$",7.65,color=NAVY)

    box(ax,8,218,232,97,WHITE,LINE)
    box(ax,8,218,232,20,SKY,LINE)
    title(ax,14,228,"c","Scalar ROH")
    marks=((18,40,r"$v_i$ (14)"),(72,58,r"$[z_i,z_i]$ (28)"),(143,23,"24"),(174,23,"24"),(205,24,"1"))
    for (x,w,label),depth in zip(marks,(1,3,2,2,1),strict=True):
        layer_card(ax,x,249,w,17,label,depth)
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
    text(ax,15,305,"1,321 parameters",7.5,color=GRAY)
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

