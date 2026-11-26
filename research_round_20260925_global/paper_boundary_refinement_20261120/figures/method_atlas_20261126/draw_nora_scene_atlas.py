"""NORA scene overview and native sampling atlas from vector geometry or a qualified exhibit."""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np


NAVY, BLUE, TEAL, RUST, OCHRE, GRAY = ("#382B26", "#AB5B38", "#356B5D", "#8F4958", "#AB5B38", "#756C65")
WHITE, LINE, PALE, SKY, MINT, CLAY, SAND = ("#FFFFFF", "#D7CBBE", "#F8F3EB", "#F8F3EB", "#E5EEE7", "#F1E2E6", "#F8F3EB")
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



from base64 import b64encode
from hashlib import sha256
from io import BytesIO
from xml.etree import ElementTree as ET
import re

import fitz
from shapely.geometry import LineString, box as shapely_box


SOURCE_SHA = "1cf9040ccf593582feaaf6f3c00b4461491263edea9a8bbe8a1d2291e57ff40f"
SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("",SVG)
ET.register_namespace("xlink",XLINK)
ROI = (21,26,8,8)
# Only the authored display PDF is used. The original source data are never opened.


def xml_node(parent, tag, **attrs):
    return ET.SubElement(parent,f"{{{SVG}}}{tag}",{k:str(v) for k,v in attrs.items()})


class Case:
    def __init__(self,path):
        data=path.read_bytes()
        if sha256(data).hexdigest()!=SOURCE_SHA:
            raise ValueError("Expected the qualified earlier author-only case figure")
        self.pdf=fitz.open(stream=data,filetype="pdf")
        self.page=self.pdf[0]
        images=self.page.get_image_info(xrefs=True)
        if self.pdf.page_count!=1 or len(images)!=12:
            raise ValueError("Unexpected authored case layout")
        self.panels={key:fitz.Rect(images[index]["bbox"]) for key,index in (("SAR",0),("N",1),("I",4))}
        if len({row["xref"] for row in images[:6]})!=1 or any(
                (row["width"],row["height"])!=(78,78) for row in images[:6]):
            raise ValueError("Source crops must share the same embedded SAR display image")
        self.png=fitz.Pixmap(self.pdf,images[0]["xref"]).tobytes("png")
        self.raw=fitz.open()
        plane=self.raw.new_page(width=78,height=78)
        plane.insert_image(plane.rect,stream=self.png)
        self.paths=self.page.get_drawings()
        self.frames={}
        masks={}
        for name in ("N","I"):
            content=[path for path in self.paths if self.panels[name].contains(path["rect"])
                     and path["type"]=="f"]
            frame,=[path["rect"] for path in content if sum(path["fill"])>2.99]
            self.frames[name]=frame
            cells=[path["rect"] for path in content if sum(path["fill"])<2.99]
            native_size=round(frame.width/np.median([cell.width for cell in cells]))
            if native_size!=44 or not np.isclose(frame.width,frame.height,atol=2e-5):
                raise ValueError("Exact original square axis frame and native-cell count")
            self.native_size=native_size
            sx,sy=frame.width/native_size,frame.height/native_size
            xy=np.array([((c.x0-frame.x0)/sx,(c.y0-frame.y0)/sy) for c in cells])
            if not np.allclose(xy,np.rint(xy),atol=3e-5,rtol=0):
                raise ValueError("Author overlay does not align to native-cell grid")
            masks[name]={tuple(point) for point in np.rint(xy).astype(int)}
        if len(masks["N"])!=274 or len(masks["I"]-masks["N"])!=4 or len(masks["N"]-masks["I"])!=4:
            raise ValueError("Exact observed signed counts")
        self.native_mask=np.zeros((self.native_size,self.native_size),dtype=bool)
        for x,y in masks['N']:
            self.native_mask[y,x]=True
        source_box=self.panels["N"]
        actual=self.frames["N"]
        self.raw_view=fitz.Rect(*((np.array(actual)-np.array((source_box.x0,source_box.y0,
                                                             source_box.x0,source_box.y0)))
                                  *(78/source_box.width)))

    def clip(self,label):
        if label=="SAR":
            return self.raw,self.raw_view
        full=self.frames["N" if label=="zoom" else label]
        if label!="zoom":
            # Source axes trim less than one source cell along the right/top edges.
            # The crop retains the author's original display extent.
            return self.pdf,full
        x,y,w,h=ROI
        sx=full.width/self.native_size
        sy=full.height/self.native_size
        return self.pdf,fitz.Rect(full.x0+x*sx,full.y0+y*sy,full.x0+(x+w)*sx,full.y0+(y+h)*sy)

    def place_svg(self, root, label, slot):
        x,y,w,h=slot
        full=fitz.Rect(0,0,78,78) if label=="SAR" else self.panels["N" if label=="zoom" else label]
        _,view=self.clip(label)
        flat=np.asarray(view)-[full.x0,full.y0,full.x0,full.y0]
        raw_clip=fitz.Rect(*(flat*(78/full.width)))
        with fitz.open() as cropped:
            surface=cropped.new_page(width=144,height=144)
            surface.show_pdf_page(surface.rect,self.raw,0,clip=raw_clip)
            png=surface.get_pixmap().tobytes("png")
        layer=xml_node(root,"g",id=f"observed-{label}")
        raster=xml_node(layer,"image",x=x,y=y,width=w,height=h,preserveAspectRatio="none")
        raster.set(f"{{{XLINK}}}href","data:image/png;base64,"+b64encode(png).decode("ascii"))
        raster.set("image-rendering","pixelated")
        if label=="SAR":
            return
        sx,sy=w/view.width,h/view.height
        def page_xy(u,v):
            return x+(u-view.x0)*sx,y+(v-view.y0)*sy

        contour_window=shapely_box(view.x0,view.y0,view.x1,view.y1)
        # Retain the source's displayed foreground cells and author outline as vector
        # objects. The visible output contains a single row of the original exhibit.
        count=0
        for path in self.paths:
            r=path["rect"]
            if not full.contains(r) or not r.intersects(view):
                continue
            items=path["items"]
            if path["type"]=="f" and path["fill"] and sum(path["fill"])<2.99:
                tag,box_rect,_=items[0]
                if len(items)!=1 or tag!="re":
                    raise ValueError("Expected the original raster-cell rectangles")
                box_rect=box_rect&view
                u,v=page_xy(box_rect.x0,box_rect.y0)
                values=tuple(round(v*255) for v in path["fill"])
                item=xml_node(layer,"rect",x=u,y=v,width=box_rect.width*sx,height=box_rect.height*sy,
                              fill=f"rgb({values[0]},{values[1]},{values[2]})")
                item.set("fill-opacity",str(path["fill_opacity"]))
                count+=1
            if path["type"]=="s" and len(items)>4 and all(item[0]=="l" for item in items):
                line=LineString([tuple(items[0][1]),*[tuple(item[2]) for item in items]])
                piece=line.intersection(contour_window)
                pieces=piece.geoms if hasattr(piece,"geoms") else (piece,)
                values=tuple(round(v*255) for v in path["color"])
                for segment in pieces:
                    if segment.geom_type!="LineString" or segment.is_empty:
                        continue
                    coords=[page_xy(u,v) for u,v in segment.coords]
                    command=" ".join(f"{'M' if i==0 else 'L'}{u} {v}" for i,(u,v) in enumerate(coords))
                    contour=xml_node(layer,"path",d=command,fill="none",
                                     stroke=f"rgb({values[0]},{values[1]},{values[2]})")
                    contour.set("stroke-width",str(min(.85,path["width"]*sx)))
                    dash=re.search(r"\[([^\]]+)\]",path["dashes"])
                    if dash:
                        contour.set("stroke-dasharray"," ".join(str(min(2,float(v)*sx))
                                                                  for v in dash.group(1).split()))
        if count==0:
            raise ValueError("No exact source foreground cells in this observed panel")


def buffered(figure,fmt,*,transparent=False):
    buffer=BytesIO()
    meta={"Date":None} if fmt=="svg" else {"CreationDate":None,"ModDate":None}
    figure.savefig(buffer,format=fmt,metadata=meta,transparent=transparent)
    return buffer.getvalue()


def mark_image(figure,svg,pdf,marks):
    height=figure.bbox_inches.height*72
    width=figure.bbox_inches.width*72
    view=pdf[0]
    gold=mpl.colors.to_rgb(OCHRE)
    parchment=(.97,.89,.78)
    for role,value in marks:
        if role=="frame":
            x,y,w,h=value
            item=xml_node(svg,"rect",x=x,y=y,width=w,height=h,fill="none",stroke=OCHRE)
            item.set("stroke-width",".95")
            view.draw_rect(fitz.Rect(x,y,x+w,y+h),color=gold,width=.95)
        if role=="edge":
            (a,b),(c,d)=value
            item=xml_node(svg,"line",x1=a,y1=b,x2=c,y2=d,stroke=OCHRE)
            item.set("stroke-width",".83")
            view.draw_line(fitz.Point(a,b),fitz.Point(c,d),color=gold,width=.83)
        if role=="grid":
            x,y,size,n=value
            for i in range(1,n):
                offset=size*i/n
                for coords in ((x+offset,y,x+offset,y+size),(x,y+offset,x+size,y+offset)):
                    a,b,c,d=coords
                    item=xml_node(svg,"line",x1=a,y1=b,x2=c,y2=d,stroke="#F7E3C7")
                    item.set("stroke-width",".3")
                    item.set("stroke-opacity",".65")
                    view.draw_line(fitz.Point(a,b),fitz.Point(c,d),color=parchment,width=.3,stroke_opacity=.65)
    assert abs(width-view.rect.width)<.002 and abs(height-view.rect.height)<.002


def write_figure(case,figure,slots,marks,dest,stem,preview):
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    for item in figure.axes[0].texts:
        assert all(figure.bbox.contains(x,y) for x,y in item.get_window_extent(renderer).get_points())
    svg=ET.fromstring(buffered(figure,"svg"))
    with fitz.open(stream=buffered(figure,"pdf"),filetype="pdf") as pdf:
        for label,slot in slots:
            x,y,w,h=slot
            src,clip=case.clip(label)
            if not src[0].rect.contains(clip):
                raise ValueError("Crop outside the authored source")
            target=fitz.Rect(x,y,x+w,y+h)
            if label=="zoom":
                # Keep mask geometry exact while rescaling the locator's outline stroke.
                zoom=ET.Element(f"{{{SVG}}}svg",width=str(w),height=str(h),viewBox=f"0 0 {w} {h}")
                case.place_svg(zoom,label,(0,0,w,h))
                with fitz.open(stream=ET.tostring(zoom),filetype="svg") as native_svg:
                    with fitz.open(stream=native_svg.convert_to_pdf(),filetype="pdf") as native_pdf:
                        pdf[0].show_pdf_page(target,native_pdf,0)
            else:
                pdf[0].show_pdf_page(target,src,0,clip=clip)
            case.place_svg(svg,label,slot)
        mark_image(figure,svg,pdf,marks)
        pdf.save(dest/f"{stem}.pdf",garbage=4,deflate=True,no_new_id=True)
        if preview:
            pdf[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(dest/f"{stem}.png")
    tree=ET.ElementTree(svg)
    tree.write(dest/f"{stem}.svg",encoding="utf-8",xml_declaration=True)
    print(stem,svg.get("viewBox"),"editable text and observed SAR layers")
    plt.close(figure)



def grid_patch(ax,x,y,step,mask,*,outside=None,inside=None,plus=(),minus=(),window=None,thin=.35):
    ox,oy,w,h=window if window is not None else (0,0,mask.shape[1],mask.shape[0])
    plus,minus=set(plus),set(minus)
    for r in range(oy,oy+h):
        for c in range(ox,ox+w):
            v=bool(mask[r,c])
            fill=NAVY if v else WHITE
            if outside is not None and outside[r,c]:
                fill=MINT
            if inside is not None and inside[r,c]:
                fill=CLAY
            sign=None
            if (c,r) in plus:
                fill,sign=TEAL,"+"
            if (c,r) in minus:
                fill,sign=RUST,"−"
            u,t=x+(c-ox)*step,y+(r-oy)*step
            box(ax,u,t,step,step,fill,LINE,thin,z=2)
            if sign:
                center(ax,u+step/2,t+step/2,sign,min(9,max(5.5,step*.9)),color=WHITE,bold=True)
    return w*step,h*step


def block_image(ax,x,y,edge_size,mask):
    # This vector-native figure layer is explicitly a schematic.
    step=edge_size/mask.shape[0]
    grid_patch(ax,x,y,step,mask,thin=.45)
    box(ax,x,y,edge_size,edge_size,"none",LINE,1,z=4)


def top_label(ax,x,y,description,bold=True):
    text(ax,x,y,description,8.8,color=NAVY,bold=bold)


def frame_map(x,y,extent,side,roi):
    a,b,w,h=roi
    step=extent/side
    return x+a*step,y+b*step,w*step,h*step


def mark_region(visual,zoom):
    x,y,w,h=visual
    zx,zy,zw,zh=zoom
    return [("frame",visual),("edge",((x+w,y),(zx,zy))),
            ("edge",((x+w,y+h),(zx,zy+zh))),("frame",zoom)]


def paint_marks(ax,marks):
    for role,value in marks:
        if role=="frame":
            box(ax,*value,"none",OCHRE,.9,z=6)
        elif role=="edge":
            edge(ax,*value,OCHRE,.7,z=6)
        elif role=="grid":
            x,y,w,n=value
            for i in range(1,n):
                z=i*w/n
                edge(ax,(x+z,y),(x+z,y+w),LINE,.2,z=6)
                edge(ax,(x,y+z),(x+w,y+z),LINE,.2,z=6)


def example():
    # A hand-specified silhouette to illustrate geometry in the public manuscript.
    core=np.zeros((14,14),dtype=bool)
    for row,left,right in ((2,4,8),(3,4,7),(4,3,10),(5,3,11),(6,4,12),(7,6,12),(8,8,12)):
        core[row,left:right]=True
    positives=((10,4),(11,5),(7,3),(10,9))
    removals=((4,3),(11,8),(8,5),(9,8))
    edited=core.copy()
    for x,y in positives:
        assert not core[y,x]
        edited[y,x]=True
    for x,y in removals:
        assert core[y,x]
        edited[y,x]=False
    assert min(8,int(core.sum())//10)>=4
    assert int(np.count_nonzero(core!=edited))==8 and edited.sum()==core.sum()
    return core,edited


def train_glyph(ax):
    y=189
    text(ax,10,y+10,"Source only",7.6,color=GRAY,bold=True)
    poly=Polygon(((87,y+12),(93,y+4),(108,y+3),(115,y+8),(113,y+15),(95,y+18)),
                 closed=True,facecolor=SAND,edgecolor=OCHRE,lw=.85,zorder=5)
    ax.add_patch(poly)
    center(ax,101,y+25,r"Polygon $G$",7.3)
    arrow(ax,(125,y+10),(153,y+10),OCHRE)
    frac=mini_coverage(ax,167,y+1,18)
    center(ax,176,y+25,r"Cell fraction $q_i$",7.3)
    arrow(ax,(195,y+10),(222,y+10),OCHRE)
    center(ax,250,y+10,r"$\mathcal{L}_{\rm BCE}$",11,color=OCHRE,bold=True)
    arrow(ax,(272,y+10),(305,y+10),OCHRE)
    center(ax,331,y+10,"Fit ROH",7.6,bold=True)


def overview(case):
    fig,ax=paper(514.8,226)
    native,edited=example()
    hx,hy,hs=9,34,132
    vx,vy,vs=166,49,106
    ox,oy,os=373,34,132
    top_label(ax,hx,24,"Observed SAR" if case else r"Native mask $N$")
    top_label(ax,vx,36,"Observed native cells" if case else "Native-cell window")
    top_label(ax,ox,24,r"Independent output $I$")
    slots=[]
    marks=[]
    roi=ROI if case else (5,5,6,6)
    if case:
        slots.extend((("SAR",(hx,hy,hs,hs)),("zoom",(vx,vy,vs,vs)),("I",(ox,oy,os,os))))
        rect=frame_map(hx,hy,hs,case.native_size,roi)
        marks=mark_region(rect,(vx,vy,vs,vs))
        marks.append(("grid",(vx,vy,vs,roi[2])))
        text(ax,hx,174,"Instance-RSDD reserve · D-FINE",7.5,color=GRAY)
    else:
        block_image(ax,hx,hy,hs,native)
        block_image(ax,ox,oy,os,edited)
        step=vs/roi[2]
        grid_patch(ax,vx,vy,step,native,window=roi,thin=.65)
        rect=frame_map(hx,hy,hs,native.shape[0],roi)
        paint_marks(ax,mark_region(rect,(vx,vy,vs,vs)))
        text(ax,hx,174,"Schematic",7.5,color=GRAY)
    text(ax,ox,174,"+4",8.4,color=TEAL,bold=True)
    text(ax,ox+22,174,"−4",8.4,color=RUST,bold=True)

    # The logic path is shared by all figures. It does not assert a particular
    # trained score, box, or pairing for the observed pixel window.
    arrow(ax,(277,67),(293,67),OCHRE)
    center(ax,326,67,r"$S_+,S_-$",10.2,color=NAVY,bold=True)
    arrow(ax,(326,78),(326,89),OCHRE)
    center(ax,326,99,"ROH",9.1,color=TEAL,bold=True)
    text(ax,345,99,r"$\widehat q_i$",8.6,color=TEAL)
    arrow(ax,(326,109),(326,121),OCHRE)
    center(ax,309,133,r"$+k$",9.4,color=TEAL,bold=True)
    center(ax,333,133,r"$-k$",9.4,color=RUST,bold=True)
    arrow(ax,(344,133),(370,133),OCHRE)
    center(ax,326,154,"Schematic decision",7.0,color=GRAY)
    edge(ax,(9,184),(505,184),LINE,.8,z=4)
    train_glyph(ax)
    return fig,slots,marks


def two_rings(mask,clip):
    outer=neighborhood(neighborhood(mask,True),True)&~mask&clip
    inner=mask&~neighborhood(neighborhood(mask,False),False)&clip
    return outer,inner


def sample_for_example(outer,inner,region):
    x,y,w,h=region
    yy,xx=np.indices(outer.shape)
    rank=(yy*17+xx*11)%101
    n=min(32,int(outer.sum()),int(inner.sum()))
    focus=(xx>=x)&(xx<x+w)&(yy>=y)&(yy<y+h)
    priority=rank+101*~focus
    p=.5+np.where(inner,1,-1)*(priority+1)/404
    def select(support):
        a,b=np.nonzero(support)
        ids=np.lexsort((a*support.shape[1]+b,abs(p[a,b]-.5)))[:n]
        return [(int(b[i]),int(a[i])) for i in ids]
    plus,minus=select(outer),select(inner)
    if n<=0 or len(plus)!=len(minus):
        raise ValueError("Both signs have a positive equal endpoint allowance")
    return plus,minus,n


def sampling(case):
    fig,ax=paper(247.9,257)
    slots=[]
    marks=[]
    if case is None:
        mask,_=example()
        roi=(5,5,6,6)
        limits=(3,2,12,12)
    else:
        mask=case.native_mask
        roi=ROI
        limits=(3,6,35,42)
    size=mask.shape[0]
    a,b,c,d=limits
    clipped=np.zeros_like(mask)
    clipped[b:d,a:c]=True
    near,far=two_rings(mask,clipped)
    plus,minus,n=sample_for_example(near,far,roi)
    visible=lambda points:[(x,y) for x,y in points if roi[0]<=x<roi[0]+roi[2] and roi[1]<=y<roi[1]+roi[3]]
    def shown(points):
        subset=visible(points)
        indices=np.rint(np.linspace(0,len(subset)-1,min(4,len(subset)))).astype(int)
        return [subset[j] for j in indices]
    show_plus=shown(plus)
    show_minus=shown(minus)
    if not show_plus or not show_minus:
        raise ValueError("The detail crop must show one valid endpoint from each signed side")

    x,y,w=10,30,78
    tx,ty,twidth=135,30,93
    top_label(ax,x,20,"Observed native mask" if case else "Schematic mask")
    top_label(ax,tx,20,"Two rings inside box")
    if case:
        slots=[("N",(x,y,w,w))]
    else:
        block_image(ax,x,y,w,mask)
    arrow(ax,(97,68),(124,68),OCHRE)
    step=twidth/size
    grid_patch(ax,tx,ty,step,mask,outside=near,inside=far,thin=.12 if case else .43)
    box(ax,tx+a*step,ty+b*step,(c-a)*step,(d-b)*step,"none",OCHRE,.9,True,z=5)
    text(ax,231,ty+b*step+4,r"$B$",7.5,color=OCHRE,bold=True)
    text(ax,155,130,r"Illustrative box $B$",7.1,color=GRAY)
    x1,y1,w1,h1=frame_map(tx,ty,twidth,size,roi)
    box(ax,x1,y1,w1,h1,"none",OCHRE,.83,z=7)
    edge(ax,(x1,y1+h1),(12,154),OCHRE,.61,z=4)
    edge(ax,(x1+w1,y1+h1),(108,154),OCHRE,.61,z=4)
    # The original-window view is organic, with a single shared set of
    # signed ticks used as a legend for the paired lists at right.
    z=96/roi[2]
    grid_patch(ax,12,154,z,mask,outside=near,inside=far,plus=show_plus,minus=show_minus,
               window=roi,thin=.5)
    box(ax,12,154,96,96,"none",OCHRE,.9,z=5)

    text(ax,126,161,"Endpoint ranking",8.3,bold=True)
    text(ax,126,175,r"$|p_i-0.5|$",8.7,color=GRAY)
    for yy,label,count,color,g in ((198,r"$S_+$",n,TEAL,"+"),(227,r"$S_-$",n,RUST,"−")):
        text(ax,125,yy,label,10.0,color=color,bold=True)
        for j in range(3):
            box(ax,160+10*j,yy-4.3,8,8,color,WHITE,.21,z=3)
            center(ax,164+10*j,yy-.1,g,6.8,color=WHITE,bold=True)
        text(ax,191,yy,r"$\ldots$",8.6,color=GRAY)
        text(ax,205,yy,rf"$n={count}$",8.0,color=color,bold=True)
    text(ax,126,247,"Schematic scores",7.2,color=GRAY)
    return fig,slots,marks


def emit_public(figure,target,stem,preview):
    renderer=figure.canvas
    renderer.draw()
    for label in figure.axes[0].texts:
        assert all(figure.bbox.contains(x,y) for x,y in label.get_window_extent(renderer.get_renderer()).get_points())
    for format_ in ("pdf","svg"):
        (target/f"{stem}.{format_}").write_bytes(buffered(figure,format_))
    if preview:
        figure.savefig(target/f"{stem}.png",dpi=168)
    plt.close(figure)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--observed-figure",type=Path)
    parser.add_argument("--preview",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    case=Case(args.observed_figure) if args.observed_figure else None
    for stem,(figure,slots,marks) in (("nora_scene_to_mask",overview(case)),
                                     ("nora_native_sampling",sampling(case))):
        if case:
            write_figure(case,figure,slots,marks,args.output,stem,args.preview)
        else:
            emit_public(figure,args.output,stem,args.preview)
            print(stem,"public vector")
    if case:
        case.pdf.close()
        case.raw.close()


if __name__=="__main__":
    main()

