"""Build the editable nine-page companion from the published schematic SVGs."""
import argparse
import base64
import re
import zlib
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "release/research_round_20260925_global/paper_boundary_refinement_20261120/figures/method_atlas_20261126"
FIGURES = [('nora_one_pixel_motivation', '1 · one pixel motivation'), ('nora_scene_to_mask', '2 · scene to mask'), ('nora_native_sampling', '3 · native sampling'), ('nora_fractional_readout', '4 · fractional readout'), ('nora_roh_architecture', '5 · ROH architecture'), ('nora_pairing_paths', '6 · pairing paths'), ('nora_overlap_accounting', '7 · overlap accounting'), ('nora_signed_budget', '8 · signed budget'), ('nora_frozen_writer', '9 · frozen writer')]
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
def fmt(n):
    return f"{n:.5f}".rstrip("0").rstrip(".") or "0"
def node(container,tag,**attrs):
    return ET.SubElement(container,tag,{k:str(v) for k,v in attrs.items()})
def native_stencil(commands,box,fill,stroke):
    x,y,w,h=box
    shape=ET.Element("shape",name="Editable contour",w="100",h="100",aspect="variable",strokewidth="inherit")
    fg=node(shape,"foreground")
    path=node(fg,"path")
    for cmd,points in commands:
        pts=[(fmt((a-x)/w*100),fmt((b-y)/h*100)) for a,b in points]
        if cmd in "ML":
            node(path,"move" if cmd=="M" else "line",x=pts[0][0],y=pts[0][1])
        elif cmd=="Q":
            node(path,"quad",x1=pts[0][0],y1=pts[0][1],x2=pts[1][0],y2=pts[1][1])
        elif cmd=="C":
            node(path,"curve",**{f"{k}{i}":p[j] for i,p in enumerate(pts,1) for j,k in enumerate(("x","y"))})
        else:
            node(path,"close")
    node(fg,"fillstroke" if fill!="none" and stroke!="none" else "fill" if fill!="none" else "stroke")
    raw=quote(ET.tostring(shape,encoding="unicode"),safe="~()*!.'-").encode()
    stream=zlib.compressobj(wbits=-15)
    return base64.b64encode(stream.compress(raw)+stream.flush()).decode()

class Page:
    def __init__(self,diagram,w,h):
        graph=node(diagram,"mxGraphModel",page="1",grid="0",pageWidth=fmt(w),pageHeight=fmt(h),pageScale="1",math="0")
        self.root=node(graph,"root")
        node(self.root,"mxCell",id="0")
        node(self.root,"mxCell",id="1",parent="0",value="Method diagram")
        self.count=0
    def cell(self,style,box,value=""):
        self.count+=1
        c=node(self.root,"mxCell",id=str(self.count+1),parent="1",value=value,vertex="1",style=style)
        x,y,w,h=box
        node(c,"mxGeometry",x=fmt(x),y=fmt(y),width=fmt(w),height=fmt(h),**{"as":"geometry"})
    def edge(self,points,style):
        self.count+=1
        cell=node(self.root,"mxCell",id=str(self.count+1),parent="1",edge="1",style="endArrow=none;startArrow=none;"+style)
        geom=node(cell,"mxGeometry",relative="1",**{"as":"geometry"})
        for what,(x,y) in zip(("sourcePoint","targetPoint"),(points[0],points[-1])):
            node(geom,"mxPoint",x=fmt(x),y=fmt(y),**{"as":what})
        if len(points)>2:
            arr=node(geom,"Array",**{"as":"points"})
            for x,y in points[1:-1]:
                node(arr,"mxPoint",x=fmt(x),y=fmt(y))
    def visit(self,e,sty=None,offset=(0,0)):
        tag=e.tag.split("}")[-1]
        if tag in ("metadata","defs"): return
        sty=dict(sty or {})
        for part in e.get("style","").split(";"):
            if ":" in part:
                k,v=part.split(":",1)
                sty[k.strip()]=v.strip()
        x0,y0=offset
        tr=e.get("transform","")
        for op,val in re.findall(r"(\w+)\(([^)]*)\)",tr):
            vals=list(map(float,re.findall(NUM,val)))
            if op=="translate": x0+=vals[0]; y0+=vals[1] if len(vals)>1 else 0
            elif op!="rotate" or vals[0]!=0: raise ValueError(tr)
        if tag=="image": raise ValueError("Raster input")
        if tag=="rect":
            x,y,w,h=[float(e.get(k,0)) for k in ('x','y','width','height')]
            self.cell("shape=rectangle;strokeColor=none;fillColor="+sty.get('fill','#000000')+";",(x+x0,y+y0,w,h))
        if tag=="path":
            chunks=re.findall(r"([MLQCzZ])([^MLQCzZ]*)",e.get("d"))
            cmds=[]
            for cmd,raw in chunks:
                vals=list(map(float,re.findall(NUM,raw)))
                cmds.append((cmd,[(a+x0,b+y0) for a,b in zip(vals[::2],vals[1::2])]))
            clip=e.get('clip-path','')
            rect=self.subclips.get(clip)
            if rect:
                from shapely.geometry import Polygon, box as clipbox
                if any(cmd not in "MLzZ" for cmd,_ in cmds):
                    raise ValueError("Unsupported clipped curve")
                polygon=Polygon([p for _,ps in cmds for p in ps]).intersection(clipbox(*rect))
                if polygon.is_empty: return
                ring=list(polygon.exterior.coords)[:-1]
                cmds=[("M",[ring[0]])]+[("L",[pt]) for pt in ring[1:]]+[("z",[])]
            pts=[p for _,ps in cmds for p in ps]
            xs,ys=zip(*pts)
            box=(min(xs),min(ys),max(.001,max(xs)-min(xs)),max(.001,max(ys)-min(ys)))
            fill=sty.get("fill","#000000"); stroke=sty.get("stroke","none")
            style=f"fillColor={fill};strokeColor={stroke};strokeWidth={sty.get('stroke-width','1')};"
            if "opacity" in sty: style+=f"opacity={float(sty['opacity'])*100};"
            if "stroke-dasharray" in sty: style+="dashed=1;dashPattern="+sty["stroke-dasharray"].replace(","," ")+";"
            seq="".join(c for c,_ in cmds)
            if seq.lower()=="mlllz" and len(set(xs))==len(set(ys))==2:
                self.cell("shape=rectangle;"+style,box)
            elif fill=="none" and set(seq)<=set("ML") and seq.count("M")==1:
                self.edge(pts,style)
            else:
                self.cell("shape=stencil("+native_stencil(cmds,box,fill,stroke)+");"+style,box)
        if tag in ("text","tspan") and e.text and e.text.strip():
            text=e.text
            fs=float(sty.get("font-size","8px").removesuffix("px"))
            # Matplotlib math places each glyph using its own absolute baseline.
            xs=[float(z)+x0 for z in re.findall(NUM,e.get("x","0"))]
            ys=[float(z)+y0 for z in re.findall(NUM,e.get("y","0"))]
            labels=[(xs[0],ys[0],text)] if len(xs)==1 else [(a,ys[min(i,len(ys)-1)],ch) for i,(a,ch) in enumerate(zip(xs,text))]
            for x,y,v in labels:
                w=max(fs*1.6,len(v)*fs*1.2)
                anchor=sty.get("text-anchor","start")
                xx=x-w*{"middle":.5,"end":1}.get(anchor,0)
                align={"middle":"center","end":"right"}.get(anchor,"left")
                bold=sty.get("font-weight","") in ("700","bold")
                italic=sty.get("font-style","") in ("italic","oblique")
                style=f"text;html=0;spacing=0;align={align};verticalAlign=middle;whiteSpace=nowrap;overflow=visible;fontFamily=Arial;fontSize={fmt(fs)};fontColor={sty.get('fill','#000000')};fontStyle={int(bold)+2*int(italic)};"
                self.cell(style,(xx,y-fs*1.07,w,fs*1.42),v)
        for child in e: self.visit(child,sty,(x0,y0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=SOURCE)
    ap.add_argument("--output",type=Path,default=HERE/"NORA_method_atlas.drawio")
    args=ap.parse_args()
    workbook=ET.Element("mxfile",host="app.diagrams.net",pages="9",compressed="false")
    for stem,title in FIGURES:
        svg=ET.parse(args.source/(stem+".svg")).getroot()
        _,_,w,h=map(float,svg.attrib["viewBox"].split())
        page=Page(node(workbook,"diagram",name=title,id=stem),w,h)
        page.subclips={}
        for clip in svg.findall('.//{*}clipPath'):
            rect=clip.find('{*}rect')
            if rect is None: raise ValueError("Unsupported clip")
            x,y,cw,ch=[float(rect.get(k,0)) for k in ('x','y','width','height')]
            if (x,y,cw,ch)!=(0,0,w,h):
                page.subclips["url(#"+clip.get("id")+")"]=(x,y,x+cw,y+ch)
        page.visit(svg)
        print(title,page.count)
    ET.ElementTree(workbook).write(args.output,encoding="utf-8",xml_declaration=True)
if __name__=="__main__": main()


