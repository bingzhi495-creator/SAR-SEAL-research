"""Build the editable five-page companion from the published schematic SVGs."""
import argparse
import base64
import re
import zlib
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
SOURCE = HERE
FIGURES = [("matched_budget_pipeline","1 · NORA"),("method_native_occupancy","2 · Residual occupancy head"),("method_bounded_selection","3 · Endpoint selection"),("method_signed_allocation","4 · Signed allocation"),("method_phoenix_controls","5 · Frozen refiner")]
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
        if tag=="path":
            chunks=re.findall(r"([MLQCzZ])([^MLQCzZ]*)",e.get("d"))
            cmds=[]
            for cmd,raw in chunks:
                vals=list(map(float,re.findall(NUM,raw)))
                cmds.append((cmd,[(a+x0,b+y0) for a,b in zip(vals[::2],vals[1::2])]))
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
            text=e.text.replace(chr(0xE238), chr(0x2112))
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
    ap.add_argument("--output",type=Path,default=HERE/"NORA_method_figures.drawio")
    args=ap.parse_args()
    workbook=ET.Element("mxfile",host="app.diagrams.net",pages="5",compressed="false")
    for stem,title in FIGURES:
        svg=ET.parse(args.source/(stem+".svg")).getroot()
        _,_,w,h=map(float,svg.attrib["viewBox"].split())
        page=Page(node(workbook,"diagram",name=title,id=stem),w,h)
        page.visit(svg)
        print(title,page.count)
    ET.ElementTree(workbook).write(args.output,encoding="utf-8",xml_declaration=True)
if __name__=="__main__": main()


