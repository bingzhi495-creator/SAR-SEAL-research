"""Render matched boundary details from the previously released SegRefiner case plate."""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import fitz

INK = (.24, .19, .16)
PAPER = (.993, .980, .951)
GREEN = (.24, .43, .34)
OCHRE = (.63, .30, .13)
LINE = (.80, .75, .67)
CASES = [
    # x, box within the same original viewport, and continuous IoU (%)
    ("(a)  Local ROH retains hull area", 283.5, (.31,.72,.62,1.), (71.93,71.95,67.43), "+4.52", GREEN),
    ("(b)  Whole-crop refinement removes excess foreground", 31.5, (.35,.61,.74,1.), (58.68,58.89,60.96), "-2.07", OCHRE),
]


def label(page, x, y, value, size=8.0, color=INK, bold=False):
    page.insert_text((x,y), value, fontname="hebo" if bold else "helv", fontsize=size, color=color)


def center(page, x, y, value, size, bold=False):
    font="hebo" if bold else "helv"
    width=fitz.get_text_length(value, fontname=font, fontsize=size)
    label(page,x-width/2,y,value,size,bold=bold)


def window(rect, roi):
    return fitz.Rect(rect.x0+rect.width*roi[0],rect.y0+rect.height*roi[1],
                     rect.x0+rect.width*roi[2],rect.y0+rect.height*roi[3])


def editable_svg(page, source_page, windows, output):
    svg = "http://www.w3.org/2000/svg"
    xlink = "http://www.w3.org/1999/xlink"
    ET.register_namespace("", svg)
    ET.register_namespace("xlink", xlink)
    tag = lambda name: f"{{{svg}}}{name}"
    drawing = ET.fromstring(page.get_svg_image(text_as_path=False))
    drawing.remove(drawing.find(tag("defs")))
    layers = drawing.findall(tag("g"))
    assert len(layers) == len(windows)
    definitions = ET.Element(tag("defs"))
    drawing.insert(0, definitions)
    plate = ET.SubElement(definitions, tag("g"), id="original-case-plate")
    plate.extend(ET.fromstring(source_page.get_svg_image(text_as_path=False)))
    for index, (layer, (target, clip)) in enumerate(zip(layers, windows)):
        scale = min(target.width / clip.width, target.height / clip.height)
        dx = target.x0 + (target.width - scale*clip.width)/2
        dy = target.y0 + (target.height - scale*clip.height)/2
        name = f"panel-view-{index}"
        mask = ET.SubElement(definitions, tag("clipPath"), id=name)
        ET.SubElement(mask, tag("rect"), x=str(dx), y=str(dy),
                      width=str(clip.width*scale), height=str(clip.height*scale))
        layer.clear()
        layer.set("clip-path", f"url(#{name})")
        shift = f"matrix({scale} 0 0 {scale} {dx-scale*clip.x0} {dy-scale*clip.y0})"
        ET.SubElement(layer, tag("use"), {f"{{{xlink}}}href": "#original-case-plate", "transform": shift})
    ET.ElementTree(drawing).write(output, encoding="utf-8", xml_declaration=True)


def render(source, out, native):
    src=fitz.open(source)
    dst=fitz.open()
    p=dst.new_page(width=513,height=286)
    windows=[]
    p.draw_rect(p.rect, color=None, fill=PAPER)
    titles=("Native mask","Local ROH","SegRefiner-LR","Local ROH detail","SegRefiner-LR detail")
    for i,title in enumerate(titles):
        center(p,53+i*101.5,12,title,8.8,True)
    for row,(heading,x,roi,scores,delta,col) in enumerate(CASES):
        y=19+row*124
        p.draw_rect(fitz.Rect(5,y,508,y+15),color=None,fill=(.938,.915,.877))
        label(p,9,y+10.7,heading,8.5,bold=True)
        label(p,376,y+10.7,"Local minus whole  "+delta+" pt",8.5,color=col,bold=True)
        # Viewports on the source PDF have identical crop extents within each case.
        panels=[fitz.Rect(x,76.2,x+95.5,171.7),
                fitz.Rect(x+105,76.2,x+200.5,171.7),
                fitz.Rect(x,204.2,x+95.5,299.7)]
        for i,clip in enumerate(panels+[window(panels[1],roi),window(panels[2],roi)]):
            cx=53+i*101.5
            target=fitz.Rect(cx-42.5,y+18,cx+42.5,y+103)
            # Preserve the source aspect ratio, including the paired detail windows.
            p.show_pdf_page(target,src,0,clip=clip)
            windows.append((target,clip))
            if i<3:
                p.draw_rect(window(target,roi),color=(.92,.59,.19),width=.8)
                center(p,cx,y+113.5,f"IoU  {scores[i]:.2f}",8.7)
            else:
                center(p,cx,y+113.5,"Lower end",7.5)
    y=279
    if native:
        p.draw_line((15,y-3),(34,y-3),color=GREEN,width=1.25)
        label(p,39,y,"Predicted boundary")
        p.draw_line((143,y-3),(162,y-3),color=(.67,.19,.15),dashes="[3 2]",width=1)
        label(p,167,y,"GT outline")
    else:
        for i,tone in enumerate((.89,.68,.49,.26)):
            p.draw_rect(fitz.Rect(14+i*5,y-8,19+i*5,y-2),fill=(tone,tone*.94,tone*.89),color=None)
        label(p,39,y,"Mask fraction")
        p.draw_line((143,y-3),(162,y-3),color=INK,dashes="[3 2]",width=1)
        label(p,167,y,"25% GT coverage")
    p.draw_rect(fitz.Rect(298,y-9,314,y-1),color=(.92,.59,.19),width=.9)
    label(p,320,y,"Shared detail window")
    out.parent.mkdir(parents=True,exist_ok=True)
    dst.save(out.with_suffix(".pdf"),garbage=4,deflate=True)
    editable_svg(p,src[0],windows,out.with_suffix(".svg"))
    dst.close()
    src.close()


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--native",action="store_true")
    args=ap.parse_args()
    render(args.source,args.out,args.native)

