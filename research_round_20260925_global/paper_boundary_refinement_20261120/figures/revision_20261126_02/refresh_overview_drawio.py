"""Refresh the overall-network page while preserving the other editable diagram pages."""
import argparse
import base64
from pathlib import Path
import re
import runpy
import xml.etree.ElementTree as ET


def overview_page(svg_path, template, converter, *, author=False):
    svg = ET.parse(svg_path).getroot()
    _, _, width, height = map(float, svg.attrib["viewBox"].split())
    diagram = ET.Element("diagram", template.attrib)
    page = converter["Page"](diagram, width, height)
    images = svg.findall(".//{*}image")
    if author:
        if len(images) != 1:
            raise ValueError("The local author overview requires one embedded SAR image.")
        bitmap = images[0]
        transform = re.fullmatch(r"matrix\(([^)]+)\)", bitmap.get("transform", ""))
        if transform is None:
            raise ValueError("Expected the native image placement matrix.")
        sx, shear1, shear2, sy, left, top = map(float, transform[1].split())
        if shear1 or shear2 or sx <= 0 or sy <= 0:
            raise ValueError("Unsupported author image transform.")
        box = (left, top, float(bitmap.get("width"))*sx, float(bitmap.get("height"))*sy)
        uri = bitmap.get("{http://www.w3.org/1999/xlink}href", "")
        if not uri.startswith("data:image/png;base64,"):
            raise ValueError("The author bitmap must be embedded as a PNG.")
        encoded = base64.b64encode(base64.b64decode(uri.split(",",1)[1])).decode()
        for parent in svg.iter():
            if bitmap in list(parent):
                parent.remove(bitmap)
                break
    elif images:
        raise ValueError("The public network diagram must consist entirely of vectors.")
    page.visit(svg)
    if author:
        page.cell("shape=image;aspect=fixed;imageAspect=0;image=data:image/png,"+encoded+";", box)
        cells = diagram.find(".//root")
        raster = cells[-1]
        cells.remove(raster)
        for index, cell in enumerate(cells):
            geo = cell.find("mxGeometry")
            if geo is not None and cell.get("style", "").startswith(("shape=rectangle;", "shape=stencil(")) and all(
                abs(float(geo.get(k,0))-v)<.02 for k,v in zip(("x","y","width","height"), box)
            ):
                cells.insert(index,raster)
                break
        else:
            raise ValueError("Could not find the input frame.")
    return diagram


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-svg", type=Path)
    parser.add_argument("--author-output", type=Path)
    args = parser.parse_args()
    if bool(args.author_svg) != bool(args.author_output):
        parser.error("--author-svg and --author-output must be used together.")
    converter = runpy.run_path(str(here.parent/"journal_20261126"/"build_NORA_method_figures.py"))
    books = [here/"NORA_core_revision.drawio", here.parent/"method_atlas_20261126"/"NORA_method_atlas.drawio"]
    for book in books:
        tree = ET.parse(book)
        root = tree.getroot()
        originals = [d for d in root.findall("diagram") if d.get("id")=="nora_network_overview"]
        if len(originals)!=1:
            raise ValueError(f"Expected exactly one overview in {book}.")
        old = originals[0]
        index = list(root).index(old)
        root.remove(old)
        root.insert(index, overview_page(here/"nora_network_overview.svg",old,converter))
        tree.write(book,encoding="utf-8",xml_declaration=True)
    if args.author_svg:
        root = ET.Element("mxfile",host="app.diagrams.net",compressed="false",pages="1")
        template=ET.Element("diagram",id="author-network",name="Author review · Overall architecture")
        root.append(overview_page(args.author_svg,template,converter,author=True))
        ET.ElementTree(root).write(args.author_output,encoding="utf-8",xml_declaration=True)


if __name__=="__main__":
    main()
