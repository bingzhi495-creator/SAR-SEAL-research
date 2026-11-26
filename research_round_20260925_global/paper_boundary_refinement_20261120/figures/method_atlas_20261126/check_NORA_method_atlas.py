from pathlib import Path
from urllib.parse import unquote
from base64 import b64decode
import re,zlib,xml.etree.ElementTree as ET
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,default=Path(__file__).with_name('NORA_method_atlas.drawio'))
p=ap.parse_args().source
doc=ET.parse(p)
pages=doc.findall('diagram')
assert len(pages)==10
for pg in pages:
    cells=pg.findall('.//mxCell')
    assert len(cells)==len({c.get('id') for c in cells})
    counts={'rect':0,'edge':0,'label':0,'stencil':0}
    for c in cells[2:]:
        s=c.get('style')
        assert 'image=' not in s and 'data:image' not in s
        if 'stencil(' in s:
            encoded=re.search(r'stencil\(([^)]+)\)',s)[1]
            shape=ET.fromstring(unquote(zlib.decompress(b64decode(encoded),-15).decode()))
            assert shape.find('foreground/path') is not None
            counts['stencil']+=1
        elif c.get('edge'): counts['edge']+=1
        elif s.startswith('text;'): counts['label']+=1
        else:
            assert s.startswith('shape=rectangle;')
            counts['rect']+=1
    print(pg.get('name'),counts)
roh=next(pg for pg in pages if pg.get("id")=="nora_roh_architecture")
values=[c.get("value") for c in roh.findall(".//mxCell")]
for label, count in (("24 × 28",2),("24 × 24",2),("1 × 24",1),("2 × 24",1)):
    assert values.count(label)==count,(label,values)
assert pages[2].get("id")=="nora_network_overview"
