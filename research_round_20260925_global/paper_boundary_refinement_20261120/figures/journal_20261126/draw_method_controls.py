"""Rebuild three illustrative diagrams from embedded published schematic fixtures."""
SELECTION = [([[0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 1, 0, 0, 0],
   [0, 1, 1, 0, 1, 0, 0, 0],
   [0, 1, 1, 1, 0, 1, 1, 0],
   [0, 0, 1, 1, 1, 1, 1, 0],
   [0, 0, 0, 0, 0, 1, 0, 0]],
  [(4, 1), (4, 2)],
  [(3, 2), (4, 3)],
  0.89),
 ([[0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 1, 0, 0, 0, 0, 0],
   [0, 1, 1, 0, 1, 0, 0, 0],
   [0, 1, 1, 1, 0, 1, 1, 0],
   [0, 0, 1, 1, 1, 1, 1, 0],
   [0, 0, 0, 0, 0, 1, 0, 0]],
  [(2, 1), (4, 2)],
  [(3, 2), (4, 3)],
  0.99),
 ([[0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 1, 0, 1, 0, 0, 0],
   [0, 1, 1, 1, 0, 0, 0, 0],
   [0, 1, 1, 1, 0, 0, 1, 0],
   [0, 0, 1, 1, 1, 1, 1, 0],
   [0, 0, 0, 0, 0, 1, 0, 0]],
  [(2, 1), (4, 1)],
  [(4, 3), (5, 3)],
  1.44)]
ALLOC = ([[0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 0],
  [0, 1, 1, 1, 1, 1, 0],
  [0, 1, 1, 1, 1, 1, 0],
  [0, 0, 0, 0, 0, 0, 0]],
 ((2, 1), (5, 1), (6, 2), (0, 4)),
 ((1, 2), (2, 2), (5, 3), (4, 4)),
 [([0, 1], [0, 1]), ([0, 1, 2], [0])])
PHOENIX = ([[0, 0, 0, 1, 1, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 1, 0],
  [0, 1, 1, 1, 1, 1, 1, 0],
  [0, 1, 1, 1, 1, 1, 1, 0],
  [0, 1, 1, 1, 1, 1, 1, 0],
  [0, 1, 1, 1, 1, 1, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0]],
 {('narrow', 'balanced'): (((0, 2), (0, 4)), ((1, 2), (1, 4))),
  ('narrow', 'count'): (((0, 2), (0, 4), (7, 3)), ((1, 2),)),
  ('narrow', 'threshold'): (((0, 2), (0, 4), (7, 3)), ((1, 2), (1, 4), (6, 3))),
  ('whole', 'count'): (((7, 1), (7, 6), (0, 2)), ((1, 2),)),
  ('whole', 'threshold'): (((7, 1), (7, 6), (0, 2), (0, 4), (7, 3)),
                           ((1, 2), (3, 3), (1, 4), (6, 3)))})

from argparse import ArgumentParser
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"font.family":"DejaVu Sans"})
OUT = None
NAVY,BLUE,TEAL,RUST,OCHRE,GRAY="#473830","#A95B36","#477163","#A14A44","#B07B3E","#756C65"

def canvas(width,height):
    fig=plt.figure(figsize=(width/72,height/72))
    ax=fig.add_axes([0,0,1,1],xlim=(0,width),ylim=(height,0),aspect="equal")
    ax.axis("off")
    return fig,ax

def label(ax,x,y,message,size=9,**kw):
    return ax.text(x,y,message,fontsize=size,va="center",color=kw.pop("color",NAVY),**kw)

def grid(ax,x,y,step,mask,adds,drops,links=False):
    mask=np.array(mask)
    for r,c in np.ndindex(mask.shape):
        site=(c,r)
        color=TEAL if site in adds else RUST if site in drops else "#CDBBA8" if mask[r,c] else "#F5EEE4"
        ax.add_patch(Rectangle((x+c*step,y+r*step),step,step,fc=color,ec="#DED3C8",lw=.42))
    if links:
        for (a,b),(c,d) in zip(adds,drops):
            ax.plot([x+(a+.5)*step,x+(c+.5)*step],[y+(b+.5)*step,y+(d+.5)*step],color=NAVY,lw=1.2)
    for cells,sign in [(adds,"+"),(drops,"−")]:
        for c,r in cells:
            label(ax,x+(c+.5)*step,y+(r+.5)*step,sign,step*.89,ha="center",weight="bold",color="white")

def save(fig,name):
    for ext in ("svg","pdf"):
        fig.savefig(OUT/f"{name}.{ext}",facecolor="white")
    plt.close(fig)

def selection():
    f,ax=canvas(510,158)
    label(ax,8,12,"Illustrative endpoint selection",10,weight="bold")
    label(ax,501,12,"+2 / −2 cells",9,ha="right")
    for i,(mask,a,d,g) in enumerate(SELECTION):
        x=8+168*i
        label(ax,x,36,["F   Distance-first","L   Bounded re-pairing","I   Independent ranking"][i],10,weight="bold")
        grid(ax,x+20,48,12,mask,a,d,i<2)
        label(ax,x,132,["Max edge  √2 px","Edge cap  √2 px","Unpaired"][i],8.7)
        label(ax,x,148,f"Gain   {g:.2f} cell areas",9.5,weight="bold",color=BLUE)
        if i<2: ax.plot([x+152]*2,[30,151],color="#DED3C8",lw=.8)
    save(f,"method_bounded_selection")

def allocation():
    mask,add,drop,actions=ALLOC
    f,ax=canvas(510,144)
    label(ax,8,12,"Four flips · 15-cell mask",10,weight="bold")
    label(ax,501,12,"Δarea = additions − removals",8.7,ha="right")
    for i,(a,d) in enumerate(actions):
        x=8+i*250
        label(ax,x,38,["(a)  Equal split","(b)  Variable split"][i],11,weight="bold")
        grid(ax,x,51,13,mask,[add[j] for j in a],[drop[j] for j in d])
        label(ax,x+103,68,f"+{len(a)} cells",10,color=TEAL,weight="bold")
        label(ax,x+103,87,f"−{len(d)} cells",10,color=RUST,weight="bold")
        delta=len(a)-len(d)
        label(ax,x+103,111,f"Area  15 → {15+delta} cells",10,weight="bold")
        label(ax,x+103,128,["Δarea  0","Δarea  +2"][i],8.5)
    ax.plot([245]*2,[31,135],color="#DED3C8",lw=.8)
    save(f,"method_signed_allocation")

def phoenix():
    mask,examples=PHOENIX
    f,ax=canvas(247,249)
    label(ax,8,12,"Frozen Phoenix XL1",11,weight="bold")
    label(ax,8,28,"Illustration · five iterations",8.5)
    for i,mode in enumerate(("threshold","count")):
        x=60+i*96
        label(ax,x+33,49,["Threshold","Exact total"][i],9.5,ha="center",weight="bold",color=BLUE)
        for j,scope in enumerate(("narrow","whole")):
            y=62+j*76
            a,d=examples[scope,mode]
            if i==0: label(ax,7,y+27,scope.title(),9,weight="bold")
            grid(ax,x,y,7.2,mask,a,d)
            label(ax,x+28,y+61,f"+{len(a)} / −{len(d)}",9,ha="center")
    ax.plot([8,239],[211]*2,color="#DED3C8",lw=.9)
    label(ax,8,226,"Narrow support · four flips",9.3,weight="bold")
    label(ax,8,242,"Free split  +3 / −1",8.7)
    label(ax,135,242,"Balanced  +2 / −2",8.7)
    save(f,"method_phoenix_controls")

def main():
    global OUT
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    OUT = parser.parse_args().output
    OUT.mkdir(parents=True, exist_ok=True)
    selection()
    allocation()
    phoenix()



if __name__ == "__main__":
    main()
