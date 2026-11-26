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
        if i<2:
            ax.plot([x+8,x+30],[132,132],lw=1.3,color=OCHRE)
            for tip in (x+8,x+30):
                ax.plot([tip,tip],[129,135],lw=1.3,color=OCHRE)
            label(ax,x+35,132,["√2 px","≤ √2 px"][i],9)
        else:
            label(ax,x+8,132,"Unpaired",8.6)
        label(ax,x,148,f"Gain   {g:.2f} cell areas",9.5,weight="bold",color=BLUE)
        if i<2: ax.plot([x+152]*2,[30,151],color="#DED3C8",lw=.8)
    save(f,"method_bounded_selection")

def allocation():
    mask,add,drop,actions=ALLOC
    f,ax=canvas(510,144)
    label(ax,8,12,"Four flips · 15-cell mask",10,weight="bold")

    for i,(a,d) in enumerate(actions):
        x=8+i*250
        label(ax,x,38,["(a)  Equal split","(b)  Variable split"][i],11,weight="bold")
        grid(ax,x,51,13,mask,[add[j] for j in a],[drop[j] for j in d])
        label(ax,x+105,62,f"+{len(a)}",11,color=TEAL,weight="bold")
        label(ax,x+135,62,f"−{len(d)}",11,color=RUST,weight="bold")
        delta=len(a)-len(d)
        for row,n in enumerate((15,15+delta)):
            y=83+row*21
            label(ax,x+104,y+3,["In","Out"][row],8.2)
            for pos in range(n):
                ax.add_patch(Rectangle((x+125+pos*4.7,y),4.2,7,fc=TEAL if pos>=15 else "#CDBBA8",ec="none"))
            label(ax,x+210,y+4,str(n),8.7,weight="bold")
        label(ax,x+104,127,f"Δarea  {delta:+d} cells" if delta else "Δarea  0 cells",8.6)
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
            if i==0:
                for r in range(5):
                    for c in range(6):
                        ring=(1<=r<=3 and 1<=c<=4 and (r in (1,3) or c in (1,4)))
                        active=scope=="whole" or ring
                        ax.add_patch(Rectangle((11+c*4.4,y+8+r*4.4),4.1,4.1,
                                               fc="#B4896E" if active else "#F5EEE4",ec="none"))
                label(ax,8,y+40,scope.title(),8.5,weight="bold")
            grid(ax,x,y,7.2,mask,a,d)
            label(ax,x+28,y+61,f"+{len(a)} / −{len(d)}",9,ha="center")
    ax.plot([8,239],[211]*2,color="#DED3C8",lw=.5)
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
