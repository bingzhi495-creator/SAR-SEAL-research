"""Rebuild original vector NORA readout and ROH diagrams; optionally add the local published ROI."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch
import fitz

ROOT=Path(__file__).resolve().parent
INK,ORANGE,GREEN,WINE,LINE,CREAM='#382B26','#AB5B38','#356B5D','#8F4958','#D7CBBE','#F8F3EB'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
def frame(h):
    fig=plt.figure(figsize=(510/72,h/72))
    ax=fig.add_axes([0,0,1,1],xlim=(0,510),ylim=(h,0),aspect='equal')
    ax.axis('off')
    return fig,ax
def txt(ax,x,y,s,sz=9,col=INK,bold=False,align='left'):
    ax.text(x,y,s,fontsize=sz,color=col,weight='bold' if bold else 'normal',ha=align,va='center')
def line(ax,x,y,xx,yy,c=LINE,w=1,dash='-'):
    ax.plot([x,xx],[y,yy],color=c,lw=w,ls=dash)
def arr(ax,a,b,c=INK):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=1,color=c))
def poly(ax,points,fill,edge='none',lw=1):
    ax.add_patch(Polygon(points,closed=True,facecolor=fill,edgecolor=edge,lw=lw))
def dot(ax,x,y,r,fc,ec='white'):
    ax.add_patch(Circle((x,y),r,facecolor=fc,edgecolor=ec,lw=.8,zorder=6))
def save(fig,out,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(out/(name+'.'+ext),dpi=160,facecolor='white')
    plt.close(fig)

def readout(out,case=None):
    fig,ax=frame(161)
    txt(ax,12,13,'Boundary context' if case else 'Native grid',11,bold=True)
    txt(ax,173,13,'Polygon coverage',11,GREEN,True)
    txt(ax,344,13,'Decoder readout',11,ORANGE,True)
    if case:
        doc=fitz.open(case)
        pix=doc[0].get_pixmap(matrix=fitz.Matrix(4,4),clip=fitz.Rect(22.5,76,89.5,143),alpha=False)
        im=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)
        ax.imshow(im,extent=(17,108,121,30),interpolation='nearest')
        txt(ax,12,139,'SAR + author outline',8.7)
    else:
        x,y,k=18,31,13
        hull=np.array([(0,.6),(1.5,.3),(4.7,2.8),(6.2,5.5),(5.5,6.3),(3.1,5.2),(.6,2.5)])*k+[x,y]
        poly(ax,hull,CREAM,GREEN,1.3)
        for i in range(8):
            line(ax,x+i*k,y,x+i*k,y+7*k,w=.45)
            line(ax,x,y+i*k,x+7*k,y+i*k,w=.45)
        ax.add_patch(Rectangle((x+4*k,y+2*k),k,k,fill=False,ec=ORANGE,lw=1.7))
        txt(ax,12,139,'Illustrative hull',8.7)
    # Identical native cell and center on both conceptual panels.
    for x in (185,356):
        y,k=31,30
        for j in range(4):
            line(ax,x+j*k,y,x+j*k,y+90,w=.65)
            line(ax,x,y+j*k,x+90,y+j*k,w=.65)
    x,y=185,31
    # The polygon intersects the central cell at (0,.25) and (1,.75):
    # mean occupied height is 0.5 of the cell.
    poly(ax,[(x,y+22.5),(x+90,y+67.5),(x+90,y+90),(x,y+90)],CREAM)
    for j in range(4):
        line(ax,x+j*30,y,x+j*30,y+90,w=.65)
        line(ax,x,y+j*30,x+90,y+j*30,w=.65)
    poly(ax,[(215,68.5),(245,83.5),(245,91),(215,91)],'#8EB5A2')
    line(ax,185,53.5,275,98.5,GREEN,1.6)
    ax.add_patch(Rectangle((215,61),30,30,fill=False,ec=GREEN,lw=1.45))
    dot(ax,230,76,2.2,INK)
    txt(ax,173,139,'Target   qᵢ = 0.50',9.3,GREEN,True)
    ax.add_patch(Rectangle((386,61),30,30,fill=False,ec=GREEN,lw=1.45))
    # Four decoder neighbors, centered at off-grid locations.
    for x,y in ((379,51),(432,51),(379,102),(432,102)):
        line(ax,x,y,401,76,ORANGE,.9)
        dot(ax,x,y,3.4,ORANGE)
    dot(ax,401,76,3,INK)
    txt(ax,344,139,'Four samples → pᵢ',9.3,ORANGE,True)
    txt(ax,173,154,'Illustrative geometry · same native cell center',8)
    save(fig,out,'nora_fractional_readout')

def architecture(out):
    fig,ax=frame(194)
    txt(ax,9,12,'Endpoint features',11,bold=True)
    txt(ax,150,12,'Scalar residual occupancy head',11,bold=True)
    cols=[GREEN,ORANGE,WINE,'#927653']
    for i,(name,n) in enumerate([('Mask / response',4),('Box position',2),('Candidate',3),('Appearance',5)]):
        y=44+i*21
        txt(ax,8,y,name,7.7)
        for j in range(n):
            ax.add_patch(Rectangle((76+j*5,y-3.5),4,7,fc=cols[i],ec='none'))
    txt(ax,8,132,'14 features',9,bold=True)
    arr(ax,(100,76),(118,76))
    stages=[(133,28,'[zᵢ, zᵢ]'),(198,24,'ReLU'),(261,24,'ReLU'),(321,1,'Residual')]
    for i,(x,n,title) in enumerate(stages):
        depth=[6,5,5,1][i]
        for j in range(depth):
            u=x+j*2.1; v=63-j*1.8
            poly(ax,[(u,v),(u+15,v-9),(u+15,v+23),(u,v+32)],['#BFD3C6','#E2B398','#DDB3B9','#B98B65'][i],cols[i],.85)
        txt(ax,x+11,112,str(n),12,cols[i],True,'center')
        txt(ax,x+11,128,title,8.2,align='center')
        if i<3: arr(ax,(x+29,77),(stages[i+1][0]-6,77))
    txt(ax,117,148,'Standardize, duplicate',8,GREEN)
    arr(ax,(341,77),(365,77))
    ax.add_patch(Circle((377,77),9,fc=CREAM,ec=ORANGE,lw=1))
    txt(ax,377,77,'+',12,ORANGE,True,'center')
    line(ax,318,39,377,39,ORANGE)
    arr(ax,(377,39),(377,67),ORANGE)
    txt(ax,318,28,'Clipped decoder logit',8.1,ORANGE)
    arr(ax,(387,77),(407,77))
    t=np.linspace(-3,3,90)
    ax.plot(414+(t+3)*4.7,89-23/(1+np.exp(-t)),color=WINE,lw=2)
    txt(ax,427,108,'Sigmoid',8,align='center')
    arr(ax,(445,77),(464,77))
    txt(ax,478,77,r'$\hat{q}_i$',13,GREEN,True,'center')
    line(ax,478,92,478,164,GREEN,.8,'--')
    line(ax,478,164,392,164,GREEN,.8,'--')
    txt(ax,385,164,'Fractional BCE',8.6,GREEN,True,'right')
    txt(ax,385,180,'Source polygon coverage',8,align='right')
    txt(ax,8,180,'Dual-reference ROH     28 → 24 → 24 → 2',8.7,WINE,True)
    save(fig,out,'nora_roh_architecture')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=Path)
    args=p.parse_args()
    for view in ('public','local'):
        out=ROOT/view
        out.mkdir(exist_ok=True)
        if view=='local' and not args.case: continue
        readout(out,args.case if view=='local' else None)
        architecture(out)

