"""Render the ROH at one endpoint. Weight matrices are affine transforms."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import numpy as np
from xml.etree import ElementTree as ET
import fitz

OUT=Path(__file__).resolve().parent/'public'
INK,OR,GREEN,WINE,LINE,PALE='#382B26','#AB5B38','#356B5D','#8F4958','#D7CBBE','#F8F3EB'
PAL=[GREEN,OR,WINE,'#90764F']
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
W,H=516,248
fig=plt.figure(figsize=(W/72,H/72))
ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(H,0),aspect='equal')
ax.axis('off')
def t(x,y,s,fs=8,c=INK,b=False,ha='left'):
    ax.text(x,y,s,fontsize=fs,color=c,weight='bold' if b else 'normal',ha=ha,va='center')
def l(x,y,X,Y,c=LINE,w=.8,d='-'):
    ax.plot([x,X],[y,Y],color=c,lw=w,ls=d)
def arrow(x,y,X,Y,c=INK):
    ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=7,lw=.85,color=c,shrinkA=0,shrinkB=0))
def band(x,y,k):
    for j,color in enumerate([PAL[0]]*4+[PAL[1]]*2+[PAL[2]]*3+[PAL[3]]*5):
        ax.add_patch(Rectangle((x+j*k,y),k-.45,5.8,facecolor=color,edgecolor='none'))
def matrix(x,y,w,h,label,c):
    # Two offset outlines give the weight panel depth; plus bias is explicit.
    ax.add_patch(Rectangle((x+2,y-2),w,h,fc=PALE,ec=LINE,lw=.6))
    ax.add_patch(Rectangle((x,y),w,h,fc='white',ec=c,lw=.95))
    t(x+w/2,y+h/2,label,8.1,c,True,'center')
def relu(x,y):
    arrow(x,y,x+26,y)
    t(x+13,y-8,'ReLU',7.1,INK,False,'center')
def target(x,y,color):
    ax.add_patch(Rectangle((x,y),10,10,fc='white',ec=LINE,lw=.7))
    ax.fill([x,x+10,x+10,x],[y+3,y+7,y+10,y+10],color=color,alpha=.55)
    l(x,y+3,x+10,y+7,color)
def plus(x,y):
    ax.add_patch(Circle((x,y),5.4,fc='white',ec=OR,lw=.9));t(x,y,'+',9,OR,True,'center')

t(8,10,'Endpoint features',10.5,b=True)
t(8,29,'Mask / response',7.8,GREEN,True)
t(8,43,r'$p_i, M_i, \ell_i, d_i$',8)
t(94,29,'Position',7.8,OR,True)
t(94,43,r'$\bar{x}_i,\bar{y}_i$',8)
t(150,29,'Candidate',7.8,WINE,True)
t(150,43,r'$\nu,\eta,s$',8)
t(213,29,'Appearance',7.8,PAL[3],True)
t(213,43,r'$L_i,\bar{L}_3,\bar{L}_5,\Delta_x L,\Delta_y L$',8)
# Four feature groups are joined before standardization.
l(8,55,306,55)
t(313,33,'14',10,b=True)
t(309,46,'features',7)
arrow(344,36,369,36)
t(347,21,'Standardize',7.3,ha='center')
band(376,28,5.4); band(376,39,5.4)
t(415,13,'Duplicate',8.4,GREEN,True,'center')
t(466,32,r'$[z_i,z_i]$',10); t(466,45,'28 inputs',7.5)
l(8,65,508,65)
t(8,78,'Affine weights',10.2,b=True)
t(96,78,'output × input',8)
t(260,78,'Clipped decoder logit',8.3,OR,True)
t(418,78,'Source supervision',8.3,GREEN,True)

def row(y,r,c,name):
    t(8,y-18,name,8.5,c,True)
    t(8,y+2,r'$[z_i,z_i]$',8.1,c)
    arrow(42,y,75,y,c)
    t(8,y+15,'Separate fit · r = '+str(r),7.0,c)
    matrix(78,y-10,40,20,'24 × 28',c)
    relu(121,y)
    matrix(150,y-10,40,20,'24 × 24',c)
    relu(193,y)
    matrix(223,y-10,36,20,str(r)+' × 24',c)
    # Each lane corresponds to a separate trained affine stack and bias set.
    ports=[y] if r==1 else [y-12,y+12]
    l(271,ports[0],271,ports[-1],c)
    l(262,y,271,y,c)
    for k,yy in enumerate(ports,1):
        arrow(271,yy,302,yy,c)
        plus(309,yy)
        arrow(316,yy,327,yy)
        t(335,yy,'σ',11,WINE,True,'center')
        arrow(343,yy,354,yy)
        q=r'$\hat q_i$' if r==1 else rf'$\hat q_i^{{({k})}}$'
        t(362,yy,q,10,GREEN)
        arrow(383,yy,408,yy,GREEN)
        t(411,yy,'BCE',8.2,GREEN,True)
        arrow(465,yy,437,yy,GREEN)
        target(470,yy-5,c)
        t(484,yy,'G' if r==1 else 'G'+str(k),8)
    # Decoder bypass is scalar, applied to every residual.
    t(296,y-32,r'$\mathrm{logit}(p_i^\epsilon)$',8,OR,ha='center')
    if r==1: arrow(309,y-24,309,y-6,OR)
    else:
        l(280,y-27,290,y-27,OR)
        l(290,y-27,290,y+4,OR)
        for yy in ports:
            l(290,yy-8,309,yy-8,OR)
            arrow(309,yy-8,309,yy-5,OR)
row(125,1,GREEN,'Scalar ROH')
l(8,151,508,151)
row(201,2,WINE,'Dual-reference')
t(78,239,r'$h \mapsto Wh+b$',8.5)
t(223,239,'Zero-initialized output layer',7.8)
t(418,239,'Polygon coverage',7.8,GREEN)
OUT.mkdir(exist_ok=True)
name='nora_roh_architecture'
for ext in ('svg','pdf','png'):
    fig.savefig(OUT/(name+'.'+ext),dpi=170,facecolor='white')
plt.close(fig)
page=fitz.open(OUT/(name+'.pdf'))[0]
assert not page.get_images()
assert all(page.rect.contains(fitz.Rect(word[:4])) for word in page.get_text('words'))
assert ET.parse(OUT/(name+'.svg')).find('.//{*}image') is None

