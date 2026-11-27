"""Revision of ordinary manuscript assets; never opens experiment payloads."""
from pathlib import Path
import json,re,shutil,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,Polygon,Circle
import pymupdf as fitz

INK='#473830'; GRAY='#756C65'; LINE='#DED3C8'; PAPER='#F8F3EB'
ORANGE='#A95B36'; GREEN='#477163'; RED='#A14A44'; WINE='#804859'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
import argparse
parser=argparse.ArgumentParser(description='Rebuild five NORA explanatory vector figures.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--figure', choices=['overview','roh','selection','allocation','phoenix'], help='Rebuild one panel family; default builds all five.')
args=parser.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
def canvas(h):
 f=plt.figure(figsize=(516/72,h/72));a=f.add_axes([0,0,1,1],xlim=(0,516),ylim=(h,0));a.axis('off');return f,a
def t(a,x,y,s,fs=8,c=INK,b=False,ha='left'):
 return a.text(x,y,s,fontsize=fs,color=c,weight='bold' if b else 'normal',va='center',ha=ha,zorder=15)
def box(a,x,y,w,h,fc=PAPER,ec=LINE):
 a.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,lw=.7,zorder=2))
def arrow(a,x,y,X,Y,c=GRAY,dash=False):
 a.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=7,lw=.8,color=c,linestyle='--' if dash else '-',shrinkA=0,shrinkB=0,zorder=5))
def line(a,x,y,X,Y,c=LINE):a.plot([x,X],[y,Y],lw=.7,color=c,zorder=3)
def stack(a,x,y,w,h,color=ORANGE,label=None):
 for k in [2,1,0]:box(a,x+k*3,y-k*3,w,h,PAPER,color)
 if label:t(a,x+w/2,y+h/2,label,8,color if False else INK,True,'center')
def save(f,name):
 f.canvas.draw();r=f.canvas.get_renderer()
 for item in f.findobj(matplotlib.text.Text):
  if item.get_text() and item.get_visible():
   b=item.get_window_extent(r)
   assert b.x0>=-1 and b.y0>=-1 and b.x1<=f.bbox.x1+1 and b.y1<=f.bbox.y1+1,(name,item.get_text())
 for ext in ['pdf','svg','png']:f.savefig(OUT/f'{name}.{ext}',dpi=170,facecolor='white')
 plt.close(f)

def overview():
 f,a=canvas(232)
 for x,label in [(8,'Frozen prediction'),(151,'Native boundary'),(297,'Occupancy'),(403,'Mask update')]:t(a,x,13,label,9,b=True)
 line(a,8,26,508,26)
 # A vector schematic is shared by both versions; no observed detector feature maps are implied.
 box(a,10,56,49,51)
 a.add_patch(Polygon([(21,96),(40,63),(49,66),(43,80),(29,99)],fc='#BCA995',ec=GRAY,lw=.8,zorder=6))
 t(a,34,120,'SAR image',7.8,ha='center')
 arrow(a,61,81,74,81,ORANGE)
 stack(a,77,59,55,42,ORANGE)
 t(a,104,73,'RF-DETR',7.5,b=True,ha='center');t(a,104,86,'or D-FINE',7.5,ha='center')
 t(a,104,119,'Frozen decoder',7.5,ha='center')
 arrow(a,135,81,153,81)
 box(a,155,57,120,50)
 for yy,label in [(69,r'$M,B$  inner / outer rings'),(86,r'$p_i$  uncertainty order')]:t(a,162,yy,label,8)
 t(a,215,120,r'$S_+,S_-$  up to 32 / side',7.7,ha='center')
 arrow(a,277,81,296,81)
 stack(a,298,57,71,50,GREEN)
 t(a,334,69,'ROH',9,GREEN,True,'center')
 t(a,334,84,'14 → 28 inputs',7.5,ha='center')
 t(a,334,99,'24 → 24 → 1 / 2',7.4,ha='center')
 arrow(a,375,70,402,61,GREEN);arrow(a,375,95,402,113,WINE)
 box(a,405,40,102,41,'#E4ECE4');box(a,405,97,102,41,'#F3E3DA')
 t(a,456,51,'Scalar scores',8,GREEN,True,'center');t(a,456,68,r'$F,L,I:\quad +k,-k$',8,ha='center')
 t(a,456,108,'Dual scores',8,WINE,True,'center');t(a,456,125,r'$+a,-(2k_s-a)$',8,ha='center')
 # Explicit mask and metadata bypasses, separated from fitting.
 line(a,131,52,131,42,GRAY);line(a,131,42,394,42,GRAY)
 line(a,394,42,394,153,GRAY);arrow(a,394,153,437,153,GRAY)
 t(a,279,33,'ID · class · box · confidence',7.2,GRAY,ha='center')
 line(a,215,124,215,166);arrow(a,215,166,437,166)
 t(a,304,159,r'Binary mask $M$',7.4,GRAY,ha='center')
 grid(a,439,149,3.4,np.array([[0,0,0,1,0,0,0,0],[0,0,1,1,1,0,0,0],[0,1,1,1,1,1,0,0],[0,0,1,1,1,1,1,0],[0,0,0,1,1,0,0,0]],bool))
 t(a,486,163,r'$M^{\prime}$',9,b=True)
 line(a,507,60,512,60);line(a,512,60,512,146);arrow(a,512,146,456,146)
 arrow(a,456,139,456,149)
 arrow(a,34,127,34,145);line(a,34,145,292,145);arrow(a,292,145,311,111)
 t(a,118,136,'Image appearance',7.4,GRAY,ha='center')
 line(a,8,178,508,178)
 t(a,8,193,'Scalar fit',8,GREEN,True)
 t(a,84,193,r'Polygon $G$',8)
 arrow(a,137,193,161,193,GREEN,True)
 t(a,173,193,r'$q_i=|C_i\cap G|/|C_i|$',8,GREEN)
 arrow(a,337,193,359,193,GREEN,True);t(a,370,193,'BCE → scalar ROH',8)
 t(a,8,215,'Dual fit',8,WINE,True)
 t(a,84,215,'PSeg COCO raster + SPSD panoptic',8)
 arrow(a,239,215,252,215,WINE,True)
 t(a,260,215,r'$(y_{i1},y_{i2})\in\{0,1\}^2$',8,WINE)
 arrow(a,337,215,359,215,WINE,True);t(a,370,215,'BCE → dual ROH',8)
 save(f,'nora_network_overview')

def roh():
 f,a=canvas(228)
 t(a,8,13,r'One endpoint $i$',9,b=True)
 groups=[(8,4,'Mask / response',GREEN),(136,2,'Position',ORANGE),(224,3,'Candidate',WINE),(333,5,'Appearance',GRAY)]
 for x,n,label,col in groups:
  t(a,x,34,label,8,col,True)
  for j in range(n):box(a,x+j*10,43,8,8,col,col)
 t(a,487,46,r'$v_i\in\mathbb{R}^{14}$',8,ha='right')
 t(a,8,67,r'$z_i=(v_i-\mu)/\sigma_v$',8)
 arrow(a,132,66,161,66)
 t(a,173,67,r'$[z_i,z_i]\in\mathbb{R}^{28}$',8,GREEN)
 t(a,329,67,'Same endpoint duplicated',7.7,GRAY)
 arrow(a,274,67,291,67,GREEN);line(a,291,67,291,75);line(a,291,75,106,75)
 line(a,106,75,106,171);arrow(a,106,113,111,113,GREEN);arrow(a,106,171,111,171,WINE)
 line(a,8,80,508,80)
 for y,r,col,name,params in [(113,1,GREEN,'Scalar ROH','1,321'),(171,2,WINE,'Dual ROH','1,346')]:
  t(a,8,y-14,name,8.4,col,True);t(a,8,y+2,params+' parameters',7.5,GRAY)
  for x,w,txt in [(111,48,'24 × 28'),(201,48,'24 × 24'),(291,43,f'{r} × 24')]:
   stack(a,x,y-12,w,23,col);t(a,x+w/2,y,txt,8,ha='center')
  for x in [163,253]:arrow(a,x,y,x+32,y);t(a,x+16,y-14,'ReLU',7.4,ha='center')
  arrow(a,338,y,358,y,col);a.add_patch(Circle((365,y),6,fc='white',ec=ORANGE,lw=.8));t(a,365,y,'+',10,ORANGE,True,'center')
  arrow(a,372,y,392,y);t(a,402,y,'σ',11,b=True,ha='center');arrow(a,413,y,431,y)
  t(a,438,y,r'$\hat q_i$' if r==1 else r'$\hat q_{i1},\hat q_{i2}$',9,GREEN)
  t(a,365,y-25,r'$\mathrm{logit}(p_i^\epsilon)$',8,ORANGE,ha='center');arrow(a,365,y-18,365,y-6,ORANGE)
 line(a,8,186,508,186)
 t(a,8,198,'Scalar target',7.9,GREEN,True)
 t(a,77,198,r'$q_i$: polygon–cell area fraction',8)
 t(a,8,216,'Dual targets',7.9,WINE,True)
 t(a,77,216,r'$y_{i1}$: PSeg COCO raster',8)
 t(a,205,216,r'$y_{i2}$: SPSD panoptic mask',8)
 t(a,355,216,'Both binary (0 / 1)',8,WINE)
 t(a,355,198,'Matrices: output × input',7.8,GRAY)
 save(f,'nora_roh_architecture')

def grid(a,x,y,step,mask,adds=(),drops=(),links=False,candidates=()):
 for row,col in np.ndindex(mask.shape):
  pos=(col,row);fc=GREEN if pos in adds else RED if pos in drops else '#CDBBA8' if mask[row,col] else PAPER
  box(a,x+col*step,y+row*step,step,step,fc,LINE)
  if pos in candidates:a.add_patch(Circle((x+(col+.5)*step,y+(row+.5)*step),step*.25,fill=False,ec=ORANGE,lw=.85,zorder=6))
 for cells,sgn in [(adds,'+'),(drops,'−')]:
  for col,row in cells:t(a,x+(col+.5)*step,y+(row+.5)*step,sgn,step*.8,'white',True,'center')
 if links:
  for (x1,y1),(x2,y2) in zip(adds,drops):line(a,x+(x1+.5)*step,y+(y1+.5)*step,x+(x2+.5)*step,y+(y2+.5)*step,INK)

def selection():
 mask=np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,1,1,1,0,0,0,0],[0,1,1,1,1,1,1,0],[0,0,1,1,1,1,1,0],[0,0,0,0,0,1,0,0]],bool)
 adds=((2,1),(4,1),(4,2),(5,2));drops=((2,2),(3,2),(4,3),(5,3))
 qa=np.array([.92,.82,.50,.30]);qd=np.array([.70,.33,.10,.20]);ids=range(4)
 dist=((np.array(adds)[:,None]-np.array(drops)[None,:])**2).sum(2)
 assign=min(itertools.permutations(ids),key=lambda p:sum(dist[i,j] for i,j in enumerate(p)))
 F=sorted(enumerate(assign),key=lambda ij:qa[ij[0]]-qd[ij[1]],reverse=True)[:2]
 rho2=max(dist[i,j] for i,j in F)
 choices=[tuple(zip(ai,di)) for ai in itertools.combinations(ids,2) for di in itertools.permutations(ids,2) if all(dist[i,j]<=rho2 for i,j in zip(ai,di))]
 L=max(choices,key=lambda p:sum(qa[i]-qd[j] for i,j in p))
 I=list(zip(np.argsort(-qa)[:2],np.argsort(qd)[:2]))
 gains=[sum(qa[i]-qd[j] for i,j in p) for p in [F,L,I]]
 assert rho2==2;np.testing.assert_allclose(gains,[.89,.99,1.44])
 f,a=canvas(190)
 t(a,8,12,'Shared mask · shared scores · +2 / −2 cells',9,b=True)
 for n,(pairs,title) in enumerate(zip([F,L,I],['F  Distance-first','L  Bounded re-pairing','I  Independent'])):
  x=8+171*n;t(a,x,34,title,9,b=True)
  ap=[adds[i] for i,j in pairs];dp=[drops[j] for i,j in pairs]
  grid(a,x+22,49,12,mask,ap,dp,n<2,adds+drops)
  t(a,x+70,133,[r'$\rho=\sqrt{2}$ px',r'Each edge $\leq\rho$','No edge constraint'][n],8,ha='center')
  t(a,x+70,152,rf'$\hat g={gains[n]:.2f}$ cell areas',9,ORANGE,True,'center')
  if n<2:line(a,x+157,30,x+157,162)
 t(a,8,178,r'$\rho$: longest retained F edge.  Orange rings: shared candidate sites.',8,GRAY)
 save(f,'method_bounded_selection')
 (OUT/'schematic_selection.json').write_text(json.dumps(dict(add_sites=adds,remove_sites=drops,add_scores=qa.tolist(),remove_scores=qd.tolist(),F=F,L=L,I=[list(map(int,p)) for p in I],rho_squared=int(rho2),gains=gains),indent=2),encoding='utf-8')

def allocation():
 f,a=canvas(172);t(a,8,13,r'Same support · same dual scores · four flips ($2k_s=4$)',9,b=True)
 mask=np.zeros((6,7),bool);mask[2:5,1:6]=True
 adds=((2,1),(5,1),(6,2),(0,4));drops=((1,2),(2,2),(5,3),(4,4))
 for n,(ap,dp,title) in enumerate([(adds[:2],drops[:2],'Equal split'),(adds[:3],drops[:1],'Variable split')]):
  x=12+n*253;t(a,x,39,title,9,b=True)
  grid(a,x,53,11,mask,ap,dp)
  t(a,x+91,65,f'+{len(ap)} added',8.5,GREEN,True);t(a,x+91,84,f'−{len(dp)} removed',8.5,RED,True)
  t(a,x+91,105,'Area: 15 → '+str(15+len(ap)-len(dp))+' cells',8.5,b=True)
  t(a,x+91,124,r'$\Delta$area = '+str(len(ap)-len(dp))+' cells',8,ORANGE)
 line(a,253,32,253,137);line(a,8,143,508,143)
 t(a,258,158,r'Common objective: $\max\ \min_r[\hat f_r(M^{\prime})-\hat f_r(S)]$',9,ha='center')
 save(f,'nora_signed_budget')

def phoenix():
 f,a=canvas(161);t(a,8,13,'Frozen Phoenix response · support × write rule',9,b=True)
 mask=np.zeros((7,8),bool);mask[1:6,1:7]=True;mask[0,3:5]=True
 patterns=[(((0,2),(0,4),(7,3)),((1,2),(1,4),(6,3))),(((0,2),(0,4),(7,3)),((1,2),)),(((7,1),(7,6),(0,2),(0,4),(7,3)),((1,2),(3,3),(1,4),(6,3))),(((7,1),(7,6),(0,2)),((1,2),))]
 t(a,9,43,'Sampled',8,b=True);t(a,9,58,r'$S_+\cup S_-$',8)
 t(a,9,100,'Full grid',8,b=True);t(a,9,115,r'$\Omega$',8)
 for n,(x,y,title) in enumerate([(94,40,'Phoenix threshold'),(300,40,'Exact total'),(94,97,None),(300,97,None)]):
  ap,dp=patterns[n];grid(a,x,y,6,mask,ap,dp)
  if title:t(a,x,29,title,8.3,b=True)
  t(a,x+58,y+18,str(len(ap)+len(dp))+' flips',8,ORANGE,True)
  if n in [1,3]:t(a,x+58,y+33,r'$2k_{\mathrm{att}}=4$',8)
 line(a,277,26,277,142);line(a,8,145,508,145)
 t(a,258,155,'One candidate and posterior; five iterations. Counts are illustrative.',7.7,GRAY,ha='center')
 save(f,'nora_frozen_writer')


if __name__=="__main__":
 builders = {'overview':overview, 'roh':roh, 'selection':selection, 'allocation':allocation, 'phoenix':phoenix}
 if args.figure:
  builders[args.figure]()
 else:
  for draw in builders.values():
   draw()
