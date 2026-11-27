"""Render the public, source-matched case figures, optionally the authorized local variants."""
from pathlib import Path
from argparse import ArgumentParser
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

BG="#fffcf6"; INK="#493e36"; MUT="#6e6155"; BORDER="#ac9d8c"
GREEN="#39745c"; RUST="#ac542c"; GT="#eaba78"; MASK="#b7ebc0"
FOCUS="#d7c09a"; PAPER="#eee9df"
coverage=LinearSegmentedColormap.from_list("coverage",[BG,"#a7bdaa","#466f59"])
change=LinearSegmentedColormap.from_list("signed",[RUST,"#f4efe6",GREEN])
matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":7.4,"svg.fonttype":"none","pdf.fonttype":42,"lines.dash_capstyle":"butt"})

def contour(ax,values):
    if values.min()<.5<values.max():
        ax.contour(np.arange(8)+.5,np.arange(8)+.5,values,levels=[.5],colors=[MUT],linewidths=.64,linestyles=[(0,(3,2))],zorder=4)

def mask_outline(mask):
    upper=np.pad(mask,1);segments=[]
    for (dy,dx),a,b in [((-1,0),(0,0),(1,0)),((1,0),(0,1),(1,1)),((0,-1),(0,0),(0,1)),((0,1),(1,0),(1,1))]:
        sy,sx=mask.shape
        y,x=np.nonzero(mask & ~upper[1+dy:1+dy+sy,1+dx:1+dx+sx]);p=np.column_stack((x,y))
        segments.extend(np.stack([p+a,p+b],axis=1))
    return segments

def figure(key,rows,records,save,private=None):
    rs=key=="rsdd"; native=private is not None
    W=516;rh=123;t0=40;H=t0+len(rows)*rh+20
    fig=plt.figure(figsize=(W/72,H/72),facecolor=BG)
    def label(x,y,s,**kw):
        opts=dict(fontsize=7.8,color=INK,va="center",ha="left");opts.update(kw)
        return fig.text(x/W,1-y/H,s,**opts)
    def rule(x1,x2,y,c=BORDER,w=.45):
        fig.add_artist(Line2D([x1/W,x2/W],[1-y/H]*2,transform=fig.transFigure,color=c,lw=w))
    starts=[7,103,199,295];size=85
    total="44 matched hulls" if rs else "425 matched hulls"
    dist="25 gains   /   11 losses   /   8 ties" if rs else "268 gains   /   154 losses   /   3 ties"
    label(7,8,"Instance-RSDD reserve  ·  D-FINE" if rs else "Gaofen-3  ·  RF-DETR fit B",fontsize=8.5,weight="bold")
    label(509,8,total+"   ·   "+dist,ha='right',fontsize=7.3);rule(7,509,18)
    baseline="Bounded (L)" if rs else "Dynamic attraction";lead="Independent (I)" if rs else "ROH ranking"
    headers=[("SAR + GT outline" if native else "GT coverage"),baseline,lead,("Foreground change" if rs else "Intersection change")]
    for x,h in zip(starts,headers):label(x+size/2,29,h,fontsize=7.2,weight="bold",ha="center")
    label(451,29,"Whole-case overlap",fontsize=7.5,weight="bold",ha="center")
    rights=["I only","L only"] if rs else ["ROH only","Dynamic only"]
    for j,idx in enumerate(rows):
        c=records[idx]; base=t0+j*rh;z=np.asarray(c['pooled']);gain=c['delta_iou_points']>0;accent=GREEN if gain else RUST
        x,y=c['focus_bin_xy'];b0,m0=c['iou'][-2:]
        if native:
            sample=np.load(private/f"{key}_{idx}.npz");gray=sample["image"];coords=sample["outline"];masks=sample["masks"];fractions=sample["truth"];n=len(gray)
        else:n=8
        edges=np.linspace(0,n,9).astype(int);zoom=[edges[x],edges[y],edges[x+4],edges[y+4]]
        label(7,base+2,"("+chr(97+j)+")  "+("Gain" if gain else "Loss"),weight='bold',color=accent)
        label(57,base+2,"ΔIoU "+f"{c['delta_iou_points']:+.2f} pt",color=accent,weight='bold')
        label(183,base+2,f"{c['swaps']} added  /  {c['swaps']} removed per edited mask",fontsize=7.1,color=MUT)
        for col,left in enumerate(starts):
            ax=fig.add_axes((left/W,1-(base+11+size)/H,size/W,size/H))
            ax.set_facecolor(PAPER);ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set(color=BORDER,lw=.5)
            if native:
                if col < 3:
                    ax.imshow(gray,cmap="gray",vmin=0,vmax=255,extent=(0,n,n,0),interpolation="nearest",resample=False)
                    if col>0:
                        ax.add_collection(LineCollection(mask_outline(masks[col-3]),colors=MASK,linewidths=.64,zorder=3))
                    line=ax.plot(coords[:,0],coords[:,1],color=GT,ls=(0,(3,2.2)),lw=.8,zorder=4)[0]
                    line.set_path_effects([pe.Stroke(linewidth=1.22,foreground="#493827"),pe.Normal()])
                else:
                    delta=masks[-1].astype(int)-masks[-2]
                    a=delta if rs else delta*fractions
                    ax.pcolormesh(np.arange(n+1),np.arange(n+1),a,cmap=change,vmin=-1,vmax=1,edgecolors='none',antialiased=False)
                    ax.plot(coords[:,0],coords[:,1],color=MUT,lw=.78,ls=(0,(3,2)),zorder=3)
            else:
                # The last matrix is already pooled in the published spatial convention.
                if col<3:
                    arr=z[0] if col==0 else z[col-4]
                    ax.pcolormesh(np.arange(9),np.arange(9),arr,cmap=coverage,vmin=0,vmax=1,edgecolors="#e7e0d6",linewidth=.22,antialiased=False)
                    contour(ax,z[0])
                else:
                    lim=max(float(np.abs(np.asarray(v['pooled'][-1])).max()) for v in records)
                    ax.pcolormesh(np.arange(9),np.arange(9),z[-1],cmap=change,vmin=-lim,vmax=lim,edgecolors="#e7e0d6",linewidth=.28,antialiased=False)
                    contour(ax,z[0])
            if col==2:
                rr=Rectangle((zoom[0],zoom[1]),zoom[2]-zoom[0],zoom[3]-zoom[1],fill=False,ec=(FOCUS if native else MUT),lw=.87,ls=(0,(5,2.8)),zorder=8);ax.add_patch(rr)
            if col==3:ax.set(xlim=(zoom[0],zoom[2]),ylim=(zoom[3],zoom[1]))
            else:ax.set(xlim=(0,n),ylim=(n,0))
            under=[("Registered crop" if native else "8 × 8 display grid"),f"IoU  {b0:.2f}",f"IoU  {m0:.2f}","Marked window"][col]
            label(left+size/2,base+103,under,fontsize=7.4,ha="center",color=MUT)
        label(393,base+20,"Intersection change (px²)",weight="bold",fontsize=7.5)
        for r,(name,num) in enumerate(zip(['Additions','Removals'],[c['addition_contribution'],c['removal_contribution']])):
            label(393,base+39+r*14,name,fontsize=7.7);label(506,base+39+r*14,("0.000" if abs(num)<.0005 else f"{num:+.3f}"),ha='right',fontsize=8)
        rule(393,507,base+64);label(393,base+77,"Net change",fontsize=7.5,weight='bold');label(506,base+77,f"{c['delta_intersection']:+.3f}",ha='right',fontsize=8.6,weight='bold',color=accent)
        rule(7,509,base+114)
    yy=H-15
    if native:
        label(7,yy,"Dashed: GT outline   ·   Solid: predicted outline",fontsize=7.1,color=MUT)
    else:
        label(7,yy,"Darker green: higher coverage   ·   Dashed: 50% GT coverage",fontsize=6.9,color=MUT)
    label(301,yy,"■",color=GREEN);label(313,yy,rights[0] if rs else "Intersection gain",fontsize=7.1)
    label(402,yy,"■",color=RUST);label(414,yy,rights[1] if rs else "Intersection loss",fontsize=7.1)
    for fmt in ['pdf','svg']:fig.savefig(save.with_suffix('.'+fmt),dpi=250,facecolor=BG)
    plt.close(fig)

def main():
    ap=ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=Path(__file__).with_name('case_aggregates.json'));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--private-npz',type=Path)
    a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True);data=json.loads(a.data.read_text(encoding='utf-8'))
    for k,rows,filename in [('rsdd',[0,1],'rsdd74_equal_placement'),('gf3',[1,3],'gf3_matched_ranking'),('gf3',[1,0,3,2],'gf3_matched_ranking_all')]:
        figure(k,rows,data[k],a.output/filename,a.private_npz)
if __name__=='__main__':main()