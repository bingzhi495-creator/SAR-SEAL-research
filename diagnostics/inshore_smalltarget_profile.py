# CPU-only profiling of PSeg-SSDD ship-size & clustering, focused on inshore small targets.
# Goal: quantify (a) how SMALL inshore ships are (px^2, COCO s/m/l), (b) how DENSE/abutting they
# are (ships/img, nearest-neighbour center dist & bbox IoU), (c) how many px a small ship spans on
# the 512x512 input and the 1/4-stride (128x128) mask feature, under the actual test pipeline
# Resize(scale=(512,512), keep_ratio=True).  Zero-hallucination: everything is computed from the JSON.
import json, math, statistics as st
from itertools import combinations

ANN='E:/SAR/seal/data/pseg_ssdd/annotations/%s.json'
SETS=['train','test','test_inshore','test_offshore']
INPUT=512          # test Resize scale (keep_ratio=True)
STRIDE=4           # M2F mask_features stride -> feature map = 512/4 = 128
SMALL=32*32        # COCO small  = 1024 px^2
MED=96*96          # COCO medium = 9216 px^2

def pct(v,p):  # percentile, no numpy dependency surprises
    v=sorted(v);
    if not v: return float('nan')
    k=(len(v)-1)*p/100.0; f=math.floor(k); c=math.ceil(k)
    if f==c: return v[int(k)]
    return v[f]+(v[c]-v[f])*(k-f)

def iou(a,b):  # a,b = [x,y,w,h]
    ax1,ay1,ax2,ay2=a[0],a[1],a[0]+a[2],a[1]+a[3]
    bx1,by1,bx2,by2=b[0],b[1],b[0]+b[2],b[1]+b[3]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=a[2]*a[3]+b[2]*b[3]-inter
    return inter/ua if ua>0 else 0.0

def load(name):
    d=json.load(open(ANN%name))
    imgs={im['id']:im for im in d['images']}
    by_img={}
    for a in d['annotations']:
        by_img.setdefault(a['image_id'],[]).append(a)
    return d,imgs,by_img

print('='*78)
print('SECTION 1 — SHIP AREA DISTRIBUTION (mask `area` field == COCO eval area)')
print('   also bbox area (w*h) for reference; native pixels')
print('='*78)
for name in SETS:
    d,imgs,by_img=load(name)
    mask_area=[a['area'] for a in d['annotations']]
    bbox_area=[a['bbox'][2]*a['bbox'][3] for a in d['annotations']]
    n=len(mask_area)
    s=sum(1 for x in mask_area if x<SMALL); m=sum(1 for x in mask_area if SMALL<=x<MED); l=n-s-m
    # bbox-edge geometry (native)
    edges=[max(a['bbox'][2],a['bbox'][3]) for a in d['annotations']]      # long edge
    shorts=[min(a['bbox'][2],a['bbox'][3]) for a in d['annotations']]     # short edge
    print('\n[%s] n_ships=%d  n_imgs=%d' % (name,n,len(d['images'])))
    print('  mask area px^2:  med=%.0f  p10=%.0f p25=%.0f p75=%.0f p90=%.0f  min=%.0f max=%.0f'
          %(pct(mask_area,50),pct(mask_area,10),pct(mask_area,25),pct(mask_area,75),pct(mask_area,90),min(mask_area),max(mask_area)))
    print('  bbox area px^2:  med=%.0f  p25=%.0f p75=%.0f'%(pct(bbox_area,50),pct(bbox_area,25),pct(bbox_area,75)))
    print('  COCO(by mask area):  small<32^2 = %d (%.1f%%)   medium = %d (%.1f%%)   large = %d (%.1f%%)'
          %(s,100*s/n, m,100*m/n, l,100*l/n))
    sb=sum(1 for x in bbox_area if x<SMALL); mb=sum(1 for x in bbox_area if SMALL<=x<MED)
    print('  COCO(by bbox area):  small = %d (%.1f%%)   medium = %d (%.1f%%)   large = %d (%.1f%%)'
          %(sb,100*sb/n, mb,100*mb/n, n-sb-mb,100*(n-sb-mb)/n))
    print('  bbox long edge px:  med=%.1f p25=%.1f p75=%.1f   short edge: med=%.1f p25=%.1f'
          %(pct(edges,50),pct(edges,25),pct(edges,75),pct(shorts,50),pct(shorts,25)))

print('\n'+'='*78)
print('SECTION 2 — DENSITY / CLUSTERING (per image; nearest-neighbour over ships in same image)')
print('='*78)
for name in SETS:
    d,imgs,by_img=load(name)
    perimg=[len(v) for v in by_img.values()]
    # add zero-ship images? all imgs have anns here; report over images that have anns
    nn_centerdist=[]   # nearest-neighbour center distance (native px), normalized by sqrt(own bbox area)
    nn_iou=[]          # nearest-neighbour bbox IoU
    nn_gap=[]          # nearest-neighbour center dist in raw native px
    multi=0
    for iid,anns in by_img.items():
        if len(anns)<2: continue
        multi+=1
        cents=[(a['bbox'][0]+a['bbox'][2]/2, a['bbox'][1]+a['bbox'][3]/2, a) for a in anns]
        for i,(cx,cy,ai) in enumerate(cents):
            best=1e18; bestiou=0.0
            for j,(ox,oy,aj) in enumerate(cents):
                if i==j: continue
                dd=math.hypot(cx-ox,cy-oy)
                if dd<best: best=dd
                ii=iou(ai['bbox'],aj['bbox'])
                if ii>bestiou: bestiou=ii
            diag=math.sqrt(ai['bbox'][2]**2+ai['bbox'][3]**2)  # own bbox diag
            nn_gap.append(best)
            nn_centerdist.append(best/diag if diag>0 else float('nan'))
            nn_iou.append(bestiou)
    dist=[1,2,3,4,5,6,7,8]
    histn={k:sum(1 for x in perimg if x==k) for k in dist}
    print('\n[%s] imgs=%d  multi-ship imgs=%d (%.0f%%)  ships/img: med=%g mean=%.2f max=%d'
          %(name,len(perimg),multi,100*multi/len(perimg),pct(perimg,50),sum(perimg)/len(perimg),max(perimg)))
    print('   ships/img hist 1..8+: '+' '.join('%d:%d'%(k,histn[k]) for k in dist)+'  >8:%d'%sum(1 for x in perimg if x>8))
    if nn_iou:
        ab=sum(1 for x in nn_iou if x>0)
        print('   nearest-neighbour bbox IoU: med=%.3f p75=%.3f p90=%.3f max=%.3f  | %% with NN IoU>0 (boxes overlap)=%.0f%%'
              %(pct(nn_iou,50),pct(nn_iou,75),pct(nn_iou,90),max(nn_iou),100*ab/len(nn_iou)))
        print('   NN center dist / own bbox-diag: med=%.2f p25=%.2f p10=%.2f  (<1.0 ~ touching/overlapping)'
              %(pct(nn_centerdist,50),pct(nn_centerdist,25),pct(nn_centerdist,10)))
        print('   NN center dist raw native px: med=%.1f p25=%.1f p10=%.1f'%(pct(nn_gap,50),pct(nn_gap,25),pct(nn_gap,10)))

print('\n'+'='*78)
print('SECTION 3 — RESOLUTION: small ships on 512 input & 1/4-stride (128) feature')
print('   test pipeline: Resize(scale=(512,512), keep_ratio=True) -> s=min(512/W,512/H)')
print('='*78)
for name in SETS:
    d,imgs,by_img=load(name)
    feat_long=[]; feat_short=[]; feat_area=[]; small_feat_long=[]
    in_long=[]
    smallcnt=0
    for a in d['annotations']:
        im=imgs[a['image_id']]; W,H=im['width'],im['height']
        s=min(INPUT/W, INPUT/H)            # keep_ratio scale
        lo=max(a['bbox'][2],a['bbox'][3]); sh=min(a['bbox'][2],a['bbox'][3])
        in_long.append(lo*s)
        fl=lo*s/STRIDE; fs=sh*s/STRIDE     # feature px (1/4 stride)
        feat_long.append(fl); feat_short.append(fs)
        feat_area.append((a['area'])*(s**2)/(STRIDE**2))   # mask area scaled to feature grid
        if a['area']<SMALL:
            smallcnt+=1; small_feat_long.append(fl)
    print('\n[%s]  scale s: med=%.3f (img med size -> 512)'%(name,pct([min(INPUT/imgs[a["image_id"]]["width"],INPUT/imgs[a["image_id"]]["height"]) for a in d['annotations']],50)))
    print('  ON 512 INPUT  long-edge px: med=%.1f p25=%.1f p10=%.1f'%(pct(in_long,50),pct(in_long,25),pct(in_long,10)))
    print('  ON 1/4 FEATURE (128x128) long-edge px: med=%.1f p25=%.1f p10=%.1f  short-edge: med=%.1f p25=%.1f'
          %(pct(feat_long,50),pct(feat_long,25),pct(feat_long,10),pct(feat_short,50),pct(feat_short,25)))
    print('  ON 1/4 FEATURE  mask-area px(grid cells): med=%.1f p25=%.1f p10=%.1f'%(pct(feat_area,50),pct(feat_area,25),pct(feat_area,10)))
    if small_feat_long:
        print('  COCO-small ships only (%d): feature long-edge px med=%.1f p75=%.1f max=%.1f  | %% of small w/ feat-long<4px=%.0f%%'
              %(smallcnt,pct(small_feat_long,50),pct(small_feat_long,75),max(small_feat_long),
                100*sum(1 for x in small_feat_long if x<4)/len(small_feat_long)))
print('\nDONE')
