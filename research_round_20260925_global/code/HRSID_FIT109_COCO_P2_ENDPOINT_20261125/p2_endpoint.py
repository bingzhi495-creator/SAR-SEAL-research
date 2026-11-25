"""Source-only test of generic COCO P2 and gray prototypes at fixed B5 additions."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from PIL import Image
from pycocotools import mask as codec
from scipy.ndimage import binary_dilation, binary_erosion


PROTOCOL = 'c70139a1a5ae2fc9488e5bb26e4c52312547ad89e264aed0b83b5bf290bd4813'
FIT_DONOR_SOURCE = '955850893e5ba313d870d4f05e5129bf0f1aded11bf150e5257f46e49b39f23d'
ARMS = ('native','independent','raw_posterior','static_attraction','direct_p2','matched_sham','learned_gray','learned_p2')
CONTROLS = ARMS[1:-1]
LEARNED = ('matched_sham','learned_gray','learned_p2')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pinned(file, digest):
    data = file.read_bytes()
    need(sha256(data).hexdigest() == digest, 'Exact artifact digest: '+file.name)
    return data


def module(file, digest, name):
    obj = ModuleType(name)
    obj.__file__ = str(file)
    sys.modules[name] = obj
    exec(compile(pinned(file,digest),str(file),'exec'),obj.__dict__)
    return obj


def bilinear(feature, coords):
    h,w,_ = feature.shape
    scaled = coords.astype(np.float32) / np.float32(4.)
    y,x = np.minimum(scaled, np.asarray([h-1,w-1],dtype=np.float32)).T
    y0,x0 = np.floor(y).astype(np.intp),np.floor(x).astype(np.intp)
    y1,x1 = np.minimum(y0+1,h-1),np.minimum(x0+1,w-1)
    a,b = (y-y0).astype(np.float32)[:,None],(x-x0).astype(np.float32)[:,None]
    upper = feature[y0,x0]*(1-b)+feature[y0,x1]*b
    lower = feature[y1,x0]*(1-b)+feature[y1,x1]*b
    return upper*(1-a)+lower*a


def reference_cells(stock):
    y,x = np.nonzero(stock)
    h,w = stock.shape
    y0,y1 = max(0,int(y.min())-5),min(h,int(y.max())+6)
    x0,x1 = max(0,int(x.min())-5),min(w,int(x.max())+6)
    sub = stock[y0:y1,x0:x1]
    structure = np.ones((3,3),dtype=bool)
    core=binary_erosion(sub,structure=structure,border_value=0)
    ring=(binary_dilation(sub,structure=structure,iterations=5,border_value=0)
          & ~binary_dilation(sub,structure=structure,border_value=0))
    origin=np.array([y0,x0])
    return np.argwhere(core)+origin,np.argwhere(ring)+origin


def l2normalized(vectors):
    return vectors/np.maximum(np.linalg.norm(vectors,axis=-1,keepdims=True),np.float32(1e-8))


def prototype_margin(core,ring,points):
    centers = np.stack((l2normalized(core).mean(axis=0),l2normalized(ring).mean(axis=0)))
    if np.any(np.linalg.norm(centers,axis=-1) <= 1e-8):
        return None
    means=l2normalized(centers)
    return l2normalized(points) @ (means[0]-means[1])


def normalized_cue(values):
    raw = values.astype(np.float64)
    centered=raw-raw.mean()
    rms=np.sqrt(np.square(centered).mean())
    return np.clip(centered/max(float(rms),1e-3),-3,3)


def frozen_deletes(scores, coords, independent, k):
    ordinals=np.arange(len(coords))
    add=np.lexsort((ordinals,-scores))[:k]
    return np.r_[coords[add,0],independent[k:]]


def prep_capture(args, infos):
    root=args.p2_root
    terminal=json.loads(pinned(root/'observed_capture/TERMINAL.json',args.p2_capture_terminal_sha))
    capture=json.loads(pinned(root/'capture/RESULT_SAFE.json',args.p2_capture_sha))
    verify=json.loads(pinned(root/'verify-capture/RESULT_SAFE.json',args.p2_verify_sha))
    verifier_terminal=json.loads(pinned(root/'observed_verify-capture/TERMINAL.json',args.p2_verify_terminal_sha))
    need(terminal['actual_exit']==verifier_terminal['actual_exit']==0
         and terminal['output_footer_verified'] and verifier_terminal['output_footer_verified']
         and terminal['phase']==capture['phase']=='capture'
         and verifier_terminal['phase']==verify['phase']=='verify-capture'
         and terminal['result_sha256']==args.p2_capture_sha and verifier_terminal['result_sha256']==args.p2_verify_sha
         and terminal['worker_sha256']==verifier_terminal['worker_sha256']==capture['worker_sha256']==verify['worker_sha256']
         == '534a880557b862c04d957490c2f7fd041cbc93e11a0546f77a0fcff494b08a31'
         and capture['status']=='HRSID_FIT109_COCO_V1_P2_CAPTURED_V1' and verify['status']=='HRSID_FIT109_COCO_V1_P2_VERIFIED_V1'
         and verify['capture_sha256']==args.p2_capture_sha and verify['exact_decoded_rgb_and_feature_file_replay'] is True
         and verify['manifest_sha256']==capture['manifest_sha256']==args.p2_manifest_sha
         and verify['weight_sha256']==capture['weight_sha256']=='bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0'
         and all(report['images']==report['files']==109 and report['p2_stride']==4
                 and not any(report[key] for key in ('fit_labels_opened','rf_outputs_opened','target_labels_opened',
                                                     'rpn_forwards','roi_forwards','model_forward_calls','optimizer_updates'))
                 for report in (capture,verify)) and capture['backbone_forwards']==109 and verify['backbone_forwards']==0,
         'Original feature capture and independent validation observed and complete')
    manifest=pinned(root/'capture/P2_MANIFEST_PRIVATE.jsonl',args.p2_manifest_sha)
    features={}
    for ordinal,line in enumerate(manifest.splitlines(),1):
        row=json.loads(line)
        iid=row['image_id']
        identity=infos[iid][0]
        need(ordinal==row['ordinal'] and iid not in features and row['source_prefix']==identity['group']
             and row['file_name']==identity['native_basename'] and row['encoded_sha256']==identity['encoded_sha256']
             and row['decoded_rgb_sha256']==identity['decoded_rgb_sha256'] and row['p2_stride']==4
             and row['native_hw']==row['transformed_hw']==row['padded_hw']==[800,800],
             'P2 reference uses exact decoded native RF image and grid')
        feature=row['feature']
        need(feature['shape']==[256,200,200] and np.dtype(feature['dtype'])==np.dtype('<f4'), 'Exact native P2 shape/dtype')
        features[iid]=feature
    need(features.keys()==infos.keys() and len(features)==109, 'Full verified source image roster')
    return features,capture,verify


def p2_plane(root, ref):
    path=(root/'capture'/ref['path']).resolve(strict=True)
    need(path.is_relative_to((root/'capture').resolve(strict=True)), 'Capture-local feature shard')
    raw=pinned(path,ref['sha256'])
    with BytesIO(raw) as io:
        value=np.load(io,allow_pickle=False)
        need(len(raw)==ref['bytes'] and not io.read(1) and value.shape==(256,200,200)
             and value.dtype==np.dtype('<f4') and np.isfinite(value).all(), 'Exact finite native FP32 P2 shard')
    return np.ascontiguousarray(np.moveaxis(value,0,-1))


def gray_plane(args, item):
    path=args.fit_images/item['native_basename']
    raw=pinned(path,item['encoded_sha256'])
    with Image.open(BytesIO(raw)) as image:
        need(image.format=='PNG' and image.mode=='RGB' and image.size==(800,800), 'Exact original native RGB PNG')
        rgb=np.asarray(image)
    need(len(raw)==item['encoded_bytes'] and sha256(rgb.tobytes()).hexdigest()==item['decoded_rgb_sha256'], 'Exact gray RGB lineage')
    return (rgb.astype(np.float32) @ np.asarray([.299,.587,.114],dtype=np.float32))/np.float32(255.)


def cue_packets(args, infos, originals, native_packets):
    features,capture,verify=prep_capture(args,infos)
    counters=Counter()
    spreads=defaultdict(list)
    for ordinal,(iid,cases) in enumerate(native_packets.items(),1):
        feature=p2_plane(args.p2_root,features[iid])
        gray=gray_plane(args,infos[iid][0])
        for pos,row in cases.items():
            coords=row['coords']
            k=row['k']
            endpoint=np.column_stack(np.divmod(coords[:,0],800))
            counters['supported_candidates']+=1
            counters['positive_k_candidates']+=bool(k)
            stock=codec.decode(originals[iid][pos]['segmentation']).astype(bool)
            core,ring=reference_cells(stock)
            row['p2']=np.zeros(len(coords))
            row['gray']=np.zeros(len(coords))
            if not len(core) or not len(ring):
                counters['missing_reference_region']+=1
                continue
            counters['both_reference_regions_nonempty']+=1
            sampled=bilinear(feature,np.concatenate((core,ring,endpoint)))
            f1,f2,fx=np.split(sampled,[len(core),len(core)+len(ring)])
            p2=prototype_margin(f1,f2,fx)
            gc=gray[core[:,0],core[:,1]].mean()
            gr=gray[ring[:,0],ring[:,1]].mean()
            gx=gray[endpoint[:,0],endpoint[:,1]]
            rad=np.abs(gx-gr)-np.abs(gx-gc)
            row['gray']=normalized_cue(rad)
            if p2 is None:
                counters['degenerate_semantic_prototype']+=1
            else:
                row['p2']=normalized_cue(p2)
                spreads['p2_raw_range'].append(float(np.ptp(p2)))
            spreads['gray_raw_range'].append(float(np.ptp(rad)))
            counters['p2_nonconstant']+=bool(np.any(row['p2']))
            counters['gray_nonconstant']+=bool(np.any(row['gray']))
            counters['p2_and_gray_nonconstant']+=bool(np.any(row['gray']) and np.any(row['p2']))
            if np.any(row['p2']) and np.any(row['gray']):
                order=np.arange(len(p2))
                counters['p2_gray_distinct_sort_ranks']+=not np.array_equal(np.lexsort((order,-p2)),np.lexsort((order,-rad)))
        if ordinal%20==0:
            print(json.dumps(dict(stage='verified_p2_and_gray_native_cues', images=ordinal)),flush=True)
    spread_report={key:{'p05_median_p95':np.percentile(values,[5,50,95]).tolist(),'count':len(values)}
                   for key,values in spreads.items()}
    return dict(counters),spread_report,capture,verify


def sham_cue(packet,q,pilot):
    m=np.column_stack((packet['p2'],np.zeros(len(packet['p2']))))
    return pilot.scramble(m,packet['depth'],q,packet['identity'])[:,0]


def fixed_point_truth(gt,matches,coco,packets,sparse,pilot):
    result=[]
    for ann in gt['annotations']:
        pair=matches.get(ann['id'])
        if pair is None:
            continue
        iid,pos=pair
        packet=packets[iid].get(pos)
        if packet is None or not (np.any(packet['gray']) or np.any(packet['p2'])):
            continue
        y=pilot.sample_rle(coco.annToRLE(ann),packet['coords'][:,0],sparse).astype(np.float64)
        result.append((packet,y))
    return result


def actual_changed(packets,matches,pilot):
    groups=set()
    count=Counter()
    for iid,pos in matches.values():
        row=packets[iid].get(pos)
        if row is None or not row['k']:
            continue
        k,r=row['k'],row['coords']
        q=row['q'][row['outer']]
        state=pilot.ranked_writes(q,r,k)
        count['positive_k_fixed_hulls']+=1
        count['p2_spatial_signal_fixed_hulls']+=bool(np.any(row['p2']))
        count['gray_spatial_signal_fixed_hulls']+=bool(np.any(row['gray']))
        if not np.any(row['p2']):
            continue
        direct=frozen_deletes(pilot.logits(q[:,0])+row['p2'],r,state,k)
        if not np.array_equal(np.sort(direct[:k]),np.sort(state[:k])):
            count['direct_p2_changed_fixed_hulls']+=1
            groups.add(row['group'])
    count['direct_p2_changed_prefixes']=len(groups)
    return dict(count),count['direct_p2_changed_fixed_hulls']>=20 and len(groups)>=8


def fitted(packets,pilot):
    coefficients,calibration={},{}
    for fold in range(5):
        train=[(r,y) for r,y in packets if r['outer']!=fold]
        populations=Counter()
        for row,y in train:
            populations[row['group']]+=len(y)
        scores,truth,weights=[],[],[]
        all_cues={key:[] for key in LEARNED}
        for row,y in train:
            q=row['q'][fold]
            scores.append(pilot.logits(q[:,0]))
            truth.append(y)
            weights.extend([1/(len(populations)*populations[row['group']])]*len(y))
            all_cues['matched_sham'].append(sham_cue(row,q,pilot))
            all_cues['learned_p2'].append(row['p2'])
            all_cues['learned_gray'].append(row['gray'])
        offset,y,w=np.concatenate(scores),np.concatenate(truth),np.asarray(weights)
        need(abs(w.sum()-1)<1e-12 and populations, 'Exact prefix-equal source calibration')
        c={key:np.concatenate(values) for key,values in all_cues.items()}
        coefficient={key:pilot.fit_coefficient(offset,data,y,w) for key,data in c.items()}
        coefficients[fold]=coefficient
        calibration[fold]=dict(prefixes=len(populations),matched_candidates=len(train),add_endpoints=len(y),
                               p2_sham_changed_coordinates=int(np.count_nonzero(c['learned_p2']!=c['matched_sham'])),
                               coefficient=coefficient)
    return coefficients,calibration


def arm_write(stock,packet,coefficients,static,sparse,pilot):
    p,k,fold=packet['coords'],packet['k'],packet['outer']
    if not k:
        return {}
    q=packet['q'][fold]
    old=pilot.ranked_writes(q,p,k)
    cue=packet['p2']
    gray=packet['gray']
    sham=sham_cue(packet,q,pilot)
    log=pilot.logits(q[:,0])
    beta=coefficients[fold]
    def proposal(score, coefficient):
        return frozen_deletes(log+coefficient*score,p,old,k) if np.any(score) and coefficient else old
    native=codec.decode(stock).astype(bool)
    y,x=np.divmod(p,800)
    writes=dict(independent=old, raw_posterior=frozen_deletes(packet['raw_p'][:,0],p,old,k),
                static_attraction=frozen_deletes(static._initial(native,y,x)[:,0],p,old,k),
                direct_p2=proposal(cue,1.),matched_sham=proposal(sham,beta['matched_sham']),
                learned_gray=proposal(gray,beta['learned_gray']),learned_p2=proposal(cue,beta['learned_p2']))
    result={}
    for arm,coords in writes.items():
        need(len(coords)==2*k==len(np.unique(coords)) and np.array_equal(old[k:],coords[k:])
             and np.isin(coords[:k],p[:,0]).all() and np.isin(coords[k:],p[:,1]).all()
             and not np.any(native.flat[coords[:k]]) and np.all(native.flat[coords[k:]]),
             'Every arm shares original SU=F deletes and signed write budget')
        mask=sparse.xor_rle(stock,coords)
        chosen=pilot.sample_rle(mask,coords,sparse)
        retained=int(codec.area(codec.merge([mask,stock],intersect=True)))
        need(np.all(chosen[:k]) and not np.any(chosen[k:]) and int(codec.area(mask))==packet['area']
             and packet['area']-retained==k, 'Actual exact signed edits and preserved area')
        result[arm]=mask
    return result


def predictions(originals,packets,coeff,static,sparse,pilot):
    worlds={arm:{iid:[dict(r) for r in source] for iid,source in originals.items()} for arm in ARMS}
    counts={arm:Counter() for arm in ARMS if arm!='native'}
    for iid,cases in packets.items():
        for pos,case in cases.items():
            changes=arm_write(originals[iid][pos]['segmentation'],case,coeff,static,sparse,pilot)
            for arm,mask in changes.items():
                worlds[arm][iid][pos]['segmentation']=mask
                counts[arm].update(candidates=1,added=case['k'],removed=case['k'])
    need(all(row==counts['independent'] for row in counts.values()),'Exact aggregate same-signed-count budgets')
    return worlds,{arm:dict(row) for arm,row in counts.items()}


def metric_comparison(gt,matches,coco,infos,worlds,ap,packets,sparse,pilot):
    groups=defaultdict(list)
    changed={name:Counter() for name in ARMS if name!='learned_p2'}
    hits=Counter()
    hit_changes=Counter()
    matched_add_budget=0
    for ann in gt['annotations']:
        iid=ann['image_id']
        g=infos[iid][0]['group']
        pair=matches.get(ann['id'])
        if pair is None:
            groups[g].append([0.]*len(ARMS))
            continue
        need(pair[0]==iid, 'Original fixed .50 match image')
        rles=[worlds[a][iid][pair[1]]['segmentation'] for a in ARMS]
        truth_rle=coco.annToRLE(ann)
        scores=codec.iou(rles,[truth_rle],[0]).reshape(-1)
        packet=packets[iid].get(pair[1])
        if packet is not None and packet['k']:
            k=packet['k']
            cells=packet['coords'][:,0]
            truth_at_sites=pilot.sample_rle(truth_rle,cells,sparse)
            matched_add_budget+=k
            local={}
            for i,arm in enumerate(ARMS[1:],1):
                applied=pilot.sample_rle(rles[i],cells,sparse)
                need(int(applied.sum())==k,'Actual same-count additions on each fixed hull')
                local[arm]=int((applied & truth_at_sites).sum())
                hits[arm]+=local[arm]
            delta=local['learned_p2']-local['independent']
            hit_changes['hulls_with_more_true_additions']+=delta>0
            hit_changes['hulls_with_fewer_true_additions']+=delta<0
            hit_changes['extra_true_pixels_on_improved_hulls']+=max(0,delta)
            hit_changes['lost_true_pixels_on_worsened_hulls']+=max(0,-delta)
        need(scores[0]>=.50-1e-12, 'Original fixed native COCO exact discrete match')
        groups[g].append(scores.tolist())
        for i,arm in enumerate(ARMS[:-1]):
            if rles[i]!=rles[-1]:
                changed[arm][g]+=1
    parent=np.asarray([np.mean(groups[name],axis=0) for name in sorted(groups)])
    values=np.asarray([r for group in groups.values() for r in group])
    need(len(groups)==109 and len(values)==496 and len(matches)==453, 'Complete original HRSID all-truth denominator')
    rng=np.random.Generator(np.random.PCG64(5217))
    draws=rng.integers(len(parent),size=(4096,len(parent)))
    geometry={name:dict(prefix_equal_iou_points=float(100*parent[:,i].mean()),hull_mean_iou_points=float(100*values[:,i].mean()),
                        reach_iou={f'{t:.2f}':int(np.count_nonzero(values[:,i]>=t)) for t in (.50,.75,.90)})
              for i,name in enumerate(ARMS)}
    diffs={}
    for i,arm in enumerate(ARMS[:-1]):
        d=100*(parent[:,-1]-parent[:,i])
        diffs[arm]=dict(AP=float(ap['learned_p2']['AP']-ap[arm]['AP']),IoU=float(d.mean()),
                        confidence95=np.percentile(d[draws].mean(axis=1),[2.5,97.5]).tolist(),
                        changed_hulls=sum(changed[arm].values()),changed_prefixes=len(changed[arm]))
    passes={a:diffs[a]['AP']>=.30 and diffs[a]['IoU']>=.20 and diffs[a]['confidence95'][0]>0 for a in CONTROLS}
    coverage=diffs['independent']['changed_hulls']>=20 and diffs['independent']['changed_prefixes']>=8
    description=dict(hulls_including_unmatched_and_noops=len(values),matched_positive_k_add_budget_each_arm=matched_add_budget,
                     exact_author_foreground_writes={arm:hits[arm] for arm in ARMS[1:]},
                     learned_minus_independent_net=hits['learned_p2']-hits['independent'],
                     learned_minus_independent_signed_counts=dict(hit_changes))
    return geometry,diffs,dict(all_control_thresholds=passes,coverage=coverage,advance=coverage and all(passes.values())),description


def run(args):
    started=perf_counter()
    pinned(args.protocol,PROTOCOL)
    pilot,reference=module(args.fit_donor/'donor.py',FIT_DONOR_SOURCE,'_p2_fit_donor').load_pilot(args.sibling_pilot)
    source,sparse,static,originals,infos,models,folds,pair,step,positive=pilot.reference_sources(args)
    packets,old_exposure=pilot.prepare(args,source,sparse,originals,infos,models,folds,pair,step)
    exposure,spreads,capture,verify=cue_packets(args,infos,originals,packets)
    need(exposure['supported_candidates']==old_exposure['supported'] and exposure['positive_k_candidates']==positive,
         'Same original frozen active cohort and reconstructed SF k')
    prelabel_at=perf_counter()
    print(json.dumps(dict(stage='all_frozen_features_checked', **exposure)),flush=True)
    helper=pilot.module(args.code/'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                        '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b','_p2_exact_labels')
    gt=helper.annotations(args,originals)
    coco,matches=source.fixed_coco(gt,originals)
    need(len(matches)==453 and len(gt['annotations'])==496, 'Same original HRSID FIT truth and assignment')
    prefit,proceed=actual_changed(packets,matches,pilot)
    fit_at=perf_counter()
    print(json.dumps(dict(stage='actual_direct_same_count_prefit_coverage', **prefit,proceed=proceed)),flush=True)
    safe=dict(status='HRSID_FIT109_COCO_P2_ENDPOINT_TEST_V1',phase='score',protocol_sha256=PROTOCOL,
              worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),accepted_P2_capture_sha256=args.p2_capture_sha,
              accepted_P2_verify_sha256=args.p2_verify_sha,P2_manifest_sha256=args.p2_manifest_sha,
              original_RF_capture_sha256=source.CAPTURE,original_RF_seed=5217,original_RF_source_fit_exposure=True,
              source_images=len(infos),source_candidates=sum(map(len,originals.values())),native_fixed_matches=len(matches),
              unmatched_original_hulls=len(gt['annotations'])-len(matches),source_hulls=len(gt['annotations']),
              exposure=exposure,cue_spread=spreads,prefit_coverage=prefit,passed_prefit_coverage=proceed,scope='HRSID_FIT109',
              scoring_new_detector_forwards=0,new_RF_or_P2_optimizer_updates=0,external_data_opened=False,
              exact_original_independent_deletes=True)
    timings=dict(label_free_capture_checks_and_cues=prelabel_at-started,fixed_matching_and_prefit_exposure=fit_at-prelabel_at)
    if not proceed:
        return dict(safe,did_coco=False,decision=dict(advance=False,reason='below_prefit_coverage'),
                    coefficient_fits=0,timing_seconds=dict(timings,total=perf_counter()-started))
    rows=fixed_point_truth(gt,matches,coco,packets,sparse,pilot)
    coeff,calibration=fitted(rows,pilot)
    after_fit=perf_counter()
    worlds,budget=predictions(originals,packets,coeff,static,sparse,pilot)
    after_write=perf_counter()
    ap={arm:pilot.full_metrics(gt, worlds[arm],originals) for arm in ARMS}
    need(all(ap[a]==reference['full_coco_points'][a] for a in ('native','independent')),'Original AP and recall parity')
    geometry,comparisons,decision,hit_counts=metric_comparison(gt,matches,coco,infos,worlds,ap,packets,sparse,pilot)
    need(all(abs(geometry[a]['prefix_equal_iou_points']-reference['fixed_all_hull_geometry'][a]['parent_discrete_iou_points'])<1e-12
             for a in ('native','independent')),'Original discrete geometry parity')
    need(not step.torch.cuda.is_initialized(), 'CPU-only endpoint scoring')
    return dict(safe,did_coco=True,coefficient_fits=15,calibration_by_fold=calibration,decision=decision,
                full_coco_points=ap,fixed_all_hull_discrete_geometry=geometry,learned_P2_minus_controls=comparisons,
                exact_signed_budget=budget,descriptive_fixed_hull_addition_hits=hit_counts,timing_seconds=dict(timings,coefficient_fit=after_fit-fit_at,
                    verified_mask_writes=after_write-after_fit,scoring=perf_counter()-after_write,total=perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('code','protocol','capture','acceptance','source-cohort','basis','heads','fit-gt','source-run',
                 'sibling-pilot','fit-donor','p2-root','fit-images','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('p2-capture-sha','p2-capture-terminal-sha','p2-verify-sha','p2-verify-terminal-sha','p2-manifest-sha'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args()
    need(not args.output.exists() and args.output.parent.is_dir(),'Fresh scoring output required')
    result=run(args)
    raw=(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    args.output.mkdir()
    with (args.output/'RESULT_SAFE.json').open('xb') as out:
        out.write(raw)
    print(json.dumps(dict(status=result['status'],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=='__main__':
    main()
