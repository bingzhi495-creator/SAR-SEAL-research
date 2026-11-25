"""Test a frozen FIT donor distance vote on the original HRSID DEV705 outputs."""

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
from pycocotools import mask as codec
import torch


PROTOCOL = '3301460ea6a21b9b6b7ec432e01de4a1d65c20091f02ecb5f28a3b4f222ece0b'
RF_WORKERS = {
    'capture':'b9eb3d4169653a7e7be634022f89dba81f17375a4ed511b45238a583647aa95b',
    'replay':'e7d4d06116539e7f2f62486557a22364bc359c3f58e3b9fe5b1b853abe256257',
    'score':'f089cc3c692edcfe3aab836dbacb28e2081c18cc5a8f1b3f29859e051424a1a3',
}
RF_TERMINALS = {
    'capture':'8351b20a3f838f5f7a31453f974a4fe2259f7b246e1a517d4cf0755566adbaf4',
    'replay':'f172ecbfc3c606b874b6c7839b35f0c0c3c5a8ae69ae20e69a2e4c575d60df9d',
    'score':'1cd9a3aeb2c83f207788e6eb9e1c4d1b3b7c09b5214b8a292565e9b3a1b93675',
}
DF_READBACK = '277965bef4230ec7fadac30e103095af3ce68f99d1b69f74a8e348c391cef035'
DF_CLOSE = '63fcd195ec3cddd59b664ac59391e561a7b248279758d190795ba439e5dca3dc'
FIT_DONOR = '955850893e5ba313d870d4f05e5129bf0f1aded11bf150e5257f46e49b39f23d'
FIT_PILOT_RESULT = 'ffd9c3d31dbbcc933b88454d1e27e8ae9b1f7afa1df115f9ee212ebaa4afdb49'
FIT_PILOT_TERMINAL = '1fe680da0e8c79e58316918c49c75f28cbdc5bf336a39cd028e724687410f871'
ARMS = ('native','distance','independent','raw_posterior','static_attraction','donor_vote')
ARCHIVED_ARMS = {'native':'N','distance':'SF','independent':'SU=F'}


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, digest):
    data = path.read_bytes()
    check(sha256(data).hexdigest() == digest, 'Exact closed artifact ' + path.name)
    return data


def load(path, digest, name):
    m = ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(pin(path,digest),str(path),'exec'),m.__dict__)
    return m


def rf_lineage(root):
    refs = {}
    for phase in RF_WORKERS:
        pin(root/f'portable_dev705_{phase}.py', RF_WORKERS[phase])
        t = json.loads(pin(root/f'observed_{phase}/TERMINAL.json', RF_TERMINALS[phase]))
        check(t['actual_exit'] == 0 and t['output_footer_verified'] and t['phase'] == phase
              and t['worker_sha256'] == RF_WORKERS[phase], 'Accepted original RF ' + phase)
        refs[phase] = json.loads(pin(root/phase/'RESULT_SAFE.json', t['result_sha256']))
    check(refs['score']['qualification']['passed'] is True, 'Original closed B5 score qualification')
    return refs


def df_lineage(root, manifest_sha, expected_checkpoint):
    readback = json.loads(pin(root/'INDEPENDENT_RESULT_SAFE.json',DF_READBACK))
    terminal = json.loads(pin(root/'EXPORT_OS_CLOSED_SAFE.json',DF_CLOSE))
    export = root/'CONTROL/EXPORT'
    result = json.loads(pin(export/'RESULT_SAFE.json',readback['export_result_sha256']))
    check(readback['status'] == 'HRSID_DFINE_E24_DEV705_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1'
          and readback['export_terminal_sha256'] == DF_CLOSE and readback['job_drained'] and terminal['job_drained']
          and terminal['original_os_exit'] == readback['original_os_exit'] == 0
          and terminal['export_result_sha256'] == readback['export_result_sha256']
          and result['manifest_sha256'] == readback['manifest_sha256'] == manifest_sha
          and result['ema_sha256'] == readback['ema_sha256'] == expected_checkpoint
          and readback['historical_stream_byte_parity'] and readback['exact_author_binary_replay'] == 70500
          and readback['images'] == result['images'] == 705 and readback['DEV'] and not readback['GT_opened'],
          'Same decoded image-only manifest and closed D-FINE binary archive')
    files = (('DEV_MASK_PREDICTIONS_PRIVATE.jsonl','predictions_sha256'),
             ('DEV_CANDIDATE_AUDIT_PRIVATE.jsonl','candidate_audit_sha256'))
    contents = [pin(export/name,readback[key]).splitlines() for name,key in files]
    check(all(len(rows)==705 for rows in contents), 'Both original D-FINE streams are complete')
    return list(zip(*contents,strict=True)), readback


def admitted_donor(rows, detail, image_id):
    p, a = json.loads(rows), json.loads(detail)
    check(p['image_id'] == a['image_id'] == image_id and len(p['coco']) == len(a['candidates']) >= 100,
          'Exactly aligned D-FINE candidates and original image')
    donors = []
    for j, (row, candidate) in enumerate(zip(p['coco'][:100],a['candidates'][:100],strict=True),1):
        rle = row['segmentation']
        check(row['image_id']==image_id and row['category_id']==1 and rle['size']==[800,800]
              and candidate['native_rank']==j and candidate['score']==row['score'], 'Frozen D-FINE native top100 identity')
        if codec.area(rle) > 0:
            donors.append(row)
    return donors


def blended(q, cue, rf, df):
    return (rf*q+df*((cue+1)/2))/(rf+df) if rf+df else q


def edit_meta(flips,k):
    return dict(flipped=np.asarray(flips),edits=2*k,adds=k,drops=k,area_delta=0)


def verified_write(stock, area, p, k, cells, sparse, old_pilot):
    check(len(cells) == 2*k == len(np.unique(cells)) and np.isin(cells[:k],p[:,0]).all()
          and np.isin(cells[k:],p[:,1]).all(), 'Original signed native coordinate write support')
    before = old_pilot.sample_rle(stock,cells,sparse)
    after_rle = sparse.xor_rle(stock,cells)
    after = old_pilot.sample_rle(after_rle,cells,sparse)
    retained = int(codec.area(codec.merge([stock,after_rle],intersect=True)))
    check(not any(before[:k]) and all(before[k:]) and all(after[:k]) and not any(after[k:])
          and int(codec.area(after_rle))==area and area-retained==k, 'Exact edited signed bitmap counts and stock area')
    return after_rle


def build_arm_rows(args,replay,ops,donor_module,old_pilot,sparse,static,df_lines):
    _,_,repair = replay.capture_inputs(args)
    replay.initialize(args,repair)
    c = replay.CTX
    p,r,cap = c.policy,c.reader,replay.cap
    h,center,scaler = c.heads[5217]
    images = ops.native.dev_file('DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json')
    worlds = {arm:[] for arm in ARMS}
    prepared = []
    counts = Counter()
    budgets = {arm:Counter() for arm in ARMS if arm != 'native'}
    old_hash = sha256()
    with (args.capture/cap.QUERIES).open('rb') as queries,(args.capture/cap.STOCK).open('rb') as stocks,(
            args.replay/'SPARSE_EDITS_PRIVATE.jsonl').open('rb') as edits:
        for ordinal,(item,q_raw,stok,edit_raw,df_raw) in enumerate(zip(images,queries,stocks,edits,df_lines,strict=True),1):
            old_hash.update(edit_raw)
            block,historic = json.loads(q_raw),json.loads(edit_raw)
            iid = item['image_id']
            check(iid == block['image_id'] == historic['image_id'] and historic['ordinal']==ordinal
                  and item['decoded_rgb_sha256']==block['image_content_sha256']
                  and item['encoded_sha256']==block['encoded_sha256']
                  and historic['group']==block['source_prefix'], 'Exact decoded RF source and edit identity')
            pool = p.spot.native_candidate_rows(block)
            check(r.legacy_stock_bytes(iid,pool) == stok, 'All original RF candidates, scores, masks and rank order')
            boxes = p.spot.author_boxes(block['raw_boxes_cxcywh'],c.torch)
            for cand in pool:
                cand['area'] = int(codec.area(cand['rle']))
                cand['box_xyxy'] = boxes[cand['query_id']].tolist()
            check(historic['mapped_identity_sha256']==r.mapped_identity_sha256(pool), 'Exact RF candidate roster')
            eligible = [cand for cand in pool if 32<=cand['area']<1024]
            dr = admitted_donor(*df_raw, iid)
            drles = [row['segmentation'] for row in dr]
            overlaps = codec.iou([cand['rle'] for cand in eligible],drles,[0]*len(drles)) if dr and eligible else None
            mutual = codec.iou(drles,drles,[0]*len(drles)) if dr else None
            counts['D_FINE_nonempty_top100'] += len(dr)
            ref = block['logit_capture']
            check(ref['relative_path'] == f'RAW_LOGITS/{ordinal:03d}.npy', 'Exact RF saved logit file order')
            path = args.capture/ref['relative_path']
            raw = r.pin(path,ref['sha256'],9*r.MIB)
            archive = BytesIO(raw)
            planes = np.load(archive,allow_pickle=False)
            check(not archive.read(1) and len(raw)==ref['bytes'] and planes.dtype==np.dtype('<f4')
                  and list(planes.shape)==ref['shape']==[len(eligible),144,144] and np.isfinite(planes).all(),
                  'Exact saved RF logit planes')
            src = dict(relative_path=ref['relative_path'],path=str(path),bytes=len(raw),sha256=ref['sha256'])
            rgb,hw = c.png.read_png(c.archive,item)
            check(hw==(800,800), 'Same native canvas')
            image_features = p.pair.image_features(rgb)
            distance_cache, new_masks, new_edits, points = {}, {}, {}, {}
            for index,(cand,witness,plane,pt) in enumerate(zip(eligible,ref['candidates'],planes,historic['points'],strict=True)):
                rank, query, score, area = (cand[key] for key in ('rank','query_id','score','area'))
                check(witness==dict(logit_row=index,query_id=query,native_rank=rank,score=score,predicted_area=area)
                      and (pt['rank'],pt['query_id'])==(rank,query), 'Exact candidate, plane and point provenance')
                coords = np.asarray(pt['coordinates'],dtype=np.int32).reshape(-1,2)
                points[rank] = coords
                independent = ops.score.get_edit(historic,5217,rank,'SU=F')
                distance = ops.score.get_edit(historic,5217,rank,'SF')
                k = independent['adds']
                check(independent['drops']==distance['adds']==distance['drops']==k and independent['edits']==distance['edits']==2*k
                      and 0<=k<=min(8,area//10,len(coords)), 'Archived exact paired SF and independent signed quotas')
                counts.update(eligible=1,supported=bool(len(coords)),positive_k=bool(k))
                repair_here = repair['ordinal']==ordinal and repair['query_id']==query
                if not k and not repair_here:
                    continue
                stock = codec.decode(cand['rle']).astype(bool)
                idn = dict(ordinal=ordinal,image_id=iid,query_id=query,native_rank=rank,logit_row=index)
                posterior = r.probability_with_repair(np,c.torch,p.features,p.posterior,plane,stock,cand['rle'],idn,src,repair)
                if not k:
                    continue
                record = p.features.extract_candidate(np,p.pair,p.rank,stock,boxes[query],score,posterior,image_features)
                check(record is not None and np.array_equal(record['coordinates'],coords), 'Exact original B5 endpoint reconstruction')
                q = p.fit_step.predict(h['S'],record['features'],center,scaler)
                sf = p.pair.choose_pairs(q[:,0]-q[:,1],min(8,area//10,len(coords)))
                stock_ind = old_pilot.ranked_writes(q,coords,k)
                check(len(sf)==k and coords[sf].reshape(-1).tolist()==distance['flipped']
                      and np.column_stack((stock_ind[:k],stock_ind[k:])).reshape(-1).tolist()==independent['flipped'],
                      'Exact original all-FIT q reproduces archived SF and SU=F actions')
                donor_idx, ambiguous = donor_module.winner(overlaps[index],mutual) if overlaps is not None else (None,False)
                counts['donor_qualified']+=donor_idx is not None
                counts['donor_ambiguous_fallback']+=bool(ambiguous)
                voted = stock_ind
                if donor_idx is not None and not ambiguous:
                    if donor_idx not in distance_cache:
                        distance_cache[donor_idx] = donor_module.distance(drles[donor_idx])
                    cue = donor_module.gather(*distance_cache[donor_idx],coords,800)
                    voted = old_pilot.ranked_writes(blended(q,cue,score,dr[donor_idx]['score']),coords,k)
                    counts['donor_admitted_positive_k'] += 1
                y,x = np.divmod(coords,800)
                writes = dict(distance=np.r_[coords[sf,0],coords[sf,1]],independent=stock_ind,donor_vote=voted,
                              raw_posterior=old_pilot.ranked_writes(record['features'][:,:,0],coords,k),
                              static_attraction=old_pilot.ranked_writes(static._initial(stock,y,x),coords,k))
                new_masks[rank],new_edits[rank] = {},{}
                for arm,pixels in writes.items():
                    new_masks[rank][arm] = verified_write(cand['rle'],area,coords,k,pixels,sparse,old_pilot)
                    new_edits[rank][arm] = edit_meta(pixels,k)
                    budgets[arm].update(edited_candidates=1,added=k,removed=k)
            for cand in pool:
                stock_row = dict(image_id=iid,category_id=1,score=cand['score'],segmentation=cand['rle'])
                worlds['native'].append(stock_row)
                masks = new_masks.get(cand['rank'],{})
                for arm in ARMS[1:]:
                    worlds[arm].append(dict(stock_row,segmentation=masks[arm]) if arm in masks else stock_row)
            prepared.append(dict(iid=iid,pool=pool,points=points,edits=new_edits,group=historic['group']))
            counts.update(images=1,mapped=len(pool))
            if ordinal%70==0:
                print(json.dumps(dict(stage='reviewed_native_and_donor_edits',images=ordinal)),flush=True)
    c.archive.close()
    check(repair['_consumed']==1 and old_hash.hexdigest()==replay.EDITS_SHA and counts['images']==705
          and counts['mapped']==67386 and counts['eligible']==42325 and counts['supported']==40897
          and all(row==budgets['independent'] for row in budgets.values()), 'Whole-cohort frozen count, repaired parity and no omissions')
    return worlds,prepared,dict(counts),{arm:dict(n) for arm,n in budgets.items()}


def read_geometry(gt, prepared, ops, approved):
    annotations = defaultdict(list)
    for ann in gt['annotations']:
        annotations[ann['image_id']].append(ann)
    groups = defaultdict(list)
    changed = Counter()
    actual_changed_groups = set()
    role_counts = Counter()
    zero = edit_meta([],0)
    for image, packet in zip(gt['images'],prepared,strict=True):
        iid = packet['iid']
        check(iid==image['id'] and (image['height'],image['width'])==(800,800), 'Original native GT axes')
        records = [dict(image_id=iid,rank=c['rank'],query_id=c['query_id'],group=packet['group'],
                        coordinates=packet['points'][c['rank']]) for c in packet['pool'] if c['rank'] in packet['points']]
        attached, census = ops.fit.attach_truth(image,annotations[iid],packet['pool'],records,ops.helpers)
        role_counts.update(census)
        for record in attached:
            if record['role']!='valid':
                continue
            alt = packet['edits'].get(record['rank'],{})
            values = [ops.geometry.candidate_geometry(record,alt.get(arm,zero))['iou'] for arm in ARMS]
            groups[packet['group']].append(values)
            if 'donor_vote' in alt and not np.array_equal(np.sort(alt['donor_vote']['flipped']),np.sort(alt['independent']['flipped'])):
                changed['fixed_hulls']+=1
                actual_changed_groups.add(packet['group'])
    group_values = np.asarray([np.mean(groups[g],axis=0) for g in sorted(groups)])
    hull_values = np.asarray([row for values in groups.values() for row in values])
    check(dict(role_counts)==approved['stock_matching_counts'] and len(hull_values)==role_counts['valid'], 'Same original fixed-matching eligibility and denominator')
    archive = approved['geometry_summary']['per_seed']['5217']['arms']
    group_means = 100*group_values.mean(axis=0)
    for arm,name in ARCHIVED_ARMS.items():
        check(abs(group_means[ARMS.index(arm)]-archive[name]['group_equal_iou_points'])<1e-9, 'Original fixed geometry parity for '+arm)
    primary = 100*(group_values[:,-1]-group_values[:,ARMS.index('independent')])
    rng = np.random.Generator(np.random.PCG64(5217))
    draws = rng.integers(len(groups),size=(4096,len(groups)))
    lo,hi = np.percentile(primary[draws].mean(axis=1),[2.5,97.5])
    metrics = {arm:dict(prefix_mean_iou_points=float(group_means[i]),hull_mean_iou_points=float(100*hull_values[:,i].mean()),
                        strict75=int((hull_values[:,i]>.75).sum()),strict90=int((hull_values[:,i]>.90).sum()))
               for i,arm in enumerate(ARMS)}
    return dict(metrics=metrics,original_geometry_parity=True,matched_hulls=len(hull_values),prefix_proxies=len(groups),
                census=dict(role_counts),primary_delta_points=float(primary.mean()),primary_95_points=[float(lo),float(hi)],
                primary_changed_fixed_hulls=changed['fixed_hulls'],primary_changed_prefixes=len(actual_changed_groups))


def run(args):
    start = perf_counter()
    pin(args.protocol,PROTOCOL)
    report = rf_lineage(args.anchor)
    assert_report=report['score']
    fit_pilot = json.loads(pin(args.fit_donor/'result/RESULT_SAFE.json',FIT_PILOT_RESULT))
    fit_terminal = json.loads(pin(args.fit_donor/'observed_pilot_retry/TERMINAL.json',FIT_PILOT_TERMINAL))
    check(fit_terminal['actual_exit']==0 and fit_terminal['output_footer_verified'] and fit_terminal['result_sha256']==FIT_PILOT_RESULT
          and fit_terminal['worker_sha256']==fit_pilot['worker_sha256']==FIT_DONOR, 'Original closed FIT deterministic donor formulation')
    donor_module = load(args.fit_donor/'donor.py',FIT_DONOR,'_dev_same_fit_distance')
    old_pilot = load(args.sibling_pilot/'pilot.py',donor_module.PILOT_SOURCE,'_dev_same_rank_math')
    portable = load(args.anchor/'portable_dev705_score.py',RF_WORKERS['score'],'_dev_official_reference')
    replay = portable.replay
    args.capture,args.intake,args.replay = (args.anchor/name for name in ('capture','intake','replay'))
    ops = portable.operators(args)
    lineage = portable.admission(args,ops)
    df_lines,df_closed = df_lineage(args.donor_run,replay.cap.INPUT_MANIFEST,donor_module.DONOR_CHECKPOINT)
    sparse = load(args.code/'RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py',
                  'cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5','_dev_exact_sparse')
    static = load(args.code/'FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py',
                  '8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6','_dev_exact_static')
    worlds,prepared,census,budgets=build_arm_rows(args,replay,ops,donor_module,old_pilot,sparse,static,df_lines)
    prepared_at=perf_counter()
    print(json.dumps(dict(stage='verified_exact_label_free_edits', **census)),flush=True)
    gt=ops.native.dev_file('DEV_ORIGINAL_GT_PRIVATE.json')
    findings=ops.native.dev_file('DEV_EMPTY_RASTERS_PRIVATE.json')
    empties={r['annotation_id'] for r in findings if r['kind']=='EMPTY_COCO_RASTER'}
    check(len(empties)==len(findings)==4 and len(gt['annotations'])==1782 and len(gt['images'])==705
          and [r['id'] for r in gt['images']]==[r['iid'] for r in prepared], 'Original official DEV population')
    metrics={}
    for arm in ARMS:
        metrics[arm]=ops.score.score_official(ops.native,gt,worlds[arm],empties)
        if arm in ARCHIVED_ARMS:
            check(metrics[arm]==assert_report['coco']['5217'][ARCHIVED_ARMS[arm]],'Original COCO AP and sensitivity parity: '+arm)
        print(json.dumps(dict(stage='scored_original_coco_roster',arm=arm)),flush=True)
    scored_at=perf_counter()
    geometry=read_geometry(gt,prepared,ops,assert_report)
    delta_ap=100*(metrics['donor_vote']['primary']['AP']-metrics['independent']['primary']['AP'])
    fits=(delta_ap>=.30 and geometry['primary_delta_points']>=.20 and geometry['primary_95_points'][0]>0
          and geometry['primary_changed_fixed_hulls']>=20 and geometry['primary_changed_prefixes']>=8)
    return dict(status='HRSID_DEV705_FROZEN_DONOR_VOTE_COMPLETE_V1',phase='score',protocol_sha256=PROTOCOL,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),FIT_donor_pilot_result_sha256=FIT_PILOT_RESULT,
                original_B5_score_terminal_sha256=RF_TERMINALS['score'],original_B5_lineage=lineage,
                D_FINE_dev_readback_sha256=DF_READBACK,D_FINE_dev_prediction_sha256=df_closed['predictions_sha256'],
                original_candidates=census,exact_signed_budgets=budgets,full_coco_fraction=metrics,continuous_fixed_geometry=geometry,
                primary_AP_delta_points=delta_ap,advances=fits,original_coco_parity=True,previously_examined_DEV=True,
                optimizer_updates=0,new_point_calibration_fits=0,new_detector_forwards=0,external_data_opened=False,
                times=dict(label_free_validation=prepared_at-start,official_COCO=scored_at-prepared_at,
                           fixed_geometry=perf_counter()-scored_at,total=perf_counter()-start))


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('anchor','archive','code','heads','fit-donor','sibling-pilot','donor-run','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for key,val in vars(args).items():
        setattr(args,key,val.resolve(strict=key!='output'))
    check(not args.output.exists() and args.output.parent.is_dir(),'Fresh observer destination')
    result=run(args)
    data=(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    args.output.mkdir()
    with (args.output/'RESULT_SAFE.json').open('xb') as file:
        file.write(data)
    print(json.dumps(dict(status=result['status'],result_sha256=sha256(data).hexdigest())),flush=True)


if __name__=='__main__':
    main()
