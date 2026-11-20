"""Complete native-order RF-DETR scoring from sealed sparse point edits.

Explicit in-memory inputs only: no loaders, models, file reads or execution entry.
Caller closes source/input/terminal identities before supplying actual GT.
The original native scorer performs all official COCO mathematics.
"""
from hashlib import sha256
from collections import Counter, defaultdict
from math import isfinite
import json

ARMS=('N','SF','SU=F','SU','LF=S','LU=S','LW=S','NP=S')
SEEDS=(5217,5218,5219)
CENTER_CHECKPOINT_SHA256='011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11'
STOCK_PREDICTIONS_SHA256='da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e'


def need(ok,message):
    if not ok:
        raise ValueError(message)


def mapped_identity_sha256(pool):
    identity=[{key:row[key] for key in ('rank','query_id','score','area')} for row in pool]
    return sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def zero_edit(arm):
    return dict(flipped=(),edits=0,adds=0,drops=0,area_delta=0,actions=0,
                action_kind='flip' if arm=='LW=S' else 'swap',
                nonpositive_actions=0,nonpositive_flips=0,predicted_gain_sum=0.)


ZERO={arm:zero_edit(arm) for arm in ARMS}


def get_edit(sparse,seed,rank,arm):
    return sparse['seeds'][str(seed)].get(str(rank),{}).get(arm,ZERO[arm])


def prepare_reconstruction(np,codec,capture_rows,sparse_rows):
    """Validate once; retain only original records and O(points) metadata."""
    need(len(capture_rows)==len(sparse_rows),'Complete equal-length captured and sparse image streams')
    prepared=[]
    seen=set()
    for ordinal,(capture,sparse) in enumerate(zip(capture_rows,sparse_rows),1):
        native,pool=capture['native_block'],capture['pool']
        image_id=native['image_id']
        need(type(image_id) is int and image_id not in seen,'Distinct original image identity')
        seen.add(image_id)
        ranks=native['native_ranks']
        need(native['raw_query_count']==164 and native['native_count']==100
             and native['unmapped_slot1_count']+len(pool)==100
             and len(native['coco'])==len(pool)==len(ranks) and ranks==sorted(set(ranks))
             and all(type(r) is int and 1<=r<=100 for r in ranks), 'Native joint-top100 mapped rank gaps')
        need(sparse['ordinal']==ordinal and sparse['image_id']==image_id
             and type(sparse['group']) is str and bool(sparse['group'])
             and sparse['group']==capture['source_prefix']
             and len(sparse['original_hw'])==2 and all(type(v) is int and v>0 for v in sparse['original_hw'])
             and sparse['mapped_identity_sha256']==mapped_identity_sha256(pool), 'Sealed image/candidate identity')
        hw=sparse['original_hw']
        by_rank={}
        previous=1.
        for rank,record,source in zip(ranks,native['coco'],pool):
            need(set(record)=={'image_id','category_id','score','segmentation'}
                 and record['image_id']==image_id and record['category_id']==1
                 and source['rank']==rank and type(source['query_id']) is int and 0<=source['query_id']<164
                 and source['score']==record['score'] and isfinite(record['score']) and 0<=record['score']<=previous
                 and source['rle']==record['segmentation'] and source['rle']['size']==hw
                 and type(source['area']) is int and source['area']==int(codec.area(source['rle'])),
                 'Original query/score/area/RLE and stable mapped order')
            previous=record['score']
            by_rank[rank]=source
        need(len({row['query_id'] for row in pool})==len(pool),'Unique original mapped query IDs')
        eligible={rank for rank,row in by_rank.items() if 32<=row['area']<1024}
        point_rows=sparse['points']
        need(len(point_rows)==len({p['rank'] for p in point_rows})
             and {p['rank'] for p in point_rows}==eligible, 'All eligible candidates, including unsupported no-ops')
        points={}
        for point in point_rows:
            rank=point['rank']
            need(point['query_id']==by_rank[rank]['query_id'],'Point ledger retains native query ID')
            coords=np.asarray(point['coordinates'])
            if coords.size==0:
                coords=np.empty((0,2),dtype=np.int32)
            need(coords.ndim==2 and coords.shape[1]==2 and len(coords)<=32 and coords.dtype.kind in 'iu'
                 and np.all((coords>=0)&(coords<hw[0]*hw[1])) and np.unique(coords).size==coords.size,
                 'Bounded disjoint sealed native points')
            if coords.size:
                stock=codec.decode(by_rank[rank]['rle']).astype(bool)
                need(not stock.reshape(-1)[coords[:,0]].any() and stock.reshape(-1)[coords[:,1]].all(),
                     'Original add/drop point orientation')
                del stock
            points[rank]=coords
        need(set(sparse['seeds'])=={str(s) for s in SEEDS}, 'All three sparse seed keys')
        for seed in SEEDS:
            supplied=sparse['seeds'][str(seed)]
            need(set(supplied)<={str(rank) for rank in eligible},'No edits on unadmitted rank or fabricated candidate')
            for rank in eligible:
                arms=supplied.get(str(rank),{})
                need(set(arms)<=set(ARMS),'Known sparse arm names only')
                coords=points[rank]
                cap=min(8,by_rank[rank]['area']//10,len(coords))
                for arm in ARMS:
                    edit=get_edit(sparse,seed,rank,arm)
                    need(set(edit)==set(ZERO[arm]),'Exact sparse edit fields')
                    indices=np.asarray(edit['flipped'])
                    if indices.size==0:
                        indices=np.empty(0,dtype=np.int32)
                    need(indices.ndim==1 and indices.dtype.kind in 'iu' and np.unique(indices).size==indices.size
                         and np.isin(indices,coords.reshape(-1)).all(),'Only unique sealed point flips')
                    add=int(np.isin(indices,coords[:,0]).sum())
                    drop=int(np.isin(indices,coords[:,1]).sum())
                    need(all(type(edit[k]) is int for k in ('edits','adds','drops','area_delta','actions',
                                                           'nonpositive_actions','nonpositive_flips'))
                         and edit['edits']==indices.size==add+drop<=2*cap and edit['adds']==add and edit['drops']==drop
                         and edit['area_delta']==add-drop and (arm=='LW=S' or add==drop)
                         and edit['action_kind']==('flip' if arm=='LW=S' else 'swap')
                         and edit['actions']==(indices.size if arm=='LW=S' else add)
                         and 0<=edit['nonpositive_actions']<=edit['actions']
                         and 0<=edit['nonpositive_flips']<=edit['edits'] and isfinite(edit['predicted_gain_sum'])
                         and (arm!='N' or indices.size==0),'Exact actual-count/area/action contract')
                need(get_edit(sparse,seed,rank,'SF')['edits']==get_edit(sparse,seed,rank,'SU=F')['edits']
                     and all(get_edit(sparse,seed,rank,a)['edits']==get_edit(sparse,seed,rank,'SU')['edits']
                             for a in ('LF=S','LU=S','LW=S','NP=S')), 'Per-candidate equal realized budgets')
        prepared.append(dict(native=native,pool=pool,sparse=sparse,points=points))
    return prepared


def _reconstruct(np,codec,prepared,seed,arm):
    need(seed in SEEDS and arm in ARMS,'Locked seed/arm')
    predictions=[]
    for image in prepared:
        for original,pool in zip(image['native']['coco'],image['pool']):
            edit=get_edit(image['sparse'],seed,pool['rank'],arm)
            if not edit['edits']:
                predictions.append(original)
                continue
            old=codec.decode(original['segmentation']).astype(bool)
            changed=old.copy(order='C')
            index=np.asarray(edit['flipped'],dtype=np.intp)
            changed.reshape(-1)[index]=~old.reshape(-1)[index]
            adds=int(np.count_nonzero(changed & ~old))
            drops=int(np.count_nonzero(old & ~changed))
            need(adds==edit['adds'] and drops==edit['drops'] and adds+drops==edit['edits']
                 and int(changed.sum())==pool['area']+edit['area_delta'], 'Actual RLE-raster edit counts and area')
            encoded=codec.encode(np.asfortranarray(changed,dtype=np.uint8))
            encoded=dict(size=encoded['size'],counts=encoded['counts'].decode('ascii'))
            need(int(codec.area(encoded))==pool['area']+edit['area_delta'],'Encoded area matches native edits')
            predictions.append(dict(original,segmentation=encoded))
    return predictions


def reconstruct_predictions(np,codec,capture_rows,sparse_rows,seed,arm):
    return _reconstruct(np,codec,prepare_reconstruction(np,codec,capture_rows,sparse_rows),seed,arm)


BASE_SCORE_KEYS={'annotations','AP','AP75','AP90','APs','AR100','all12'}


def legacy_native_block(native):
    """Restore original insertion order, never infer bytes from sorted queries."""
    keys=('image_id','raw_query_count','native_count','unmapped_slot1_count','native_ranks','coco')
    need(set(native)==set(keys),'Exact historical native block fields')
    rows=[]
    for row in native['coco']:
        need(set(row)=={'image_id','category_id','score','segmentation'}
             and set(row['segmentation'])=={'size','counts'},'Exact native COCO and RLE fields')
        rle=row['segmentation']
        need(type(rle['counts']) is str,'Original ASCII RLE counts')
        rle['counts'].encode('ascii')
        rows.append(dict(image_id=row['image_id'],category_id=row['category_id'],score=row['score'],
                         segmentation=dict(size=rle['size'],counts=rle['counts'])))
    return dict(image_id=native['image_id'],raw_query_count=native['raw_query_count'],
                native_count=native['native_count'],unmapped_slot1_count=native['unmapped_slot1_count'],
                native_ranks=native['native_ranks'],coco=rows)


def verify_native_baseline(native_scorer,capture_rows,historical_native_bytes,expected_sha256):
    """Verify old stock hash, parsed science and exact legacy JSONL byte stream."""
    need(type(historical_native_bytes) is bytes
         and sha256(historical_native_bytes).hexdigest()==expected_sha256,'Pinned historical stock bytes')
    start=0
    digest=sha256()
    mapped=0
    for capture in capture_rows:
        end=historical_native_bytes.find(b'\n',start)
        need(end>=start,'Complete original native JSONL line')
        old=historical_native_bytes[start:end+1]
        native=capture['native_block']
        need(json.loads(old)==native,'Exact parsed original prediction science fields')
        rebuilt=native_scorer.encoded(legacy_native_block(native))
        need(rebuilt==old,'Exact historical no-sort field/RLE order and bytes')
        digest.update(rebuilt)
        mapped+=len(native['coco'])
        start=end+1
    need(start==len(historical_native_bytes) and digest.hexdigest()==expected_sha256,
         'No omitted or extra original images')
    return dict(predictions_sha256=expected_sha256,images=len(capture_rows),mapped_candidates=mapped,
                exact_historical_bytes=True,exact_parsed_science=True)


def score_official(native_scorer,gt,predictions,empty_ids):
    """Original score_variants and its official adapter; add only AP50 indexing."""
    output=native_scorer.score_variants(gt,predictions,set(empty_ids))
    need(set(output)=={'primary','sensitivity'},'Original two GT populations')
    for kind in ('primary','sensitivity'):
        row=output[kind]
        need(set(row)==BASE_SCORE_KEYS and len(row['all12'])==12
             and all(isfinite(v) and (v==-1 or 0<=v<=1) for v in row['all12'])
             and isfinite(row['AP90']) and (row['AP90']==-1 or 0<=row['AP90']<=1)
             and row['AP']==row['all12'][0] and row['AP75']==row['all12'][2]
             and row['APs']==row['all12'][3] and row['AR100']==row['all12'][8],
             'Exact original standard COCO summary')
        row['AP50']=row['all12'][1]
    return output


def fixed_match_geometry(np,gt,prepared,fit_step,helpers,geometry_module):
    """Use the original stock .50 matcher once, including valid K=0 records."""
    need([image['id'] for image in gt['images']]==[row['native']['image_id'] for row in prepared],
         'Original GT and prediction image order')
    annotations=defaultdict(list)
    for ann in gt['annotations']:
        annotations[ann['image_id']].append(ann)
    records=[]
    edits={seed:{} for seed in SEEDS}
    audit=Counter()
    for image,row in zip(gt['images'],prepared):
        need([image['height'],image['width']]==row['sparse']['original_hw'],'Original mask/GT canvas')
        points=[dict(image_id=image['id'],rank=cand['rank'],query_id=cand['query_id'],
                     group=row['sparse']['group'],coordinates=row['points'][cand['rank']])
                for cand in row['pool'] if cand['rank'] in row['points']]
        attached,counts=fit_step.attach_truth(image,annotations[image['id']],row['pool'],points,helpers)
        need(len(attached)==len(points) and all(
            all(actual[k]==original[k] for k in ('image_id','rank','query_id','group'))
            and np.array_equal(actual['coordinates'],original['coordinates'])
            for actual,original in zip(attached,points)), 'Matching must retain fixed original point identities')
        records.extend(attached)
        audit.update(counts)
        for record in attached:
            key=tuple(record[k] for k in ('image_id','rank','query_id'))
            for seed in SEEDS:
                edits[seed][key]={arm:get_edit(row['sparse'],seed,record['rank'],arm) for arm in ARMS}
    output=geometry_module.evaluate_geometry(records,edits)
    output['stock_matching_counts']=dict(audit)
    return output


def summarize_scores(coco):
    """All seed means and differences; undefined -1 never becomes zero."""
    def mean(values):
        return None if any(v is None or v==-1 for v in values) else sum(values)/len(values)
    means={}
    contrasts={}
    metrics=('AP','AP50','AP75','AP90','APs','AR100')
    for kind in ('primary','sensitivity'):
        means[kind]={}
        for arm in ARMS:
            rows=[coco[seed][arm][kind] for seed in SEEDS]
            means[kind][arm]={metric:mean([r[metric] for r in rows]) for metric in metrics}
            means[kind][arm]['all12']=[mean([r['all12'][j] for r in rows]) for j in range(12)]
        contrasts[kind]={}
        for arm in ARMS:
            if arm=='SU':
                continue
            difference={}
            for metric in metrics:
                one,two=means[kind]['SU'][metric],means[kind][arm][metric]
                difference[metric]=None if one is None or two is None else 100*(one-two)
            contrasts[kind]['SU_minus_'+arm]=difference
    return means,contrasts


def edit_statistics(prepared):
    result={}
    for seed in SEEDS:
        arms={}
        for arm in ARMS:
            count=Counter()
            gain=0.
            for row in prepared:
                for candidate in row['pool']:
                    edit=get_edit(row['sparse'],seed,candidate['rank'],arm)
                    count.update(candidates=1,edited=int(edit['edits']>0),unchanged=int(edit['edits']==0))
                    for key in ('edits','adds','drops','area_delta','actions','nonpositive_actions','nonpositive_flips'):
                        count[key]+=edit[key]
                    gain+=edit['predicted_gain_sum']
            arms[arm]=dict(count,predicted_gain_sum=gain,
                          nonpositive_action_fraction=(count['nonpositive_actions']/count['actions']
                                                       if count['actions'] else None),
                          nonpositive_flip_fraction=(count['nonpositive_flips']/count['edits']
                                                     if count['edits'] else None))
        result[seed]=arms
    return result


def score_matched_population(np,codec,native_scorer,*,gt,empty_ids,capture_rows,sparse_rows,
                             historical_native_bytes,historical_native_sha256,historical_scores,
                             fit_step,geometry_helpers,geometry_module):
    """Numerical composition, also usable with small synthetic populations.

    This accepts already loaded labels. It grants no input or prelabel admission.
    At most one variant's encoded predictions and one candidate raster are built.
    """
    prepared=prepare_reconstruction(np,codec,capture_rows,sparse_rows)
    parity=verify_native_baseline(native_scorer,capture_rows,historical_native_bytes,historical_native_sha256)
    baseline=score_official(native_scorer,gt,_reconstruct(np,codec,prepared,SEEDS[0],'N'),empty_ids)
    need(all({k:baseline[kind][k] for k in BASE_SCORE_KEYS}==historical_scores[kind]
             for kind in ('primary','sensitivity')),'N reproduces exact historical COCO12/AP90 and both GT populations')
    coco={seed:{'N':baseline} for seed in SEEDS}
    for seed in SEEDS:
        for arm in ARMS:
            if arm=='N':
                continue
            predictions=_reconstruct(np,codec,prepared,seed,arm)
            coco[seed][arm]=score_official(native_scorer,gt,predictions,empty_ids)
            del predictions
    geometry=fixed_match_geometry(np,gt,prepared,fit_step,geometry_helpers,geometry_module)
    gate=geometry_module.locked_gate({seed:{arm:coco[seed][arm]['primary'] for arm in ARMS} for seed in SEEDS},
                                     geometry['summary'])
    means,contrasts=summarize_scores(coco)
    return dict(baseline_parity=dict(parity,exact_historical_scores=True),coco=coco,geometry=geometry,
                mean_COCO=means,contrasts_points=contrasts,budgets=edit_statistics(prepared),gate=gate)


def score_dev(np,codec,native_scorer,*,gt,empty_ids,capture_rows,sparse_rows,historical_native_bytes,
              historical_scores,interpolation_mismatches,fit_step,geometry_helpers,geometry_module):
    """Fixed center5202 E24 DEV705 scorer, invoked only after external admission.

    Caller verifies natural readout exit, independent readback, all sealed hashes
    and original GT bytes before loading labels. This pure function neither opens
    GT nor replaces that gate. The N digest and full numerical scores are checked
    again here. No forward, fitting, seed selection or checkpoint selection.
    """
    old=historical_scores
    need(type(interpolation_mismatches) is int and interpolation_mismatches==0,
         'Zero independently observed CPU interpolation disagreements')
    need(old['status']=='HRSID_DEV705_E24_NATIVE_COCO_COMPLETE'
         and old['checkpoint_sha256']==CENTER_CHECKPOINT_SHA256
         and old['predictions_sha256']==STOCK_PREDICTIONS_SHA256
         and old['original_dev_gt_sha256']==native_scorer.FILES['DEV_ORIGINAL_GT_PRIVATE.json']
         and old['images']==705 and old['excluded_empty_gt_sensitivity']==4
         and old['max_dets']==[1,10,100] and old['units']=='fraction'
         and old['official_TEST_opened'] is False and old['official_TEST_AP'] is False,
         'Exact historical center checkpoint, stock prediction, original GT and score protocol')
    need(len(gt['images'])==len(capture_rows)==len(sparse_rows)==705 and len(gt['annotations'])==1782
         and len({image['id'] for image in gt['images']})==705
         and all(image['height']==image['width']==800 for image in gt['images'])
         and [cat['id'] for cat in gt['categories']]==[1]
         and len({ann['id'] for ann in gt['annotations']})==1782
         and len(empty_ids)==len(set(empty_ids))==4
         and set(empty_ids)<={ann['id'] for ann in gt['annotations']}
         and sum(len(row['pool']) for row in capture_rows)==67386
         and old['primary']['annotations']==1782 and old['sensitivity']['annotations']==1778,
         'Complete unchanged DEV705/1782 and the four original empty-only exclusions')
    result=score_matched_population(np,codec,native_scorer,gt=gt,empty_ids=empty_ids,
        capture_rows=capture_rows,sparse_rows=sparse_rows,historical_native_bytes=historical_native_bytes,
        historical_native_sha256=STOCK_PREDICTIONS_SHA256,
        historical_scores={k:old[k] for k in ('primary','sensitivity')},fit_step=fit_step,
        geometry_helpers=geometry_helpers,geometry_module=geometry_module)
    return dict(status='RFDETR_MATCHED_POINTS_DEV705_SCORING_COMPLETE',seeds=list(SEEDS),arms=list(ARMS),
                images=705,native_mapped_candidates=67386,primary_annotations=1782,sensitivity_annotations=1778,
                detector_checkpoint_sha256=CENTER_CHECKPOINT_SHA256,stock_predictions_sha256=STOCK_PREDICTIONS_SHA256,
                interpolation_mismatches=0,model_forwards=0,optimizer_updates=0,FIT=False,DEV=True,TEST=False,
                GT_opened_by_function=False,loaded_GT_used=True,AP=True,
                public_scope='Fixed DEV recipe development; geometry.private_witness remains private',**result)
