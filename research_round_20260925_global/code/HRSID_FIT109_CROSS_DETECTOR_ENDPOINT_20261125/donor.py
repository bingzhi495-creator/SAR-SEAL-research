"""Source-only fixed-budget RF endpoint ranking with one verified D-FINE distance cue."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.ndimage import distance_transform_edt


PROTOCOL = '5fd53f79435444fa55a8c4eb314571c2232089de63bd7f7cae04c3226fe00834'
PILOT_SOURCE = 'dd35a407d09786af0f6aa3cc61fe86f25edc9a3c3af7a212a9310639720b6960'
PILOT_RESULT = 'f0bdbd6557a7cd40589a0a8007b081d5ce07bf23f867b34714095043a0e60323'
PILOT_TERMINAL = 'b18b108089aa99e295360635d351caa03f96ff6b99b363565073896867ab0ec2'
DONOR_READBACK = 'cd4c2a6a7859319429546f8b9a2ab31f6cfbf7076569ad44611c51f77a5f1d35'
DONOR_TERMINAL = '0db73ab9393d8eb9e80a27dd47d00293577447025a096414ea3ab87ab0e6f5e9'
DONOR_RESULT = '7d9b19175bc445fef6fb1387b287b1c32002d7c74054cc16ae4431963e25f3ad'
DONOR_PREDICTIONS = 'c1d4d0dcdebbe47b083e24ef96be5db57d4655caea919703d183ea5544e0c0fa'
DONOR_CHECKPOINT = '82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8'
DONOR_TOOL_WAIT = '60817f8c0db5c4cecfc9f63048121bb5f04063e37607ba2d7b17a53c08f21526'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, 'Exact closed pin: ' + path.name)
    return raw


def load_pilot(root):
    t = json.loads(pinned(root / 'observed_pilot/TERMINAL.json', PILOT_TERMINAL))
    r = json.loads(pinned(root / 'result/RESULT_SAFE.json', PILOT_RESULT))
    need(t['actual_exit'] == 0 and t['output_footer_verified'] and t['result_sha256'] == PILOT_RESULT
         and t['worker_sha256'] == r['worker_sha256'] == PILOT_SOURCE and not r['decision']['advance'],
         'Closed sibling finding and accepted score path')
    path = root / 'pilot.py'
    src = ModuleType('_closed_sibling_rank')
    src.__file__ = str(path)
    sys.modules[src.__name__] = src
    exec(compile(pinned(path, PILOT_SOURCE), str(path), 'exec'), src.__dict__)
    return src, r


def donors(root, infos):
    audited = json.loads(pinned(root / 'INDEPENDENT_READBACK_SAFE.json', DONOR_READBACK))
    closed = json.loads(pinned(root / 'CONTROL/OWNER_CLOSURE_PRIVATE.json', DONOR_TERMINAL))
    result = json.loads(pinned(root / 'CONTROL/EXPORT/RESULT_SAFE.json', DONOR_RESULT))
    wait = json.loads(pinned(root / 'ORIGINAL_TOOL_WAIT_PRIVATE.json', DONOR_TOOL_WAIT))
    need(audited['status'] == 'DFINE_E24_FIT114_IMAGE_CAPTURE_INDEPENDENT_ACCEPTED'
         and result['status'] == 'DFINE_E24_FIT114_IMAGE_CAPTURE_COMPLETE'
         and wait['terminal']['exit_code'] == closed['original_exit'] == 0 and closed['resource_closed']
         and closed['result_sha256'] == audited['result_sha256'] == DONOR_RESULT
         and audited['terminal_sha256'] == DONOR_TERMINAL and audited['original_tool_wait_sha256'] == DONOR_TOOL_WAIT
         and result['ema_sha256'] == DONOR_CHECKPOINT and result['core_images'] == 109
         and audited['FIT'] and result['FIT'] and not any(audited[k] or result[k] for k in ('TEST','DEV','GT_opened','AP'))
         and audited['raw_probability_arrays_verified'] and result['split_reports'] == audited['split_reports'],
         'Independent closed D-FINE image-only capture authority')
    census = audited['split_reports']['core']
    need(census['images'] == 109 and census['exact_author_binary_replay'] == 10900
         and census['predictions_sha256'] == DONOR_PREDICTIONS, 'Exact original top100 witness')
    raw = pinned(root / 'CONTROL/EXPORT/core/CANDIDATES_PRIVATE.jsonl', DONOR_PREDICTIONS)
    output = {}
    for ordinal, line in enumerate(raw.splitlines(), 1):
        block = json.loads(line)
        iid = block['image_id']
        image = infos[iid][0]
        need(iid not in output and block['ordinal'] == ordinal and block['group'] == image['group']
             and block['basename'] == image['native_basename'] and block['rgb_sha256'] == image['decoded_rgb_sha256'],
             'Same closed original image ID, decoded RGB, prefix and name')
        rows, identities = block['coco'][:100], block['candidates'][:100]
        need(len(rows) == len(identities) == block['exact_author_binary_replay'] == 100, 'Original D-FINE top100 present')
        keep = []
        prior = 1.
        for j, (row, meta) in enumerate(zip(rows, identities, strict=True), 1):
            rle = row['segmentation']
            area = int(codec.area(rle))
            need(row['image_id'] == iid and row['category_id'] == 1 and rle['size'] == [800,800]
                 and 0 <= row['score'] == meta['score'] <= prior and j == meta['native_rank']
                 and area == meta['mask_area'], 'Exact D-FINE mask/rank/score archive identity')
            if area:
                keep.append(row)
            prior = row['score']
        output[iid] = keep
    need(output.keys() == infos.keys(), 'All and only the same 109 decoded FIT images')
    return output


def winner(iou, mutual):
    if not len(iou):
        return None, False
    best = int(np.argmax(iou))
    if iou[best] < .50:
        return None, False
    near = iou >= .50
    near[best] = False
    ambiguous = bool(np.any(near & (mutual[best] < .50)))
    return best, ambiguous


def distance(rle):
    h, w = rle['size']
    x, y, bw, bh = map(int, codec.toBbox(rle))
    need(bw > 0 and bh > 0, 'Only nonempty donor masks enter the signed distance cue')
    x0, y0, x1, y1 = max(0,x-3), max(0,y-3), min(w,x+bw+3), min(h,y+bh+3)
    mask = codec.decode(rle)[y0:y1,x0:x1].astype(bool)
    padded = np.pad(mask, 1, constant_values=False)
    signed = distance_transform_edt(padded) - distance_transform_edt(~padded)
    return np.clip(signed[1:-1,1:-1], -3, 3) / 3, (x0,y0,x1,y1)


def gather(field, crop, cells, width):
    y,x = np.divmod(cells, width)
    x0,y0,x1,y1 = crop
    inside = (x0 <= x) & (x < x1) & (y0 <= y) & (y < y1)
    values = np.full(cells.shape, -1.)
    values[inside] = field[y[inside]-y0,x[inside]-x0]
    return values


def donor_packets(originals, packets, donor_world):
    exposure = Counter()
    for iid, cases in packets.items():
        available = donor_world[iid]
        positions = list(cases)
        rles = [row['segmentation'] for row in available]
        overlaps = codec.iou([originals[iid][p]['segmentation'] for p in positions], rles, [0]*len(rles)) if positions and rles else None
        mutual = codec.iou(rles, rles, [0]*len(rles)) if rles else None
        cache = {}
        exposure['nonempty_donor_masks_top100'] += len(rles)
        for i, pos in enumerate(positions):
            case = cases[pos]
            coords, k = case['coords'], case['k']
            best, ambiguous = winner(overlaps[i], mutual) if overlaps is not None else (None,False)
            case.update(is_sibling=False, cue=np.zeros(coords.shape), occupied=None, sibling_score=0.)
            exposure['supported'] += 1
            exposure['positive_k'] += bool(k)
            exposure['best_donor_iou50'] += (best is not None)
            exposure['ambiguous_donor_abstentions'] += int(ambiguous)
            if best is None or ambiguous:
                continue
            if best not in cache:
                cache[best] = distance(rles[best])
            cue = gather(*cache[best], coords, 800)
            score = available[best]['score']
            centered = score * (cue - cue.mean(axis=0))
            case.update(is_sibling=True, cue=centered, occupied=(cue+1)/2, sibling_score=score)
            exposure['admitted_donor'] += 1
            exposure['admitted_donor_signal'] += bool(np.any(centered))
            exposure['admitted_donor_signal_positive_k'] += bool(np.any(centered) and k)
    return exposure


def prefitting_exposure(matches, packets, pilot):
    groups = set()
    counts = Counter()
    for iid, pos in matches.values():
        case = packets[iid].get(pos)
        if not case or not case['k'] or not case['is_sibling']:
            continue
        q = case['q'][case['outer']]
        p,k = case['coords'],case['k']
        s,r = case['stock_score'],case['sibling_score']
        vote = (s*q+r*case['occupied'])/(s+r) if s+r else q
        before, after = pilot.ranked_writes(q,p,k), pilot.ranked_writes(vote,p,k)
        counts['admitted_donor_positive_k_fixed_matches'] += 1
        counts['admitted_donor_nonconstant_positive_k_fixed_matches'] += bool(np.any(case['cue']))
        if not np.array_equal(np.sort(before),np.sort(after)):
            counts['actual_deterministic_changes'] += 1
            groups.add(case['group'])
    counts['actual_deterministic_changed_prefixes'] = len(groups)
    return dict(counts), counts['actual_deterministic_changes'] >= 20 and len(groups) >= 8


def rename_arms(document):
    if not isinstance(document, dict):
        return document
    names = {
        'duplicate_vote': 'deterministic_donor_vote',
        'native_matched_sibling': 'native_matched_donor',
        'native_matched_sibling_signal_positive_k': 'native_matched_donor_signal_positive_k',
    }
    return {names.get(key, key): rename_arms(value) for key, value in document.items()}


def run(args):
    t0 = perf_counter()
    pinned(args.protocol, PROTOCOL)
    pilot, accepted = load_pilot(args.pilot_run)
    source,sparse,static,originals,infos,models,folds,pair,step,expected_positive = pilot.reference_sources(args)
    donor_world = donors(args.donor_run,infos)
    packets,_ = pilot.prepare(args,source,sparse,originals,infos,models,folds,pair,step)
    exposure = donor_packets(originals,packets,donor_world)
    need(exposure['supported'] == 6421 and exposure['positive_k'] == expected_positive, 'Exact original RF B5 support and budgets')
    before_labels = perf_counter()
    print(json.dumps(dict(stage='closed_donor_and_original_rf_join_verified', **exposure)),flush=True)
    helper = pilot.module(args.code/'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                           '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b','_donor_exact_labels')
    gt = helper.annotations(args,originals)
    coco,matches = source.fixed_coco(gt,originals)
    need(len(matches) == accepted['native_matches'] == 453 and len(gt['annotations']) == 496, 'Original source denominator')
    coverage, proceeds = prefitting_exposure(matches,packets,pilot)
    t_gate = perf_counter()
    print(json.dumps(dict(stage='pre_fit_same_count_change_coverage', **coverage, coverage_pass=proceeds)),flush=True)
    shared = dict(status='HRSID_FIT109_CROSS_DETECTOR_ENDPOINT_COMPLETE_V1',phase='score',protocol_sha256=PROTOCOL,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),source_sibling_pilot_sha256=PILOT_RESULT,
                  original_source_run_sha256=pilot.SOURCE_RESULT,d_fine_readback_sha256=DONOR_READBACK,
                  d_fine_top100_source_rows_sha256=DONOR_PREDICTIONS,d_fine_checkpoint_sha256=DONOR_CHECKPOINT,
                  rf_capture_sha256=source.CAPTURE,source_point_head_seed=5217,detector_source_fit_exposure=True,
                  point_calibration_uses_original_outer_prefix_folds=True,source_images=len(infos),
                  candidates=sum(map(len,originals.values())),hulls=len(gt['annotations']),native_matches=len(matches),
                  native_unmatched=len(gt['annotations'])-len(matches),source_prefix_proxies=len(folds),
                  candidate_exposure=dict(exposure),prefitting_exposure=coverage,prefit_coverage_pass=proceeds,
                  detector_forwards=0,new_point_head_optimizer_updates=0,external_data_opened=False)
    preliminary = dict(label_free_preparation=before_labels-t0, fixed_matching_and_prefit_coverage=t_gate-before_labels)
    if not proceeds:
        return dict(shared, fitted_scalar_coefficients=0, did_full_coco=False,
                    decision=dict(advance=False, reason='prefit_coverage_below_floor'),
                    timing_seconds=dict(preliminary,total=perf_counter()-t0))
    rows = pilot.fixed_truth_records(gt,matches,coco,packets,sparse)
    betas,fit = pilot.fit_folds(rows)
    t_fit = perf_counter()
    worlds,counts = pilot.build_worlds(originals,packets,betas,static,sparse)
    t_writes = perf_counter()
    ap = {arm:pilot.full_metrics(gt, worlds[arm], originals) for arm in pilot.ARMS}
    t_ap = perf_counter()
    geometry, diffs, match_exposure, decision = pilot.geometry_and_gate(gt,matches,coco,infos,worlds,packets,ap)
    for arm in ('native','distance','independent','native_posterior','static_attraction'):
        need(ap[arm] == accepted['full_coco_points'][arm] and geometry[arm] == accepted['fixed_all_hull_geometry'][arm],
             'Original control AP and fixed-group geometry parity: ' + arm)
    need(not step.torch.cuda.is_initialized(),'CPU-only finished replay')
    return rename_arms(dict(shared, fitted_scalar_coefficients=20,did_full_coco=True,full_coco_points=ap,
                fixed_all_hull_geometry=geometry,paired_comparisons_aligned_minus_control=diffs,
                fixed_match_exposure=match_exposure,decision=decision,fitted_fold_safe=fit,
                exact_original_controls_reproduced=True,
                signed_endpoint_count={arm:dict(value) for arm,value in counts.items()},
                timing_seconds=dict(preliminary,deterministic_fit=t_fit-t_gate,validated_edits=t_writes-t_fit,
                                    full_coco=t_ap-t_writes,total=perf_counter()-t0)))


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for path in ('protocol','code','capture','acceptance','source-cohort','basis','heads','fit-gt',
                 'source-run','pilot-run','donor-run','output'):
        p.add_argument('--'+path,type=Path,required=True)
    args=p.parse_args()
    need(not args.output.exists() and args.output.parent.is_dir(),'Fresh result destination')
    result=run(args)
    raw=(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    args.output.mkdir()
    with (args.output/'RESULT_SAFE.json').open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(status=result['status'],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__ == '__main__':
    main()
