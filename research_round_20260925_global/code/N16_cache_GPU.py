"""N16 all100 FIT cache. ROOT-owned once; no scout training or AP.
New background emptyset proxy is shared by both future scouts, not native maskGT.
Frozen original lineage/source/runtime is preserved; no automatic retry.
"""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

OWNER_SHA="ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924"
RAW_KEYS=("receipt","phase","admission","phase_binding","bootstrap")
THREAD_ENV=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")

def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def api(module, name):
    value = getattr(module, name, None)
    require(callable(value), 'ROOT must adapt/review missing external ABI: ' + name)
    return value


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'),
            'Malformed absolute pinned reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def closed(closure, ref):
    ref_shape(ref)
    require(closure.get(ref['path']) == ref, 'Missing exact complete SOURCE reference: ' + ref['path'])
    return bound(ref)


def load(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal_raw_only(ref, phase, code):
    # Reads only original RAW JSON; no SOURCE hash/import or process probe.
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name == phase,
            'Original RAW locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and raw.get('error') is None and
            raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual held exit/reap/original birth absence required before any SHA/import')
    if 'phase' in raw:
        require(raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Preserved RAW resource/admission scope differs')
    return raw


def check_pin(pin, raw, phase, code):
    require(type(pin) is dict and set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and type(pin['returncode']) is int and pin['returncode'] == code and
            pin['original_identity'] == raw['original_identity'] and pin['admission_sha256'] == raw['admission_sha256'],
            'Original actual owner differs from its immutable identity/admission pin')


def ninth_failed_raw_only(spec):
    extra = spec['additional_preserved_independent_CPU_failure']
    require(type(extra) is dict and set(extra) == {'RAW_ref', 'phase', 'pin', 'returncode'} and
            type(extra['returncode']) is int and extra['returncode'] == 1,
            'Independent failed9 needs exactly RAW_ref/phase/pin/returncode1')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= 97 and
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 11 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 11 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 is separate; total terminal pins must equal normal+9')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9 has ALREADY passed its RAW-only identity check.
    rows = spec['RAW_inputs']
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= minimum and type(rows) is list and len(rows) == expected and
            len({row['receipt']['path'] for row in rows}) == expected, 'All original normal RAW ancestors required')
    normals = [terminal_raw_only(row['receipt'], row['phase'], 0) for row in rows]
    failed_ref = spec['prior_failed_CPU_receipt']
    failed_phase = Path(failed_ref['path']).parent.name
    failed = terminal_raw_only(failed_ref, failed_phase, 1)
    extra = spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra) == 7 and
            all(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
                type(row['returncode']) is int and row['returncode'] == 1 for row in extra),
            'Do not alter the original extra failed7 list or preserved failed8')
    failures = [failed] + [terminal_raw_only(row['receipt'], row['phase'], 1) for row in extra]
    failed_paths = [failed_ref['path']] + [row['receipt']['path'] for row in extra]
    require(len(set(failed_paths)) == 8 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
            'Normal owners and preserved failed8 owners must remain distinct')
    all_rows = [dict(row, returncode=0) for row in rows] + [dict(receipt=failed_ref, phase=failed_phase, returncode=1)] + extra
    pins = spec['terminal_identity_pins']
    require(type(pins) is dict and set(pins) == {row['receipt']['path'] for row in all_rows},
            'Original normal-plus-failed8 pins must remain unchanged')
    for row, raw in zip(all_rows, normals + failures):
        check_pin(pins[row['receipt']['path']], raw, row['phase'], row['returncode'])
    require(len(pins) == expected + 8 and
            spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'] not in
            {row['receipt']['path'] for row in all_rows},
            'Legacy normal+failed8 pins required; independent failed9 owner distinct')
    # FIRST HASH: failed9 and every normal/legacy failed8 RAW owner passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(row[key] in closure.values() for row in rows for key in RAW_KEYS if key != 'phase') and
            failed_ref in closure.values() and all(row['receipt'] in closure.values() for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




SCHEMA = "ROOT-N16-all100-FIT-action-cache-GPU-spec-v1"
CPU_SCHEMA = "ROOT-N16-all100-FIT-action-cache-CPU-spec-v1"
CARRIER_KEYS = tuple(["carrier_GPU_spec","carrier_GPU_report","carrier_GPU_RAW","carrier_GPU_admission","carrier_CPU_spec","carrier_CPU_source","carrier_CPU_qualification","carrier_CPU_RAW","carrier_CPU_admission","carrier_evaluation_source"])
EXPECTED_PROTOCOL = json.loads("{\"schema\":\"N16-two-scout-fixed-E24-FIT-selector-locked-v1\",\"locked_UTC\":\"2026-10-14T06:02:44.021Z\",\"locked_before_N16_actual_FIT_cache\":true,\"teacher\":\"qualified native0 E24\",\"fixed_PR\":{\"seed\":202610120701,\"epoch\":24,\"parameters\":40497,\"strong_DEV_AP\":68.09938494745545},\"population\":{\"FIT_images\":741,\"FIT_instances\":1661,\"natural_queries_per_image\":100,\"DEV_images\":187,\"DEV_natural_queries\":18700},\"targets\":{\"positive\":\"last original native matcher assigned raw instance; original mask_weight1\",\"negative\":\"explicit NEW common background-emptyset proxy, only original sampler.neg_inds with background class and mask_weight0\",\"ignored\":\"never assigned empty; invalid target excluded\",\"no_extra_annotation\":true},\"candidate\":{\"P\":\"actual original GPU L>0\",\"Omega\":\"abs(actual original GPU L)<=2\",\"C\":\"Omega union P union Chebyshev radius8 dilation(P)\",\"K\":\"count(Omega)\",\"query_filter\":\"none by score or GT\",\"K0\":\"no edit; no training candidate samples\"},\"sampler\":{\"point_seed_base\":202610140160000,\"matching_loss_seed_base\":202610122001,\"max_points_per_image\":4096,\"proposal\":\"2048 per active matched/background stratum; 4096 if only one; uniform active query then uniform C point, replacement\",\"target_distribution\":\"uniform valid active query within image, then uniform its C point; each image one equal update\",\"importance\":\"(active_query_count_in_stratum/all_active_query_count)/proposal_stratum_probability\",\"loss\":\"mean(weight_i * cross_entropy_i); no per-batch sum-weight renormalization or additional class weights\",\"algorithm\":{\"RNG\":\"Torch CPU private Generator seeded point_seed_base+dataset_index\",\"group_order\":[\"matched\",\"background\"],\"calls\":\"per existing group one randint(n_active_queries,(count,)); then one randint(|C_query|,(1,)) per sampled draw in draw order\",\"sample_order\":\"matched samples followed by background samples; replacement duplicates preserved\",\"weights_dtype\":\"float32\",\"GT_point_sampling\":\"only matcher stratum; no utility/help/harm or pixel GT resampling\"}},\"scout\":{\"inputs\":\"nonaffine LN(F256) and LN(e256), eps1e-5; unclipped raw L/8\",\"layers\":[513,32,3],\"activation\":\"ReLU\",\"parameters\":16547,\"tensors\":4,\"paired_identical_initial_state\":true,\"normal_Linear_init\":true,\"utility_classes\":[\"help\",\"harm\",\"noop\"],\"error_classes\":[\"FP\",\"FN\",\"correct\"]},\"training\":{\"seed\":202610140161,\"order_seed\":202610140160001,\"epochs\":24,\"curve_endpoints\":[8,24],\"AdamW_lr\":0.0003,\"AdamW_weight_decay\":0.0001,\"AdamW_betas\":[0.9,0.999],\"AdamW_eps\":1e-8,\"LR_schedule\":\"constant\",\"update_unit\":\"one same cached image batch per arm; same order/points/weights\",\"empty_image\":\"both skip; count actual updates\",\"mixed_precision\":false,\"gradient_clip\":null},\"selection\":{\"canonical\":\"FP32 scout logits transferred to CPU, stable NumPy FP64 softmax/score\",\"utility\":\"p_help-p_harm\",\"error\":\"2*(p_FP+p_FN)-1\",\"main\":\"atmost K, strictly score>0\",\"supplement\":\"exact K real points only\",\"ties\":[\"score descending\",\"abs native L ascending\",\"flat index ascending\"],\"rank_tolerance\":0,\"qualification\":\"selected indices exact; no allclose substitution\"},\"main_GATE\":{\"metric\":\"official COCO mask AP full natural all100 at fixed E24 scout endpoint; all original class/S0/order unchanged\",\"reference\":\"max(error_same24_AP,68.09938494745545)\",\"GO\":\"utility-minus-reference >=1 AP point and utility>paired error\",\"nonpositive\":\"stop current recipe, no extra budget/seed\",\"positive_under1\":\"at most one prelocked extra seed then original goal repeat rule\",\"no_DEV_epoch_or_threshold_selection\":true},\"strong_controls\":[\"original strong fixed E24\",\"same-trained error scout\",\"dense error+actual_flip indicator: delta*(2p_error-1)\",\"adapted two-class MagNet: 2sigmoid(-absL)*abs(tanh((L+u)/2))\"],\"cost\":{\"canonical_unshared_first\":true,\"shared_projection\":\"future engineering only after both-arm equal numeric qualification\",\"all100_include_background\":true,\"count_all_C_PR_for_dense_controls\":true,\"report_label_generation_separately\":true},\"numeric\":{\"absolute\":0.0001,\"relative\":0,\"actual_after_sign_exact\":true},\"execution\":{\"ROOT_single_leaf\":true,\"CPU_affinity\":[0,1],\"threads\":2,\"allocator_MiB\":4096,\"whole_GPU_observed_MiB\":8192,\"deadline_seconds\":1800,\"automatic_retry\":false},\"claims\":{\"training_started\":false,\"new_AP\":false,\"new_primitive\":false,\"TGRS_ready\":false}}")

def write(path, value):
    with Path(path).open("x",encoding="utf8",newline="\n") as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write("\n")

def validate_protocol(spec,closure):
    protocol=read(closed(closure,spec["locked_protocol"]))
    require(protocol==EXPECTED_PROTOCOL,"Exact locked N16 protocol differs")
    return protocol


def tenth_failed_raw_only(spec):
    extra=spec['additional_preserved_N16_cache_failure']
    require(extra['returncode']==1 and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path']!=spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'],
        'Original N16 failed owner10 must remain separate from normal and old failure owners')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw


def eleventh_failed_raw_only(spec):
    extra=spec['additional_preserved_N16_cache_target_dtype_failure']
    require(extra['returncode']==1 and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in (spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'],
            spec['additional_preserved_N16_cache_failure']['RAW_ref']['path']),
        'Original N16 target-dtype failed owner11 must remain separate')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw

def source_gate(path, spec, cpu=False):
    require(spec['schema'] == (CPU_SCHEMA if cpu else SCHEMA) and
        type(spec['expected_normal_RAW_count']) is int and spec['expected_normal_RAW_count'] >= 105 + int(cpu) and
        spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] and spec['deadline_seconds'] == 1800 and
        spec['automatic_retry'] is False and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
        'New N15 actual full ancestry/schema/owner resource recipe required')
    ninth_failed_raw_only(spec)
    failed10=tenth_failed_raw_only(spec)
    failed11=eleventh_failed_raw_only(spec)
    raws, closure = pre_gate(spec, spec['expected_normal_RAW_count'])
    for key in CARRIER_KEYS + ('entry_source', 'locked_protocol', 'cache_body_source', 'labels_source', 'FIT', 'original_fit_cache_report', 'native_context_spec', 'region_common_source', 'original_fit_cache_entry_source', 'action_diagnostic_source'):
        closed(closure, spec[key])
    closed(closure, spec['additional_preserved_independent_CPU_failure']['RAW_ref'])
    if not cpu:
        require(Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True), 'Current N15 entry differs')
    protocol = validate_protocol(spec, closure)
    result=read(closed(closure,spec['opportunity_ROOT_result']))
    opportunityQ=read(closed(closure,spec['opportunity_CPU_qualification']))
    require(result['status']=='ROOT_N15_FIXED_E24_ACTION_OPPORTUNITY_FULLY_QUALIFIED' and result['GO'] is True and
        result['CPU_Q']==spec['opportunity_CPU_qualification'] and result['CPU_actual_exit_code']==0 and
        result['GPU_actual_exit_code']==0 and opportunityQ['status']=='ROOT_N15_FIXED_E24_ACTION_OPPORTUNITY_SAME_GPU_INPUT_CPU_QUALIFIED_V1' and
        opportunityQ['same_GPU_input_numerical_qualified'] is True, 'Actual qualified opportunity GO required')
    original = read(closed(closure, spec['carrier_GPU_spec']))
    cpu_spec = read(closed(closure, spec['carrier_CPU_spec']))
    q = read(closed(closure, spec['carrier_CPU_qualification']))
    report = read(closed(closure, spec['carrier_GPU_report']))
    require(original['seed'] == cpu_spec['seed'] == q['seed'] == 202610120701 and
        original['head_endpoint_epoch'] == cpu_spec['head_endpoint_epoch'] == q['head_endpoint_epoch'] == 24 and
        q['status'] == 'ROOT_N13_SAMEK_FIXED_FULLDEV_CPU_QUALIFIED_V1' and
        q['evaluation_spec'] == report['spec'] == spec['carrier_GPU_spec'] and q['spec'] == spec['carrier_CPU_spec'] and
        q['source'] == report['source'] == original['entry_source'] == spec['carrier_evaluation_source'] and
        q['reader_source'] == cpu_spec['entry_source'] == spec['carrier_CPU_source'] and
        q['evaluation_report'] == spec['carrier_GPU_report'] and
        q['evaluation_GPU_RAW'] == spec['carrier_GPU_RAW'] and
        q['evaluation_GPU_admission'] == report['GPU_admission'] == spec['carrier_GPU_admission'] and
        q['CPU_admission'] == spec['carrier_CPU_admission'] and
        q['original_native_374pair_6FN_cohort_preserved'] is True and
        q['exact_GPU_L_regions_all18700_CPU_checked'] is True and
        q['independent_two_PR_formula_all_selected_pixels_CPU_checked'] is True,
        'Only original qualified strongest701 LOG E24 carrier is permitted; no N14/E48 substitution')
    require(Path(original['GPU_admission_file']).resolve(strict=True) == Path(spec['carrier_GPU_admission']['path']).resolve(strict=True) and
        Path(cpu_spec['CPU_admission_file']).resolve(strict=True) == Path(spec['carrier_CPU_admission']['path']).resolve(strict=True),
        'Carrier admission locators must resolve to actual bound refs')
    require(all(ref in closure.values() for payload in (original, cpu_spec) for ref in payload['sources']),
        'Full original carrier GPU/CPU Source closure required')
    context = dict(original)
    for key in ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt', 'additional_failed_RAW_inputs',
                'terminal_identity_pins', 'phase', 'entry_source', 'output_directory', 'deadline_seconds'):
        context[key] = spec[key]
    context['CPU_admission_file' if cpu else 'GPU_admission_file'] = spec['CPU_admission_file' if cpu else 'GPU_admission_file']
    lineage = load(original['endpoint_lineage_source'], '_N15_original_E24_lineage')
    lineage.check_recipe(original)
    gate = lineage.own_gate(path, context, spec['entry_source']['path'] if cpu else __file__, closure, raws, cpu=cpu)
    fail10=spec['additional_preserved_N16_cache_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail10[k])
    failrow=dict(receipt=fail10['RAW_ref'],phase=fail10['phase'],admission=fail10['admission'],
        phase_binding=fail10['phase_binding'],bootstrap=fail10['bootstrap'])
    failed_owner=lineage.owner_join(failrow,failed10,closure)
    require(failed_owner['scientific_spec']==fail10['spec'] and failed_owner['entry']==fail10['entry_source']['path'] and
        failed_owner['cuda_visible_devices']=='0','Preserved N16 actual failure10 original SOURCE/spec/ad owner differs')
    fail11=spec['additional_preserved_N16_cache_target_dtype_failure']
    for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec'):closed(closure,fail11[k])
    failrow11=dict(receipt=fail11['RAW_ref'],phase=fail11['phase'],admission=fail11['admission'],
        phase_binding=fail11['phase_binding'],bootstrap=fail11['bootstrap'])
    failed_owner11=lineage.owner_join(failrow11,failed11,closure)
    require(failed_owner11['scientific_spec']==fail11['spec'] and failed_owner11['entry']==fail11['entry_source']['path'] and
        failed_owner11['cuda_visible_devices']=='0','Preserved N16 actual failure11 original SOURCE/spec/ad owner differs')
    gi = lineage.owner_ref_join(context, gate['owners'], spec['carrier_GPU_RAW'], spec['carrier_GPU_admission'],
        spec['carrier_evaluation_source'], spec['carrier_GPU_spec'], '0')
    ci = lineage.owner_ref_join(context, gate['owners'], spec['carrier_CPU_RAW'], spec['carrier_CPU_admission'],
        spec['carrier_CPU_source'], spec['carrier_CPU_spec'], '')
    require(gi < ci < spec['normal_RAW_count'] and original['RAW_inputs'] == spec['RAW_inputs'][:gi] and
        cpu_spec['RAW_inputs'] == spec['RAW_inputs'][:ci], 'Carrier actual GPU0/CPU0 chronological prefixes differ')
    for ancestor in (original, cpu_spec):
        require(all(spec['terminal_identity_pins'].get(key) == value for key, value in ancestor['terminal_identity_pins'].items()),
            'Preserved carrier identity/admission pins differ')
    gate.update(lineage=lineage, context=context, carrier_Q=q, carrier_CPU=cpu_spec, carrier_report=report, protocol=protocol)
    gate.update(lineage.endpoint_gate(context, closure, gate['owners']))
    gate["evaluation"] = load(spec["carrier_evaluation_source"], "_N16_strong_E24_loader")
    return gate

def run(spec_path):
    started=time.monotonic(); path=Path(spec_path).resolve(strict=True)
    require(Path(spec_path).is_absolute(), 'Absolute ROOT spec required')
    spec=read(path); gate=source_gate(path,spec); closure=gate['closure']
    require(spec['cache_budget']['raw_tensor_upper_bytes']==51322624484 and
            spec['cache_budget']['max_raw_packet_bytes']==536870912 and
            spec['cache_budget']['serialized_total_upper_bytes']==60129542144 and
            spec['cache_budget']['sum_original_pixels']==119780015 and
            spec['point_seed_base']==EXPECTED_PROTOCOL['sampler']['point_seed_base'] and
            spec['matching_loss_seed_base']==EXPECTED_PROTOCOL['sampler']['matching_loss_seed_base'],
            'Exact full741 cache bound/seed required')
    import shutil, psutil
    output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and
            shutil.disk_usage(output.parent).free >= 60129542144+2*1024**3, 'Fresh output and prelocked disk margin required')
    require(psutil.Process().cpu_affinity()==[0,1], 'Actual CPU01 before numeric imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(), 'Original Torch/cold CUDA required')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks_api
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0','Original numeric libraries required')
    context=read(closed(closure,spec['native_context_spec']))
    require(context['FIT']==spec['FIT'] and context['checkpointmap']['native0']==gate['cache_report']['teacher_checkpoint'] and
            all(r in closure.values() for r in context['sources']), 'Same native E24/FIT full context closure required')
    cache7=load(spec['original_fit_cache_entry_source'],'_N16_old_qualified_cache')
    common13=load(spec['region_common_source'],'_N16_original_region_common')
    libraries=common13.imported_files(torch,np,cv2,masks_api,COCO,closure,gate['admission'],ndimage)
    modules=[load(context[k],'_N16_'+k) for k in cache7.MODULES]
    native,legacy,pref,initializer,native_common,old,diag,joint,preflight=modules;old.torch=torch
    legacy=load(context['evaluation_source'],'_N16_native_E24_teacher_lineage');teachers=legacy.qualified_training(context)
    register_all_modules(init_default_scope=False);init_default_scope('mmdet')
    construction=dict(teachers['native0']['training_spec'],training_source=context['training_source'],
        checkpointmap=context['checkpointmap'],qualificationmap=context['qualificationmap'],
        order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    model,anchor,policy,loading=native.prepare_training_model(torch,pref,initializer,native_common,MODELS,
        revert_sync_batchnorm,Config,tuple(CocoDataset.METAINFO['classes']),construction,teachers['native0'])
    native.imported_library_gate(torch,model,context)
    require(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve()==closed(closure,context['interpolation_SOURCE']).resolve(),
        'Original renderer interpolation SOURCE required')
    model.requires_grad_(False);model.eval();model.backbone.with_cp=False
    require(len(anchor)==len(model.state_dict())==610 and len(tuple(model.parameters()))==451 and
        not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'Frozen E24 native610/451 required')
    heads,states,proofs=gate['evaluation'].load_heads(torch,gate['context'],gate)
    head=heads['LOG_PR'];pr_anchor={k:v.clone() for k,v in states['LOG_PR'].items()}
    require(len(head.state_dict())==8 and sum(p.numel() for p in head.parameters())==40497 and not head.training,
        'Original strongest LOG E24 8/40497 head required')
    dataset=CocoDataset(ann_file=str(closed(closure,spec['FIT'])),data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco=COCO(str(closed(closure,spec['FIT']))); ids=[dataset.get_data_info(i)['img_id'] for i in range(len(dataset))]
    require(len(dataset)==len(coco.imgs)==741 and len(coco.anns)==1661 and ids==list(coco.imgs) and set(coco.cats)=={1},
        'Full canonical FIT741/1661 only required')
    sumN=sum(r['height']*r['width'] for r in coco.imgs.values())
    require(sumN==119780015 and max(r['height']*r['width'] for r in coco.imgs.values())==306257 and
        all(str(Path(dataset.get_data_info(i)['img_path']).resolve(strict=True)) in closure for i in range(741)),
        'Actual full FIT geometry/image closure differs')
    old_report=read(closed(closure,spec['original_fit_cache_report']))
    require(old_report['images']==741 and old_report['raw_GT']==1661 and len(old_report['packet_refs'])==741,
        'Original matched FIT cache source/report qualification required')
    body=load(spec['cache_body_source'],'_N16_actual_all100_cache_body')
    helper=load(spec['labels_source'],'_N16_actual_sampled_labels')
    diag15=load(spec['action_diagnostic_source'],'_N16_qualified_N15_domains')
    require(not torch.cuda.is_initialized(),'Every admission/source/lineage/budget check precedes CUDA')
    torch.cuda.init();torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0);model.to('cuda:0');head.to('cuda:0');output.mkdir()
    write(output/'START.json',dict(schema='N16-all100-FIT-cache-start-v1',source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],phase=spec['phase'],protocol=gate['protocol'],teacher_loading=loading,
        head_state_proofs=proofs,original_FIT_cache=spec['original_fit_cache_report'],budget=spec['cache_budget'],
        new_background_empty_set_supervision=True,optimizer_created=False,heads_trained=False,no_AP=True,
        imported_libraries=libraries))
    rows=[];packet_refs=[];native_L_refs=[];bytes_written=0;raw_bytes=0;points=0;total_GT=0
    module_map=dict(pref=pref,old=old,diag=diag,joint=joint,preflight=preflight,cache7=cache7,common=native_common)
    with (output/'TRACE.jsonl').open('x',encoding='utf8',newline='\n') as trace:
        for index in range(741):
            require(time.monotonic()-started<1800, 'Prelocked bounded cache deadline; no automatic retry')
            old_ref=old_report['packet_refs'][index];old_row=old_report['rows'][index]
            require(old_row['packet']==old_ref and old_ref in closure.values(),'Qualified old packet ref required')
            old_packet=torch.load(str(closed(closure,old_ref)),map_location='cpu')
            require(old_packet['source']==old_report['source'] and old_packet['spec']==old_report['spec'] and
                old_packet['schema']==cache7.PACKET_SCHEMA, 'Exact old packet/source/spec identity required')
            row,packet=body.cache_image(torch,np,cv2,ndimage,module_map,model,dataset,coco,spec,context,
                old_packet,old_row,index,head,diag15,helper)
            require(row['dataset_index']==index and row['image_id']==ids[index] and row['natural_queries']==100 and
                row['points']<=4096 and row['raw_tensor_payload_bytes']<=536870912,'Actual image/cache bound differs')
            packet.update(source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],
                protocol=spec['locked_protocol'],teacher_checkpoint=context['checkpointmap']['native0'],
                fixed_PR_checkpoint=gate['context']['endpoint']['checkpoints']['LOG_PR'],original_cache_packet=old_ref)
            actual_L=packet.pop('natural_native_L')
            L_target=output/('NATIVE_L_%04d.pth'%index)
            L_packet=dict(schema='N16-actual-GPU-natural100-L-carrier-v1',native_L=actual_L,
                identity=packet['identity'],source=gate['source'],spec=gate['spec_ref'],
                GPU_admission=gate['admission_ref'],phase=spec['phase'],dataset_index=index)
            with L_target.open('xb') as stream:torch.save(L_packet,stream)
            L_replay=torch.load(str(L_target),map_location='cpu')
            require(pref.tree_equal(torch,L_packet,L_replay),'Exact natural100 L carrier readback differs')
            L_ref=reference(L_target);native_L_refs.append(L_ref)
            packet['natural_native_L_carrier']=L_ref
            target=output/('PACKET_%04d.pth'%index)
            with target.open('xb') as stream:torch.save(packet,stream)
            replay=torch.load(str(target),map_location='cpu')
            require(pref.tree_equal(torch,packet,replay),'Full new cache packet double-readback differs')
            ref=reference(target);bytes_written+=ref['bytes']+L_ref['bytes'];raw_bytes+=row['raw_tensor_payload_bytes'];points+=row['points'];total_GT+=row['raw_GT']
            require(bytes_written<=60129542144 and raw_bytes<=51322624484+64*1024**2,'Actual cumulative cache bound exceeded')
            row['packet']=ref;row['natural_native_L_carrier']=L_ref;packet_refs.append(ref);rows.append(row)
            trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
            del old_packet,packet,replay,actual_L,L_packet,L_replay
            print(json.dumps(dict(cached_images=index+1,sampled_points=points,bytes_written=bytes_written)),flush=True)
    final=pref.cpu_tree(torch,model.state_dict())
    require(total_GT==1661 and all(torch.equal(final[k],anchor[k]) for k in anchor) and len(final)==610 and
        all(torch.equal(v.cpu(),pr_anchor[k]) for k,v in head.state_dict().items()) and
        not any(p.grad is not None for p in model.parameters()) and not any(p.grad is not None for p in head.parameters()),
        'Frozen native610 and PR8 must remain unchanged with zero gradients')
    peak=torch.cuda.max_memory_allocated(0);model.cpu();head.cpu();heads.clear()
    del model,head,heads,anchor,final,states;gc.collect();cleanup=pref.clear_gpu(torch)
    require(cleanup['allocated_after']==0 and psutil.Process().cpu_affinity()==[0,1] and
        torch.get_num_threads()==torch.get_num_interop_threads()==2,'Exact CUDA cleanup/resources differ')
    for ref in spec['sources']:bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['GPU_admission_file'])==gate['admission_ref'],'Final source/spec/admission changed')
    write(output/'REPORT.json',dict(schema='N16-all100-FIT-action-cache-GPU-report-v1',
        status='N16_FULLFIT_ALL100_ACTION_CACHE_COMPLETE_PENDING_INDEPENDENT_CPU',source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],phase=spec['phase'],start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),
        packet_refs=packet_refs,native_L_refs=native_L_refs,rows=rows,images=741,raw_GT=1661,natural_queries=74100,sampled_points=points,
        raw_tensor_payload_bytes=raw_bytes,serialized_packet_bytes=bytes_written,recipe=gate['protocol'],
        fixed_PR_checkpoint=gate['context']['endpoint']['checkpoints']['LOG_PR'],teacher_checkpoint=context['checkpointmap']['native0'],
        original_native610_unchanged=True,original_PR8_unchanged=True,new_background_empty_set_supervision=True,
        native_negative_mask_supervision_is_not_claimed=True,one_teacher_forward_per_image=True,teacher_forwards=741,
        weights_updated=False,optimizer_created=False,backward_calls=0,scouts_trained=False,AP_evaluated=False,
        no_new_DEV_prediction_or_GT_labels=True,no_TEST=True,elapsed_seconds=time.monotonic()-started,
        torch_peak_allocated_bytes=peak,cleanup=cleanup,CPU_qualification_pending=True))
    print(json.dumps(dict(status='N16_FULLFIT_ALL100_ACTION_CACHE_COMPLETE_PENDING_INDEPENDENT_CPU',
        report=reference(output/'REPORT.json'))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args();require(args.owner_execute,'Single actual ROOT owner required');run(args.spec)
