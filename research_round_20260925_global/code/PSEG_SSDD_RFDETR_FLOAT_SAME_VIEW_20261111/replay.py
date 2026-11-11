"""Replay the closed P125/P269 FIT cohort with a polygon-only label overlay."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

OVERLAY_SHA = "97162d23a33b9387fe0b47a117f7e5b1f9dde7471033f244ab336cee1ca9039b"
PLAN_SHA = "31fa8bf86d347597c79beefbec1f8884b93820826e7b0d1e56c65011034f3fea"
MODES = ("integer_native","integer_center","float_native","float_center")
PAIRS = (("integer_native","integer_center"),("float_native","float_center"),
         ("integer_native","float_native"),("integer_center","float_center"))

def need(ok,reason):
    if not ok:
        raise ValueError(reason)

def pinned(path,checksum,limit=1024*1024):
    need(path.is_absolute() and path.resolve(strict=True)==path and 0<path.stat().st_size<=limit,
         "Exact bounded source "+path.name)
    raw = path.read_bytes()
    need(sha256(raw).hexdigest()==checksum,"Pinned SHA "+path.name)
    return raw

def module(path,checksum,name):
    raw = pinned(path,checksum)
    mod = ModuleType(name)
    mod.__file__=str(path)
    sys.modules[name]=mod
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod

def offline(event,_):
    if event in ("socket.connect","socket.getaddrinfo","socket.gethostbyname"):
        raise PermissionError("Offline FIT worker")

def read_prior(prior,exposure):
    p269 = module(exposure/"target_exposure.py",
                  "d93e127fa49bb7bb34be4a56d326abdcb92d9fbe31547c38d63805e8fb32ca19","_p396_p269")
    cap = module(prior/"capture_matcher_cost.py",p269.CAPTURE_SOURCE,"_p396_p125")
    rep = module(prior/"REPLAY_CODE/matcher_cost_runtime.py",p269.REPLAY_SOURCE,"_p396_runtime")
    actual = json.loads(pinned(exposure/"ACTUAL/RESULT_SAFE.json",
        "d13b16956f4ca8518f89cc9f0d32f168d6acd19b24931cff6538a9df66104508"))
    wait = json.loads(pinned(exposure/"TERMINAL_PRIVATE.json",
        "463d2216c5cdf56d8f17b7761f7bbcf04ded502fad16693bda551c8d5bd1b043"))
    need(wait["original_os_wait"] and wait["original_os_exit"]==0 and wait["worker_closed"] and not wait["resource_stop"]
         and wait["result"]["sha256"]=="d13b16956f4ca8518f89cc9f0d32f168d6acd19b24931cff6538a9df66104508"
         and actual["FIT_only"] and actual["DEV"] is actual["TEST"] is False
         and actual["exact_input_and_target_packet_views"]==30 and actual["capture_views"]==32,
         "Exact original closed P269 replay")
    report = json.loads(pinned(prior/"CAPTURE/CAPTURE_SAFE.json",p269.CAPTURE_REPORT))
    closure = json.loads(pinned(prior/"CONTROL/ORIGINAL_TERMINAL_PRIVATE.json",p269.CAPTURE_CLOSURE))
    old = json.loads(pinned(prior/"REPLAY_ACTUAL/RESULT.json",p269.REPLAY_REPORT))
    old_wait = json.loads(pinned(prior/"REPLAY_CONTROL/ORIGINAL_TERMINAL_PRIVATE.json",p269.REPLAY_CLOSURE))
    need(old["protocol"]==p269.REPLAY_PROTOCOL and old["capture_protocol"]==p269.CAPTURE_PROTOCOL
         and old["exact_native_anchors"]==old["exact_native_solver_pair_images"]==30
         and old_wait["exit"]==0 and old_wait["original_OS_wait"] and not old_wait["resource_stop"]
         and old_wait["result_sha256"]==p269.REPLAY_REPORT,"Exact P125 prior replay custody")
    rows = rep.validate_closed(report,closure,p269.CAPTURE_PROTOCOL,p269.CAPTURE_REPORT,cap)
    need(report["tranche"]=="cohort32" and len(rows)==32 and report["exact_stock_traces"]==30
         and sum(r["status"]=="ZERO_TARGET_AUGMENTED" for r in rows)==2,"Exact 32 original views")
    return p269,cap,rep,report,rows

def read_overlay(bundle,cap,items,overlay):
    data = bundle.read("annotations/fit.json")
    need(len(data)==498760 and sha256(data).hexdigest()==cap.FIT_GT_SHA,"Exact frozen integer FIT labels")
    integer = json.loads(data)
    floating = json.loads(pinned(overlay,OVERLAY_SHA,2*1024*1024))
    need({k:v for k,v in integer.items() if k!="annotations"}=={k:v for k,v in floating.items() if k!="annotations"}
         and {im["id"] for im in integer["images"]}=={r["image_id"] for r in items}
         and len(integer["images"])==741 and len(integer["annotations"])==len(floating["annotations"])==1661,
         "Same FIT-only census and non-annotation metadata")
    src_i,src_f = defaultdict(list),defaultdict(list)
    for a,b in zip(integer["annotations"],floating["annotations"],strict=True):
        need({k:v for k,v in a.items() if k!="segmentation"}=={k:v for k,v in b.items() if k!="segmentation"}
             and [[int(x) for x in ring] for ring in b["segmentation"]]==a["segmentation"]
             and a["category_id"]==1,"Exact row order and polygon-only restoration")
        src_i[a["image_id"]].append(a)
        src_f[b["image_id"]].append(b)
    return src_i,src_f

def make_views(image,prep_i,prep_f,stock,native,p269,seed,torch):
    from common_mask_supervision import ROW_ID
    from paired_fit_transform import _check_output,_prepare,_rng,_rng_bytes
    from trace_resize import capture_transform
    def fields_equal(a,b,ignored):
        return set(a)==set(b) and all(torch.equal(a[k],b[k]) and a[k].dtype==b[k].dtype
                                      for k in a if k not in ignored)
    need(fields_equal(prep_i,prep_f,{"masks"}),"Exact original nonmask target equality")
    views = {}
    for kind,prepared in (("integer",prep_i),("float",prep_f)):
        source = _prepare(image,prepared)
        for sample in ("native","center"):
            p269.seed_native(seed)
            with capture_transform(native,mode=sample) as trace:
                pixels,target = stock(image.copy(),{k:v.clone() for k,v in source.items()})
            _check_output(pixels,target,source)
            views[kind+"_"+sample]=(pixels,target,trace,_rng_bytes(_rng()))
    p,a,trace,rng = views["integer_native"]
    for name,(other,b,steps,state) in views.items():
        need(torch.equal(p,other) and p.dtype==other.dtype and steps==trace and state==rng,
             "Identical pixels, transform trace and post-augmentation RNG for "+name)
        need(fields_equal(a,b,{"masks"}),"Identical nonmask target tensors and row IDs for "+name)
        need(ROW_ID in b and len(b[ROW_ID])==len(b["masks"]),"Retained row IDs agree with masks")
    return p,{name:v[1] for name,v in views.items()},"crop" if any(step["op"]=="crop" for step in trace) else "direct"

def masks_only(targets,torch):
    mask = {name:target["masks"].bool() for name,target in targets.items()}
    n = len(mask["integer_native"])
    support = {name:m.flatten(1).any(1) for name,m in mask.items()}
    details = {}
    for name in MODES:
        details[name] = dict(rows=n,nonempty=int(support[name].sum()),empty=n-int(support[name].sum()))
    changed = {}
    for a,b in PAIRS:
        diff = mask[a]^mask[b]
        changed[a+"_vs_"+b] = dict(changed_masks=int(diff.flatten(1).any(1).sum()),xor_pixels=int(diff.sum()),
                                   support_only_first=int((support[a]&~support[b]).sum()),
                                   support_only_second=int((~support[a]&support[b]).sum()))
    guard = {}
    for kind in ("integer","float"):
        a,b = kind+"_native",kind+"_center"
        replace = support[a]&support[b]
        guarded = torch.where(replace[:,None,None],mask[b],mask[a])
        need(torch.equal(guarded.flatten(1).any(1),support[a]),"Support-preserving center replay")
        guard[kind] = dict(center_applied=int(replace.sum()),native_fallback=n-int(replace.sum()),
                           changed_masks=int((guarded^mask[a]).flatten(1).any(1).sum()))
    return dict(modes=details,comparisons=changed,guard=guard,
                all_four_nonempty=int(torch.stack(tuple(support.values())).all(dim=0).sum()))

def add_numbers(left,right):
    for key,val in right.items():
        if isinstance(val,dict):
            add_numbers(left.setdefault(key,{}),val)
        else:
            left[key]=left.get(key,0)+val

def run(prior,exposure,overlay,protocol,output,cpu_core):
    start = perf_counter()
    need(os.sched_getaffinity(0)=={cpu_core} and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and os.environ.get("OMP_NUM_THREADS")=="1" and sys.version_info[:3]==(3,12,14),
         "Isolated one-thread offline CPU runtime")
    pinned(protocol,PLAN_SHA,16384)
    sys.addaudithook(offline)
    p269,cap,replay,report,rows = read_prior(prior,exposure)

    import torch
    from rfdetr.datasets import _torchvision as native
    from rfdetr.datasets.coco import ConvertCoco,make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.utilities.tensors import make_collate_fn
    need(not torch.cuda.is_available() and {k:version(k) for k in cap.VERSIONS}==cap.VERSIONS
         and {k:version(k) for k in report["decoder_versions"]}==report["decoder_versions"]
         and all(cap.sha(Path(import_module(k).__file__).read_bytes())==v for k,v in cap.NATIVE_SOURCE.items()),
         "Exact original RF-DETR and decoder sources")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for name,digest in p269.PINNED_SOURCE.items():
        module(exposure/"GEOMETRY"/(name+".py"),digest,name)
    from common_mask_supervision import ROW_ID
    items = json.loads(pinned(prior/"FIT741_IMAGE_ONLY_MANIFEST_PRIVATE.json",cap.FIT_INDEX_SHA))["items"]
    need(len(items)==741 and [r["dataset_index"] for r in items]==list(range(741)),"Exact original FIT roster")
    zip_data = pinned(prior/"SSDD_FIT741_ORIGINAL_INPUTS_20261107_01.zip",cap.FIT_BUNDLE_SHA,34*1024*1024)
    need(len(zip_data)==33079338,"Exact original FIT bundle length")
    plan = cap.view_plan(items,"cohort32")
    convert = ConvertCoco(include_masks=True,include_keypoints=False,cat2label={1:0},num_keypoints_per_class=[])
    stock = make_coco_transforms_square_div_64(
        "train",cap.RESOLUTION,multi_scale=False,expanded_scales=False,skip_random_resize=True,
        patch_size=12,num_windows=1,aug_config=None,scale_jitter=True,gpu_postprocess=False,keypoint_flip_pairs=None)
    collate = make_collate_fn(block_size=12,pack=True,pad_targets_to=None)
    branches = {"direct":{},"crop":{}}
    counts = Counter()
    with ZipFile(BytesIO(zip_data)) as bundle:
        original,floating = read_overlay(bundle,cap,items,overlay)
        for ordinal,(reference,(item,repeat)) in enumerate(zip(rows,plan,strict=True),1):
            seed = 5301000+2*item["dataset_index"]+repeat
            iid,digest = item["image_id"],item["image_sha256"]
            jpeg = bundle.read("images/"+digest+".jpg")
            need(len(jpeg)==item["image_bytes"] and cap.sha(jpeg)==digest,"Exact original JPEG")
            image,scale = decode_pil_image_bytes(jpeg,draft_size=None)
            h,w = item["original_hw"]
            need(image.mode=="RGB" and image.size==(w,h) and scale==(1.,1.),"Original JPEG canvas")
            _,a = convert(image,dict(image_id=iid,annotations=original[iid]))
            _,f = convert(image,dict(image_id=iid,annotations=floating[iid]))
            pixels,targets,branch = make_views(image,a,f,stock,native,p269,seed,torch)
            native_target = targets["integer_native"]
            n = len(native_target["labels"])
            need(n==reference["target_rows"],"Exact native augmented row count")
            counts[branch+"_views"]+=1
            if n:
                need(reference["status"]=="STOCK_TRACE_EXACT","Original packet eligible")
                packet = replay.load_packet(torch,prior/"CAPTURE",reference,plan,p269.CAPTURE_PROTOCOL,cap)
                samples,packed = collate([(pixels,{k:v for k,v in native_target.items() if k!=ROW_ID})])
                tgt, = packed.to_list("cpu",non_blocking=False)
                need(samples.tensors.shape==(1,3,cap.RESOLUTION,cap.RESOLUTION)
                     and not bool(samples.mask.any())
                     and cap.state_digest({"pixels":samples.tensors})==packet["input_sha256"],
                     "Exact original packed A pixels")
                need(all(tgt[k].dtype==packet["target_"+k].dtype and torch.equal(tgt[k],packet["target_"+k])
                         for k in ("masks","boxes","labels")),"Exact original packed A target")
                counts["original_exact_tensor_views"]+=1
                del packet
            else:
                need(reference["status"]=="ZERO_TARGET_AUGMENTED","Keep original empty augment")
                counts[branch+"_original_empty_views"]+=1
            add_numbers(branches[branch],masks_only(targets,torch))
            counts["equal_nonmask_rng_and_rows_views"]+=1
            if ordinal%8==0:
                print(json.dumps(dict(event="P396_FIT_VIEW_REPLAY_PROGRESS",ordinal=ordinal)),flush=True)
    need(counts["equal_nonmask_rng_and_rows_views"]==32 and counts["original_exact_tensor_views"]==30
         and counts["crop_original_empty_views"]==2 and counts["direct_original_empty_views"]==0
         and not torch.cuda.is_initialized(),"Full original witness and CPU-only closure")
    result = dict(status="P396_SSDD_RFDETR_FLOAT_SAME_VIEW_ADMISSION_COMPLETE",FIT=True,DEV=False,TEST=False,
                  AP=False,images=32,model_forward_calls=0,optimizer_updates=0,counts=dict(counts),branches=branches,
                  overlay_sha256=OVERLAY_SHA,protocol_sha256=PLAN_SHA,p269_capture_report_sha256=p269.CAPTURE_REPORT,
                  actual_float_training_started=False,seconds=perf_counter()-start)
    data = (json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest())),flush=True)

if __name__=="__main__":
    p = ArgumentParser(allow_abbrev=False)
    for name in ("prior","closed-p269","overlay","protocol","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cpu-core",type=int,required=True)
    args = p.parse_args()
    run(args.prior,args.closed_p269,args.overlay,args.protocol,args.output,args.cpu_core)

