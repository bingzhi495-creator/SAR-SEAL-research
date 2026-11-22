"""Capture and independently reconstruct FIT-only RF-DETR native masks."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import os
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZIP_STORED, ZipFile

MIB = 1024**2
FIT_IMAGES, RGB_GROUPS, IMAGES, Q, SIZE = 741, 730, 109, 100, 144
CKPT = "bbe80576d9d8f3ff1b4dc308bd87f199a707ad9dc44c4d48d68533c309aeb9bb"
ACCEPTANCE = "9ec3e2fe0fb841e520142c8d31c8b5830f1d5f7295c06fbd9eac98359ad21200"
REQUEST = "733f29ce60796a3fd47e689e4c72482c3353a16f6ef6cd2e4e622e01f4d3f103"
REPORT = "7a508e02797bd684182becf1eda5e2f12b4430bcac9938f368483898efa7f1df"
CLOSURE = "bdddfba15d0a51f4201690810c728fac817401e42fa671bf0f06fb8c2ab78a9d"
FIT_MANIFEST_SHA = "5bf021523d1ba3507a5ef7ba6a731da8699c37eb32d6b8ea962c52dcfbaafb47"
FIT_ROSTER_SHA = "315f7cfad9a3782865989ced212a6eeea1052e31251a01f24dbe2758bbc4f0e9"
PROTOCOL_SHA = "11a74f36a4b662cffd1a78fcea7b77b9e5d5e18d4543cc191ac296642752fb02"
SELECTION = "FIT_IMAGE_SELECTION_PRIVATE.json"
POPULATION = "PARITY_EXPORT"
WORKER_REL = "SSDD_FLOAT_PRIMARY_DEV187_20261112/float_dev187.py"
WORKER_SHA = "5463f294191929bceaac3b04fb6c17179c05066eaa29b6d859ef4943cca686c7"
COMPLETE = "PSEG_TARGET_RFDETR5201_FIT109_CAPTURE_COMPLETE_V1"
ACCEPTED = "PSEG_TARGET_RFDETR5201_FIT109_CAPTURE_INDEPENDENTLY_VERIFIED_V1"
SCOPE = dict(DEV=False, FIT=True, TEST=False, AP=False, GT_opened=False, optimizer_updates=0)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def read_exact(path, expected, limit=64*MIB):
    need(path.is_file() and path.stat().st_size <= limit, "Exact bounded former input " + path.name)
    raw = path.read_bytes()
    need(digest(raw) == expected, "Hash mismatch " + path.name)
    return raw


def as_document(path, expected, limit=64*MIB):
    return json.loads(read_exact(path, expected, limit))


def encoded(doc):
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def module(path, expected):
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(read_exact(path, expected), str(path), "exec"), m.__dict__)
    return m


def ref(path, expected):
    return dict(path=path.as_posix(), bytes=path.stat().st_size, sha256=expected)


def select_rgb_groups(rows, count):
    groups = {}
    for row in sorted(rows,key=lambda x:x["image_id"]):
        groups.setdefault(row["rgb_sha256"],row)
    need(len(groups) >= count, "Insufficient exact decoded RGB groups")
    ranked = sorted(groups.values(),key=lambda row: (
        digest(("PSEG-SSDD-FIT109-20261122-v1:"+row["rgb_sha256"]).encode("ascii")),row["image_id"],
    ))
    return ranked[:count], len(groups)


def roster(archive_path, root):
    import numpy as np

    need(root.name == "train" and root.is_dir(), "Independent train-only FIT layout")
    with ZipFile(archive_path) as archive:
        info = archive.getinfo("manifest.json")
        need(info.file_size == 861009 and info.compress_type == ZIP_STORED, "Exact bounded image-only FIT manifest")
        raw = archive.read(info)
    need(digest(raw) == FIT_MANIFEST_SHA, "Exact accepted image-only FIT archive manifest")
    payload = json.loads(raw)["rows"]
    need(len(payload) == FIT_IMAGES and len({x["image_id"] for x in payload}) == FIT_IMAGES,
         "Exact FIT-only logical image roster")
    rows, content, names = [], [], set()
    for i, meta in enumerate(payload):
        name = meta["original_basename"]
        image_id = meta["image_id"]
        old_ref = meta["original_image_ref"]
        hw = meta["original_shape"]
        p = root / name
        need(type(image_id) is int and image_id >= 0 and meta["dataset_index"] == i
             and Path(name).name == name and p.suffix == ".jpg" and name not in names
             and p.resolve(strict=True) == p and len(hw) == 2 and all(type(v) is int and v > 0 for v in hw),
             "Exact original FIT manifest and original dimensions")
        item = dict(image_id=image_id, original_hw=hw, image_ref=dict(path=p.as_posix(),bytes=old_ref["bytes"],
                                                                    sha256=old_ref["sha256"]))
        rgb, relative = checked_rgb(item,root,np)
        need(relative == name,"Exact FIT basenames")
        rows.append(dict(item,rgb_sha256=rgb))
        content.append([image_id,name,old_ref["bytes"],old_ref["sha256"]])
        names.add(name)
    need({p.name for p in root.iterdir()} == names | {"_annotations.coco.json"}
         and digest(encoded(content)[:-1]) == FIT_ROSTER_SHA,"Exact closed source image layout and content roster")
    chosen, groups = select_rgb_groups(rows,IMAGES)
    need(groups == RGB_GROUPS, "Exact number of decoded FIT groups")
    return dict(schema="PSeg-SSDD-FIT109-RGB-selection-v1",source_fit_images=FIT_IMAGES,
                unique_rgb_groups=groups,expected_ids=[x["image_id"] for x in chosen],items=chosen)


def setup(args):
    worker = module(args.code / WORKER_REL, WORKER_SHA)
    stage, modules = worker.load_stock(args.code)
    api = modules["ssdd_phase_dev_admission"]
    former = args.historical
    old_req = as_document(former / "STAGE_REQUEST_PRIVATE.json", REQUEST, MIB)
    old_report = as_document(former / "ACTUAL/EXPORT_REPORT_PRIVATE.json", REPORT, MIB)
    closure = as_document(former / "CLOSED_STAGE_PRIVATE.json", CLOSURE, MIB)
    need(old_req["stage"] == "EXPORT_CENTER" and old_report["mode"] == "center"
         and old_report["training_seed"] == 5201 and old_req["archive_ref"]["sha256"] == CKPT
         and closure["report_ref"]["sha256"] == REPORT
         and closure["request_ref"]["sha256"] == REQUEST,
         "Original qualified center 5201 carrier")
    accepted_path = former / "INDEPENDENT_ACCEPTANCE_PRIVATE.json"
    read_exact(accepted_path, ACCEPTANCE, MIB)
    worker.verify_export_acceptance(stage, api, {
        "acceptance_ref":ref(accepted_path, ACCEPTANCE), "closure":closure["closure"]
    }, "center", pair_sha=old_req["training_pair_ref"]["sha256"], evaluation_protocol=old_req["evaluation_protocol"])
    selected = roster(args.fit_archive,args.fit_root)
    items = [{key:value for key,value in row.items() if key != "rgb_sha256"} for row in selected["items"]]
    return worker, stage, modules, api, old_req, selected, items


def write_array(out, rel, value, np):
    buf = BytesIO()
    np.save(buf, value, allow_pickle=False)
    raw = buf.getvalue()
    with (out / rel).open("xb") as handle:
        handle.write(raw)
    return dict(path=rel, bytes=len(raw), sha256=digest(raw))


def checked_rgb(item, root, np):
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    source = item["image_ref"]
    image_path = Path(source["path"])
    need(image_path.is_relative_to(root), "Original image-only rectangle")
    image_bytes = read_exact(image_path, source["sha256"], 16*MIB)
    need(len(image_bytes) == source["bytes"], "Original JPEG bytes")
    image, resize = decode_pil_image_bytes(image_bytes, draft_size=None)
    x = np.asarray(image)
    need(image.mode == "RGB" and resize == (1., 1.) and x.dtype == np.uint8
         and x.shape == (*item["original_hw"],3), "Original decoded native RGB")
    return digest(x.tobytes(order="C")), image_path.relative_to(root).as_posix()


def historical_block(manifest_row, preds):
    begin, count = manifest_row["prediction_start"], manifest_row["prediction_count"]
    out = preds[begin:begin+count]
    need(len(out) == count and all(p["image_id"] == manifest_row["image_id"] and p["category_id"] == 1
                                  for p in out), "Exact original COCO block order")
    return out


class Capture:
    def __init__(self, args, selection, items, stream, np, torch, codec):
        self.root = args.output
        self.items, self.selection = items, selection
        self.images_root, self.stream = args.fit_root, stream
        self.np, self.torch, self.codec = np, torch, codec
        self.stats = Counter()
        self.hash = sha256()

    def wrap(self, original):
        def capture(raw, **kwargs):
            stock = original(raw, **kwargs)
            self.observe(raw, stock, kwargs["image_id"], kwargs["original_hw"], kwargs["postprocess_type"])
            return stock
        return capture

    def observe(self, raw, stock, image_id, hw, postprocess_type):
        np, torch, codec = self.np, self.torch, self.codec
        i = self.stats["images"]
        item = self.items[i]
        rows = stock["coco"]
        need(item["image_id"] == image_id and list(hw) == item["original_hw"]
             and all(x["image_id"] == image_id and x["category_id"] == 1 for x in rows),
             "Exact FIT image identity and native COCO output")
        scores, labels, queries = postprocess_type(num_select=Q)._select_topk(raw["pred_logits"])
        native = stock["native"]
        labels, queries, scores = labels[0], queries[0], scores[0]
        need(torch.equal(scores, native["scores"]) and torch.equal(labels, native["labels"]),
             "Original author stable joint score and class slot identity")
        mapped = torch.nonzero(labels == 0, as_tuple=False).flatten()
        ranks = (mapped+1).tolist()
        qids = queries[mapped].tolist()
        need(len(set(qids)) == len(ranks) == len(rows)
             and [float(scores[x-1].item()) for x in ranks] == [x["score"] for x in rows],
             "All original mapped native ranks and query identities")
        area = [int(codec.area(p["segmentation"])) for p in rows]
        selected = [j for j, n in enumerate(area) if 32 <= n < 1024]
        qr = [qids[j] for j in selected]
        rr = [ranks[j] for j in selected]
        rgb_hash, src = checked_rgb(item, self.images_root, np)
        need(rgb_hash == self.selection["items"][i]["rgb_sha256"], "Exact selected decoded RGB identity")
        original_planes = raw["pred_masks"][0]
        index = torch.tensor(qr, device=original_planes.device, dtype=torch.long)
        eligible = original_planes.index_select(0, index).cpu().contiguous()
        values = eligible.numpy().astype("<f4", copy=False)
        need(values.shape == (len(rr), SIZE, SIZE) and bool(np.isfinite(values).all()), "Finite native logit planes")
        logits = write_array(self.root, f"LOGITS/{image_id}.npy", values, np)
        fallbacks = []
        for start in range(0, len(rr), 8):
            rendered = torch.nn.functional.interpolate(eligible[start:start+8,None], size=hw,
                                                       mode="bilinear", align_corners=False)[:,0]
            for offset, logits_cpu in enumerate(rendered):
                j = start + offset
                rank, query = rr[j], qr[j]
                stock_mask = native["masks"][rank-1,0].cpu()
                if torch.equal(logits_cpu > 0, stock_mask):
                    continue
                gpu = torch.nn.functional.interpolate(original_planes[None,query:query+1], size=hw,
                                                      mode="bilinear", align_corners=False)[0,0]
                need(torch.equal((gpu > 0).cpu(), stock_mask), "Original CUDA render does not recover exact mask")
                p = np.ascontiguousarray(gpu.sigmoid().cpu().numpy(),dtype="<f4")
                saved = write_array(self.root, f"OVERRIDES/{image_id}_{rank}.npy", p, np)
                fallbacks.append(dict(native_rank=rank, query_id=query, **saved))
        doc = dict(image_id=image_id, original_hw=list(hw), stock_prediction_start=self.stats["mapped_candidates"],
            stock_prediction_count=len(rows), source_image_name=src, image_sha256=item["image_ref"]["sha256"],
            rgb_sha256=rgb_hash, native_ranks=ranks, query_ids=qids, native_boxes_xyxy=native["boxes"][mapped].tolist(),
            logit_ranks=rr, logit_query_ids=qr, logits_ref=logits, posterior_overrides=fallbacks,
            joint_native_query_ids=queries.tolist(), joint_native_labels=labels.tolist(),
            joint_native_scores=scores.tolist())
        line = encoded(doc)
        self.stream.write(line)
        self.hash.update(line)
        self.stats.update(images=1,mapped_candidates=len(rows),eligible_planes=len(rr),native_overrides=len(fallbacks))
        if (i+1) % 24 == 0:
            print(json.dumps({"event":"PSEG_TARGET_RFDETR5201_FIT_IMAGE_CAPTURE",**self.stats}), flush=True)


def scoped_items(items, expected_ids, pilot):
    need(len(items) == len(expected_ids) == IMAGES and len(set(expected_ids)) == IMAGES,
         "Exact independent FIT109 selection")
    selected = []
    for item, image_id in zip(items,expected_ids,strict=True):
        path = pilot.original_ref(item["image_ref"],max_bytes=pilot.MAX_JPEG_BYTES,suffix=".jpg")
        hw = tuple(item["original_hw"])
        need(path.parent.name == "train" and item["image_id"] == image_id and hw[0]*hw[1] <= pilot.MAX_PIXELS
             and (not selected or path.parent == selected[0][0].parent),
             "Only authorized FIT JPEGs inside a single train-only folder")
        selected.append((path,item["image_ref"],image_id,hw))
    need(len({x[0] for x in selected}) == IMAGES, "Exact distinct selected FIT paths")
    return selected


def write_document(path, doc):
    raw = encoded(doc)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return ref(path,digest(raw))


def write_population(selected, infer, *, output_directory, checkpoint_sha256, recipe_sha256, event_sink):
    rows, images = [], []
    for item in selected:
        _, _, image_id, hw = item
        predictions, summary = infer(item)
        need(len(summary["retained_native_ranks"]) == len(predictions)
             and summary["raw_query_count"] == summary["native_count"] == Q,
             "Exact stock Q100 prediction identities")
        images.append(dict(image_id=image_id,original_hw=list(hw),prediction_start=len(rows),
                           prediction_count=len(predictions),**summary))
        rows.extend(predictions)
    start = perf_counter()
    output_directory.mkdir()
    pred = write_document(output_directory/"PREDICTIONS.json",rows)
    manifest = write_document(output_directory/"IMAGE_MANIFEST.json",dict(
        schema="RFDETR-E24-FIT109-image-index-v1",checkpoint_sha256=checkpoint_sha256,
        recipe_sha256=recipe_sha256,prediction_offset_base=0,native_rank_base=1,predictions_ref=pred,
        images=images,prediction_count=len(rows),
    ))
    return dict(predictions_ref=pred,image_manifest_ref=manifest,
                serialization_and_optional_write_seconds=perf_counter()-start)


def run_capture(args, setup_values):
    import numpy as np
    from pycocotools import mask as codec
    import torch

    worker, stage, modules, api, request, selection, items = setup_values
    stage.admit_runtime("EXPORT_CENTER")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] >= 6*1024**3,
         "Original single CUDA card under available-memory limit")
    torch.cuda.set_per_process_memory_fraction(min(.35, 6*1024**3 / torch.cuda.mem_get_info()[1]))
    arm_args = stage.train_args(request)
    _, arms = worker.read_training_pair(api,request["training_pair_ref"],**arm_args)
    arm = arms["center"]
    archive_ref = request["archive_ref"]
    archive = api.reference(archive_ref,max_bytes=1024*MIB,suffix=".ckpt")
    need(archive_ref["sha256"] == arm["checkpoint_sha256"] == CKPT
         and archive_ref["bytes"] == arm["checkpoint_bytes"] and archive.name == api.ARCHIVE_NAME,
         "Exact independently accepted FIT741 float center E24 EMA archive")
    args.output.mkdir()
    (args.output/"LOGITS").mkdir()
    (args.output/"OVERRIDES").mkdir()
    selection_ref = write_document(args.output/SELECTION,selection)
    sys.addaudithook(stage.reject_network)
    paths = {Path(x["image_ref"]["path"]) for x in items}
    guard, seen = stage.scope_guard(allowed=paths|{archive},images=paths,data_root=args.fit_root)
    sys.addaudithook(guard)
    original = api.load_sources()
    native_worker = original["worker"]
    native_worker.select_items = scoped_items
    native_worker.write_population = write_population
    model, train = stage.load_recipe(modules["ssdd_phase_init"],args.fit_root.parent)
    stage.configure_export()
    start = perf_counter()
    with (args.output / "CAPTURE_PRIVATE.jsonl").open("xb") as stream:
        sink = Capture(args, selection, items, stream, np, torch, codec)
        stock = native_worker.run_e24_dev_export(
            archive_ref,items,selection["expected_ids"],model,train,recipe_sha256=api.RECIPE_SHA,
            validate_archive_progress=api.bind_archive_validator(arm,long_worker=original["long"]),
            boundary_api=original["boundary"],pilot_api=original["pilot"],
            export_e1_image=sink.wrap(original["raw_export"].export_e1_image),
            canonical_export=original["canonical"].export_coco_mask_predictions,
            output_directory=args.output/POPULATION,event_sink=stage.print_event,
        )
        stream.flush()
        os.fsync(stream.fileno())
    need(seen == paths and sink.stats["images"] == IMAGES and stock["checkpoint_sha256"] == CKPT,
         "Exact selected FIT data access and accepted center EMA")
    return dict(status=COMPLETE, **SCOPE, **sink.stats, detector_forwards=IMAGES, checkpoint_sha256=CKPT,
        original_export_acceptance_sha256=ACCEPTANCE, predictions_sha256=stock["predictions_ref"]["sha256"],
        image_manifest_sha256=stock["image_manifest_ref"]["sha256"],selection_sha256=selection_ref["sha256"],
        capture_stream_sha256=sink.hash.hexdigest(), worker_seconds=perf_counter()-start,
        original_forward_seconds=stock["timing_seconds"]["synchronized_cuda_forward"])


def load_npy(root, info, np, *, expected_rel):
    need(info["path"] == expected_rel, "Exact native plane sidecar path")
    data = read_exact(root / info["path"], info["sha256"], 12*MIB)
    need(len(data) == info["bytes"], "Exact native plane document length")
    return np.load(BytesIO(data),allow_pickle=False)


def run_readback(args, setup_values, protocol):
    import numpy as np
    from pycocotools import mask as codec
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(),
         "CPU-only independent probability readback")
    *_, selection, items = setup_values
    root = args.capture
    terminal = as_document(args.capture_terminal, args.capture_terminal_sha256, MIB)
    need(terminal["actual_exit"] == 0 and terminal["worker_sha256"] == digest(Path(__file__).read_bytes())
         and terminal["output_footer_verified"] is True and terminal["phase"] == "capture",
         "Original observed capture naturally closed")
    capture = as_document(root/"RESULT_SAFE.json", terminal["result_sha256"], MIB)
    need(capture["status"] == COMPLETE and capture["protocol_sha256"] == protocol
         and capture["checkpoint_sha256"] == CKPT and capture["images"] == capture["detector_forwards"] == IMAGES
         and all(capture[k] == v for k,v in SCOPE.items()), "Closed actual capture identity and scope")
    chosen = as_document(root/SELECTION,capture["selection_sha256"],MIB)
    preds = as_document(root/POPULATION/"PREDICTIONS.json",capture["predictions_sha256"])
    manifest = as_document(root/POPULATION/"IMAGE_MANIFEST.json",capture["image_manifest_sha256"],4*MIB)
    need(chosen == selection and manifest["schema"] == "RFDETR-E24-FIT109-image-index-v1"
         and manifest["checkpoint_sha256"] == CKPT and manifest["prediction_offset_base"] == 0
         and manifest["native_rank_base"] == 1 and manifest["predictions_ref"]["sha256"] == capture["predictions_sha256"]
         and len(preds) == manifest["prediction_count"] == capture["mapped_candidates"]
         and len(manifest["images"]) == IMAGES,"Exact closed native FIT predictions and image selection")
    data = read_exact(root/"CAPTURE_PRIVATE.jsonl", capture["capture_stream_sha256"])
    lines = [json.loads(line) for line in data.splitlines()]
    counts = Counter()
    expected_files = {"LOGITS":set(), "OVERRIDES":set()}
    for row, frozen, item, picked in zip(lines, manifest["images"], items, chosen["items"], strict=True):
        identity, hw = row["image_id"], row["original_hw"]
        original = historical_block(frozen, preds)
        need(identity == frozen["image_id"] == item["image_id"] and hw == frozen["original_hw"] == item["original_hw"]
             and row["stock_prediction_start"] == frozen["prediction_start"]
             and row["stock_prediction_start"] == counts["mapped_candidates"]
             and row["stock_prediction_count"] == len(original) and row["native_ranks"] == frozen["retained_native_ranks"]
             and row["image_sha256"] == item["image_ref"]["sha256"]
             and all(p["segmentation"]["size"] == hw for p in original), "FIT image and full native roster")
        rgb, name = checked_rgb(item, args.fit_root, np)
        need(row["rgb_sha256"] == rgb == picked["rgb_sha256"] and row["source_image_name"] == name,
             "Original image identity and decoded view")
        ranks, qids = row["native_ranks"], row["query_ids"]
        n = len(ranks)
        boxes = np.asarray(row["native_boxes_xyxy"],dtype=np.float64).reshape(n,4)
        need(len(qids) == len(set(qids)) == n and bool(np.isfinite(boxes).all()), "Unique mapped queries and finite author boxes")
        joint_q, joint_label, joint_s = (row["joint_native_query_ids"],row["joint_native_labels"],row["joint_native_scores"])
        need(len(joint_q) == len(joint_label) == len(joint_s) == Q
             and len(set(zip(joint_q,joint_label))) == Q and all(type(q) is int and 0 <= q < Q for q in joint_q)
             and all(label in (0, 1) for label in joint_label)
             and all(np.isfinite(v) and 0 <= v <= 1 for v in joint_s)
             and all(joint_s[i-1] > joint_s[i] or (joint_s[i-1] == joint_s[i]
                     and (joint_q[i-1],joint_label[i-1]) < (joint_q[i],joint_label[i]))
                     for i in range(1,Q))
             and [i+1 for i,l in enumerate(joint_label) if l==0] == ranks
             and [joint_q[r-1] for r in ranks] == qids and [joint_s[r-1] for r in ranks] == [x["score"] for x in original],
             "Exact stable native joint order and historical confidence")
        selected = [i for i,p in enumerate(original) if 32 <= int(codec.area(p["segmentation"])) < 1024]
        rr, qr = [ranks[i] for i in selected], [qids[i] for i in selected]
        need(rr == row["logit_ranks"] and qr == row["logit_query_ids"], "Exact eligible logit order")
        rel = f"LOGITS/{identity}.npy"
        expected_files["LOGITS"].add(Path(rel).name)
        logits = load_npy(root,row["logits_ref"],np,expected_rel=rel)
        need(logits.shape == (len(rr),SIZE,SIZE) and logits.dtype == np.dtype("<f4")
             and bool(np.isfinite(logits).all()), "Finite exact eligible planes")
        overrides = {x["native_rank"]:x for x in row["posterior_overrides"]}
        need(len(overrides) == len(row["posterior_overrides"]) and set(overrides) <= set(rr), "Valid native fallback rank roster")
        for start in range(0,len(rr),8):
            rendered = torch.nn.functional.interpolate(torch.from_numpy(logits[start:start+8,None]), size=hw,
                                                       mode="bilinear",align_corners=False)[:,0].numpy()
            for k, values in enumerate(rendered):
                j = start+k
                stock = codec.decode(original[selected[j]]["segmentation"]).astype(bool)
                match = np.array_equal(values>0,stock)
                rank, query = rr[j], qr[j]
                need(match == (rank not in overrides), "Exact native arbitrary-size raster fallback accounting")
                if not match:
                    side = overrides[rank]
                    rel = f"OVERRIDES/{identity}_{rank}.npy"
                    expected_files["OVERRIDES"].add(Path(rel).name)
                    value = load_npy(root,side,np,expected_rel=rel)
                    # Extremely small logits can round sigmoid to exactly 0.5 in either class.
                    need(side["query_id"] == query and value.shape == tuple(hw) and value.dtype == np.dtype("<f4")
                         and bool(np.isfinite(value).all()) and value.min() >= 0 and value.max() <= 1
                         and not np.any((value != .5) & ((value > .5) != stock)), "Native GPU posterior class-side parity")
        counts.update(images=1,mapped_candidates=n,eligible_planes=len(rr),native_overrides=len(overrides))
    need(all(capture[k] == value for k,value in counts.items()) and counts["images"] == IMAGES
         and all({p.name for p in (root/dr).iterdir()} == names for dr,names in expected_files.items()),
         "Whole original sidecar and plane-resource independent readback")
    args.output.mkdir()
    return dict(status=ACCEPTED, **SCOPE, **counts, detector_forwards=0, checkpoint_sha256=CKPT,
        native_raster_parity=True, original_export_acceptance_sha256=ACCEPTANCE,
        predictions_sha256=capture["predictions_sha256"],image_manifest_sha256=capture["image_manifest_sha256"],
        selection_sha256=capture["selection_sha256"],capture_stream_sha256=capture["capture_stream_sha256"],
        capture_result_sha256=terminal["result_sha256"])


def no_label_files(event, values):
    if event != "open" or not isinstance(values[0],(str,bytes)):
        return
    name = Path(os.fsdecode(values[0])).name.casefold()
    if name == "_annotations.coco.json" or name.endswith("_float_polygon_only_private.json"):
        raise PermissionError("Capture admits image-only FIT metadata, with labels sealed")


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("mode", choices=("preflight","capture","verify"))
    for k in ("code","historical","fit-archive","fit-root","output","protocol","capture","capture-terminal"):
        p.add_argument("--"+k,type=Path,required=k in ("code","historical","fit-archive","fit-root","protocol"))
    p.add_argument("--capture-terminal-sha256")
    args = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and sys.flags.utf8_mode, "Isolated exact Python")
    for k,v in vars(args).items():
        if isinstance(v,Path):
            setattr(args,k,v.resolve(strict=k!="output"))
    protocol = digest(read_exact(args.protocol,PROTOCOL_SHA,MIB))
    sys.addaudithook(no_label_files)
    context = setup(args)
    if args.mode == "preflight":
        print(json.dumps({"status":"PSEG_TARGET_RFDETR5201_FIT109_PREFLIGHT_OK",**SCOPE,
                          "images":IMAGES,"source_fit_images":FIT_IMAGES,"unique_rgb_groups":RGB_GROUPS,
                          "selection_sha256":digest(encoded(context[-2])),"detector_forwards":0}))
        return
    need(args.output is not None and not args.output.exists() and args.output.parent.is_dir()
         and not args.output.is_relative_to(args.fit_root) and not args.fit_root.is_relative_to(args.output),
         "Fresh scope-specific destination")
    result = run_capture(args,context) if args.mode=="capture" else run_readback(args,context,protocol)
    result.update(protocol_sha256=protocol, source_sha256=digest(Path(__file__).read_bytes()))
    data=encoded(result)
    with (args.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"],result_sha256=digest(data))),flush=True)


if __name__ == "__main__":
    main()

