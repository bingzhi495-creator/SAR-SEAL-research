"""Read-only target-trained RF-DETR DEV187 raw-mask sidecar and independent readback."""
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

MIB = 1024**2
IMAGES, STOCK_ROWS, Q, SIZE = 187, 16161, 100, 144
CKPT = "bbe80576d9d8f3ff1b4dc308bd87f199a707ad9dc44c4d48d68533c309aeb9bb"
ACCEPTANCE = "9ec3e2fe0fb841e520142c8d31c8b5830f1d5f7295c06fbd9eac98359ad21200"
REQUEST = "733f29ce60796a3fd47e689e4c72482c3353a16f6ef6cd2e4e622e01f4d3f103"
REPORT = "7a508e02797bd684182becf1eda5e2f12b4430bcac9938f368483898efa7f1df"
CLOSURE = "bdddfba15d0a51f4201690810c728fac817401e42fa671bf0f06fb8c2ab78a9d"
PRED = "c81599bf9c627e798a78fa47397a10afe11a0a50491330b5fc9ba583fd1b22c4"
MANIFEST = "78390a66fbf01eea400f6528d699f40f3aa4acbf2129afbbd8d91d8224304bd4"
WORKER_REL = "SSDD_FLOAT_PRIMARY_DEV187_20261112/float_dev187.py"
WORKER_SHA = "5463f294191929bceaac3b04fb6c17179c05066eaa29b6d859ef4943cca686c7"
COMPLETE = "PSEG_TARGET_RFDETR5201_DEV187_CAPTURE_COMPLETE_V1"
ACCEPTED = "PSEG_TARGET_RFDETR5201_DEV187_CAPTURE_INDEPENDENTLY_VERIFIED_V1"
SCOPE = dict(DEV=True, FIT=False, TEST=False, AP=False, GT_opened=False, optimizer_updates=0)


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
    art = former / "ACTUAL/ARTIFACTS"
    preds = as_document(art / "PREDICTIONS.json", PRED)
    manifest = as_document(art / "IMAGE_MANIFEST.json", MANIFEST, 4*MIB)
    old_info = old_report["result"]
    need(old_info["predictions_ref"]["sha256"] == manifest["predictions_ref"]["sha256"] == PRED
         and old_info["image_manifest_ref"]["sha256"] == MANIFEST
         and len(preds) == manifest["prediction_count"] == STOCK_ROWS and len(manifest["images"]) == IMAGES
         and manifest["prediction_offset_base"] == 0 and manifest["native_rank_base"] == 1
         and all(old_info[key] is False for key in ("GT_read", "TEST", "AP", "training")),
         "Original frozen stock bytes and report")
    items, ids = api.load_image_manifest(old_req["image_manifest_ref"], Path(old_req["dev_root"]))
    need(len(items) == len(ids) == IMAGES and ids == [r["image_id"] for r in manifest["images"]],
         "Whole aligned label-free DEV187")
    return worker, stage, modules, api, old_req, preds, manifest, items


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
    def __init__(self, args, preds, manifest, items, images_root, stream, np, torch, codec):
        self.root = args.output
        self.preds, self.manifest, self.items = preds, manifest, items
        self.images_root, self.stream = images_root, stream
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
        item, historical = self.items[i], self.manifest["images"][i]
        rows = historical_block(historical, self.preds)
        need(item["image_id"] == historical["image_id"] == image_id
             and list(hw) == item["original_hw"] == historical["original_hw"] and stock["coco"] == rows,
             "Exact historical image, rank, score and mask parity")
        _, labels, queries = postprocess_type(num_select=Q)._select_topk(raw["pred_logits"])
        native = stock["native"]
        scores, _, _ = postprocess_type(num_select=Q)._select_topk(raw["pred_logits"])
        labels, queries, scores = labels[0], queries[0], scores[0]
        need(torch.equal(scores, native["scores"]) and torch.equal(labels, native["labels"]),
             "Original author stable joint score and class slot identity")
        mapped = torch.nonzero(labels == 0, as_tuple=False).flatten()
        ranks = (mapped+1).tolist()
        qids = queries[mapped].tolist()
        need(ranks == historical["retained_native_ranks"] and len(set(qids)) == len(ranks) == len(rows),
             "All original mapped native ranks and query identities")
        area = [int(codec.area(p["segmentation"])) for p in rows]
        selected = [j for j, n in enumerate(area) if 32 <= n < 1024]
        qr = [qids[j] for j in selected]
        rr = [ranks[j] for j in selected]
        rgb_hash, src = checked_rgb(item, self.images_root, np)
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
        doc = dict(image_id=image_id, original_hw=list(hw), stock_prediction_start=historical["prediction_start"],
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
            print(json.dumps({"event":"PSEG_TARGET_RFDETR5201_IMAGE_CAPTURE",**self.stats}), flush=True)


def run_capture(args, setup_values):
    import numpy as np
    from pycocotools import mask as codec
    import torch

    worker, stage, modules, api, original_request, preds, manifest, items = setup_values
    stage.admit_runtime("EXPORT_CENTER")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] >= 6*1024**3,
         "Original single CUDA card under available-memory limit")
    torch.cuda.set_per_process_memory_fraction(min(.35, 6*1024**3 / torch.cuda.mem_get_info()[1]))
    request = dict(original_request, output_directory=(args.output / "PARITY_EXPORT").as_posix())
    args.output.mkdir()
    (args.output/"LOGITS").mkdir()
    (args.output/"OVERRIDES").mkdir()
    req_file = args.output/"IMAGE_ONLY_REQUEST_PRIVATE.json"
    req_file.write_bytes(encoded(request))
    request = worker.read_request(stage, req_file, digest(req_file.read_bytes()), "EXPORT_CENTER")
    sys.addaudithook(stage.reject_network)
    original_loader = api.load_sources
    start = perf_counter()
    with (args.output / "CAPTURE_PRIVATE.jsonl").open("xb") as stream:
        sink = Capture(args, preds, manifest, items, Path(request["dev_root"]), stream, np, torch, codec)
        def capture_loader():
            sources = original_loader()
            sources["raw_export"].export_e1_image = sink.wrap(sources["raw_export"].export_e1_image)
            return sources
        api.load_sources = capture_loader
        outer = stage.run_export(request, worker.float_api(api), modules["ssdd_phase_init"], event_sink=stage.print_event)
        stream.flush()
        os.fsync(stream.fileno())
    stock = outer["result"]
    need(stock["predictions_ref"]["sha256"] == PRED and sink.stats["images"] == IMAGES
         and sink.stats["mapped_candidates"] == STOCK_ROWS and stock["checkpoint_sha256"] == CKPT,
         "Exact original full stock prediction digest")
    fresh = as_document(Path(stock["image_manifest_ref"]["path"]), stock["image_manifest_ref"]["sha256"], 4*MIB)
    fresh["predictions_ref"]["path"] = manifest["predictions_ref"]["path"]
    need(fresh == manifest, "Full original manifest parity apart from intentionally new path")
    return dict(status=COMPLETE, **SCOPE, **sink.stats, detector_forwards=IMAGES, checkpoint_sha256=CKPT,
        original_export_acceptance_sha256=ACCEPTANCE, predictions_sha256=PRED, image_manifest_sha256=MANIFEST,
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
    *_, request, preds, manifest, items = setup_values
    root = args.capture
    terminal = as_document(args.capture_terminal, args.capture_terminal_sha256, MIB)
    need(terminal["actual_exit"] == 0 and terminal["worker_sha256"] == digest(Path(__file__).read_bytes())
         and terminal["output_footer_verified"] is True and terminal["phase"] == "capture",
         "Original observed capture naturally closed")
    capture = as_document(root/"RESULT_SAFE.json", terminal["result_sha256"], MIB)
    need(capture["status"] == COMPLETE and capture["protocol_sha256"] == protocol
         and capture["predictions_sha256"] == PRED and capture["image_manifest_sha256"] == MANIFEST
         and capture["checkpoint_sha256"] == CKPT and capture["images"] == capture["detector_forwards"] == IMAGES
         and all(capture[k] == v for k,v in SCOPE.items()), "Closed actual capture identity and scope")
    data = read_exact(root/"CAPTURE_PRIVATE.jsonl", capture["capture_stream_sha256"])
    lines = [json.loads(line) for line in data.splitlines()]
    counts = Counter()
    expected_files = {"LOGITS":set(), "OVERRIDES":set()}
    for row, frozen, item in zip(lines, manifest["images"], items, strict=True):
        identity, hw = row["image_id"], row["original_hw"]
        original = historical_block(frozen, preds)
        need(identity == frozen["image_id"] == item["image_id"] and hw == frozen["original_hw"] == item["original_hw"]
             and row["stock_prediction_start"] == frozen["prediction_start"]
             and row["stock_prediction_count"] == len(original) and row["native_ranks"] == frozen["retained_native_ranks"]
             and row["image_sha256"] == item["image_ref"]["sha256"], "Original image and native roster")
        rgb, name = checked_rgb(item, Path(request["dev_root"]), np)
        need(row["rgb_sha256"] == rgb and row["source_image_name"] == name, "Original image identity and decoded view")
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
        original_export_acceptance_sha256=ACCEPTANCE, predictions_sha256=PRED, image_manifest_sha256=MANIFEST,
        capture_stream_sha256=capture["capture_stream_sha256"], capture_result_sha256=terminal["result_sha256"])


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("mode", choices=("preflight","capture","verify"))
    for k in ("code","historical","output","protocol","capture","capture-terminal"):
        p.add_argument("--"+k,type=Path,required=k in ("code","historical"))
    p.add_argument("--protocol-sha256")
    p.add_argument("--capture-terminal-sha256")
    args = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and sys.flags.utf8_mode, "Isolated exact Python")
    for k,v in vars(args).items():
        if isinstance(v,Path):
            setattr(args,k,v.resolve(strict=k!="output"))
    context = setup(args)
    if args.mode == "preflight":
        print(json.dumps({"status":"PSEG_TARGET_RFDETR5201_DEV187_PREFLIGHT_OK",**SCOPE,
                          "images":IMAGES,"mapped_candidates":STOCK_ROWS,"detector_forwards":0}))
        return
    need(args.output is not None and not args.output.exists() and args.output.parent.is_dir(), "Fresh scope-specific destination")
    protocol = digest(read_exact(args.protocol, args.protocol_sha256, MIB))
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

