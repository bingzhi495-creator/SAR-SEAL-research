"""Read back a terminal, accepted-carrier FIT109 feature capture on CPU without a model or labels."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile


OLD_QUERIES = "62737cefda8d04fe44069e0e9e515fb1081038ebea78ccbfcbad491c80e3da15"
OLD_ACCEPTANCE = "7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f"
ENDPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
MIB = 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def read_pinned(path, expected_sha, limit, expected_size=None):
    require(exact_digest(expected_sha) and not path.is_symlink(), "Required immutable SHA-256 and ordinary artifact")
    with path.open("rb") as file:
        size = os.fstat(file.fileno()).st_size
        require(0 < size <= limit and (expected_size is None or size == expected_size), "Bounded matching byte count")
        data = file.read(size + 1)
    require(len(data) == size and sha256(data).hexdigest() == expected_sha, "Independent file byte identity")
    return data


def index_file(root, ref, fixed_name, limit):
    require(ref["name"] == fixed_name and type(ref["bytes"]) is int and ref["bytes"] > 0,
            "Exact indexed filename and size")
    return read_pinned(root / fixed_name, ref["sha256"], limit, ref["bytes"])


def typed(array, shape, kind):
    import numpy as np
    require(array.shape == shape and array.dtype == np.dtype(kind), "Exact array shape and dtype")
    if array.dtype.kind == "f":
        require(bool(np.isfinite(array).all()), "Finite floating-point payload")


def read_npy(data, shape):
    import numpy as np
    stream = BytesIO(data)
    values = np.load(stream, allow_pickle=False)
    typed(values, shape, "<f4")
    require(stream.read(1) == b"", "Complete original NPY block")
    return values


def row_arrays(data, n):
    import numpy as np
    shapes = {"image_id":((), "<i8"), "ranks":((n,), "<i8"), "query_ids":((n,), "<i8"),
              "areas":((n,), "<i8"), "scores":((n,), "<f8"), "boxes_xyxy":((n,4), "<f4"),
              "raw_logits":((n,144,144), "<f4")}
    result = {}
    with ZipFile(BytesIO(data)) as z:
        names = z.namelist()
        require(len(names) == len(shapes) and set(names) == {k+".npy" for k in shapes}, "Exact original NPZ arrays")
        for name, (shape, kind) in shapes.items():
            member = name + ".npy"
            limit = int(np.prod(shape)) * np.dtype(kind).itemsize + 2048
            require(z.getinfo(member).file_size <= limit, "Bounded NPY member")
            stream = BytesIO(z.read(member))
            value = np.load(stream, allow_pickle=False)
            typed(value, shape, kind)
            require(stream.read(1) == b"", "No extra bytes inside the original NPY array")
            result[name] = value
    return result


def original_rows(block, codec):
    import numpy as np
    ranks, queries, scores, areas = [], [], [], []
    joint = (block["joint_query_ids"], block["joint_labels"], block["joint_scores"])
    require(all(len(a) == 100 for a in joint) and len(block["all_rles"]) == len(block["ship_scores"]) == 164,
            "Closed complete native joint ranking")
    for pos, (qid, label, sc) in enumerate(zip(*joint), 1):
        require(type(qid) is int and 0 <= qid < 164 and type(label) is int and label in (0,1),
                "Original native query identity")
        if label == 0:
            require(sc == block["ship_scores"][qid], "Original mapped score")
            queries.append(qid)
            ranks.append(pos)
            scores.append(sc)
            areas.append(int(codec.area(block["all_rles"][qid])))
    require(queries == block["native_ids"] and ranks == block["native_ranks"] and len(queries) == len(set(queries)),
            "Exact original mapped rank/query join")
    boxes = np.asarray(block["raw_boxes_cxcywh"], dtype="<f4")
    typed(boxes, (164,4), "<f4")
    x, y, w, h = boxes.T
    half = np.float32(.5)
    full = np.stack((x-half*w, y-half*h, x+half*w, y+half*h), axis=1)
    boxes = np.clip(full * np.float32(800), 0, 800) / np.float32(800)
    qids = np.asarray(queries, dtype="<i8")
    return dict(ranks=np.asarray(ranks,dtype="<i8"), query_ids=qids, scores=np.asarray(scores,dtype="<f8"),
                areas=np.asarray(areas,dtype="<i8"), boxes_xyxy=boxes[qids])


def verify_binary(planes, query_ids, rles, torch, codec):
    import numpy as np
    for start in range(0,len(planes),8):
        values = torch.from_numpy(planes[start:start+8,None])
        raster = torch.nn.functional.interpolate(values, size=(800,800), mode="bilinear", align_corners=False)[:,0] > 0
        for restored, qid in zip(raster.numpy(),query_ids[start:start+8]):
            require(np.array_equal(restored,codec.decode(rles[int(qid)])), "Exact original per-query native raster")


def terminal(owner):
    handle = owner.get("original_tool_handle")
    valid_handle = (type(handle) is int and handle > 0) or (type(handle) is str and re.fullmatch("[1-9][0-9]*",handle))
    require(owner.get("status") == "POINTREND_CAPTURE_OWNER_TERMINAL_OBSERVED"
            and owner.get("original_process_terminal") is True and type(owner.get("original_exit")) is int
            and owner["original_exit"] == 0 and valid_handle, "Actual original terminal tool observation and zero exit required")
    require(all(exact_digest(owner.get(k)) for k in ("index_sha256","result_sha256","capture_source_sha256")),
            "Owner binds the original source, result and complete index")


def compare_rows(rows, expected, positions, image_id):
    import numpy as np
    require(rows["image_id"].item() == image_id, "Per-image typed identity")
    for name, value in expected.items():
        chosen = value[positions]
        require(np.array_equal(rows[name], chosen), "Historical row equality: " + name)


def check_old_raw(old, ordinal, block, expected, retained):
    ref = block["logit_capture"]
    filename = "RAW_LOGITS/{:03d}.npy".format(ordinal)
    require(ref["relative_path"] == filename and ref["shape"] == [len(retained),144,144], "Exact prior native plane locations")
    lineage = [dict(logit_row=i, query_id=int(expected["query_ids"][p]), native_rank=int(expected["ranks"][p]),
                    score=float(expected["scores"][p]), predicted_area=int(expected["areas"][p])) for i,p in enumerate(retained)]
    require(ref["candidates"] == lineage, "Every historical retained logit row and identity")
    return read_npy(read_pinned(old/filename, ref["sha256"], 9*MIB, ref["bytes"]), (len(retained),144,144))


def verify(args):
    # No new source, old/new capture or science payload lookup precedes this step.
    owner = json.loads(read_pinned(args.owner_record,args.owner_sha256,MIB))
    terminal(owner)
    require(not args.output.exists() and args.output.parent.is_dir(), "Fresh independent output required")
    read_pinned(args.capture_source,owner["capture_source_sha256"],MIB)
    result = json.loads(read_pinned(args.capture/"RESULT_SAFE.json",owner["result_sha256"],MIB))
    index = json.loads(read_pinned(args.capture/"INDEX_PRIVATE.json",owner["index_sha256"],2*MIB))
    require(result["status"] == "RFDETR_POINTREND_FIT109_CARRIER_CAPTURE_COMPLETE"
            and result["index_sha256"] == owner["index_sha256"] and exact_digest(result["protocol_sha256"])
            and result["raw_acceptance_sha256"] == OLD_ACCEPTANCE
            and result["FIT"] is True and all(result[k] is False for k in ("DEV","TEST","AP","GT_opened"))
            and result["head_updates"] == 0 and result["images"] == result["model_forwards"] == 109,
            "Complete previous FIT-only worker and exact output bindings")
    require(all(result[key] is True for key in ("exact_original_candidates_and_masks","exact_original_eligible_logits",
                                                "exact_all_query_basis_contraction")), "Original worker parity checks completed")
    history = json.loads(read_pinned(args.old_readback,OLD_ACCEPTANCE,2*MIB))
    require(history["status"] == "RF_CENTER5202_FULL_CAPTURE_INDEPENDENTLY_ACCEPTED_V1"
            and history["mode"] == "fit" and history["images"] == 109 and history["source_checkpoint_sha256"] == ENDPOINT
            and history["queries_sha256"] == OLD_QUERIES and history["raw_logit_rows"] == 6738,
            "Accepted previous frozen native carrier")
    prior_bytes = read_pinned(args.old_capture/"QUERIES_PRIVATE.jsonl",OLD_QUERIES,128*MIB)
    prior_lines = prior_bytes.splitlines()
    del prior_bytes
    images = index["images"]
    require(len(prior_lines) == len(images) == 109 and len({r["image_id"] for r in images}) == 109
            and len({r["group"] for r in images}) == 109, "Full unique historical image and source-group roster")
    import numpy as np
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = perf_counter()
    mapped = eligible = degenerate = bytes_spatial = bytes_rows = 0
    expected_files = set()
    for ordinal,(ref,line) in enumerate(zip(images,prior_lines),1):
        block = json.loads(line)
        image_id = ref["image_id"]
        require(type(image_id) is int and image_id == block["image_id"] and type(ref["ordinal"]) is int and ref["ordinal"] == ordinal
                and type(ref["group"]) is str and ref["group"] == block["source_prefix"], "Exact ordered image and source-group join")
        expected = original_rows(block,codec)
        keep = np.flatnonzero((expected["areas"] >= 32) & (expected["areas"] < 1024))
        n = len(keep)
        require(type(ref["rows"]) is int and ref["rows"] == n, "All original area-eligible rows")
        spatial_name = "SPATIAL_{:03d}_PRIVATE.npy".format(ordinal)
        rows_name = "ROWS_{:03d}_PRIVATE.npz".format(ordinal)
        spatial = index_file(args.capture,ref["spatial"],spatial_name,22*MIB)
        read_npy(spatial,(256,144,144))
        bytes_spatial += len(spatial)
        del spatial
        payload = index_file(args.capture,ref["row_archive"],rows_name,12*MIB)
        new = row_arrays(payload,n)
        bytes_rows += len(payload)
        del payload
        compare_rows(new,expected,keep,image_id)
        old_planes = check_old_raw(args.old_capture,ordinal,block,expected,keep)
        require(np.array_equal(old_planes.view(np.uint32),new["raw_logits"].view(np.uint32)), "Bit-exact original retained logits")
        del old_planes
        verify_binary(new["raw_logits"],new["query_ids"],block["all_rles"],torch,codec)
        b = new["boxes_xyxy"]
        require(bool(np.all(b[:,2:] >= b[:,:2])), "No inverted original author boxes")
        degenerate += int(np.count_nonzero(np.any(b[:,2:] == b[:,:2],axis=1)))
        mapped += len(expected["ranks"])
        eligible += n
        expected_files.update([spatial_name,rows_name])
        del new
    actual_files = {f.name for pattern in ("SPATIAL_*","ROWS_*") for f in args.capture.glob(pattern)}
    require(actual_files == expected_files and len(actual_files) == 218, "Exact indexed feature and row file inventory")
    require(mapped == result["mapped"] == 10488 and eligible == result["eligible_rows"] == 6738
            and degenerate == result["zero_area_author_boxes"] == 262 and not torch.cuda.is_initialized(),
            "Closed FIT-only row totals and CPU-only independent check")
    safe = dict(status="RFDETR_POINTREND_FIT109_CARRIER_INDEPENDENT_ACCEPTED", FIT=True, DEV=False, TEST=False,
                images=len(images), groups=len(images), mapped_rows=mapped, eligible_rows=eligible, zero_area_author_boxes=degenerate,
                independent_original_native_masks_exact=eligible, bitwise_old_raw_planes_exact=eligible,
                full_finite_fp32_spatial_maps=len(images), spatial_map_shape=[256,144,144], indexed_files=len(actual_files),
                bytes_spatial=bytes_spatial, bytes_rows=bytes_rows, native_rank_score_area_and_normalized_box_exact=True,
                independent_basis_contraction_recomputed=False, original_worker_reported_all_query_basis_contraction=True,
                basis_readback_limit="Projected query vectors and bias were not saved. The independent check tests complete finite spatial maps and historical raw planes and native binary masks.",
                original_tool_handle=owner["original_tool_handle"], original_exit=owner["original_exit"], original_process_terminal=True,
                owner_record_sha256=args.owner_sha256, protocol_sha256=result["protocol_sha256"],
                capture_source_sha256=owner["capture_source_sha256"], result_sha256=owner["result_sha256"],
                index_sha256=owner["index_sha256"], original_raw_acceptance_sha256=OLD_ACCEPTANCE,
                original_native_queries_sha256=OLD_QUERIES, endpoint_sha256=ENDPOINT,
                verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), independent_verification_seconds=round(perf_counter()-start,3),
                model_forwards=0, optimizer_updates=0, GT_opened=False, AP=False)
    raw = (json.dumps(safe,sort_keys=True,indent=2) + "\n").encode()
    with args.output.open("xb") as file:
        require(file.write(raw) == len(raw), "Complete independent output")
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps(dict(status=safe["status"],sha256=sha256(raw).hexdigest(),images=len(images),eligible_rows=eligible)),flush=True)


def cli(argv=None):
    p = ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("owner-record","capture-source","capture","old-capture","old-readback","output"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--owner-sha256",required=True)
    verify(p.parse_args(argv))


def self_test():
    from tempfile import TemporaryDirectory
    import unittest
    import numpy as np

    class ClosedChecks(unittest.TestCase):
        def test_terminal_precedes_missing_science_inputs(self):
            from types import SimpleNamespace
            with TemporaryDirectory() as where:
                base = Path(where)
                file = base/"owner"
                for observation in ({"status":"running","original_exit":0},
                                    {"status":"POINTREND_CAPTURE_OWNER_TERMINAL_OBSERVED","original_exit":1,
                                     "original_process_terminal":True,"original_tool_handle":8}):
                    raw = json.dumps(observation).encode()
                    file.write_bytes(raw)
                    with self.subTest(observation=observation), self.assertRaisesRegex(ValueError,"original terminal"):
                        verify(SimpleNamespace(owner_record=file,owner_sha256=sha256(raw).hexdigest()))

        def test_idx_file_bytes_and_fixed_names(self):
            with TemporaryDirectory() as where:
                root = Path(where)
                (root/"one").write_bytes(b"123")
                ref = dict(name="one",bytes=3,sha256=sha256(b"123").hexdigest())
                self.assertEqual(index_file(root,ref,"one",5),b"123")
                for bad in (dict(ref,name="../one"),dict(ref,bytes=2),dict(ref,sha256="0"*64)):
                    with self.assertRaises(ValueError):
                        index_file(root,bad,"one",5)

        def test_typed_archive_and_exact_original_row_identity(self):
            arrays = dict(image_id=np.array(7,dtype="<i8"),ranks=np.array([3],dtype="<i8"),query_ids=np.array([4],dtype="<i8"),
                          areas=np.array([35],dtype="<i8"),scores=np.array([.6],dtype="<f8"),boxes_xyxy=np.ones((1,4),dtype="<f4"),
                          raw_logits=np.zeros((1,144,144),dtype="<f4"))
            def packed(fields):
                output = BytesIO()
                np.savez(output,**fields)
                return output.getvalue()
            restored = row_arrays(packed(arrays),1)
            identity = {k:arrays[k] for k in ("ranks","query_ids","scores","boxes_xyxy","areas")}
            compare_rows(restored,identity,np.array([0]),7)
            with self.assertRaises(ValueError):
                compare_rows(restored,dict(identity,ranks=np.array([4])),np.array([0]),7)
            for changed in (dict(arrays,scores=arrays["scores"].astype("<f4")), dict(arrays,extra=np.array([1])),
                            dict(arrays,raw_logits=np.full((1,144,144),np.nan,dtype="<f4"))):
                with self.assertRaises(ValueError):
                    row_arrays(packed(changed),1)

        def test_cpu_probability_threshold_and_native_rle(self):
            import torch
            from pycocotools import mask as codec
            raw = np.zeros((1,144,144),dtype="<f4")
            empty = codec.encode(np.zeros((800,800),dtype=np.uint8,order="F"))
            verify_binary(raw,[0],[empty],torch,codec)
            with self.assertRaises(ValueError):
                verify_binary(raw+1,[0],[empty],torch,codec)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ClosedChecks))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(self_test())
    cli()
