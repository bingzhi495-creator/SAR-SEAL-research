"""Stream native PaddleDetection Mask RT-DETR masks from an ordered image roster."""

from time import perf_counter

from MASK_RTDETR_AUTHOR_READER_IDS_20261111.reader_ids import bind_reader_ids

from author_stream import write_author_stream

NATIVE_ROWS = 100
INPUT_SHAPE = (1, 3, 640, 640)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def create_endpoint_model(config, checkpoint, *, paddle, np):
    """Build one-class author inference and load the complete native EMA state."""
    from ppdet.core.workspace import create, load_config
    from ppdet.modeling.post_process import DETRPostProcess
    from ppdet.utils.checkpoint import load_weight

    use_gpu = paddle.device.get_device().startswith("gpu:")
    cfg = load_config(str(config))
    cfg.update(use_gpu=use_gpu, num_classes=1, worker_num=0)
    reader_cfg = cfg["TestReader"]
    require(reader_cfg["sample_transforms"] == cfg["EvalReader"]["sample_transforms"]
            and reader_cfg["batch_size"] == 1 and not reader_cfg["shuffle"] and not reader_cfg["drop_last"],
            "The native stock batch-one reader must match evaluation transforms")
    cfg["TestDataset"].update(dataset_dir=None, anno_path=None)
    model = create(cfg.architecture)
    model.load_meanstd(reader_cfg["sample_transforms"])
    post = model.post_process
    require(type(post) is DETRPostProcess and post.with_mask and post.use_focal_loss and post.num_classes == 1
            and post.num_top_queries == NATIVE_ROWS and post.mask_threshold == .5
            and not post.use_avg_mask_score and post.bbox_decode_type == "origin",
            "Fixed native Mask RT-DETR postprocess")
    state = paddle.load(str(checkpoint))
    initial = model.state_dict()
    require(initial.keys() == state.keys() and len(initial) == 646
            and all(initial[key].shape == state[key].shape and initial[key].dtype == state[key].dtype for key in initial),
            "Expected complete one-class author EMA state")
    load_weight(model, str(checkpoint))
    for name, value in model.state_dict().items():
        actual = value.numpy()
        require(value.place.is_gpu_place() == use_gpu and np.isfinite(actual).all()
                and actual.tobytes() == state[name].numpy().tobytes(), "Exact native EMA tensor load")
    model.eval()
    return model


def author_blocks(model, logical_images, image_shapes, *, paddle, np, stats, synchronize):
    """Yield one stock paired box/mask export per image, retaining no dense prediction stack."""
    from ppdet.core.workspace import create
    from ppdet.metrics.coco_utils import get_infer_results

    use_gpu = paddle.device.get_device().startswith("gpu:")
    dataset = create("TestDataset")()
    dataset.set_images([path for _, path in logical_images], do_eval=False)
    reader = create("TestReader")(dataset, 0)
    binding = bind_reader_ids(logical_images, dataset.get_imid2path())
    require(len(dataset) == len(reader) == len(image_shapes), "One native reader batch per image")
    with paddle.no_grad():
        for (logical_id, hw), batch in zip(image_shapes.items(), reader, strict=True):
            reader_id, = batch["im_id"].numpy().reshape(-1).tolist()
            require(binding.get(reader_id) == logical_id, "Native reader order differs from the image-only roster")
            require(tuple(batch["image"].shape) == INPUT_SHAPE and batch["image"].dtype == paddle.float32
                    and batch["image"].place.is_gpu_place() == use_gpu
                    and not any(key.startswith("gt_") for key in batch), "Stock FP32 image-only reader batch")
            shape = batch["im_shape"].numpy()
            scale = batch["scale_factor"].numpy()
            expected_scale = [[INPUT_SHAPE[2] / hw[0], INPUT_SHAPE[3] / hw[1]]]
            require(shape.tolist() == [[640., 640.]] and np.allclose(scale, expected_scale, rtol=0, atol=2e-8),
                    "Native input resize and scale")
            synchronize()
            start = perf_counter()
            result = model(batch)
            synchronize()
            stats["forward_seconds"] += perf_counter() - start
            require(set(result) == {"bbox", "bbox_num", "mask"}
                    and result["bbox"].dtype == paddle.float32
                    and all(value.place.is_gpu_place() == use_gpu for value in result.values()),
                    "Actual author model outputs must be on the expected device")
            boxes = result["bbox"].numpy()
            counts = result["bbox_num"].numpy().reshape(-1)
            masks = result["mask"].numpy()
            require(boxes.shape == (NATIVE_ROWS, 6) and counts.tolist() == [NATIVE_ROWS]
                    and masks.shape == (NATIVE_ROWS, *hw) and (boxes[:, 0] == 0).all()
                    and np.isfinite(boxes).all() and np.logical_or(masks == 0, masks == 1).all(),
                    "Every native mask row is retained at the original image size")
            block = get_infer_results(
                dict(im_id=np.asarray([[logical_id]], dtype=np.int64), bbox=boxes, bbox_num=counts, mask=masks),
                {0: 1}, bias=0, save_threshold=0,
            )
            expected_scores = boxes[:, 1].tolist()
            require([row["score"] for row in block["bbox"]] == [row["score"] for row in block["mask"]] == expected_scores,
                    "Exact stock scores and candidate order")
            stats["images"] += 1
            stats["nonempty_masks"] += int(np.count_nonzero(masks.reshape(NATIVE_ROWS, -1).any(1)))
            del result, boxes, counts, masks
            yield block


def export_native_endpoint(path, logical_images, image_shapes, *, model, paddle, np, synchronize):
    """Export one full native arm as a digest-bound paired per-image stream."""
    if [image_id for image_id, _ in logical_images] != list(image_shapes):
        raise ValueError("The image path roster and original shapes must have identical order")
    stats = {"images": 0, "nonempty_masks": 0, "forward_seconds": 0.}
    started = perf_counter()
    blocks = author_blocks(model, logical_images, image_shapes, paddle=paddle, np=np, stats=stats, synchronize=synchronize)
    digest = write_author_stream(path, image_shapes, blocks, per_image=NATIVE_ROWS)
    return dict(**stats, predictions_sha256=digest, candidates_per_image=NATIVE_ROWS,
                rows=NATIVE_ROWS * stats["images"], elapsed_seconds=perf_counter()-started,
                GT_opened=False, official_TEST_opened=False)
