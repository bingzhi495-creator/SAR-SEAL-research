"""Check the native RF-DETR FIT/DEV reader against separately admitted SSDD inputs."""
import os
from pathlib import Path, PurePosixPath
import random


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _id_set(values, count):
    _require(type(values) is list and len(values) == count, "exact split ID list length")
    _require(all(type(value) is int for value in values), "integer COCO image IDs")
    ids = set(values)
    _require(len(ids) == count, "unique COCO image IDs within each split")
    return ids


def _direct_mask(mask_api, segmentation, height, width):
    if isinstance(segmentation, dict):
        _require(segmentation.get("size") == [height, width], "source RLE size")
        counts = segmentation.get("counts")
        _require(isinstance(counts, (list, str, bytes)) and bool(counts), "source RLE counts")
        encoded = mask_api.frPyObjects(segmentation, height, width) if isinstance(counts, list) else segmentation
    else:
        _require(isinstance(segmentation, list) and bool(segmentation), "source polygon parts")
        _require(all(isinstance(part, list) and len(part) >= 6 and len(part) % 2 == 0
                     for part in segmentation), "nondegenerate polygon representation")
        encoded = mask_api.merge(mask_api.frPyObjects(segmentation, height, width))
    decoded = mask_api.decode(encoded)
    _require(decoded.shape == (height, width) and bool(decoded.any()), "nonempty original-size source mask")
    return decoded != 0


def _check_target(torch, target, image_id, original_size, current_size):
    _require(target["image_id"].tolist() == [image_id], "unchanged native image ID")
    _require(target["orig_size"].tolist() == list(original_size), "unchanged original H/W")
    _require(target["size"].tolist() == list(current_size), "actual target H/W")
    labels, boxes, masks = target["labels"], target["boxes"], target["masks"]
    count = labels.numel()
    _require(count > 0 and labels.dtype == torch.int64 and tuple(labels.shape) == (count,)
             and bool((labels == 0).all()), "positive instance rows use label zero")
    _require(boxes.dtype == torch.float32 and tuple(boxes.shape) == (count, 4)
             and bool(torch.isfinite(boxes).all()), "finite boxes aligned with labels")
    _require(masks.dtype == torch.bool and tuple(masks.shape) == (count, *current_size)
             and bool(masks.flatten(1).any(dim=1).all()), "one nonempty bool mask per instance")
    _require(tuple(target["area"].shape) == tuple(target["iscrowd"].shape) == (count,)
             and bool(torch.isfinite(target["area"]).all()) and bool((target["iscrowd"] == 0).all()),
             "aligned noncrowd instance fields")
    _require(all(value.device.type == "cpu" for value in (labels, boxes, masks)), "CPU targets")
    return count


def _check_sample(torch, mask_api, dataset, split_root, sample, role):
    image_id = sample["image_id"]
    index = dataset.ids.index(image_id)
    metadata = dataset.coco.loadImgs(image_id)[0]
    relative = PurePosixPath(metadata["file_name"])
    _require(not relative.is_absolute() and ".." not in relative.parts and bool(relative.parts)
             and "\\" not in metadata["file_name"], "relative native image filename")
    image_path = split_root.joinpath(*relative.parts).resolve(strict=True)
    _require(image_path == Path(sample["image_path"]) and image_path.is_file(), "exact admitted sample path")

    height, width = metadata["height"], metadata["width"]
    _require(type(height) is int and type(width) is int and height > 0 and width > 0, "original COCO H/W")
    image, scales = dataset._decode_image(image_id)
    _require(image.mode == "RGB" and image.size == (width, height) and scales == (1.0, 1.0),
             "native original-size RGB decode")
    annotations = dataset._load_target(image_id)
    active = [row for row in annotations if row.get("iscrowd", 0) == 0]
    _require(all(row["category_id"] == 1 for row in active), "source ship category one")
    _, original = dataset.prepare(image, {"image_id": image_id, "annotations": annotations})
    raw_count = _check_target(torch, original, image_id, (height, width), (height, width))
    _require(raw_count == len(active), "all chosen noncrowd rows survive original box validation")
    for row, native_mask in zip(active, original["masks"]):
        expected = _direct_mask(mask_api, row.get("segmentation"), height, width)
        _require(torch.equal(native_mask, torch.from_numpy(expected)), "original mask equals direct COCO raster")

    image_tensor, transformed = dataset[index]
    _require(image_tensor.device.type == "cpu" and image_tensor.dtype == torch.float32
             and tuple(image_tensor.shape) == (3, 576, 576) and bool(torch.isfinite(image_tensor).all()),
             "finite native CPU image tensor at 576")
    final_count = _check_target(torch, transformed, image_id, (height, width), (576, 576))
    _require(final_count <= raw_count and (role == "train" or final_count == raw_count), "native retained instances")
    boxes = transformed["boxes"]
    _require(bool(((boxes >= 0) & (boxes <= 1 + 1e-6)).all()) and bool((boxes[:, 2:] > 0).all()),
             "normalized positive-size cxcywh boxes")
    return dict(image_id=image_id, original_hw=[height, width], transformed_chw=[3, 576, 576],
                original_instances=raw_count, transformed_instances=final_count, masks_equal_direct_coco=True)


def run_probe(dataset_root, model_config, train_config, expected_ids, samples):
    """Use exact train/valid ID lists and one caller-admitted positive per split.

    samples maps each role to {image_id, image_path}. The path is the canonical
    resolved image file, including when the split view uses an admitted link.
    """
    roles = {"train", "valid"}
    _require(type(expected_ids) is dict and set(expected_ids) == roles, "exact FIT and DEV membership roles")
    _require(type(samples) is dict and set(samples) == roles, "one frozen positive per role")
    members = {"train": _id_set(expected_ids["train"], 741), "valid": _id_set(expected_ids["valid"], 187)}
    _require(members["train"].isdisjoint(members["valid"]), "disjoint FIT and DEV image IDs")
    for role, sample in samples.items():
        _require(type(sample) is dict and set(sample) == {"image_id", "image_path"}
                 and type(sample["image_id"]) is int and sample["image_id"] in members[role]
                 and type(sample["image_path"]) is str and Path(sample["image_path"]).is_absolute(),
                 "fixed positive sample identity and path")
    root = Path(dataset_root)
    _require(root.is_absolute() and root.resolve(strict=True) == root and root.is_dir(), "canonical admitted data root")
    for role in roles:
        annotations = root / role / "_annotations.coco.json"
        _require(annotations.is_file() and annotations.resolve(strict=True) == annotations,
                 "ordinary native split annotation file")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only worker environment")

    import torch
    from pycocotools import mask as mask_api
    from rfdetr.config import AugmentationBackend, MultiScale, RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.datasets.coco import CocoDetection, _train_split_cat2label
    from rfdetr.training.module_data import RFDETRDataModule

    _require(not torch.cuda.is_initialized(), "CUDA context remains absent at entry")
    _require(type(model_config) is RFDETRSegNanoConfig and model_config.resolution == 576
             and model_config.num_channels == 3 and model_config.num_classes == 1
             and model_config.segmentation_head is True and model_config.use_grouppose_keypoints is False,
             "fixed Nano single-class mask recipe")
    _require(type(train_config) is SegmentationTrainConfig and train_config.dataset_file == "roboflow"
             and Path(train_config.dataset_dir) == root and train_config.square_resize_div_64 is True
             and train_config.augmentation_backend == AugmentationBackend.TV
             and train_config.multi_scale is MultiScale.OFF and train_config.scale_jitter is True,
             "fixed FIT/DEV reader and transform recipe")
    _require(_train_split_cat2label(root) == {1: 0}, "available train-derived category mapping")
    data = RFDETRDataModule(model_config, train_config)
    data.setup("fit")
    datasets = {"train": data._dataset_train, "valid": data._dataset_val}
    for role, dataset in datasets.items():
        _require(type(dataset) is CocoDetection and Path(dataset.root) == root / role,
                 "actual native Roboflow COCO split")
        _require(_id_set(dataset.ids, len(members[role])) == members[role], "exact admitted split image IDs")
        _require(dataset.cat2label == {1: 0} and dataset.label2cat == dataset.coco.label2cat == {0: 1},
                 "native label mapping round trip")

    random.seed(5201)
    torch.random.default_generator.manual_seed(5201)
    observations = {role: _check_sample(torch, mask_api, dataset, root / role, samples[role], role)
                    for role, dataset in datasets.items()}
    _require(not torch.cuda.is_initialized(), "CUDA context remains absent at exit")
    return dict(status="NATIVE_SSDD_DATA_SANITY_COMPLETE_PENDING_ROOT", qualified=False,
                split_images={role: len(ids) for role, ids in members.items()}, samples=observations,
                cat2label={1: 0}, label2cat={0: 1}, augmentation_seed=5201,
                model_executed=False, training=False)
