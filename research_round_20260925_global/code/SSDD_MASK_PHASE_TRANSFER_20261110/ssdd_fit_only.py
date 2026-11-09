"""Isolate the SSDD FIT741 cohort while retaining the stock RF-DETR loader."""

from pathlib import Path
import re

from rfdetr.datasets.coco import CocoDetection
from rfdetr.training.module_data import GradAccumAlignedDataset, RFDETRDataModule


FIT_IMAGES = 741
ACCUMULATION = 4
FIT_VISITS = 744
FIT_PADDING = FIT_VISITS - FIT_IMAGES
ANNOTATIONS = "_annotations.coco.json"
JPEG_NAME = re.compile(r"[A-Za-z0-9_.-]+\.jpg")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fit_filenames(layout):
    require(layout.is_absolute() and layout.is_dir() and layout.resolve(strict=True) == layout,
            "Expected a canonical SSDD FIT-only layout")
    train = layout / "train"
    require(list(layout.iterdir()) == [train] and not train.is_symlink() and train.is_dir()
            and train.resolve(strict=True) == train, "SSDD FIT-only layout must contain only its own train directory")
    files = set()
    for path in train.iterdir():
        require(not path.is_symlink() and path.is_file() and path.resolve(strict=True) == path,
                "SSDD FIT files must be direct canonical files")
        require(path.name == ANNOTATIONS or JPEG_NAME.fullmatch(path.name),
                "SSDD FIT files must be the native annotation and direct JPEGs")
        files.add(path.name)
    require(ANNOTATIONS in files and len(files) == FIT_IMAGES + 1, "Expected the SSDD FIT741 file roster")
    return files - {ANNOTATIONS}


class FitOnlyData(RFDETRDataModule):
    def __init__(self, model_config, train_config, fit_layout):
        self.fit_layout = Path(fit_layout)
        self._fit_ready = False
        require(model_config.segmentation_head and not model_config.use_grouppose_keypoints
                and train_config.dataset_file == "roboflow" and Path(train_config.dataset_dir) == self.fit_layout
                and train_config.batch_size == 1 and train_config.grad_accum_steps == ACCUMULATION
                and train_config.num_workers == 0 and train_config.run_test is False,
                "Expected the SSDD FIT-only segmentation recipe")
        super().__init__(model_config, train_config)

    def setup(self, stage):
        require(stage == "fit" and self._dataset_val is None and self._dataset_test is None,
                "Only SSDD FIT setup is permitted")
        self._fit_ready = False
        require(Path(self.train_config.dataset_dir) == self.fit_layout, "SSDD FIT layout changed after construction")
        names = fit_filenames(self.fit_layout)
        # RF-DETR 1.12.0 otherwise builds validation as part of its stock FIT setup.
        self._dataset_val = object()
        try:
            super().setup(stage)
        finally:
            self._dataset_val = None
        native = self._dataset_train
        require(type(native) is CocoDetection and Path(native.root) == self.fit_layout / "train"
                and len(native) == FIT_IMAGES and len(set(native.ids)) == FIT_IMAGES,
                "Expected the native SSDD FIT741 dataset")
        images = native.coco.dataset["images"]
        require(len(images) == FIT_IMAGES and all(type(image["file_name"]) is str for image in images)
                and {image["file_name"] for image in images} == names,
                "SSDD FIT annotation names must match the isolated JPEG roster")
        self._fit_ready = True

    def train_dataloader(self):
        require(self._fit_ready and self._dataset_val is None and self._dataset_test is None,
                "Complete isolated SSDD FIT setup before creating a loader")
        require(getattr(self.trainer, "world_size", 1) == 1 and self._effective_grad_accum_steps() == ACCUMULATION,
                "Expected a single-process SSDD FIT loader with accumulation four")
        loader = super().train_dataloader()
        aligned = loader.dataset
        require(type(aligned) is GradAccumAlignedDataset and aligned._dataset is self._dataset_train
                and loader.batch_size == 1 and loader.num_workers == 0 and loader.drop_last
                and len(loader) == len(aligned) == FIT_VISITS and len(aligned._pad_indices) == FIT_PADDING,
                "Expected the stock 744-visit SSDD FIT loader with three alignment duplicates")
        return loader

    def _forbid_nonfit(self, *args, **kwargs):
        raise RuntimeError("SSDD DEV, test, and prediction loaders are forbidden during FIT")

    _build_test_dataset = val_dataloader = test_dataloader = predict_dataloader = _forbid_nonfit
