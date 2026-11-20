"""Exercise original author decoding of non-square query masks without data or weights."""
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("_target_test", HERE / "target_capture.py")
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)
base = target.load_source("author_export_reference.py", target.BASE_SHA)
capture = target.load_source("native_capture_reference.py", target.CAPTURE_SHA)
author_source = Path(sys.argv[1])
base.admit_source(author_source, Path(sys.argv[2]))
with TemporaryDirectory(dir=HERE, ignore_cleanup_errors=True) as tmp:
    base.prepare_export_codec(Path(sys.argv[3]), Path(tmp) / "CODEC")

    import numpy as np
    import torch
    from pycocotools import mask as codec
    from dfine_seg.infer.torch_model import TorchModel
    author = sys.modules[TorchModel.__module__]
    torch.set_num_threads(1)

    class Fixed(torch.nn.Module):
        def forward(self, value):
            return dict(pred_logits=torch.linspace(3., -3., 300).reshape(1, 300, 1),
                        pred_boxes=torch.tensor([.5, .5, .4, .6]).reshape(1, 1, 4).expand(1, 300, 4).clone(),
                        pred_masks=torch.full((1, 300, 160, 160), .75))

    model = TorchModel.__new__(TorchModel)
    model.input_size = (640,640)
    model.keep_ratio = model.rect = False
    model.channels = 3
    model.device = "cpu"
    model.model = Fixed()
    model.task = "segment"
    model.n_outputs = 1
    model.debug_mode = False
    model.conf_thresh = 0.
    model.labels_to_use = []
    model.apply_nms = True
    model.nms_iou_thresh = .70
    model.binarize_masks = True
    model.mask_threshold = .50
    for hw in ((73,121),(121,73),(800,800)):
        forward,nms,hook,old_nms = capture.bind_observers(model, author, torch)
        try:
            row = dict(image_id=73, original_hw=list(hw))
            prediction,audit,planes = capture.capture_row(np.zeros((*hw,3), dtype=np.uint8), row,
                model,author,torch,np,base,forward,nms)
            assert planes.shape == (1,160,160)
            assert len(audit["candidates"]) == len(prediction["coco"]) == 1
            assert prediction["coco"][0]["segmentation"]["size"] == list(hw)
            box = audit["candidates"][0]["box_xyxy"]
            assert abs(box[0] - .3*hw[1]) < 1e-3 and abs(box[1] - .2*hw[0]) < 1e-3
            assert codec.decode(prediction["coco"][0]["segmentation"]).shape == hw
        finally:
            hook.remove()
            author.nms = old_nms
print("Actual author CPU synthetic landscape, portrait, 800-square and below100 lineage passed")

