"""Synthetic native postprocessor and mapped-row contract checks, no model or data."""
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("_full_capture_contract",Path(__file__).with_name("capture.py"))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
core = entry.load_core(Path(sys.argv[1]))
base,post = entry.setup(None,core)

import numpy as np
import torch
from pycocotools import mask as codec
from rfdetr.models.postprocess import PostProcess

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
raw = dict(
    pred_logits=torch.linspace(-3.,3.,328,dtype=torch.float32).reshape(1,164,2),
    pred_boxes=torch.full((1,164,4),.25,dtype=torch.float32),
    pred_masks=torch.linspace(-1.,1.,144*144,dtype=torch.float32).reshape(1,1,144,144).expand(1,164,144,144).contiguous(),
)
joint = post.aligned_query_rows(raw,(31,47),torch=torch,np=np,codec=codec,processor=PostProcess)
native = base.native_rows(raw,image_id=7,hw=(31,47),torch=torch,
    postprocess=PostProcess(num_select=100,upsample_masks_to_image_size=True),np=np,codec=codec)
reconstructed = entry.native_block(7,joint)
assert base.encoded(reconstructed) == base.encoded(native)
assert reconstructed["native_ranks"] == joint["native_ranks"]
assert reconstructed["unmapped_slot1_count"] > 0
assert any(b-a>1 for a,b in zip(reconstructed["native_ranks"],reconstructed["native_ranks"][1:]))
mask = (torch.nn.functional.interpolate(raw["pred_masks"][:,0:1],size=(31,47),
        mode="bilinear",align_corners=False)>0)[0,0]
expected = codec.encode(np.asfortranarray(mask.numpy(),dtype=np.uint8))
assert dict(size=expected["size"],counts=expected["counts"].decode("ascii")) == joint["all_rles"][0]
negative_or_zero = torch.tensor([-1.,0.])
assert torch.equal(negative_or_zero>0,negative_or_zero.sigmoid()>.5)
assert not torch.cuda.is_initialized()
print("PASS native joint class mapping, exact stock bytes, original rank gaps and strict zero-logit sign")

