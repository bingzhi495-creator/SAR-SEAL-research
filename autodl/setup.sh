#!/bin/bash
# AutoDL 环境安装（A100/A800 80GB）。AutoDL 基础镜像通常已带 torch+cuda；若版本不符按下面装。
set -e
# 若需指定 torch（cu118）：
# pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine "mmcv==2.1.0" "mmdet==3.3.0"
pip install pymupdf 2>/dev/null || true
python -c "import torch,mmcv,mmdet; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'mmcv',mmcv.__version__,'mmdet',mmdet.__version__)"
echo "OK. 接着改 carrier/configs/autodl_swinl_pseg.py 顶部 3 个数据路径, 然后 bash autodl/run.sh"
