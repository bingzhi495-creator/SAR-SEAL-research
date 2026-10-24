# SAMRefiner 实际运行准备：依赖与未闭合条件

2026-10-25，北京时间。本文依据固定作者源码、官方发布元数据及一次DGX安装登记查询；没有安装、Torch/模型导入、真实图像读取、权重下载、SAM推理或新AP。主线仍是固定50轮PLAIN/FULL实验，不因依赖准备改动其配方或当前环境。

## 已取得的最小证据

[作者API桥接候选](code/U57_SAMREFINER_REAL_BRIDGE_DRAFT_20261025_01/README.md)的7项小NumPy测试和独立源码审阅通过；[固定候选适配器](code/U57_SAMREFINER_FIXED_CANDIDATE_ADAPTER_DRAFT_20261025_01/README.md)另有13项有限标准库测试。这些只检查替身后端/数组边界，不能证明实际SAM运行、图像对应、内部候选顺序或AP。

本次在DGX当前训练环境仅查询distribution元数据，未导入torch、numpy、cv2或FastGeodis：

| 分发包 | 安装登记版本 |
|---|---|
| torch | 2.10.0+cu130 |
| torchvision | 0.25.0+cu130 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0.86 |
| scipy | 1.15.3 |
| matplotlib | 3.11.2 |
| FastGeodis | 未找到该分发包登记 |
| segment-anything | 未找到该分发包登记 |

“未登记”不是实际import失败的证据，也不排除本地或vendored实现；登记版本不能证明C++/CUDA扩展ABI可用。Torch版本中的cu130也不能代替nvcc、toolkit或CUDA_HOME实际检查。

## FastGeodis 的官方构建与发布证据

固定官方源码提交为0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e。[构建依赖](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/pyproject.toml)列出setuptools、torch>=1.5.0和wheel；[setup.py](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/setup.py#L8)还使用pkg_resources及PyTorch的C++/CUDA扩展构建，Linux涉及OpenMP。是否构建CUDA路径取决于实际环境与构建条件，不能从已安装Torch的版本字符串直接断定。

本次官方[PyPI发布元数据](https://pypi.org/pypi/FastGeodis/json)查询显示当前1.0.5仅有源码tar.gz，所列历史发行文件也未见wheel；[官方GitHub releases](https://github.com/masadcv/FastGeodis/releases)未列上传资产。该事实限本次查询时点，不承诺未来发行情况。

已查[构建CI](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/.github/workflows/build.yml#L17)和[发布配置](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/.github/workflows/build-sdist-publish.yml#L16)未取得aarch64＋CUDA13专项测试证据。这表示兼容性尚未证实，不能推断一定不可编译。

## 作者SAM变体与接口

固定SAMRefiner提交为4bb7f95738c7a2e805f189210fb09e02a2197557。[README](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/README.md#L21)的基础SAM方案使用ViT-H及Meta官方sam_vit_h_4b8939.pth入口；HQ-SAM另列。文件名中的短串不能当作本次独立核验过的完整权重摘要，当前未下载或加载该权重。

作者要求使用仓库内segment-anything。[外层调用](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/sam_refiner.py#L133)依赖[vendor模型的forward_with_image_embeddings](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/segment-anything/segment_anything/modeling/sam.py#L134)及未阈值化掩膜路径；不能仅安装同名标准包便称作者复现。README示例的“sam-vit_h”与[实际registry键vit_h](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/segment-anything/segment_anything/build_sam.py#L47)存在差异，后续应按固定源码核对实例化入口。

## 下一项实际验证及范围

后续使用独立候选环境，先实际检查CPU架构/Python ABI、C++编译器/OpenMP、nvcc/toolkit/CUDA_HOME及目标GPU架构、Torch实际导入/扩展ABI、setuptools/pkg_resources/wheel。再依据构建日志与小型真实扩展调用验证CPU/CUDA路径，不能把安装命令退出0直接当作GPU功能可用。此步骤尚未执行。

实际模型试点还需核验固定作者代码及权重、模型状态、原图路径与真实尺寸/像素/掩膜对应、真实返回行对应和资源峰值。空掩膜策略须在看AP前明确；若采用显式透传，应标为适配策略并保持全部100候选、原类别和S0。候选桥接echo的ID不能替代内部对应核验。

本页不新增外部方法分数。SAMRefiner、DiffSARShipInst和SegS-YOLO的本项目同协议实例AP仍未取得，见[外部比较状态](U57_EXTERNAL_METHOD_COMPARISON_STATUS_20261025.md)。主实验最新计数以[README](../README.md)和[当前安全标量](results/U57_U60_ACTUAL_SCALARS_20261021.json)为准。
