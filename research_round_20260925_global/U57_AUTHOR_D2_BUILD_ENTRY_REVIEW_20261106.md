# 作者 _C 的最小构建入口与运行依赖边界

2026-11-06。固定作者提交 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`。本次只读完整 setup.py、setup.cfg、初始化链和 PyTorch 2.10 构建入口；未 SSH、下载归档/安装、科学导入、编译或测试。Root 负责实际操作和原始资格。下列是新的执行事实，不重复 CARAFE 已通过的算子检查。

## 先构建扩展可以不安装整套作者运行依赖

作者 setup.py 顶层直接依赖标准库、setuptools 和 Torch；版本读取仅按文本解析 detectron2/__init__.py，没有导入 detectron2。get_extensions 仅进入 Torch 的扩展构建器、文件列表与 CUDA 构建选择。setup.cfg 只有 isort/mypy 配置，未定义构建依赖或命令别名；已核仓库根目录没有 pyproject.toml。

因此可在专用构建副本/venv 中先执行 `python setup.py build_ext --inplace`，只生成作者 `_C`，不必先运行 pip install 作者包或先满足 fvcore/iopath/yacs 等完整运行依赖。install_requires 是安装元数据，不等于该命令在执行前会导入或安装其中所有包。作者也没有声明 setup_requires；所核 setuptools 80.9.0 对此类即时获取只处理 setup_requires，而非直接把整份 install_requires 当作构建依赖。

最低 Python 侧是可工作的 setuptools、已安装 Torch 及其自身导入依赖；Torch 2.10 构建器直接使用 typing_extensions，TorchVersion 使用其内部 vendored packaging。作者额外声明的独立 packaging 不能因此误说成所有最小编译路径均会先导入的外部包。Ninja 是 Torch BuildExtension 的默认可选后端，已存在时应确保专用 venv/bin 在 PATH；需要真实 C++ 编译器、Python/Torch 头、链接库及所选 CUDA 13 开发工具链。**作者没有锁定 setuptools 或 wheel 的版本；build_ext 本身也不是 bdist_wheel，不要求先造作者 wheel。** 如需安装构建工具，仅写入专用环境并保存实际版本，不能把工程选定版本伪称作者声明。

这是旧仓库的一次限定原入口构建，不表示完成作者包安装、数据适配或模型复现。新环境显式借用 base Torch/MMCV site-packages，应准确称依赖复用，不能称完全隔离。

来源：[完整作者 setup.py](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/setup.py)、[setup.cfg](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/setup.cfg)、[PyTorch 2.10 构建器导入](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/cpp_extension.py#L22)、[内部版本解析](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/torch_version.py#L4)、[setuptools 80.9.0 依赖获取分支](https://github.com/pypa/setuptools/blob/v80.9.0/setuptools/__init__.py#L52)。实际安装的 setuptools 仍由 Root 记录，本文未假定等于该核查版本。

## 两个源码副作用必须如实记录

即使只调用 build_ext，setup 仍会调用 get_model_zoo_configs，在该独立 checkout 内生成或更新 model_zoo/configs 的链接；不能将所有生成文件出现都误认成手改数学源码。BUILD_NIGHTLY 若置为 1 会改写版本文件，应显式保持为 0，并让 D2_VERSION_SUFFIX 为空。原 archive/固定提交保持，构建仅在独立副本中进行。

来源：[版本分支](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/setup.py#L17)、[配置生成](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/setup.py#L110)。

## 构建成功与正常 import detectron2._C 需要分别核验

后续正常导入扩展会先执行 detectron2/__init__.py，再进入 utils/env.py。该链直接导入 NumPy/Torch，并在环境配置中要求 fvcore 和 PyYAML；检查最低 fvcore 0.1.2、PyYAML 5.1，OpenCV 只在可导入时配置。**部署仍应遵循 setup 更窄的 fvcore 0.1.5 系列约束**，不能只拿初始化链更宽的断言代替作者运行需求。该发现解释了为何可以先编译，之后却仍可能因缺少 Python 运行依赖而不能常规导入。

作者完整 install_requires（不包含可选 all/dev extras）是：Pillow>=7.1；matplotlib；pycocotools>=2.0.2；termcolor>=1.1；yacs>=0.1.8；tabulate；cloudpickle；tqdm>4.29.0；tensorboard；fvcore>=0.1.5,<0.1.6；iopath>=0.1.7,<0.1.10；dataclasses 仅 Python<3.7；omegaconf>=2.1；hydra-core>=1.1；black；packaging。当前 Python 3.12 不应无条件安装 dataclasses 回移植包。

可核的候选依赖来源：fvcore 0.1.5.post20221221 满足声明范围，PyPI 提供 50217 字节 sdist，SHA256 为 f2fb0bb90572ae651c11c78e20493ed19b2240550a7e4bbb2d6de87bdd037860；该版本没有登记现成 wheel，且 JSON requires_dist 为空不能被解释为“没有依赖”。iopath 0.1.9 满足范围，提供 py3-none-any wheel，27367 字节、SHA256 为 9058ac24f0328decdf8dbe209b33074b8702a3c4d9ba2f7801d46cb61a880b6e；登记依赖为 tqdm、portalocker，另有旧 Python 的 dataclasses 条件项。它们是可尝试的固定候选，不是已在本 DGX 运行验证的组合；实际安装/依赖闭合仍由 Root 完成，仅进入专用 prefix。

来源：[D2 初始化](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/__init__.py#L3)、[运行环境门](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/utils/env.py#L85)、[fvcore 固定 PyPI 元数据](https://pypi.org/pypi/fvcore/0.1.5.post20221221/json)、[iopath 固定 PyPI 元数据](https://pypi.org/pypi/iopath/0.1.9/json)。

## 对 Root 当前构建配置的结论

GPU 隐藏、FORCE_CUDA=1、显式 CUDA_HOME 与实测架构、限定 MAX_JOBS 的路线符合已核构建选择；先记录实际 NVCC 版本及所用编译器，再尝试原源码。先不为猜测的 Torch API 问题修改 DCN/CARAFE；真实失败后只修已定位的接口。该阶段的产物是扩展构建证据，不是数值有效性、完整网络训练或 AP。


## 实际执行更新

[原作者扩展现已构建成功](U57_AUTHOR_D2_BUILD_ACTUAL_20261106.md)，未安装整套训练依赖或修改作者数学算子；普通包导入、所需真实算子和完整模型仍待验证。
