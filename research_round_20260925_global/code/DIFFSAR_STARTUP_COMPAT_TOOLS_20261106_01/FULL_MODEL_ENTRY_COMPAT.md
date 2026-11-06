# DiffSARShipInst 完整模型入口：静态兼容审阅增量

日期：2026-11-06。作者固定提交：`bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`。本稿只记录源码、依赖和后续检查设计；审阅者未 SSH、未导入科学库、未构造或执行模型。完整训练、单批损失/反向和同协议 AP 均不能由本稿认定完成。

## 原仓库的具体启动问题与最小修改边界

Root 已实际尝试原作者包的 GPU 隐藏导入，首先在 `backbone/__init__.py` 的 `FPN_1p_5p` 导入处失败。这是 Python 包导入错误，尚不是模型或 CUDA 算子失败。下表其余问题来自固定归档静态核查，实际出现顺序仍以运行记录为准。

| 文件与问题 | 固定源证据 | 只在独立兼容副本采用的最小修改 |
|---|---|---|
| `detectron2/modeling/backbone/__init__.py:5` 导入不存在的 `FPN_1p_5p`、`FPN_1p_4p` | `fpn.py` 实际定义 `FPN:37`、`SCJE_FPN:292`，完整作者 detectron2/diffsarshipinst 源码中只有该导入提及两个缺失名称。 | 该行仅保留 `FPN, SCJE_FPN`。不补写或替换 FPN。 |
| 同文件 `:18` 导入 `.fan.FAN`，`:19` 导入外部 `detectron2_backbone...bifpn` | 已逐文件核验的作者归档不存在 `fan.py` 和 `detectron2_backbone` 包。SSDD/HRSID 训练 YAML 均选 `build_resnet_scje_fpn_backbone`，没有调用这些缺失扩展。 | 删除这两条未用导入；动态 `__all__` 无须另外修改。不创造 FAN/BiFPN 替代物。 |
| `diffsarshipinst/loss.py:20` 无条件导入 `poly_focal_loss_jit` | 作者归档没有自带 fvcore；该名称只在该导入和 `:239` 注释调用出现。实际 `:238` 调用 `sigmoid_focal_loss_jit`。官方 fvcore `0.1.5.post20221221` 源码分发中的 `nn/__init__.py`、`nn/focal_loss.py` 均无 poly 导出。 | 仅删除未使用的 poly 导入；保留实际 sigmoid focal loss 及全部损失权重。不能将此描述为把 PolyLoss 换成 FocalLoss。 |
| `detectron2/data/transforms/transform.py:46` 默认值 `Image.LINEAR` | Pillow 9.1 官方源码将 `LINEAR` 映射到 `BILINEAR`，其 `Resampling.BILINEAR=2`；Pillow 10.0 官方删除列表明确指定同义替代。已知目标环境 Pillow 12.3 已无旧别名。 | 仅改为 `Image.Resampling.BILINEAR`，保持插值选择。无需为一个别名降级 Pillow。 |

这些修改只触及三个 Python 文件。必须保留原归档、原源目录和已编译扩展；兼容副本记录逐项差异及其余文件一致性。`fpn.py`、ResNet/DCN、CARAFE、动态头、匹配器和损失公式均不在这组修改内。作者 `fpn.py:584–604` 的真实工厂仍用 ResNet 构造 `SCJE_FPN`，并保留 `FirestLevelUpSample` 和 `LastLevelMaxPool`。

源码：[backbone 导出](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/__init__.py)、[实际 FPN 与工厂](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/fpn.py#L584)、[作者损失](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/loss.py#L20)、[作者插值默认值](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/data/transforms/transform.py#L46)。

Pillow 同义证据：[9.1.0 映射](https://github.com/python-pillow/Pillow/blob/9.1.0/src/PIL/Image.py#L56)、[10.0.0 删除及替代列表](https://github.com/python-pillow/Pillow/blob/10.0.0/docs/deprecations.rst#L93)、[12.3.0 的 Resampling 定义](https://github.com/python-pillow/Pillow/blob/12.3.0/src/PIL/Image.py#L147)。fvcore 的精确版本证据来自[官方 PyPI 元数据](https://pypi.org/pypi/fvcore/0.1.5.post20221221/json)：50,217 字节源码分发，SHA-256 `f2fb0bb90572ae651c11c78e20493ed19b2240550a7e4bbb2d6de87bdd037860`；本轮只在内存读取这两个文本成员，未安装或执行该包。

## 真正需要的依赖与单次导入检查

作者包 `__init__.py` 会依次导入 config、detector、mapper、TTA、Swin。即使选择 R101，Swin 模块仍在导入时请求 timm 的 `DropPath`、`to_2tuple`、`trunc_normal_`；detector 会导入 `matplotlib.pyplot`；D2 config 导入链会请求 `cloudpickle`、YAML 和 OmegaConf。Torch/Torchvision、fvcore、iopath、yacs、NumPy、Pillow、pycocotools 和既有 MMCV 扩展也应沿当前已经确认的前缀使用，不重新解析安装到共享 base。

作者 [README](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/README.md) 和 [docs requirements](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/docs/requirements.txt) 未锁定 timm 版本。若本轮选 `timm==0.6.13`，应称本轮兼容版本，不能称精确恢复作者环境。该官方 tag 的旧导入路径确实导出全部三个接口；`DropPath(drop_prob, scale_by_keep=True)`、`to_2tuple(value)`、`trunc_normal_(tensor, std=.02)` 覆盖作者 Swin 的实际调用。helpers 已用 `collections.abc.Iterable`，没有旧 `torch._six` 依赖；该路径也导出 D2 可选 ViT/MViT 构造时所需的 `Mlp`。接口核对不能替代 Python 3.12/Torch 2.10 完整包导入实测。

官方依据：[timm 0.6.13 导出](https://github.com/huggingface/pytorch-image-models/blob/v0.6.13/timm/models/layers/__init__.py)、[DropPath](https://github.com/huggingface/pytorch-image-models/blob/v0.6.13/timm/models/layers/drop.py#L157)、[helpers](https://github.com/huggingface/pytorch-image-models/blob/v0.6.13/timm/models/layers/helpers.py)、[trunc_normal_](https://github.com/huggingface/pytorch-image-models/blob/v0.6.13/timm/models/layers/weight_init.py#L43)、[作者调用](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/swintransformer.py#L17)。完整 wheel 的声明依赖仍应按该版本元数据冻结；不能为了只取三个符号而谎报安装完整性。

不应机械补装无关依赖：固定作者 diffsarshipinst 和 detectron2 源码未发现 einops import；TensorBoard 仅在 `TensorboardXWriter.__init__` 请求；black 在 `LazyConfig.to_py` 内请求；Hydra 在 override/定位辅助路径内请求，普通 YAML `get_cfg/merge_from_file` 不必经过这些可选功能。目标已有 Matplotlib、SciPy、OpenCV 和 pycocotools，无须因通用 requirements 再次下载安装。后续若实际进口暴露新缺项，保留 traceback，再逐项处理。

建议独立兼容副本只做一次 GPU 隐藏的导入/config 检查：确认解析的是新副本与专用前缀；正常 `import diffsarshipinst`；`get_cfg()` 后调用作者 `add_DiffSARShipInst_config`，再合并作者目标训练 YAML；检查注册表中 meta architecture/backbone 存在及关键配置值。此阶段不构造模型、不读数据/权重、不前向、不反向、不实例化 Trainer/评估器/日志 writer。它通过只能确认入口和配置可用。

## 完整单批前向、损失、反向的最短路线

之后另行申请、限定并真实监督一次数值检查。直接用 D2 `build_model(cfg)` 构造已注册作者网络，避开 `train_net.py` 的 Trainer、评估器和 writer。该[构造函数](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/meta_arch/build.py#L16) 明确不自动加载权重；因此未经显式加载的单批只属初始化网络接口检查，不能证明复现论文训练配方。

作者单类配置真正由 `MODEL.DiffSARShipInst.NUM_CLASSES` 驱动，不能只改通用 `ROI_HEADS.NUM_CLASSES`。SSDD/HRSID 训练 YAML 已为单类，并指定 R101、SCJE_FPN、500 proposals、6 heads、调制 DCN；导入检查要核对它们。`add_DiffSARShipInst_config` 已建立 SCJE_FPN 节点，不需再创造新节点。

训练输入是长度为 batch size 的列表，每项有 `image` 和 `instances`。图像为三通道 CHW，按 SSDD 作者 RGB/像素均值标准差约定；Instances 保留真实 `(H,W)`，`gt_boxes=Boxes(xyxy)`、`gt_classes` 为从 0 开始的 int64 单类标签、`gt_masks=BitMasks(N,H,W)`。务必用 BitMasks 对象：作者 `loss.py:378` 在索引后取 `.tensor`，传裸掩膜 tensor 会违背接口。作者 mapper 已硬编码 `annotations_to_instances(..., mask_format="bitmask")`，与 YAML 中展示的 polygon 标签字段不能混淆。

`model.train(); loss_dict=model(batch)` 进入作者 `forward:521`，依次走归一化/骨干、扩散 proposal、完整动态头及 criterion；对返回损失之和反向。预先记录 seed、输入尺寸/实例数、配置、加载初始化、算子来源和资源限额，检查损失与所需梯度有限，并清理所有 GPU 引用。首个样例用非空有效实例，避免把空目标分支当成掩膜损失已覆盖。合成输入只能证明该合成输入下接口可运行；真实 FIT 数据映射和公平同预算训练仍需单独验证。

完整 R101/500 proposals/6 heads 网络不能沿用两个微型算子案例的 1 GiB GPU 上限，也不能为过限静默删掉骨干、head、DCN 或 CARAFE 后称完整模型通过。可先独立规划更小输入的完整网络诊断，但其尺寸限制必须公开，不能当作作者 512 训练协议的等价实验。

接口依据：[作者训练配置](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml)、[作者配置注册](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/config.py#L19)、[作者 forward/targets](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L521)、[作者 mapper](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/dataset_mapper.py#L119)。

结论边界：本组兼容设计保留目标 R101/SCJE 方法的实际数学路径，但包含显式源码补丁。后续结果应称“作者固定源码、公开最小启动兼容补丁下的执行”，在真实单批和训练验证完成前不称完整复现成功，更无同协议 AP/SOTA 结论。


发布说明：上文输入已按更正附注修正为RGB；SSDD为512、HRSID为1024。实际执行进展见[运行环境与D2算子结果](../../U57_AUTHOR_RUNTIME_AND_D2_OPERATOR_ACTUAL_20261106.md)。本稿其余部分保持静态设计范围，不将实际算子或导入通过解释为完整模型/AP完成。
