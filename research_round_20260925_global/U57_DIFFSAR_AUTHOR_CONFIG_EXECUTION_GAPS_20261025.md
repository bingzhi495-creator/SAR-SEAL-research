# DiffSARShipInst作者配置的实际执行缺口

2026-10-25（北京时间）。在唯一DGX恢复训练运行期间，独立智能体通过GitHub连接限定只读作者仓库7份文件。固定提交：bdc3362372213bc8c3a1f6d5b7b74dc224d1b871。未SSH、构建、导入模型、下载权重或运行训练/评价；本文不是复现结果，也不是整个仓库所有传递导入的审计。

本次核查把“有作者代码，应补强对照”的笼统待办收窄为数据注册、初始化、真实环境与评价接线四个必要事项。不能直接运行一个带SSDD文件名的配置后便称完成作者方法复现。

| 核查部分 | 作者文件中的事实 | 对本项目执行的影响 |
|---|---|---|
| 数据名称 | 两份SSDD YAML使用coco_2017_train/val；builtin对应标准COCO目录 | 必须将统一实例数据明确注册，核单类映射、图像及标注，不依文件名猜数据身份 |
| 初始化 | 两份MODEL.WEIGHTS为空，入口调用resume_or_load | 作者确切初值与外部预训练尚未闭合；既不假定COCO初始化，也不无依据称随机初始化 |
| 骨架/候选 | 配置声明R101、四阶段deform、SCJE FPN；训练500 proposals、测试1000；采样步数1 | 与项目Mask2Former/R50载体不同，不能用项目原生67.071 AP充当该作者复现 |
| 输入/训练 | RGB512，训练声明crop/flip；batch1、AdamW、LR3.5e-5、WD1e-4、18560 iterations；衰减14848/17632 | 不能按次数直接写20轮，也不能和当前50轮贴标签即称同预算 |
| 评价 | MASK_ON、测试DETECTIONS_PER_IMAGE=100、TTA关闭；入口使用COCOEvaluator | 仍需真实segm、类别、空预测、原图几何和共同COCO helper一致性证据 |
| 环境 | setup编译C++/CUDA扩展，仅检查Torch最低版本 | 现有DGX ARM/GB10/Torch2.10的兼容性未证明；需要隔离环境实际构建及所用算子真实前后向，不凭静态代码判定能跑或不能跑 |

## 数据注册疑点的有限闭合

register_coco.py确实存在，但仅版权行与两个注册函数重导出；没有注册调用、SSDD路径、类别配置或覆写逻辑。已读train_net.py没有直接导入该文件、DatasetCatalog注册/移除或训练前路径重注册。已读builtin初始化分支执行register_all_coco(root)，root取DETECTRON2_DATASETS，默认datasets；两个通用数据名称对应coco/train2017、coco/val2017及标准annotation路径。

这排除了“该三行文件已隐藏配置SSDD”的具体猜测。它不证明作者实际用了COCO图像：作者可能把SAR数据摆在相同目录结构，需要数据证据。也不证明整个仓库所有传递导入均无其他注册副作用，本轮未扩大到全仓库执行/审计。

## 下一项实际工作

先在独立适配中明确数据注册与合法可核初值，再真实验证依赖/算子和统一segm评价。分别标明“作者声明配置适配”与“统一预算比较”；保留R101、候选预算、增强和成本差异，不悄改成R50后继续称完整作者复现。共同短预算只能作相同条件的诊断，论文主比较还需充分训练/收敛证据。

这项外部强对照尚未启动，当前唯一科学执行仍为固定主实验的一次性诊断恢复。该记录没有带来新AP，也不构成SOTA或创新成立证据。

来源：[README](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/README.md)、[train配置](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml)、[test配置](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-test.yaml)、[builtin](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/data/datasets/builtin.py)、[register_coco](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/data/datasets/register_coco.py)、[入口](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/train_net.py)、[setup](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/setup.py)。

## 公开默认入口的初始化补核

同一固定提交另限定补读4份文件：公开默认参数的resume开关关闭；DefaultTrainer将MODEL.WEIGHTS传给checkpointer，没有另选COCO/ImageNet。DetectionCheckpointer只有在路径非空时解析URL/获取路径，空值继续交给外部fvcore的Checkpointer.load。所核detector构造调用build_backbone，FPN构造转入build_resnet_backbone，未见另选预训练checkpoint的指令，部分卷积有显式Xavier初始化。

因此，已核公开默认入口没有指定COCO/ImageNet权重，不能把空WEIGHTS解释为自动加载COCO。完整“必保持纯构造初值”的结论仍缺实际fvcore版本/最终空值分支和更下游ResNet构造；当前依赖主分支不能替代作者当时环境。论文实际训练的初始化仍未知，不将其写成“作者从零训练”或“作者隐式使用COCO”。

另外，FPN在导入时调用ext_loader.load_ext加载carafe_forward/carafe_backward。真实环境验证除了作者setup的扩展，还必须检查这一CARAFE前后向依赖；仅普通Detectron2安装成功不足以证明整条执行链就绪。此次只读，没有运行或构建这些扩展。

补核来源：[默认入口](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/engine/defaults.py#L109)、[checkpoint分支](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/checkpoint/detection_checkpoint.py#L34)、[detector构造](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L238)、[FPN构造与扩展](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/fpn.py#L584)。
