# DiffSAR 配置入口的补充核查与严格适配

日期：2026-11-06。固定作者提交 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`。本稿及自有 adapter 为静态/configuration 层产出，不代表完整网络前后向或训练通过。

Root 的实际记录表明：三文件五处启动补丁应用后，作者包已能正常导入；其余 1,118 个原文件 SHA 保持一致，原作者源未改。该次检查随后在 SSDD YAML 合并时报 `MODEL.BIFPN_1p_5p` 不存在，整次原失败必须保留，不能改记成功。

## 先复原作者真实入口

固定归档根目录的 `train_net.py:240–245` 顺序是 get_cfg、add_DiffSARShipInst_config、**add_model_ema_configs**、merge_from_file、merge_from_list、freeze；没有 add_deeplab 或其他 addX。必须按精确相对路径读取根入口，不能以文件名后缀匹配误取 projects 下同名脚本。

EMA 函数的精确位置为 `diffsarshipinst.util.model_ema.add_model_ema_configs`，源码 `:162–171` 只登记 `MODEL_EMA` 及 ENABLED/DECAY/DEVICE/USE_EMA_WEIGHTS_FOR_EVAL_ONLY/YOLOX。此前最短入口稿遗漏该调用，应在合并完整作者 YAML 前恢复。该遗漏独立于 BIFPN 报错：调用 EMA 本身不会登记 `MODEL.BIFPN_1p_5p`。

来源：[根训练入口](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/train_net.py#L236)、[EMA 配置注册](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/util/model_ema.py#L162)。

## BIFPN 节点为什么属于未使用元数据

两个作者训练 YAML 的第 56–65 行均含 `MODEL.BIFPN_1p_5p`，含五个叶字段：FUSE_TYPE=`sum`、IN_FEATURES=`[res2,res3,res4,res5]`、NORM=`' '`、OUT_CHANNELS=`256`、REPEAT=`1`。但两者真正选用的 BACKBONE.NAME 均为 `build_resnet_scje_fpn_backbone`。

固定作者 defaults、DiffSAR initializer、EMA initializer 均未登记 BIFPN 节点；作者 detectron2/diffsarshipinst/根训练 Python 全文也没有读取该配置字段的代码。已删除的外部 BiFPN import 指向归档不存在的包。现存真实 SCJE 工厂 `fpn.py:584–604` 只读取 MODEL.SCJE_FPN，调用 ResNet 和实际 SCJE_FPN，并保留作者上/下采样模块。不能因此补造一个 BiFPN 网络，或把 SCJE 换成其他实现。

本轮对两个 train YAML 的受限缩进 key 清单，与 defaults.py、DiffSAR config.py 和 EMA config initializer 的静态赋值路径作比较，计入 `CN({'ENABLED': ...})` 内联定义后，唯一未声明子树是 `MODEL.BIFPN_1p_5p` 及其五个叶字段。该静态清单有助于收窄缺口，但不替代真实严格 YAML merge，不能保证不存在后续类型或运行错误。

来源：[作者 SSDD YAML](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml#L53)、[作者 HRSID YAML](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-HRSID-train.yaml#L53)、[实际 SCJE 工厂](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/fpn.py#L584)、[作者模型 defaults](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/config/defaults.py)、[DiffSAR 配置注册](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/config.py#L19)。

## 最薄自有 adapter

`diffsar_config_compat.py` 不含任何 import、作者完整源码或私有路径。`register_inactive_bifpn_schema(cfg)` 仅通过现有 cfg 类型创建一个严格节点，登记上述五个有确定类型的叶字段；同名节点已存在就拒绝覆盖。没有全局或局部 `new_allowed`，不删除或修改作者 YAML，不修改任何已有 active 字段，也没有实现 BiFPN。

`verify_inactive_bifpn_schema(cfg)` 在 merge 后只读核对 meta architecture 必须为 DiffSARShipInst、backbone 必须为指定 SCJE 工厂，以及五个 metadata 的键、值和类型必须一致。偏离这些边界就拒绝继续。两函数均返回简短状态字典，明确 `implements_bifpn=False`。

调用方复原真实作者 initializer 后使用：

```python
cfg = get_cfg()
add_DiffSARShipInst_config(cfg)
add_model_ema_configs(cfg)  # from diffsarshipinst.util.model_ema
register_inactive_bifpn_schema(cfg)
cfg.merge_from_file(explicit_author_yaml)
verify_inactive_bifpn_schema(cfg)
```

这组步骤不自动构造模型、不加载数据或权重。调用方仍须核对原 YAML 的 RGB、SSDD 512/HRSID 1024，以及完整作者 R101、SCJE、500 proposals、6 heads、单类、SAMPLE_STEP=1 等有效配置；不能将不实施 BiFPN 的元数据登记宣传为新增架构。

`test_diffsar_config_compat.py` 仅用标准库和轻量配置容器验证：登记后所有既有 active 配置逐值保持；已有节点不覆盖；合并后 guard 对错误 backbone/meta architecture、额外字段、错误值及 bool 冒充整数拒绝，并保持只读。三项测试通过。没有导入 Detectron2、Torch 或执行作者代码；真实 CfgNode 严格合并仍由 Root 独立检查并保留实际结果。
