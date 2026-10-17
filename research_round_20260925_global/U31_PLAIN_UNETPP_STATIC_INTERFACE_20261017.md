# 普通完整 UNet++ 主对照：静态接口与来源

日期：2026-10-17。仅阅读 D:/SAR/UNet++ 中指定源码、历史配置及既有普通结构说明；未导入 Torch/NumPy，未构造模型、读取权重/图像/标注/日志数值，也未读取当前运行的资格或 Ops 文件。本说明不是已完成的普通主干数值验证。Root 仍独占科学执行。

## 历史 clean 的实际类

实际文件为 `D:/SAR/UNet++/model_unetpp_clean.py`，类名 `CleanUNetPlusPlus`（35 行），119 行提供别名 `UnetPlusPlus = CleanUNetPlusPlus`。这是本地直接实现，文件仅依赖 Torch，不是 `segmentation_models_pytorch.UnetPlusPlus` 的同名转发。

`configs/baseline_hrsid_clean_unetpp400.yaml:27–35` 与 `configs/baseline_ssdd_clean_unetpp400_scale_full.yaml:33–41` 明确指定该 module/class、三通道、`base_channels=32`、`norm_type=group`、关闭深监督。`train_semantic.py:53–57` 按配置动态导入该类和 kwargs，再处理可选归一化替换。这确认历史 clean 配置指向哪个类，不追认任何既存权重必然由当前源码字节训练而来。

`CleanUNetPlusPlus` 的默认构造参数为 `num_classes=1, in_channels=3, base_channels=32, deep_supervision=False, norm_type='group'`。其五层宽度是 32/64/128/256/512（38–48 行）；`make_norm` 默认最多 8 组，可显式选 BatchNorm 或 affine InstanceNorm（6–16 行）。所有 `ConvBlock` 均为两次 `3×3 Conv(bias=False)→Norm→ReLU`（19–29 行）。下采样为 MaxPool2d(2)，上采样为按目标节点实际尺寸的双线性插值、`align_corners=False`，不先压缩通道（50–71 行）。

**已经有直接特征接口**：`forward_features(x)`（78–104 行）返回含 `x0_1/x0_2/x0_3/x0_4` 的字典。400 输入的最终特征为 `[B,base_channels,400,400]`；默认 `[B,32,400,400]`，末尾有 ReLU。直接取 `forward_features(x)['x0_4']` 即可旁路四个 1×1 语义头，不需 hook 或新增 factory。这里没有实际执行。

## 与 full 的差异不能混为模块效应

| 项目 | 历史 clean 默认 | `model_full.py` 卷积资源版本 |
| --- | --- | --- |
| 五尺度宽度 | 32/64/128/256/512 | 64/128/256/512/1024 |
| 归一化与块层序 | 默认 GN；两次 Conv→Norm→ReLU | BN 为主；编码复合块与解码块组织不同，`con2`/DSConv 还有 GN |
| 上采样 | 无参数双线性，拼接前保留深层通道数 | 十条独立 ConvTranspose2d，先将通道减半 |
| 最终共享特征 | `x0_4`，默认32通道，ReLU后 | `CONV0_4` 后64通道，末尾普通卷积，可带符号 |
| 额外结构 | 无 attention/DSConv/Transformer/坐标头 | 注意力复合编码、H/4方向融合、分层解码等 |

把 clean 设置为 `base_channels=64, norm_type='batch'` 可对齐五尺度宽度及主要归一化种类，但**不等于参数、计算、上采样或块层序匹配**；双线性保留深层通道还会使拼接输入宽度不同。本次未计算或运行参数统计，不给虚构的配平数量。历史 clean32 适合作为另一个明确的旧基线，不能把其直接与 full64 的差值全部归因于所加模块。

## 更贴近 full 的普通结构锚点

另一个已存在的直接普通实现为：

`D:/SAR/UNet++/archive/old_code/Unet++/model1.py`，类 `UnetPlusPlus`。

注意路径中是 `archive/old_code/Unet++/model1.py`，**不是** `archive/old_code/root_legacy_models/model1.py`；后者已有 Transformer/坐标等不同路径，不可按相同文件名替代。既有 `E:/SAR/seal/research_round_20260925_global/UNETPP_SELF_DESIGNED_STRUCTURE_RECONSTRUCTION_20261015.md:11–13` 也明确区分两者。

这个普通类的实际 ABI 只有 `UnetPlusPlus(num_classes, deep_supervision=False)`（35–40 行），没有 `input_size`、`base_channels` 或 `norm_type` 构造参数。其宽度固定为 64/128/256/512/1024；五个 stage 是普通双卷积 Conv→BN→ReLU 两次（21–28、56–60 行）；十个嵌套解码节点使用 BN→ReLU→Conv→BN→ReLU→Conv（12–19、42–54 行）；十条上采样边均为独立 `ConvTranspose2d(kernel_size=4,stride=2,padding=1)`（64–75 行）。没有 CBAM、SimAM、DSConv、Transformer 或坐标头。

它与 full 的尺度、独立上采样边及最高分辨率普通节点更直接对应。400 输入静态对应空间尺寸 400/200/100/50/25，`CONV0_4` 在 133–134 行产出 `[B,64,400,400]`；400 可被16整除，没有上述尺度拼接的静态尺寸冲突。它仍不是与 full 等参数/等计算的模型，也不能孤立证明某一复合模块的因果作用；但更适合作为本次“普通完整 UNet++”的主结构锚点。Root 已选择该候选，**不混称历史 clean**。

原类没有 `forward_features`；136–143 行继续执行语义输出。最小新包装应保持原 99–134 行逐节点计算、拼接顺序和所有共享模块不变，在 `x_0_4` 处返回，旁路四个 `final_super_0_*`。若使用同一个既有实例，应保留其模块/参数对象、训练模式与 requires_grad flags，避免暗中重建随机参数或冻结状态。原四个语义头不属于新的特征损失图，须单独声明，不应要求它们获得实例损失梯度。

测试参考可在 eval 模式下临时将原实例的 `final_super_0_4` 替为 Identity，让原 forward 返回真实 `CONV0_4`，并在 finally 恢复原 head；再与新特征包装逐位比较。临时 Identity 只作测试 oracle，不作为正式训练时反复改模块的实现。本轮仅准备一个延迟注入 Torch 的公共测试，不实现包装或执行数值。

当前确定的是源码接口与更合理的普通对照来源。400 实际前后向、梯度集合、共同初始化、完整参数/计算预算及与 full 的同输入渲染/实例头关系仍需 Root 后续独立验证；本说明不读取或改变当前 N28 任务。
