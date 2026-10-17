# 完整 UNet++ 主干静态接线核对

日期：2026-10-17。范围：仅阅读指定原始 Python 源码和普通公开研究说明；没有导入 Torch、构造或运行网络、读取原图或权重，也未访问当前 N28 的科学载荷、SOURCE 或资格链。本文记录源码中的计算关系及待接接口，不是完整结构、梯度、内存、历史训练或实例效果已经验证的结论。未修改原模型或已发布适配器。

下文 `D` 指 `D:/SAR/UNet++/`；`P` 指 `E:/SAR/seal/github_release_20261011/research_round_20260925_global/`。P 下引用均为公开源码副本，不执行其中的入口、载荷读取或资格检查函数。当前方法范围以 `E:/SAR/seal/refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_PLAN.md` 为准。

## 1. 直接依赖、构造和特征出口

`D/model_full.py:1–6` 直接依赖 Torch、NumPy、同目录 `dsconv.py` 的 `DSConv` 和 `head.py` 的 `CoordConv`。`D/dsconv.py:2–6` 仅继续导入 Torch、NumPy 和标准库；`D/head.py:1–6` 仅导入 Torch 和标准库。所读 DSConv 是 Python/Torch 实现，没有需要另行编译的 DSConv 扩展依赖。实际加载时仍需明确这两个同名模块来自上述文件，避免搜索路径中的其他 `dsconv` 或 `head` 被误导入。`dsconv.py:8` 会全局忽略 warnings，这是导入副作用，不是科学验证。

当前计划的卷积资源版本必须显式传入：

```python
UnetPlusPlus(
    num_classes=1,
    deep_supervision=False,
    input_size=400,
    use_x2_transformer=False,
    instantiate_x2_transformer=False,
    include_legacy_x4_transformer=False,
)
```

构造参数见 `D/model_full.py:274–292`；Transformer 分配见 378–387 行。`instantiate_x2_transformer=None` 会被改成 `True`，因此只设置 `use_x2_transformer=False` 仍会分配巨大矩阵。原文件的 `__main__` 使用 400 输入但未显式修改默认 `input_size=512` 和 Transformer 开关，不能直接充当该资源版本的验证入口。这是明确关闭旧层的配置，不是开启旧层版本的忠实复现。

`stage_0` 固定输入三通道（309 行）。按原 forward 的池化与转置卷积，400 正方形输入对应下列静态尺寸：

| 节点 | 静态形状，不含 batch |
| --- | --- |
| `x_0_0` | `64 × 400 × 400` |
| `x_1_0` | `128 × 200 × 200` |
| `x_2_0`，三分支经 `con2` 融合后 | `256 × 100 × 100` |
| `x_3_0` | `512 × 50 × 50` |
| `x_4_0` | `1024 × 25 × 25` |
| `x_0_4 = CONV0_4(...)` | `64 × 400 × 400` |

编码与方向融合见 392–418 行，嵌套解码见 435–463 行。400 可被 16 整除，原各级转置卷积在这一尺寸上能静态对齐；这不是已运行的 shape 测试。`CONV0_4` 使用前置 BN 模式，最后是卷积而非 sigmoid/ReLU，其特征可带符号。

当前 forward 在 465–472 行返回语义输出，未提供 `forward_features`。实例共享特征应取 463 行的 `x_0_4`。新版本宜显式从这里返回，不能把当前 forward 的一张二值语义图当作各实例输出。

## 2. 有效路径与预期无实例损失梯度的参数

在上述关闭 Transformer、且实例损失只使用 `x_0_4` 的合同下，进入特征路径的主要模块是：`stage_0`、`cbam_stage_0…4`、独立的 `stage_2`、两方向 DSConv 的被选卷积及偏移/归一化、`con2`、十条 `upsample_*` 边、六个 `cbam_stage_up_*` 的有效支路，以及 `CONV0_1…4`。这是静态祖先关系，不保证任意输入下每个梯度元素非零。

需要提前声明的整参数无梯度范围如下：

| 参数前缀或模块 | 原因与源码 |
| --- | --- |
| `CONV3_1.*`、`CONV2_1.*`、`CONV2_2.*`、`CONV1_1.*`、`CONV1_2.*`、`CONV1_3.*` | 295–302 行有声明；实际低分辨率解码在 439–454 行使用对应 `cbam_stage_up_*`，未调用这些普通模块。不要误排除实际使用的 `CONV0_1…4`。 |
| `cbam_stage_up_{1_1,1_2,1_3,2_1,2_2,3_1}.bn1.{weight,bias}` | `CBAM_block_up.forward` 的 261–263 行先计算 `bn1(x)` 和 ReLU，随后 `conv1(x)` 再次直接读取原输入，前一结果被覆盖。训练模式下其 running buffers 仍可能更新，不能写成“模块完全未运行”。 |
| `final_super_0_1.*` 至 `final_super_0_4.*` | 对仅返回 `x_0_4` 的实例接口，四个旧语义头均不属于实例损失祖先图。若沿旧 forward 且关闭深监督，前面三个不调用，第四个仍执行但其输出被实例接口丢弃；两种情况的计算和 BN 状态副作用不同。 |
| `ds_stage_2x.dsc_conv_y.{weight,bias}` | morph=0 仅使用 `dsc_conv_x`。见 `D/dsconv.py:48–64,79–89`。 |
| `ds_stage_2y.dsc_conv_x.{weight,bias}` | morph=1 仅使用 `dsc_conv_y`，同上。 |
| `encoder_x2` / `encoder_x4` | 本资源合同应不实例化；若保留但关闭，则另成为未使用参数，不能默认为零成本。 |

另外，DSConv 两个偏移分量中仅一个用于该方向，中心偏移固定为零；x 方向路径还存在 `detach().clone()` 与部分切片重写（`dsconv.py:166–180`），y 方向路径写法不同（234–244 行）。因此“整个参数张量有梯度”与“每个通道/元素都有非零梯度”不同；正式 B0 应核实际期望集合，不应事后临时忽略意外缺梯度，也不应以每元素非零作为旧结构必然满足的要求。

`ChannelAttention.pool` 的滤波核为 buffer，65 行计算结果未参与输出；`CBAM_block.pool` 也没有进入其 forward 返回路径。这些不是额外有效的可学习抗混叠模块。需分开报告分配参数、优化器参数、实际梯度/更新参数及 buffers。

## 3. DSConv 多图 batch 的具体索引问题

`D/dsconv.py:285–290` 将输入整理为 `[B*H*W, C]`；296–305 行构造每张图的展平 `base`。307–316 行随后执行：

```python
base_y0 = base + y0 * self.height
base_y1 = base + y1 * self.height
index_a0 = base_y0 - base + x0
index_c0 = base_y0 - base + x1
index_a1 = base_y1 - base + x0
index_c1 = base_y1 - base + x1
```

四个采样索引都把 batch 偏移抵消了，化简后只剩单图范围的坐标。319–322 行用这些索引读取已包含全部 batch 的展平输入，因此 **B>1 时后续样本的采样值来自第一张图的特征地址**。其偏移仍可由各自样本产生，故不意味着各输出必然完全相同；但已足以证明图像身份串扰。B=1 时 `base=0`，这一特定问题不触发。

当前一图一次共享主干、只对 query 分批的 MVP 可明确限制图像 batch=1。它不能据此宣称旧 DSConv 已通过多图批次隔离。若未来支持 B>1，应在独立版本修正并检验；不能改旧文件后追认历史版本。即使修正索引，训练态 BatchNorm 本身也有跨样本统计耦合，不能把 eval 下图像隔离与训练态 BN 性质混为一谈。本文未执行该修复或数值反例。

## 4. CoordConv 不构成 N² 注意力，但 hook 后继续执行有副作用

`D/head.py:15–22` 的 `AddCoords.pa` 是 `1×1 Conv → BN → ReLU → 1×1 Conv → BN → sigmoid`，45–47 行逐位置乘回输入。它没有空间两两相似度、矩阵乘或 `[HW,HW]` 张量；固定通道宽度下空间计算/存储量随 HW 线性增长，不是全空间 N² attention。BN 的统计依赖不等于 N² 注意力。

31–43 行从当前输入的 `x_dim/y_dim` 动态生成坐标，51–54 行拼接；`CoordConv` 在 66–81 行再执行 BN/ReLU/卷积。400 正方形没有固定 512 坐标阻断。坐标除数是维度减一，不能外推到空间维度为 1 的退化输入。

若只用临时 forward hook 保存 `CONV0_4`，仍继续执行原 backbone 尾部，就会继续构造坐标、计算旧 head，并在 train 态更新旧 head 的 BN buffers。丢弃返回值不会取消这些前向计算。建议下一独立适配版本真正返回特征；这里没有修改已经发布的 hook 版本，也未宣称旧 head 是当前资源失败的原因。

## 5. 公开 teacher 加载与几何事实

公开实际 loader 位于：

`P/multimodal_data_pilot_20260928/native_ordinal_two_mode_mask_adapter_learning_preparation_20261008/train.py:246–290`。

其读取关系为 `LoadImageFromFile(to_float32=True) → LoadAnnotations(with_bbox=True, with_mask=True) → Resize(scale=(800,800), keep_ratio=True) → PackDetInputs → model.data_preprocessor(..., training=True)`。Pack 的 metainfo 包含 `img_id,img_path,ori_shape,img_shape,scale_factor`。源码明确校验 `data['inputs']` 为 `[1,3,800,800]`，并另外复制样本、移除 `gt_instances/ignored_instances/gt_sem_seg` 后交给原生前向。因此不能把原 loader 描述成完全不读取标注；GT 被移出的是模型前向输入。

同文件 264–278 行另外用 `cv2.IMREAD_GRAYSCALE` 解码原 uint8 灰度，nearest resize 到实际内容尺寸后补零为 `gray800`。这一路是辅助灰度目标，不等于 teacher 的三通道归一化输入，不能直接拿来充当完整主干输入。

公开覆写配置位于：

`P/multimodal_data_pilot_20260928/native_coco_comparator_micro_preflight_preparation_20261008/run.py:101–139`。

它设置 `data_preprocessor.pad_mask=True`、`pad_size_divisor=32`，以及 `BatchFixedSizePad(size=(800,800), img_pad_value=0, pad_mask=True, mask_pad_value=0, pad_seg=False)`；声明 keep-ratio 800 resize、无数据增强。**确切 teacher mean/std 与颜色转换仍由官方模型配置继承，本次未核当前实际声明，留待 Root 补齐，不按常见配置猜填。** 这也不代表旧语义 UNet++ 训练采用相同归一化。

公开几何审阅源：

`P/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/training_path_inference_control_source_20261009/evaluate_common_cpu.py:246–260`。

其原尺寸为 `(H,W)`，比例 `s=800/max(H,W)`，实际内容尺寸为 `(int(H*s+0.5), int(W*s+0.5))`，`scale_factor=[actual_w/W, actual_h/H]`，最终画布为 800×800。应保留实际整数尺寸，不能只保留理想连续缩放比例。

## 6. 400 分支与实例读出的最小接口

旧原生字段捕获可参照前述 `train.py:197–243`：最终 `mask_embed` 前的 postnorm 查询名为 `h`，形状 `[1,100,256]`；`F` 为 `[1,256,200,200]`；十级 classes/masks 分别为 `[1,100,2]` 和 `[1,100,200,200]`。`h` 是 mask MLP 输入，不能与经 MLP 后的 mask embedding 不加说明地互换。

每图接线至少需要：

- 身份与几何：`image_id`、真实图像来源、`ori_shape`、实际 `img_shape`、`scale_factor`、`batch_input_shape=(800,800)`。
- 输入合同：实际三通道图像/teacher 输入的颜色、dtype、归一化、padding 声明，以及明确的 800→400 缩放规则。可以定义共用实际 teacher 归一化输入再缩小的候选，但这是待固定的新分支合同，不是本次已验证的数值路径。
- 实例合同：最终查询 h、原始 M200、原实际未二值化 native logits L、原 `prediction_query_ids`、类别和冻结 S0。不得用二值 RLE 或另一强头的掩膜冒充 L。
- 输出合同：共享 `U=[1,64,400,400]`，小头按查询生成 δ400；全部原查询均保留，query 分块不能重跑或改变图像主干。

准确旧渲染实现位于上述 `train.py:54–62`，以及公开 `training_path_inference_control_source_20261009/evaluate_common.py:258–280`：

1. 原 M200 双线性插值至 800×800，`align_corners=False`。
2. 取 `[:actual_h,:actual_w]`，去掉右侧/底部填充。
3. 再双线性插值至原 `(H,W)`，`align_corners=False`，得到实际 native logits L。

新 δ400 应独立采用 **400→800→内容裁剪→原图** 的对应渲染后加到 L。不能先把 δ400 降到 200，再调用只接受 `(1,1,200,200)` 的旧 `render_native`，然后声称等价；插值复合通常不相等。精确零残差、边缘/非方图几何仍须后续数值检验。

原公开预测用类别概率的 `topk(100, sorted=False)` 得到实际 `query_indices`，而不是假定导出行号就是 query ID（`evaluate_common.py:267–280`）。它原本按掩膜质量乘类别概率计算官方分数。当前固定 S0 结构对照必须保留基底分数和实际 query 映射，不能对新掩膜重算质量分数，再把排序变化归为纯几何收益。零残差与原 L 相加、按原 `>0` 二值化，是待验证的身份合同，不是本次已经完成的预测复现。

## 7. 当前结论

400 输入、完整卷积主体与共享 U 的静态接口可具体实现；没有发现需要另编译 DSConv 或 CoordConv 固定 512 尺寸造成的接线阻断。已明确的实质限制是旧 DSConv 多图索引串扰、仅 hook 后仍执行旧输出头的成本/状态副作用，以及必须提前声明的 inactive 参数集合。teacher 实际归一化、400 分支数值定义、有效梯度集合、真实资源成本、原图渲染和零残差身份仍未验证。本说明不替代 B0，不启动任何训练，不降低既定资源限制。
