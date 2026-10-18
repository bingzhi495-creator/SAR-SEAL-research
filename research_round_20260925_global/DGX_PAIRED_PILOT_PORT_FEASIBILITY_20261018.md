# DGX 配对短试点移植可行性

<!-- U50_DGX_PILOT_ACTUAL_ACCEPTED_20261018_01 -->

后续实测：原U50试点函数、入口和Linux监督器已实际完成DGX两臂各3次更新，6次真实step，2图/28不同匹配实例；45秒，GPU分配峰值约2.17GiB，RSS约11.43GiB。接口资源可行性已获得两图范围证据；尚无全量、长训、AP或加速证据。[实际记录](U50_DGX_PAIRED_PILOT_ACTUAL_AND_FULL_INPUT_NEXT_20261018.md)。


<!-- U49_ACTUAL_TWO_IMAGE_EXPORT_ACCEPTED_20261018_01 -->

U49现已实际退出0并经Root接收：原FIT索引54/710两张图，共28个匹配实例；五字段逐值、逐字节、独立紧凑存储和真实CPU序列化重载检查均通过。worker耗时4.609秒，采样RSS峰值588648448 B；两个私有包合计43366454 B（约43.37 MB）。CUDA始终未初始化，没有模型更新或AP。741图/1661实例仅为原划分元数据，不能称已完成全量张量导出。DGX传输、远端运行和加速仍待实际检验。[核验摘要](results/U49_MATCHED_CPU_EXPORT_ACTUAL_20261018.json)。下文“尚未导出”是此前形成时的历史状态。


**SOURCE-ONLY，未执行。** 仅审普通发布源码，未读实际数据、权重、资格链或运行状态，未连接DGX、导出或传输文件。DGX的GB10/aarch64、Python3.12.14、Torch2.10.0+cu130、NumPy1.26.4来自Root已核事实。

可避开完整教师环境，但必须另建实验。原[U40 core](code/U40_FULL_METHOD_CONTINUOUS_TRAIN_DRAFT_20261017_01/train_core.py)426–432锁定Python3.8/Torch2.0.1/NumPy1.23.5，455–464又把版本纳入checkpoint身份；旧Adam/RNG不应跨版本接回。DGX任何更新均不计入U40，不推进其配对前沿，也不承诺速度或收益。

## 最小数据包

[U39._update](code/U39_FULL_METHOD_CROSS_IMAGE_RESOURCE_DRAFT_20261017_01/resource_probe.py)153–241实际消费如下CPU输入：

| 字段 | 类型/形状 |
| --- | --- |
| rgb400 | FP32 `[1,3,400,400]`，原U37真实RGB及原几何输出 |
| e / M | FP32 `[G,256]` / `[G,200,200]`，原GT-matched顺序、真实coarse logits |
| L / targets | FP32 `[G,H,W]`，原图native logits及原逐实例0/1目标 |
| metadata | 原index/image_id/query_ids/ann_ids、H/W/content/800画布、原来源和逐字段摘要 |

**matched-only足够，F可省。** 原F只由reader验证，未进新UNet；不能用教师F替代RGB。保留741图/1661原匹配GT，不搬每图100个L、不重匹配、不合并实例、不从L重建M。S0/类别不作为训练特征。

现有代码不证明独立包已经存在：[旧reader](code/N28_DENSE_ROUTING_TRAIN_SOURCE_ONLY_20261017_01/reader.py)130–175选取matched e/M/L并核GT摘要，[U37](code/U37_FULL_RGB_MATCHED_READER_DRAFT_20261017_01/rgb_matched_reader.py)87–153仍现场解码JPEG。须由Root后续实际逐图导出，逐字段核原reader值精确相等、GT/query身份与顺序、dtype/shape/digest；紧凑CPU存储避免view携带整份100-query storage。不能把继承来源当成新导出已验证。

理论字节基数：RGB为`741×3×400²×4`；matched M为`1661×200²×4`；L+FP32 targets为`8ΣG_iH_iW_i`。H/W未读，不给实测总大小。省去F理论减少`741×256×200²×4`字节；不是测得文件大小。

## 代码、初始化与最小差异

保留原17角色清单作谱系记录，但不能原样运行Windows控制。最薄数值组合是：

- [U34 shared_initialization.py](code/U34_FULL_PLAIN_INIT_DRAFT_20261017_01/shared_initialization.py)及其`SOURCE_SHA256`七角色：FULL的model_full/dsconv/head/feature包装、PLAIN的model1/feature包装、共同instance_adapter；对应普通文件见`code/U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/`和`code/UNETPP_FULL_INSTANCE_ADAPTER_DRAFT_20261017_01/`。
- 原`code/U35_REAL_IMAGE400_CHUNK1_DRAFT_20261017_01/image_geometry.py`，以及U39 `_update/_partition`、`code/U36_REAL400_CPU_SAVED_ACTIVATIONS_DRAFT_20261017_01/probe_body.py`指定辅助；不调用旧run/probe。
- 新增薄包reader和DGX试点worker，取代旧reader及运行接线；保持B1/400/FP32/query1、每图一次encode/AdamW、全G的1/G BCE+softDice。保留旧inactive/BN边界，不混入U46修复候选。

新鲜试点不需教师checkpoint、MMDetection/MMCV、GT JSON或已训练U40权重；原DSConv本路径为普通Torch运算。按U34 CPU共同104状态/相同adapter初始化，核实际配对相等；同seed不证明跨版本初值相同。如要机器间同初值，另导出pristine两臂state_dict并实核，仍不成为原链续训。

## 新环境必须核的接口

1. **加载：** 显式CPU、`weights_only=True`和纯tensor/primitive包。2.10默认已改变，旧隐式load不能照抄；不为兼容直接恢复旧optimizer/RNG或任意allowlist。[官方2.10加载](https://docs.pytorch.org/docs/2.10/generated/torch.load.html)、[view/序列化](https://docs.pytorch.org/docs/2.10/notes/serialization.html)
2. **CPU保存：** 保留仅encode的`save_on_cpu(pin_memory=True)`。U36:344–390观察器依赖pack/unpack属性及二元payload，2.10实际兼容尚待极小梯度/Adam对照；失败不能静默关闭offload。计数字节是累计事件，不是峰值/带宽。[官方接口](https://docs.pytorch.org/docs/2.10/autograd.html#torch.autograd.graph.save_on_cpu)
3. **配方：** AdamW仍3e-4/1e-4/(.9,.999)/1e-8、foreach/fusedFalse；实际核step/moments/device，BN与优化器跨图保留。新RNG另存，不声称跨版本bitexact。首次CUDA前显式关TF32、benchmarkFalse、deterministicTrue，无AMP/compile；新旧TF32控制不混用。[CUDA设置](https://docs.pytorch.org/docs/2.10/notes/cuda.html)、[复现边界](https://docs.pytorch.org/docs/2.10/notes/randomness.html)
4. **GB10资源：** 128GB是CPU/GPU共享DRAM；offload不等于释放同量物理内存或加速。记录RSS、available/swap、Torch allocated/reserved/峰值和设备观测，不能相加双计。4GiB allocator不是系统总占用上限；原Windows整卡8GiB门与句柄控制不能直接搬，`nvidia-smi`不支持字段不记为0。[NVIDIA UMA说明](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

下一最小动作由Root安排：仅导出原metadata已选54/710，固定两臂同序`[54,710,54]`各3步；全G参与、BN/Adam连续、初始零残差、梯度及释放实测，失败保留。通过后才考虑741包与独立完整训练。后续已完成[U49导出源码](code/U49_MATCHED_CPU_EXPORT_DRAFT_20261018_01/README.md)和[U50试点核心源码](code/U50_DGX_PAIRED_PILOT_BODY_DRAFT_20261018_01/README.md)，尚未实际导出、传输或运行DGX；远端入口和监督仍须闭合。2图短试点也不证明全人口、收敛、AP或创新成立。