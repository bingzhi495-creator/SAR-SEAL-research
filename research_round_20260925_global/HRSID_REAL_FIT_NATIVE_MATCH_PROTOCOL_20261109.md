# HRSID 真实空目标与稠密 FIT 输入资格

从上一轮独立验收的 2,937 张官方 PNG FIT 及派生训练标注中，预先固定两张实际模型输入。沿用 576 分辨率、原生 scale jitter、水平翻转和 block 12 packed collate。按 image ID 升序，以 520100000 加零起始序号取种子。无目标组取首个被裁为空的 FIT 视图，稠密组取首个经原生直接缩放且完整保留 163 个实例的 FIT 视图。两个查询宽度复用同一份只存本地的张量。 [全量输入审计](HRSID_OFFICIAL_PNG_FIT_PIXEL_LABEL_AUDIT_PROTOCOL_20261109.md)

| 验证对象 | 固定条件和通过条件 |
| --- | --- |
| 图像与标注来源 | P109 核验后的官方 PNG 归档、9,260 条有效 FIT 训练标注和相同原生增强。CPU 选择记录 0/163 个目标。 |
| 查询对照 | 100 与 164 个查询每组，共 13 个训练组。固定 E24 初始化。164 组每一组内精确保持原有前 100 个参考点和查询特征，新增槽位按固定规则初始化。其它权重完全相同。 |
| 无目标损失 | 两个宽度均用批大小 1、原生训练模式 BF16。最后一层、辅助层和编码器分支均返回 0 个匹配，分类损失有限，框 L1、GIoU、掩膜 BCE 和 Dice 为 0，合成损失及梯度有限。 |
| 稠密匹配 | 使用同一 163 目标 FIT 视图。记录每个分支及匹配组的查询数、匹配数和未分配目标数。若两侧候选足量，100 组每组匹配 100 个，164 组每组匹配 163 个。两侧损失和梯度均有限。 |
| 执行与边界 | 两宽度分别以原生前向、匹配、损失和反向核验。每进程 GPU 分配上限 6 GiB，批大小 1，优化器和 EMA 的实际更新数均为 0。无 DEV、TEST 或 AP。异常保留原退出，修复后另锁。 |

Group DETR 在每组内独立执行一对一匹配，RF-DETR 的训练查询表按组排列。增大查询容量用于避免已实测的训练匹配上限。用这两例可以确认具体代码路径及分组预算，后续目标域训练仍需统一输入规则和查询宽度。正式 COCO maxDets 由评估协议单独固定。 [Group DETR 原文](https://arxiv.org/abs/2207.13085) · [RF-DETR 固定版本模型](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L150-L163) · [逐组匹配实现](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/matcher.py#L829-L851)

CPU 选择逻辑的两项合成测试已通过。实际张量选择和后续 CUDA 核验分别留存原进程结果。 [输入选择源码](code/HRSID_REAL_FIT_NATIVE_MATCH_20261109/select_native_fit.py) · [选择逻辑测试](code/HRSID_REAL_FIT_NATIVE_MATCH_20261109/test_select_native_fit.py)

## 真实输入已生成

CPU 原进程退出 0，用时 11.20 秒。已按上述规则保存 0 目标裁剪与 163 目标直接缩放各一份真实 FIT 张量。独立重读校验了张量包 SHA-256、两组 1×3×576×576 输入和对应 0/163×576×576 掩膜张量，163 张实例掩膜逐个非空。原张量仅留本地。CUDA 模型核验另行执行。 [输入选择实测](results/HRSID_REAL_FIT_CPU_INPUTS_ACTUAL_SAFE_20261109.json)

## CUDA 原生诊断的执行锁定

CPU 输入提交为 `80741886c069213c39e37f47591953a5a52b8204`，执行提交独立绑定，两侧 GPU 使用同一私有张量 SHA-256。沿用已验收 E24 在线权重。164 宽度在每组原 100 行后附加 64 行。新增参考点为零，新增查询特征用独立 CPU 随机种子 5201164。其余模型状态保持原值。每个宽度在独立进程中先跑空目标，再跑稠密输入。CPU 初始化复用 P87 入口，随后释放临时优化器与 EMA，GPU 流程只执行前向和反向并检查完整模型状态一致。匹配和扩容核验的四项合成测试通过。 [CUDA 诊断源码](code/HRSID_REAL_FIT_NATIVE_MATCH_20261109/probe_native_match.py) · [合成测试](code/HRSID_REAL_FIT_NATIVE_MATCH_20261109/test_probe_native_match.py)

