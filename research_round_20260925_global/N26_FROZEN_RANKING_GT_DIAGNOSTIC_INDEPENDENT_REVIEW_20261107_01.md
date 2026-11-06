# N26 冻结排名 / GT 敏感性诊断：独立有限静态复核

2026-11-07。结论：在下列指定源码接口范围，**未确认需要阻止执行的具体缺陷**。这不是 COCO/API 数值通过、实际输入闭合、CPU 资源合格或科研结果认证。

**冻结范围**：`N26_FROZEN_RANKING_GT_DIAGNOSTIC_DRAFT_20261107_01/worker.py`，21,167 bytes，SHA256 `20f01f1b30e4522849d7ab4fb00836cad9872541144cd3e05321296f11c865cd`；同目录 README；独立普通发布副本中 CPU COCO adapter v1 的 `segm_copy` / `evaluate_population`；独立 locator ledger 的 schema、GT/R/三种子质量结果及 audit 的角色和字节身份 metadata。没有跟随 metadata 中的数据或资格引用。

- **整数先行闸门**（worker 183–208）：顺序保留共同 R 和三个预指定种子；四个整数 case 先完成，再逐一与冻结 AP 以绝对 `1e-8` 百分点比较。任一不符即抛错，所有 float case 保持未尝试；计算中异常也不会进入 float。没有更改阈值、重试或挑种子路径。
- **冻结预测完整性**（69–91、98–116）：限定四份完整 18,700 条、每图 100 条、187 个原图 ID 和已映射 category1。逐记录比较除 score 外所有字段及原位置，RLE 尺寸/压缩类型、bbox 和有限 score 均检查；score 原样传入，不重乘 qhat、不归一化、不排序、不重映射。三个 audit 只读字节身份不解析，其乘法正确性没有被新证明。ledger 角色要求三个固定 E24 / `S0*qhat` 来源，执行副本按原角色 size/SHA 闭合。
- **GT 与工作状态隔离**（43–67、93–96）：原始 JSON 按图像/标注 ID 比较，图像及非几何 dataset/annotation 字段一致，仅允许 annotation 的 segmentation/bbox/area 不同。每个 case 新建 COCO，给 dataset 独立 deepcopy 后 createIndex，避免 `_prepare` 改写向其余 case 或原始 GT 泄漏。实际两份 GT 是否满足这些条件尚未读取验证。
- **原 API 与配对统计**：只调用固定普通 helper 的纯 `evaluate_population`，不调用 native main 或其 37050 资格入口。helper 全量 deepcopy 预测并只去 bbox，原 COCO loadRes / COCOeval(segm) 对完整187图使用 category1、maxDets[1,10,100]，无自写 AP 或分片均值。每种 GT 都保留共同 R、三种子及各配对差值；三个差值全部有定义才算均值和 n−1 样本 SD，不用部分种子替代。
- **失败记账**（183–208、290–298）：evaluate 成功返回后先保存 math completed，再验证 summary；有效返回被确认后才标 AP computed。后续 case、时限或末尾源/输入复核失败，已保存的有效指标不被清空，整体失败退出且始终 qualified=false。新输出原子替换；外部硬杀仅可能留下最后一个中间报告，不能视为完成。

**未执行的范围**：没有重跑作者 10 项替身测试，没有科学库导入、SSH、模型/AP 执行，未读取/哈希真实预测、GT、audit 数值正文、CP、active SOURCE 或真实资格链。未核实历史编码器来源、候选 float 的官方/物理真值地位或真实 COCO ABI。Root 仍须另锁原 CPU owner、预算、库来源、结束输入及原始终态并做实际后核验；静态报告及 worker 的外部条件布尔值均不授予这些资格。当前结论只适用于上述冻结版本和指定接口。
