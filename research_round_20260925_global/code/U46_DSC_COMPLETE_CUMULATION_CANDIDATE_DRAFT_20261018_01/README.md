# U46：DSC 完整累计候选，未执行

仅新建候选源码，不修改 U33 原件或任何已冻结训练来源。基于普通发布副本 `code/U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/dsconv.py`：15207 bytes，SHA256 `b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0`。

只有三处文本变化，其余字节（包括换行）保持：

1. 第166行：`y_offset.detach().clone()` → `y_offset.clone()`。
2. 第177行、第241行：`range(1, center)` → `range(1, center + 1)`。

这是已知 Dynamic Snake Convolution（DSC）累计坐标机制的一致性修复候选，不是新算子或创新声明。Root 已报告接受的 U45 原始坐标诊断：B1/K9/1×1，morph0 的 AD 为 `[0,1,2,3,0,3,2,1,0]`、FD 为 `[1,1,2,3,0,3,2,1,1]`；morph1 两者均为后一向量。本任务没有读取该实际载荷或重复执行诊断。

候选意图是中心偏移固定为零、两侧全部九点按距离向外累计。对选定变形坐标之和、`extend_scope=1`，两个方向选中九通道的预期标量 VJP 与中心有限差分均为 `[1,2,3,4,0,4,3,2,1]`，未选的另一半通道为零。此预期尚未在候选上运行；不代表双线性插值、全 DSConv、完整网络、资源门或 AP 已验证，也不扩展为任意核大小正确性的保证。

本次只做字节级三处替换、逆变换核对、原件未变确认及 AST 解析；没有导入 Torch/NumPy、运行测试或数值、读取数据/权重/实际控制链。候选改变前向端点累计及梯度，不能覆盖旧冻结源码、接续旧训练时静默换入，或将新旧结果合并。后续探针、执行来源和实际运行由 Root 另行负责。

候选 `dsconv.py`：15206 bytes，SHA256 `321e9f23681d1da17ad2ecec008e4a58f64b66df4ad2bf78b68309a7fae820ea`。


The CPU entry reuses the U45 coordinate diagnostic admission/report schema, with a distinct phase, exact U46 source pins and output directory. The probe records all measurements regardless of agreement; `candidate_expected_pattern_observed` compares the predeclared complete-accumulation VJP. The reused mismatch status string is a compatibility label, not an original-source identity claim. Source identity is closed by the entry. No U40 training source, checkpoint or recipe is changed.
