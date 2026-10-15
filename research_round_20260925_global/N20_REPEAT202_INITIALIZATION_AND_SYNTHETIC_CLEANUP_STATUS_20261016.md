# N20 第二种子初始化与合成GPU清理探针

2026-10-16。状态：seed202610150202的源码初始化及一次合成GPU清理探针已实际完成并由Root核验；SAR训练与AP复验尚未开始。首201小幅正收益的[完整结果及预锁停止规则](N20_FIRST201_FULLDEV_QUALIFIED_RESULTS_AND_REPEAT_DECISION_20261016.md)保持。

初始化仅使用模型源码和随机状态，不读取SAR数据、缓存或已训练权重。CPU01、两线程、CUDA为空；新初态两臂各五张量/18,851参数，逐值相同、存储独立、空间权重全零；外部CPU随机状态不变。新digest为`baacb4ac28395e21db9bc821b0afb6c4c7dfd5b861e31d4af0c380ec581a1e5e`，与首201不同。原CPU179实际0（3.516秒），新pure180实际0后Root外部资格通过；模型训练未执行。

原failure24只留下最终CUDA清理断言，没有残留分配数值，不能仅凭它确认根因。静态比较发现旧成功N16额外清cuBLAS workspace。为区分残存state与库workspace，仅执行一次无SAR数据的实际GPU探针：用相同五张量head，两臂各8次AdamW更新，每步4096个合成点。输入/loss/logits限制在内层作用域，随后执行新202 tail中完全相同的清理代码片段，检查CPU历史和顺序副本保存。

| 阶段 | allocated字节 | reserved字节 |
|---|---:|---:|
| before_owned_release | 17653760 | 295698432 |
| after_original_model_optimizer_release | 17039360 | 58720256 |
| after_complete_owned_state_release | 17039360 | 58720256 |
| after_library_workspace_release | 0 | 0 |

在这个合成负载中，完整state释放未继续降低残留，调用当前Torch2.0.1+cu118的`_cuda_clearCublasWorkspaces`后allocated/reserved均为0，支持此负载的库workspace解释。它不能证明历史failure24唯一根因，也不能替代完整SAR训练的收尾验证；该私有API的行为限定在已测试环境。严格0门没有放宽。

原GPU180实际0（3.250秒），新pure181实际0后Root外部资格通过；原pending文件不改。当前181正常/26保留科学失败/207终态。两项执行均不产生新的SAR AP、性能收益、创新或TGRS资格。

新训练tail只改预锁seed202准入和训练后资源清理：先保存纯CPU历史和epoch_orders，再释放模型/优化器及完整state、清库workspace。保持实际训练数学、固定缓存、采样点/标签/权重/顺序、24轮、E8/E24、AdamW、17,784更新和72,843,264点曝光。新tail完整训练尚未执行；完整入口还须闭合当前前缀、首201继续许可、新初始化来源与所有历史失败，不能直接运行未绑定草稿。

公开源码为通用未绑定入口：[初始化](code/N20_REPEAT202_SOURCE_ONLY_INITIALIZATION_ENTRY_UNBOUND_20261016_01.py)、[合成清理探针](code/N20_REPEAT202_GPU_CLEANUP_PROBE_ENTRY_UNBOUND_20261016_01.py)、[训练tail](code/N20_REPEAT202_TRAIN_GPU_RUN_TAIL_UNBOUND_DRAFT_20261016_01.py)、[tail接线说明](code/N20_REPEAT202_TRAIN_GPU_RUN_TAIL_UNBOUND_DRAFT_20261016_01.md)。实际执行使用独立冻结副本及本机资格输入，公开范围不含这些载荷；不承诺仅凭草稿一键复现。
