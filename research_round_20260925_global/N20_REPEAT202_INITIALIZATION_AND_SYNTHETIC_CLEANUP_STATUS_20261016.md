# N20 第二种子初始化与合成GPU清理探针

**最新状态（2026-10-16，覆盖下文历史快照）：** 第二种子202610150202原训练55247实际退出0，新纯终态前缀182实际0先于报告读取。两臂24轮/17,784更新完成并保存E8/E24；独立CPU182原任务97964已启动，仍待终态核验与完整DEV评价。首201主差值仍+0.197741 AP点；没有第二种子AP或重复收益结论。当前182正常/26保留科学失败/208终态，活动CPU不计完成。[最新状态](N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。

2026-10-16。状态：seed202610150202的源码初始化及一次合成GPU清理探针已实际完成并由Root核验；同配方24轮正式训练任务已启动（原supervisor55247），未读取活动批训练进度；第二种子AP复验尚未执行。首201小幅正收益的[完整结果及预锁停止规则](N20_FIRST201_FULLDEV_QUALIFIED_RESULTS_AND_REPEAT_DECISION_20261016.md)保持。

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

新训练tail只改预锁seed202准入和训练后资源清理：先保存纯CPU历史和epoch_orders，再释放模型/优化器及完整state、清库workspace。保持实际训练数学、固定缓存、采样点/标签/权重/顺序、24轮、E8/E24、AdamW、17,784更新和72,843,264点曝光。新tail已接入冻结的正式任务；其正常结束与完整端点仍待核验。当前入口绑定首201继续许可、新初始化及所有历史失败；未绑定发布组件不能直接替代完整入口。

公开源码为通用未绑定入口：[初始化](code/N20_REPEAT202_SOURCE_ONLY_INITIALIZATION_ENTRY_UNBOUND_20261016_01.py)、[合成清理探针](code/N20_REPEAT202_GPU_CLEANUP_PROBE_ENTRY_UNBOUND_20261016_01.py)、[训练tail](code/N20_REPEAT202_TRAIN_GPU_RUN_TAIL_UNBOUND_DRAFT_20261016_01.py)、[tail接线说明](code/N20_REPEAT202_TRAIN_GPU_RUN_TAIL_UNBOUND_DRAFT_20261016_01.md)。实际执行使用独立冻结副本及本机资格输入，公开范围不含这些载荷；不承诺仅凭草稿一键复现。


## 正式训练入口与运行状态

原supervisor55247运行中，当前181正常/26保留失败/207终态；该活动任务尚未加入正常链。仅第二种子202，没有203或额外训练预算。未读取活动payload、SOURCE或资格链用于同步，未宣称实际已达到任何epoch或更新数。

[新front组件](code/N20_REPEAT202_TRAIN_GPU_FRONT_UNBOUND_DRAFT_20261016_01.py)在新202配置下运行当前own gate，并以首201真实历史配置检查旧失败记录；不会把旧初始化改成202。新初始化CPU179与探针GPU180的原spec确实没有旧`expected_normal_RAW_count`字段，新增两个精确schema/owner分支接纳原件；旧F/P/cache继续要求原字段严格相等，未更改任何冻结原件。

[三个metadata-only测试](code/test_N20_REPEAT202_PARENT_COUNTER_ABI_SOURCE_ONLY_20261016_01.py)先实际RED复现问题，最小修补后GREEN；覆盖两类新父记录、旧父门严格保留和异常不得扩展到其他owner/新增字段。测试不导入Torch/NumPy，不读取SAR数据。它们只支持接口修复，不替代真实训练、收敛、AP或科学效用验证。
