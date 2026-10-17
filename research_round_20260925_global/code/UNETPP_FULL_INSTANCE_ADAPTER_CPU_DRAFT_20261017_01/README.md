# 共享实例接口的受控CPU行为验证：未执行草稿

2026-10-17。为[共享实例接口](../UNETPP_FULL_INSTANCE_ADAPTER_DRAFT_20261017_01/README.md)准备真实tiny-nn测试入口。仅AST通过（be6a7b），没有导入Torch或执行任何测试。

未来由Root统一拥有原任务。完整五个源码角色entry/body/adapter_source/test_source/runtime_source及所有输入须先经过闭合引用和原13字段准入核验，之后才导入Torch。CPU固定0、1，两线程，空CUDA可见设备，RSS与最小可用RAM各16GiB，单任务截止由spec预锁且不超过1800秒；无自动重试。

设计为实际执行三项检查：未经改动的原公共测试（非零读出时逐query、分批、置换与每图一次主干）；零初始化残差精确为零；真实tiny网络对共享特征及小头的有限非零反向梯度。最后一项显式确认未用于损失的旧语义尾部没有梯度；没有优化器或真实模型训练。

所有实际开始/成功测试ID、错误、失败和原始测试输出应保留；失败不生成成功Q。执行前后检查CUDA冷状态，测试期间拦截lazy初始化并保留尝试记录；恢复外部CPU随机状态。输出ADAPTER_CONTRACT_REPORT.json、UNITTEST_STDOUT.txt，以及仅在测试通过后的待Root接收Q。

schema：ROOT-UNETPP-full-instance-adapter-contract-CPU-spec-v1；operation：CPU_UNETPP_FULL_INSTANCE_ADAPTER_CONTRACT；prelock：UNETPP-full-instance-adapter-contract-prelock-v1。五个源码角色之外，还需原当前Root资格、固定测试预锁、完整sources、全套原始运行控制与输出目录。此目录没有绑定这些引用，也没有授予实际执行或Root资格。

本草稿不实例化用户原full、DSConv或Transformer，不加载旧权重/真实影像/GT，不检查400网格渲染、不计算AP、不证明整网训练显存或完整结构有效。实际执行及Root原任务接收均尚未进行，不能把源码中的测试/报告字段当作已发生的结果。

## 后续实际执行更新

本目录源码在beae064c43a7325c8d68cc243b86ee841c13a5ef预先发布后已由Root受控执行，三项tiny真实nn测试实际通过并接收；此前未执行描述保留为版本历史。[结果及范围](../../UNETPP_SHARED_ADAPTER_CPU_RESULT_AND_FULL_FEATURE_STATUS_20261017.md)。原full、真实影像/400渲染和AP仍未执行。
