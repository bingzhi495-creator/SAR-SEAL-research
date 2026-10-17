# N28：单轮执行分段修订

2026-10-17。执行前源码，尚未实际运行。仅将原八轮任务分段改为单轮；原24轮预算、两臂配方与完整检查点结构保持。首段须先取得原E05完整端点的实际CPU恢复检查资格，再在GPU首次更新前严格恢复并核对双优化器及全部RNG。

run_train.py与train_body.py采用v2执行规格、预锁和待接收报告；checkpoint schema仍为原v1。factory、reader、train_core及resource runtime复用原冻结来源，不修改它们或原权重。额外断言用于检查完整state恢复，不增加模块或更换超参数。

[恢复预锁与分段修订](../../../refine-logs/N28_DENSE_ROUTING_20261017_01/RESOURCE_RECOVERY_PLAN.md)。尚无本版本实际续训、完整24轮或新AP结果。
