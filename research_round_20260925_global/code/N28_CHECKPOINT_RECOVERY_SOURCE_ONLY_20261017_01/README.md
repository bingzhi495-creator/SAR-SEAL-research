# N28：原E05完整状态恢复检查

2026-10-17。执行前源码，尚未实际加载原检查点。run_probe.py为单次原CPU入口；checkpoint_probe_body.py核完整两臂模型/BN、原AdamW矩与步数3705、完整五轮历史、图序/CPU RNG恢复及CUDA RNG的CPU载荷。test_checkpoint_recovery_metadata.py仅两项标准库捏造元数据检查。

不训练、前向、反向或评价，不读取GT，不初始化CUDA。实际CUDA RNG恢复必须由下一GPU任务另验。输出待原任务实际终止和Root接收；不自行授予科学资格，不把原八轮任务记为成功。

[恢复预锁与分段修订](../../../refine-logs/N28_DENSE_ROUTING_20261017_01/RESOURCE_RECOVERY_PLAN.md)。本目录不含权重、数据、执行绑定或资格载荷。
