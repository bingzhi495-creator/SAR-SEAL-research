# N28：实例条件密集路由的初始化与接口源码

2026-10-17。状态：已实现、冻结并完成静态审阅和4项纯标准库路由行为检查；真实数值尚未执行，正式结构训练尚未启动。

本小型适配沿用户model_new1的up_x连接，单层冻结mask feature F256与自然query e/原coarse M生成完整delta_M；两组均可修改全mask。SELECTIVE保留其他出边D，WHOLE_NODE改用同一已计算T=G(D)，跨尺度均U(T)。同尺度共享纯CBAM，独立上采样边，宽度16/32/64/128/256。它是查询条件的实例精修适配，不是完整历史语义UNet++或多层FPN输入的复现。

- factory.py：无数值导入的工厂，公开make_head(torch,mode)与末层readout；两臂共享参数接口、零初始化残差头。
- test_routing_metadata.py：用可追踪令牌检验真实forward复用的route_node出边分配、单次门控和错误mode拒绝；通过不能替代网络数值与真实效用。
- run_initialize.py / initialize_body.py：原单任务GPU初始化与第0张FIT图的第一个真匹配实例核验；保存三个原始seed共同初始状态，disposable副本做零残差恒等、域外可写、非零路由差异与两次反向，不创建优化器、不做step。
- resource_runtime.py：已用资源控制源码的精确副本，作为runtime_source加载Control，不执行旧训练CLI。

种子202610172801/2802/2803；原native_L为base，不拿二值R或CPU重放替代。F/e LayerNorm无affine，F200→800→actual裁剪→original沿既有GPU投影。旧N13 full-F载体及新N16自然query/L的身份、摘要和实用数值关系必须先核。当前仅第一真实接口资格，不授予全FIT训练资格，不报告AP/DEV/TEST。

[执行前计划及正式训练前缺口](../../../refine-logs/N28_DENSE_ROUTING_20261017_01/EXPERIMENT_PLAN.md)；[N27实测依据](../../N27_MATCHED_SUPPORT_CENSUS_RESULT_20261017.md)。模型、优化器、数据、GT、载体、RAW/SPEC/Q和绑定凭据不随本源码发布，因此尚非一键运行镜像。
