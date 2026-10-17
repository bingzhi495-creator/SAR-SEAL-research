# N28：全FIT配对密集路由训练源码

2026-10-17。源码于执行前冻结；首seed的0→8轮原任务已启动，2026-10-17 00:04 UTC复查同一原任务仍运行，尚未终态接收，没有已核验的完成轮次或AP结果。此前三seed共同初始化与首个真实实例GPU接口已实际通过；该首例不代替本轮新reader对全部741图/1661真实匹配实例的实际加载。

两组SELECTIVE/WHOLE_NODE各2,978,555参数，采用同一原初始化factory、完整单层F200、自然query e/M/native_L、全mask可写及完整实例GT。原up_x节点的唯一区别是其他出边保留D或改用同一个T=G(D)，跨尺度均U(T)。固定S0的正式评价另行冻结，不使用DEV选轮/调参。

- reader.py：逐图验证已资格完整F与N16自然query/L的来源、摘要、原matcher及GT；不缓存30GB全集，不用抽样点代替dense特征。
- train_core.py：共同图序、GT连续微批4、每图每臂一次AdamW step；全图均衡BCE+Dice，完整保留741图/1661GT，BN微批语义明确。
- train_body.py：3seed中的授权首seed，固定24轮按8轮分段；逐轮新文件保存两个模型/BN、两个原优化器完整state、私有图序Generator和CPU/CUDA RNG及实际计数，续段严格恢复。
- run_train.py：原单任务GPU入口、资源及待接收产物；实际结束与Root接收由外部原进程证据确定。
- factory.py/resource_runtime.py：已冻结原初始化factory与资源源码的精确发布副本。实际spec.factory_source复用原冻结路径，以严格恢复原CP来源；发布目录不会替换该本机绑定。
- 两个metadata测试：纯标准库原人口身份/破坏拒绝与图序/GT微批权重行为检查；不冒充真实reader、优化器恢复或NN训练效用。

完整配方、重复seed门槛与停止规则见[训练执行前计划](../../../refine-logs/N28_DENSE_ROUTING_20261017_01/TRAIN_EXPERIMENT_PLAN.md)。每epoch每臂741次更新，24轮17,784次；AdamW lr3e-4/wd1e-4，FP32，无scheduler/AMP/增强/早停。数据、权重、GT、checkpoint、RAW/SPEC/Q/资格及运维绑定不发布，故尚非一键复现镜像。

[初始化实际结果](../../N28_DENSE_ROUTING_INITIALIZATION_RESULT_20261017.md)；[N27完整几何依据](../../N27_MATCHED_SUPPORT_CENSUS_RESULT_20261017.md)。当前无新AP、正式TEST、跨集或核心创新成立结论。

后续状态：首段原任务已受控超时退出，失败历史已核验并保留；检查点恢复尚未核验，不能将原八轮记为成功。[失败及恢复要求](../../N28_FIRST_TRAINING_RESOURCE_ABORT_20261017.md)。
