# 真实full卷积特征CPU64检查：未执行草稿

2026-10-17。仅AST及Root静态审阅，未导入或运行网络。它接续已完成的tiny接口验证，不能把前者成功转写为本次成功。

spec=ROOT-U30-full-feature-CPU-spec-v1，operation=CPU_U30_FULL_FEATURE_CONTRACT；九个角色为entry/body/runtime/legacy/dsconv/head/feature/test/adapter。完整闭合来源、原十三字段准入及资源核对先于Torch导入；仅Root原叶可执行，CPU0、1/两线程，CUDA不可见，RSS和最小可用RAM各16GiB，截止预锁且不超过1800秒，无自动重试。

实际原full及两个直接依赖需由Root后续独立冻结后绑定；使用dsconv/head/model_full原import字面量隔离加载，finally恢复原模块别名与外部CPU随机状态，拦截并记录CUDA lazy初始化尝试。

仅两项：不改动原64尺寸feature测试源；以及真实full→feature wrapper→共享adapter、B1/Q2、eval+autograd，非零末层读出。反向之前从ABI固定effective/inactive参数清单；所有effective梯度存在且有限，有限零张量/中心零offset不误判失败；所有预期inactive梯度为空。记录图像/查询/粗logit梯度、一次wrapper/特征调用、原legacy forward零次、四个语义头均零调用及全部head BN buffers逐元素不变。

输出FULL_FEATURE_CONTRACT_REPORT.json、UNITTEST_STDOUT.txt，只有通过才写待Root实际终态接收的Q。成功也仅证明随机合成64尺寸eval特征路径，明确不授予400资源、train态BN、旧权重、真实影像/GT/teacher/渲染、优化器、训练/AP或完整方法资格。本目录不包含实际spec、资格链或运行载荷。
