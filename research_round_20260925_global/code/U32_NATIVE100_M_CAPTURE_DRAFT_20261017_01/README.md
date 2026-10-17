# 未绑定研发草稿，未完成数值资格

2026-10-17。为真实FIT691/image871/9GT补采原100查询的M200和F256，使用原冻结teacher真实forward，不从L或二值掩膜反推M。复用已核N22构造，初始化仍检查32项元数据/影像ref并构造未用冻结PR头；只解码前向1图，仍执行原30loss/10阶段matcher/GT核验，不是无GT推理。原N16 L绝对1e-4且符号精确；对原N22完整L/e/class/GT/assignment/export/top/quality/S0均严格精确一致；不执行domains/PR点/训练/AP/DEV。来源和原13字段准入先于Torch，Root单叶/CPU01两线程/CUDA0/allocator4GiB/wholeGPU观察8GiB/RSS与可用RAM各16GiB/1800秒一次原任务，无自动重试。只有原实际终态及Root接收后才记录结果。当前仅AST与静态审阅，尚未执行。

代码不包含本机spec/资格/原数据/权重。主线范围见[完整方案](../../../refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_PLAN.md)。

## 启动前路径对齐修正

首个metadata绑定在科学任务启动前退出1，没有新RAW、模型前向或计数变化。147个旧来源内容摘要全部不变，仅一个N21 Q路径采用正斜杠，而Windows Path格式化为反斜杠。新spec逐ref规范化路径；新Control子类仅对旧ref作相同路径拼写变换，然后完整调用原closed的成员/字节/摘要核验。旧spec、资格和失败准备副本均保留，不放宽任何文件内容或数值门槛。新纯标准库公共回归先缺函数RED后实际GREEN，未导入Torch；它只验证路径身份，不是M100采集成功。修正来源在新冻结目录保存后，才进行第一次真实科学任务。
