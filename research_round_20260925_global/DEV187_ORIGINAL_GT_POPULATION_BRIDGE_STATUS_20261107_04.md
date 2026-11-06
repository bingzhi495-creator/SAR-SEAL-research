# DEV187 原GT字节身份与候选人口/尺寸桥（2026-11-07，尝试04）

状态：**`PASSED_ORIGINAL_GT_BYTES_AND_CANDIDATE_POPULATION_DIMENSIONS_ONLY`**。GT字节身份和候选人口/尺寸桥已通过。

新独立预锁在输入读取前保存。仅沿固定attempt03私有manifest的明确advertised GT引用，按stat→完整118,319字节/SHA256精确身份→JSON容器解析的顺序核验，未匹配不比较人口。**本次明确获准并实际读取原DEV GT正文**；不改GT，不向预测输入标注。

候选/GT各 187/187 图像，unique image IDs各 187/187；集合一致 True，height/width差异 0。GT 380 条、unique annotation IDs 380，类别仅1 True，未知图像链接 0、无效类别链接 0。候选ordinal与GT.images序列关系 **EXACT_EQUAL**，没有重排行或改GT。

本次范围只到原GT byte身份、187人口/尺寸和380唯一标注的COCO链接桥，不能自授Root完整执行资格。独立当前N28 image_ids数组角色没有明确输入，**N28当前顺序证据仍未知**；没有为此追随qualification/capture/package/active scientific SOURCE。当前GT.images顺序一致与否不能替代N28原执行角色证明。

旧01/02失败及03候选组成历史全部保留。此前定位过程也实际解析过带annotation ID映射的独立容器，不能说整个流程未触及GT相关数据。没有像素/掩膜栅格化、Torch/模型、预测、AP、SSH或DGX传输，未改运行中训练/调度。物理标注质量、像素对齐、官方发布/许可与官方TEST独立性均不在本次证明范围；DEV已用于开发。

实际耗时 0.031 秒，CPU 0.062 秒，子进程峰值工作集 19.58 MiB。单工作线程、BELOW_NORMAL、可用时CPU2/3、512MiB Job硬内存限额、外部300秒时限。原GT closed ref、逐项身份/locators、prelock和实际receipt仅私有保存。 实际工具328ee5/0退出码0；阻塞监督等待子进程后传播成功码，工具无存活session。资源仅为子进程Windows peak working set/pagefile，未测监督RSS或全机内存，未独立退出后扫PID。

[公开聚合记录](results/DEV187_ORIGINAL_GT_POPULATION_BRIDGE_AGGREGATE_20261107_04.json)。
