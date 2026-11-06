# DEV187 原图候选完整内容覆盖：预锁组合尝试03（2026-11-07）

状态：**`CANDIDATE_ONLY_COMPLETE_CONTENT_COVERAGE_187_LOGICAL_183_OBJECTS` / CANDIDATE_ONLY**。在新目录先保存独立content-address composition预锁，再读取固定SHA的原187 inventory及attempt02实际闭合记录。逐行用原expected SHA256、长度、height/width精确关联已验证原JPEG对象，保持全部原image_id和ordinal；不近似匹配、不改GT、不删除或合并逻辑样本。

| 尝试 | 实际结果 | 完整候选manifest |
|---|---|---|
| 01 | 首行旧定位缺失，退出1，0/187 | 无 |
| 02 | 原basename规则选中183/187，退出1，4行缺同名文件 | 无 |
| 03 | 本次组合 187 个逻辑输入 / 183 个物理原字节对象，4 行精确同字节复用 | True |

旧尝试和失败计数保持历史原样。逻辑编码字节总计 **7,274,234**，独特物理对象字节 **7,118,253**。manifest的dataset_index为原inventory ordinal 0..186，原dataset_index私有另记；GT仅沿用advertised原整数引用、expected380及类别映射0→1。

**证据范围。** 原图完整SHA256、长度及JPEG header来自attempt02实际核验的183对象闭合记录；本轮没有重新读图像字节、哈希图像或读取header，仅核当前183明确objects的regular-file存在/大小与组合前后stat稳定：True。新组合仅解决四个历史basename缺失的内容寻址，保持187逻辑样本。此前定位诊断实际解析过带annotation ID映射的独立定位JSON容器，不能声称整个流程从未触及GT相关元数据；未将其中标注用于组合或预测，DEV GT正文未打开/哈希/stat。

**本次只形成完整内容候选，不授Root最终预测输入资格。** 当前N28/currentDEV顺序/角色桥、GT人口身份仍未获独立准入；未送DGX，未forward、预测、mask或AP；未改科学SOURCE、权重、资格链、训练或调度。像素完整解码、GT对齐、官方版本和许可未新证明，内部DEV已用于开发，不是新官方TEST。

本轮metadata组合耗时 0.063 秒，CPU 0.062 秒，子进程峰值工作集 21.04 MiB。单工作线程、BELOW_NORMAL、可用时CPU2/3、512MiB Job硬内存限额与300秒外部时限。原图/逐项manifest、映射、字节摘要、locators和prelock全部仅私有保存。 实际工具0f22b4/0退出码0；阻塞监督等待子进程后传播成功退出码，工具无存活session。资源指标仅为子进程Windows peak working set/pagefile，未测监督RSS或全机内存，也未独立退出后扫PID。

[公开聚合记录](results/DEV187_ORIGINAL_IMAGE_CANDIDATE_COMPOSITION_AGGREGATE_20261107_03.json)。
