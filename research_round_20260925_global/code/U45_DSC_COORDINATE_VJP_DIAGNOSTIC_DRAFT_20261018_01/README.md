# U45：原 DSC 坐标映射的局部 AD / FD 诊断

2026-10-18，**UNBOUND / NOT NUMERICALLY RUN**。probe.py 唯一接口：`run_diagnostic(torch, DSC, report)`，返回同一 caller-owned report；失败保留 partial 并抛出。探针本身不含 main、runner、修复版本或资格逻辑；本轮不导入 Torch、不跑测试/模型/数值。

原普通发布副本：`github_release_20261011/research_round_20260925_global/code/U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/dsconv.py`，15207 B，SHA256 `b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0`。`DSC` 类在93行，构造器95行，`_coordinate_map_3D(offset, if_offset)` 在115行；仅调用该方法，绝不调用 `deform_conv`、插值、DSConv/Conv/BN。原 morph0 detach 位于166行，两处循环位于177/241行。实际 owner 必须先关闭该原源再注入 DSC；report中的常量指纹只是预期，探针不自行证明实际 callable 来源。

固定 B1/C1/K9/H=W1、18个独立CPU double offset，extend_scope=1、if_offset=True。分别对 morph0 的 `sum(y)`、morph1 的 `sum(x)` 求完整18通道AD，以及每通道对称中心FD；步长1/4096、绝对容差1e-9、相对容差0。offset用确定且非对称的二进制可精确表示数值，不使用随机生成。临时设置默认double使原内部linspace也是double，开启确定性算法；结束/异常均恢复调用者默认dtype、确定性/warn-only及grad上下文，并检查CUDA始终未初始化。

预期仅描述**原版**：morph0有效9通道AD `[0,1,2,3,0,3,2,1,0]`、FD `[1,1,2,3,0,3,2,1,1]`；morph1有效9通道AD/FD均为后者，非活动9通道均0。报告逐通道AD/FD、正负扰动目标、原预期与实测匹配、两morph有效通道差异和调用计数。实际是否符合预期必须由Root后续单一owned CPU叶取得，不从静态预期填充observed。

FD只检查当前前向函数；morph1即使AD/FD一致，也不证明原循环覆盖了全部端点或符合预期递推定义。

数值差异作为 `coordinate_AD_FD_mismatch_observed` 诊断记录返回，不把发现缺陷误报成修复/算法成功；外层可将该真实差异用于原版数值RED断言。本探针不实施修复，也不自动启动重试。坐标梯度差异不能直接证明原训练失效、mask/AP变化、SAR机制或论文创新；H=W1并未测试实际采样/边界/完整网络。后续修复需先取得Root真实原版数值证据，再写独立候选并统一验证。

## 后补 owned CPU 入口（仅源码）

`entry.py --spec ABS` 由Root原owned driver按Source pin执行；本任务未运行该入口。spec为 `ROOT-U45-coordinate-diagnostic-spec-v1`，operation为 `CPU_U45_DSC_COORDINATE_VJP_DIAGNOSTIC`，source_roles恰entry_source/probe_source/dsconv_source；sources另由Root明确提供当前父、pure prefix及metadata控制引用。import前逐项关闭Source、核entry自身与原probe/DSC指纹，核CPU0/1、CUDA_VISIBLE_DEVICES空及三项BLAS/OpenMP线程变量2；之后才导入Torch/NumPy并设Torch两线程、检查CUDA冷。

限额字典固定为 allowed_cpus=[0,1]、threads=2、maximum_RSS_bytes=minimum_available_RAM_bytes=17179869184、deadline_seconds=120、CUDA_visible_devices=''。原owned driver负责整叶强制资源/超时约束；入口另采样RSS与可用RAM，采样峰不冒称绝对峰。

唯一文件输出是新绝对D目录的REPORT.json，schema `U45-coordinate-vjp-diagnostic-report-v1`；不生成Q。含实际spec/source引用、库版本/路径、完整诊断、耗时/采样RSS及冷CUDA、Source字节、probe控制/模块别名恢复flags，qualified始终False。不声称闭合全部二进制依赖。成功只要求数值完成与范围/资源合同，不要求一定发现AD/FD差异或符合预设系数；失败尽量保存报告并保留原异常，绝不写入未由本入口新建的输出目录。
