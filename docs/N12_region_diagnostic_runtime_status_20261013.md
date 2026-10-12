# N12 区域诊断运行状态

日期：2026-10-13。当前状态：**GPU_CPU_ACTUAL0_ROOT_QUALIFIED_DIAGNOSTIC_ONLY**。GPU02和独立CPU02均已实际正常结束并完成ROOT核验。[完整限定结果](N12_sameK_region_qualified_result_20261013.md)已记录，不作为训练收益或创新。

## GPU01：依赖来源核验失败，已实际终止

| 项目 | ROOT 已确认事实 |
| --- | --- |
| 实际退出 | 原始 held handle 为 WAIT_OBJECT_0，exit code=1；ROOT 已收回进程，原始 birth 已不存在 |
| 用时 | 19.922 秒 |
| 整卡观测峰值 | 1743 MiB；是 whole-GPU 观测，不能解释为本进程的独占显存 |
| 退出阶段 | common.imported_files 的实际文件来源核验 |
| 科学计算边界 | 尚未开始本轮新区域计算；没有新的方法负结果 |

ROOT 使用 stdlib PathFinder 与实际安装文件树完成来源诊断：SciPy 1.10.1 的 egg-info/SOURCES 清单遗漏 **120 个实际 py/pyd/dll 文件**，包括生成的 __config__.py 和 ndimage 二进制；cv2 的实际 **15 个文件全部已绑定**。本次失败说明绑定清单未覆盖实际 SciPy 文件来源，不能据此判定几何区域假设或点头方法失效。

GPU01/v01 与其实际 exit1 记录冻结保留，不覆盖、不改写成正常结束。

## GPU02：来源闭包修正后实际正常结束

ROOT 已按实际安装文件树补齐绑定并开展单次 GPU02 执行。原始 held handle 的实际退出码为 0，ROOT 已收回进程并确认原始 birth absent；用时 80.609 秒，整卡观测峰值 3545 MiB。独立 CPU02 已实际退出0、收回并确认birth absent，用时256.078秒；ROOT先核验37正常终端和4保留失败，再核对Q/SOURCE/统计闭合。v02 保持区域选择和指标算法不变；变更范围为实际包树来源核验与绑定、加入 GPU01 为第四份保留失败，以及相应历史资格前缀核验和 ABI 说明。

历史 N10 Q 仍按其生成时原有两条 additional_failed_RAW_inputs 的**精确前缀**核验；新增失败保存在后续链中，不反向添加到旧资格对象，也不以更新后的失败列表冒充旧 Q 的输入。正常祖先与失败祖先都须按实际身份及原始终止记录核验。

ROOT 为科学执行所有者；这是一轮重新准备后的单次执行，没有自动重试。独立CPU区域raster、整数计数和距离直方图等逐项通过。主检验旧区域错误覆盖22.53%→几何同K24.67%，oracle仅+0.000728650；预登记弱诊断门通过，效用与创新仍未建立。

## 结果可解释前的要求与适用范围

GPU02 只有在原始实际 terminal0、进程收回与 birth absence 核验完成，并经独立 CPU 复算及资格闭合后，才可解释新的区域覆盖、距离和条件 oracle 结果。运行成功本身也不证明研究假设成立；若实际执行再次失败，保留失败原因与原始状态。

N12 使用冻结原生输出和内部 DEV 的固定 query–GT 人口，比较旧 logit 带与逐查询相同 K 的几何支持；uncertainty top-K 仅作旧带集合一致性检查。RefineMask width2 原图适配为另列实际面积的补充对照，不冒充同 K 或完整作者复现。固定 .50/.75/.90 槽位的人口分别报告，保留 FN，不跨槽位作差。

本轮是**same-K 区域覆盖与条件 oracle 支持诊断**，没有新头训练；不是 AP、SOTA 比较、共享模型能同时达到的性能上限或 TGRS 投稿证明。不承诺训练收益。通过诊断后仍须另登记新 FIT 采样和公平完整训练；未通过也只限制这一特定几何规则与诊断协议。

协议及背景见 [N12 实验前登记](../refine-logs/N12_REGION_COVERAGE_20261013_01/EXPERIMENT_PLAN.md)、[执行跟踪](../refine-logs/N12_REGION_COVERAGE_20261013_01/EXPERIMENT_TRACKER.md)、[N11 已核验限定结果](N11_original_editable_band_qualified_result_20261013.md)和 [仓库当前研究状态](../README.md)。本页由普通发布文档与 ROOT 确认事实形成，没有读取原始 SOURCE、SCI、缓存、SPEC、admission 或 runtime 目录。
