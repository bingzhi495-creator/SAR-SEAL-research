# N17受控时限中断与五段执行补充

2026-10-15。第二段原GPU59160（原范围[47,94)）在1800.329秒被资源监督中止，原actualexit1/held1/reaped/birth消失，abort为QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE。整卡观察峰值3258MiB、最低可用RAM25.65GB、无观察错误；这是执行时限事件，不能判定方法失效或主STOP。Root先通过全部123正常+20失败的纯原RAW，之后核Source/spec/admission/phase/bootstrap和REPORT不存在。

中断后仅文件元数据观察显示47个capture、manifest和10种prediction文件已写；REPORT不存在，未取得原执行资格。没有读取或哈希这些失败载荷以复用，失败partial不计入合格结果，原件全部保留。首段已核Source03原actual0及47图捕获/导出metadata不改，独立CPU数值仍待完成。

新执行在启动前锁定五个物理分区，原科学人口及四个逻辑分区不变：

| 物理分区 | 数据序号范围（左闭右开） | 合格GPU owner序号 | 状态 |
|---|---|---:|---|
| 0 | [0,47) | 122 | 复用已actual0的Source03与其完整metadata，CPU资格待 |
| 1 | [47,70) | 123 | 新Source04，原监督64568已启动，结果待核 |
| 2 | [70,94) | 124 | 新Source04，尚未启动 |
| 3 | [94,141) | 125 | 新Source04，尚未启动 |
| 4 | [141,187) | 126 | 新Source04，尚未启动 |

只有执行任务变小；训练24轮/初始化171/数据/同query原生八邻与中心copy/C/K/固定PR/类别分数和导出query顺序/10种策略/主E24atmost/强参考和重复停止门均不变。完整187图/380GT/18700自然query统一COCO，不平均分段AP。GPU单一ownedleaf、CPU01两线程、allocator4096MiB/整卡8192MiB、RSS16GiB/1800秒、无自动重试保持。额外Source检查和分段使评价总墙钟时间增加，不冒充更多训练或方法成本提升。

新GPU04、numericCPU02和最终COCO02需保留第20次完整RAW/owner/Source/REPORT缺失链；旧N16的四段legacy载体仍是四段，不随新N17物理分区改写。原CPU01/final01仅支持先前4分区，保留为历史版本，本次新版本实际启动资格尚未完成。后续只有五个实际结束和完整独立数值证据闭合，才报告统一全量AP。没有新方法收益、创新或TGRS结论。

[最新状态](N17_FIXEDDEV_EXECUTION_PREPARATION_STATUS_20261015.md) · [锁定方法协议](N17_LOCAL_NATIVE_LOGIT_ERROR_SELECTOR_EXPERIMENT_PLAN_20261015.md) · [直接先例](N17_LOCAL_NATIVE_L_ERROR_SCOUT_TARGETED_PRIOR_CHECK_20261015.md)

## 本次启动

第二物理分区 [47,70) 已启动，原监督句柄64568，Source04。当前只确认父监督在运行；是否完成、数值是否合格及AP均待原结束后核验。
