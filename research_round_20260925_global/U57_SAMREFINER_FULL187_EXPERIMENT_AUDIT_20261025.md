# SAMRefiner完整187图：独立审阅与主张边界

2026-10-25。实际审阅者为GPT-5.6-sol xhigh独立子代理；指定GPT-5.4 Codex MCP接口不可用，本报告不冒称该指定接口审计。审阅仅只读代码、已关闭结果与执行摘要，并读取固定上游源码；没有SSH、独立重跑或直接读取远端GT/预测载荷。

结论：**WARN，未发现阻断性完整性FAIL；有限主张partial、置信medium。**

| 检查 | 判断 | 证据与限制 |
|---|---|---|
| GT来源 | WARN | 代码从独立DEV COCO JSON载入380标注，预测另经loadRes；未从模型输出制造GT。审阅者未独立检查远端GT全文与上游数据集身份 |
| 评分及归一化 | PASS | 原S0与元数据保持，仅改segmentation；保留原始COCO12，百分制只是乘100 |
| 结果存在与原终态 | PASS（有限） | 已保存接收及观测摘要互相一致，原tool/SSH/worker均0；审阅不独立证明所有远端执行和引用文件字节 |
| 指标实际调用 | PASS | CPU循环调用N28指标定义，再调用原helper，执行COCOeval evaluate/accumulate/summarize；输出进入完整结果 |
| 范围 | WARN | 单一固定作者/default执行种子、历史选型接触的内部DEV187/380；不是FULL_E50、正式TEST或重复/跨集证据 |
| 评价类型 | real_gt（有限） | 内部GT实评代码路径，远端payload未被审阅者独立检查；不是合成或自监督proxy |

独立审阅的[固定上游作者实现](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/sam_refiner.py)及[提示工具](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/utils.py)哈希匹配执行pins。tuple首项确为精修掩膜；作者逐行在三个分支中按预测IoU选取，并按原行顺序堆叠，没有发现跨行重排路径。原网格、uint8二值及Fortran RLE编码未发现具体错误。不以缺少动态逐行trace为由事后重绑S0。

可支持：[实际结果](U57_SAMREFINER_FULL187_COMPARISON_EXECUTION_20261025.md)记录在本次原候选/S0条件下，SAM AP42.770654，低于native67.071057与固定R68.099385。不能支持SAM总体失效、同训练成本优越性、FULL结构有效、SOTA、核心创新或TGRS就绪。

AP损失仍可能混合几何变化、固定S0与新几何的质量失配，以及作者内部三分支选择；当前聚合指标不能唯一分解。下一项固定native匹配的TP/FP/FN与面积比诊断属于事后探索，仅覆盖native已匹配人口。全100覆盖或原S0分层可作为后续补充。coarse/refined最佳匹配不能证明置换，也不能用于改写原AP。

本报告未包含原始数据、GT全文、权重或私有运维引用。完整结果和范围保留，指定审阅接口的不可用不阻止既定主训练接续。
