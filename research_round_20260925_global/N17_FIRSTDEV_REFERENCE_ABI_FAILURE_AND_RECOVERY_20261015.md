# N17 首次固定DEV接口失败与恢复

2026-10-15。原首段GPU监督18818真实退出1；原句柄已回收、原birth消失。全部既有122个正常任务与18个失败保留，140个身份pin。首段未生成START、REPORT或输出目录；不是训练失败、方法失效或新AP。24轮配对训练及独立E8/E24 CPU资格仍成立。

根因是入口把旧Root结果的`control_results[name].COCO`当作三字段文件引用；实际是包含stats、metrics、iouType、maxDets、原GT及crowd语义的内嵌字典。错误来自本项目接口实现，不能归因于GPU速度、模型能力或数据。原01源码、spec、admission和终止记录不改写。

恢复采用独立02源码、新spec/admission和新输出目录；仅修正COCO字段读取并保留第18次失败的完整身份/Source门。COCO.stats与结果stats逐项一致，12项metrics保持原0..1尺度、segm/maxDets[1,10,100]、原GT、无score过滤和官方crowd/ignore语义。先用同形内嵌字典最小合成检查和一次定向审阅，再由Root单次启动；没有自动重试。

训练预算、初始化、候选C、K、query/类别分数/PR、8个评价profile及E24 atmost主门均保持。四段完整187图合并、独立CPU选点/PR/mask/RLE与统一COCO通过后，才判定收益与是否追加预登记初始化。单段、FIT loss或合成接口检查均不能证明涨点。

从GLDet得到的当前N17仅为同query局部原生logit信息的低成本检验，不是GLDet复现或完整局部梯度网络。先尝试模块是可行研发策略；若涨点，仍须排除额外容量、预算、评价因素，并说明与直接先例的具体差异。尚无核心创新、SOTA或TGRS投稿资格。

[固定DEV最新状态](N17_FIXEDDEV_EXECUTION_PREPARATION_STATUS_20261015.md) · [完整训练资格](N17_ENDPOINT_CPU_QUALIFICATION_ENTRY_AND_STATUS_20261015.md) · [预登记计划](N17_LOCAL_NATIVE_LOGIT_ERROR_SELECTOR_EXPERIMENT_PLAN_20261015.md)


## 第二次原失败：Root SOURCE漏项

独立02原监督11194真实退出1（183.953秒），原句柄已回收、原birth消失。COCO内嵌dict检查已通过，随后读取`native_official_PREDICTIONS.json`时因新spec Sources缺少原结果导出文件而拒绝。Root构造清单遗漏，不是模型错误。仍未生成START/REPORT/新预测/输出目录。

Root先通过全部122正常+19失败的纯原RAW，再核验12个旧结果导出的原strictref/字节/SHA并补入下次Source闭包；两个实际复用控制均为187图/18700自然query，逐图原qids顺序完整吻合。原01/02保持冻结，第19次失败完整保留，总141身份pin。下一独立03仅保留failure19及修正绑定，数值算法和原主门不变；尚未启动。
