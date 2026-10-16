# N20 两种子结果：限定范围的独立完整性复核

日期：2026-10-17。审阅后端为 **gpt-5.6-sol**，作为可用的跨模型替代；Codex MCP及技能指定的GPT-5.4后端不可用，不能冒称使用了它们。审阅者独立读取文件，未修改文件、运行模型、数值、COCO或资格gate。

**总评：WARN；主张支持度：provisional partial。预锁续验门槛不通过，正确预览动作是STOP_CURRENT_RECIPE。** 主规则判断置信度高；完整端到端科学主张置信度中等。本报告时点的第二seed post接收尚未完成，原Q的pending/qualified=false不能当方法失效。

| 检查 | 结论 | 实际证据与限制 |
| --- | --- | --- |
| GT路径 | PASS（限定当前评价路径） | 有效入口从固定dev.json构建COCO，要求187图、380标注和原始顺序；未发现本评价中从模型预测生成GT。仅核数据集标注结构与加载路径，未重新追溯上游标注制作。 |
| 分数归一化 | PASS | 原始COCO统计为0–1，主解释只乘100；max(CENTER,R)是比较对象而非预测归一化分母。 |
| 数字文件及状态 | WARN | Q/DETAILS的十组control_results一致；十个预测文件存在，记录数各18700。WARN来自post接收尚待完成。 |
| 实际评价调用 | PASS | assembled入口最后的run调用实际coco_metrics；该函数调用COCOeval(segm)的evaluate/accumulate/summarize，固定类别1、imgIds、maxDets=[1,10,100]，无score filter。 |
| 范围与来源 | WARN | 重复使用的内部DEV187；两个scout初态digest不同，每seed内两臂匹配，但共享冻结载体。seed201恢复链、seed202正常训练链；未重新展开全部优化器轨迹。正式TEST及跨集未做。 |
| 评价类型 | PASS | real_gt / internal reused DEV / official COCO mask AP；不是模拟或模型生成GT代理。 |

实际检查的源码位置包括 `evaluate_full187.py:2693`（单位）、`:2731`（完整合并）、`:2816`（GT）、`:2840`（指标调用），实际 `qualify_evaluation_cpu.py:626`（官方COCO包装），两seed解释器`:12,:33`及锁定协议。这里记录行号供本地冻结源码定位；没有将资格链、完整预测或标注复制到发布仓库。

读取范围包括本轮assembled评价入口、两份拼装片段、当前spec的必要字段、Q/DETAILS、十个预测文件的存在/字节/记录数、解释器、锁定协议、first201资格结果，以及一跳载体spec、固定dev.json和实际COCO helper。未遍历完整 SourceMap，也未读原图、掩膜或权重；不能称全项目、全历史或所有载荷的完整审计。

第一轮审阅所给的历史TEST台账文件名缺少TEXT_EVIDENCE，因而未读取；实际普通台账见[历史TEST使用的文本证据](HISTORICAL_TEST_USE_LEDGER_TEXT_EVIDENCE_20261016.md)。本报告不将其内容冒称为第一轮已独立审过，也不把本轮未评估formal TEST解释为历史从未接触TEST。后续补充如有，应独立记录。

能支持的有限描述是：两个预锁下游初始化下LOCAL均略高于CENTER，但平均主差值仅+0.045039 AP点且seed202低于强R，未达到续验门槛。不能支持总体胜过强R、总体有效/失效、正式测试泛化、SOTA、新颖性或TGRS就绪。E8或exact等次要输出不能替代主终点。

该审阅不替代独立post的实际退出、Source/spec绑定及完整接收；无需因WARN重算十组COCO。数值、条件和下一步见[主结果说明](N20_TWO_SEED_MAIN_RESULTS_20261017.md)。

## E项补充：历史TEST台账已独立核读

审阅者随后按更正路径读取普通台账，E项仍为WARN。台账第17行记录历史旧稿的特定PSeg-SSDD competitor表曾使用best-on-test checkpoint，以及两项SAR模块复现按近岸子集选型；证据是历史文稿直接披露，未重新认证原始进程、权重身份或时间链。范围不能扩展到全部SSDD实验。

台账第24行明确不重新判断当前N20的具体TEST使用；不能由旧稿披露反推本轮N20权重、阈值或停止规则经过TEST选型。本轮formal_TEST=false也不能解释成历史从未查看过TEST。未来正式评价须披露历史接触，不把相同测试集重新称为未接触确认集；一般化仍需冻结配方后的透明外部评价和独立跨集证据。

本补充仅读取已发布普通台账，不重新打开正在接收中的科学Source、spec、Q或载荷；其余A–F结论不变。

## Root进展注：原post失败27（2026-10-17，非原独立审阅结论）

独立post原任务19115实际退出1，原件保留为failure27；新的RAW-only失败前缀已实际退出0（a151e0）。当前为220个正常终态、27个保留失败、247个总终态身份及228个legacy pins，失败post不计入正常完成。已定位为两个按seed索引的字典在JSON往返后由整数键变为字符串键，直接比较Python对象因而失败；完整数值及预锁`STOP_CURRENT_RECIPE`预览保持不变，这是接收接口问题，不能当作模型失败。第二seed的Root接收仍未完成。fixture10项已实际退出0（aea237），Source装配已实际退出0（8336a3）；独立恢复原session5111已启动（launch5020e5），尚无actual终态，Root接收仍pending；不能写成已恢复，也不把合成回归或启动当作实际资格。冻结Source、Q和数值产物不改，不重算十组COCO。

本段仅补充Root提供并核验的执行进展，不将上述跨模型审阅改写为修复版的集成审计或Root接收，也不改变原WARN/provisional partial结论。
