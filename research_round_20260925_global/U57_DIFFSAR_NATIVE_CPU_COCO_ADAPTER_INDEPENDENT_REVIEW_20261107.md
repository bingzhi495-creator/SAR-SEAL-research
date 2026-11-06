# 原生DiffSAR CPU COCO薄适配器：有限独立静态审阅

2026-11-07。只读新v1 `worker.py`（17615字节，SHA256 `f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748`）、README、四项标准库/API替身测试源码及此前审过的原生预测v1接口。未重跑测试，未导入科学库、读取真实GT/prediction/CP/实际资格或绑定SOURCE，未SSH或执行COCO，未修改冻结候选。

**结论：指定接口的有限静态复核未确认具体阻断。代码就绪和替身测试不证明真实数据、库ABI、指标或执行资格已通过。**

| 聚焦项 | 静态核对结果 |
|---|---|
| 原COCO数学与完整人口 | 直接调用原`COCO.loadRes`、`COCOeval(...,'segm')`及evaluate/accumulate/summarize，没有本地IoU、匹配或AP数学。完整187张、category1、maxDets[1,10,100]一次总体评价；不平均分片AP。全局空预测也通过原COCO空结果对象和原评价器处理，不手填零分。 |
| 预测输入保持 | `segm_copy`深复制完整列表，仅从副本删除bbox，使原API按mask面积分箱；不改原文件/list、score、类别或顺序，不前截100。允许每图零条及超过100条；总数和逐图数与原producer记录相接，不要求18700条教师查询。 |
| 人口及原GT | 核producer全部187图顺序与唯一ID、原图尺寸、完整输出计数、已应用一次0→1类别映射、原最终37050 checkpoint/ref及原producer source；实际COCO GT加载后核187唯一图、380唯一标注、category1及尺寸。适配器不重建/改写GT几何或再次映射类别。整数版本的历史身份仍依赖Root提供的原固定字节身份，不是从名称重新认证。 |
| 编解码来源接口 | 预测v1实际记录的`actual_library_refs`含`pycocotools_mask`与`pycocotools_extension`；CPU使用同名两键，要求完整path/bytes/SHA ref相等。另核实际导入的五个库ref与Root明确canonical源集，而不是以版本字符串替代。预测器本身没有计算AP，CPU定义的canonical COCO数学不自动重新认证历史结果。 |
| 状态及失败诚实性 | 调用入口前置状态与完成状态分开；进入后异常保留started=True/completed=False。完整12项统计返回后，completed/AP_computed/metrics先保存；随后资源或源复核失败只改FAILED/exit1，保留已算结果但不授科学资格。qualified始终False。原−1未定义统计保留，百分制视图用None而非伪造0。 |

Root仍须实际核验原最终端点及完整预测的终态、来源、输出、出生/资源回收后，才发放新的CPU lease。适配器对原资格ref做字节身份核验、消费Root内联admission的语义投影；这些值不能代替真实外部审计，也不能修改冻结训练/预测lease或资格文件。schema失败在REPORT建立前以原stderr/exit1保留，不伪造“完成”。

尚缺实际37050状态与完整187预测、正确原GT/manifest、可相接的canonical库与mask二进制部署、实际CPU owner及总体COCO运行和Root最终接收。本次未证明这些条件；四项旧替身测试只证明其有限调用与拒绝行为。`COCO_math_started`是进入调用阶段的记录，单凭它不能认定AP已完成；应结合completed、AP_computed、metrics和外部原始终态解释。
