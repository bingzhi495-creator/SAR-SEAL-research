# N24 完整DEV匹配普查：执行前锁定

2026-10-17。状态：PRELOCKED_NOT_EXECUTED。只诊断，不选模块、训练或补救N23。N23已按原门槛STOP，不追加seed、调参或融合。

## 数据与范围

- 固定原DEV187图、380条原GT，原类别、segmentation、crowd/ignore及图像顺序不改。五组为native、冻结强R、A_COMMON、A_ACTIVE、B_C；每组全部18700个自然query，原score、类别、query身份及预测顺序保持。
- 使用已完成预测和官方GT；本阶段不读全量native_L/F缓存，不算Ω oracle，不前向模型。Root执行前绑定全部输入身份和原五组COCO12，禁止按IoU或结果筛图/GT/query。
- 全部380条GT保留记录，包括零IoU、低质量、未匹配及crowd/ignore；未匹配query不冒充真实背景。有效非忽略GT分母另列，不预设等于380。

## 固定计算

1. 沿原official.coco_metrics运行原COCOeval，segm、maxDets=[1,10,100]、原面积定义、无额外score过滤。回调只观察原对象并复制元数据；不改evaluate/accumulate/summarize、排序或匹配算术。五组完整12项stats须精确复现已有结果，任一差异停止诊断并保留原因。
2. 逐图/类别记录原dtIds、gtIds、dtScores、dtMatches、gtMatches、dtIgnore、gtIgnore及官方IoU矩阵；以明确sidecar关联原query/预测ordinal，不能从排序后位置猜query。保存crowd/ignore原因，矩阵含crowd时沿官方语义标注，不与普通GT几何IoU混称。
3. 每阈值0.50:0.05:0.95分别报告原score顺序贪心匹配与最大一对一匹配。top10/top100均按官方稳定score顺序截取，明确是检测数限制，不是GT筛选；all100自然query人口与top10评价分别列示。GT端只取该官方切片的非忽略、非crowd GT；候选端为截取后的全部检测，边为官方IoU>=阈值。不得先按贪心dtIgnore删除候选，因为该标志可能依赖原关联。
4. 最大匹配只用纯增广路求最大基数，不优化AP。遍历固定为官方检测顺序、原GT顺序；输出具体配对仅为确定性代表，最优配对不一定唯一。数量差仅称greedy-association opportunity，不唯一归因score。逐GT最佳IoU/query分别在top10/top100内求；并列按query_id再原ordinal，0及无候选显式保留，不将最佳重叠当模型身份或真实matcher。
5. 从原贪心匹配重建AR100：对all-area/category1/maxDets100，各阈值累计非忽略TP/非忽略GT，再沿官方有效切片取均值，对照stats[8]。最大匹配另报阈值召回，不替代官方AR100。分母0标undefined，不产生NaN；重建不符即停止并定位。

## 输出与解释

输出输入身份清单、五组COCO12复现检查、逐GT最佳重叠/零低质/匹配状态、逐图阈值与top10/100匹配表、总体计数及AR100重建核对；全GT分母和crowd/ignore审计分开。保留query重复竞争、多GT共用最佳query及非唯一匹配说明。原输入不改，结果由独立实际终态及Root接收后才称完成；回调或脚本不得自认证。

若top100最佳IoU仍低于目标阈值，记为现有候选几何覆盖不足；若最大匹配明显高于贪心，记为关联机会；若top10低而top100充分，记为排序/候选预算相关机会。几类可并存，不能由单个差值决定机制。诊断不提供AP上界、部署GT oracle、因果证明或新AP收益，也不证明必须改整网/跨尺度。下一阶段只据完整分布锁一个可否证假设与强通用替代对照，当前不预选网络。
