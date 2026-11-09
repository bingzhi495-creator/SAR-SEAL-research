# 固定四补四删的多阈值动作目标

484 图实测中需要七或八次正确翻转的 AP75 可达配对有 7 个，两套 BCE 均只实现 1 个。扩训练池在同候选仅多救回一例，AP75 专门保持项的难例支持条件未通过。下一假设是在固定八次动作中，用 COCO 十个既定阈值共同提供训练信号，检验在哪些实例中正确翻转的效用呈阶梯变化。原 AP75 评估要求保留。 [训练支持结果](RFDETR_FIT_AP75_BROAD_SUPPORT_PROTOCOL_20261109.md)

给定原预测与真值面积 A 和 G、原交集 I，以及实际正确翻转数 H，动作后交集等于 I+H−4，预测面积保持不变。阈值 m/20 对应整数判定 (20+m)(I+H−4)≥m(A+G)，m 为 10 到 19。对单个原匹配定义 U(H) 为通过的固定阈值数。在每个可达的两侧命中数组合内，分别取相应个数的最高分正确与错误点，得到该组合最大分。最多 25 个组合。将相对于最好可达阈值数的缺口加到动作分上取最大，再减所有达最佳阈值数动作中的最高分。差除以八，得到结构训练项。若所有可行动作的 U 相同，附加项置零。

关键对照在同一候选和同一求解器上用线性命中数缺口。线性项的最大惩罚逐候选匹配阈值项的最大惩罚，共用候选权重和损失系数，便于区分阈值阶梯与样本筛选或权重。保留原 BCE 和全量前四排序。点输入、头、训练预算与四补四删保持可比。真值只用于训练目标和独立计分。P166 在固定位置排序上学习补删比例，本项保持补删各四个，学习位置分数。Precision@k 结构替代损失以及实例质量排序已有直接先例。 [Precision@k](https://proceedings.mlr.press/v37/kar15.html) · [Rank & Sort](https://openaccess.thecvf.com/content/ICCV2021/papers/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.pdf) · [COCO 评估器](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) · [P166 配额结果估计](RFDETR_FIT_QUOTA_OUTCOME_PROTOCOL_20261109.md)

当前完成损失原型和小动作穷举核对，尚未执行真实拟合。下一步在原 484 图编码组五折上预先固定完整驱动和推进条件。报告阈值数真正变化的独立内容组、阶梯与线性差异、逐阈值得失及同组配对区间。历史 114/111、DEV 与 TEST 不参与选型。固定匹配下的阈值命中属于机制指标，最终效用以独立角色完整候选 COCO AP 核验。
