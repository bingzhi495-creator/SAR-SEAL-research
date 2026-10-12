# N06 固定强 E24 一对一误差分析协议

预锁日期：2026-10-12。本协议在读取本次误差结果前保存。分析历史 DEV 的固定输出，不训练、不修改分数/阈值/后处理、不选择新权重；不能称正式未接触测试或方法效用证明。

## 输入与执行

- 合格 E24 native0 固定 ALL_PREDICTIONS 和原始 DEV187图/380实例/category1；全部100候选保留，GT crowd/ignore由绑定官方 COCO SOURCE实际语义处理。来源经原固定导出重算资格记录连接。
- CPU01/线程2、CUDA不可见；NumPy1.23.5、绑定pycocotools SOURCE与实际binary。单一所有者、隐藏后台、1800秒上限，无自动重试。16个相关正常科学RAW与已保存CPU接口故障RAW先行核验，再核验输入/源码绑定与导入科学库。
- 完整COCO mask AP的12项统计与已固定E24重算结果比较，容差1e-12；保留原imgIds、category、maxDets[1,10,100]、area和10 IoU阈值[.50:.05:.95]。本文不新增效用指标选择。

## 实际匹配与计数

一对一关系取官方area=all、maxDet100的evalImgs gtMatches/dtMatches/gtIgnore/dtIgnore，不能用可复用GT最优查询替代实际召回。COCO各area执行会改变GT字典的_ignore，取all对应的evalImgs数组，避免误用最终large状态。

所有10阈值统计TP/FN/FP/ignore；额外固定前K=1/10/100的score稳定排序前缀。同一greedy匹配前缀不被后续低分候选改变；前K的GT只在对应匹配DT位于前K时计入匹配。主要详细记录IoU0.50和0.75，保存所有未匹配GT与DT。固定score/rank统计仅描述，不搜索最佳过滤阈值。

FP诊断只对实际未ignore、未匹配DT，按其到所有all口径非ignore/noncrowdGT的最大mask IoU互斥划分：

- maxIoU >= 当前匹配阈值：duplicate_or_assignment_conflict（重复或匹配竞争候选）。
- 0.10 <= maxIoU < 当前阈值：geometry_candidate（几何不足候选）。
- maxIoU < 0.10，或没有可评GT：background_candidate（背景候选）。

这些是可复核操作标签，不是TIDE完整误差工具或AP反事实因果分解；不能由类别数量直接推断错误对AP的贡献。100查询尾部大量低分背景预测需与前10数量区分。

GT记录实际一对一匹配、未匹配、忽略；对0.75未匹配GT标记其是否在0.50实际匹配。可复用query的bestIoU只作为辅助覆盖，不称实际召回。匹配TP保存mask IoU、Dice及像素TP/FP/FN，GT与预测同原图shape解码。

## 分层与产物

- 图中实例数固定分组：0、1、2–4、>=5；使用可核验GT信息，不猜测岸线、海况或传感器。
- 描述性互斥面积组：area<=1024、1024<area<=9216、area>9216；明确这是分层表的互斥口径。官方COCO APS/APM/APL依旧使用其标准重叠闭区间，二者不得混淆。
- 写REPORT和DETAILS：完整人口、12统计、各阈值/前K计数、分层、完整匹配与未匹配、异常状态和证据边界；固定输入与实际运行来源可连接。图像清单定位供解释，不挑最好图替代总体。

## 决策

执行成功后先报告观察，再提出解释与竞争解释。按真实错误的频率、影响和可利用信息只选择一条新假设，查新其直接先例并预锁强简单替代。分析完成不保证找到可改进机制；证据不足时保留结果，总目标仍未完成。SOURCE检查、12项复算一致和误差计数均不能单独支持创新或TGRS投稿结论。
