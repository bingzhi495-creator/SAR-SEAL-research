# AP75 临界掩膜的保护对象与拟合支持

上一轮将 SSDD 主组的 AP75 下降定位为匹配边跌破门槛，涉及三个原真阳性。单纯提高平均正确翻转可能改变这些临界实例。先检验失守边在原生掩膜上是否已达标，还是只有逐点 BCE 修补后才达标。两种保护对象分开计数。保留原 COCO 匹配、置信度、候选与同一组历史角色。逐项复现三个旧 AP75 端点和已验收匹配转移。原失守边的来源使用同一检测和同一真值实例核验。 [上一步实测](RFDETR_FIT_AP75_MATCH_ATTRIBUTION_PROTOCOL_20261109.md)

同一个四补四删动作中，正确编辑数 C 在零到八之间。用原生相交像素 I、并集像素 U 定义 S=4I−3U。编辑后裕量等于 S+7(C−4)，所以保持或达到 IoU 0.75 至少需要 max(0,4−floor(S/7)) 个正确编辑。给定每侧的正确点数和总点数，可精确枚举选四后的最少及最多命中数。只有规则允许的选择中既有通过又有不通过时，才认为该训练候选存在排序可控制的失守。

在原 290 图点拟合角色内部按编码组分五折。每折只用其余四折拟合相同 BCE、头、种子和 96 步预算，再在排除组上得到同预算动作。统计原生达标、交叉拟合 BCE 达标及仅 BCE 达标三类的可控实例数和内容组数。折间没有同编码图。冻结的原检测器仍见过 FIT，各角色结果只说明训练信号是否存在。该统计不把模型的拟合内预测当成部署级锚点。

历史 114/111 图只用于解释已有失败。此轮不新拟合拟议保持损失，不改动作、门槛或正式评估权限。既有 SSDD 分组均有曝光，HRSID DEV705 若用于后续跨集开发评估，须在打开方法结果前固定配方并核对接触台账。实例分割局部细化与保护已有正确预测均有先例，差异候选只限于由固定预算像素代数决定的可达保持条件。 [RefineMask](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html) · [Positive-Congruent Training](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html)
