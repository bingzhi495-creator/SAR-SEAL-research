# HRSID FIT 上的原生 164 查询与候选保留对照

固定已验收的 HRSID COCO-only E24 EMA、原生 576 预处理、FP32 与关闭 TF32。仅使用独立核验的官方 PNG FIT2937 与同一派生 COCO 标注。此前 DEV705 的原生 top100 映射后仅覆盖 104/1,778 个严格目标。SSDD 的先例在 100 查询条件下，原映射与全查询的 0.90 一对一覆盖相差一个，不能据此判定 HRSID 的 164 查询结果。 [HRSID 冻结误差普查](HRSID_DEV705_NATIVE_CANDIDATE_COVERAGE_PROTOCOL_20261109.md) · [SSDD 原始查询结果](RFDETR_E24_RAW_QUERY_GEOMETRY_ACTUAL_20261107.md)

主样本覆盖全部 109 个原分区母景代理前缀，每组取 `SHA-256(b"HRSID-FIT-Q164-P192-v1\\0" + PNG 编码 SHA-256 的原始 32 字节)` 最小者，并以文件名打破并列。以派生标注计数超过 100 的五图为独立压力层。交叠图共享一次前向，两个层分别聚合。导出不读 GT 正文。固定 RGB 指纹和模型后，每张图前向一次。原生两槽 top100 使用原 PostProcess；全 164 船舶槽用同一个 PostProcess 和相同原图栅格。逐图核对原生已映射船掩膜的查询编号、分数、顺序和 RLE 全部一致。

四组使用同一个候选掩膜池。原组保留两槽 top100 过滤非船槽后的 m 个候选。分数对照取船舶槽前 m 个。主要等量对照仍用船舶分数顺序，先保留非空掩膜，不足 m 才以空掩膜补齐。上界组保留全部 164 查询。前三组逐图船舶候选数相同，所有组保留原分类分数。空掩膜替换属于普通筛选对照。RF-DETR 作者的全局两槽选取与 Mask2Former 作者的前景得分及掩膜质量评分说明候选筛选和排序已有直接先例。 [RF-DETR 固定版本](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py) · [Mask2Former 原文](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html)

仅在原导出工作进程关闭并核对完整字节摘要后，CPU 打开 FIT GT，按空、crowd、ignore 互斥排除。报告 0.50、0.75、0.90 的独立、最大一对一和原排序贪心覆盖，严格独立覆盖的增加、失守、原最佳落在 [0.75, 0.90) 的恢复、空候选数和成本。主决策用 0.90 最大一对一覆盖，并用固定种子 5209 的 4,096 次图组配对重采样计算每有效 GT 的差异区间。

全 164 若仅挽回少于原严格一对一缺口的 5%，或增量只分布于少于 10 个主图组，停止本轮选择不足假设，后续考察固定候选几何。若通过上界门，等量非空组须比原组增加至少 2 个百分点的严格覆盖、图组配对区间下界大于零，且 0.75 覆盖不降，才进入固定筛选对照的前瞻评价。分数对照用于判断收益能否由普通前景排序解释。上界有空间而等量组未过门时，保留选择假设但停止这条筛选规则。压力层不参与门槛。FIT 图已进入训练，本轮无 AP、新训练、DEV 或 TEST，也不主张跨数据集效用。
