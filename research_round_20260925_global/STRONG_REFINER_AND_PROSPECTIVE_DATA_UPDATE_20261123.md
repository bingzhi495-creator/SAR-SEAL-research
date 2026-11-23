# SAR 船舶实例边界精修对照与独立数据核验（2026-11-23）

本次核对作者论文、代码、数据说明和权重链接。未下载数据或权重，未运行模型。以下记录不能用于比较 SAR 精度或给方法排名。

## 已核实接口与权重入口，待实测：SegRefiner-LR

Wang 等（NeurIPS 2023）的 COCO 实例推理配置使用原图、粗二值实例掩膜、检测框及其分数和类别。现存图像和冻结检测输出可以提供这些输入，无需重新提取检测器特征。作者发布的 LR 权重端点经 HTTP HEAD 核验返回 200，附件名为 `segrefiner_lr_latest.pth`，声明长度为 474,603,251 字节。权重内容、哈希及本地运行兼容性仍待核验。[论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Abstract-Conference.html) · [作者配置](https://github.com/MengyuWang826/SegRefiner/blob/53419a2d38ea3da0b6e2be77e5b45e139195a0b3/configs/segrefiner/segrefiner_coco.py) · [作者权重入口](https://drive.google.com/file/d/1FrhbdwNyTlQYNbF9IgFF2tY3iQXHFcau/view)

默认门槛是原图上**粗掩膜的前景面积 ≥512 像素**。合格实例从掩膜外接框向各边扩 20 像素，裁至图像范围后缩放至 256×256。精修输出贴回原图后按 `≥0.5` 二值化。低于门槛的掩膜原样输出。代码先拼接合格实例，再拼接小实例，检测框和分数同序移动。若接入本项目，应按原候选编号恢复顺序，保留同分排序，报告全量候选中的合格数及固定匹配船体中的合格数。取消 512 门槛或另加固定编辑数，应单列为改编实验。[作者筛选、拼接及回贴代码](https://github.com/MengyuWang826/SegRefiner/blob/53419a2d38ea3da0b6e2be77e5b45e139195a0b3/mmdet/models/detectors/segrefiner_instance.py#L25-L93)

## 已核实方法，所查权重链接失效：Boundary Patch Refinement（BPR）

Tang 等（CVPR 2021）直接用图像和已有实例掩膜提取边界小块。作者单图示例默认截取 64×64 区域，放大至 128×128 推理。重叠处取前景分数的算术平均，仅对覆盖像素以 `>0.5` 更新。作者 README 中的三个权重入口（Cityscapes 两个、COCO 一个）以及维护者在问题 #47 补充的两个入口均返回 `Link does not exist.`。本次未取得可核验的作者权重，文中可引用其作为边界局部更新的先例，尚无本项目 BPR 实测结果。[论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.pdf) · [单图实现](https://github.com/chenhang98/BPR/blob/9eafc3f1f6f6c4cfcdd0ceaf9f42bb2d07f90f4f/demo/inference_img.py#L137-L186) · [权重表](https://github.com/chenhang98/BPR/tree/9eafc3f1f6f6c4cfcdd0ceaf9f42bb2d07f90f4f#models) · [维护者补充链接](https://github.com/chenhang98/BPR/issues/47)

## 数据资格：本次未确认新增独立轮廓基准

| 来源 | 作者材料所能确认的事实 | 本研究可用范围 |
| --- | --- | --- |
| MariSat（2026） | 1,260 张图像，作者从 OpenAerialMap 选择光学正射影像，结合 SAM 3 和 CVAT 人工校正生成实例标注。[作者论文 §2.1–2.4](https://arxiv.org/html/2608.29852v1) | 有船舶实例轮廓。影像模态不满足原生 SAR 评估条件。 |
| SDFSD-v1.0（2024） | 846 景亚米级 SAR 影像，含 211 景 Umbra。作者在整景影像上绘制矩形矢量框，再切片。[作者论文 §2](https://www.mdpi.com/2072-4292/16/21/3952) | 能用于检测研究。现有矩形不能充当船舶轮廓真值。 |
| Umbra Open Data | 提供港区等地点的 SAR 时间序列。官方 STAC 规范定义了采集标识和孔径中心 UTC 时刻。[官方数据入口](https://umbra.space/open-data/) · [采集元数据规范](https://docs.canopy.umbra.space/docs/collect-metadata) | 可按采集标识预先锁定整景并组织盲法描边。独立性须核对采集重叠，尤其是已使用 Umbra 数据的对照。此项是候选采样方案，尚未形成有真值的基准。 |
