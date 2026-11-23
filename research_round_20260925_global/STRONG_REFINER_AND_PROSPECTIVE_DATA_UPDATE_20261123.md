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

## 补查：作者轮廓的实际可获得性（2026-11-23）

| 候选来源 | 核验结果 | 判定 |
| --- | --- | --- |
| Sentinel-1 Test01 | 核对作者仓库提交 `b205e83d82f9868a1441118e132e900c5eccee2d` 的目录和 COCO JSON。60 张 800×800 图对应 207 条船舶记录，207 条 `segmentation` 均为空数组。切片记录也没有原始采集产品号。项目说明提及 Labelme 描边，公开文件仅能核验框。 [作者说明](https://github.com/Bighandsome-handsome/SAR-ShipDetection-Project/blob/b205e83d82f9868a1441118e132e900c5eccee2d/README.md) · [公开标注](https://github.com/Bighandsome-handsome/SAR-ShipDetection-Project/blob/b205e83d82f9868a1441118e132e900c5eccee2d/Test01/annotations/instances_val2017.json) | 不纳入船体轮廓评分。 |
| SISS-YOLO（2025） | §3.1 描述作者从 OpenSARShip 2.0 的 24 景 Sentinel-1A 影像挑选切片，用 Labelme 绘制船舶多边形。作者可用性声明指向原始 OpenSARShip，§3.1 也明确原始切片无掩膜。暂未核实新增多边形的公开下载。原文的类别内 80／10／10 分配不能证明母景独立。 [原文及数据声明](https://www.mdpi.com/2072-4292/17/17/3118#sec3dot1-remotesensing-17-03118) | 需要原始多边形、未扩增切片键和采集产品号，才可按母景核查。 |
| S&R-Net（2025） | §IV-A 报告在 41 景 Sentinel-1 的 11,340 个 OpenSARShip 1.0 单船切片上补充实例掩膜，2,318 个切片放入测试。Fig. 5 展示 VV/VH 双极化切片和掩膜。官方原始入口提供切片及 AIS 元数据，未核实新增掩膜公开发布、逐点描边方式或测试切片的母景分离。 [论文](https://doi.org/10.1109/JSTARS.2025.3622241) · [OpenSARShip 官方入口](https://opensar.sjtu.edu.cn/openSAR.html) | 需要原始掩膜、母景产品号、极化配对及标注说明。若仅有栅格掩膜，按像素参考报告。 |
| OSSDD（2026） | 作者 §4 从 VH 强度阈值提取候选连通域、人工剔除误检，随后由旋转框填充最终二值掩膜。此信息与项目先前对公开训练前缀的标签核验一致。 [作者论文](https://arxiv.org/html/2608.01963v1) · [官方数据](https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset) · [实物标签核验](results/OSSDD_TRAIN_PREFIX_LABEL_REPRESENTATION_ACTUAL_SAFE_20261110.json) | 可检验检测盒覆盖。二值字段描述旋转框，不能支撑船体边界结论。 |

本次从原始论文、作者目录与公开标注字段取得资格证据，没有让任何模型读取这些候选来源来挑选研究条件。尚未核实可按采集独立分组、可直接下载的新增船体轮廓。既有 Umbra 双参考验证仍使用预锁定的采集与接收规则。
