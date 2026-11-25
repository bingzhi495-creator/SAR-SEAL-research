# Sentinel-1 Test01 公开标注实件与新采集资格复核

2026-11-26。从作者 GitHub 固定提交读取公开 COCO 标注 62,724 字节及数据说明 4,882 字节，并分别以 Git Blob SHA-1 和 SHA-256 验证。完整 Git 树的 60 个 PNG 成员名与 COCO 图像名逐项相同，尺寸均为 800×800，60 图有 207 条船舶检测记录。公开记录的 207 个 segmentation 均为空数组。初始发布提交和当前提交中的 COCO Blob 相同，其余三个路径历史提交只更新说明。作者提及通过光学资料辅助的 LabelMe 标注，已发布文件可核验的几何内容为框。[固定作者标注](https://github.com/Bighandsome-handsome/SAR-ShipDetection-Project/blob/b205e83d82f9868a1441118e132e900c5eccee2d/Test01/annotations/instances_val2017.json) · [作者数据说明](https://github.com/Bighandsome-handsome/SAR-ShipDetection-Project/blob/b205e83d82f9868a1441118e132e900c5eccee2d/Test01/readme.markdown)

图像记录只有尺寸、ID 和文件名，无法分辨卫星产品和母景。说明文字称影像来自上海与杭州湾多个 Sentinel-1 SLC 时点，但没有裁片至产品的对应关系。COCO 许可证表列 CC BY-NC-SA 2.0，单图记录无许可证字段，仓库元数据未识别总许可证。因此记录作者许可表及限制，不作扩大授权解释。可确认公开框和影像成员的身份，未取得可用实例掩膜或确定可隔离的原始采集组。未下载原始像元、运行模型或读取项目未开放的验证标注。 [机器核验摘要](results/SAR_TEST01_PUBLIC_INSTANCE_METADATA_20261126_SAFE.json)

当前本地真值候选仍按既有证据使用。SAR-SSP 作者两个划分共用 31 个 AIR 母景，同队列已评估。Instance-RSDD 已评估 112 个官方测试母景和 13 个作者训练余留母景。UEIKAP 相对既往初始评分曾追加 13 个母景，但该追加现已评分，且发布的船体有尾迹案例选择限制。HRSID 发行现有图块也都用于源训练、开发或历史重评。 [SAR-SSP 来源证据](SAR_SSP_PROVENANCE_NOTE_20261125.md) · [Instance-RSDD 余留实测](INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_20261123.md) · [UEIKAP 新增家族](UEIKAP_ADDITIONAL_FAMILIES_ACTUAL_20261123.md)

对新增官方渠道复核时，LS-SSDD 作者说明仍指向 v1 框标签，分割目录在论文里为未来更新预留。PID-SAM 作者 Git 树未列 FSS-Seg 或可下载数据资产。MOS-Ship 的原文明确 DOTA 四点旋转框，并将 SAR 重采样到光学网格。没有找到满足本轮冻结实例评估的新公开集。现已配好 12 个 Umbra 独立采集的双盲评材料，待两份原始 JSON 各自回传，再按原接收和评分接口验证。 [LS-SSDD 作者说明](https://www.mdpi.com/2072-4292/12/18/2997) · [PID-SAM 作者仓库](https://github.com/Fm336/PID-SAM) · [MOS-Ship 作者标注](https://www.mdpi.com/2072-4292/18/2/274#sec2dot1dot3) · [Umbra 评分入口](code/UMBRA12_DUAL_REVIEW_SCORE_20261123/README.md)
