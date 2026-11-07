# HRSID 原字节输入与既有完整像素核验的对齐

2026-11-07。逐项比较两个已关闭阶段的指纹记录。当前 v04 JPG 输入与此前 U43 实际解码使用的 **5,604 个逻辑文件**按原文件名对齐，全部文件长度和 SHA-256 一致，两份 train/test 标注 JSON 的长度与 SHA-256 也一致。U43 保留了原退出 1 和全人口访问证据，发现 13 个空栅格结果，其中训练 9、测试 4。输入同一性支持复用该运行的证据范围。此次没有重新读取图像或标注正文、训练或评价。 [v04 JPEG 输入](HRSID_JPG_INPUT_AUDIT_STATUS_20261107_04.md) · [U43 实际记录](results/U43_CANONICAL_POPULATION_EMPTY_RASTER_FAILURE_ACTUAL_20261018.json)

按存档 JPEG 编码字节比较，SSDD FIT741 和 DEV187 与 HRSID 原 train3642、test1962 的四组 SHA-256 交集均为 0。HRSID 自身有两个跨划分相同编码字节组。该检查的范围是精确编码副本，像素近似重复、母景和传感器来源重叠尚未核验。HRSID 的 JPG 副本与作者 PNG 的像素等价性也仍待证实。 [作者数据说明](https://github.com/chaozhong2010/HRSID#dataset-download-link) · [标注版本差异](HRSID_ANNOTATION_PROTOCOL_DIFF_STATUS_20261107_05.md)

未来固定跨集评价用原 `test2017.json` 的 1,962 图和 5,922 条标注，保留空栅格记录和 COCO 原有语义。旧 U43 在 Windows 上使用的编解码扩展与当前 DGX CPU 评价扩展字节不同。正式计分前应以选定运行时核对已知空栅格记录，并预锁任何另报的敏感性视图。历史 HRSID 测试已用于部分方法和配置选择，应报告为固定配方的跨集评价。若之后从 SSDD 冻结端点直接迁移，须另核对模型训练谱系与目标数据接触，再说明零目标调参的范围。 [空栅格规则核查](U43_EMPTY_RASTER_COCO_PROTOCOL_REVIEW_20261018.md) · [历史测试用途](HISTORICAL_TEST_USE_LEDGER_TEXT_EVIDENCE_20261016.md) · [安全汇总](results/HRSID_U43_V04_INPUT_IDENTITY_ACTUAL_SAFE_20261107.json)
