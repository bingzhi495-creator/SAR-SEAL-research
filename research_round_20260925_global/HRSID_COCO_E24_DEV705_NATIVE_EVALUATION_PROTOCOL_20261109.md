# HRSID COCO 初值 E24 的 DEV705 固定评分

仅评估原生训练预算验收后的 E24 EMA。入口检查实际训练提交、COCO 来源、164 查询初值、FIT 索引、E100 调度和 17,640 次更新的归档标记，严格重建同宽模型。E1/E8 留作固定过程归档。正式 TEST 不进入此次模型选择或评估。 [E24 训练锁定](HRSID_COCO_NATIVE_FIT_E24_PROTOCOL_20261109.md) · [DEV705 审计](HRSID_OFFICIAL_PNG_DEV705_AUDIT_PROTOCOL_20261109.md)

推理进程只读取预锁的 705 个官方 DEV PNG 成员，每图核对原成员 CRC、编码和解码后的 RGB 指纹。固定原生 576 输入、FP32、关闭 TF32，掩膜按原 800×800 编码。保留原生跨两个分类槽的稳定 top100，随后映射船舶槽 0 至 COCO 类别 1，丢弃未映射的槽 1。除原生选择外无置信度过滤。记录每图 164 个原始查询、100 条排序输出及映射后数量。导出进程不读取评估 GT。 [原生 PostProcess](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py)

独立确认导出关闭后，CPU 进程使用 pycocotools 2.0.11。主评估保留全部 1,782 条原始 GT，含 213 条分数顶点及 4 条空栅格记录。敏感性评估只排除这 4 条，同一预测流和图像分母均保持。预测使用 COCO RLE，掩膜预测面积由编码计算。报告 AP、AP75、AP90、小目标 AP、AR100、映射数量和推理成本。开发角色来自官方 TRAIN 的预锁划分。独立模型和外部方法的数字必须按数据、训练和评价范围区分。 [COCO 作者工具](https://github.com/cocodataset/cocoapi)

四项固定运行时合成测试通过，覆盖 164 查询下与原生排序及 RLE 的一致性、并列分数顺序、空预测、GT 敏感性和错误模型来源拒绝。真实推理和评分等 E24 原训练验收后执行。 [评分源码](code/HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py)

## 修复版 E24 接入

原评分入口严格核验原端点字段集合。修复版增加过滤源码摘要与累计空目标数，直接送入旧入口会被拒绝。独立兼容层对原评分程序锁定源码字节，并要求独立 E24 原进程验收产生的 SHA-256 锁定回执。回执需确认修复协议、原退出 0、24 轮及 17,640 次优化与 EMA 更新、无资源中止、精确过滤源码摘要、实际累计空行数和 E24 权重摘要。两种评分模式都用同一验收回执和权重摘要。只在内存里把两项新字段核验后交回原检查器，其余权重、配置、时钟、导出和计分规则沿用原实现。

指定 Windows Python 与 pycocotools 环境实际通过 4 项新接口测试和 3 项原评价回归测试。原评分程序、训练源码和运行资产均未修改。实际评分仍待原训练闭合及独立验收。 [兼容入口](code/HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py) · [测试](code/HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/test_hrsid_native_dev705_filtered.py)

