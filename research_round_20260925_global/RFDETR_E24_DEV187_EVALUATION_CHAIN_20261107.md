# E24 的固定 DEV187 导出和评价链

2026-11-07。已用验收过的原图归档和原生布局记录，对齐 187 个固定图像位置及原始高宽。187 个位置含 183 份不同 JPEG 字节，保留原人口和顺序。私有图像清单不含实例标注。它将用于 E24 归档验收后的单次 GPU 导出。当前只有元数据核验和合成接口测试，真实 E24 预测和 AP 待运行。

GPU 原生路径固定使用第 4,464 步归档的 EMA、FP32 eager、官方 576 图像变换和原图大小的后处理。先对 200 个查询与类别组合做原生稳定排序，保留 100 行，再按槽位 0 映射船舶类别。完整保留已有顺序与分数，零预测图留在索引。第一、第二张固定图在同一原始进程中先检查 RLE 回读，随后原进程继续全 187 张。推理进程按明确清单读取图像，禁止访问 GT。输出按图流式保存，分开记录读图和变换、同步的前向、后处理与 RLE、写盘时长。 [导出工作体](code/RFDETR_NATIVE_E24_DEV_EXPORT_WORKER_CANDIDATE_20261107/README.md) · [RF-DETR 原生后处理](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py)

CPU 评价先用图像顺序、候选排名、每图字节摘要和计数校验导出，再在原始像素网格对 380 条 GT 做 COCO segm。整数标注为主，候选浮点标注单列，两版各保留原始 GT 和全部图像。诊断分开报告已导出映射候选的最高掩膜 IoU 覆盖，以及整数 COCO 分配后沿用同一预测行得到的两版几何误差。前者的范围是原生选出后实际映射的候选。排序前的全部查询能力需要另行输出和检查。后者是条件配对，不能解释整集 AP 的所有变化。原生在线评价还会缩放变换后的 GT，因此其数值需独立列示。 [分组桥](code/RFDETR_NATIVE_DEV187_OUTPUT_BRIDGE_CANDIDATE_20261107/bridge.py) · [CPU 适配器](code/RFDETR_NATIVE_DEV187_CPU_EVAL_ADAPTER_CANDIDATE_20261107/adapter.py) · [原生在线回调](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/callbacks/coco_eval.py#L1424-L1517)
