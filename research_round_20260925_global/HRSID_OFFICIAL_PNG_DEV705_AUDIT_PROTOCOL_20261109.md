# HRSID 官方训练集内 705 张开发图的输入审计

开发集来自既有预锁 HRSID 官方 TRAIN 角色划分。全包官方 TRAIN 3,642 张与 11,047 条标注。FIT 2,937 张，开发角色 705 张、1,782 条原始标注。109 个 FIT 与 28 个开发 P 前缀组互斥，前缀为原母景的代理。官方 TEST 历史接触边界见独立记录。 [作者数据](https://github.com/chaozhong2010/HRSID) · [角色与历史](HRSID_CROSSSET_EVIDENCE_BOUNDARY_20261108.md)

只打开官方 PNG 归档内的原 `annotations/train2017.json` 及预锁开发角色的 705 个 PNG 成员。核对全包回执与中央目录、成员 CRC、原始尺寸和解码像素。比较训练角色已闭合的编码与 RGB 指纹，两类重复分别计数。保留开发角色的全部原始 annotation ID、面积、浮点顶点及其它字段，逐条调用 pycocotools 2.0.11 栅格化并单记空实例。原始开发 GT 及图像清单仅本地存放。训练像素和正式 TEST 成员均在本轮范围外。

原始全记录评分口径预先固定。若需要另报有效栅格敏感性，同一组预测和目标角色下独立报告。当前为输入审计，未运行模型、训练或 AP。三项合成测试通过，覆盖只读范围、浮点及空 GT 保留、CRC 错误和像素重复。 [审计源码](code/HRSID_OFFICIAL_PNG_DEV705_AUDIT_20261109/hrsid_png_dev_audit.py) · [合成测试](code/HRSID_OFFICIAL_PNG_DEV705_AUDIT_20261109/test_hrsid_png_dev_audit.py)

## 实际输入审计与独立读回

原进程退出 0，工作体用时 14.60 秒。全部 705 个开发 PNG 通过成员 CRC 和 800×800 像素解码。编码和 RGB 指纹各有 705 个唯一值，两种指纹与 2,937 张 FIT 均无精确重复。28 个开发 P 前缀与 FIT 的 109 个前缀互斥，该检查保留母景代理的局限。原始 1,782 条标注完整保留，其中 213 条含分数顶点，1,778 条得到非空掩膜，4 条为空。独立重读官方 TRAIN 正本，全部开发图像与标注字段逐项一致，独立重算的空栅格 ID 集一致。正式 TEST 未打开，模型和 AP 未执行。 [审计汇总](results/HRSID_OFFICIAL_PNG_DEV705_AUDIT_ACTUAL_SAFE_20261109.json) · [独立读回](results/HRSID_OFFICIAL_PNG_DEV705_AUDIT_READBACK_SAFE_20261109.json)

