# RF-DETR SSDD 原生输入布局实际结果

2026-11-07。将既有独立封存的 FIT741 与 DEV187 原始输入归档传至 DGX 新目录。实际工具和 SSH 均退出0。构建器逐个核对两份归档 SHA-256、成员 CRC 与 SHA-256、清单身份、COCO 图像顺序、尺寸和类别。每份写入原生目录的 JPEG 与源内容重新比较 SHA-256，两份原标注直接以原字节保存为各划分的 _annotations.coco.json。原图与标注均留在执行侧。

| 原生目录 | 逻辑图像 | COCO 实例 | 按编码字节去重的 JPEG |
|---|---:|---:|---:|
| train 对应 FIT | 741 | 1,661 | 730 |
| valid 对应 DEV | 187 | 380 | 183 |

两组图像 ID 和 JPEG SHA-256 的交集均为零，类别均为原始船舶 category 1。DEV 是已有内部验证集。编码字节去重支持这一层的隔离检查，场景级身份仍待独立核验。本次没有构造 test 目录或读取官方测试组。[FIT 来源](U57_FIT_DGX_INPUT_TRANSFER_20261107.md) · [DEV 来源](DEV187_DGX_RAW_INPUT_STAGE_STATUS_20261107_06.md)

[布局构建器](RFDETR_SSDD_NATIVE_INPUT_LAYOUT_DRAFT_20261107_01/build_native_view.py)只读取明确指定的已选归档及新输出。下一阶段的[原生 DataModule 探针](RFDETR_REAL_SSDD_DATA_QUALIFICATION_DRAFT_20261107_01/probe.py)将比较全部划分 ID、检查类别 1 到训练标签 0 的双向映射，并对每组预先固定一个正例核对原图、COCO 栅格和 576 原生变换。本次仅完成布局的实际字节验收。DataModule 尚未执行，SAR 训练和项目 AP 待后续实测。

后续同一原生布局已完成[两个固定真实样本的数据入口实测](RFDETR_NATIVE_SSDD_DATA_ENTRY_ACTUAL_20261107_01.md)。
