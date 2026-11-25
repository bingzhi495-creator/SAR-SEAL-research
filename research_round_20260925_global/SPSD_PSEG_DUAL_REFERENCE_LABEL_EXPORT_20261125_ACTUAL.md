# 双参考训练标注导出实测

从已验收的 915 张训练图及锁定的 901 个精确解码内容组导出两套私有 COCO 掩膜。FIT 和源域内部验证分别保存。两边使用 PSeg 原始影像 ID、文件名和原生尺寸。训练只需读取 FIT 标签与其输入绑定。原图、逐像素标注、逐图哈希和 ID 映射均留在私有产物中。 [预锁](code/SPSD_PSEG_DUAL_REFERENCE_LABEL_EXPORT_20261125/PRELOCK.md)

| 角色 | 图像 | 解码内容组 | PSeg 船体 | SPSD 船体 |
| --- | ---: | ---: | ---: | ---: |
| FIT | 732 | 721 | 1,579 | 1,580 |
| 源域内部验证 | 183 | 180 | 411 | 410 |
| 全部已准入 | 915 | 901 | 1,990 | 1,990 |

PSeg 保留原实例 ID、面积、边框、类别及 crowd 字段，分割字段以作者浮点多边形的原生 COCO 栅格 RLE 保存。SPSD 对原 RGB 实例 ID 计算正面积 RLE 和同栅格紧框，以共享影像 ID 归属，由原影像 ID 与原船体 ID 的顺序分配唯一实例编号。逐图核对原始元数据、轮廓或 PNG 字节、RLE 集合及导出行散列与已验收训练名册一致。

正式导出进程和随后单独的 COCO 读回均实际退出 0。独立读回复算 PSeg 浮点多边形的 RLE，证明其评分元数据等同原件。SPSD 每个 RLE 的面积和紧框通过复算。两个角色间编码图像、解码图像组和原影像 ID 交集均为零。没有读取历史测试图像或标签，未执行模型推理或优化。母景身份仍未核验，内部验证只支持解码图像组范围。

[导出工作程序](code/SPSD_PSEG_DUAL_REFERENCE_LABEL_EXPORT_20261125/export_train_labels.py)读取原作者 TRAIN 标注及已锁定的训练清单。运行时保留两份作者档案、准入和划分回执、各自的私有清单以及预锁。通过程序参数说明指定已有产物和新输出路径，并在使用前核对安全汇总和独立读回。

[汇总](results/SPSD_PSEG_DUAL_REFERENCE_LABEL_EXPORT_20261125_SAFE.json) · [独立读回](results/SPSD_PSEG_DUAL_REFERENCE_LABEL_EXPORT_20261125_READBACK_SAFE.json)
