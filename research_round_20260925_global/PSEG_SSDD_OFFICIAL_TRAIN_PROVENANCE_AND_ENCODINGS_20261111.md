# PSeg-SSDD 作者发布的训练标注来源与两种坐标编码

此前固定 SSDD DEV187 的两种子结果对标注版本敏感。以当前项目整数多边形评分，中心采样相对原生的 mask AP 差值为 +1.680 和 +0.806 点。使用归档小数坐标重新评分，差值为 −3.113 和 −3.897 点。本轮核查这两类几何编码是否确实存在于作者发布的训练标签。 [已有配对与标注敏感性](SSDD_DEV187_GEOMETRY_AND_GT_SENSITIVITY_20261111.md)

## 三段作者归档的远端原字节证据

使用作者仓库的公开 Drive 发布，正常 HTTP 范围请求得到三次 206，完整对象大小均为 1,043,241,197 字节。对本地候选 RAR5 逐个读取并验证 32,658 个记录头 CRC。只对所选训练记录读取压缩负载。合并条件为归档中相邻且同属获准训练目录，无跨记录跳读。两套 XML 加一份 COCO 的 1,857 条记录头与压缩负载远端、本地逐段 SHA-256 相同。未依赖其他压缩成员解压这些记录。 [作者发布入口](https://github.com/TianwenZhang0825/Official-SSDD) · [作者 Drive 文件](https://drive.google.com/file/d/1glNJUGotrbEyk43twwB9556AdngJsynZ/view) · [RAR5 公开格式](https://www.rarlab.com/technote.htm)

| 记录 | 数量 | 远端实取字节 | 解压字节 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 合并版 COCO 训练表 | 1 | 374,884 | 1,526,458 | SHA-256 相同 |
| 合并版 VOC TRAIN | 928 | 751,443 | 2,501,632 | SHA-256 相同 |
| 单独 PSeg VOC TRAIN | 928 | 428,584 | 1,231,959 | SHA-256 相同 |

本轮认证范围是已比较的训练成员。整包其他成员、图像及 TEST 没有重新认证，未读取正式 TEST 负载、原图像素或运行模型。已验收的本地 COCO 训练成员解压 SHA-256 为 `4565548ce05136dba8072a4b1c1fb668f3d415e3c980ebfe05b5a8998dec1ac5`，与项目现有小数训练源逐字节相同。 [远端范围实测](results/PSEG_SSDD_OFFICIAL_TRAIN_RANGE_MATCH_ACTUAL_SAFE_20261111.json) · [本地训练成员](SSDD_DEV187_GEOMETRY_AND_GT_SENSITIVITY_20261111.md#本地候选归档的-train-标注字节核验)

## 同一训练人口的坐标证据

两套 VOC 各有 928 图、2,041 个船实例、27,873 个顶点，实例和顶点顺序及整数坐标全部一致。原 COCO 明确船类 ID 为 0，有同一批 2,041 个实例。其中 1,972 个包含非整数值，55,746 个坐标中的 53,090 个为非整数。直接比对只有 69 条轮廓相同。将 COCO 每个非负坐标向下取整或截断，2,041 条均按原实例与顶点顺序精确重现 VOC。两种四舍五入规则各只命中 69 条。项目当前整数多边形因而与作者 TRAIN VOC 的顶点相同。原 COCO 的浮点轮廓也是作者实际发布的另一种编码。这个观察无法确定上游标注转换的先后关系。第一次读取中将类别 ID 假定为 1，修订按原声明的 0 重新运行，最终独立原进程退出 0。 [逐顶点比较](results/PSEG_SSDD_TRAIN_VOC_COCO_VERTICES_ACTUAL_SAFE_20261111.json)

另从远端验证了四个可视化和转换脚本的压缩记录。VOC 可视化按整数顶点调用 OpenCV 填充，并在背景影像上绘制边框。COCO 可视化调用 PyCOCO 的 `showAnns`。这些程序说明发布者实际提供了两类展示入口。正式评分的像素面积与坐标约定仍应由明确的标注文件和统一评估程序锁定，展示脚本无法替代。 [作者论文 §3.3](https://www.mdpi.com/2072-4292/13/18/3690#sec3dot3-remotesensing-13-03690) · [程序记录远端核验](results/PSEG_SSDD_OFFICIAL_PROGRAM_RANGE_MATCH_ACTUAL_SAFE_20261111.json)

下一次 PSeg 跨库对照应明确区分原始发布的 COCO 浮点标注与 VOC 顶点对应的项目整数 COCO 口径。固定模型、预测及评分器后报告两套指标，不能按结果选择版本。现有整数优势与小数劣势均保留。正式 TEST 的标注编码及评估资格仍待单独核验。 [候选正式划分与已知重复](U57_INTERNAL_DATASET_PROTOCOL_TRACE_20261024.md#候选归档划分与测试图像重复审计)
