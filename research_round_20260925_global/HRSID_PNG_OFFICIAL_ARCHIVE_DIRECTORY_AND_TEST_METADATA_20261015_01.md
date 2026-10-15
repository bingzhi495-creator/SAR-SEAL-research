# HRSID官方PNG压缩包：完整目录与TEST标注元数据实核

2026-10-15。独立的数据来源准备，不属于N20运行数据、训练或评价。本记录只报告已实际取得并由Root复核的目录与JSON元数据；没有下载整包、读取图像像素、解码mask或计算AP。

## 官方入口与取得范围

来源为[作者仓库固定提交README](https://github.com/chaozhong2010/HRSID/blob/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/README.md)指向的[官方PNG包](https://drive.google.com/file/d/1xgXi8KC3MDWuu7Yp4n2J-LOPYLRszHAc/view)。公开文件标题为HRSID_png.zip，远端声明总长5,698,302,315B，Last-Modified为Wed, 01 Jul 2020 15:10:52 GMT。这是远端元数据，不是全包下载或全包SHA认证。

仅以有界HTTP Range取得ZIP尾、完整central directory、指定JSON所需部分。Root独立解析ZIP64目录并核TEST文件SHA256、CRC32和JSON关联；没有运行下载包内程序。完整目录672,944B，SHA256为4a54303e8cce24f0330cddd2e6d3c12205c8b0002859d0b5bf534efb531a2dce。目录包含5,612个条目：5,604个PNG、5个JSON、3个目录，全路径无重复。条目存在不等于对应PNG像素已取得或验证。

| JSON成员 | 目录声明解压后字节 | 实际完整取得 | CRC32 |
|---|---:|---|---|
| inshore_offshore/inshore.json | 9,566,861 | 否 | b4e1d56c |
| inshore_offshore/offshore.json | 14,666,349 | 否 | 07a1d410 |
| annotations/test2017.json | 8,608,088 | 是，Root独立复核 | 18a3bf81 |
| annotations/train2017.json | 15,641,078 | 否，下载短响应后停止 | 18c7bae6 |
| annotations/train_test2017.json | 24,250,109 | 否 | 82b0e3d1 |

除TEST外，表内长度与CRC仅为完整目录声明，不能写为已验证成员内容。

## 实际TEST元数据

完整文件8,608,088B，SHA256为eb611754afc74c34222fa52cc04a500d1b84f956782fcbcc0e91d9766c4ac937，CRC32与目录18a3bf81一致。Root独立JSON检查得到：

- 1,962张图、5,922条实例标注、1类ship；1,961个图像ID被标注引用，1张没有标注关联。
- 图像ID、标注ID和文件名均无重复；标注图像/类别引用无孤项。JSON图像尺寸均为800×800；所有segmentation均为非空polygon列表。
- 1,962个图像文件名均在同包完整目录唯一匹配一个PNG条目。

上述是标注结构与目录关联检查，没有解码polygon、验证mask面积/拓扑、核图像内容、划分母景或执行跨集评价。没有训练集完整JSON，不能认证同包TRAIN/TEST的ID交集、并集或官方样本关系，也不能从目录总数相减冒充实读TRAIN数量。

## 与先前Git候选TRAIN分开记录

[此前固定作者Git提交候选JSON](HRSID_OFFICIAL_ANNOTATION_INTAKE_20261015_01.md)为11,592,317B、3,573图/9,870实例；同PNG包TRAIN目录声明15,641,078B，文件字节版本不同。尚未取得后者，不知道实例关系与差异成因。不得把Git候选TRAIN与本包TEST拼接后宣称同一官方版本完整划分；此前note的未知状态保留为历史范围，不用新TEST追认旧候选版本。

## 实际失败和停止边界

本轮HTTP请求计数到第11次即停止，累计计费响应7,361,454B，末轮上限12请求/9MiB、每请求至多30秒。第10次TRAIN请求只有空AssertionError，未保存独立长度/耗时，原因仍unknown。第11次返回206且Range/Content-Range相符，但实际1,375,769B少于预期1,454,719B，9.234秒；明确为响应长度不足，没有发第12次或复用partial作为TRAIN。

第11首跨度有意包含local header：固定30B+文件名26B+extra0；实际deflate起点5690230748，取回跨度5690230692–5691685410。离线核对没有起点变量误用；read1循环接受单次短读并继续，遇空返回后总长度门拒绝。这一说明只解释取数失败，不把它计入N20科学执行失败或模型失效。后续若继续取得TRAIN，必须另立有界获取任务、保存独立响应长度/耗时，并完整CRC/JSON验证，不默认扩容重试。

## 对跨数据集证据的意义和剩余门槛

本轮排除了“尚无官方TEST标注文件”和“文件名无法关联同包目录”两个具体障碍，但仍未具备跨数据集AP资格。仍需闭合同包TRAIN或固定配方评价所需完整版本、实际图像/真mask、原有TEST暴露与预训练史、协议、母景关系和许可覆盖。获取官方TEST不自动恢复盲测资格，也不证明zero-shot或场景独立。

当前N20不因此换数据、追加训练或提前选型。只发布本自写说明与官方入口，不上传完整标注、ZIP片段、第三方README/LICENSE、临时签名URL、原始图像或下载私有状态。许可覆盖尚未确认。
