# 官方HRSID PNG：两个编码文件的有界获取试点

2026-10-15。已实际取得同一官方PNG归档中的两个完整PNG编码文件，并由Root独立核文件字节和容器完整性。未解码IDAT/像素、未读取GT掩膜栅格、未执行模型或AP；不是完整数据集取得或跨集资格。实际原图不随Git发布。

## 来源与事前选取

使用[同包目录与TRAIN/TEST实核](HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)已核的[作者PNG归档入口](https://drive.google.com/file/d/1xgXi8KC3MDWuu7Yp4n2J-LOPYLRszHAc/view)。从固定TRAIN和TEST各取file_name字典序第一项，只依赖标注文件名，不依赖模型输出、错误或AP。归档总长5698302315B、Last-Modified为Wed, 01 Jul 2020 15:10:52 GMT；整包SHA仍未知。

| 划分 | 归档成员 | 完整PNG字节 | ZIP CRC32 | PNG SHA256 |
|---|---|---:|---|---|
| TRAIN | images/P0001_0_800_7200_8000.png | 1769250 | 9d760dda | 83a09088ce3c695fa4e7d40f9573a4b292c81be2423d9dbb0584fbb161c70010 |
| TEST | images/P0001_0_800_10190_10990.png | 1407681 | 2e129a7f | 1e3f032f595494587c9e37223ef83c3eceb1d071342fb38240636ff34445c2d3 |

## 实际执行与Root核验

新试点02最多16请求/4MiB应用body、每段至多256KiB；原生curl HTTP1.1直连，curl45秒和持有进程55秒取消，无代理、账号凭据、redirect、retry或刷新URL。实际14请求、2745196B应用body（含两个512B local-header请求），curl计数同值；14条curl均实际退出0并回收，206/HTTP1.1/Content-Range/Content-Length/Last-Modified一致。请求耗时相加68.199秒，含小分块连接开销；不是模型推理速度，也不足以推断整包下载速度。

Root独立检查各响应实际文件长/SHA与连续范围，解析ZIP local header的filename/method/flags/尺寸/CRC，核其与既有完整central directory一致。TRAIN header62B、data_offset1159278、压缩1585057B；TEST header64B、data_offset101、压缩1159115B。完整raw-deflate流独立解压为PNG编码字节，消耗全部压缩字节，与保存PNG逐byte一致，最终文件SHA及ZIP CRC符合上表。

Root独立遍历PNG容器，签名、chunk framing/CRC、首个唯一IHDR和末尾IEND通过；两文件IHDR均800×800、bit depth8、color type2（RGB）、compression/filter/interlace均0。TRAIN有216个IDAT块，TEST172个。此处没有解压IDAT或验证像素/标注的空间对齐，RGB头也不能证明SAR像素内容质量。

## 原失败和具体修复

试点01只完成一次512B请求，curl0/206；通用header解析器错误拒绝重复Content-Security-Policy头后停止，没有取得完整PNG。它不是网络或文件损坏的证据。旧脚本、日志和512B保留，不在试点02复用。

新增回归真实RED（aa1e8e）复现同一重复CSP异常，再GREEN（46c10a）：非关键头保留多值，关键Range/Length/mtime/ETag/encoding冲突拒绝，相同关键值重复显式归一。一次AST/手工差异检查（86ccb9）限定为header多值解析和fresh输出路径；原选图、网络预算、ZIP/PNG校验函数不变。修复后只执行这个新试点并停止。操作脚本/临时URL配置和原文件均私有，未加入N20科学Sources；该失败不计入科学failed26。

## 支持与不支持的结论

本次支持两个明确源成员的编码文件完整获取、目录身份及PNG头/容器检查，说明这条官方影像字节获取路径实际可行。其余5602个成员未由本试点取得；影像像素、GT mask解码和对齐、完整版本/历史暴露/母景协议、正式跨集训练或评价仍未完成。没有利用这两个文件调整本轮N20方法、阈值或模型。完整研究目标仍未达成。
