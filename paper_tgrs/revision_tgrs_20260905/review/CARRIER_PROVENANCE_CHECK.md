# 载体来源最小 P0 预检（第一步）

生成时间 UTC：2026-09-05T05:39:54.728130+00:00

六个指定最终迭代成员及两个 soup 文件均存在，配置种子与成员声明一致。HRSID 三成员使用 MSIoUM2FHead 和辅助权重 0.5，三份对应训练日志均记录 loss_maskiou；PSeg-SSDD 三成员使用标准 Mask2FormerHead，辅助权重字段不存在，三份指定日志均未检出 loss_maskiou。

本步骤只读配置、历史日志和文件元数据，不读取 checkpoint 或缓存内容，不证明全部预测字节来源，不认证论文结果。逐字段位置、完整配置 SHA-256、日志证据原行及片段哈希均见同名 JSON。

## 成员与文件元数据

| 数据集 / seed | head | 辅助权重 | 最终文件 | 大小（bytes） | mtime UTC | 首条辅助损失 |
|---|---|---|---|---:|---|---:|
| HRSID / 0 | MSIoUM2FHead | 0.5 | iter_100872.pth | 583462603 | 2026-07-06T14:34:32.249877+00:00 | 0.3336 |
| HRSID / 2 | MSIoUM2FHead | 0.5 | iter_100872.pth | 583547659 | 2026-07-11T22:11:28.702137+00:00 | 0.3204 |
| HRSID / 3 | MSIoUM2FHead | 0.5 | iter_100872.pth | 583520651 | 2026-07-12T13:28:11.180992+00:00 | 0.3271 |
| PSeg-SSDD / 0 | Mask2FormerHead | 字段不存在 | iter_8352.pth | 532852651 | 2026-07-28T16:49:54.610768+00:00 | 未检出 |
| PSeg-SSDD / 2 | Mask2FormerHead | 字段不存在 | iter_8352.pth | 532863659 | 2026-07-28T18:12:20.780476+00:00 | 未检出 |
| PSeg-SSDD / 3 | Mask2FormerHead | 字段不存在 | iter_8352.pth | 532858731 | 2026-07-28T19:34:31.027039+00:00 | 未检出 |

全部 load_from 均为 pretrained/m2f_r50_coco.pth。日志同时记录 torchvision://resnet50 的 backbone 初始化，以及随后载入该 COCO checkpoint；两条信息应一起理解。PSeg 辅助权重字段不存在，不等于保存配置显式设为 0。

## 各成员文本证据

### HRSID / s0

- 配置：[c0_maskiou_v2_m2f_hrsid.py](E:/SAR/seal/runs/c0_maskiou_v2/s0/c0_maskiou_v2_m2f_hrsid.py)。SHA-256：8a5d8913136106557f9a9e930b0a16665e06b9a1115e7858bb8bb710874abf00。
- load_from 行 90；head 行 239；seed 行 296；work_dir 行 511。
- 对应日志：[20260706_104731.log](E:/SAR/seal/runs/c0_maskiou_v2/s0/20260706_104731/20260706_104731.log)，共 3631 行；训练记录 1008 行，含 loss_maskiou 的训练记录 1008 行，全文件该词出现 1008 次。
- 首条训练记录行 2345；最终迭代保存消息行 3631。
- last_checkpoint 小文本声明：E:\SAR\seal\runs\c0_maskiou_v2\s0\iter_100872.pth。
- 当前成员文件：[iter_100872.pth](E:/SAR/seal/runs/c0_maskiou_v2/s0/iter_100872.pth)。

### HRSID / s2

- 配置：[c0_maskiou_v2_m2f_hrsid.py](E:/SAR/seal/runs/c0_maskiou_v2/s2/c0_maskiou_v2_m2f_hrsid.py)。SHA-256：cbdeb4747310531511ae67043846f35b419a571395eb4d97cf8c39a783b862eb。
- load_from 行 90；head 行 239；seed 行 296；work_dir 行 511。
- 对应日志：[20260711_153205.log](E:/SAR/seal/runs/c0_maskiou_v2/s2/20260711_153205/20260711_153205.log)，共 3633 行；训练记录 1008 行，含 loss_maskiou 的训练记录 1008 行，全文件该词出现 1008 次。
- 首条训练记录行 2345；最终迭代保存消息行 3633。
- last_checkpoint 小文本声明：E:\SAR\seal\runs\c0_maskiou_v2\s2\iter_100872.pth。
- 当前成员文件：[iter_100872.pth](E:/SAR/seal/runs/c0_maskiou_v2/s2/iter_100872.pth)。

### HRSID / s3

- 配置：[c0_maskiou_v2_m2f_hrsid.py](E:/SAR/seal/runs/c0_maskiou_v2/s3/c0_maskiou_v2_m2f_hrsid.py)。SHA-256：1655c51cbfeed222343f80f6145db18eb967eb741bb14479c566c74a9bc60e33。
- load_from 行 90；head 行 239；seed 行 296；work_dir 行 511。
- 对应日志：[20260712_065351.log](E:/SAR/seal/runs/c0_maskiou_v2/s3/20260712_065351/20260712_065351.log)，共 3631 行；训练记录 1008 行，含 loss_maskiou 的训练记录 1008 行，全文件该词出现 1008 次。
- 首条训练记录行 2345；最终迭代保存消息行 3631。
- last_checkpoint 小文本声明：E:\SAR\seal\runs\c0_maskiou_v2\s3\iter_100872.pth。
- 当前成员文件：[iter_100872.pth](E:/SAR/seal/runs/c0_maskiou_v2/s3/iter_100872.pth)。

### PSeg-SSDD / s0

- 配置：[cp_m2f_cocoinit_pseg.py](E:/SAR/seal/runs/cp_pseg/s0/cp_m2f_cocoinit_pseg.py)。SHA-256：5c1be27b6ae0bcbe4ce6649dc215b589196e3673351b03931f9f57dbcc41d3c3。
- load_from 行 84；head 行 226；seed 行 282；work_dir 行 497。
- 对应日志：[20260728_232720.log](E:/SAR/seal/runs/cp_pseg/s0/20260728_232720/20260728_232720.log)，共 2571 行；训练记录 167 行，含 loss_maskiou 的训练记录 0 行，全文件该词出现 0 次。
- 首条训练记录行 2306；最终迭代保存消息行 2562。
- last_checkpoint 小文本声明：E:\SAR\seal\runs\cp_pseg\s0\iter_8352.pth。
- 当前成员文件：[iter_8352.pth](E:/SAR/seal/runs/cp_pseg/s0/iter_8352.pth)。

### PSeg-SSDD / s2

- 配置：[cp_m2f_cocoinit_pseg.py](E:/SAR/seal/runs/cp_pseg/s2/cp_m2f_cocoinit_pseg.py)。SHA-256：a0b98e0cbf8242e7634ed75557c60f5689c4415047ca374fa8ce59daaafc72f6。
- load_from 行 84；head 行 226；seed 行 282；work_dir 行 497。
- 对应日志：[20260729_005053.log](E:/SAR/seal/runs/cp_pseg/s2/20260729_005053/20260729_005053.log)，共 2575 行；训练记录 167 行，含 loss_maskiou 的训练记录 0 行，全文件该词出现 0 次。
- 首条训练记录行 2306；最终迭代保存消息行 2564。
- last_checkpoint 小文本声明：E:\SAR\seal\runs\cp_pseg\s2\iter_8352.pth。
- 当前成员文件：[iter_8352.pth](E:/SAR/seal/runs/cp_pseg/s2/iter_8352.pth)。

### PSeg-SSDD / s3

- 配置：[cp_m2f_cocoinit_pseg.py](E:/SAR/seal/runs/cp_pseg/s3/cp_m2f_cocoinit_pseg.py)。SHA-256：a6c2c54ddb493d3061b191838a8764d0f469c8ad469eff3e4095ceee58cc323e。
- load_from 行 84；head 行 226；seed 行 282；work_dir 行 497。
- 对应日志：[20260729_021239.log](E:/SAR/seal/runs/cp_pseg/s3/20260729_021239/20260729_021239.log)，共 2571 行；训练记录 167 行，含 loss_maskiou 的训练记录 0 行，全文件该词出现 0 次。
- 首条训练记录行 2306；最终迭代保存消息行 2560。
- last_checkpoint 小文本声明：E:\SAR\seal\runs\cp_pseg\s3\iter_8352.pth。
- 当前成员文件：[iter_8352.pth](E:/SAR/seal/runs/cp_pseg/s3/iter_8352.pth)。

## Soup 和缓存声明

- HRSID soup：[E:/SAR/seal/runs/c0_maskiou_v2/soup_s023/iter_100872.pth](E:/SAR/seal/runs/c0_maskiou_v2/soup_s023/iter_100872.pth)；177111581 bytes；mtime UTC 2026-07-16T14:44:49.180142+00:00。
- PSeg-SSDD soup：[E:/SAR/seal/runs/cp_pseg/soup_s023/iter_8352.pth](E:/SAR/seal/runs/cp_pseg/soup_s023/iter_8352.pth)；176437637 bytes；mtime UTC 2026-07-28T19:34:44.789869+00:00。

HRSID 成员/输出声明见 carrier/soup.py:39–45；PSeg 成员/平均调用/输出声明见 run_ssdd_queue.sh:53–69。只读代码，没有执行其中的平均或检查。

### HRSID：BPR fitting single-view dump

- 源码定位：[run_bpr_queue.sh:23](E:/SAR/seal/run_bpr_queue.sh:23)，片段至行 25。
- 源码定位：[run_bpr_queue.sh:37](E:/SAR/seal/run_bpr_queue.sh:37)，片段至行 43。
- 声明输出：[train_preds.pkl](E:/SAR/seal/runs/bpr/train_preds.pkl)；存在=True；大小=62425072 bytes。
- 相关历史载入/写出路径消息共 2 条，逐条保存在 JSON。消息仅证明日志记录了这些路径，不证明当前文件字节未被替换。

### HRSID：BPR input TTA dumps

- 源码定位：[run_bpr_queue.sh:23](E:/SAR/seal/run_bpr_queue.sh:23)，片段至行 25。
- 源码定位：[run_bpr_queue.sh:73](E:/SAR/seal/run_bpr_queue.sh:73)，片段至行 88。
- 声明输出：[test_inshore.pkl](E:/SAR/seal/runs/bpr/test_inshore.pkl)；存在=True；大小=6495965 bytes。
- 声明输出：[test_offshore.pkl](E:/SAR/seal/runs/bpr/test_offshore.pkl)；存在=True；大小=13422935 bytes。
- 声明输出：[test_official.pkl](E:/SAR/seal/runs/bpr/test_official.pkl)；存在=True；大小=19919228 bytes。
- 相关历史载入/写出路径消息共 6 条，逐条保存在 JSON。消息仅证明日志记录了这些路径，不证明当前文件字节未被替换。

### HRSID：rerank raw TTA dumps

- 源码定位：[run_scic_queue.sh:40](E:/SAR/seal/run_scic_queue.sh:40)，片段至行 41。
- 源码定位：[run_scic_queue.sh:54](E:/SAR/seal/run_scic_queue.sh:54)，片段至行 68。
- 源码定位：[c0_v2_scic_stack_hrsid.py:15](E:/SAR/seal/carrier/configs/c0_v2_scic_stack_hrsid.py:15)，片段至行 23。
- 声明输出：[train_stack.pkl](E:/SAR/seal/runs/scic/train_stack.pkl)；存在=True；大小=58578923 bytes。
- 声明输出：[test_official_stack.pkl](E:/SAR/seal/runs/scic/test_official_stack.pkl)；存在=True；大小=32343119 bytes。
- 相关历史载入/写出路径消息共 4 条，逐条保存在 JSON。消息仅证明日志记录了这些路径，不证明当前文件字节未被替换。

### PSeg-SSDD：BPR input TTA dumps

- 源码定位：[run_ssdd_queue.sh:73](E:/SAR/seal/run_ssdd_queue.sh:73)，片段至行 90。
- 源码定位：[run_ssdd2_queue.sh:63](E:/SAR/seal/run_ssdd2_queue.sh:63)，片段至行 70。
- 声明输出：[stack_test.pkl](E:/SAR/seal/runs/ssdd/stack_test.pkl)；存在=True；大小=2385612 bytes。
- 声明输出：[stack_ins.pkl](E:/SAR/seal/runs/ssdd/stack_ins.pkl)；存在=True；大小=721584 bytes。
- 声明输出：[stack_ofs.pkl](E:/SAR/seal/runs/ssdd/stack_ofs.pkl)；存在=True；大小=1664371 bytes。
- 相关历史载入/写出路径消息共 6 条，逐条保存在 JSON。消息仅证明日志记录了这些路径，不证明当前文件字节未被替换。

### PSeg-SSDD：BPR fitting single-view dump

- 源码定位：[run_ssdd2_queue.sh:16](E:/SAR/seal/run_ssdd2_queue.sh:16)，片段至行 21。
- 源码定位：[run_ssdd2_queue.sh:45](E:/SAR/seal/run_ssdd2_queue.sh:45)，片段至行 51。
- 声明输出：[train_preds.pkl](E:/SAR/seal/runs/ssdd/train_preds.pkl)；存在=True；大小=14190209 bytes。
- 相关历史载入/写出路径消息共 4 条，逐条保存在 JSON。消息仅证明日志记录了这些路径，不证明当前文件字节未被替换。

### PSeg-SSDD：rerank raw TTA dumps

- 源码定位：[run_ssddr11_queue.sh:13](E:/SAR/seal/run_ssddr11_queue.sh:13)，片段至行 25。
- 源码定位：[run_ssddr11_fix.sh:9](E:/SAR/seal/run_ssddr11_fix.sh:9)，片段至行 21。
- 声明输出：[ssdd_train_stack.pkl](E:/SAR/seal/runs/scic/ssdd_train_stack.pkl)；存在=True；大小=17691065 bytes。
- 声明输出：[ssdd_test_stack.pkl](E:/SAR/seal/runs/scic/ssdd_test_stack.pkl)；存在=True；大小=4404911 bytes。
- 声明输出：[ssdd_inshore_stack.pkl](E:/SAR/seal/runs/scic/ssdd_inshore_stack.pkl)；存在=True；大小=1002904 bytes。
- 声明输出：[ssdd_offshore_stack.pkl](E:/SAR/seal/runs/scic/ssdd_offshore_stack.pkl)；存在=True；大小=3402344 bytes。
- 相关历史载入/写出路径消息共 8 条，逐条保存在 JSON。消息仅证明日志记录了这些路径，不证明当前文件字节未被替换。

## 剩余边界

PSeg 最终 inshore/offshore R11 输出声明为 ssdd_r11fix_inshore/offshore.json，原始匹配缓存也改为子集专用文件；full 仍是 ssdd_r11_full.json。不要把修正前的 ssdd_r11_inshore/offshore 误绑为最终输出。源码定位见 JSON final_output_declarations。

本步骤把保存配置、对应训练日志、指定成员文件和生成脚本路径整理成可复查记录。尚未完成权重内部验证、soup 数值核验、历史源码固定、全部预测缓存与论文表格的不可变绑定。各队列存在“已有输出就跳过”的复用守卫，因此路径与文件存在不能封闭来源链。

只对小配置与 last_checkpoint 文本计算整文件 SHA-256；日志仅对选中原行片段计算 SHA-256。没有哈希大检查点，没有反序列化 pth/pkl，没有导入 torch/mmengine/mmdet，没有训练、推理、评估或安装。原项目文件未修改；本次仅写这两个审查文件。
