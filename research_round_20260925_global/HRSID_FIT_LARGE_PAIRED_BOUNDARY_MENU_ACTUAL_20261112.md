# HRSID 大船的等量边界补删上限

复用原 RF-DETR 的 28 张 FIT 图、50 个固定锚、四个文件名前缀代理组和每臂 100 个留出实例折。固定查询、框、类别分数、排序和两折留出区域。每个锚用同一臂原排序中已定义的第一个补充查询，在原掩膜两像素边界带内分别产生等量增加与删除两个候选。补充掩膜支持新增，原掩膜与补充掩膜的差异支持删去。对照匹配每个方向的像素数和深度分布。候选选择不用真值。 [预锁协议](code/HRSID_FIT_LARGE_PAIRED_BOUNDARY_MENU_20261112/PROTOCOL.md) · [原补充查询试验](HRSID_FIT_LARGE_COMPLEMENTARY_QUERY_RESULT_20261111.md)

| 组等权整掩膜 IoU（%） | 原生 A | 中心 B |
| --- | ---: | ---: |
| 原锚 | 89.510 | 88.185 |
| 补充查询候选，强制加 | 89.496 | 88.109 |
| 补充查询候选，强制删 | 89.354 | 88.190 |
| 补充查询候选，真值三选一 | 89.663 | 88.281 |
| 补充查询候选，加入安全约束的真值三选一 | 89.661 | 88.281 |
| 同深度轮廓候选，加入同安全约束的真值三选一 | 89.666 | 88.265 |

A／B 分别 61／72 折有成对编辑预算，每种方向合计 6,676／7,949 个候选像素。中心 B 的约束上限较原锚 **+0.09564** 点，较同深度轮廓上限 **+0.01543** 点。33 个目标在 3 个代理组获得正编辑，严格 IoU 0.90 达标保持 42 折。所选编辑无新增分裂或连通桥，最大单邻船新增侵入 0.652%。强制新增有一次连通桥，最大侵入 1.178%。 独立逐折复算确认 28 折按原锚计入，其中 6 折无补充查询、22 折有补充查询但任一方向无合格编辑。只看有编辑预算的折，三个有支持组的均值增量均小于 0.24 点。两方向共享 K 的要求限制了可编辑范围。实际受限上限低于预先锁定的 +0.50 和 +0.25 点，按门槛停止此候选规则的方向拟合。 [完整聚合与门槛](results/HRSID_FIT_LARGE_PAIRED_BOUNDARY_MENU_ACTUAL_SAFE_20261112.json)

以前的真值逐像素上限允许在更广的原区域自由选补删位置，B 组同口径曾有 +3.315 点。此处只有两个预测导出的整掩膜选项，显示这套候选的编辑位置和等量约束不足以利用原有几何余量。相似实例掩膜投票、边界小块细化和轮廓向内外移动均有先例。本实验只对当前 FIT 锚和固定候选菜单证伪，实际检测器在 FIT 图上训练过。 [先前真值编辑上限](HRSID_FIT_LARGE_FIXED_QUERY_AND_EDIT_ENVELOPE_RESULT_20261111.md) · [MNC 掩膜投票](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf) · [BPR](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html) · [SharpContour](https://xyzhang17.github.io/SharpContour/)

首次外层启动因选用的解释器缺少资源监控依赖退出，科学子进程未启动。换回原监控环境后，实际 CPU 科学进程退出 0，作业树清空，约 11.15 秒。原锚及逐折栅格计数复现，三项几何用例通过。没有模型前向、训练、DEV、TEST 或 AP。复算入口依赖已闭合的 P383 与原 FIT 候选档案，提交保留代码和聚合结果。 [实现](code/HRSID_FIT_LARGE_PAIRED_BOUNDARY_MENU_20261112/boundary_menu.py) · [几何用例](code/HRSID_FIT_LARGE_PAIRED_BOUNDARY_MENU_20261112/test_fixture.py)

