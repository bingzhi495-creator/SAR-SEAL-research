# RF-DETR 热启动端点的全部原始查询覆盖实测

四个已验收端点在同机固定 DEV187 重新前向，共比较 74,800 个原始掩膜。用原图双线性插值及 logits>0 生成二值掩膜。66,495 个已归档保留掩膜逐像素相同。原进程退出 0，实际 72.28 秒，独立复核了原件、阈值计数、全部旧保留覆盖和分组总计。 [固定诊断协议](RFDETR_WARM_RAW100_COVERAGE_PROTOCOL_20261108.md)

| 端点 | IoU≥0.90，保留查询覆盖 | IoU≥0.90，全部 100 查询覆盖 | 相对 E24 原有覆盖丢失 | 丢失中原始查询仍达标 | 丢失中原始查询也未达标 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E24 | 100 | 101 | — | — | — |
| A 原生热启动 | 84 | 86 | 37 | 1 | 36 |
| B 均匀 BCE | 79 | 80 | 39 | 0 | 39 |
| D 校正分层 BCE | 77 | 79 | 42 | 1 | 41 |

对每个 GT 取掩膜最大 IoU，忽略候选分数。三臂在 0.50 和 0.75 处的原始覆盖与对应保留覆盖一致，0.90 的原有丢失中，原始查询仍达标仅 A 的 1/37、B 的 0/39、D 的 1/42。按预设规则，三个端点均优先研究严格几何回落及热启动过程。仅在当前固定模型上改保留规则，可补回的这类 E24 丢失很少。最高 IoU 允许同一候选覆盖多个 GT，不是 COCO AP 或一对一最优匹配。本次未改预测分数、更新参数或打开 TEST。证据范围为这一单种子、短程热启动和已用的内部开发集。 [全部计数](results/RFDETR_WARM_RAW100_COVERAGE_ACTUAL_SAFE_20261108.json)

## 下一项假设与直接先例

下一项先在 FIT 检验，固定 E24 教师和学生面对相同增强图像时，能否找到真值确认的高质量教师掩膜及其在学生端丢失的边界信息。跨模型按同一 GT 配对，训练前统计资格覆盖和重复增强稳定性。若可用，再比较无锚定、同位置同权重的额外硬 GT 和教师软掩膜三个条件。候选目标是保住原始查询的严格几何覆盖，同时在统一 AP 上超过原生强参考。这个新假设未执行，本轮 B/D 的停止结论保持。

固定教师软掩膜和同 GT 查询配对已有直接先例。CoMFormer 作者实现包含可选的旧掩膜 sigmoid 软 BCE。Li 等将教师和学生分别与 GT 匹配，再蒸馏同 GT 的实例输出。Noisy Boundaries 展示了教师边界噪声及粗细监督的对照。新工作需验证可观察的 SAR 几何回落、可靠信号筛选及超过额外硬 GT 对照的实际收益，方可讨论贡献。 [CoMFormer 原文](https://openaccess.thecvf.com/content/CVPR2023/papers/Cermelli_CoMFormer_Continual_Learning_in_Semantic_and_Panoptic_Segmentation_CVPR_2023_paper.pdf) · [固定源码](https://github.com/fcdl94/CoMFormer/blob/640496b58100fe2e3ab5bf7e9eb79818246bee44/continual/method_wrapper/set_criterion.py#L125-L129) · [同 GT 查询蒸馏原文](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Li_Knowledge_Distillation_for_Efficient_Instance_Semantic_Segmentation_with_Transformers_CVPRW_2024_paper.pdf) · [Noisy Boundaries 原文](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Noisy_Boundaries_Lemon_or_Lemonade_for_Semi-Supervised_Instance_Segmentation_CVPR_2022_paper.pdf) · [固定源码](https://github.com/zhenyuw16/noisyboundaries/blob/0c83823fe3634edb1274f4b20829822e49a25d65/mmdet/models/roi_heads/mask_heads/fcn_mask_head_ntm.py#L145-L197)
