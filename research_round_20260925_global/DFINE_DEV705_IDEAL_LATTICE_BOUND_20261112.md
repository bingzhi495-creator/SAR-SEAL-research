# D-FINE 细长小船的理想掩膜网格上界

FIT109 的同预算真值选点明显改善轮廓，而直接使用模型概率排序未超过原掩膜。随后在已使用的 D-FINE E24 DEV705 上，固定原检测框和 1,329 个小型细长船的原匹配，检验当前 160×160 掩膜网格能被严格证实限制了多少目标。未改写掩膜、重新训练、计算 AP 或访问官方 TEST。 [FIT109 选点](DFINE_E24_FIT109_PROBABILITY_RANK_20261112.md) · [冻结协议](code/DFINE_DEV705_IDEAL_LATTICE_BOUND_20261112/PROTOCOL.md)

设单位原生像元与源多边形的相交面积为 $q_i$，整目标面积为 $T$，预测为 $b_i$。连续 IoU 达到 $t$ 等价于 $\sum_i[(1+t)q_i-t]b_i\ge tT$。第一步在作者框内自由选像元。第二步利用理想半像素双线性插值在单一插值区间沿行、列的单调阈值模式，分别放松行与列的共享系数，再取两者较小的上界。上界低于阈值且超过数值容差时，可以证实理想表示无法达到该阈值。实际 CUDA FP16 后处理的有限精度另有条件。

| 连续阈值 | 原掩膜明确达标 | 框内自由栅格已排除 | 框内自由栅格可达的明确原掩膜失配 | 其中理想 160 网格进一步排除 | 来源代理组 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 885 | 17 | 424 | 2（0.47%） | 2/27 |
| 0.90 | 2 | 206 | 1,120 | 21（1.88%） | 13/27 |

0.90 的 21 个额外排除均未被理想 480 网格的放松上界排除。它们在更密网格上的实际可达性仍待验证。框内自由栅格的排除可同时涉及框范围和原生像元离散化。原掩膜均保留在作者框内，连续相交面积的两种复算最大差为 $2.84\times10^{-13}$，未见原掩膜超过理想 160 放松上界。后者检验的是实际 1,329 个掩膜。0.75 的实际汇总将 3 个容差内匹配计入 888 个原始非负裕量；上表依照容差规则列 885 个明确达标。0.90 没有原始裕量接近零的匹配，主结论一致。

0.90 的新增排除比例 1.88%，低于预设 10%；受影响来源代理组 13/27，也低于预设一半。本次不启动单纯扩大全图掩膜网格的训练。尚有 1,099 个框内自由栅格可达而当前未达标的匹配，理想 160 放松未能在这些样本上给出排除证书。放松可能较宽，无法将这批样本全部归为可由原网格实现。 [实测](results/DFINE_DEV705_IDEAL_LATTICE_ACTUAL_SAFE_20261112.json) · [读标注前绑定](results/DFINE_DEV705_IDEAL_LATTICE_PRELABEL_SAFE_20261112.json) · [评分源码](code/DFINE_DEV705_IDEAL_LATTICE_BOUND_20261112/score_grid.py)

通用掩膜表示工作已研究重建质量和细粒度点预测。这里测量的是固定检测框和连续 SAR 轮廓下可证实的网格限制。 [DCT-Mask](https://openaccess.thecvf.com/content/CVPR2021/html/Shen_DCT-Mask_Discrete_Cosine_Transform_Mask_Representation_for_Instance_Segmentation_CVPR_2021_paper.html) · [PointRend](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)
