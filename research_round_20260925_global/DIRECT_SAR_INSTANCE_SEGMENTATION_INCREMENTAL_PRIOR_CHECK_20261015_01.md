# SAR船舶实例分割：有限增量先例核查

日期：2026-10-15。状态：普通文献核查，STOP EDIT；未复现论文或形成新科学结果。初查限三次定向检索，随后仅追加一次准确DOI补核；不是完整查新，也不改变N20已锁端点。

| 工作 | 证据等级 | 数据与实例掩膜指标 | 训练设置 |
|---|---|---|---|
| O²Former，2025，arXiv:2506.11913v1 | 已核v1论文全文；此处按预印本记录，不确认期刊版 | PSeg-SSDD训练/测试928/232；HRSID为3642/1962。表II/III的离岸/近岸mask AP分别为71.9/63.2、71.7/43.1，不能平均成全量AP | §IV-C：500轮、batch 8、AdamW；ImageNet预训练ResNet-50用于HRSID，再用HRSID权重微调SSDD |
| DACF-net，DOI 10.1080/17538947.2026.2696657 | 仅官方出版域搜索索引摘录；未成功直开全文或核表/PDF/代码 | 索引报告DACF-Net‡在HRSID/PSeg-SSDD的segmentation AP为63.3/65.7；划分比例分别13:7、8:2 | 索引§4.2报告端到端训练12轮、AdamW，batch分别2/4；均未独立确认 |

O²Former原文§IV-A称沿用官方划分，§IV-C又写随机抽样；具体文件ID、种子及该措辞如何对应尚未闭合。其Mask2Former扩展同时修改查询生成和方向特征，属于直接实例掩膜任务，不能当作只做bbox的工作。全文来源：[O2Former: Direction-Aware and Multi-Scale Query Enhancement for SAR Ship Instance Segmentation](https://arxiv.org/html/2506.11913v1)。

DACF-net索引页首标注IJDE及在线日期2026-09-08；摘要和性能比较段报告上述AP，§4.2摘录报告12轮。这里的日期、数值和预算都只能记作“官方域索引报告”，不是Root复核或复现结论。原始出版入口：[DACF-net: a dynamic attention and context fusion network with geometry-aware multi-scale refinement for sar ship instance segmentation](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2696657)。

N20检验冻结教师及固定PR条件下，最终渲染像素F的3×3邻域与中心复制对照的小读出；这与上述网络级特征/查询联合训练存在区别，但区别本身不等于创新。方向、多尺度与上下文增强已有直接先例，不能宣称首次。当前仍缺可比预算、初始化重复和跨集证据；论文TEST或分场景AP也不能与本项目当前DEV直接排名。上述文献只补充最近先例与未来公平比较范围，不替代N20固定主端点、停止规则或实际资格。
