# SAR 船舶实例分割论文收尾

**交付截止为北京时间 2026 年 11 月 24 日 22:00。** 交付英文审阅稿、LaTeX、可编辑图源、核验过的主结果和复现入口。作者元数据保留占位。

| 截止（北京时间） | 分工 | 验收 |
| --- | --- | --- |
| 11 月 22 日 22:00 | 实验组与独立复核组 | 核验 HRSID 官方划分的冻结精修结果；跟进第二个 RF-DETR 检测器种子的训练、固定输出与同量对照；完成经典空间吸引力对照或记录实际状态。 |
| 11 月 23 日 18:00 | 主控与图件组 | 冻结正文证据，填写主结果及失败范围，核对来源与有效样本数。以实测生成结果和案例图，保留可编辑矢量源。 |
| 11 月 24 日 22:00 | 主控统稿与独立评审 | 核对主张、数字、引用、来源和复现入口，编译检查 IEEE 双栏 PDF，验证 GitHub 交付和 TGRS 投稿判断。 |

论文主线是冻结实例候选后，学习原生像元的分数占据率，并在相同端点和实际编辑量下检验固定配对的影响。查询条件与保留角度的邻域顺序均已完成原有对照，按预锁判据结束。传统等面积像元交换先例已补入正文。正式复验保留原始负例并注明官方划分的历史接触。

GPT‑6.2 Astra 最高推理档已独立核对同量主结论、跨库估计与直接先例。最终采用通过核验的实验形成投稿建议。 [论文主张核对](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/paper-claim-audit/SKILL.md) · [证据约束学术写作](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-writing/SKILL.md) · [TGRS 作者说明](https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/)

11 月 22 日的独立审阅以发布结果复算主效应，并逐条追溯适用对象、比较组和计量单位。正文已写明 PSeg-SSDD 迁移使用冻结 D-FINE、历史测试文件来自固定镜像，以及 0.90 阈值下预测策略与真值上界各自的结果。相关工作补入动态关键点船舶实例分割的原始研究。 [数字核验](https://github.com/OIerYangJZ/Science-Skill/tree/main/skills/number-audit) · [同行评议](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/peer-review/SKILL.md) · [Gao 等，JSTARS](https://doi.org/10.1109/JSTARS.2024.3383779)
