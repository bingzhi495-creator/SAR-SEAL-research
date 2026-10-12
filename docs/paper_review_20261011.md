# 论文源码发布审阅（2026-10-11）

**结论：可原样保留为历史源码档案；三个入口均不能直接作为当前研究的提交稿。** 本次只读 `paper/`、`paper_tgrs/` 的 87 个 `.tex/.bib` 及 `paper/tgrs.tex` 引入的两个 `benchmark/*.tex`。数字、结果“已确认”等注释均为未重核草稿陈述；未访问 GT、权重、结果 JSON、科学 SPEC/日志，未联网、散列、编译或科学执行。

| 入口 | 实际稿件主线与发布问题 |
|---|---|
| `paper/seal.tex` | 摘要第32行及结论第384–392行写 oracle 诊断、卷积 mask refinement、MNE 统计先验及 HRSID 协议。不是当前 live F/e0 原图残差实验；“control proving the mechanism”（第380行）和“resolution-limited”（第386行）不能直接继承为当前机制证据。 |
| `paper/tgrs.tex` | 第6–25行明确标记 **STALE DRAFT / DO NOT SUBMIT AS-IS**，并指出正文 headline 未更新。第54、434行仍保留旧修正收益叙事；第416行把分类 Top-1/Top-5 差距并入“跨任务高IoU约束”，分类指标本身不测 IoU；第418行把特定50轮配方结果推广为 DETR 家族需要某类结构，范围过宽。 |
| `paper_tgrs/main.tex` | 第34行标题及 `sec_intro.tex:10`、`sec_disc.tex:33` 主张 frozen-model TTA/soup、mask-IoU fusion、BPR、re-ranking；与另两入口和当前 joint 主线均不同。`sec_exp.tex:38` 明述在 HRSID TEST 上选融合超参数，`:110/119` 明述 best-on-test 及 inshore 选点；这些设置不能表述为盲测或统一公平比较。 |

**需要重写的证据边界。** 当前 A/B 都有原生30损失及同权重的额外原图点BCE：A监督原生 L，B监督 L+u；故已控制额外监督，但 B 仍增加9441参数与 F/e0 信息，B>A 不独立证明 query 条件机制。论文应分开冻结读出、joint 训练、暴露 DEV、锁定官方 TEST、不同初始化/计算预算；阶段24后重建优化器追加48不是连续72轮。已知 posterior 预算、共享 posterior 几何/评分、conditional bias 不应改名为新原语。几何与评分交叉是有交互的条件诊断，不能把指标差值称为独立因果贡献；旧 oracle 的逐步差值也不自动证明普遍、顺序无关的机制分解。当前新实验结果尚不能由这些旧稿提供。

`paper_tgrs/sec_disc.tex:23/28` 的 preservation 必要性、幅度输入的“sensor ceiling”，以及 `paper/seal.tex:53`、`paper/tgrs.tex:75` 的重叠导致记忆结论，需要各自可核验的证据与适用条件；源码文字本身不证明这些断言。`paper_tgrs/sec_disc.tex:33` 的 epoch 比较也与 `sec_exp.tex:30/80` 的预算/初始化 caveat 不足以形成统一公平对照。

**依赖与版本。** 三入口的活动 `\input` 均有对应 `.tex`，静态 `\cite` 键在各自加载的 `.bib` 中均能解析，未发现加载集合内重复键；不等于文献真实、引用准确或能编译。`paper/tgrs.tex:414/426` 依赖目录外 `benchmark/tables.tex`、`benchmark/seg_tables.tex`，已纳入清单。图像/PDF、IEEEtran 和其他 TeX 包没有打开或验证，不在此次源码清单内。`_v27_backup/`、`revision_tgrs_20260905/`、`*_v*.tex`、park/draft 文件保留为明确的历史版本；`seal_refs_used.bib` 与 `seal_refs.bib` 文本相同，但名称不证明引用审计。作者/机构/日期、代码 URL 等占位仍需重写（`paper_tgrs/main.tex:37–38`、`sec_disc.tex:37`）。

**归档与敏感信息。** `paper_source_allowlist_20261011.txt` 只列89个可保留的源码相对路径，不选定 canonical submission、不认证结果、不自动授权发布其他文件。对这些源码检查私钥标记、常见凭证/令牌、带密码 URL、私网/回环地址，未检出候选；此为静态模式检查。科学载荷、图片/PDF、编译产物与环境文件排除在此次清单之外，未读取其内容。
