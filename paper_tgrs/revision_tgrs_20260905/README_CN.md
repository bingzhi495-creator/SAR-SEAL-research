# TGRS 论文内部工作稿（2026-09-25 证据更新）

本目录为 2026-09-05 的独立修订副本。原始稿件、实验脚本、训练环境及预测文件未被覆盖。

## 先看这些文件

- `main.pdf`：IEEE 双栏工作正文，实际页数以重新编译的 PDF 为准。
- `supplement.pdf`：真实候选掩码、边界修正样例和载体读出结构，3 页。
- `figs/fig1_framework_cvpr.png`：CVPR 风格总架构图。
- `figs/figm_fusion_cvpr.png`：掩码关联与门控投票机制图。
- `figs/figm_bpr_cvpr.png`：边界补丁、UNet 与受限写入图。
- `REVISION_REPORT_CN.md`：本轮实际修改与剩余研究问题。
- `CONFIRMATORY_EXPERIMENTS_CN.md`：按优先级整理的确认性实验执行方案。
- `AUTHOR_INFORMATION.md`：目前作者顺序为 HOU BINGZHI、ZHAO PANZI，名单及单位尚未最终确定。

## 本版改变了什么

标题、摘要、引言、相关工作、方法、实验解释和讨论均已实质重写。2026-09-25 又补入原八视角候选池的完整强简单对照：1,962 图精确历史回放的每视角 top-1 官方 mask AP 为 56.4，门控为 53.8；近岸与远海也均是 top-1 更高。复用原 BPR 后，top-1/门控的官方 AP 为 58.8101/58.5646，近岸 44.2000/43.5540，远海 69.0596/69.1425；旧重排器与 OOF 重训对照尚未运行。稿件已停止把候选数门控写作必要新算法贡献，现定位为探索性评估。区分 HRSID 辅助质量监督与 PSeg-SSDD 标准 head，公开原有选择协议，不宣称架构不变、因果规律或整体 SOTA。

三张主方法图使用用户指定的 **内置 imagegen 生图技能**，并按后续要求改成具有特征层、掩码栈、局部放大、UNet 结构和重排示意的 CVPR 风格。生成的图形明确为概念示意；真实 SAR、掩码、PR 曲线及结果数值保留其科学来源。所有提示词、迭代与最终选择记录在 `review/IMAGEGEN_COMPLETE_RECORD.md`。

## 使用与编译

在安装了 IEEEtran、pdflatex 和 bibtex 的 Windows 环境运行 `./build.ps1`。该脚本仅编译论文和补充材料，不训练模型、不运行评估、不安装依赖。LaTeX 源码、所需 PDF/PNG 图件和参考文献均随包提供。历史单次迁移脚本不是编译入口。

## 验证范围

- 既有结果数值按历史记录保留；另加同池 top-1 诊断行和复用旧 BPR 的诊断段，其完整 AP 和哈希分别见 `research_round_20260920/manuscript/SAME_POOL_TOP1_FULL_DIAGNOSTIC_20260925.md` 与 `research_round_20260920/manuscript/TOP1_BPR_HISTORICAL_DIAGNOSTIC_20260925.md`。旧“433 个数值完全不变”校验不再覆盖新增行。
- LaTeX 编译通过，引用/标签完整，无 overfull，正文与补充 PDF 字体均嵌入。
- 新概念图使用原生像素，没有通过插值放大冒充分辨率；正文嵌入尺寸保证至少 300 DPI。
- 6 个成员 checkpoint 与 2 个 soup 的文件存在性、配置身份、种子及日志链已静态核查。没有加载其权重或重跑模型，不能把这些检查当成预测与权重的完整字节绑定。

这是可供作者审读的完整工作稿。独立确认实验、部分预测来源绑定、计算成本测量及作者信息仍需完成；当前版本不构成录用保证或投稿就绪声明。
