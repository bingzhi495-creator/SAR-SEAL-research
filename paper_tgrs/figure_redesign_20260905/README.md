# TGRS 论文审读与图件重绘

交付日期：2026-09-05。全文审读对象是下载目录的 `main_submit.pdf`（15 页）；绘图需求来自 `E:/SAR/seal/docs/PROMPT_FOR_FIGURE_AI.md`。原论文、原图、原实验记录均未修改。

先看 `TGRS_REVIEW_AND_FIGURES.pdf`：前部是完整中文审读报告，后部是五张原尺寸矢量图及中文说明。仅看图可用 `TGRS_FIGURE_PROOFS.pdf`。`review/PAPER_AUDIT_CN.md` 是可编辑的报告原文。

## 文件与用途

| 图 | 可编辑源文件 | 投稿图件 | 说明 |
|---|---|---|---|
| Fig. 1 总览 | `fig1_overview.tex` | `fig1_overview.pdf` | 离线权重准备与图像推理分开 |
| Fig. 2 融合 | `fig2_consensus.tex`、`derive_fig2_assets.py` | `fig2_consensus.pdf` | 同一真实 top-16 子集的矩阵与分支 |
| Fig. 3 BPR | `fig3_bpr.tex` | `fig3_bpr.pdf` | 真实补丁、准确定位与可写环带 |
| Fig. 4 PR | `make_pr.py` | `fig4_pr.pdf`、`fig4_pr.svg` | 保留已有 16×101 个采样点 |
| 补充 Fig. 1-B | `fig1b_readout.tex` | `fig1b_readout.pdf` | 实际读出依赖与张量形状 |

`previews/` 含每图 150 dpi 彩色与灰度预览。`assets_fig*/` 含全部用到的真实素材；`sources/` 含 PR 原始图及原绘图脚本。`latex_integration.tex` 提供四个图环境和英文图注，路径以 `paper_tgrs/` 为基准，请分别放回正文相关位置。

## 图件验收

以 `review/FINAL_FIGURE_MEASUREMENTS.json`、各 `*_checks.json`/`*_verification.json` 和 `ACCEPTANCE.md` 为准。五张独立图均有两档物理字号 7/8 pt，字体全嵌入，无 Type 3，机械文字重叠检测为零；方法图有两档线宽，PR 图有三档。宽度按双栏 516 pt / 单栏 252 pt 设计。

用户说明中的 7/8 pt 是本次制作规则；IEEE 当前图形页建议约 9–10 pt，不能将本次字号写成 IEEE 统一的最低标准。通过图件验收也不等于实验复现、统计显著性或投稿就绪认证。真人一秒阅读测试未实施。

## 必须随图保留的科学限定

1. **基座身份**：保存的训练配置与日志有非零辅助 `loss_maskiou`。关闭推理混合系数不等于训练时未改动标准 Mask2Former。冻结指训练完成后的推理状态。
2. **Fig. 2 是 top-16 算子重放**：真实完整 800 候选池下，图中两种子对应 10/9 成员并均保留种子。图注中不能删掉这一区别。
3. **Fig. 3 的素材是单视图 coarse masks**：不是完整 TTA 融合输出。该窗口由原脚本按改动像素最多选出，不能称随机典型案例。
4. **PR 图只做显示重绘**：本轮没有重新评估预测 dump。独立坐标往返检查验证了显示一致性，详见 `review/QA_PR_ROUNDTRIP.json`。
5. **训练与投稿协议仍需修正**：HRSID 的融合设置在测试集上选择；BPR 的验证划分按 patch；若干公平性、新颖性和理论表述超出当前证据，完整分析见报告。

因此，本包是图件与投稿前审读成果，不是可直接提交的论文版本。

## 重建

需要现有 LaTeX/TikZ、Python、numpy、Pillow、OpenCV（仅 Fig. 2 派生）、PyMuPDF、matplotlib；报告额外需要 reportlab 和中文字体。所有数据输入都在包内，不需要模型、不需要 GPU，也不会启动训练或评估。

在本目录执行：

```powershell
python derive_fig2_assets.py
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape fig1_overview.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape fig2_consensus.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape fig3_bpr.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape fig1b_readout.tex
python make_pr.py
python build_review.py
```

本机已使用现有 `C:/Users/houbingzhi/anaconda3/python.exe` 和 `E:/miktex/miktex/bin/x64/pdflatex.exe`。`build_review.py` 在需要时从 Codex 的已安装依赖目录加载 reportlab；中文排版使用 Windows 宋体/黑体，换机器时可修改字体路径。

`tools/verify_fig.py` 与 `tools/_overlap.py` 是原仓库验收脚本的副本。示例：

```powershell
python tools/verify_fig.py --min-pt 7 --min-dpi 600 fig1_overview.pdf fig2_consensus.pdf fig3_bpr.pdf fig1b_readout.pdf
python tools/verify_fig.py --target 252 --min-pt 7 fig4_pr.pdf
```

详细资产来源、每张图的英文图注、固定文案的必要修正、原生采样与嵌入 dpi 的区别，见 `review/FIG*_NOTES.md` 和 `review/FIG1_AND_PR_NOTES.md`。本包未使用图像生成模型，也没有虚构 SAR、掩码、热图、实验曲线或性能数字。
