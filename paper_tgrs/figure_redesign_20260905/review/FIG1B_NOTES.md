# Fig. 1-B supplementary readout anatomy

交付：`fig1b_readout.tex`、`fig1b_readout.pdf`、`previews/fig1b_readout_150dpi.png`、`previews/fig1b_readout_gray_150dpi.png`、`assets_fig1b/`（7 张逐字节复制的真实资产及选取的原始 manifest）、`review/fig1b_metrics.json`。

建议作为补充图使用，避免占用主文篇幅。该图解释分割读出分支的数据依赖与实际张量形状，不声称证明上采样导致了全部边界误差，也不把展示资产误当成最终完整输出。

## 推荐英文图注

**Supplementary Fig. 1. Segmentation readout of the Mask2Former-R50 carrier.** The ResNet feature pyramid feeds a pixel decoder, which supplies multi-scale memory to the transformer decoder and a stride-four mask feature F. Learned queries are decoded and projected by the mask embedding MLP; the filled-dot node denotes the channel-wise inner product with F. The resulting 100 × 200 × 200 mask logits are bilinearly upsampled to the 800 × 800 chip grid. The SAR image, channel-norm feature visualizations and example query-logit plane are existing exports from one single-view forward pass of the souped carrier on HRSID image 1528. The low-resolution stack shows only query 36 as a representative front plane; the full-resolution stack is symbolic. Feature glyphs are not to geometric scale, and colors are independently normalized visualizations rather than comparable raw amplitudes. This diagram shows the segmentation readout only: the carrier was trained with an auxiliary mask-quality loss, whose inference score mixing is disabled in the final recipe. The auxiliary training branch, classification branch, iterative attention-mask updates and final instance selection are omitted for clarity.

## 修改目的

1. 原稿把 `pred_masks.png`（9 个经过显示阈值筛选的实例叠加图）画成 `100×800×800` 张量；本版换为完全符号化的输出张量，不使用这张资产。
2. 实际 pixel decoder 的 multi-scale memory 显式进入 transformer decoder；query output 通过 mask embedding MLP 后才与 mask feature 做通道内积。双路径在读出节点汇合，避免将查询矩阵当作空间特征图。
3. C2–C5 四个缩略图改为等显示尺寸，并逐行标真实通道与空间形状，牺牲按张量像素数缩放的装饰几何以保留小尺度特征可读性。这是对原规格的明确布局偏离；caption 明示图标不按空间尺寸等比例绘制。所有源栅格未做插值或美化。
4. 不使用“never updated”“unmodified”或“上采样证明导致 halo”；整图范围仅为已训练载体中的分割读出分支。

## 资产与来源

来源目录：`E:/SAR/seal/paper_tgrs/figs/assets/`。全部依据该目录现存 `manifest.json`，本轮没有执行模型、重新生成科学数据或检验 tensor raw 数组。

| 复制资产 | 用途 | manifest 中真实形状/展示处理 |
|---|---|---|
| chip.png | SAR 输入 | 800×800 展示图 |
| feat_c2_x4.png | C2 | 256×200×200；通道 L2 norm、分位裁剪，最近邻 ×4 |
| feat_c3_x8.png | C3 | 512×100×100；通道 L2 norm、分位裁剪，最近邻 ×8 |
| feat_c4_x16.png | C4 | 1024×50×50；通道 L2 norm、分位裁剪，最近邻 ×16 |
| feat_c5_x32.png | C5 | 2048×25×25；通道 L2 norm、分位裁剪，最近邻 ×32 |
| maskfeat_x2.png | mask feature F | 256×200×200；通道 L2 norm；最近邻 ×2 |
| mask_logits_q36_x4.png | 低分辨率 logits 张量的一个示例前平面 | 单个 q36 的 200×200 logit 平面；最近邻 ×4；不是全部 100 个平面 |

`queries_x3.png` 未用：它是 transformer 最后一层的查询特征，不是输入 learned query embedding；将它放到查询输入处会错误标注阶段。输入 queries 和输出 100×800×800 张量均使用符号图形。

代码定位：`carrier/seal_m2f_head.py:81` 的 decoder post-normalization、`:82` 的 mask embedding MLP、`:83` 的 `einsum("bqc,bchw->bqhw", ...)`；`:119` 的 mask feature 与 multi-scale memories，以及后续 transformer 输入路径。

## 实测验收

编译命令（新交付目录内）：`pdflatex -interaction=nonstopmode -halt-on-error fig1b_readout.tex`。

使用原仓库验收脚本，以 7 pt 字号、600 dpi 门槛检查该独立 PDF；同时运行 `_overlap.py`。通过 WIDTH/FONT/EMBED/RASTER/STROKE/OVERLAP 六项。

| 项目 | 实测 |
|---|---|
| PDF 画布 | 516.000×122.000 PDF pt（显式 bp 画布边界） |
| 宽高比 | 4.23:1 |
| 物理字号 | 恰好 7.00 / 8.00 pt |
| 文本词数 | 33（PyMuPDF 空白分词，含形状及运算点） |
| 精度/AP 数字 | 0 |
| 描边 | 0.50 / 0.80 pt，两档 |
| 字体 | 2 个嵌入 Type1 Times 系列字体，无 Type3 |
| 栅格 | 7 张，最低实际展示 DPI 约 1372 |
| 真重叠 | 0（文字/文字、文字/栅格、文字/路径检测均无命中） |
| 锁图标 | 0 |
| 预览 | 150 dpi RGB 与灰度各一份，已打开检查 |

注：独立 PDF 的训练身份与图注必须一起使用；不能移除图注中的辅助质量损失说明后重新称为 stock/unmodified 训练网络。此图不代替完整训练图，不包括后续 TTA、共识融合、BPR 或 re-ranker。
