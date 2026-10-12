# O2Former 版本、作者与比较数值核查

核查日期：2026-09-05。范围仅限公开文献与当前 LaTeX 文本；未编辑正文或 `refs.bib`，未运行任何实验。

## 可执行结论

1. 应纠正旧条目中的 **Yang Li → Yunrui Li、Xin He → Xiaoyu He**。这不只是依据后续期刊论文推断：arXiv v1 的正文首页已经列出这两位全名。[arXiv v1 首页](https://arxiv.org/html/2506.11913v1)
2. O2Former 已有 **2026 年 JSTARS 正式出版记录**，期刊版为六位作者，末位新增 Amir Hussain。作者顺序得到 Crossref 出版社登记信息与作者所属机构记录共同支持。[Crossref DOI 登记](https://api.crossref.org/works/10.1109/JSTARS.2026.3677088)、[机构出版记录](https://pure.kfupm.edu.sa/en/publications/o2-former-direction-aware-and-multiscale-query-enhancement-for-sa/)
3. 当前比较表的数值能够明确追溯到 **arXiv:2506.11913v1，2025-06-13**。不要只替换参考文献年份而保留未注明版本的“Table 3”：期刊作者接受稿已将 PSeg-SSDD/HRSID 比较表重新编号为 Table III/IV。[v1 版本记录](https://arxiv.org/abs/2506.11913)、[带正式 DOI 的作者接受稿](https://www.researchgate.net/publication/403111230_O2Former_Direction-Aware_and_Multi-Scale_Query_Enhancement_for_SAR_Ship_Instance_Segmentation)
4. 推荐保留 `o2former2025` 作为数值与旧实现描述的明确来源，新增 `o2former2026` 用于引言/相关工作中的正式出版状态；无须修改任何实验数值。

## 正式书目信息及证据边界

| 字段 | 核实结果 | 来源与限制 |
|---|---|---|
| 作者及顺序 | Fei Gao; Yunrui Li; Xiaoyu He; Jinping Sun; Jun Wang; Amir Hussain | Crossref、机构记录、接受稿首页一致 |
| 期刊 | IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing | 三方一致 |
| 年份 | 2026 | 三方一致；作者北航主页另列 2026-04-01 |
| DOI | 10.1109/JSTARS.2026.3677088 | 出版社登记及作者主页一致 |
| 卷/最终页码 | 19 / 11633–11645 | KFUPM 机构记录给出，共 13 页；尚未直接读取 IEEE 最终排版页眉 |
| 题名拼写 | O²Former: Direction-Aware and Multiscale Query Enhancement for SAR Ship Instance Segmentation | 机构终卷页记录采用 Multiscale；Crossref 与作者接受稿采用 Multi-Scale，属于版本/元数据拼写差异 |
| IEEE 文献号 | 11455159 | Crossref 所给 IEEE PDF 链接包含该编号 |

来源：[机构记录及导出 BibTeX](https://pure.kfupm.edu.sa/en/publications/o2-former-direction-aware-and-multiscale-query-enhancement-for-sa/)、[作者北航主页](https://shi.buaa.edu.cn/hexiaoyu/zh_CN/lwcg/173753/content/33701.htm)、[Crossref 原始记录](https://api.crossref.org/works/10.1109/JSTARS.2026.3677088)。

**元数据并未完全同步。** 本次直接请求 Crossref API 成功，其 `author` 已有六人，但 `volume` 尚空、`page` 仍为 `1-14`。不应把早期页码 `1–14` 与机构给出的卷 19 拼接成一个混合条目。下方正式 BibTeX 使用机构终卷页信息，并在此保留来源差异。

**最终全文限制。** [IEEE 文章页](https://ieeexplore.ieee.org/document/11455159)本次返回 JavaScript/机器人验证页面；PDF 端点 HEAD 返回 PDF 类型，但实际 GET 返回 HTTP 418，未读取 PDF 内容，因此不能称 IEEE 最终出版稿全文已核验。能够读取的是带同一 DOI、六位作者、明确声明为未最终编辑作者版本的接受稿公开全文。该稿的模板页眉仍有旧 LaTeX 示例信息，不能把示例 `VOL. 18, NO. 9, SEPTEMBER 2020` 当出版卷期。[作者接受稿及版本声明](https://www.researchgate.net/publication/403111230_O2Former_Direction-Aware_and_Multi-Scale_Query_Enhancement_for_SAR_Ship_Instance_Segmentation)

## 原比较数值来自哪个版本

以下为论文已报告数值的文献核对，不代表本地复现或同协议比较。AP 单位为百分数。

| 模型与场景 | AP | AP50 | AP75 | arXiv v1 位置 | DOI 作者接受稿位置 |
|---|---:|---:|---:|---|---|
| O2Former，HRSID inshore | 43.1 | 69.5 | 45.9 | Table III | Table IV，同值 |
| O2Former，HRSID offshore | 71.7 | 97.5 | 91.0 | Table III | Table IV，同值 |
| O2Former，PSeg-SSDD inshore | 63.2 | 91.6 | 81.2 | Table II | Table III，同值 |
| O2Former，PSeg-SSDD offshore | 71.9 | 98.0 | 92.9 | Table II | Table III，同值 |
| Mask2Former，HRSID inshore | 17.9 | 36.9 | 16.2 | Table III | Table IV，同值 |
| Mask2Former，HRSID offshore | 60.6 | 95.2 | 74.9 | Table III | Table IV，同值 |
| Mask2Former，PSeg-SSDD inshore | 52.5 | 83.0 | 59.9 | Table II | Table III，同值 |
| Mask2Former，PSeg-SSDD offshore | 67.2 | 94.0 | 83.2 | Table II | Table III，同值 |

数值来源：[arXiv v1 Table II/III](https://arxiv.org/html/2506.11913v1)、[接受稿 Table III/IV](https://www.researchgate.net/publication/403111230_O2Former_Direction-Aware_and_Multi-Scale_Query_Enhancement_for_SAR_Ship_Instance_Segmentation)。这支持“两个已读版本的上述条目一致”，不支持“最终 VOR 所有表格与预印本完全一致”。

arXiv 元数据页显示 v1 是 2025-06-13 提交的版本，正文首页五位作者依次为 Fei Gao、Yunrui Li、Xiaoyu He、Jinping Sun、Jun Wang。应避免给该 v1 条目加入六人期刊版的末位作者。[版本与正文](https://arxiv.org/abs/2506.11913)、[v1 首页](https://arxiv.org/html/2506.11913v1)

## 推荐 BibTeX

正式引用使用机构记录中的终卷页和题名拼写；作者与 DOI 同时受 Crossref 支持。未直接读取终稿题名，故保留上述 Multi-Scale/Multiscale 差异说明。

```bibtex
@article{o2former2026,
  author  = {Fei Gao and Yunrui Li and Xiaoyu He and Jinping Sun and Jun Wang and Amir Hussain},
  title   = {{O$^{2}$Former}: Direction-Aware and Multiscale Query Enhancement for {SAR} Ship Instance Segmentation},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume  = {19},
  pages   = {11633--11645},
  year    = {2026},
  doi     = {10.1109/JSTARS.2026.3677088}
}

@article{o2former2025,
  author  = {Fei Gao and Yunrui Li and Xiaoyu He and Jinping Sun and Jun Wang},
  title   = {{O$^{2}$Former}: Direction-Aware and Multi-Scale Query Enhancement for {SAR} Ship Instance Segmentation},
  journal = {arXiv preprint arXiv:2506.11913v1},
  year    = {2025},
  doi     = {10.48550/arXiv.2506.11913},
  url     = {https://arxiv.org/abs/2506.11913v1},
  note    = {Version 1, June 13, 2025}
}
```

## 最小正文更新建议（由根 agent 执行）

- `refs.bib:34–39`：纠正预印本两位作者的姓名；另外添加正式期刊条目。
- `sec_intro.tex:17`、`sec_related.tex:8`：方法概述引用可转为 `o2former2026`。
- `sec_exp.tex:150,163`：HRSID 文献数值保持 `o2former2025`，表脚明确写 `arXiv v1, Table III`。可用完整句：`Literature values are transcribed from the arXiv v1 tables of O2Former \cite{o2former2025}; the work was subsequently published in JSTARS \cite{o2former2026}.`
- PSeg 文献对照数字说明保留 v1 来源；本地 OAEM 实现若依据 2025 版本实现，仍保留预印本引用，不能因出版更新推定本地实现对齐了期刊版全部机制。
- 若采用一个正式条目覆盖全文，必须在比较表脚仍显式标明历史数字取自 arXiv v1 及具体表号。两个条目更易复核。

以上行号是本次读取时的定位，根 agent 合并后可能变化。本次只生成此核查文档。
