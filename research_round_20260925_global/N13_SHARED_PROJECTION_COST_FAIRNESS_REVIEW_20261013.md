# N13 后续质量—成本假设：共享投影的公平强对照审阅

2026-10-13；**普通静态研究审阅，尚未实施优化或测量速度。当前固定 N13 两臂、三种子、48 epoch、SOURCE 和资格链均不改。** 本文补充未来 utility / error × actual flip 比较的成本边界，不新增文献或正式实验臂。

结论：学生的 F256→32 可每图共享，稠密实际 flip 对照的 PR-FL91 也应共享第一层的 F256→91 贡献。只优化学生、把逐 query 重复计算第一层的朴素 PR 当成本基线，会夸大“避免全域 PR”的收益。共享后的 PR 仍有逐 query 的非线性计算，是否值得省略须在同等优化下实测。

## 公开 SOURCE 所证明的计算顺序

本轮只读 `github_release_20261011` 独立发布副本和两份普通 proposal/addendum；没有读取原 SCI、运行载荷、数据或资格文件，没有数值导入、哈希或科学执行。

| 独立发布副本，目录均在 `research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/` 下 | 静态依据 |
|---|---|
| `N07_frozen_E24_original_detail_point_refinement_source_20261012_01/cache_fit.py:269–277` | `sampled_features` 每16通道执行原图 renderer，再按原图 flat 点索引 gather，拼成 F_samples；没有在粗网格先 LN。 |
| 同 N07 的 `qualify_evaluation_cpu.py:233–238` | 独立 CPU renderer 显式为200→800 bilinear、裁剪有效 h/w、再以 `crop[:,None]` bilinear 到 H/W，均 `align_corners=False`。它是独立 reader 的公式依据，不冒称本轮读取了原绑定 GPU renderer。 |
| `N10_PR_FL91_direct_precedent_control_source_20261013_01/point_heads.py:31–37,78–89` | 原 PR 为257→91、92→91、92→91、92→1；先对原域采样的 F 作无 affine LN256，eps=1e-5。`c=clip(L,-8,8)/8` 注入每层；返回 ungated residual，旧 logits 另有 absL 门。e/patch 被验证但不参与残差公式。 |
| 同 N10 的 `evaluate.py:483–508` | 先每16通道构建完整原域 F，再逐 query gather；head 的像素块为2048。 |
| `N13_sameK_fixed_DEV_evaluation_source_20261013_01/evaluate.py:201–212,259` | 复用上述原域 F；只在真实 arm region 取点，使用 ungated residual 写 `L+r`，域外保持原 L。 |

因此正确共享对象是**同图、同冻结 head 权重、同原域像素上的归一化特征贡献**，不是任意粗网格投影。现有总参数40497不变；这项工程优化不构成新的学习机制。

## 给 actual flip 对照的可用优化

设原域逐像素特征为 f_x，n_x=LN(f_x)，把第一层权重分成 W_F 和 w_c。实数算术下原第一层恰为

\[
V_x=W_F n_x,\qquad h_{1,qx}=\operatorname{ReLU}(V_x+w_c c_{qx}+b_1).
\]

V 是91维，可以每图只计算一次，bias 只加一次；其余两隐藏层和最终层仍保留原 c 注入和非线性。每 query 从同一 V 读取并完成后续计算，得到完整固定 PR 候选 `P'=1[L+r>0]`，再算 `δ_actual=1[P!=P']`。此路径不用 GT，必须覆盖排序所需的全域候选；不能在只计算 K 点后称实际全域 flip 对照。

学生和三类 error head 均有自己的 F256→32 权重，可复用同一 n_x，但不能把不同 head 的 V 当相同结果。不同 PR 种子、端点或两臂权重也不能混用 V。冻结推理可共享；训练期间跨每实例 optimizer 更新复用旧 V 会改变梯度或使用过期参数，本文不授权这种训练改动。

LN 与 bilinear 一般不交换。`W_F LN(renderer(F))` 不能换成 `renderer(W_F LN(F粗网格))`；后者是不同函数。应保持200→800、有效裁剪、H/W取样及 LN 的顺序。若未来做流式取样，需保持完整原图的坐标、边界与两段插值规则；不能用局部 tile 尺寸重新缩放或改为单段插值来冒称等价。

## 静态算量与真实存储账

以下只数 Linear 的 MAC，排除 bias、LN、ReLU、插值、描述符、排序、取点、访存、启动和导出，不是速度或能耗结果。令 N=HW，Q为参与比较的原生 queries，T=ΣK_q，U为稀疏支持的去重像素并集大小。

| 计算路径 | Linear MAC |
|---|---:|
| 朴素全域 PR | `40223 QN`，其中 `257×91+2×92×91+92=40223` |
| 共享第一层 F 的全域 PR | `23296 N+16927 QN`，其中共享 `256×91=23296` |
| 学生或 error 的全域8451参数 head | `8192 N+224 QN`；逐 query 四量→32及32→3分别128、96 |
| 所选支持上的 PR，跨 query 去重后共享 F 贡献 | `23296 U+16927 T`；也可用朴素 `40223 T`，需选择实际更合适且合格的实现 |

Q=100时，共享全域 PR 为 `1715996 N` MAC，而朴素为 `4022300 N`；这个差值只能说明 Linear 重复计算减少。它不能预测完整速度。学生仍须执行 selector 的全域逐 query 激活、exact K 排序和所选点 PR；不应把“8451参数”当整个系统容量或把 K 当真实翻转数。

完整 FP32 V_PR 缓存为 `364 N` 字节；每个 V32为 `128 N` 字节。原域 F256为 `1024 N` 字节，若另保留归一化 F 又增加同量；100 query 的原 L 为 `400 N` 字节，每份完整 L'/residual 同量，uint8 δ为 `100 N` 字节。示例1024²原图，V91单独364 MiB、F256单独1 GiB；这些是公式示例，不是本数据实测峰值，也不能相加后当实际同时驻留量。

可选两种同函数执行方式：完整 V91缓存后逐 query/2048点处理；或原域像素块 S 内取得 F、LN、V，再处理该块各 query并流式维护全局 exact K。块内仅 F[S,256] 与 V[S,91]就合计 `1388 S` 字节，另有 LN临时量、原 L、后续逐 query 激活、输出、排序和 allocator。后者不必同时保存所有 L'/δ，但必须保持全图分数与 flat tie-break 的 exact K，不能每 tile 独立选 K。共享缓存的生成、转置/contiguous、读写和释放都应计入成本。

稀疏 PR 可只为去重并集 U 构建 V，或直接逐点运行原 head；为很小 T 强行生成整图 V91反而可能更慢。学生和 dense baseline 均应获得合理 chunk/共享优化，不能只为一方挑选有利缓存模式。公共 native forward、原域 F取样、描述符、质量估计、排序、I/O和导出同时报告；另报 refinement 增量，避免骨干掩盖选择器开销。

## 信息与数值等价边界

拆分首层在实数下等价，**尚不能声称 FP32逐 bit 等价**。原 `cat(n,c)` 的257项 Linear 与先256项矩阵乘再加 `w_c c+b` 可能使用不同 reduction/FMA/GEMM；chunk形状、布局和 kernel也可能改变舍入。近零 ReLU、后续残差以及最终 `L+r>0` 可因此改变。+0与−0本身在严格 >0规则下都判0，不能把符号零差异直接说成 mask翻转；真正风险是近零值跨过0。

未来若要称“同一个固定 PR 候选”，须另行锁定新实现并由所有者实际资格核验，明确是字节级 logits一致还是 mask/δ级一致；后者不得写成前者。近似 allclose、较少 MAC或符号大部分相同，不能替代所声明的等价门。原生 all100 L、query对应、类别、分数、顺序、K与导出规则继续保持；不能用 GT选择容差或在见到失败后放宽判据。若优化改变候选 δ，就不能把结果无条件归因于同一 PR 的支持分配。

utility标签 `u=δ(2e−1)` 及实际 flip 强替代 `δ_actual[2(p_FP+p_FN)−1]` 的信息边界仍保持：压缩描述符 Z 的错误估计不是自动等于 Bayes行动收益；共享投影也不会让像素净收益等于 IoU/AP。实例质量加权仍是待检验代理，不是 AP最优理论。

## 收缩后的未来可否证假设

可发布为：**在相同冻结 PR、输入与训练曝光、exact K和评价规则下，预估支持后只执行所选 PR点，能否相对已共享第一层并合理分块的 actual flip 强对照，保持预登记质量区间且降低完整推理成本？**

未来先完成数值资格，再以同硬件、同 warm-up/计时边界、相同同步及 I/O规则测量完整时间和峰值存储。若优化后的稠密对照消除了成本优势，或稀疏预估未保持锁定 AP/高阈值质量区间，则不支持这份数据及配方下的质量—成本假设；不能以朴素 dense更慢、单独 MAC或局部像素收益挽救结论。该审阅不给当前 N13增臂、重训或改预算，也不宣称创新成立或达到 TGRS。

相关普通文档：`N13_LEARNED_EDIT_UTILITY_NEXT_HYPOTHESIS_20261013.md`、`N13_ACTION_UTILITY_AND_ALLOCATION_DIRECT_PRIOR_ADDENDUM_20261013.md`。上述源内行号来自本次独立发布副本的静态文本，不代表读取或重新核验正在运行的科学载荷。
