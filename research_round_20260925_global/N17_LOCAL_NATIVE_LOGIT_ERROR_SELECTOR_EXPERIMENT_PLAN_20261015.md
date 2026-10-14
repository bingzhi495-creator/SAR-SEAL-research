# N17：ERROR 八邻输入的一组配对检验

**状态：UNBOUND、未执行。** 仅依据 ROOT 提供的已核验摘要与 FIT 元数据编写；本作者未读取科学载荷、资格链或 Git。N16 的固定 E24 首种子 DEV 已合格且触发 STOP：UTILITY AP=67.6667317718、ERROR AP=67.8959321748，强固定 PR 参考 (R=68.09938494745545)。这不证明所有 selector 或空间信息无效；DEV 也不是正式 TEST。

**唯一待检验假设：** 同参数量、同初始化和同训练预算下，同 query 的局部 native-L 信息能使 ERROR 选点优于中心复制控制及强固定 PR。只启动下述一组对照，不声明拓扑机制、核心创新、GLDet GIM 复现或跨划分外部 AP 优势。

## 固定两臂及唯一差异

| 项目 | LOCAL-ERROR | CENTER-ERROR |
|---|---|---|
| 追加八维 | 同 query 3×3 的八邻 native L | 同一中心 native L 复制八份 |
| 共同模型 | 521→32→3、ReLU、四张量、16803 参数 | 完全相同 |
| 共同原513维 | 原点 F256/e256 各自 non-affine LN（eps1e-5），原始中心 L/8 | 完全相同 |

追加值固定为 FP32 `c(v)=clip(v,-8,8)/8`，顺序 NW/N/NE/W/E/SW/S/SE；原513维的中心 L/8 不新增裁剪。越界邻坐标逐轴 clip 到本图边界（replicate），不跨 query、不跨 flat 行；两臂使用同一 c。以上是首次取数前须进入新锁协议的工程定义，不在 DEV 调整裁剪或 padding。

以新的私有 scout 初始化种子 **202610150171** 生成共同原四张量；扩展 hidden.weight 的原513列精确复制，新增32×8列全0，另三张量不变，再向两臂复制完全相同的四张量。两臂初始函数相同；新增八列均可训练。重复种子固定为 **202610150172、202610150173**，下文简称171/172/173；初态 digest 由 ROOT 在首次绑定前固定。这些种子只改变私有 scout 初始化；共享冻结 teacher/PR，不是 full-backbone 多种子，也不补旧 N16 seed163。

## 样本、预算与无 GT 部署

固定 FIT741 与旧内部 DEV187；冻结 teacher、强701 E24 PR、原 F/e/native-L、原 class/S0/query order。两臂共享原每图4096点、重复抽样顺序、ERROR 的 FP/FN/correct 三类标签与 importance：有效 query→C 内点分布不变；两 stratum 各2048，单 stratum4096；使用原 mean(weight×CE)，不新增类别重采样或归一化。

负 query 的监督仍是既有**新增 empty-set 代理**，不称标注已证明的真实背景；ignored 不转空集。候选仍 C=Ω∪P∪Chebyshev₈(P)，Ω为absL≤2，P为L>0，K=Ω点数；K0 原样保留。score 仍 `2(pFP+pFN)-1`：实际 GPU logits 回 CPU 做 canonical NumPy FP64 softmax；主 atmost K 且 score>0，tie 为 score降/absL升/flat index升，选择后应用原 PR raw residual。GT/标签不进入部署排序。

AdamW、lr=3e-4、wd=1e-4、betas=(0.9,0.999)、eps=1e-8、恒定 LR、无 AMP/额外预算均沿原配方；固定24轮，每臂17784更新、72843264点曝光、相同图序与点序。**主指标仅 E24 atmost 全18700自然 query 的官方 mask AP（百分点）**；E8 与 exact 仅次要曲线，不选 winner、不救主门。

## 薄缓存与必要执行顺序

ROOT 已核 FIT 的741份 full same-query native_L_refs 元数据；实际提取与独立 CPU 同输入资格尚未执行。新薄 cache 只按既有 query/flat 坐标 gather 八邻值，不新跑 teacher、不保存邻域 F、不使用 DEV 构造 FIT 样本。3035136点×8×4=**97124352 bytes** 原张量，741个 NumPy 文件另计 header/manifest；2048点块×8×4=65536 bytes。

执行顺序仅：ROOT 新协议/来源绑定 → 薄 cache 提取及独立 CPU 坐标、边界、重复点、FP32值与原标签/权重/次序 exact 回放 → 两臂共同初态与冻结状态核验 → 配对24轮及端点 CPU 资格 → 固定 DEV GPU/独立 CPU 完整读回 → 判定。资格应保存 actual GPU 输入；新增 scout 数值与 FP64排序的 selected indices exact 分开核验，PR 仍 absolute1e-4/rtol0 且 after-sign exact；CPU 近似 logits 不替代实际 GPU logits 定义排名。完整自然100 mask/RLE、原 scores/order、官方 COCO12 保留；接口或数值资格失败不得改容差或算方法 STOP。

每实际 leaf 保持 CPU01/两线程、1800秒、单 leaf、无自动重试及原资源门。提取/资格/训练/评价分别监督；不预估可靠 wall 时长，不因截止缩范围或复用失败 partial。ROOT 负责实际资源、Source/RAW/资格与后续发布；本计划不启动科学执行。

## 首组预锁判定与证据边界

令 AL、AC 为 LOCAL/CENTER 的 E24 atmost AP，Δ=AL−max(AC,R)。用未舍入原数值，不加阈值容差；任一主 AP undefined 则规则不可评估，不造0或借次要指标。

| 首组171结果 | 唯一后续权限 |
|---|---|
| Δ≤0 | STOP 当前 N17 配方，不补种子或训练预算 |
| 0<Δ<1 | 最多补一组172；不立即授权173 |
| Δ≥1 且 AL>AC | 补两组172、173作初始化重复 |

若首组落在0<Δ<1，完成172后，仅以两组已合格、可评估主结果计算 `mean(AL)-max(mean(AC),R)`：该值≥1且**两组各自AL>AC**才允许173，否则停止。最终三组 meanDelta 采用同一公式 `mean(AL)-max(mean(AC),R)`；即使 meanDelta≥1且三组各自胜AC，也仅支持共享冻结 teacher 下的 scout 初始化效用，不构成创新证据。

正收益也仅支持当前冻结 teacher、代理标签、数据划分与短预算下的邻域输入效用。中心复制控制匹配参数量与初态，但不匹配八邻值分布/协方差或有效输入秩，收益可能来自局部置信度、容量使用或优化；尚不能专门归因空间排列。保值破位控制及跨集/正式 TEST 验证只在主门正收益后另行预锁；这里不扩大实验清单，也不把同 teacher 的 scout 重复称独立 backbone 证据。


## 基础模块状态（2026-10-15）

[独立普通kernel/init源码](code/N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py)和[ABI](code/N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.md)已保存，原107行快照一次针对性接口审阅未见确定阻断；[可移植6项合成测试](code/test_N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_SYNTHETIC_ONLY_20261015_01.py)原实际0（0.022秒）。覆盖同query/非方图/边角/1×1/饱和/空点/非法索引、独立NumPyexact、旧四tensor/补零/RNG和初态函数一致性。新forward(F,e,L,extra8)需要新入口适配，旧三参入口不能直接使用。测试仅合成输入/旧普通factory，未生成741真实邻域cache、未完成真实输入数值资格、未启动N17训练/DEV评价；不得以此宣称涨点。
