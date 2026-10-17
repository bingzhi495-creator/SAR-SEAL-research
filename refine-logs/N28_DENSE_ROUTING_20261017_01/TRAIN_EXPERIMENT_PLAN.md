# N28：全FIT配对密集路由训练预锁

2026-10-17。执行前状态：N28三seed共同初始化与首个真实实例GPU接口已actual0并Root接收，正式训练尚未执行。每组2,978,555参数；F/M投影首例逐位相同、零残差/全域写入/有限梯度通过。首例不代表全741图已完成新读取器核验；本次在原已资格全量F的基础上，按新reader逐图核SHA、tensor摘要、真实matcher及GT，并完整保留所有实例。

## 唯一假设与控制

SELECTIVE只门控跨尺度传输，其他出边保留D；WHOLE_NODE相同T=G(D)也覆盖其他出边。两臂同一factory来源、参数接口与初始状态、冻结F/e/M/native_L、全mask写入、监督、图序/微批、优化器及端点。原5尺度小型实例条件UNet++适配结构不变，不加入额外模块或修复。原保存native_L是base，完整delta_M经200→800→actual裁剪→original双线性(false)后相加。两臂输出同100自然query的实例mask；固定类别/ordinal/原完整S0，不由新mask重算score、不NMS或过滤，不用连通域代理AP。

使用原FIT741图、1661非空真实匹配GT，忽略/负query不伪标背景。训练仅原GT配对，未匹配query的行为必须在完整100query评价中如实呈现。旧完整F是单层pixel-decoder mask feature，学习层级不是原多层FPN；不重新teacherforward、不读取DEV/TEST参与监督。

## 冻结训练配方

- 三个full64bit seed为202610172801、202610172802、202610172803；先只执行2801。每seed同一已保存virgin共同state同时初始化两臂，不从probe扰动或历史选择终点恢复。
- 总24轮，原完整数据每轮恰741图/1661GT。每图每臂仅一次AdamW更新，所以每轮741次、24轮每臂17,784次。图内按原GT顺序连续微批4，尾批保留；微批实例均值loss乘chunk_GT/image_GT后累加，图末统一step。含BN，因此这是固定微批下的图均值，不能称与一次完整G forward数值等价。
- 每GT完整原图监督：loss=0.5*(前景BCE均值+背景BCE均值)+softDice，Dice平滑1，实例等权。原1661GT已由N27核面积均0<area<H*W；任何不符均停止，不补标签/删样。各图目标等权，不采用GT面积加权；另报实例加权描述loss。
- AdamW lr=3e-4、weight_decay=1e-4、betas=(0.9,0.999)、eps=1e-8、foreach=False/fused=False；FP32，无scheduler、混合精度、梯度裁剪、数据增强、早停或DEV选端点。
- 私有CPU orderGenerator以该完整seed初始化，每轮randperm741，同一order给两臂。无dropout或数据随机增强。仅按(0,8)、(8,16)、(16,24)三个实际任务段执行；逐轮保存新checkpoint，包含两个模型/BN、两个优化器全部状态、私有orderGenerator及CPU/CUDA RNG、真实更新计数和历史。续段严格恢复它们；不得将仅加载权重、重建优化器称连续24轮。

## 固定评价与后续门槛

第8/16轮只保存训练曲线/状态，不看DEV选型；第24轮固定端点评价完整原DEV187图/380GT/100query，报告COCO12、AP75、AR、域内外几何及成本。评价代码/源和真实预测接口必须在评价前另锁，不因早期loss选择最佳轮次。该内部DEV不冒充未接触正式TEST。

主比较为同seed SELECTIVE−WHOLE_NODE；同时保留原强R AP68.09938495及N26已知质量排序三seed均值69.53736694。用SELECTIVE相对max(WHOLE_NODE,69.53736694)的AP点差作为保守效用筛选门槛；N26均值是额外强参考，不冒充与本轮种子一一配对或同类训练预算。首seed差≥1且SELECTIVE胜WHOLE_NODE则完成其余2seed；0<差<1且SELECTIVE胜WHOLE_NODE最多补2802，只有两组均为正且平均差≥1才补2803；差≤0结束当前配方扩展。保留所有结果和实际停止规则，不仅报最高seed。

公平配对支持路由组织效用；即使超过强参考也不能直接认定根因抑制、首创或TGRS就绪。幅值衰减/普通注意力、充分训练和泛化仍需后续针对性验证。若没有效用，则结论限定当前适配、24轮配方和冻结教师；不宣称用户所有UNet++设计失效，也不据此重开已停止N20/N23配方。

## 执行与完整性

每段单次GPU0任务，分配上限4GiB、整卡观测8GiB、CPU0/1线程2、RSS≤16GiB/可用RAM≥16GiB、1800秒，无自动重试。观察超时只复查原session；只有actual终态后才接收。原SRC/绑定/前段checkpoint冻结不改，运行期间不为发布读取或哈希绑定载荷。公开源码先于执行；每段actual0、持有身份、fresh RAW FIRST及Root接收后再发布摘要。任何失败完整保留，不能从未资格产物或中断轮次宣称成功。

当前无新AP、正式TEST或跨集结果，核心创新与论文投稿依据尚未成立。

