# U52：共享特征梯度累积工程候选（仅源码，未执行）

本候选落实项目已有普通文档 `code/U34_FULL_PLAIN_INIT_DRAFT_20261017_01/SHARED_FEATURE_VJP_REVIEW.md` 的同一工程建议。它是普通链式法则的计算组织调整，**不是新模型、学术创新、DSConv修复或已验证提速**。原 U40/U50/其源码、配方、优化器状态和预算均未改动。当前没有 runner、自动切换、自动重试或实际模型/数据执行。

## 源码证据与瓶颈边界

只读公开普通副本：U39 `resource_probe.py:186–187` 一次 `adapter.encode(x)`，但 `:199` 直接把该完整图输出传给每个query，`:217` 各自执行 `loss.backward(retain_graph=ordinal+1<G)`。因此在当前图没有特殊截断的情况下，每个query都沿同一个编码图反向传播。它复用前向，**没有自动复用编码器反向**。U40公开设计明确保留这一实现，并锁 `VJP=False`；窄范围源码检查未找到本候选已执行的证据。

原 adapter SHA9ad221… 的 `encode`（`:45–72`）涵盖 backbone 和 `feature_projection`；`predict`（`:89–98`）只使用已经投影的共享特征、query_projection、mask_projection、head。它不另跑主干，没有头部BN/dropout。因此当前实现存在明确切分边界，值得做一个独立工程候选。

Root提供的单次DGX观察：warm G1 PLAIN0.907s/FULL1.360s；另一图G27 PLAIN10.179s/FULL15.541s。它们的原图大小、GT数、读出/原图渲染量不同，且有背景GPU服务；不能据此归因重复编码器反向是主要瓶颈，也不能算本地/DGX速度优势。查询头、原图loss、传输/同步及保存张量解包都可能重要。需要同图、同状态的受控测量。

## 代码与原接口

`shared_encoder_update.update_shared_encoder(torch,c,reader,index,adapter,optimizer,params,effective,inactive,helpers,step,stage,arm_record,report)` 与原 U39 `_update` 参数相同。Torch由caller注入；模块顶层只有标准库导入。未来caller必须显式登记这个新候选并闭合原 adapter/U36 helpers/source，不得悄悄替换或猴子补丁正在执行的 U39/U40/U50。这里没有接上任何现有运行入口，也不自行赋予资格。

保留原一次读取/一次编码、CPU保存激活、query1原顺序、原图render+native L、每实例原 BCE+softDice(smooth1)/G、原始零末层校准、完整梯度检查及每图一次AdamW。唯一反向组织改变：

1. `u=adapter.encode(x)` 保持其完整原图；令 `v=u.detach().requires_grad_(True)`，不额外clone、不原位修改共享存储。
2. 原顺序每个query用 `predict(v,e,M)` 和相同loss，`loss.backward(retain_graph=False)` 累积读出参数梯度与 `v.grad`，立即释放该query头图。
3. 所有G完成后，调用一次 `u.backward(v.grad,retain_graph=False,create_graph=False)`，再执行原梯度验证和一次AdamW。不额外除G，不在中间清掉叶子梯度，不提前step/裁剪/weight decay。

若编码参数θ与读出参数φ分离、没有绕开u的编码辅助损失，则实数一阶导满足 `∇θΣℓq = J_EᵀΣ∂ℓq/∂u`。这是本候选的条件性数学依据，不是FP32逐位等价证明。FP32把“各query的参数VJP相加”换成“先加特征梯度再VJP”，会改变舍入与内核累积顺序；不能声称数值已相等。

最小实现会拒绝：不同参数对象共享可写存储、编码/读出参数共享或未覆盖、重复/额外优化器参数、参数或模块反向变换hook、需要梯度的teacher/输入、非FP32/AMP。它只支持当前单Tensor编码输出；多输出需独立完整梯度接口，不能仅取一支。编码器辅助loss、高阶梯度、自定义非线性梯度处理或未来可学习e/M不在支持范围。正常forward计数hooks允许，当前BN仍只按原训练前向执行一次；FULL旧的被覆盖BN分支行为并未修补。原DSConv b4b612…也没有被U46候选替换。

当前额外叶子梯度约19.53MiB（1×32×400×400 FP32）；参数/Adam、encoder forward活跃skip和保存张量仍存在，不能保证降低峰值、消除OOM或提高速度。CPU保存的forward事件不变；重复解包是否减少及实际传输成本必须测量，不能把事件累计字节当驻留内存或节省量。

## 未来独立验证的最少充分范围（尚未执行）

必须避开零末层陷阱：第1步原final为零，encoder梯度可能全部为零，只比较首步不能验证候选。先从共同完整state用同一次标准warm-up使头非零；从该相同参数、BN、Adam(m/v/step)及CPU/CUDA RNG状态分开运行原法和候选，**G27**逐张量比较loss、特征梯度、所有有效参数的None/有限性/梯度、更新后参数、BNbuffers、Adam每一项及RNG；inactive保持None。随后各自沿同一G1检验carried state。只换本次反向组织，不混U46、AMP、分辨率、loss、读出或参数变化。

在看到结果前预锁逐张量绝对/相对梯度与状态阈值、相对误差分母下限、失败规则；同时保留baseline-vs-baseline重复对照以识别数值噪声。不能事后放宽阈值、用全局范数掩盖个别参数失败或把第一步全零当通过。数值合同通过后，才比较同步整步耗时、各阶段耗时、实际encoder反向次数、CPU保存/解包事件、GPU峰和RSS，至少在同图同状态控制背景服务；新增boundary校验成本也应纳入整步时间。任何OOM/资源失败保留，不自动重试。

当前仅一个微型标准库metadata顺序/非法G与step检查：缺模块RED `de8833` → GREEN `e6cbdf`，未导入Torch/NumPy、未访问实际Ops/RAW/Q/冻结SOURCE/图像/GT/权重、未SSH或进行数值运行。该检查只验证代码实际使用的query序号规则，不是梯度、性能、训练效用或AP证据。
