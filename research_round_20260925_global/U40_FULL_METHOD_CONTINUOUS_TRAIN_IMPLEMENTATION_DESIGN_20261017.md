# U40：完整主干连续训练的最小实现评议

状态：planned / not run。只读普通已发布源码与主计划，未导入 Torch/NumPy，未读原图、标注、缓存、权重、运行中的 SOURCE/RAW/Q 或 Ops/H/SCI。本文不构成训练准入，不改变 U39 试点。Root 统一拥有科学执行和后续绑定。

依据：公开 `research_round_20260925_global/code/` 下 U39 `resource_probe.py` SHA `378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d`、U34 `shared_initialization.py` SHA `4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292`、U37 `rgb_matched_reader.py` SHA `678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025`。本次所读普通 `refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_PLAN.md` SHA `73bbd42f79a3f07fcfb2fe0e6b0f98297664eb9d8d4e3362ae283e04f6de7d4e`。路径均以 `E:/SAR/seal` 为工作区、`github_release_20261011` 为公开副本根；不读取其运行绑定版本。

## 1. 可以直接复用的数值核心

建议新 `train_core.py` 只实现分段、恢复和保存，固定 SHA 私有加载 U39；直接调用其 `_partition`、`_bn_signature`、`_update` 及固定 U36 helper。不能调用 U39 的 `probe`/`_run_arm`：两者明确硬编码三次丢弃试点及对应状态，后者不能冒充正式训练器。

U39 `_update`（153–242行）已经是一张图一次更新：原 U37 reader 加载 RGB400、原 GT 匹配 e/M/native-L；encode 一次且只在 encode 外包 canonical `save_on_cpu(pin_memory=True)`；逐实际 G 的 query1 原尺寸 BCE+smooth1 softDice 各乘 1/G；最后一次 backward 才释放共享图；对预声明 effective 参数做一次 AdamW。无需复制 loss、渲染、反传、梯度检查或 optimizer 检查。传入的 `step` 必须是该臂全局更新编号 1..37050，而非叶内编号：这样首步零头检查只执行一次，Adam 状态检查持续到真实全局 step。

保持 FP32、B1、400、query1、AdamW lr3e-4/wd1e-4/betas(.9,.999)/eps1e-8、foreach=False/fused=False，不引入 scheduler、AMP、VJP、checkpoint 重算、增强或新监督。GT 只入 loss，F 仅保留旧 reader 校验用途，旧头和其他 inactive 参数仍分配但不进优化器。保留各原算子及 active BN 的正常训练更新，不把 unused `bn1` 的 buffer 一律冻结。

## 2. 一次叶子的最小接口与所有权

建议接口：`train_segment(torch,np,c,context,modules,source_refs,segment,parent_checkpoint_refs,output_directory,report=None)`。只接 CPU/JSON 元数据和已闭合模块，不接模型、optimizer、GPU tensor 或已反序列化 checkpoint 对象。返回 JSON 报告与科学输出引用，不返回模型/状态张量。运行器、原准入/硬时限/终态接收不放进此函数。

`segment` 至少固定完整 seed `202610173101`、两臂顺序 `[PLAIN,FULL]`、共同全局区间 `[start,stop)`、全37050图序摘要、source/recipe/人口身份。两臂每叶处理完全相同区间；只有双方均成功保存并被接收，配对前缀才能从 start 推进到 stop。不因某臂更快而额外训练。第一叶调用原 U34 `build_pair`，保留其实际104公共状态及 adapter 同源回执；之后叶只从各自准确父状态恢复，不能再次以公共初始化覆盖已训练参数。

CPU 模型/检查点字典由内部拥有，不从上层临时索引出 active model 作为调用参数。最小方案每叶仅在 CUDA 尚冷时调用一次原 U34 pair factory；其源码178–188、277行明确要求 CUDA 冷，不能在第一臂结束后再次调用。之后沿 U39 的做法：内层训练函数从内部容器 pop 当前臂，另一臂保持 CPU；恢复叶对当前臂严格加载其完整父状态，覆盖新构造的临时初值，再恢复 RNG。重复构造不计为新的训练初始化。所有构造耗时与 CPU 峰值如实计入本叶，不在本设计另造单臂 factory。

每臂同一模型、同一 optimizer 连续完成该区间。每图结束只清梯度和图/样本引用，不在图间 empty_cache 或重新构造模型；保留 Adam/BN 真实驻留。每臂保存后释放模型、optimizer、checkpoint CPU 快照及所有局部别名，再 GC/sync/clearCublas/empty_cache/sync，核 GPU allocated/reserved 为零后才能进入下一臂。失败 traceback 不保留 GPU frame；仍按 U39 保留真实已返回更新数。

## 3. 恢复的对象必须完整，而不只是权重

检查点在更新完成、图和梯度释放后生成，所有张量转 CPU，只写独占科学输出目录，禁止发布权重。保存同步完成再释放；临时文件写完、字节摘要和载荷自检成功后原子形成本次新端点，不覆写父端点。不同时持有两臂大快照，不做多余整模型 deepcopy；CPU/pinned 高水位仍由资源控制观察。

每臂端点至少包括：

- 完整 model state（包含所有 BN buffers、inactive 声明状态和 adapter），严格名称/shape/dtype；有效/失活名称及排序的 optimizer 参数名清单。
- AdamW 全部 state 与 param_groups：每参数 `step/exp_avg/exp_avg_sq`、超参和位置对应。不能只重建一个同超参 AdamW 就称续训；也不能依赖参数数量相同来推定映射正确。
- 全局完成更新数、下一图序游标、epoch/轮内位置、完整图序摘要和累计实际 trace 摘要；完整人口、原匹配和所有来源身份，父检查点 ref、初始化回执身份。
- Python、NumPy、Torch CPU、当前 CUDA generator 的各自 RNG 状态；若采用额外私有生成器也必须保存其状态与算法/版本。PLAIN/FULL 各保存自己的 RNG，不能在叶间用另一臂或新构造消耗后的 RNG 覆盖。

恢复顺序为：CPU 加载来源已验证的自有 checkpoint；构建精确原模型并 strict load 全状态；核 inactive 名单/参数顺序；设同一4GiB分配上限，模型转 GPU；按同顺序建 AdamW 再 load 全 optimizer state；核 moment 对应/shape、step 与全局游标。按照原 Torch2.0.1 非 capturable AdamW 的实际 state device 规则恢复，不能把 CPU step 标量一概强搬 CUDA。最后恢复各 RNG，随后才取下一张训练输入，避免初始化/加载吞掉恢复后的随机序列。BN buffers 必须逐项等于存盘值；跨叶不重新初始化/重估 BN。

“状态逐位恢复”和“整个 CUDA 训练轨迹逐位可重放”是不同主张。前者必须实核；后者只有在已固定算子/库/确定性配置下做过实际分段对照才能报告，不能由固定 seed 推定。

## 4. 图序、50轮与1800秒边界

主预算每臂50×741=37050更新，每轮完整1661个原匹配实例；两臂共74100更新，资源试点不抵扣。`202610173101` 是完整首种子，不能写成整数3101执行。先只启动首组；其余种子遵循既有完整DEV端点规则，不由训练中间值选择。

正式预锁需明确图序而不仅写“fixed”。最小建议是每轮原 dataset_index 0..740、连续50轮，两臂共享同一显式37050列表及摘要，没有隐式打乱。如果 Root 决定预锁按轮打乱，则须在训练前由具名私有生成器一次生成并保存完整列表及其规则；各轮均验证恰好741唯一索引。运行只按已保存列表与游标消费，不在叶边界重新抽签；本文不自行更换主计划图序。

每叶两臂共同 `[start,stop)`，不要求每叶恰好一轮。1800秒包含两臂训练、输入校验、恢复/保存、收尾和原执行成本，不是每臂各1800秒。分段长度在看DEV之前锁定；依据 U39 已完成实测及保存/恢复成本留余量选取，但少数图耗时不能保证所有图均适配。不要照搬9GT单图5.8秒估算全人口，也不能承诺一轮能装入1800秒。可预锁轮内段界与跨轮游标，末段不足时严格按剩余预算截断，不丢尾样本。

如果真实叶超时/OOM/校验失败，保留已发生更新、partial和父端点，不自动降低配额、改变 query chunk、跳图或重跑。中间未保存状态不能说成“精确续训”；若需要从已接收端点重放或改变分段，须明确另立恢复尝试，保留失败的实际计算账本，不能把重复尝试抹成原轨迹。若一臂已成功保存、另一臂失败，配对前缀仍不推进，已完成臂状态保留，不悄悄再次训练它。

磁盘也必须在预锁前估算。Root 本轮普通运维反馈 D 约350.34GiB、E约130.72GiB空闲；本评议没有自行读取运行盘/载荷。每个双臂完整 model+Adam 端点暂估约1.4GB量级，必须以首次真实保存字节量核实，不能写成精确容量。以未决定的256图/段举例，37050更新需145个配对端点，累计约200GB量级，另有原存档、日志、原子写临时文件和余量；更短段会显著增加保存次数。正式合同记录预计端点数/总空间和实测修正，保留既有冻结原件，不自动清理旧 checkpoint 以腾空间。

## 5. 最小必要新增验证与日志

U38 已支持全人口读取/公共初始化；U39 若通过，支持指定连续图序的暂存与 Adam 驻留，但不验证检查点恢复。按 Root 最新范围，不另开 CPU/GPU 资格叶：首次正式段实核保存后的全状态/名称映射/计数，下一正式段在更新前实核恢复的 model/BN/Adam/RNG/游标与父端点一致，随后核实际下一图和 global step 增量。不得多执行、丢弃或重演已计入正式预算的 optimizer 更新来伪造新训练。仅状态 roundtrip 与后续连续计数仍不证明“不中断执行 vs 跨进程恢复”的全轨迹 bitexact；报告不作这个额外主张。若将来确实需要该主张，才另立同输入连续/分段对照，不把它设成本轮串行前置资格。

正式逐图继续复用 U39 有效梯度/失活梯度、Adam step/finite 与语义头零调用检查。日志可在完成检查后压缩为 index/image/G/query摘要、global_step、loss/BCE/Dice、检查计数、真实更新返回、耗时和资源摘要；完整失败名字/原因保留。不把全部逐参数列表积累37050步在内存，也不保存原图/GT。每段保留实际图序trace，恢复后确认无遗漏/重复且双方同前缀。

资源报告区分 GPU allocated/reserved、pinned保存/恢复累计事件、进程RSS、系统可用RAM。累计恢复字节不是驻留量；释放图后RSS不下降也不自动等于泄漏。跨叶退出会清除进程缓存，不能由每叶终点低RSS声称单进程50轮稳态；报告实际采用的分段执行和段内增长，保持原4GiB allocator、8GiB整卡观测、16GiB RSS/可用RAM及硬期限，不静默放宽。

## 6. 固定端点评价闭合与停止边界

两臂均完成37050更新并实际退出后，使用固定最终端点做完整内部DEV评价；不在中途看AP、不挑轮次。先接齐完整DEV原100候选 M/e/L/S0/类别/几何来源，再封存两臂完整预测后统一读GT；训练匹配G缓存不替代评价全100接口，U32单图也不是完整DEV数据证据。固定原分数、候选顺序、渲染和原计划指标/效用门；不因正式训练成本改变比较对象或阈值。

这份设计的下一实现是一个小型状态管理 core，将恢复检查直接纳入正式段，不新增资格叶，不是新网络或新实验家族。U39尚未通过时不启动正式训练；通过后不再凭空增加多轮相同资源资格。正式训练开始前只需将明确的分段/图序/状态合同、固定评价输入准备范围和已有资源结果落实到执行预锁。源码准备与同步不等于数值验证、训练完成或主方法有效。
