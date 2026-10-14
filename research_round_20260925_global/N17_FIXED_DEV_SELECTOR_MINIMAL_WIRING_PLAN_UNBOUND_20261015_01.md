# UNBOUND — N17 fixed DEV 最小选择接线

仅静态接口方案，未读取实际 SCI/Source/spec/Q/权重/缓存或执行实验/Git。依据公开 [N16选择body](code/N16_ALL100_SCOUT_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261014_01.py)、[N16 shard GPU捕获源码](code/N16_FIXED_E24_ALL100_SCOUT_SHARD_GPU_UNBOUND_20261014_01.py)及 [N17预锁计划](N17_LOCAL_NATIVE_LOGIT_ERROR_SELECTOR_EXPERIMENT_PLAN_20261015.md)。以下载体可用性只是公开源码 ABI 推断，实际引用与完整资格仍由 ROOT 核验。

1. **固定评价对象。** models[(epoch,arm)]只含 E8/E24 × LOCAL_ERROR/CENTER_ERROR；两臂各做 atmost 主策略及 exact 补充，共8个scout profile。主比较仅 E24 atmost 的 AL、AC 与固定 R=68.09938494745545；E8/exact不得换主指标或挑最好。保留187图/380GT/自然100 query、原 class/S0/query导出顺序及每策略官方COCO12项stats。旧body的 MAIN_CONTROLS/UTILITY 分支不能原样套用：两臂都是 FP/FN/correct，均调用原 error_score，不调用 utility_score、不擅补新 dense/MagNet 参赛臂。

2. **最少数值改动。** 原scout调用 model(F,e,rawL)改为 model(F,e,rawL,extra8)。LOCAL在当前自然q的原[H,W] native-L网格，用真实flat→(y,x)坐标及固定NW/N/NE/W/E/SW/S/SE顺序，逐轴replicate边界，先gather8再clip[-8,8]/8；邻值可位于C外。CENTER用同一rawL.clip/8复制8。原513维scalar仍rawL/8、不新增裁剪；不resize/跨query/以C点顺序假造网格。新factory为 make_scout(torch,202610150171,base_factory)，4tensor16803；载入实际已CPU资格的E8/E24全部训练态，新增列无需保持0或非0、两臂无需必分化。

3. **选择与PR完全沿用。** P=L>0，Ω=absL≤2，C=Ω∪P∪Chebyshev₈(P)，K=|Ω|；GT/标签不进入C、score、排名。每个(端点,臂)在C上升序2048块只forward一次，其原始实际GPU FP32 logits可共享给该模型的atmost/exact。四模型之间不共享forward或projection。GPU logits→CPU稳定NumPy FP64 softmax，score=2(pFP+pFN)−1；atmost只选strict score>0且≤K，exact恰K真实点；tie依score降、absL升、flat升，ranktol0。K0不编辑；空P保持原域定义。选定真实索引升序2048块运行同一冻结701/E24 PR，actualGPU raw u及FP32 L+u定义after>0；域外mask原样。selected-PR对同query完整C参考raw u/combined绝对1e-4、rtol0及strict sign exact；不兼容是数值资格失败，不能换baseline/容差或算科学STOP。

4. **旧捕获是否足够。** 公开 shared_actual_image 保存完整 native_L FP32[100,H,W]、e0[100,256]、union_flat_indices[U]及对应 F_unique[U,256]；union是全部100个C的空间并集，含K0 query的C。故在C/K完全不变且binding成立时，它们在数学/接口上足够给N17全部C点中心F/e/rawL，以及C外的L邻值；无需八邻F或全幅F256。strip_event_to_shared 显式逐位核原GPU observed F/e/L 后才去重，validate_shared_for_CPU 独立核C/K/union。但只存sampled L的点chunk不足以取任意八邻；旧N16 GPU logits也不能冒充新N17 logits。

   旧evaluate_all100_scout_selectors签名要求完整 F_original[256,H,W]。已捕获F_unique不能直接传入该签名，也不能补零伪造全幅F。必要adapter应提供 gather_F(flat)->FP32[N,256]：searchsorted union、exact索引覆盖检查，再读F_unique对应行；e用e0[q]，L/邻值读完整同query网格。原始GPU full-F渲染路径则可继续原gather；两种来源都须 ROOT 实际来源与输入逐位核验，不能由此方案宣称任一实际载体已可复用。

5. **来源、捕获和参考复用条件。** ROOT先闭合原4 GPU shard/manifest/完整CPU资格与真实bundle refs、original_GPU_binding、所有注册members/图ID/shape/自然query/原class-S0-order，并验证新训练Source/spec/report/checkpoints/CPUQ及新实验锁。内部member字符串不是physical三字段ref，原bundle字节和binding不得改。native/R已合格预测仅在真实同carrier/selector/teacher/冻结PR701-E24/18700人口/export binding核对后复用；验证前只记待绑定，不宣称复用成功。新增N17 capture保留原input-carrier binding，另绑定当前evaluation及两端点checkpoint/新kernel，保存实际GPU extra8、logits、selected索引和u/combined；CPU独立坐标/公式与实际GPU logits所定义的FP64选点exact分开核验。不得用近似CPU logits替换GPU排名。

6. **判定和费用不变。** 首组 Δ=AL−max(AC,R)，未舍入值、无阈值宽容；Δ≤0 STOP，0<Δ<1最多172，Δ≥1且AL>AC才172/173；两组后是否可173及最终mean规则仍完整沿N17锁。undefined主AP不造0、不借次要指标。新增邻域gather/输入传输/scout/F/PR/排序/RLE/捕获IO/来源hash均计实际成本；共享的full-C参考属于数值诊断成本，不可相减冒充独立部署FPS。资源/分段/Source/RAW/外层literal schema由ROOT随后按既定规则绑定，本稿不实现未知科学外层、不改变预算/门或追加实验。
