# N16 独立 CPU 部署选点与同 GPU 点输入 PR 核验主体（UNBOUND）

普通研发稿，无实际科学资格、训练或 AP 结果。ROOT 负责实际 GPU 已结束的 report/capture manifest、全部真实 capture 文件及原 native/selector/source/admission/RAW/checkpoint 资格闭合，CPU empty-CUDA/资源门、Source 首次读取顺序、实际输出绑定与全 DEV 范围。本主体不读文件、权重、GT、Source/Spec/Q/RAW，不渲染 CPU F，不调用 CPU scout。

接口为 `proof_query(torch,np,cv2,helper,packet,formulas)` 和 `proof_all100_queries(...,query_captures,formulas,evalmode='atmost',query_callback=None)`。后者逐 query 消费且要求自然0…99完整、同 mode、无缺漏/重复/额外记录；捕获解码和实际身份由 ROOT 外层处理，必须逐图覆盖全部既定图像。输出仅为传入 capture 的机械核验，不能单独形成实际 Q。

ROOT 提供两条独立原 PR 路径的闭合 callable 字典，键严格为 `original_head`、`independent_PR`，各 `callable(F_CPU_FP32[N,256],e_CPU_FP32[N,256],L_CPU_FP32[N]) -> (u_CPU_FP32[N],after_CPU_FP32[N])`。after 必须等于该路径自己的 FP32 `L+u`；不是直接复制 GPU after。ROOT 包裹实际原 `independent_PR(torch,state,F,L)->(u,L+u)` ABI，e 可按原数学不使用。实际40497/8state、factory/manual来源独立及冻结证明由 ROOT 负责，函数名称不证明这些事实。

新的纯 query 核心 packet 严格键：`schema,query_id,evalmode,native_L,e0,K,C_flat_indices,common_C_PR_chunks,controls`。schema=`N16-all100-scout-selection-same-GPU-input-query-capture-v1`；query_id 为 Python int0…99；mode atmost/exact；native_L 为 CPU FP32[H,W]，e0 为 CPU FP32[256]；K 为真实 `abs(L)<=2` 点数；C_flat_indices 为 CPU INT64[|C|] 升序且无重复。物理真实 packet 的 binding/ref/identity/source等包络由 ROOT 在外层验证，不能混入纯 query 核心以代替验证。

`common_C_PR_chunks` 是完整 C 升序2048切块的 list；K0必须为空，不能用虚拟值补块。每 PR chunk 严格键为 `flat_indices,sampled_F,e0,native_L,GPU_raw_residual,GPU_combined`：分别 CPU INT64[n]、FP32[n,256]、FP32[256]、FP32[n]、FP32[n]、FP32[n]，都是实际原 GPU 执行的回读（名称 GPU 不表示解码后仍在 CUDA）。最后块1…2048，其余2048。ROOT 真实存储可共享实际保存 F 的物理 union及精确索引，解码出同值 CPU view；不得 CPU 重新渲染/近似重造 F。

atmost controls 严格为 UTILITY/ERROR/DENSE_ERROR_FLIP/MAGNET；exact 仅两 scout 补充。每 control 严格键：`C_scout_chunks,selected_PR_chunks,actual_selected_indices,actual_mask`。scout chunk 与 PR chunk 共用四输入键，但输出为 `scout_logits` CPU FP32[n,3]；UTILITY/ERROR/dense各有完整实际 C scout 回读，MagNet无 scout 块。main selected_PR_chunks 完整覆盖该 control 实际升序选中点，strong为空并直接用 common C 原实际动作。actual_selected_indices 是 CPU INT64[s]，actual_mask 是完整 CPU BOOL[H,W] 的实际 GPU-evaluation 输出，不是 CPU近似前向自行生成的目标。K0时所有块、选中索引为空且 mask 完全 native；C本身可非空。

CPU 独立使用 CV2 UINT8 centered17×17 dilation/constant0/iterations1，emptyP捷径，保留原 FP32 `L>0`/`absL<=2`；核 C/K exact。scout评分只用实际 GPU logits 经冻结 helper 的 NumPy FP64 stable softmax，denseflip只用实际 common GPU `(P!=A)` 和该 control 实际 GPU error logits，MagNet只用实际 GPU native L/rawu。CPU原公式近似输出永不用于排序。atmost只 score>0且至多K；exact含真实零/负分C点；score降/absL升/flat升无任何 rank tolerance，实际 selected indices 必须完全相同。

每 full-C/selected PR 块分别运行两 CPU 原公式，要求 residual/combined absolute<=1e-4、rtol0和严格 `after>0` sign exact。所有 scout/main选中点 F/e/L须与实际 common C同点 bitexact，不能换点输入。main GPU selected/fullC同点 u/combined另核 absolute1e-4及strictsign；失败无fallback。最终新mask用实际 GPU combined取 sign 重建，与完整 actual_mask exact，并单独核全部非选中点 native exact；两CPU公式的sign相同才能支持其完整mask同样exact。计算只拼接一维 u/combined/logits，不拼接整块256维F，CPU公式每次最多2048点。

实际训练混合query/重复采样PR与部署C批次输入/符号兼容门仍由 ROOT另核。此主体不能证明 CPU renderer等价、CPU scout近似logits的独立部署排名相同、GPU AP有效性、standalone速度、收敛或创新。尤其即便CPU/GPU scout logits allclose，也不能把CPU scout logits用于此actual-GPU-selection proof；近tie索引变化不代表实际GPU记录选择错误。

合成验证：8项CPU-only测试通过，覆盖四主control/两exact补充、near-tie中CPU近似logits会反转排序但实际GPU-logits proof仍通过、K0/emptyP/positiveP/真实noop、非正分atmost/真实exact、PR abs误差/严格sign失败、篡改F/e/C/K/选点/外支撑mask/NaN/缺块拒绝、2048+2、公式after契约、自然all100完整/顺序和无GT接口。所有fixture和formula均人工CPU stub，不是实际科学核验。
