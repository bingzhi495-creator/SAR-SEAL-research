# N16 最小 all100 scout 部署评价主体（UNBOUND）

本文件只实现计算主体，不是完整实际科学 runner，未提供或注册科学资格。ROOT 负责实际来源闭合、资格/资源/终止门、训练状态冻结、原 renderer、native S0/class/query order、真实捕获文件、CPU 独立 replay、导出和官方 COCO。没有 GT/target/文件/权重加载接口，不改冻结 helper。

接口 `evaluate_all100_scout_selectors(torch,np,ndimage,helper,F_original,e0,native_L,utility_scout,error_scout,frozen_PR,*,evalmode='atmost',point_callback=None,synthetic_only=False)` 返回自然 query0…99 的生成器；每次仅返回一个 query 的独立 CPU 结果。实际输入是同设备、已实际渲染的 FP32 CUDA `F_original[256,H,W]`、`e0[100,256]`、`native_L[100,H,W]`；三个模型必须已 eval/frozen。调用者不可收集全部 query 输出后再声称本接口保持小内存。

每 query 固定 `P=L>0`、`Omega=abs(L)<=2`、`C=Omega|P|binary_dilation(P,ones17x17,iterations1,border0)`；K=真实 Omega 点数。K0 完整保留 native，所有前向与排序次数为0。C/选中点严格升序，所有模型2048分块。scout 输出实际 FP32 logits；CPU NumPy FP64 使用冻结 helper 评分，按 score降、absL升、flat升排序，无 ranking tolerance。主 atmost 仅选择真实 C 中 score>0 的至多K点；exact 单独调用只返回两 scout 的补充结果，含真实零/负分点，不虚构 noop。

主 UTILITY/ERROR 先各自计算全 C scout，再只对升序选中点调用原 PR。独立 common full-C 原 PR 输出实际原设备 FP32 residual 和原设备 `L+u`，用于数值参考；选中点 residual/combined 必须绝对误差<=1e-4（rtol0）且严格 `>0` sign 完全相同。失败抛 `NumericQualificationFailure`，附 details；不静默 fallback、不放宽阈值。最终 mask 从 native 复制，选中点直接取实际原设备 combined 的 sign，保持外支撑 native 完全相同，不用 FP64/CPU 加法重造实际动作。

DENSE_ERROR_FLIP 独立再跑 error scout，以 common-C 实际 Boolean `(P!=A)` 乘 `2p_error-1` 排序；MAGNET 使用冻结 helper 的未平滑、稳定二类概率式。两强控选择后仍取原 PR rawu/combined，不把评分当新 correction。它们在该诊断主体复用 common full-C 输入，因此各自 standalone 的必要全C PR成本不能记为0；输出明确记录 required calls 和 shared reference。此程序不共享两 scout 或两个 error 使用之间的 projection。

逐 control cost 记录实际 scout/selected-PR/gather/score/sorting/callback 调用、点数与耗时；common full-C PR另列。shared diagnostic time 只说明本次共享诊断执行；不能通过减法推导 cheap 部署速度或声称外部作者实现等成本。standalone 部署总耗时须另按明确范围实际测量。

可选 point_callback 收取 private CPU 实际点块：query/control/mode/stage、升序 flat indices、sampled F、query e0、native L、scout logits 或原 PR residual/combined；无虚拟 file ref。ROOT 负责保存真实整包与来源绑定，捕获/回调耗时单列。per-query 返回 native/C/Omega/K、每 control 的 Boolean mask/selected indices/selected native L/rawu/combined/FP64 C scores，供 ROOT export。

仍需 future actual qualification：训练混合 query/重复采样2048块与部署全C/选中点批次的相同实际输入 residual/combined absolute1e-4、strict sign/utility label exact；真实 CPU 独立 scout 公式及 canonical FP64 selected indices 必须 exact。局部 selected/fullC 门与合成测试均不能替代此门或证明 AP、效率、创新、CPU renderer 等价。

验证：独立 CPU-only synthetic tests 8项通过，覆盖正/负K0、empty C、非正分 atmost 与真实 exact 支撑、2048+2、真实noop、稳定饱和 MagNet、abs误差失败及微小误差仍 sign失败、GT-free接口/冻结模式/非有限输入。模型 stub/人工张量只验证流程；未读 actual SCI/GT/cache/weights/Source/Spec/Q/RAW，未科学执行或操作 Git。
