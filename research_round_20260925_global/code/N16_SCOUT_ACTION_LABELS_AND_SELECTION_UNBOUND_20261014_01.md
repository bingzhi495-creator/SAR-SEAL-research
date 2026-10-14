# N16 labels / scores / physical selection 普通合同

状态：UNBOUND、未登记、未训练。只读获准普通N15方法/ABI文档和纯count helper；本稿只执行stdlib/NumPy合成测试，不读actual SCI/State/Source/Spec/Q/RAW/GT/权重/缓存，不Git，不构成科学资格。ROOT另锁真实assignment、共同background监督扩展、采样/importance与资源门。

接口均无文件I/O、模型或训练：

- `utility_labels(native_P,G,fixed_after_A)` → 同shape NumPy INT64，0=help、1=harm、2=noop。help为原错误且实际PR纠正，harm为原正确且实际PR破坏，noop为A=P；未被PR改变的FP/FN仍是utility noop。
- `error_labels(native_P,G)` → 同shape INT64，0=FP、1=FN、2=correct。两个label接口要求真实NumPy BOOL、同shape、ndim>=1；1D空S合法，拒绝masked数组或隐式dtype转换，真实内部masks互斥覆盖，不修改输入。
- `softmax3(logits)`、`utility_score(logits)`、`error_score(logits)` 输入finite FP32/FP64 `[...,3]`，统一CPU NumPy **FP64**稳定softmax。utility=p_help−p_harm；error=2(p_FP+p_FN)−1。ROOT已决定实际GPU FP32 logits转CPU后用同helper；不能以GPU FP32 allclose替代排名/选点indices exact资格。
- `choose_candidates(native_L,C,score,K,mode='atmost')` → 新BOOL同shape，部署接口不接GT。主atmost仅选真实C内score>0、最多K；补充exact恰K真实C点，可含真实零/负分点，不凭空造noop。要求非bool整数0<=K<=|C|，统一(score降、absL升、flat index升)，score/零/tie均直接比较，无ranktolerance；K0/空C返回空selection。
- `apply_selected_residual(native_L,actual_residual,C,selected)` 保留调用者同一FP32/FP64算术，只在selected⊆C应用真实PR residual；外selected含外C逐位native不变。operand dtype/shape必须相同，不隐式cast，拒绝非有限combined。
- `dense_error_flip_score(actual_delta,error_logits)` → FP64 **排序分数** `δ_actualflip×(2p_error−1)`，δ必须是真BOOL `(native_P!=actualfixed_A)`，不是raw signed PR residual。无flip点分数0；选点仍应用原PR rawu，不能把score当新correction。需要全C PR产生δ，成本不能视同仅scout排序。
- `stable_sigmoid(values)` 为分段FP64 sigmoid；`magnet_score(native_L,correction_u)` 的u为原fixed PR raw residual，用冻结稳定等价式 `2σ(-abs(L))*abs(tanh((L+u)/2))`，等于二类公式 `(1−abs(2σ(L)−1))*abs(2σ(L+u)−1)`，无平滑。避免FP64正饱和sigmoid取消误差；中等数literal仅互核。这是MagNet式适配，非作者逐位复现、非IoU/AP最优理论；dense/fullC PR、额外排序/IO等成本不能宣称equal。

cache Torch桥接应由body调用两label接口：P=(native_L>0)、A=(actual_combined>0)，CPU BOOL转NumPy后生成INT64再转Torch。其 `delta` 若为BOOL应明确指actual flip `(P!=A)`，`error`为native error `(P!=G)`；raw signed PR residual另存，不混同。函数不根据原 `mask_weights=0` 推断空GT：ignore/原target validity与新的双方共同background监督扩展是caller需要独立核验、预锁并披露的语义。部署排序不得从GT labels生成score。

两label函数输出INT64用于CE，不自动保证原方法草案的25S缓存界。保存若压到uint8须显式核范围0..2和训练long转换；若保存两INT64须计16S bytes，再加raw PR、flip/error/importance等实际tensor payload。ROOT按真实字段重新锁资源。

TDD合成结果：10项behavior tests通过。覆盖8种二进制真值组合、手算softmax/dense actualflip排序/MagNet、large20/40/1000稳定数值、错误dtype/shape/NaN/masked数组、K0/空输入、score下一可表示数及极小正分无epsilon、外C逐位native与全四点16候选池×81分数组合的独立Python排序。共7776个合法K/模式选点核对。测试只验证这些纯函数，不能证明真实target assignment、scout数值或科研效用。
