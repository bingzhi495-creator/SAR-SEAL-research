# N15 独立 CPU same-GPU-input replay 草稿合同

状态：普通 UNBOUND / UNEXECUTED。完整入口只做AST静态检查；编写仅读获准普通tmp摘录和新GPU草稿，没有读取actual科学Source、State、Spec、Q、RAW、GT或权重，也未开展科学执行、Git发布。ROOT是唯一实际科学owner，负责终止审计、真实绑定、admission和单次准入。

入口 `N15_FIXED_ACTION_OPPORTUNITY_CPU_UNBOUND_DRAFT_20261014_01.py`，schema=`ROOT-N15-fixed-E24-action-opportunity-CPU-spec-v1`，明确 `--spec` 与 `--owner-execute`。主head固定原strong701 E24 LOG PR，40497参数/8个FP32 tensor；不加载N14 MEMORY、不替换E48、不开训练或AP。

先独立failed9 RAW/pin，再全部normal>=104和旧failed8的RAW/pins，通过held exit、ROOT reap、original-birth absent和identity/admission原pin之后，才首次SHA或科学模块导入。normal动态而非截断历史；legacy pins=normal+8，另独立failed9 total pins=normal+9。GPU diagnostic完整normal前缀比CPU少一；最新 `owner_ref_join(...,'0')` 必须连接完成的本次诊断GPU。历史carrier GPU0/CPUempty与CPU自己entry/cudaempty严格分开。运行中的本次CPU Q/RAW不作为输入。

复用GPU普通完整稿 `source_gate(path,spec,cpu=True)` 和 `validate_protocol(spec,closure)`。原锁JSON与内嵌EXPECTED_PROTOCOL逐字相等；预锁要求**绝对误差<=1e-4，rtol=0**，不能沿用旧N13/N14相对容差。CPU spec新增完成GPU真实refs：`diagnostic_GPU_source/spec/report/admission/RAW`；source作为共享gate模块。它们与report、START、capture manifest、187个真实capture、374 pair diagnostics、population、image_checks/trace及original E24 prediction refs全部闭合，完成GPU实际exit0原owner是CPU的新最后normal祖先。

CPU自己的 `CPU_admission_file`、`CPU_resource_fields` 使用ROOT实际JSON key paths核 affinity[0,1]、thread_limit2、deadline1800、automatic_retryfalse、cuda_visible_devices空；四线程环境变量与Torch两线程实际核验。Torch CUDA全程cold。common imported-files加独立psutil/Torch/Numpy源码或binary及实际stdlib Fraction闭合，不能调用GPU不存在的 imported_files ABI。

CPU新 `CPU_replay_budget` 恰8非负整数键：`maximum_image_capture_file_bytes`、`maximum_image_capture_tensor_payload_bytes`、`maximum_selector_tensor_bytes`、`required_available_RAM_bytes`、`maximum_process_RSS_bytes`、`auxiliary_output_budget_bytes`、`minimum_free_output_bytes`、`minimum_available_RAM_margin_bytes`。前两项为完成GPU manifest真实最大值；selector=400×原最大图pixel数；RAM margin>=1GiB，可用RAM>=16GiB且覆盖capture+selector+margin。每图load前、每2048点call及末尾核deadline/RSS；不声称采样RSS是完整峰值证书。

capture每图一真实wholepacket，members不是ref。CPU读取实际GPU union-F[U,256]、e0[1,100,256]和每pair的C_indices、native_L、C/Ω-only residual+combined；原full100 GPU L从既有immutable selector读回。CPU不渲染F/L、不跑native model。候选用CPU独立OpenCV17×17/constant0 dilation核Chebyshev8的C=Ω∪P∪dilate8(P)，P=L>0，Ω=abs(L)<=2，Pempty回Ω，K=|Ω|包含K0；C完全不由GT选择。shared union必须逐pair C并集精确相同，payload=`1032U+21ΣC+16ΣK+102400+Σceil(N/8)`。

两PR路径是原factory.forward residual与独立显式LN(F,eps1e-5)+clamp(L,-8,8)/8四层coarse reinjection公式；分别保留u和L+u，不能通过combined-L倒推u，也不能用factory内部Ω门控logits替代C外部ungated结果。C升序2048和原Ω升序2048分别重放；两公式各与实际GPU residual/combined绝对1e-4、strict>0 sign相同，二者相互及C∩Ω/Ω-only也必须符合该门。

原native RLE与selector sign逐字相等，Ω-only完整binary与原strongE24实际refined RLE完全相同，且native/refined TPFPFN符合锁定pair。GT只从该GPU capture的little-endian packbits读回并核padding roundtrip、原pixel数、非空和锚点；不会重新解码原GT annotation。count oracle与原独立array oracle逐branch核真实selected support/真实noop/TPFPFN与完整mask，CPU独立pair JSON和所有Fraction分布必须等GPU保存的实际诊断。

人口只374既定native .50 pair；6未匹配GT单列UNEVALUABLE，无query、无虚构0IoU、不入374均值。主B=atmostK oracle_C−atmostK oracle_Ω；次A=oracle_Ω−actualapplyΩ不决定GO。GT替换oracle仍同C、同atmostK；exactC只能用真实noop填充，不能伪造zero动作。GO三门全部用Fraction：ΣB>=374/200、至少20pair B>=1/200、C中oracleIoU>=9/10 pair数比Ω增加>=4。

成功Q/details绑定完整实际report/captures/资格闭包；写出bytes计算stdout SHA，不读运行时自身Q。Q始终 `current_owned_CPU_terminal_self_certified=False`，ROOT须实际CPU exit0/reap/birth absent后再独立资格。任何数值/sign差异输出 `NUMERICAL_QUALIFICATION_FAILURE` 并退出1，其他绑定/执行问题另列资格失败；不当作scientific STOP、不放宽门、不自动重试，也不抹去继承failed9。本次新失败RAW由ROOT另保留。

成功仅说明本批真实GPU输入的计算读回资格以及原privileged条件IoU诊断一致。未资格CPU renderer/fullAP、训练、selector可学性、部署、TEST/cross-data、创新/TGRS或方法总体优劣；合成数学测试不构成actual科学资格。
