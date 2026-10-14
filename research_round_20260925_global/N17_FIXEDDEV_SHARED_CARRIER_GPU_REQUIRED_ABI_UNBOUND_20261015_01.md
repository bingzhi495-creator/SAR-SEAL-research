# N17 fixedDEV：共享载体与GPU外层必要ABI（一页，UNBOUND）

2026-10-15；仅静态接口清单。只读许可普通源码/设计；未读live SCI/权重/SPEC/RAW，未运行SAR或Git。ROOT告知CPU121及完整训练已合格，本稿不自行复核actual文件；静态字段不授予载体资格。

1. **真实存储。** 旧N16每图由torch.save生成一个物理 `.pth`：schema=`N16-actual-GPU-image-capture-bundle-v1`，严格键为schema/binding/dataset_index/image_id/members/member_registry/member_count/raw_tensor_payload_bytes。物理三字段ref在manifest row的bundle及physical_capture_refs/capture_refs中；`i%03d/SHARED_ACTUAL_INPUTS.pth`只是members内部字符串，不是第二个文件。shared schema=`N16-actual-GPU-image-shared-inputs-v1`，字段dataset_index/image_id/identity/native_packet/selector/natural_query_ids/native_export_query_ids/union_flat_indices/F_unique/e0/native_L/C_indices/C_counts/K_counts/U/actual_GPU_F_captured/CPU_F_rerendered。native_L为CPU FP32[100,H,W]，e0[100,256]，F_unique[U,256]，union为CPU INT64[U]；共享member本身没有original_GPU_binding，原绑定是bundle.binding=manifest.capture_binding=report.capture_binding。

2. **只继承原成功四段。** [0,47)、[47,94)、[94,141)、[141,187)。ROOT从原N16完整合格结果定位四GPU report/manifest/原owner与其完整CPU资格，核这些精确refs及Source闭合；每row的dataset_index/image_id/identity/native_packet/selector/qids/bundle/shared_actual_image_inputs/member_locators/natural_queries100均与原行及父report绑定一致。核bundle严格键、registry/成员角色/大小、shared原身份、qids排列、全100 C/K与union exact覆盖。原完整资格不可由member的True标记或模板schema替代，也无需重做无关旧史审计。原bytes、binding、registry不改，不混不同shard。

3. **模型注入。** ROOT训练结果须为`ROOT_N17_PAIRED24_GPU_CPU_TRAINING_FULLY_QUALIFIED`、qualified/eligible_for_fixed_DEV_evaluation为True，并闭合GPU120和CPU121原actual0、双方完整Source/Q/details及lock/cache/init/factory/body/checkpoints/FIT_endpoints。从其同一E8/E24 checkpoint顶层models加载LOCAL_ERROR/CENTER_ERROR，完整四FP32张量16803、strict=True、冻结eval/无grad；不取旧payload['snapshot']，不要求训练后八列零/非零。make_scout(torch,seed171,base_factory)必须绑定正确旧factory；原冻结701/E24 PR独立绑定。模型参数在评价前后逐位不变。

4. **实际数值入口。** `evaluate_all100(torch,np,ndimage,helper,legacy_evalbody,neighbor_module,native_L,e0,union_flat_indices,F_unique,models,frozen_PR,point_callback=…,check_callback=…,synthetic_only=False)`。L/e/F移到同CUDA0 FP32，union保留CPU INT64；gather_F仅以严格升序union searchsorted并exact索引覆盖读F_unique，不伪造全幅F。ROOT实际验证旧shared资格、完整身份、all100 union及输入转移逐位一致；只保存sampled L的chunk不足以生成八邻。新scout四参输入/邻域padding、raw中心L/8沿锁，不引用DEV调整规则。

5. **捕获callback及格式差异。** event含query_id/control/scout_epoch/evalmode/stage/chunk_first/point_chunk2048/flat_indices/sampled_F/e0/native_L；scout另有extra8及actual_GPU_logits/scout_logits，PR另有GPU_raw_residual/GPU_combined。它们是实际device detached张量，outer须在回调内同步CPU clone/保存，不能延后借用；核输入F/e/L与旧shared逐位一致后方去重，新extra8也保存供独立CPU回放。旧member-ID/writer只接受UTILITY/ERROR，不能直接写N17 arms；新schema/registry/CPU解码合同须先锁定，原input bundle+member+binding与新evaluation/model binding分开保留。不得拿旧N16 logits当新模型输出；捕获预算计extra8、实际字段/别名与序列化，不照抄旧104C公式。

6. **完整评价。** 新body产生E24/E8×两arm×atmost/exact八profiles，主仅E24-atmost；四模型各自一次C forward，其同模型实际GPU logits共享两mode。原native/R predictions只在同teacher/PR/selector/图/query/class/S0/order/导出绑定后复用；生成新预测时按native_export_query_ids排列且仅替换segmentation。四段合并必须187图/380GT/18700自然query，各profile COCO12；selected索引由实际GPU logits的CPU FP64排名定义，CPU近似logits不替代。新共域/selected PR u与combined绝对1e-4/rtol0且after-sign exact，随后完整独立CPU。分项及共享诊断成本非可相减部署FPS；首组门/STOP仍按N17锁。ROOT确认真实载体后再冻结科学outer。

来源：普通N16 shard GPU的shared_actual_image/ImageBundleCaptureSink/lookup_CPU_bundle_member/report writer；N17 fixedDEV wiring、已STOP EDIT的evaluate_all100 body及CPU121普通Source。没有新增科学资格、AP或创新结论。


## Root实际元数据核查（2026-10-15，仍待owned GPU张量读取）

在CPU121实际0及纯RAW122前缀通过后，Root核旧合格N16四段report和capture manifest物理哈希及完整metadata绑定：187个唯一图像、380GT、18700自然query，范围[0,47)/[47,94)/[94,141)/[141,187)；每row bundle在原物理ref清单及当前Source闭包中，shared成员名与registry角色关联、原query导出置换和selector行一致。这是实际元数据证据；新的GPU原owned任务仍须严格加载bundle/shared并核实际张量和坐标，不能由此提前宣称新预测或AP。

固定PR工厂的实际SHA与旧资格链一致，Root已核forward：PR只计算LN(F)与截断L注入的三隐藏层及raw residual；e和49维patch只保留finite/shape/device/range接口检查，不进入PR计算。显式zeros49沿原评估，不引入新patch采样；动作仍用selected L+raw residual，不能替换为head中仅Omega门控的logits。此核查收敛了PR接口疑点，未更改科学配方。
