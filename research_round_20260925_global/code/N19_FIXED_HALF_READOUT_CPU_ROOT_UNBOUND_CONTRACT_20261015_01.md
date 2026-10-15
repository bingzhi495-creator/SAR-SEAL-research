# N19 固定r=.5方向读出：CPU全187缓存动作探针（UNBOUND）

STOP EDIT。仅普通源码/合同、AST对照与一项stdlib元数据fixture；未读实际科学载荷、未执行模型/COCO或Git。Root独立冻结并单次原owner执行。旧N16/N17/N18 Source、结果及STOP不改；本轮不训练质量头，也不把max-any-GT IoU冒充assigned-GT质量。

入口 `N19_FIXED_HALF_READOUT_CPU_ROOT_UNBOUND_DRAFT_20261015_01.py --spec ABSOLUTE_SPEC --owner-execute`。own schema `ROOT-N19-fixedDEV-full187-fixed-half-readout-CPU-spec-v1`；normal/expected真实完整前缀>=137，当前21失败、pins=normal+21。30个typed21前门函数/类AST exact N18，仅换own CPU_SCHEMA。CPUempty/CPU01/线程2、RSS16GiB、最低可用RAM16GiB、1800秒、allocator4096/wholeGPU8192、noRetry保留。

Root从N18 spec克隆，更新own entry/phase/ad/output/RAW/pins/pureprefix。新增exact字段集合：`prior_N18_root_result_ref,expected_prior_N18_root_result,scoring_body_source,n19_locked_protocol,expected_n19_locked_protocol`。四个ref/expected锁字段取普通 `N19_EXPECTED_PROTOCOL_AND_PRIOR_N18_UNBOUND_CONTRACT_20261015.json` 对应值。原 `locked_protocol` 必须保留N16祖先ref，不能覆写成N19锁，以保持 inherited_source_gate与legacy_gate字面ABI；无derived Spec或monkeypatch。

prior N18 envelope完整逐型JSON相等，schema/status按Root实际合同，owner136 exact8 record复用completed_record核actualCPUempty0、原Source/spec/ad/phase/bootstrap/完整前缀及原pending Q；其summary、details、priorN17ref和original_actual须对应。N19锁全JSON exact；唯一DIR_HALF主、固定.5、strictpositive/atmost/ties、无搜索与原资源/范围保持。

scoring body为冻结3190B SHA `3a1a1883cadd2277eea7759d2b4d524a614366869d208a81dcedf18db1532b97`；API `score_and_select(logits,native_L_C,C_flat,K)`，输入原actualGPU FP32 help/harm/noop logits。内部NumPy FP64稳定softmax；SYMMETRIC/DIR_HALF/GLOBAL_HALF/REVERSED_HALF四输出中只后三项导出新预测。P0/P1严格由原FP32 L>0判定。score>0、预算至多K、score降/absL升/flat升，固定系数无参数搜索，不选最好control替主。

直接复用GPU_outer_source.legacy_gate得到已qualified原N16四段 legacy_rows/manifests/populations/legacy_root。每真实旧image bundle只载入一次，按自然query core ID读取E24-atmost UTILITY的C_scout_chunks.scout_logits和common_C_PR_chunks.GPU_combined，核实际成员registry、query/epoch/role/chunk/共享输入ID，均为原2048升序C。只检查/使用本轮所需成员，不调用完整decode、模型factory、head/scout/teacher forward或PR公式重放。

每query另重建SYMMETRIC：选点必须exact旧actual_selected_indices，native外S/common A内S所得mask必须exact保存packbits（little/padding0）并且原编码RLE逐值等于已qualified E24_UTILITY_atmost预测。绝不使用rank容差、静默fallback或换动作。K0时scout/common流必须空，S空并输出native；C可非空，不能伪造C的零logits或声称观测过动作。

DIR_HALF/GLOBAL_HALF/REVERSED_HALF只把各自S应用于保存的同一common-C实际A。逐图依原qids导出自然100，所有image/category/score/S0/query顺序及非segmentation字段来自原native预测，不计算新quality/S0。全187图/380GT/18700query；绝不只选374匹配群体。原N10 segmentation编码与原official.coco_metrics直接闭合复用。

只输出三份新18700 *_PREDICTIONS.json；每份独立调用一次原官方完整187 COCO，保存全部12 stats/inline COCO语义，不平均分段AP。旧SYM/ERROR/R的pred refs与stats12沿用qualified legacy_root，不重算其AP。187份 IMAGE_*_PROOF.json保存原bundle ref、qids、C/K、各S及消费成员ID、symmetric exact三证据；DETAILS.json收集refs/原baseline refs和统计。

固定主Δ为 `DIR_HALF_AP-max(SYMMETRIC_AP,ERROR_E24_atmost_AP,strong_R_AP,GLOBAL_HALF_AP,REVERSED_HALF_AP)`，原COCO stats[0]从比例转百分点。合法undefined -1保持，不得当负一百分点参与成功判断。Δ<=0 STOP当前固定half假设；>0仅探索性正信号，新的重复必须另行事前锁，不在本轮准入。对照不能替主。此比较仅对预锁对照集合，不自动表示全项目/外部最优。

Q.json pending status `N19_ALL18700_FIXED_HALF_CACHED_ACTION_PROBE_COMPLETE_PENDING_ROOT_ACTUAL0`，含三control_results/原baseline_results、details strictref、主预览、全scope/计数、Source/ad/锁绑定与21失败计数。qualified/ROOT_actual_CPU0_verified/current_owned_CPU_terminal_self_certified仍False。Root须原actual0+pure0及闭合输出核验后另行资格；预览不能live自证。

本轮是新支持上的缓存动作模拟，不是新selected PR batch实际执行：deployment_numerical_qualification/new_selected_PR_batch_qualified/standalone_performance_qualified均False。若有正信号，未来仍需真实新selected GPU PR与独立CPU资格、事前锁定新重复；不宣称新定理、期望IoU/AP最优、性能、创新或TGRS成功。最终全Source SHA、自身Source/spec/ad、RNG/库与资源核验不删。

一项stdlib tinyfixture覆盖137 owner/header、不误读继承cpu_shard_owner_index、DIR唯一主/不被较高control替代、undefined不称成功；旧入口RED、新入口GREEN。此fixture不验证actual logits/掩码/COCO或科学终止资格。另仅一次30前门AST保持对照和定向静态接口审阅。
