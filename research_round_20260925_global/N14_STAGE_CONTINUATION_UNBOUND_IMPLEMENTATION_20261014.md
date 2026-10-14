# N14 独立stage续训草稿 v2

状态：**UNBOUND、未执行、未读取实际SCI/weights/cache/SOURCE/RAW/Q/live输出**。只从允许的普通v1/module文本准备新文件，静态AST语法检查通过；双agent静态审阅完成并修正owner语义连接和累积证据门。v1及所有实际冻结原件没有改动。ROOT需进一步review/adapt资格ABI、绑定实际父leaf与完整SOURCE，再决定实际执行。

入口：`N14_CONTEXT_MEMORY_PAIRED_STAGE_CONTINUATION_UNBOUND_DRAFT_20261014_02.py`。schema为`ROOT-N14-LOG-E24-memory-paired-stage-continuation-GPU-spec-v2`。只接受(first,last)=(9,16)或(17,24)，fixed_stage_epochs=24、optimizer_reset_both=False，原seed/module_seed/order_seed、训练数学、LOG支持、固定cache、scores、FIT741/1661、K、AdamW配置不变。

## 资格及新spec字段

除v1原字段外，新spec必须完整绑定：

- parent_N14_GPU_source / GPU_spec / GPU_report / GPU_admission / GPU_RAW。
- parent_N14_CPU_source / CPU_spec / CPU_admission / CPU_RAW / CPU_qualification。
- parent_N14_CPU_endpoint_numeric_protocol（父CPU的独立协议JSON ref）。

以上字段都使用`parent_N14_`前缀，GPU/CPU各source/spec/ad/RAW/Q及其完整来源closure都在新sources内。第九失败先通过RAW-only/pin检查，再校所有normal与原失败8及其pins；**任何SOURCE SHA之前必须完成所有实际终止门**。normal>=97，legacy pins=normal+8，独立failed9不混入旧pins，total pins=normal+9。原失败记录及identity/admission pins保持相同。

当前合同要求父GPU/CPU是父normal prefix之后连续追加的两个实际owner：parent_GPU_spec.normal=expected-2，parent_CPU_spec.normal=expected-1；正常RAW prefix及旧/新增失败记录不变。再调用原lineage.owner_ref_join，把GPU cuda_visible_devices='0'、CPU cuda_visible_devices=''分别连接到指定source/spec/ad/RAW，返回索引必须expected-2/expected-1。这里cuda字段是字符串，不能当bool GPU_empty字段替代。

父CPU Q的stage8 ABI按CPU稿作者提供的静态结构写入：schema `N14-stage8-training-endpoint-CPU-qualification-v1`，status `ROOT_N14_MEMORY_PAIRED_STAGE8_ENDPOINT_CPU_QUALIFIED_V1`；明确匹配training_GPU_source/spec/report/RAW/admission及双方checkpoint refs、数值/订单/trace证据。其terminal_self_certified=False并不代替实际CPU RAW0/held/reap/birthabsence和owner fullSOURCE门。Q.details是独立文件ref，也闭合。

**stage16 CPU Q须ROOT另行adapt后才能用于17—24。** 代码预期同式stage16 schema/status，完整26576 step/order/trace-prefix证据；不能直接拿只检查stage8的CPU源码或Q改个标题。预期动态键是`all_AdamW_parameter_steps_exact26576`、`complete_paired26576_trace_order_identity_online_means_exact`、`final_stage16_full_FIT_mean_BCE_CPU_GPU_compatible`、`private_sixteen741_orders_and_final_state_exact`。这是草稿ABI要求，不声称该资格程序已执行或已形成。

原GPU训练protocol与父CPU数值protocol分开。CPU_endpoint_numeric_protocol与CPU spec.endpoint_numeric_source是同一JSON ref；六keys为atol/rtol/locked_before_CPU_execution/locked_after_GPU_training/scope/bound_completed_GPU_report。它在父GPU实际完成后、父CPU attempt之前锁定，不能称在原GPU前预注册。本续训绑定相同ref并验证对应completed GPU report、有限正tol与两个锁定flag；新leaf仍沿用原GPU训练protocol。

## 状态恢复与预算

实际模型state只能来自同独立CPU Q绑定的父N14双checkpoint。旧E24 carrier/curve继续作为原始provenance和FIT cache资格；不调用旧E24 load_heads来重开优化，不把它写成实际resume state。

构造独立BASE/MEMORY对象后，CPU载入两arm实际state；保持原参数顺序（MEMORY root memory_keys/values在child base.*之前），严格检查完整FP32模型与AdamW moments、单group全覆盖、原lr/betas/eps/weight_decay与foreach/fused/capturable等选项。head先toGPU，再load optimizer_state；恢复前后所有saved值与options逐值相等，不reset。

每个step必须是有限real scalar tensor，取值精确等于1661*(first-1)，没有int截断。完成leaf后全部参数step精确1661*last；moment finite、shape、dtype及exp_avg_sq非负。checkpoint完整保存并逐树readback验证model/optimizer/private order/meta，数值状态可resume；实际下一leaf仍必须独立CPU资格和实际RAW/SOURCE闭合，GPU报告自身不能完成该资格。

用独立proof Generator从原order_seed复现全部父epoch的741顺序、hash及最终state，再将实际order Generator用set_state恢复，继续draw。history/orders/activity完整deepcopy前缀并累加；父present=1661*(first-1)，最终present=1661*last。每leaf13288 trace行，每行两arm，stage_update延续，不是26576行。

首leafinitial_state_digests、common_E24 proofs、original START与cold/behavior preflight证据只继承，并核对origin source/spec/ad绑定；不在已经训练的MEMORY上重做MEMORY==BASE或zero-output断言。v2父checkpoint的origin/trace_chain须与父report相同；父trace chain长度为parent_end/8，均闭合。新增resume_state_digests明确记录真实续训输入。

leaf只新增8*1661=13288更新/arm；累积更新是1661*last（stage16=26576，stage24=39864）。报告仍写“E24共同父模型后的新stage续训”，不叫原连续E40/E48。原stage24主端点不改变；FIT online/final BCE仅训练和数值端点证据，不能替代后续固定DEV AP或科研创新证据。

本次只形成普通草稿和说明，未导入Torch、未构造模型、未运行科学训练或数值重放。ROOT负责实际review/qualification/admission、冻结新SOURCE及发布范围内同步；此文件不报告已同步或已执行成功。
