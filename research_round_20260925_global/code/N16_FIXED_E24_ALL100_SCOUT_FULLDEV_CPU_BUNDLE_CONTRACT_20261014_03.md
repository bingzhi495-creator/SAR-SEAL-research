# N16 fullDEV CPU03：逐图真实 bundle 读取

状态：普通 UNBOUND，AST/微型 CPU 合成回归通过，STOP EDIT；没有执行真实评价或读取实际 SCI/SOURCE/RAW/SPEC/Q/数据/权重，不操作 Git。CPU01/02 与 adapter01 留存。新入口 CPU03、新 adapter02 只改变存储读取。

每图一个真实 `{path,bytes,sha256}` bundle ref，187图合计187个 physical capture refs。包精确字段为 schema/binding/dataset_index/image_id/members/member_registry/member_count/raw_tensor_payload_bytes，schema=`N16-actual-GPU-image-capture-bundle-v1`。member IDs 是包内字符串，绝不伪造虚拟文件ref，也不嵌入本文件自身SHA。strict registry 与实际成员逐项重推一致：kind/query_id/epoch/evalmode/control/stage/chunk_first/points/raw_tensor_payload_bytes。

CPU loader 每图只 torch.load 一个闭合 bundle；manifest 的bundle真ref、图身份、shared ID、全部 member_locators 与header/members/registry逐值核对。复用GPU04纯 `validate_CPU_image_bundle`、`lookup_CPU_bundle_member(...,member_use=set)` 与 `decode_CPU_query_core_bundle(...,member_use=set)`。所有query/core/chunk与原2048升序point/action接口严格绑定，最后 members==registry==实际used集合；不允许多余或缺失成员，合法共享引用可复用。

core存储schema=`N16-actual-GPU-query-core-and-bundle-member-IDs-v3`，仅ref改内部ID；decoder仍返回原冻结 proof_query 的9字段。实际 union F/e0/100L 保留一次，实际GPU scout logits定义严格FP64排名，实际完整mask little-packbits仍独立保存。双CPU PR公式 absolute1e-4/rtol0/strictsign、原Ω/commonC交集、18700 natural queries、12条原score/order/RLE与官方COCO12 readback全部不变。每图释放bundle再读取下一图；pickle/runtime/COCO、完整注册表和decoder临时gather另占RAM，仍受16GiB RSS/1800秒CPU门，无实测时长保证。

CPU SPEC schema/ROOT_FIELDS/数字锁沿用CPU02，额外Source绑定新adapter02与实际完成GPU04 source及全部187真bundle/manifest/predictions。GPU REPORT=`N16-fixed-E24-all100-scout-fullDEV-GPU-bundle-report-v2`，status=`N16_FIXED_E24_ALL100_SCOUT_GPU_BUNDLE_COMPLETE_PENDING_INDEPENDENT_CPU_PROOF`；manifest=`N16-GPU-all18700-image-bundle-capture-manifest-v2`。Q继续pending ROOT actual CPU0，不用自报qualified晋级。

当前普通前门仍normal110/failed13/pins123，完整继承CPU02 RAW-first13，不猜仍运行叶的未来终止结果。如果ROOT实际确认新失败14，须另获精确RAW合同并另存下一版本，CPU03当前13门不能直接套用。此存储性能修复不构成新方法、AP收益或学术创新。

合成测试文件：`tmp/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_BUNDLE_ADAPTER_SYNTHETIC_TEST_20261014_03.py`。只构造微型CPU张量与内存member IDs：全100/12策略、原逆序导出与分数、全部member覆盖、K0/C空或非空通过；虚拟ref、registry端点错、缺失member、跨端点chunk、padding坏位、orphan成员、共享F数值损坏及错误binding共8项拒绝。RLE为确定性stand-in，不冒称真实COCO、GPU provenance、模型训练或实际资格通过。原proof主体保持冻结。
