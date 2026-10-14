# N17 配对 ERROR 训练：最小外层接线 ABI

**普通静态报告；未执行、未读 live CPU119/科学载荷，无 Git。** 仅核文末两份发布代码。只有新缓存经 ROOT 核实际0及完整资格，才激活 GPU owner120、normal120/failed17/pins137；失败 partial 不用。

| 必要改接 | 已发布旧接口/硬编码与 N17 要求 |
|---|---|
| 自己的门 | 外层L315的N16 schema/旧锁/失败12前门不能直接用于N17。新门须RAW120+17→Source、独立N17 schema/锁、新entry/spec/phase/GPU_admission_file及CUDA0 own_gate；旧锁仅祖先。CPU final02/CPU119门也不能直接充当GPU120门。 |
| 两层缓存资格 | 外层L442按normal−2/−1推旧cache owner，会误要求118/119；应显式105 GPU0/106 CPUempty及旧ROOTcache/CPUQ/ref/前缀。另CPUempty join新119，闭合其source/spec/ad/RAW、Q/manifest/ROOT资格；旧ROOTcache不证明新八维。新Q键等待ROOT真实合同，不猜qualified。 |
| loader(index) | 沿外层L554–573原thin packet；再读合格neighbors8.npy FP32[4096,8]，对齐manifest packet/carrier/image/shape/point_order digest及samples query_ids/flat_indices。LOCAL用8列，CENTER用原samples.native_L的clip[-8,8]/8复制；原labels/importance/点序不变，不读fullL/teacher。 |
| 构造与初态 | 旧body L48 factory两参、硬16547不可照用；新调用N17 make_scout(torch,202610150171,closed_old_factory.make_scout)，两臂四张量/16803/独立存储，原513列匹配、新8列0，整171 init digest同锁。保留make_paired_state→cpu_snapshot→train_until的外层层次。 |
| arm/forward | body ARMS/LABEL_KEYS及L105/180/224硬UTILITY，改LOCAL_ERROR/CENTER_ERROR，**两臂error_labels**。训练L127/端点L203补各自extra8；共同一次loader、mean(weight×CE)、4096点一更新，不拆成两更新。utility_labels/delta可留FIT动作诊断，两臂score均ERROR。 |
| START/存档 | 外层L513/530/537/577–619硬161/旧arm/16547/N16 schemas必须改。新START/INITIAL/E8/E24/REPORT绑定原cache+新119Q/manifest/ROOT资格+N17锁/init/kernel/base_factory/body真ref；记录16803、171、每臂17784更新/72843264曝光，完整四参AdamW、order/history/RNG与CPU读回，不选best。 |
| 库/资源 | 沿外层L503–527冷Torch2.0.1+cu118、CPU01/两线程/RAM16GiB、TF32/AMP关闭、cuDNN deterministic；Source/库门后GPU0，allocator4GiB/wholeGPU8GiB/1800秒。新增np.load需NumPy1.23.5及npy模块Source/ad记录；旧imported仅torch/functional/AdamW/LayerNorm/Linear。保留外部CPU/CUDA RNG、double-readback、allocated0及最终Source/spec/ad/环境门。 |

原 sampling/matching/ORDER_SEED、AdamW、24/E8配方保留。训练/FIT CE不等于DEV排名资格/AP；端点CPU及后续实际GPU输入排名/mask资格仍必要。首次绑定前锁新接口/真实refs，不改旧Source。

依据：[GPU外层03](../github_release_20261011/research_round_20260925_global/code/N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_03.py)、[训练body02](../github_release_20261011/research_round_20260925_global/code/N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py)。
