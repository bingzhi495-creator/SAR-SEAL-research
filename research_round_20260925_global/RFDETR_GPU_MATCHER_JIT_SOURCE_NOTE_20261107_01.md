# GPU matcher 的 JIT 与子进程补录

2026-11-07。仅静态核查普通固定 RF-DETR 1.12.0 源码、官方 PyPI 小 wheel 和上游 Triton 源码。没有安装或导入 Torch、torch_linear_assignment、Triton，没有编译、运行模型或访问 SSH/Ops/数据。此补录不改已冻结的 CPU 探针或控制件。

**当前 microbatch1 的分割基线，按固定 RF 源码应使用 CPU SciPy 分配。** 不能因为安装 torch-hungarian 就断言本次 matcher 必须 JIT。若后续进入合格的 GPU 分配分支，compile=False 又不足以关闭依赖自身的 Triton JIT。新 GPU 执行应按真实路由与编译边界单独取得审计资格。

## 已核包字节和分发名称

从官方 PyPI 0.1.0rc0 JSON 取得并在内存中核验 wheel，16,894 字节，SHA256 为 1823ae4d37c870b49e1a0fa460dc9a39649f015210cf5bdb7299507d82cccdeb。只用标准库 ZIP 解读，未安装。distribution 名为 torch-hungarian，导入模块名为 torch_linear_assignment。metadata 声明 torch>=1.12.0、scipy>=1.6.0，以及 Linux 条件下的 triton。文档中另一个安装名不能替代这里实际核验的 distribution 身份。 [官方版本元数据](https://pypi.org/pypi/torch-hungarian/0.1.0rc0/json) · [核验的 wheel](https://files.pythonhosted.org/packages/1e/20/2fadf1e9664179f5fb5845da1b359187fba8526c282a36e42d2a1d4661db/torch_hungarian-0.1.0rc0-py3-none-any.whl)

wheel 中 assignment.py 为 163 行，SHA256 324798da4199356ed13cb78f040d3b45e7f04ff95605cceccde197cc9884365d。_triton.py 为 309 行，SHA256 d043d0a48efddcb92af66b49b5ca03a951e37f89bacc23acc77b29e9d5a91e53。以下函数与行号以该 wheel 为准，作者仓库当前分支不代替已发布包字节。

## RF 路由与 compile=False

固定 RF 提交 639eac4555ceb89ef399ccda3fde716bbcfe34cb 的 matcher.py 第 1130 行只是构造 matcher。真正路由在 _compact_path_applicable、_compact_mask_path_applicable、_match_many 和 forward。检测 compact 需要即时 batch>1 且无 mask，mask hybrid 同样显式要求 batch>1，还要求 CUDA、无 keypoints/固定行 padding，并达到避免跨图计算的 728,000 元素门槛。当前 batch1 不满足这些条件，累积4不会改变它。full 路径第 1124–1127 行把最终对角成本 .cpu()，再由 _assign_compact_cost_matrix 使用 SciPy linear_sum_assignment。 [实际条件](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/matcher.py#L355-L460) · [full 路径](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/matcher.py#L1084-L1127)

CUDA compact 路由若成立，才进入 _assignment.assign_many_bucketed → _solve_to_indices → torch_linear_assignment.batch_linear_assignment。RF 的编译开关另外控制 L1 成本的 torch.compile，该开关没有传给分配依赖。当前 compile=False 会关闭该 L1 编译请求，但未来 GPU solver 的 @triton.jit 仍独立生效。 [分配调用](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/_assignment.py#L46-L91) · [L1 编译入口](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/matcher.py#L184-L229)

wheel 的 assignment.py 第 64–73、112–134 行进一步要求输入 is_cuda、Linux、Torch>=2.4、CUDA 可用且 torch.version.cuda 非空、输入设备 compute capability>=8.0，并能导入 Triton 模块。不满足则警告一次，移到 CPU SciPy 计算，再把 assignment 送回输入设备。它没有供 RF compile 参数控制的公共 backend 开关。

_triton.py 第 53 行声明 @triton.jit。batch_linear_assignment 第 299–309 行处理 dtype、空形状和矩阵转置，非空输入才进入 _solve。_solve 第 265–279 行启动 kernel，NUM_ROWS、NUM_COLUMNS、BLOCK_N、IS_FP64 为专门化条件，num_warps=1。因此将来改变目标数或 dtype 可能产生新专门化。没有实际缓存和路由证据时，不能认定某次调用一定重新编译。

## Triton 3.6.0 中真实的子进程源

官方 v3.6.0 对应提交 7c56a5e40f7fd928dfd5c72902d5def0097db73a。它只是本次查阅的候选版本，实际环境必须按 inventory 绑定真实安装字节。

| 阶段 | 已核源码与条件 |
|---|---|
| JIT kernel | JITFunction.run 在内存 kernel cache 未命中时进入 _do_compile，编译器还有磁盘缓存。首次进程调用、磁盘缓存命中和完全冷编译应分开记录。 [runtime/jit.py](https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/python/triton/runtime/jit.py#L635-L680) |
| PTX 工具探测 | NVIDIA compiler 的 get_ptxas_version 用 subprocess.check_output([ptxas路径, "--version"])。get_ptxas 在 arch>=100 时选择 ptxas_blackwell，否则选择 ptxas。实际路径由 Triton knobs/环境及安装包决定。 [compiler.py 第 31–40 行](https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/third_party/nvidia/backend/compiler.py#L31-L40) |
| PTX→cubin | make_cubin 组装 ptxas argv，包含目标架构、临时 PTX 和输出文件，subprocess.run(check=True) 执行。只有真正走到此编译阶段才产生该调用。 [compiler.py](https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/third_party/nvidia/backend/compiler.py#L420-L495) |
| 主机 C 扩展 | CudaUtils 和 CudaLauncher 调用 compile_module_from_src 构建 cuda_utils 与 __triton_launcher。缓存有效则可直接加载，缓存缺失或加载失败则 _build。_build 使用 CC，未设时优先 gcc，再选 clang，最后 subprocess.check_call(cc_cmd)。该编译器还可能产生自己的后代工具进程，具体须实际观察。 [driver.py](https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/third_party/nvidia/backend/driver.py#L54-L61) · [launcher](https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/third_party/nvidia/backend/driver.py#L620-L635) · [构建与缓存](https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/python/triton/runtime/build.py#L18-L90) |
| 平台探测 | build.platform_key 会调用 platform.architecture。CPython 3.12.14 的该函数在 Unix 可经 _syscmd_file 调用 ["file","-b",解释器真实路径]，即使未进入 C 编译，也可能出现这个子进程。失败可被标准库捕获，但审计仍须保留事件。 [Triton 调用点](https://github.com/triton-lang/triton/blob/7c56a5e40f7fd928dfd5c72902d5def0097db73a/python/triton/runtime/build.py#L56-L75) · [CPython 原实现](https://github.com/python/cpython/blob/v3.12.14/Lib/platform.py#L602-L669) |

这条正常链中已核的外部程序是 ptxas、主机 C 编译器及可能的 file。源码没有要求必经 nvcc。自定义 CC、Triton build 实现或工具环境可改变链路，因此不能仅按进程名推断合法性，也不能由 wheel 很小推断运行时无编译。

## Root 的最小审计范围

当前 batch1 配方应保留原 CPU solver，并在真实回执记录 batch/targets/compile 条件及观察到的分配路线。它本来就是该配方的官方路径，无须为“使用 GPU matcher”改数据或批大小。没有触发 solver JIT 也不表示未来更大 batch 已验证。

未来若资格明确包含 GPU 分配，应另绑定包源、CUDA/Triton 版本、实际工具真实路径及字节、必要 argv 形态、CC/Triton knobs、受控临时和缓存目录。审计只准入已核调用链的具体编译或平台探测，记录其父子关系、退出码和冷/暖缓存状态；其余进程请求保留为拒绝。原 worker 的资源身份和原生工具进程后代均需要在同一资源边界中观测和回收，不能只凭 Python audit hook 宣称覆盖本机编译器所有后代行为。网络限制继续覆盖实际科学进程范围。

不应直接重用旧 CPU 探针的“拒绝全部子进程”结论来评价 GPU 算法。合法但未准入的编译被拒绝时，应记为执行/审计资格未完成，不能写成方法无效。此处没有给任何现成 audit/control 文件放宽权限。

未来较大 batch 若改为显式 CPU solver，应另命名后端适配并固定相同的成本、dtype、mask 采样、query groups 与返回索引语义。最小总成本和并列最优解的具体索引是不同的可比性问题，需固定相同实际成本作求解质量核对，不能承诺逐次训练轨迹相同。CPU 传输和求解耗时也属于实际训练成本。不要通过注册 matcher forward hook 来“仅观测”后端，因为 criterion 第 1373–1389 行会因 forward hooks 关闭其多层 fast path，可能改变测量对象。 [hook 路由条件](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/criterion.py#L1373-L1389)
