# DiffSARShipInst SSDD：03 单一 IAMR 共享输出边界诊断候选

日期：2026-11-07。状态：**03 边界诊断源码候选，未绑定、未执行**。本目录作者未 SSH、未导入科学库、未读真实图像/标注/主训练 SOURCE。原01、02代码和输出均冻结不改。02的一次性诊断预算已经消耗；03需要 Root 另立一个明确的新 hypothesis 与 max1 预锁，不能复用02预算，也不能自动执行。无论03结果如何，闭合并恢复主训练，不自动延伸为 per-head 或其他泛化探针。

## 唯一研究问题和既有证据

Root 提供的02实际证据：唯一失败scope为 `head.iamr.`，14个参数、591,880个梯度元素已经存在但全部为0；其余8个scope非零。02原工具、SSH、worker均退出1，Root `d2b7ad/0` 已闭合。01没有保存逐组统计，仍是细节未知的独立失败，不能用02数据追溯补写01。

03只问一个问题：原总损失对 IAMR 原始共享8通道输出 M 的梯度 `G_M = ∂sum(original_loss_dict.values())/∂M` 是 None、有限且全0，还是有限且非0？若它非有限，明确记为 NONFINITE。这里“非0”是精确元素计数，不用容差或把负数排除。

固定作者 `head.py:251` 仅一次 `mask_feat=self.iamr(mask_feat)`，`:284` 返回同一共享 mask_feat；`detector.py:565–570` 把这个共享对象交给主输出和全部辅助输出，而非6次调用IAMR。hook只装在原 `model.head.iamr` 整个模块的输出，返回 None，不替换/clone/detach/mutate Tensor；调用原 Tensor.retain_grad()，必须恰调用一次。只执行原 total.backward()，没有 autograd.grad、额外 backward、per-head 梯度或新的损失。

原 backward、synchronize 和资源 check 返回后，先保存 `IAMR_shared_output_gradient`，再进入02已有的逐组扫描/保存及完全相同的 all9非零断言。旧门槛不放宽，原失败不升级。该共享总梯度无法分解各 head 或各样本的贡献，可能含求和抵消；本诊断刻意不再增加探针。

| 观测 | 支持的最窄结论 | 不能直接推断 |
|---|---|---|
| NONE | 当前保留输出没有 materialize 梯度；结合hook身份/次数检查再判读路径 | 方法失效或某个非线性饱和 |
| FINITE_ALL_ZERO | 当前原总损失在共享输出边界的总梯度为0 | 某一head独自梯度为0、sigmoid已饱和、或具体下游模块有错 |
| FINITE_NONZERO且IAMR参数仍全0 | 总梯度已到达共享边界，需另研究IAMR内部Jacobian/归约及其零空间 | 已确认具体ReLU/初始化原因 |
| NONFINITE | 边界梯度非有限，保留为运行诊断异常 | 既有方法总体无效 |

`IAMR_shared_output_gradient` 精确字段含 boundary、objective、hook_calls、return_policy、original_output_retained、output_shape/dtype/requires_grad、gradient_present/shape/dtype、finite、max_abs、nonzero_elements、classification。NONE时 finite/max_abs/count均null；NONFINITE时 finite=false、max_abs=null，避免把NaN/Inf写成合法数值。记录标志 `IAMR_shared_output_observed_before_scope_assertion=true`。不因这些新增分类另设或放宽原参数梯度通过门槛。

run_full 的 finally 移除hook句柄并释放持有的原输出强引用；即使remove抛错也释放引用。`IAMR_shared_boundary_cleanup` 记录 hook_installed、hook_calls、hook_removed、held_original_output_released。之后仍执行原模型/图释放、gc、CUDA同步、cuBLAS workspace清理及最终allocated/reserved0/0。若清理失败，不授资格。

风险：retain_grad额外保留一个共享8通道梯度，会增加少量 allocator/RSS 并影响时序；保留原8GiB/16GiB等限额，不据此承诺与02逐位一致。有限非零总梯度与全零参数梯度也可能与内部Jacobian零空间或参数归约抵消有关，需数据支持后另立研究，不能直接确定饱和。小标准库测试只核hook返回None、保留原对象身份、拒绝第二次调用、remove与释放引用，不执行Tensor计算。

官方2.10接口依据：[forward hook](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)、[retain_grad](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.retain_grad.html)。作者路径依据：[共享输出](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/head.py#L227)、[主/辅助共用mask_feat](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L565)。

接口 schema/operation/PRELOCK 与冻结02一致，spec仍11字段、24项library refs。Root须给不同run_id/输出目录/新一次性lease，例如既有前缀后追加 `03_BOUNDARY`；并在外部预锁明确本次唯一共享边界问题、max1、不得自动跟进、不复用02额度、旧失败保留。worker不自行创建授权或安排后续实验。下文沿用的数据/配置/资源合同仍适用。

本次沿用完整作者 R101/SCJE 在一张真实训练图像上的原路径：原训练 mapper → 完整模型 → 24个原损失 → 损失之和的一阶反向，只增加上述一个共享边界观察。没有优化器、参数更新、评估或 AP；不能证明论文复现完成、公平效用、优于其他方法或达到 TGRS 要求。所有参数采用作者构造函数的随机初始化，没有加载 checkpoint；单张随机初始化检查的损失数值不可与训练结果比较。

## 固定来源及数学范围

作者固定提交 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`。只静态读取 Root 已结束的原 ZIP；原 ZIP 为 2,971,918 字节，SHA256 `3b990faabb5ea14bb0d40b64cc11e5ae5af27eb5843241c405ffc4acc05bc33e`。Root 的独立兼容副本包含已单独记录的三个文件、五处最小启动修补，不能描述成未修改的原环境。本 worker 只绑定并核验 Root 提供的兼容收据和选定实际模块/二进制来源，不重新修改作者源码。

目标只允许 `configs/DiffSARShipInst-SSDD-train.yaml`，原文件 7,065 字节、SHA256 `48d6e3a67a19a737ce22d0659db02da0a878dbde9a5483e5c6ed240fbc87fddb`。保留 R101、SCJE_FPN、四阶段调制 DCN、500 proposals、6 heads、SAMPLE_STEP=1、单类、深监督及原损失权重。`SAMPLE_STEP=1` 是作者的采样配置值；本次训练前向并不执行推理 DDIM，不能称已检验一步推理。

SSDD 原 YAML 的格式是 **RGB**，并非 BGR。原 mapper 先随机水平翻转；原 crop 开启时按原随机分支直接 resize，或先 resize 到 `[400,500,600]` 中的尺寸并随机裁剪，再最终 `ResizeShortestEdge(512,max512)`。最终保持比例，最长边 512，不强制拉伸为正方形。图像保持原 mapper 的 CHW uint8，归一化由作者模型完成。YAML 的 `MASK_FORMAT=polygon` 不改变 mapper 的硬编码 `mask_format="bitmask"`；必须得到 `Instances/Boxes/BitMasks`。另一个 HRSID YAML 使用 RGB/1024，不能混作本次配置。

作者根 `train_net.py:240–243` 的真实配置入口包含 EMA 初始化，不能只调用 add_Diff。Root 实际早期 merge 记录 `170848/1` 因缺失旧 `MODEL.BIFPN_1p_5p` 节点失败，原失败必须保留。本候选显式调用：`get_cfg()` → `add_DiffSARShipInst_config(cfg)` → 作者 `diffsarshipinst.util.model_ema.add_model_ema_configs(cfg)` → 独立 `register_inactive_bifpn_schema(cfg)` → `merge_from_file(原SSDD YAML)` → `verify_inactive_bifpn_schema(cfg)` → 原 mapper → `build_model(cfg)` → `model.train()` → `model([record])` → `sum(losses.values()).backward()`。

独立 adapter 来源于同轮兼容审阅的 `diffsar_config_compat.py`，只有原 YAML 的5个明确 typed leaves：FUSE_TYPE=`sum`、IN_FEATURES=`[res2,res3,res4,res5]`、NORM=`' '`、OUT_CHANNELS=256、REPEAT=1。已有同名节点拒绝覆盖，merge 后拒绝非 DiffSARShipInst/SCJE 路线、额外键或值/类型偏离。该节点在目标 SCJE 路线不参与数学，并未实现 BiFPN。无 `set_new_allowed` 或宽松 merge，原 YAML 不动。adapter 自身必须另行 source ref 绑定。EMA 原 YAML 为禁用，不构造 EMA 更新器。

该 adapter 独立冻结为 2,220 字节、SHA256 `31bb4935787510af70f96f098b60a6c9eb71472ad6f4760bf20bbaa01d250619`，worker 也强制这一来源。兼容审阅独立静态检查确认本候选的24损失键和梯度 scope 成立；其3项 stdlib schema 测试通过，不替代 Root 对真实 strict CfgNode merge 或完整模型的执行。

不使用 Trainer、Checkpointer、评估器或数据加载器。`build_model` 不自动加载权重。显式记录上述 schema 注册与 adapter 验证，不覆盖作者实际模型/训练配置。

参考原源码：[训练配置](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml)、[mapper](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/dataset_mapper.py)、[模型与构造](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py)、[损失与 BitMasks 接口](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/loss.py#L356)。

## Root 固定真实数据的最小合同

Root 创建**独立** sample.json 和独立原图副本，核定 FIT741 的原始顺序与非空真实目标，记录它们自己的字节数与 SHA256。不能从运行中绑定的主载荷复制、读取或哈希来准备这次检查。数据准备必须依照原资格链在允许的终态后进行。

sample.json 必须只有以下字段：

| 字段 | 精确内容 |
|---|---|
| schema | `Root-DiffSARShipInst-SSDD-single-FIT-sample-v1` |
| dataset / partition / fit_image_count | `pseg_ssdd` / `FIT` / 整数 `741` |
| selection_rule | `FIRST_NONEMPTY_IN_ROOT_CANONICAL_FIT_ORDER` |
| root_verified_fit_membership | `true`，是 Root 的来源核验声明 |
| source_image_key | 原始图像标识，非空字符串 |
| fit_split_origin_ref / annotations_origin_ref | `{path,bytes,sha256}`，原 split/标注的来源元数据 |
| image_ref | 独立真实原图的 `{path,bytes,sha256}` |
| record | 一个原图对应的 Detectron2 dataset record，见下文 |

两个 origin refs **只作为来源元数据保存，不由 worker 打开/哈希**；worker 不读取全量 split、全量 GT 或重算 FIT 成员关系。其可信性来自 Root 外部核验，报告明确 `full_FIT_membership_independently_recomputed=false`。worker 实际读取/核验的是这份 sample.json 和这一张 image_ref，不能将此提升为整个数据集完整性结论。

`record` 必须只有 `file_name,image_id,height,width,annotations`。`file_name` 精确等于独立 image_ref.path；height/width 是原图尺寸，不是人为指定的模型输入尺寸；`image_id` 是稳定原标识。annotations 保留该图全部真实有效非 crowd 实例，不人为挑选目标以通过检查。每项至少有 `bbox,bbox_mode,category_id,segmentation,iscrowd`：category_id 整数0、bbox_mode 整数1（XYWH_ABS）、iscrowd 整数0、原始有限正面积 xywh。

segmentation 接受原真实 polygon 的非空坐标列表，或 JSON 可表示的 COCO compressed RLE `{size:[H,W],counts:"..."}`。原 mapper 的 transform 路径直接 decode RLE，因此不接受未经官方 COCO 转换的 uncompressed counts 列表；若原来源是 uncompressed RLE，Root 只可按官方 COCO 无损格式转换后记录转换，不可改变掩膜几何。不得用矩形框、预测结果或手造掩膜代替真实实例标注。

原随机增强可能导致目标全部为空。遇到该情形本次检查失败并保留记录，不能挑另一个 seed/目标自动重试。若原 mapper 过滤少数空实例，报告原标注数与保留实例数；至少须留下一个真实非空实例。

## Spec、lease 与启动接口

CLI：`worker.py --spec /canonical/PATH.json --spec-sha <SHA256>`。顶层仅 stdlib，模块可被 Root 惰性导入以读取常量/校验合同，不能在此阶段调用 main。

spec 必须精确包含 11 个字段：

```text
schema = Root-DiffSAR-SSDD-full-single-FIT-probe-spec-v1
operation = CUDA_DIFFSAR_SSDD_FULL_SINGLE_FIT_FP32_LOSS_BACKWARD_V01
run_id = DiffSAR-SSDD-full-single-FIT-v01-<Root唯一后缀>
source_refs = {worker: {path,bytes,sha256}, config_adapter: {path,bytes,sha256}}
library_sources = {worker.LIBRARY_NAMES 中各键: {path,bytes,sha256}}
config_ref = 原SSDD YAML ref
sample_ref = 独立sample.json ref
compatibility_receipt_ref = Root兼容副本来源/补丁收据 ref
root_lease_ref = Root lease ref
output_directory = 不存在的新绝对POSIX路径
prelock = worker.PRELOCK，逐项完全一致
```

`LIBRARY_MODULES` 是 worker 中固定的 23 个正常 import 路径（包含实际作者 model_ema）；额外 `tv_extension` 是正常加载的 Torchvision `_C.so`，不直接 `import torchvision._C`。Root 可在 GPU 隐藏的普通导入取实际 refs。D2 `_C` 必须是已实际构建的 20,097,880 字节、SHA256 `8b3323891f77ce4d2e9fdc330336cc600e821c0f77b90f398714f4d6dc6398af`。三个启动修补文件均在所选模块 refs 内。worker 逐项核验真实 import 的 __file__，并要求 D2、diffsarshipinst 和 YAML 属于同一兼容源码根；结束再核验全部选定 refs、sample 与原图。仅认证选定模块/二进制，不声称认证所有传递依赖。

lease schema 为 `Root-DiffSAR-SSDD-full-single-FIT-probe-lease-v1`，status=`ROOT_GRANTED`；operation/run_id/output_directory/prelock 必须与 spec 连接；worker_ref、sample_ref、config_ref、config_adapter_ref 必须分别匹配 spec 中对应 ref。还必须有 boot_id、single_attempt=true、main_scientific_processes_reaped=true、exclusive_project_scientific_leaf=true、whole_device_exclusive=false。lease 不是自授资格，Root 必须先实际确认主科学组已终止、资格验收完成，再独占登记一次执行。

保持专用 shared-base venv 的原 Python 可执行路径，不能 resolve symlink 后换成 base Python。Root 父监督给 child `PYTHONPATH=Path(bound_detectron2_init).parent.parent`、`PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1`、CUDA_VISIBLE_DEVICES=`0`。shared-base .pth 复用的事实必须保留，不称完全隔离环境。

## 资源、报告与资格边界

Root 已选择本次新独立 profile：CPU0/1，Torch/OMP/MKL/OpenBLAS 2 线程，CUDA allocator **8 GiB**、group RSS **16 GiB**，启动可用 RAM 至少48 GiB，运行 floor16 GiB，外部硬 deadline1800秒，单次、无重试。这不是公平速度/效率比较；报告本次成本。主训练原资源限额不随之改变。

worker 自查亲和性、RAM、自己的 peak RSS/时间；Root 外部父监督必须沿已闭合流程采样**整个拥有进程组**、记录真实 childbirth、设置 PDEATHSIG、超时终止并回收，最终外核同 boot、原 birth 消失、剩余组为空、实际原 SSH/工具终态。worker 自报退出0不授科学资格，也不自动恢复主训练；Root 做完外部终态验收后立即接续主训练。

固定平台为 Linux aarch64 / Python3.12.14 / Torch2.10.0+cu130 / Torchvision0.25.0+cu130。FP32 指训练参数与一阶梯度，AMP/TF32 关闭；不调用 model.float()，保留作者刻意建立的 float64 扩散系数 buffers。seed 固定40244023，调用顺序记录为 config 后 seed → 原 mapper → build_model → forward；不强制新的 deterministic 算法，不保证跨硬件逐位一致。

输出 `OUTPUT/REPORT.json`，schema `DiffSAR-SSDD-full-single-FIT-probe-report-v1`。候选成功状态为 `FULL_AUTHOR_SSDD_SINGLE_REAL_FIT_LOSS_BACKWARD_PASSED_PENDING_ROOT`，worker_exit_code=0、qualified=false 始终保留。必须有原始最终及5组辅助的 loss_ce/loss_bbox/loss_fiou/loss_masks 共24项，全部有限/可微，主 loss_masks>0，真实目标非空；反向后所有**已有**梯度有限，bottom_up/backbone、mask tower 与6个动态 head 各有非零梯度。未产生梯度的参数名字另列；作者源码可能声明但未调用某些模块，不能据此直接认定实现或方法失效。

本检查只新增上述单一不替换输出的forward hook，不修改作者 forward 源码、不替换匹配器/动态 mask、不改变损失。清理先移除hook并释放持有的原输出及函数局部模型/图引用，gc/synchronize，再按固定 Torch2.10 官方 API `_cuda_clearCublasWorkspaces()` 清理持久 workspace，最后 empty_cache；要求最终 allocated/reserved 均0、峰值≤8GiB。该清理仅发生在数值阶段之后。源码依据：[Torch2.10 CUDA 注释](https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/cuda.rst)、[绑定](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/cuda/Module.cpp)。

任何异常保留 `FAILED_PRESERVED_NO_RETRY` 和 traceback。OOM 只记 `RESOURCE_OOM_NOT_METHOD_INVALIDITY`，其他首错只记未解决执行错误；均不解释为方法无效。无静默降低分辨率/候选数/heads、AMP fallback、容差放宽或自动重试。AP=false、optimizer_steps=0、formal_training_credit=0、new_training=false、TGRS_success=false。

本目录的 stdlib 测试只检查样本元数据合同、失败前保存观察的顺序以及新增hook的原输出身份/单调用/清理，虚构占位元数据不进入实际运行。不导入科学库，不证明模型可构造、真实数据可解析、CUDA 前向/反向可通过、真实零梯度原因或资源足够。Root 若发现新的真实接口首错，应保留原候选和失败证据，另建修订，不能修改已经绑定的源。
