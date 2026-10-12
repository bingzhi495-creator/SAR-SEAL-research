# N13 冻结缓存关系复核及独立 CPU 资格协议修订

状态：新协议登记，尚未执行或取得资格。原 N13 CPU v01 实际退出 1 与固定 index3 诊断实际退出 1 保留，不改写原 SOURCE、容差或结果。本修订不改变 48 轮、三个种子、LOG_PR/GEO_PR 配对、采样、网络与优化器的学术实验配方。

## 原检查与问题范围

原 GPU 全 FIT 缓存已实际完成 741 图、1661 个匹配实例。CPU v01 要求整幅重建 logits 与 GPU logits 满足 rtol=atol=1e-4，在 index3 失败。有限诊断中 FP32 有 1 个像素违反该门，FP64 转回 FP32 有 3 个；该图三个 query 的符号及 abs(L)<=2 分带未变。结果仅限此图，不证明全体采样顺序或区域一致，也不能据此宣布方法失效、唯一原因或放宽容差。

## 新独立关系叶：强字节判据

使用已冻结的 matched M200、原 query ID、图像尺寸及保存的 GPU matched L，独立执行原 bilinear 渲染链：200→800，按有效 img_shape 裁切，crop[:,None]→原图。固定 FP32、align_corners=False、指定原 Torch 2.0.1+cu118 运行环境。

主要输入为 contiguous NCHW [1,100,200,200]，按原 query ID 放入保存的 matched M，其余通道为 dummy 零。未匹配原 M 没有保存；dummy 不是原全 100 query 数据重建，历史 full-M stride 没有直接证据。主要成功条件是全部 741 图、1661 query 的输出与冻结 matched L 的全部 FP32 字节一致，包含 signed zero；不使用数值容差。小 G 输入重放只作诊断，不能替代主要判据。任一差异则保存完整失败报告并退出 1，禁止自动重试或兜底。

关系叶不加载教师、标注文件或模型，不做 matching/AP，也不写缓存资格。报告通过还必须由 ROOT 核验原持有进程句柄、实际退出、回收、原出生实例消失和完整来源，之后才可写单独的关系资格记录。

## 新 CPU v2 缓存资格

CPU v2 的独立责任是：核验上述实际关系叶和 ROOT 资格链；直接使用冻结 GPU L 精确复算区域、K、点索引、标签和重复采样语义，核验保存 packet/carrier 与旧合格缓存关系；保留原 sampled F/e 等实践数值 1e-4 强门。CPU 全幅 L 重放及原 allclose 结果逐图记录为跨设备数值诊断，不作为 v2 资格门，也不得记作原 v01 门通过。

这是显式改版的资格责任分工，不能宣称 CPU 与 CUDA 的 full-L 1e-4 一致，也不验证重新运行 backbone 的数值。新 Q/schema、关系记录和训练入口明确标记 v2；原失败结果、原 v01 SOURCE 和历史前缀完整保留。后续训练仅在 v2 实际结束并经 ROOT 复核后准入。

## 执行及可证结论

全38正常 + 全6失败原 RAW 首先核验，再进行哈希、载荷读取和数值导入；正常关系叶完成后才扩展前缀。每叶 CPU 0/1、2线程、allocator≤4GiB、全 GPU 观察≤8GiB、RAM≥16GiB、1800秒，无自动重试。上传使用运行前独立源码副本，不读写正在运行的绑定原件。

源码审阅通过仅说明可供 ROOT 绑定实际运行，不能证明数值关系或任务效果。关系通过仅支持指定环境中的冻结 matched M→L 关系；缓存通过仅允许开展配对训练。效用、机制、创新、SOTA 与 TGRS 结论仍需公平完整训练、重复种子和正式外部证据。

## 静态依据与限定

[PyTorch v2.0.1 CUDA bilinear](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu) 的普通 NCHW 分支没有跨通道混合，但实际分支还取决于 layout。[MMDetection v3.3.0 Mask2Former](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/mask2former_head.py) 的 einsum 与 [PyTorch einsum](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/Linear.cpp)、[bmm](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/LinearAlgebra.cpp) 支持标准 eager dense NCHW 的条件推导，不能代替历史 layout 记录或全体实际字节检查。不能把 canonical100 称为已证明的批大小修复或此前 CPU 差值的唯一原因。

独立审阅：新 replay_gpu.py / ABI.md 一次静态关键门审阅 PASS；审阅者未读取 SCI、哈希载荷、导入数值库或执行实验。审阅范围包括 RAW-first、身份/来源闭包、全体 query 映射、两段原渲染、signed-zero 字节门、失败后无兜底及报告不自认证实际终止。
