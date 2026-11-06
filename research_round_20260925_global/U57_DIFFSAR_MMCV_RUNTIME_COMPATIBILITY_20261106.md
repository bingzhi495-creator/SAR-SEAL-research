# DiffSARShipInst：CARAFE 2.1 接口与最小构建准入补核

2026-11-06。限定作者提交 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`、MMCV `v2.1.0`、PyTorch `v2.10.0` 的原始源码与官方文档。先读已有 CARAFE/mapper 执行缺口记录，本文仅补实际安装版本之后的具体结论。**只读静态核查，无 SSH、安装、科学导入、测试执行、GT/权重读取或新 AP。** DGX 的安装信息由 Root 提供，不视为本审阅者独立验证。

## 可以先使用已安装 MMCV 2.1.0，不应因主版本先降级

作者 FPN 的 CARAFEFunction 在第 687、718 行调用扩展；官方 MMCV 2.1.0 的 carafe.py 在第 142、173 行调用同名扩展。逐调用去除空白后的文本完全相同；MMCV 1.7.2 与 2.1.0 两次调用亦一致。向前为六个 Tensor 加三个命名整数参数，反向为九个 Tensor 加同样三个参数；MMCV 2.1.0 的 C++ pybind 第 515–524 行逐项承接。作者加载器仍使用同一 `mmcv._ext` 命名空间。因此这里没有由 1.x/2.x 本身造成的已发现参数接口冲突。

这只支持先检查当前安装，**不证明实际二进制来源、Torch/CUDA ABI、GB10 kernel 或完整 DiffSAR 模型可运行**。Root 新盘点确认存在 `_ext.so`、CUDA 13.0 NVCC 和 venv ninja；这些事实也不能替代实际前后向。

来源：[作者两次调用](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/fpn.py#L687)、[MMCV 2.1 Python 接口](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/carafe.py#L142)、[C++ 注册](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/csrc/pytorch/pybind.cpp#L515)、[1.7.2 接口](https://github.com/open-mmlab/mmcv/blob/v1.7.2/mmcv/ops/carafe.py#L141)。

## 有限真实算子检查应覆盖什么

作者实际 FPN 第 241 行构造 256 通道、放大两倍的 CARAFEPack；第 790–794 行默认核宽 5、组数 1、编码核宽 3、压缩通道 64。第 821–830 行在 pixel shuffle 后按组内核位置做 softmax；CARAFEFunction 本身不再归一化 mask。tiny 检查可缩小通道，但必须明确其并非完整 256 通道 FPN 或完整模型。

官方 CUDA 核对输出位置采用整数除法映射至输入中心，奇数窗口超出输入视为零；输入通道按等宽连续组分配 mask，mask 在输出网格取值。故显式 PyTorch 参考可先在输入网格做带零填充的 unfold，再仅沿两个空间轴重复至放大网格，与按组/核位置重排的 mask 相乘求和。不能对 mask 再做相同空间重复，也不能用双线性插值代替该聚合。

最小检查应在作者核宽/组数/倍率下，对两个以内非方形 FP32 小输入，以提前固定误差容限比较输出、输入特征梯度及 mask 梯度；同时要求非空、有限、完整反向和清理。若先对随机 mask 归一化，应使所比较的 mask 张量成为一致的 leaf，避免把 softmax 链式梯度与算子 mask 梯度混淆。Root 执行侧仍需来源、实际终态、资源与进程组核验；本核查不执行这些步骤。

来源：[作者 Pack 参数](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/fpn.py#L787)、[CUDA 输出邻域/通道映射](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/csrc/common/cuda/carafe_cuda_kernel.cuh#L125)、[直观同等公式](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/csrc/common/cuda/carafe_naive_cuda_kernel.cuh#L31)、[官方测试范围](https://github.com/open-mmlab/mmcv/blob/v2.1.0/tests/test_ops/test_carafe.py#L23)。官方测试仅作为范围参考，本次没有运行或下载其二进制 fixture。

## Detectron2 侧：先核三个构建准入，不预改数学代码

1. 作者 setup.py 第 66–70 行只有在 CUDA 可见且 CUDA_HOME 可用，或显式 FORCE_CUDA 时选 CUDAExtension。否则 `_C` 安装成功仍可能是 CPU-only；必须核其实际 CUDA 算子。Root 已找到真实 NVCC，非登录 PATH 为空的疑问不再应当扩大为“没有工具链”。
2. 如果隔离构建隐藏 GPU、同时强制 CUDA，PyTorch 2.10 默认架构推断可能得到空列表，须显式指定经实机确认的架构。该版本已列出 12.1/12.1a，不能宣称 GB10 架构静态不受支持；也不能在未测硬件/编译器前随意选其一。
3. PyTorch 构建器会检查 NVCC 存在及 CUDA 版本，CUDA 主版本与 cu130 不匹配时会报错。应显式使用真实 CUDA 13.0 开发工具链，并保留编译环境/日志；wheel 版本本身不是工具链证据。

一项应避免的误修复：作者 DCN 虽引用旧 THCAtomics 头，PyTorch 2.10 仍保留它到 ATen Atomic 的兼容转发，并保留 Half 原子加法包装。所核 DCN/ROIAlign/NMS 文件也没有定位到可证明必然失败的旧 Tensor API 调用。**本轮未发现已证实的 PyTorch 2.10 删除 API 阻断，不等于全库兼容性已经证明。** 只有真实构建错误出现后，才针对对应依赖接口做最小兼容修改；不替换 DCN、CARAFE 或数学实现后仍冒称原作者方法。

来源：[作者构建选择](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/setup.py#L66)、[PyTorch 2.10 架构推断](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/cpp_extension.py#L2502)、[工具链检查](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/cpp_extension.py#L500)、[兼容转发头](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/THC/THCAtomics.cuh)、[Half 原子包装](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/Atomic.cuh#L282)。编译器、架构和可见设备选择的原则另见 [PyTorch 2.10 扩展官方文档](https://docs.pytorch.org/docs/2.10/cpp_extension.html)。

## 最小后续顺序

先在 Root 已授权的独立有限间隙检查现有 MMCV 2.1 CARAFE 原扩展的真实输出/梯度，避免先降级或重装训练环境。通过后再在独立 checkout/环境构建作者 `_C`；明确 CUDA_HOME、按实机指定架构、限制编译并发并保存实际版本与日志。然后检查作者所用 modulated DCN、RoIAlign/NMS 的真实前后向，以及完整模型单批训练和原网格导出。即使这些均通过，统一数据注册、初始化来源、充分训练、共同 COCO 评价和最终 AP 仍需另行完成。

本文是可操作的静态补核，未授予运行资格，也未声称强对照已完成。


## 后续真实执行

[CARAFE两个固定FP32 CUDA样例](CARAFE_ACTUAL_OPERATOR_PROBE_20261106.md)现已完成并通过Root接收。静态签名一致已补上有限实际算子证据，完整作者D2扩展及模型仍未验证。
