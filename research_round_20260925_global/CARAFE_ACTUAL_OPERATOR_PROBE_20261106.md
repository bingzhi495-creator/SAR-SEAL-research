# CARAFE真实FP32 CUDA前后向：两个固定算子案例

2026-11-06。现有MMCV2.1.0扩展在DGX GB10上实际运行成功：两个固定非方/边界样例，kernel5/group1/scale2，输出与features、detached masks的两个一阶梯度均符合预锁容差。它清除了该算子的有限运行疑点，不代表完整DiffSARShipInst、Detectron2._C、256通道CARAFEPack、模型训练或AP已经通过。

作者固定FPN与MMCV2.1.0的前后向参数顺序及kwargs经原始源码核对一致。现有112037752字节扩展文件、CUDA13.0 nvcc和venv ninja均实际存在；此前nvcc仅未在非登录PATH中。未降级MMCV、改作者算子或修改base环境。依赖静态差异及官方来源见[兼容性补核](U57_DIFFSAR_MMCV_RUNTIME_COMPATIBILITY_20261106.md)。

| 固定案例 | 输出最大绝对误差 | feature梯度最大误差 | mask梯度最大误差 |
|---|---:|---:|---:|
| 1×4×3×5，非方带符号信号 | 2.98023224e-8 | 1.19209290e-7 | 2.38418579e-7 |
| 1×4×2×3，边界主导信号 | 3.72529030e-9 | 1.86264515e-8 | 1.19209290e-7 |

原生与参考使用独立输入/掩膜叶张量，参考为明确零填充unfold、仅特征H/W重复和分组加权求和，不调用CARAFE原生算子。梯度使用非恒定带符号上游信号；归一化掩膜先detach，再测对mask叶的梯度。输出atol/rtol=2e-5/2e-4，两个梯度均5e-5/5e-4；全部误差比远小于1，不放宽门槛或自动重试。它是明确标注的合成算子一致性检查，不是数据集真值或模型精度指标。

实际环境：Linux/aarch64、Python3.12.14、Torch2.10.0+cu130、CUDA13.0、MMCV2.1.0、NVIDIA GB10 capability12.1，CPU0/1及线程2。单次预锁120秒、allocator1GiB、所属组RSS4GiB、可用RAM16GiB下限，启动48GiB；物理设备共享。parent约4.204秒，组RSS峰值1069563904字节、可用RAM最低73441906688字节，违规为空。CUDA allocated峰值101376字节，清理后allocated/reserved均0；这些不是整卡使用量或速度基准。

原工具/SSH/worker均0，Root后核7个parent及6个worker/库/lease引用、同boot出生身份及所属组为空，随后Root接收。只核选定四个库源/扩展，不宣称传递二进制全闭合；独立静态审阅未见具体阻断，实际运行资格由Root承担，没有冒称指定MCP全面审计。未加载图像、GT、数据集或模型，没有训练和AP，主训练计数不变。

下一步是隔离路径中的作者Detectron2._C CUDA构建及所需deform-conv/RoIAlign/NMS真实检查，再验证完整模型与统一数据注册/评价，之后才可开展外部公平长训。不凭旧THC头或版本差异预先改动数学算子，不把该检查加入方法AP主表。

[完整安全误差与资源记录](results/CARAFE_FIXED_OPERATOR_ACTUAL_20261106.json)；[自有检查源码](code/DIFFSAR_CARAFE_OPERATOR_PROBE_DRAFT_20261106_01/README.md)。
