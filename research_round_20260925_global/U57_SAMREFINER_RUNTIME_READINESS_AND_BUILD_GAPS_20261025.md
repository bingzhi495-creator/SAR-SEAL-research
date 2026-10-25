# SAMRefiner 实际运行准备：依赖与未闭合条件

2026-10-25，北京时间。本文依据固定作者源码、官方发布元数据及一次DGX安装登记查询；初稿只做元数据核查；后续工具链检查、GPU隐藏库导入、预构建下载失败及离线暂存见本页实际更新。当前已另行完成FastGeodis隔离构建与GPU隐藏导入，详见后续实际更新；GPU算子在两个固定合成样例上已实际检查（见下节）；SAM模型/科学图像及AP仍未执行。主线仍是固定50轮PLAIN/FULL实验，不因依赖准备改动其配方或当前环境。

## 已取得的最小证据

[作者API桥接候选](code/U57_SAMREFINER_REAL_BRIDGE_DRAFT_20261025_01/README.md)的7项小NumPy测试和独立源码审阅通过；[固定候选适配器](code/U57_SAMREFINER_FIXED_CANDIDATE_ADAPTER_DRAFT_20261025_01/README.md)另有13项有限标准库测试。这些只检查替身后端/数组边界，不能证明实际SAM运行、图像对应、内部候选顺序或AP。

初次在DGX训练环境仅查询distribution元数据，未导入torch、numpy、cv2或FastGeodis：

| 分发包 | 安装登记版本 |
|---|---|
| torch | 2.10.0+cu130 |
| torchvision | 0.25.0+cu130 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0.86 |
| scipy | 1.15.3 |
| matplotlib | 3.11.2 |
| FastGeodis | 未找到该分发包登记 |
| segment-anything | 未找到该分发包登记 |

“未登记”不是实际import失败的证据，也不排除本地或vendored实现；登记版本不能证明C++/CUDA扩展ABI可用。Torch版本中的cu130也不能代替nvcc、toolkit或CUDA_HOME实际检查。

## FastGeodis 的官方构建与发布证据

固定官方源码提交为0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e。[构建依赖](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/pyproject.toml)列出setuptools、torch>=1.5.0和wheel；[setup.py](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/setup.py#L8)还使用pkg_resources及PyTorch的C++/CUDA扩展构建，Linux涉及OpenMP。是否构建CUDA路径取决于实际环境与构建条件，不能从已安装Torch的版本字符串直接断定。

本次官方[PyPI发布元数据](https://pypi.org/pypi/FastGeodis/json)查询显示当前1.0.5仅有源码tar.gz，所列历史发行文件也未见wheel；[官方GitHub releases](https://github.com/masadcv/FastGeodis/releases)未列上传资产。该事实限本次查询时点，不承诺未来发行情况。

已查[构建CI](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/.github/workflows/build.yml#L17)和[发布配置](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/.github/workflows/build-sdist-publish.yml#L16)未取得aarch64＋CUDA13专项测试证据。这表示兼容性尚未证实，不能推断一定不可编译。

## 作者SAM变体与接口

固定SAMRefiner提交为4bb7f95738c7a2e805f189210fb09e02a2197557。[README](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/README.md#L21)的基础SAM方案使用ViT-H及Meta官方sam_vit_h_4b8939.pth入口；HQ-SAM另列。文件名中的短串不能当作本次独立核验过的完整权重摘要，当前未下载或加载该权重。

作者要求使用仓库内segment-anything。[外层调用](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/sam_refiner.py#L133)依赖[vendor模型的forward_with_image_embeddings](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/segment-anything/segment_anything/modeling/sam.py#L134)及未阈值化掩膜路径；不能仅安装同名标准包便称作者复现。README示例的“sam-vit_h”与[实际registry键vit_h](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/segment-anything/segment_anything/build_sam.py#L47)存在差异，后续应按固定源码核对实例化入口。

## 下一项实际验证及范围

后续使用独立候选环境，先实际检查CPU架构/Python ABI、C++编译器/OpenMP、nvcc/toolkit/CUDA_HOME及目标GPU架构、Torch实际导入/扩展ABI、setuptools/pkg_resources/wheel。再依据构建日志与小型真实扩展调用验证CPU/CUDA路径，不能把安装命令退出0直接当作GPU功能可用。后续更新只完成部分前置检查，实际编译与真实GPU功能仍未验证。

实际模型试点还需核验固定作者代码及权重、模型状态、原图路径与真实尺寸/像素/掩膜对应、真实返回行对应和资源峰值。空掩膜策略须在看AP前明确；若采用显式透传，应标为适配策略并保持全部100候选、原类别和S0。候选桥接echo的ID不能替代内部对应核验。

本页不新增外部方法分数。SAMRefiner、DiffSARShipInst和SegS-YOLO的本项目同协议实例AP仍未取得，见[外部比较状态](U57_EXTERNAL_METHOD_COMPARISON_STATUS_20261025.md)。主实验最新计数以[README](../README.md)和[当前安全标量](results/U57_U60_ACTUAL_SCALARS_20261021.json)为准。


## 工具链与GPU隐藏导入的实际更新

实查aarch64、Python3.12.14/cpython-312及Python.h存在；nvcc13.0.88和C++13.3.0版本命令均退出0，nvcc目标列表含compute_121。setuptools79.0.1、wheel0.48.0、ninja1.13.2及pip26.2.1有安装登记。pkg_resources实际导入和分发版本解析成功，候选CUDA头文件及cudart库文件存在。文件存在不是编译成功或GPU核函数证据。

CUDA_HOME环境变量未设不构成必然阻断。[PyTorch2.10](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/cpp_extension.py#L86)还查PATH中的nvcc及默认安装根；[FastGeodis判据](https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/setup.py#L22)使用模块发现值、CUDA可用性与FORCE_CUDA。

另一次有界诊断明确令GPU不可见，实际导入Torch、pkg_resources和cpp_extension：Torch2.10.0+cu130、构建CUDA13.0，模块发现CUDA_HOME且nvcc文件存在，实际版本解析(2,10,0)，显式12.1生成compute_121/sm_121参数；CUDA未初始化。约1.56秒、峰值RSS约489MiB。未加载模型、科学数据/原图或执行科学张量任务。cpp_extension导入会触及CUDA可用性检查，因此GPU隐藏条件不能省略；结果不证明真实GPU可用性、扩展ABI/核函数或SAM运行。

## 一次预构建下载失败与离线来源修正

主训练16640严格接收并确认科学进程已结束后，尝试一次600秒上限、附10秒终止宽限的隔离软件准备。GPU不可见、CPU0/1、nice19、单个编译任务，原训练环境不升级、不自动补装依赖。该步骤不是模型训练或评价。

原始工具及SSH均退出1：DGX直接下载源码包发生TLS握手超时。实际确认失败发生在创建venv、启动pip或编译器之前，原进程组已不存在。失败记录保留，原尝试限额耗尽；不把它记为方法失效、种子或科学更新，不用后续元数据查询0升级原失败。主训练已从合格16640父状态继续。

随后按既有下载安排，在本机取得[官方FastGeodis1.0.5源码包](https://files.pythonhosted.org/packages/eb/dd/8220b4787908c4ba14f97f92a09690524184199efabffac537d1afaba314/FastGeodis-1.0.5.tar.gz)，核验42,680字节、SHA256为2869ecfa8d6ab0b54284d8f819e68458da4cc7b10e5282bd69d4a0fcadc3fead，传入原候选目录并重新核验。包内setup.py此前仅在内存核到与固定0f6cc1e6提交一致。第三方完整源码包不随本仓库发布。

该次修正时只完成离线源码暂存，尚未再次安装或编译，原下载失败状态另行保留。未来继续须针对已修正的离线来源另锁有界步骤，先核旧目录没有环境/编译产物，不得换目录盲目重试；依赖准备和桥接代码仍不构成外部同协议AP。


## 已完成：使用已核验源码的离线隔离构建

在后续主训练16896端点严格接收后，预先另锁一次有界离线安装：同一候选目录、同一已核验源码，启动前确认原失败状态及其副本相同、没有venv或BUILD.log，不重新下载、不换目录。新操作使用独立状态文件，原下载失败未升级或覆盖。独立审阅确认输入/旧记录保护、实际外层600秒强制终止和真实返回码传播，没有将预锁字段当作执行证据。

本次原工具、SSH、pip和隔离导入均真实退出0，开始到结束约2分24秒；Root随后确认原进程组为空且无未解决观察。pip目标为全新venv，通过其中的路径文件复用已装Torch，不请求升级原训练包。这支持安装目标和执行路径的限定事实，不能当作整个原环境前后所有字节未变的证明。

安装产物为aarch64/cpython-312原生扩展；readelf确认含CUDA设备代码段，cuobjdump列出sm_121。它支持目标代码已生成，不证明实际GPU核函数能正常执行。导入时GPU不可见、CUDA未初始化，没有加载SAM模型、权重、SAR图像或执行AP。安全标量见[本次依赖构建结果](results/U57_SAMREFINER_FASTGEODIS_NATIVE_BUILD_SAFE_STATUS_20261025.json)。源码包、二进制、环境路径配置及完整构建日志不随本仓库发布。科学训练随后从合格16896父状态继续。

下述最小功能检查在提出时待执行，现已按原规则实际完成（结果见本页最新更新）：[官方1.0.5 API](https://github.com/masadcv/FastGeodis/blob/1.0.5/FastGeodis/__init__.py#L37)为generalised_geodesic2d(image, softmask, v, lamb, iter=2)。按[官方示例约定](https://github.com/masadcv/FastGeodis/blob/1.0.5/README.md#L109)，使用9×9连续float32、非种子1/种子0、v=1e10及两次扫描。一例全零图、中心种子、lamb=0，检查结构、有限非负、种子约0及轴向邻点约1；另一例强度阶跃、左侧种子、lamb=1，只检查结构、有限非负、种子约0及跨边界正代价，不虚构完整距离图。

未来应分别明确调用CPU与CUDA并同步GPU，CUDA不可用须报告失败而非退回CPU。可记录两者最大差异，不强制逐位相等或互当GT；[作者明确讨论扫描精度取舍](https://github.com/masadcv/FastGeodis/blob/1.0.5/README.md#L21)。此项只检验极小合成输入的二维算子功能，不是SAMRefiner或SAR模型效果，该段最初成稿时未运行；后续实际结果及适用范围见下节。


## 已完成：四次固定微型CPU/CUDA算子调用

后续主训练17152严格接收并确认原科学进程已回收后，Root独占执行两个预先固定9×9样例，在CPU与CUDA各调用一次。原始工具、SSH、worker均退出0；Root核源字节、结果结构与原进程组为空，没有用后续查询0替代原退出。两种子距离均0，欧氏样例轴向邻点及阶跃样例跨边界检查点均1；所有结构、有限非负和CUDA同步检查通过。两组完整输出的观测最大差为0，仅适用于这两个样例，不是一般等价性证明，也不是完整距离图GT。

[独立检查源码与运行范围](code/U57_FASTGEODIS_CPU_CUDA_SMOKE_20261025_01/README.md)及[安全结果](results/U57_FASTGEODIS_FIXED_CPU_CUDA_CHECK_SAFE_RESULT_20261025.json)已保存。PyTorch峰值分配3584字节、清理后分配0，进程峰值RSS741330944字节；分配峰值不含全部驱动/上下文内存，不认证whole-GPU。约1.49秒为包含库导入的worker时间，不是SAM或稳态推理速度。该单元任务另行计数，不并入原训练/输入验证管线账户或正式预算。

主训练已继续。SAMRefiner作者模型、权重装载、真实图像/返回行对应与同协议AP仍未验证。作者固定提交的22个必要源码/许可文件（149194字节）仅在私有目录准备并按官方Git blob核对，未安装或运行；完整第三方源码和权重不随Git发布。ViT-H官方权重初次有界传输达到300秒上限并保留624951296字节前缀，失败状态保留；随后按同一ETag及HTTP206 Content-Range条件续传，完整文件及模型加载尚未核验，不能据元数据或部分文件宣称模型就绪。


## 官方ViT-H权重获取完成，模型仍未验证

上述续传后来真实退出0，条件HTTP206的表示版本与字节范围已核，最终文件为2564550879字节。本地流式SHA256为a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e；这是本次本地计算的摘要，不宣称取得Meta独立公布的完整SHA256，也不把ETag当作SHA256。原300秒传输终止记录保留。文件尚未由本任务传到DGX或加载，不能据下载完成宣称SAM模型、真实图像或AP已验证。[获取安全状态](results/U57_SAM_VITH_OFFICIAL_ASSET_ACQUISITION_SAFE_STATUS_20261025.json)仅保存来源与摘要，权重本体不上传Git。
