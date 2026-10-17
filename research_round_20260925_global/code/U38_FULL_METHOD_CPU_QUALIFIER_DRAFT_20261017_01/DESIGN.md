# U38 CPU 资格函数草稿

仅普通未绑定源码。当前只做 AST 与纯 Python metadata 检查；没有执行 Torch/NumPy、模型、JPEG/GT/cache/权重读取，没有接触活动 N28 的 SOURCE/RAW/Q。后续执行、Source 闭合与硬时间/内存边界由 Root wrapper 负责；本文件没有 runner/binder、子进程或重试逻辑。

## 精确接口

`qualify(torch,np,c,context,modules,source_refs,image_control=None,report=None)` 返回 JSON 可序列化的资格报告；调用者可传空 `report={}`，异常时保留已完成图数、阶段与错误，函数仍抛异常，不以部分人口返回成功。`c` 沿原 `closed/check/psutil` ABI；独立 image_control 如需使用仍须 Root 事先准入。时间记录允许零，不凭计时数值推断执行成功。

modules/source_refs 恰有11角色：`shared_initialization/rgb_matched_reader`，U34 原七角色 `full_legacy/full_dsconv/full_head/full_feature/plain_legacy/plain_feature/instance_adapter`，以及 U37 两角色 `matched_reader/image_geometry`。前两模块固定公开 SHA，其他精确 SHA 来自其已核常量；所有模块文件在调用前经 `c.closed`、路径、长度与字节 SHA 核对。函数不自行加载任意模块或 checkpoint。

已读公开源：`P/research_round_20260925_global/code/U34_FULL_PLAIN_INIT_DRAFT_20261017_01/shared_initialization.py`（SHA `4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292`）和 `code/U37_FULL_RGB_MATCHED_READER_DRAFT_20261017_01/rgb_matched_reader.py`（SHA `678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025`），其中 `P=E:/SAR/seal/github_release_20261011`。当前未执行这些函数。

## 两阶段，均保持 CUDA 冷

先调用原 U37.prepare，按原 dataset_index 0..740 完整流式调用 reader.load。原 N28 继续负责 F/e/M/native-L/GT 的 CPU FP32/有限值/shape/digest、匹配与 Source 检查；U37 负责原 image 引用、JPEG尺寸/EXIF、RGB400 和几何。U38 不复制这些校验，不改变 e/M/L/GT，不重新匹配，不需要 native100 M 或 S0。附加核对逐图 identity/G/query 顺序、完整741图与1661实例计数，保存尺寸/G及身份摘要，逐图释放 tensor。

L 是原 N16 native_L 按 GT匹配 query 选择的结果，不是强 PR 修补后的 L。正式 B1 人口仍 FIT741/1661；691只是已用 B0 单图索引，本函数不筛出训练子集。

全部读入通过后释放 reader，再实际调用 U34.build_pair。由原实现核16公共根/104状态（68参数张量、36BN buffers）、同 seed202610173101 公共 adapter、原inactive及未映射状态不变。报告保留其实际初始化回执和参数计数；不做模型 forward，不创建 optimizer，不保存或返回权重。返回前删除两 CPU adapter/原模型，仅保留 JSON 摘要，并用弱引用检查这些模型对象已经释放。CPU allocator/RSS未立即下降不等于模型对象仍存活或内存泄漏。

每阶段检查 `torch.cuda.is_initialized()==False` 和 FP32 默认dtype。后续调用者必须在加载依赖前也自行保证冷CUDA；该函数无法逆转此前初始化。库导入来源、时间/16GiB等资源控制仍由执行 wrapper 负责，不能把本草稿当成已获资源资格。

## 确定性资源图目录

只使用完整已验 metadata，不用 loss、AP 或失败表现选图：small 按 `(G,原H×W,dataset_index)` 取最小；max_area、max_G、max_G_area 分别取最大，均按最小 dataset_index 破平局。候选目录依 small→max_area→max_G→max_G_area 首次出现去重，再显式追加同一 small 作返回观察。

另输出 Root 当前拟优先的三图 proposal：small→max_G_area→small。报告真实 unique图数、shape/G覆盖；若角色落在同图或同shape如实保留，不能声称已经覆盖所有风险。四极值目录不是必须全跑的GPU门，三图也未在本函数执行，更不能代替741样本稳态结论。resource_candidates 是纯 metadata 函数，可用微小 synthetic rows 验证稳定tie和去重。

报告资源量只有逐图加载后及初始化前后的边界 RSS/available RAM 样本、最大边界采样RSS和耗时，不冒称连续峰值、pinned驻留量、速度或泄漏证明。报告不包含原始像素、GT mask 或权重张量，不写文件；Root 可按发布范围保存摘要。

成功仅表示本次 CPU 原 reader/RGB 与公共初始化检查完成；`new_training=False`、`GPU_resource_gate_passed=False`。不存在 AP/DEV/TEST 评价、科学方法门、正式训练起算或全主干数值等价结论。
