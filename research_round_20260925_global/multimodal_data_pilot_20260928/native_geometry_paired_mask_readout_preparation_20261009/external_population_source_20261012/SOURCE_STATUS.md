# 外部数据集清单接口

状态：仅独立 SOURCE 元数据计划；未绑定真实数据、未执行科学评价、不可启动训练或评价。

`population_contract.py` 使用 Python 3.8 兼容的纯标准库，无文件读写，无 Torch/CUDA，无标签、预测或结果访问。

- `freeze_population(images, source, gt_pin=None, audit_domain='external-dense-audit-v1')`：图片项严格为 `image_id / relative_path / height / width`；来源严格为 `dataset_id / version / split / source_id`。保留全部图片和空场景，按整数 ID 排序。ID 唯一，图片相对路径使用规范 POSIX 文本，拒绝逃逸、驱动器、Windows 别名和重复路径。GT 元数据 pin 格式为 `path / bytes / sha256`，不会读取对应文件。返回清单、六控制定义、全人口期望计数、抽查计划、存储估计和 canonical JSON 内容 SHA256；该摘要不能代替真实文件验证。
- `deterministic_dense_audit(image_ids, domain, limit=16)`：按紧凑 ASCII JSON `[domain, image_id]` 的 SHA256 排名选最多 16 张，ID 打破并列；返回 ID 升序。输入排列、尺寸、标签、预测、结果均不影响抽查。少于 16 张时全部选中。域与清单应在查看 GT/预测/结果前由 ROOT 冻结，不得事后更换域重抽。
- `validate_dense_descriptors(fields, arm, numeric_summary=None)`：检查 A 的 `native_masks / classes` 或 B 的上述字段加 `feature / coefficient` 的固定 native200 形状、FP32 和摘要格式；可拒绝标量摘要的 NaN/Inf/bool。不会读取张量或证明摘要、实际有限性和重放结果。

六控制保持当前评价器身份：`A_native_official`、`A_native_zero_budget`、`B_native_official`、`B_native_posterior_budget`、`B_posterior_official`、`B_posterior_budget`。B 后验几何是无界 `L+u`。official 质量在本控制掩膜上计算前景 sigmoid 均值；budget 为该几何支持和指定概率图的交并比代理；分数均乘以本臂类别概率。A zero-budget 使用自身原生概率，B native posterior-budget 使用自身后验概率。每图各控制固定 100 个候选，保留空掩膜、零分数；无 NMS 或分数截断，不假设 A/B 类别相同。

未来全人口 COCO 评价、候选身份、RLE/导出及计数检查必须覆盖所有 N 张、各控制 100N 条。密集独立数值重放只覆盖冻结的抽查图片；N > 16 时不得称全人口密集重放。原始 GT 总数须由未来独立完整解析取得，本接口保留为 pending，不能沿用内部 DEV 的 380。

存储计划显式使用每张 **73,064,000 bytes**，乘保留图片数；仅为当前固定 800 输入/native200、两臂字段契约的计划估计，排除导出/元数据/文件开销，不代表未知目标图片原尺寸推理内存或训练预算。

当前评价器硬编码 DEV 187 图/380 GT/18700 候选及 DEV 哈希。本接口没有改动该评价器；未来新建动态评价器必须独立绑定实际数据、GT、来源、原始 owned 执行及完整 CPU 资格。路径防逃逸目前是文本层约束，实际根目录解析、软链接/junction containment 仍需 ROOT 检查。HRSID 数量、数据版本、划分、重复地理区域及测试接触历史均未核验；不得写死数量或称盲测。
