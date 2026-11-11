# Windows 原生 CUDA 浮点 SSDD 固定共同行短窗

两个短训臂按原生、逐步中心顺序运行。原 P333 Windows 所有者、JobObject、资源保护、RF-DETR Q100、576 输入、原生 BF16、AdamW、EMA 和 E100 调度全部沿用。原整数 FIT 通过原有入口检查后，数据集使用独立核验的 P397 浮点目录，直接由原 FitOnly 加载器打开其中的 COCO 标注。两个目录的 JPEG 字节清单相同，P397 标注与 P395 官方源多边形文件逐字节相同。 [冻结协议](PROTOCOL_PRELOCK.json)

每臂预锁 32 个 FIT 访问、8 次 AdamW 和 8 次 EMA。按原逻辑保留两臂有序的非空共同目标，原空掩膜保护移除数为零，允许保留无目标访问。实际资格还要求图像、非掩膜字段、增强轨迹和随机状态一致，并至少出现一个掩膜差异。阶段仅限 FIT，零完整轮次，且不生成模型权重或 AP。

## 当前核对

在无 CUDA 的 CPU 预检中核验了原整数 FIT 目录、P395 浮点标注、P397 的 741 张 JPEG 和 1,661 条标注、P398 共同行人口回执及原 P330/P333 闭合证据。三项保护回归通过，覆盖原始数据核验先于浮点目录接入、目录别名拒收和两臂实际关闭顺序。CPU 预检不打开共享 GPU 状态，不导入 Torch。首次配置因 P398 摘要少一位被拒收，后续用新请求通过预检。 [预检汇总](PREFLIGHT_RESULT_SAFE.json)

## 操作

用绝对规范路径准备私有请求，字段为 `status`、`training_protocol`、`implementation_commit`、`operation_commit`、`output_root`、`worker`、`p333_owner`、`p333_worker`、`p330_report`、`p330_terminal`、`p333_pair`、`p395_report`、`p397_report`、`p398_report`、`float_layout`、`float_overlay`、`protocol_document` 和 `coco`。文件引用包含 `path`、`bytes`、`sha256`。`status` 使用 `SSDD_WINDOWS_FLOAT_COMMON_SHORT_REQUEST_V1`，`training_protocol` 是预锁协议原字节的 SHA-1。实施与操作身份使用独立的 40 位标识。`output_root` 是原 Windows 执行根目录下以 `SSDD_WINDOWS_Q100_FLOAT_COMMON_` 开头的新目录。私有请求和原权重不属于发布文件。

执行 `windows_float_common_short.py --config <request> --config-sha256 <sha256> --preflight-cpu` 可重复核验输入。原 Windows GPU 所有者正常交接且资源满足原门槛后，使用 `--run` 运行。原入口要求 16 GiB 宿主可用内存、12 GiB CUDA 即时可用，最大 CUDA 分配 6 GiB、worker RSS 10 GiB、每臂 300 秒。资源与活动状态由原始 Windows 所有者和子进程保护检查。

独立验收使用 `--accept --original-wait <receipt> --original-wait-sha256 <sha256> --original-stdout <stdout>`。等待回执须保存原所有者出生身份、原退出状态、请求及源摘要和原外层输出摘要，状态为 `SSDD_WINDOWS_FLOAT_COMMON_SHORT_ORIGINAL_OWNER_WAIT_V1`。验收核对两个原 JobObject、资源回执、32 个实际消费批次和八次更新，写出新的资格汇总。当前未启动 CUDA 训练。

