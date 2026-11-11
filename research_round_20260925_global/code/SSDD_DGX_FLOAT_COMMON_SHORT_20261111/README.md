# PSeg-SSDD 浮点标注的 DGX 固定共同行短窗

本目录提供 5201 种子的短训资格入口和独立验收。两臂均使用 P395 恢复的官方源多边形，按原生、逐步中心的次序运行。两臂复用原始 RF-DETR Q100、576 输入、BF16、优化器、调度、EMA 和 FIT 数据加载器。唯一训练干预是同一浮点标注上的掩膜采样。两臂保留相同的有序非空交集，允许保留零共同实例的访问。每臂消费 32 个实际 FIT 访问，完成 8 次 AdamW 和 8 次 EMA 更新。 [预锁协议](PROTOCOL_PRELOCK.json)

浮点替换局限在当前原生 FIT 数据集实例，发生在原始标注读取后、原始 COCO 栅格化和增强前。运行时验证 741 张图和 1,661 条标注的图像及实例身份、顺序和全部非掩膜字段，再核对浮点顶点截断与固定整数坐标一致。冻结原整数 JSON 的坐标数学上为整数，序列化时可带小数点。原目录和整数索引保持原样。

## 已执行核对

Windows 的原生 RF-DETR 1.12.0 读入实测对照了两种数据集构造。一组在固定整数布局使用本目录的浮点读取适配，另一组直接加载同一份浮点 JSON。全部 741 张图、1,661 条实例的实际标注读取逐项相同，浮点覆盖改变其中 1,606 条的顶点。按排序图像 ID 选取的前 32 张图使用同尺寸空白图像完成原始 COCO 张量转换，43 条标注的全部张量逐位一致。测试还验证了钩子退出后原始读取恢复。该核对未解码真实影像或执行模型、优化器。早期虚拟环境启动和整数坐标序列化类型的两次失败记录均保留。 [聚合核对结果](RESULT_SAFE.json)

资格入口和输入接线已验证。DGX 的真实训练、原进程退出和独立验收待执行。P396 的 32 个固定增强视图与本目录的 Windows 栅格化核对使用独立样本流程。实际运行时仍需核对优化器消费的 32 个批次。短窗的证据范围为训练接线和优化器资格。

## 准备和运行

准备规范路径的原 P242 DGX owner、Linux 资源保护、P333 固定共同行接入、P330 聚合证据、P395 浮点 FIT 文件、P396 聚合证据及本目录的冻结协议。运行器中的摘要锁定各文件的精确字节。请求使用 UTF-8 JSON，由操作者计算完整 SHA-256。请求需要以下键。

- `status` 使用 `SSDD_DGX_FLOAT_COMMON_SHORT_REQUEST_V1`。
- `training_protocol` 使用协议原字节的 SHA-1 `0cf70c678b8095c6256cd678e03994408a86e56b`。`implementation_commit` 使用该实现的 Git 提交 ID。
- `python` 使用原隔离 DGX 解释器的绝对路径。`code_root` 指已核验的代码目录。`lease_directory` 指原两个共享 GPU 锁的目录。`work_root` 是该目录下尚不存在的独立子目录。
- `worker`、`stock_owner`、`linux_guard`、`common_worker`、`p330_report`、`float_overlay`、`p396_report`、`protocol_document` 各使用 `path`、`bytes`、`sha256` 三个字段的文件引用。`stock_binding_sha256` 使用原 P242 冻结绑定的 SHA-256。

用原解释器的 `-I -B -X utf8` 运行 `dgx_float_common_short.py`，传入 `--request`、`--request-sha256` 和 `--check-source`。源码和输入摘要核对不会启动训练或 CUDA。原 DGX 准入要求宿主可用内存至少 40 GiB、CUDA 即时可用至少 12 GiB、原生 BF16、两项原 GPU 所有权锁空闲。已固定每臂 6 GiB CUDA 分配上限、8 GiB worker RSS 和 300 秒资源保护。满足真实准入后，运行环境使用 `CUDA_VISIBLE_DEVICES=0`、`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，并以 `--run` 替换源码核对参数。

使用原 SSH 会话保存两条 arm start 和 owner 结束事件的实际输出、退出状态、进程出生信息与进程组闭合。独立验收的回执格式由 `verify_external_wait` 检查。验收使用 `CUDA_VISIBLE_DEVICES=-1`，传入 `--verify-closed --original-wait --original-wait-sha256 --output`。验收会复核两臂原退出、逐批共同实例和掩膜证据、训练计数、初始化和原优化器条件。输出应写入独立新路径。协议限定 FIT、零完整轮次，且不生成权重、DEV、TEST 或 AP 输出。
