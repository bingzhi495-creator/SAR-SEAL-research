# U57 owner ESRCH 修复候选

## 新增时间快照：2026-10-21T14:23Z（北京时间 22:23）

修复源的实际控制接口检查已取得原始 SSH 0：科学 SPEC 内容、正式父状态、原预锁和训练配方保持不变，除替换 owner 源以外，其余 770 个源码绑定保持相同。实际使用 `make_control(spec, psutil_module)`（psutil 6.1.1）构造控制器，771 路径 allowlist、修复 owner 源的字节读取，以及 CPU 0/1 下 4 次资源检查通过；没有导入 Torch / NumPy。上述仍不是完整 owner 命令行或模型训练资格证明。

新的 corrected-source 准入已单独、前瞻限定最多 1 次；旧 capture-recompute 的最多 1 次额度仍已耗尽，不重置。原始数值任务 88171 已于上述时间启动，计划从正式 10,496 恢复至 10,752，Root 已登记原始执行句柄。**截至该启动快照，任务尚未结束，未取得成功接纳，新增合格更新仍为 0，正式前缀仍为 10,496。** 这里不宣称工作进程成功、完整 owner 命令行通过或模型收益。下文的 UNEXECUTED 标签专指启动前候选状态，启动后属于已尝试执行、结果待真实终态核验。
**启动前状态：UNEXECUTED SCIENTIFIC CANDIDATE。启动后状态见新增时间快照，尚无成功接纳证明。** 这是控制层的最小运行可靠性修复，不是网络模块、学术创新或模型涨点结果。

本目录是独立普通源码副本，供审阅和复现针对性元数据回归。它不包含可直接启动项目训练的执行配置、凭据、数据、标注或权重。请同时阅读 [失败、修复与证据边界](../../U57_OWNER_ESRCH_FAILURE_AND_FIX_20261021.md)。

## 内容与唯一源码变更

- `linux_owner.py`：`proc_identity` 的异常分支由仅捕获 `FileNotFoundError` 改为捕获 `(FileNotFoundError, ProcessLookupError)`。除此之外保持原代码字节不变。
- `test_metadata.py`：通过模拟进程状态读取的 I/O 结果验证上述行为；可相对定位同目录的候选源码。

进程在被枚举后退出时，读取其状态可能出现 ESRCH。新增分支将该单个进程视为已消失，同时保持权限错误、其他 I/O 错误与格式错误继续传播。该修复不放宽 PID 出生标识、资源限制、进程结束核验或训练成功接纳规则。[Python 官方异常说明](https://docs.python.org/3.12/library/exceptions.html#ProcessLookupError)

## 已完成的验证

本机原实现的针对性回归实际返回 1，修复候选实际返回 0；独立静态审阅未发现该单处修改的具体阻塞项。DGX 上实际完成 2 个标准库针对性测试，原始 SSH 返回 0，并检查真实自身进程身份、新 `make_control` 接口签名及固定资源限制。

DGX 验证范围为 **metadata-only Linux validation**：没有导入 Torch / NumPy，没有运行模型，没有执行完整 owner 命令行。测试通过不能替代实际训练准入、恢复、结束与接纳证据。

## 运行元数据测试

在本目录运行：

```text
python -B test_metadata.py
```

这只执行针对性元数据测试。不要把直接启动 `linux_owner.py` 视为已经配置好研究训练；真实执行需另有明确、有限的 corrected-source 准入。

## 固定候选指纹

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `linux_owner.py` | 37,956 | `6c6e2867bc7a36624f09e6483eadaaf6cd2f8340001e839000b7d85c5ac6659e` |
| `test_metadata.py` | 2,897 | `2e4be450ab4f155cb9f39e8cf1c818deeea6eb61be425ecdc737e92234cff7a9` |

本文记录时，正式每臂连续前缀仍为 10,496，本机独立前缀为 2,816，原目标为每臂 37,050 次更新。本轮失败尝试新增合格更新为 0；论文机制、优越性、重复种子、跨数据集与 TGRS 成熟度均未由本修复确认。
