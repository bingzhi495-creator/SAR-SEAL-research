# N20 DEV 九邻：旧 carrier 的可复用范围（普通源码静态核对）

**结论：N16/N17 shared 没有完整 `F_original[256,H,W]`；它保存100个 C 的空间并集上的真实 GPU F。已有原生 packet→F256→GPU renderer 路径可能免去新 teacher capture，但不能只靠该稀疏 shared 证明全部九邻齐备。** 本文未读真实 packet、Source/资格链或运行代码。

| 普通源码位置 | 确定的实际字段/行为 |
| --- | --- |
| `tmp/N16_FIXED_E24_ALL100_SCOUT_SHARD_GPU_ROOT_UNBOUND_DRAFT_20261014_01.py:921–941`，`shared_actual_image` | 对 natural q=0..99 构造 C，`A=union(C_q)`，`union_flat_indices=flatnonzero(A)` 为升序 INT64[U]；从 GPU 原图 F 按 `(flat//W,flat%W)` gather，保存 CPU FP32 `F_unique[U,256]`。 |
| 同文件 `:937–941` | shared schema=`N16-actual-GPU-image-shared-inputs-v1`；另含 `native_L[100,H,W]`、`e0[100,256]`、`C_indices[100]`、`C_counts/K_counts[100]`、`identity/native_packet/selector`、natural/export query IDs、`actual_GPU_F_captured=True`、`CPU_F_rerendered=False`。**完整 L 不等于完整 F。** |
| 同文件 `:878–903`，`original_image` | `n10.packet_check(torch,reader,coco,packet,row,index,ref,gate)` 返回 `fields,identity,qids`；使用 `fields['F256']`、`e0/M200/final_class_logits`，再 `n10.original_feature(torch,renderer,F,identity)` 形成临时完整 `F_original[256,H,W]`。此函数没有 teacher forward；随后仅持久化上面的并集 F。 |
| `tmp/N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_04.py:1074–1088,1228–1237` | 从旧物理 bundle 的 `members[row['shared_actual_image_inputs']]` 取同 shared，直接把 `F_unique/e0/native_L` 转 GPU；未新增稠密空间 F。新 point chunks 还删除重复的 `sampled_F/e0/native_L`（`:1103–1117`）。N17 的 `extra8` 是 native-L 八邻，不能冒充 feature 八邻。 |

坐标仍是 `identity.original_shape=(H,W)` 的行优先 flat=`y*W+x`。对目标中心集合 A 逐轴 clamp 原图整数±1，令 B 为九邻并集、D=B\A、E=|D|。旧 `F_unique` 精确覆盖 B∩A；只缺 D。一般 C 的外边缘还会产生一圈 D，其他 query 的 C 可覆盖其中一部分，但“C 含 P 的8像素 dilation”并不保证 C 对±1邻接闭合。严格边界 `0≤E≤min(8U,HW−U)`；A为空或已覆盖全图时 D为空。缺点不能补零、借最近已存点或 CPU 近似重渲染；图像边界按坐标 clamp 复用合法边界像素。

最小可复用执行路径：Root 闭合旧 bundle/native_packet/renderer refs 后，沿**既有** `original_image` 的原生 packet→F256→同GPU两级 bilinear renderer/channel16 路径重建 F；新 gather 在 A 上须与旧真实 `F_unique` 逐字节相等，再仅保存 D 的 `extra_F[E,256]`。这是复用原捕获源特征、重新 GPU 渲染，不是新的 teacher forward，也不是直接重用已保存完整 F。外部 `native_packet` 的物理内部键路径未在这次准许的 writer 中展开；准确已核 ABI 是 `packet_check` 的返回 `fields['F256']`，Root 需按既有资格 hook 闭合该来源，不能自行假定 packet 顶层结构或 F 原生空间尺寸。

新增 raw carrier 若沿用 N20 坐标合同，单图为 `1024E + 8E + 72U` bytes（extra F、extra INT64 flats、INT64[U,9] map）；旧中心/e0/L继续引用，不重复保存。若只持久化 extra F/flat 并独立重建 map，下界为 `1032E` bytes，需另立明确保存合同。稠密新增 F 则是 `1024HW` bytes；两者都另计头/序列化/资格读回/RAM，不承诺实际 E、总耗时或 deadline。每图旧原生 carrier 可用性及 E 必须由 Root 的已闭合 refs/坐标实数确认；本次静态检查不冒充实际覆盖资格。

STOP EDIT：无 actual 科学读取、执行、Git 或新评价 outer。
