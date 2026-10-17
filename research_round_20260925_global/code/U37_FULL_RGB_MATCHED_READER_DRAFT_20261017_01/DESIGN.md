# U37：原匹配实例 reader 加真实 RGB400 的组合草稿

状态：普通未绑定源码；仅 AST/源码审阅，未导入或运行 Torch/NumPy/PIL，未打开缓存、JPEG、标注、权重或运行资格。没有训练 runner，没有全人口实际验收或正式训练资格；不改变 U33/U35。

公开源根 `P=E:/SAR/seal/github_release_20261011`。原 reader 位于 `P/research_round_20260925_global/code/N28_DENSE_ROUTING_TRAIN_SOURCE_ONLY_20261017_01/reader.py`，SHA `82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870`。新几何复用 `code/U35_REAL_IMAGE400_CHUNK1_DRAFT_20261017_01/image_geometry.py`，SHA `65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7`（均相对 `P/research_round_20260925_global/`）。原 `train_core.py` 只用于核 ABI，不直接用于新主干训练。

## 最小接口与来源要求

`prepare(torch,np,c,context,*,matched_reader,image_geometry,source_refs,image_control=None)` 注入已加载的原两模块；source_refs 两角色精确为 `matched_reader/image_geometry`，每项是 `path/bytes/sha256`。模块文件须由原 `c.closed(ref)` 接纳、路径与实际模块一致，并符合上述源码 SHA。原 `matched_reader.prepare(torch,np,c,context)` 完整调用，原缓存、匹配、GT、teacher/PR 来源核验均不复制或放松。

返回 reader 保留 `.records`、`.input_references` 和 `.load(index)`。每次 load 先调用旧 reader，原 `F/e/M/L/targets/G/query_ids/identity` 的值及对象引用原样保留，只添加：

| 新字段 | 语义 |
| --- | --- |
| `rgb400` | CPU FP32 `[1,3,400,400]`，新 RGB 分支输入 |
| `geometry` | 同一公开 make_geometry 的实例，含 prepare_rgb/render_delta |
| `rgb_receipt` | 原 JPEG 引用、原/内容尺寸、EXIF、RGB 转换与新输入约定；不冒称 teacher 预处理复现 |

影像只取原 `identity['image']` 的精确绝对 `path/bytes/sha256`，并追加到 metadata 输入引用清单。旧 N28 reader 自身未把 JPEG 加入 Source，调用者必须事先登记原 Source；或者显式传入一个独立、已准入、事先收集这些引用的 `image_control`，只由其 `closed/check` 开图。没有登记就失败，不在运行中临时注册、不尝试同名文件、不在拒绝后自动换控制器。第二个 Control 的准入与 wrapper/PIL/运行库 Source 闭合仍由 Root 完成，Python 对象具有同名方法本身不构成准入证明。

已读原 writer/普通 U31 review 表明 identity 包含 image 引用；但原 N28 reader 并未单独保证它存在。因此本草稿逐记录要求该字段；实际缺失时失败。如果后来确需桥接，必须另提供 Source 登记的逐图 metadata，至少有 `dataset_index/image_id/完整原identity/精确image ref`，先核其来源与一一对应，再另审实现；当前没有猜测或补齐桥接数据。

JPEG 要求 format=JPEG、PIL 尺寸正好是原 `(W,H)`、EXIF Orientation 缺失或1，不自动转向。仅 convert RGB 得原 uint8，之后原几何 `make_geometry(torch, original_shape, actual_img_shape,800,400)`：保持比例 resize 到原记录内容尺寸、右下零 padding 至800、再到400；RGB /255，非 teacher mean/std。实际内容尺寸必须满足现成几何的 round-half-up 规则，否则失败，不静默重算。GT 和 L 仍是原 `(H,W)`，delta 用返回同一 geometry 渲染后才可加到 L；此处不执行该加法或损失。

## L 的来源已经静态确认：native，不是强 R 输出

原 reader.py:115 从 `fit['existing_native_L_refs'][index]` 开独立 carrier，:134 取键 `native_L` 为 `[100,H,W]`，:165 仅按原 GT 匹配 query ids 选择 G 条。因此本包装的 `L` 仍是这个 native 基底。

对应公开上游 `code/N16_cache_body.py:220–235` 从原 native `model.loss` 观测的最终 `mask_stages[-1]` 取 M，经 `diag.render_original` 得 `natural_L`，复制为 `natural_cpu`；:269–283 的固定 PR 头另算 sampled residual/combined，:300 保存在 `samples.actual_PR_residual/actual_combined`，未加回 natural_cpu。`code/N16_cache_GPU.py:506–515` 把 `natural_native_L` 分离存入 schema `N16-actual-GPU-natural100-L-carrier-v1` 的 `native_L`，并登记 `natural_native_L_carrier` 引用。N28 检查 `fixed_PR_checkpoint` 是共同来源约束，不能据此把返回 L 称为 R 修补后 L。

这是公开生成/读取源的含义确认，未读取本轮实际 carrier；实际字节身份仍由原 reader/Control 核验。不能据此断言该历史 natural L 与 U32 新补采渲染结果逐位相同，也不重新授予 U32/S0 导出资格。若正式 B1 将基底改成 R_L，必须另有真实 R_L 来源和明确合同；本草稿不做这一改动。

## 保留范围

原 reader.prepare 仍严格要求 FIT741/1661 的原始 metadata/annotation 身份；load 返回的是每图 G 个原 GT 匹配查询，不是全100候选，不增补 S0 或 background 监督。未来B0单图检查可由调用者预锁 dataset_index=691 的子集；正式B1仍为完整FIT741/1661，任何人口调整另立协议；本wrapper不选择训练子集。

原 train_core 的200输出渲染及头训练循环没有被复用为400主干 runner；不能直接把新字段交给旧 train_epoch 就声称完成全主干接线。当前交付仅是保留旧 reader 的真实 RGB 输入组合接口，几何、实际图像身份与完整训练适用性尚待 Root 后续验证。
