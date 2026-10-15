# N19 固定 directional-half 评分合同（UNBOUND）

普通纯 NumPy 函数，无文件访问、GT、模型、训练、AP、阈值调参或资格声明。ROOT 另行闭合原实际 GPU logits/原 L/C/K/类顺序/来源，并独自执行科学流程；N16/N17 STOP 不变。

`score_and_select(logits, native_L_C, C_flat, K) -> dict`：logits 为有限 FP32/FP64 [N,3]，类顺序 help/harm/noop；L 为有限同长 FP32/FP64 [N]；C_flat 严格 INT64 [N]、非负升序且唯一；K 为非 bool 的整数（含 NumPy integer），0≤K≤N。拒绝 None、NaN/Inf、数值伪装索引、错误形状/重复/乱序坐标和非法 K。N=0/K=0 合法，返回空支持；不把未定义输入当作零概率。

概率沿普通 `N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py` canonical：先转换 FP64，逐行减最大值、exp、按行求和相除，运算顺序相同；本函数限定二维 [N,3]。输入保持不变。

四个固定键顺序为 SYMMETRIC、DIR_HALF、GLOBAL_HALF、REVERSED_HALF。令 ph/pm 为 help/harm 概率，P=(原 L>0)，L=0 属 P0：

|policy|P0|P1|
|---|---|---|
|SYMMETRIC|ph−pm|ph−pm|
|DIR_HALF 主读出|ph−.5pm|.5ph−pm|
|GLOBAL_HALF 普通偏置控制|ph−.5pm|ph−.5pm|
|REVERSED_HALF 方向控制|.5ph−pm|ph−.5pm|

每值恰为 `{score, selected_flat_sorted, selected_C_positions}`。score 为 FP64[N]；支持只取严格 score>0，按 score 降序、abs(原L)升序、真实 flat 升序截最多 K 个；不设近零/排名容差、不补负分/零分/noop。返回两个 INT64[M] 均按 C 的升序坐标排列，满足 `C_flat[selected_C_positions]==selected_flat_sorted`。固定系数 .5，不暴露 r/mode/score threshold 参数。body 不应用 PR、不生成 mask/RLE；outer 用同实际 C 动作应用支持，并单独逐位重建旧 SYMMETRIC 支持/mask。

一次独立手算 fixture：由 log(手写概率)恢复概率，覆盖原 P0/P1/L0、正负零分、score/absL/flat 平局顺序、四策略不同支持、K0/空C与必要输入拒绝。先 NotImplementedError RED，随后 GREEN（1 test，exit0）；未使用实际科研输入/随机镜像/训练。AST 另作静态语法核对。源码/契约不认证部署校准、IoU/AP最优或科学资格。
