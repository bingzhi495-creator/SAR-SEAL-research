# N13 全 FIT 冻结 M→L GPU 关系实际观察

2026-10-13，ROOT 原持有单次执行句柄已实际结束、回收并确认原出生实例消失，exit0。全38旧正常与6旧失败首先核验，再检查8696项来源闭包和本次全部输出；本次成为normal38（共39正常）。

全部741图、1661匹配query的 canonical100 FP32原两段渲染输出与冻结 GPU matched L逐字节一致，包括signed zero；mismatch image/query/pixel均0。subsetG诊断同样全体字节一致，不能把100通道输入改动认定为此前CPU差值的修复原因。

原执行157.109秒，全GPU观察峰值2689MiB，最低RAM35671105536字节，GPU cleanup allocated/reserved均0。此为whole GPU observed，不是每进程WDDM完整显存认证。

canonical100未匹配槽位仍是dummy零。关系结果仅限保存matched M、保存L、指定Torch运行环境和渲染链；不恢复原未匹配M，不重跑教师，不认证历史完整M stride、骨干/匹配数值，也不证明CPU与CUDA full-L 1e-4一致。原CPUv01和index3诊断退出1保持失败。ROOT关系记录仅为新CPUv2资格的一项前提，不是缓存Q。

当前缓存CPU v2尚未执行，固定48轮训练尚未开始，没有新增AP/创新/TGRS证据。下一步先独立复核精确冻结GPU L上的区域、draw、标签和采样F，再准入预锁配对训练。
