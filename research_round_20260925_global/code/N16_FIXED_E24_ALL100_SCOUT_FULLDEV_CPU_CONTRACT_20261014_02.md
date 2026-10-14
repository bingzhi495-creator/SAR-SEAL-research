# N16 fullDEV CPU 入口 v02：仅继承第13个失败门

状态：普通 UNBOUND、AST/静态结构核验完成、STOP EDIT；未实际执行科研，未读取实际 SCI/SOURCE/RAW/SPEC/Q/缓存/权重/GT，不操作 Git。旧CPU01、adapter01及其合成测试保留不改。

入口：`tmp/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_ROOT_UNBOUND_DRAFT_20261014_02.py`。以已冻结普通 GPU03 (`tmp/N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_03.py`) 第1–440行、TRAIN_SCHEMA前全部纯 RAW/Source gate 为新前缀；只改开头文档描述。CPU_SCHEMA已与新独立CPU契约相同。CPU01从def tree_equal起全部后缀原样接入。AST证明所有复制前门函数与GPU03精确相同、所有CPU后缀函数与CPU01精确相同，没有调整排名、选点、PR、捕获解码、掩膜、COCO或容差。

CPU SPEC 现在需要 `additional_preserved_N16_fullDEV_packet_ABI_failure`，字段与旧失败10/11结构一致：RAW_ref、phase、pin、returncode1、admission、phase_binding、bootstrap、entry_source、spec、original_report_absent=True。GPU03前门对该失败先terminal_raw_only失败1（abort=None）和check_pin，且与normal/旧失败8及独立失败9/10/11/12区分，再允许pre_gate中的任何SOURCE SHA。随后SOURCE闭合实际GPU0失败owner、原spec/entry/ad、相位及原REPORT缺失，绝不将失败叶晋级成功。旧8+9+10+11+12规则与pins结构保持。

未来completed fullDEV GPU0后CPU本叶：normal110、total_failed_RAW_count13、total_terminal_identity_pin_count123；normal+旧失败8 pin字典仍118，其余五条额外失败各自独立pin。completed_GPU_source应绑定实际新GPU03成功叶，由ROOT提供完成GPU0后的报告/RAW/ad/Source引用，不能用失败13的source/report冒充completedGPU。ROOT_FIELDS不变，训练GPU107/CPU108、下一成功DEVGPU109 owner索引不变；CPU own110。实际SOURCE字段、全部闭合refs与原退出0由ROOT绑定。

其余必需SPEC/Q/捕获/资源/原COCO合同全部沿用CPU01同stem.md。Q仍 `N16_ALL18700_SCOUT_FULLDEV_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0`，qualified=False/self-certified=False。adapter01继续使用实际GPUlogits定义排名、双CPU PR公式 absolute1e-4/rtol0/strictsign、little-packbits独立实际掩膜与自然18700 query/12策略完整证明；不改任何科学算法。

验证：本次AST_OK，GPU03纯前门与CPU01完整后缀均逐函数AST exact；GPU03作者已交付普通失败13 fake标准库门的4项合成回归通过，CPU01微型全100/12策略/K0及7拒绝测试保持适用。不重复实际科学或无关测试，不把结构复用称为本叶实际资格通过。
