# N26 冻结两GT输入：私有本机运输状态（2026-11-07）

状态：**`LOCAL_PRIVATE_FROZEN_N26_ZIP12_BYTES_VERIFIED_NOT_TRANSFERRED`**。新独立预锁后，仅消费固定独立ledger的共同R、三个预锁seed完整quality predictions、三份score audit、两GT、float mapping与原ledger。源字节不改，不换R/seed，不重训或重算qhat。

ZIP_STORED实际 **12成员**，**12成员**逐项EOF/CRC/完整SHA256回读闭合，包字节数 **36,998,025**。原始输入读取 **36,990,939字节**（单输入≤16MiB、源输入总量≤64MiB），原pins与读取前后stat一致：True。运输包结构固定为4预测、3audit、2GT、mapping、原ledger和新runtime refs，共12项。

**本次实际读取了原GT/预测/score/audit JSON的原始bytes作hash/打包，但没有解析这些数值正文、没有计算指标。** 仅解析独立locator ledger metadata；GT与全部预测/score原样入ZIP，未重新序列化。新未绑定runtime_refs只将各role ref路径映射到Root指定新远端根，bytes/SHA和各seed旧AP/score_rule声明保持，不产生新AP或科学资格。

此前定位阶段实际读过三个已停止旧N26 SPEC及其三份直接EVAL_REPORT metadata，此范围保留；不能声称未读实际执行metadata或未访问GT相关数据。本次没有跟多级qualification/capture/package链，没有NumPy/Torch/COCO、像素、预测forward、AP、SSH/HTTP、activeSource/weights或ops改动，也不冒用native37050适配器。

**未创建远端根、未传输至DGX。** Root须另核远端新根占用、做有界传输/接收核验，并为科学诊断另立prelock/source资格。原整数主结果与三seed保留，不挑最好seed、不选择有利GT；浮点N26尚无新结果。

耗时 0.187秒，CPU 0.172秒，子进程峰值工作集 20.37MiB。单工作线程、BELOW_NORMAL、可用时CPU2/3、Job512MiB硬限额、外部300秒时限。原包、payload、逐成员hash、runtime refs、locators和private builder仅私有保存，不上传GitHub。 实际工具2a7a82/0退出码0；阻塞监督等待子进程后传播成功码，工具无存活session。内存仅为子进程Windows peak working set/pagefile，未测监督RSS/全机内存，未独立退出后扫PID。

[公开聚合记录](results/N26_FROZEN_SENSITIVITY_TRANSPORT_AGGREGATE_20261107_01.json)。
