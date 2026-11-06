# DEV187 私有原字节传输包：本机准备状态（2026-11-07，尝试05）

状态：**`LOCAL_PRIVATE_ZIP186_ORIGINAL_BYTES_VERIFIED_NOT_TRANSFERRED`**。新独立预锁在本机源数据读取前保存；只做数据运输准备，不授37050端点预测资格或当前N28数组角色。

实际运输读取183份原JPEG对象，完整长度/SHA256再次闭合，原图总计 **7,118,253 字节**；保留全部187逻辑ID、原顺序、尺寸、0→1及expected380。ZIP使用STORED，**186项**（183 images + 原GT bytes + 新未绑定远端runtime manifest + 原local candidate manifest），**186项**逐成员EOF/CRC/完整SHA256回读通过，最终ZIP字节数 **7,439,654**。

原GT118,319字节按原SHA固定，只运输原bytes，未重新解析或序列化；复用实际04的187人口/尺寸/顺序桥。原candidate03文件未改，原bytes另存入包；新runtime manifest仅将图像和GT路径改成Root指定远端根，其余schema和逻辑内容保持。原始源文件读取前后状态稳定：True。原GT和原candidate与对应ZIP成员的完整SHA一致，证据仅私有保存。

**未创建、探测或传输到DGX远端根，未SSH；远端新根是否已占用须Root在接收时另核。** 该包只在本机私有保存，不证明N28角色、37050模型资格、像素/标注质量、官方TEST独立性或许可。没有图像header/pixel load、重编码、GTmask、Torch、预测、AP或ops改动。此前定位过程实际读过带标注ID映射的元数据容器，本次也明确读取原GT字节，不能声称流程没有GT相关访问。

耗时 0.156 秒，CPU 0.172 秒，子进程峰值工作集 22.29 MiB。单工作线程、BELOW_NORMAL、可用时CPU2/3、512MiB Job硬限额及300秒外部时限；原图单件2MiB/总量128MiB上限。原包、逐成员SHA清单、两个manifest、GT与locators/private builder不上传GitHub，公开仅本状态与聚合。 实际工具c46b01/0退出码0；阻塞监督等待子进程后传播成功码，工具没有存活session。资源值仅为子进程Windows peak working set/pagefile，未测监督RSS/全机内存，未独立退出后扫PID。

[公开聚合记录](results/DEV187_RAW_INPUT_TRANSPORT_AGGREGATE_20261107_05.json)。
