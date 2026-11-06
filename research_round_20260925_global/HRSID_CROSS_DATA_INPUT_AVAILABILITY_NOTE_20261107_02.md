# HRSID配套输入仍缺：指定目标与三份JSON的有界核对

2026-11-07。**旧图像目录指向的目标仅有358个JPG，名称与本地完整split直接匹配0，不能直接作为完整HRSID实例基线输入。** 上一份目录报告保留为历史快照；本次按扩展后的明确范围，只检查指定目标一层元数据，依次读取三份已知JSON（每份小于32MiB，前后stat稳定），未读取或哈希影像、解码像素/mask、导入科学库、SSH或训练。

| 本次实际核查 | 结果 |
|---|---|
| 目标直接影像文件 | 358个JPG，共66,018,309字节；无子目录或待跟随link |
| train JSON | 3642图、11047实例、1类；全部实例为结构合法的数值polygon |
| test JSON | 1962图、5922实例、1类；全部实例为结构合法的数值polygon |
| merged JSON | 5604图、16951实例、1类；全部实例为结构合法的数值polygon |
| train/test文件名关系 | 交集0，并集5604；与目标basename精确匹配均为0，缺5604 |
| split内部ID | 各自imageID、annotationID唯一；跨split分别复用1962、5922个ID |

两split合计16969实例，仍与merged的16951不等；不能静默用merged代替split。跨split裸ID复用不代表重复图，加载应分别保留split命名空间。JSON声明的尺寸也不证明358个现存文件的实际像素尺寸；仅文件名存在或结构合法polygon，均不证明原始编码、mask可解码、空间对齐、官方release或母景独立。JSON的license条目也没有补足外部包逐文件许可。

已有本地JPG命名主归档和两组图像ZIP/RAR仍在。最快下一步是另立有限的**本地归档成员目录检查**，确认是否含配套5604图；若相符，再解到新私有目录、核原字节及像素/mask对齐。当前不必把再次下载整份PNG作为唯一道路，也不改名或改后缀挪用这358个文件。历史TEST选型使用与既有重复证据保持，不因恢复原图而成为盲测。

本次没有新增AP、跨集泛化或完整数据资格结论。私有定位和前10个名称样本不发布；[本次安全聚合](results/HRSID_CROSS_DATA_INPUT_AVAILABILITY_SAFE_20261107_02.json)、[历史版本及split差异](HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)、[数据获取与许可范围](../docs/DATA_ACCESS.md)。
