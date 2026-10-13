# N13 第二种子冷启动与48轮接续源码状态

2026-10-13 08:06（Asia/Shanghai）快照。第二种子首段GPU训练已启动；本文不是终止报告或效用结果。

## 当前已执行与未完成的工作

种子701原GPU任务在两臂完整8轮产物保存后、收尾报告处实际退出1。新独立CPU恢复实际退出0，ROOT闭合完整41条正常与7条失败来源、8740项来源文件后采用两个E08检查点。原GPU失败、缺失报告和九件产物保持原状；恢复没有增加训练更新，不代表AP收益。

第二种子202610120702的LOG_PR/GEO_PR已通过ROOT准备并启动原持有任务41567。新cold v03入口仅允许702/703的1–8轮；准备绑定8753项来源，历史41正常/7失败及48个身份固定值。每臂预锁13,288更新，最终仍为48轮、79,728更新和三个配对种子。当前尚无该任务实际终止码、端点CPU资格或AP。观察超时只等待原任务，不重启。

第三种子尚未启动；下一顺序为第二种子终止及独立CPU端点核验、第三种子冷启动及核验，再按三个种子交错推进9–16、17–24、25–32、33–40、41–48轮。

## 连续48轮源码：静态通过，未执行

新N13_continuous48_adopted_endpoint_training_source_20261013_04的train.py、qualify_train_cpu.py和ABI.md已停写。语法检查与一次独立静态审阅通过，ROOT已核对关键分支。训练配方、11个训练数值函数和9个CPU数值函数与独立cold v03发布文本完全一致。

成功的v3/v4端点按精确历史GPU0/CPU0引用定位，不要求处于当前历史尾部。种子701的原v2检查点只经指定CPU恢复v3与ROOT采用记录v1进入唯一采用分支；原GPU1永远保留在失败链。接续加载原AdamW状态、私有图像顺序RNG和累计历史；本轮SOURCE不会冷重置这些状态。正常来源至少45/46条和完整7条失败要求不变。

源码静态通过不等于运行或数值通过。实际接续仍需ROOT在每个原任务真实终止后，检查完整来源并单次绑定新准入。

## 固定完整DEV评价设计

完整DEV为历史187图/380GT，每臂保留18,700个自然query；原生分数、类别、顺序及空预测规则固定，不删预测、不按GT选query。8/24轮只作学习曲线，48轮为唯一主端点，不按DEV选择最好轮次或改预算。

新评价接口设计使用共享的真实GPU原生L载荷精确重建LOG/GEO同K区域，独立CPU以OpenCV几何及显式PR公式检查，再重算官方COCO指标。实际原图尺寸和资源需求尚需ROOT核查；800×800假设不是原图上界。评价源码正在编写，尚未执行、未取得新AP。

原full-L CPU allclose在6/741图失败的记录仍仅作诊断。当前BCE端点一致性、几何oracle和源码审阅均不能替代方法效用、跨数据集或TGRS投稿证据。

## 导航

- [第一种子实际恢复结果](N13_SEED701_E08_CPU_RECOVERY_ACTUAL_QUALIFIED_RESULT_20261013.md)
- [完整评价接口](N13_FIXED_DEV_EVALUATION_INTERFACE_CONTRACT_20261013.md)
- [冷启动ABI](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_qualified_endpoint_training_source_20261013_03/ABI.md)
- [连续48轮ABI](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_continuous48_adopted_endpoint_training_source_20261013_04/ABI.md)
