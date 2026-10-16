# 固定实例配对的动作机会计数工具

状态：源码和可移植合成单元测试已完成，8项测试实际退出0；没有真实FIT全C采集、新训练或新AP。这里不包含私有接口清单、资格链和本机运维配置。

oracle.py仅依赖标准库fractions。调用者必须保证固定真实GT配对、相同候选C及固定PR实际动作身份；计数无法检查这些事实。负query的空集训练代理不能当作真实背景实例。

- beneficial_oracle：同at-most K下，固定PR可用有益动作的最优配对IoU。
- ideal_candidate_oracle：同一C内，理想翻转原FN/FP的配对IoU。
- router_outcome：路由实际有益/有害翻转及未改变选点的计数。
- threshold_crossings：以精确Fraction比较0.50至0.95的十个配对IoU阈值。

I/U为原交并计数，U=I+FN+FP，真实GT非空。可用有益FN追加A、FP移除R，t=min(K,A+R)，最优a=min(A,t)、r=t-a，IoU=(I+a)/(U-r)。证明见函数说明。空并集显式NA，计数不一致拒绝；该结论限定静态、独立且坐标唯一的翻转。不是AP上界、模型总体能力界、因果结论或新学术原语。

在本目录执行python -m unittest -v test_oracle.py。8项测试含2750个小穷举场景，覆盖K0、空并集、动作可行性、路由误改及精确阈值等号；仅读取同目录实现，数据均为合成整数/Fraction，不读取真实数据、权重或运行资格输入。单元通过不能代替后续真实采集和独立核验。
