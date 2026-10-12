# AutoDL turnkey 部署：GeoKernel M3 ShapePrior 6-run pilot

你只做 4 步：①租实例 ②上传一个 tar ③跑 setup ④跑 pilot，然后把 `results/` 发我裁决。
**我无法替你登录/租用/付费 AutoDL**——下面让这件事尽量一键。

## 0. 要传什么（很小，~52MB，无需 510MB 权重）
- 代码（carrier/ protocol/ mne/ seeding/ diagnostics/ + 配置）
- PSeg 数据：标注 1.1M + 图像 ~51M（pilot 从配方训，backbone 自动下 torchvision resnet50 ~100MB，实例需联网）

## 1. 本地打包（在本机 E:\SAR\seal 跑）
```bash
bash autodl/pack_for_autodl.sh      # 生成 seal_autodl.tar.gz (~52MB)
```

## 2. 租 AutoDL 实例
- 镜像：选 **PyTorch 2.0.1 / CUDA 11.8 / Python 3.10**（与本机 torch 2.0.1+cu118 对齐）。
- GPU：1× 4090/3090 即可；**租 2-3 卡可并行更快**（6 个 pilot 自动分到各卡）。
- 把 `seal_autodl.tar.gz` 上传到实例的 `/root/autodl-tmp/`（数据盘，重启不丢）。

## 3. 服务器端 setup（一次）
```bash
cd /root/autodl-tmp && tar xzf seal_autodl.tar.gz      # 解到 /root/autodl-tmp/seal
cd /root/autodl-tmp/seal && bash autodl/setup_env.sh   # 装 mmcv/mmdet + 改路径
```
`setup_env.sh` 做：pip 装 mmengine==0.10.4 mmcv==2.1.0 mmdet==3.3.0；把配置里 `E:/SAR/seal`→当前根、`E:/PSeg_SSDD`→`<根>/PSeg_SSDD`。

## 4. 跑 pilot（6 个 8 轮实验，自动分卡并行）
```bash
cd /root/autodl-tmp/seal && bash autodl/run_pilot.sh
```
跑完结果在 `results/pilot_metrics.json`。打包发我：
```bash
bash autodl/pull_results.sh         # 生成 results.tar.gz，下载发我
```

## 6-run pilot 是什么（已设计好，见 docs/GEOKERNEL_variants_full.md）
| Run | 配置 | 作用 |
|---|---|---|
| R0 | a0r_refine_pseg | rank-1 基线（地板）|
| R1 | sp_static_pseg | M3 真静态（容量地板）|
| R2 | sp_blind_pseg | M3 几何盲（content 对照）|
| R3 | sp_geo_pseg | M3 几何（机制）|
| R4 | sp_geo_sar_pseg | M3 几何 + SAR 损失（最想赢的格）|
| R5 | a0r_refine_pseg + SAR损失 | 仅损失效应 |

**裁决（我来做）**：GEO(R3) 是否明显甩开 BLIND(R2)，且 ≥0.7×(BLIND−STATIC)？甩开→升 36/50 轮完整训练；打平→转"等参对照表审计"framing。

## A800-80G 优化（你打算租的卡）
- **单卡也能并发**：80GB ÷ ~20GB/job = **4 个 arm 同时跑** → 5 个 pilot 约一波（~1h）跑完，不必串行。`run_pilot.sh` 已自动按显存算并发（可 `PER_JOB_GB=24 bash autodl/run_pilot.sh` 调）。
- **R0 交叉校验**：pilot 的 R0(rank-1) 用本机同配方，A800 上应复现 ~0.615 AP75 → 验证移植正确。

## pilot 过了之后：完整训练（已备好，参数化）
```bash
# 36ep × 3 seed × {geo, blind, static, baseline}（A800-80G 4 并发，~9-12h）
ARMS="sp_geo_pseg sp_blind_pseg sp_static_pseg a0r_refine_pseg" SEEDS="0 1 2" bash autodl/run_full.sh
# 50ep headline（仅胜出格）
FULL_ITERS=11600 ARMS="sp_geo_sar_pseg" SEEDS="0 1 2" bash autodl/run_full.sh
```
结果在 `results/full_metrics.json`（含每 arm 3-seed AP75 均值）。发我 → 我用 bootstrap 出 CI + headline 裁决。

## 我能怎么进一步帮你
- 你把 `results.tar.gz` 发我 → 我出 GO/NO-GO 裁决 + 下一步（完整训练配置 / 3-seed 矩阵）。
- 若你愿意把实例的命令输出贴给我，我可逐步带你跑、调错。
