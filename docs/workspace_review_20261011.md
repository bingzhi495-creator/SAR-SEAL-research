# 工作区文件盘点与发布边界（2026-10-11）

本次仅枚举文件名并读取大小元数据，没有读取数据、模型、结果JSON、日志内容、档案内部或凭据，没有运行科研计算，也没有移动、删除或修改现有文件。用途分类由目录名和扩展名推断，不能替代源码审查、实验核验或许可审核。

枚举范围为本工作区；先使用 `rg --files --hidden`，再以 `--no-ignore` 补全忽略项，排除所有 `.git` 内部文件。2026-10-11 06:21:56 UTC开始采集，06:29:21 UTC完成长路径补查。记录 **259,758个文件、41个顶层目录/根文件组、534.01 GiB逻辑容量**（573,388,076,922字节）。第一遍2,523项Windows长路径失败，使用扩展路径补查后全部取得大小；最终元数据错误0，枚举错误输出0。逐文件路径、大小、分类提示见 [CSV清单](workspace_inventory_20261011.csv)。

## 顶层目录统计

“主要扩展名”按文件数排序取前三；大小按逻辑字节累计。

| 目录 | 文件数 | 逻辑容量 | 主要扩展名及数量 |
|---|---:|---:|---|
| `runs` | 3,519 | 248.11 GiB | .log 711；.pth 689；.json 604 |
| `research_round_20260925_global` | 168,561 | 206.30 GiB | .npz 70575；.png 46105；.json 21536 |
| `refs` | 11,323 | 44.10 GiB | .py 4739；.txt 4479；.yaml 494 |
| `research_round_20260920` | 14,330 | 11.50 GiB | .npz 9401；.gz 1981；.jpg 1182 |
| `delivery` | 49,518 | 9.07 GiB | .json 18902；.py 15764；.md 7177 |
| `（根目录文件）` | 207 | 5.11 GiB | .sh 117；.py 43；.md 16 |
| `pretrained` | 11 | 4.77 GiB | .pth 11 |
| `outputs` | 5,120 | 3.47 GiB | .py 2229；.pyc 2188；[none] 282 |
| `tmp` | 1,599 | 486.88 MiB | .pyc 432；.py 380；.xsd 256 |
| `机械臂ROS入门教程-小五` | 3 | 410.95 MiB | .rar 2；.zip 1 |
| `paper_tgrs` | 634 | 320.38 MiB | .png 341；.tex 71；.pdf 64 |
| `data` | 387 | 185.78 MiB | .jpg 358；.json 18；.jsonl 4 |
| `smartfrost_v4` | 1,525 | 121.95 MiB | .js 1092；.step 371；.json 19 |
| `work_dirs` | 1,961 | 24.42 MiB | .py 710；.log 632；.json 619 |
| `smartfrost_integrated` | 110 | 20.18 MiB | .png 45；.cjs 37；.md 10 |
| `node_modules` | 183 | 17.69 MiB | .js 65；.md 26；.txt 10 |
| `SmartFrost_完整包_20260923` | 33 | 9.62 MiB | .cjs 8；.md 6；.png 6 |
| `paper` | 73 | 4.43 MiB | .log 17；.pdf 14；.tex 8 |
| `docs` | 64 | 2.79 MiB | .md 57；.pdf 6；.txt 1 |
| `build_pdf` | 43 | 1.64 MiB | .log 16；.tex 7；.pdf 7 |
| `carrier` | 267 | 1.35 MiB | .py 216；.pyc 51 |
| `SAR_maskrcnn_ssdd` | 34 | 0.34 MiB | .py 13；.log 12；.json 9 |
| `diagnostics` | 35 | 0.23 MiB | .py 27；.pyc 8 |
| `benchmark` | 17 | 0.22 MiB | .tex 4；.log 3；.py 3 |
| `SAR_point_rend_ssdd` | 19 | 0.20 MiB | .py 7；.log 6；.json 6 |
| `SAR_instaboost_ssdd` | 19 | 0.19 MiB | .py 7；.log 6；.json 6 |
| `SAR_dcn_ssdd` | 19 | 0.19 MiB | .py 7；.log 6；.json 6 |
| `SAR_yolact_ssdd` | 19 | 0.19 MiB | .py 7；.log 6；.json 6 |
| `SAR_solo_ssdd` | 25 | 0.18 MiB | .py 9；.log 8；.json 8 |
| `tools` | 29 | 0.16 MiB | .py 14；.ps1 11；.pyc 4 |
| `protocol` | 16 | 0.06 MiB | .py 9；.pyc 7 |
| `.pytest_cache` | 6 | 0.05 MiB | [none] 4；.md 1；.tag 1 |
| `refine-logs` | 9 | 0.04 MiB | .md 9 |
| `mne` | 8 | 0.03 MiB | .pyc 5；.py 3 |
| `autodl_package` | 8 | 0.03 MiB | .py 5；.sh 2；.md 1 |
| `autodl` | 11 | 0.02 MiB | .sh 9；.md 2 |
| `seeding` | 5 | 0.02 MiB | .py 3；.pyc 2 |
| `losses` | 3 | 0.01 MiB | .py 2；.pyc 1 |
| `__pycache__` | 2 | 0.01 MiB | .pyc 2 |
| `automation` | 1 | 0.00 MiB | .py 1 |
| `smartfrost_v3` | 2 | 0.00 MiB | .cjs 1；.js 1 |

容量主要集中在 `runs` 与两个 `research_round`。全工作区扩展名中，`.pth`为2,464个、462.53 GiB，约占逻辑容量86.6%；数量最多的是`.npz`80,004个、`.png`47,144个、`.json`42,606个和`.py`36,159个。扩展名无法确认这些文件的实际内容或是否仍被使用。

## 用途评议与GitHub发布范围

仅按扩展名计，`research_round_20260920`有246个`.py`、2,254,688字节；`research_round_20260925_global`有11,104个`.py`、126,959,921字节。这包含历史副本、打包依赖等可能项，不能直接当作待发布的维护源码规模。

- **核心源码候选**：`carrier`、`protocol`、`tools`、`losses`、`seeding`、`automation`；当前研究实现还分散在`research_round`。README应指定唯一维护入口、配置与依赖，区分现用源码、未执行候选和历史版本。
- **基线与参考**：`refs`、`SAR_*`、`benchmark`、`diagnostics`。外部实现应保留原来源和许可证；目录存在不能证明基线已复现，`SAR_*`中的配置/运行痕迹不能替代公平比较证据。
- **论文材料**：`paper`、`paper_tgrs`、`build_pdf`。源码、图稿和构建产物宜分别说明；文件名不能证明论文结论已验证。
- **数据与权重**：`data`、`pretrained`以及其他目录中的模型/数组/档案。GitHub宜提供来源、许可、下载方法和固定划分说明，按发布权限决定是否提供单独资产；本次没有审核来源、真实性或授权。
- **历史实验与交付**：`research_round_*`、`runs`、`outputs`、`work_dirs`、`delivery`。这些保存研究记录、快照和包的可能性高，但不能由名字判断成功、失败或冗余。README应把可运行最小源码与历史档案分开，并说明各快照对应关系；本次没有去重或清理。
- **缓存与独立项目**：`tmp`、`node_modules`、`__pycache__`、`.pytest_cache`候选为依赖/缓存；SmartFrost、机械臂ROS教程属于其他工程项目。`mne` 是本项目历史证据场组件，不能仅凭缩写误列为独立项目。发布时应明确各组件是否为当前主线依赖，避免让用户误装整套工作区。目录名称不是删除依据。

README最小结构建议：研究问题与当前证据范围 → 唯一维护源码地图 → 环境/数据获取与许可 → 固定运行及评估入口 → 基线来源与比较限制 → 已完成和待验证事项 → 历史档案说明。不能把全目录上传、多个准备脚本或大量输出文件等同于“全部可复现”，也不能由这次盘点判断为何没有可靠增益。

## 覆盖限制

清单是逐项采集的非原子快照，运行中的任务可能继续新增或改变文件；扫描期间另新增的`README.md`、`RESEARCH_GOAL.md`不保证纳入该快照，本次新建的CSV与本评议也不计入上述数量。容量未按文件内容去重，可能重复累计交付镜像、硬链接与压缩包；它不是磁盘实际占用。档案内部、空目录、`.git`以及工作区外被源码引用的数据/权重不在清单内；未跟随目录链接。凭据内容未检查，文件名盘点不是秘密扫描。所有用途标签均为提示，本次不认证学术有效性、当前版本权威性或可复现性。
