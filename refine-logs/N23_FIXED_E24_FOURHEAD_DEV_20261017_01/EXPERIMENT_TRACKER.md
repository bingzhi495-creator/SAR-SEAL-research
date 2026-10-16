# N23固定E24评价执行表

| Run | 范围 | 目的 | 状态 |
| --- | --- | --- | --- |
| GPU0 | [0,47) | 四头原Omega导出及逐query R校准 | ACTUAL0_ROOT_ACCEPTED |
| GPU1 | [47,94) | 同上 | ACTUAL0_ROOT_ACCEPTED |
| GPU2 | [94,141) | 同上 | ACTUAL0_ROOT_ACCEPTED |
| GPU3 | [141,187) | 同上 | ACTUAL0_ROOT_ACCEPTED |
| CPU完整COCO | 全187/380GT/18700query | 统一官方AP及固定门槛 | ACTUAL0_ROOT_ACCEPTED_STOP |

B67.807324，A_COMMON67.788929、A_ACTIVE67.603337，R68.099385，主Delta=-0.292061。预锁STOP，不加种子、预算、lambda调参或融合。详见[完整结果](../../research_round_20260925_global/N23_FIXED24_FULLDEV_RESULT_AND_STOP_20261017.md)。
