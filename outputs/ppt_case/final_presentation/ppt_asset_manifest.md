# DSTA 6 页汇报｜素材与指标来源

| 页码 | 主要素材 | 指标来源 | 真实性口径 |
|---|---|---|---|
| P01 | DAIR 原始车端 `010482.jpg`、路侧 `010699.jpg` | `world_tracks.parquet`、`frame_pairs.parquet` | 原始公开数据；并非消防、防汛现场数据 |
| P02 | `time_alignment_ppt.png`、`spatial_mapping_ppt.png` | 两目录下 `*_metrics.csv` | 时间扰动为受控注入；空间测试观测真实 |
| P03 | `missing_compensation_ppt.png`、`reliability_weighting_ppt.png` | 对应指标 CSV | 真实参考轨迹；缺失、乱序和双通道噪声为仿真 |
| P04 | `consistency_validation_ppt.png` | `consistency_metrics.csv` | 候选匹配不等于身份真值；异常为受控注入 |
| P05 | `dsta_cube_ppt.png` | 统一观测表、统一立方体状态表 | 真实标准化与严格聚合数据；T 非地理高度 |
| P06 | 可编辑证据矩阵与适配层 | 五项冻结实验 CSV | 跨实验指标不合成为端到端精度，不外推目标场景性能 |

## 冻结指标

- 时间：正确配对率 67.33%→98.02%；P90 103.60→5.70 ms。
- 空间：固定/质量感知平均量化误差 12.40→8.60 m；144/186 个网格，预算 ≤200。
- 缺失：40% 缺失时 Baseline/DSTA RMSE 1.332/0.179 m。
- 可信度：整体等权/动态加权 RMSE 1.168/0.450 m。
- 一致性：OR FPR/Recall/F1=13.0%/77.6%/83.2%；联合=2.2%/52.9%/68.6%。
