# DSTA 时空一致性联合校验

## 口径

真实部分是 DAIR 官方帧对、候选匹配的事件时间、世界坐标和同轨迹历史连续性残差。异常标签全部是独立测试副本上的 **synthetic anomaly labels**。候选匹配并非跨源身份真值，因此本实验验证的是受控扰动下的校验能力，不宣称真实身份匹配准确率。

正常校准集只用于计算 99% 单维阈值、稳健中位数/MAD 尺度和联合分数阈值；测试标签未参与调参。联合方法对时间、空间、轨迹残差分别稳健归一化，再按每条记录实际可用的维度计算 RMS 分数。轨迹缺失被显式保留。

## 中等强度结果

| 方法 | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| 单时间阈值 | 0.905 | 0.526 | 0.665 | 0.081 |
| 单空间阈值 | 0.940 | 0.515 | 0.665 | 0.049 |
| 时间/空间 OR | 0.898 | 0.776 | 0.832 | 0.130 |
| 多维联合 | 0.973 | 0.529 | 0.686 | 0.022 |

## 文件

- `consistency_validation_ppt.png` / `.svg`：PPT 主图（PNG 300 DPI）。
- `consistency_metrics.csv`：Precision、Recall、F1、FPR 与混淆计数。
- `consistency_class_recall.csv`：各异常类型检出率。
- `consistency_sensitivity.csv`：低/中/高扰动强度敏感性结果。
- `consistency_samples.parquet`：三档扰动的逐点结果。
- `cube_consistency_layer.parquet`：中等强度独立实验层，不覆盖统一真实数据表。
- `consistency_validation_config.json`：固定种子、切分、扰动和阈值配置。
- `data_audit.md`：真实键关联及可用性审查。

## 限制

空间候选在生成时已受 5 m 门限截断，不能代表无条件真实配对分布；正常参考也可能包含误匹配。轨迹残差仅在已有历史点足够且时间间隔合理时可用。所有性能数字只适用于所声明的受控异常强度，不能解释为真实事故检出率。
