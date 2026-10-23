# DSTA 技术攻关 1｜最终验收报告

验收日期：2026-10-23  
最终结论：**通过，可交付。** DSTA 五项核心算法实验、双粒度统一数据流、三维时空立方体和下游 EVT2_006 验证已形成完整证据链。

## 1. 通过项

### 演示页面

- `apps/dsta_cube.py` 已增量整合五种独立场景：真实数据立方体、乱序缺失补偿、可信度动态加权、一致性联合校验、五项指标概览。
- 真实统一数据与实验数据分层读取，未向 `unified_observations.parquet` 或 `unified_cube_states.parquet` 写入实验字段。
- 页面明确说明补偿轨迹不是 EVT2_006 成员，双通道是受控仿真，815 条匹配是候选关联而非身份真值。
- 联合一致性结果同时展示误报率下降与召回率下降，不宣称所有指标全面提升。
- 真实三维立方体保留图层切换、全局/EVT2_006 局部视角、时间切片和数据 Schema 视图。

### PPT 图片与文档

- 五项算法的 PNG 和 SVG 主图全部存在。
- 立方体 PNG/SVG、真实轨迹输入图和下游事件演化图全部存在。
- `final_figure_index.md` 已冻结图片编号、来源、口径、指标、图注和真实性边界。
- `ppt_storyboard.md` 已冻结为 8 页结构，每页包含标题、核心结论、图形、指标、讲述词和真实性说明。

### 指标一致性

- 时间对齐：正确配对率 67.33%→98.02%，P90 时间误差 103.60→5.70 ms。
- 空间映射：固定/密度/质量感知平均量化误差 12.40/11.33/8.60 m；网格数 144/126/186，统一预算 ≤200。
- 乱序缺失补偿：40% 缺失时 Baseline/DSTA RMSE 1.332/0.179 m，DSTA 恢复覆盖率 100%。
- 可信度动态加权：退化阶段等权/动态加权 RMSE 2.150/0.508 m，10 个固定随机种子。
- 中强度一致性校验：OR 的 F1/FPR/Recall 为 0.832/0.130/0.776；联合方法为 0.686/0.022/0.529。
- 自动检查确认上述一致性指标与 CSV 和两份最终 Markdown 文档一致。

### 数据与运行

- 真实统一观测表：8,327 条；真实网格窗口表：830 条。
- Python 语法检查通过：`apps/dsta_cube.py`、`app.py`。
- Streamlit `AppTest` 首屏及五个场景均无未捕获异常。
- 乱序缺失场景的 10%、20%、30%、40% 交互路径可运行；已额外验证 40% 场景。
- 实际启动 `streamlit run apps/dsta_cube.py` 成功，健康检查返回 HTTP 200，随后正常停止服务。
- 原事件分析 `app.py` 保留；本轮没有修改上游算法、训练模型或 Step 1–8.1 数据结果。

## 2. 限制项（不影响交付）

- 时间对齐的网络延迟和抖动、缺失乱序、双通道噪声及一致性异常标签均为明确标注的受控扰动，不是现场网络或事故标签。
- 可信度实验是“真实参考轨迹 + 双通道仿真”，不能称为真实车路双源融合，也没有独立测绘级位置真值。
- 815 条空间匹配受同类别、5 m 门限和帧内匈牙利匹配约束，是候选关联而非身份真值。
- 联合一致性方法在中等强度下降低 FPR 的同时降低 Recall 和 F1，应定位为更保守的取舍。
- EVT2_006 是候选交通模式异常，不是已确认事故或拥堵；112 条为观测记录数，相对优先级不是事故概率。

## 3. 最终可交付文件

- 交互演示：`apps/dsta_cube.py`
- 原事件回放：`app.py`
- PPT 图片索引：`outputs/ppt_case/final_figure_index.md`
- 8 页汇报规划：`outputs/ppt_case/ppt_storyboard.md`
- 本验收报告：`outputs/ppt_case/final_acceptance_report.md`
- 统一 Schema：`outputs/ppt_case/cube/unified_data_schema.md`
- 立方体主图：`outputs/ppt_case/cube/dsta_cube_ppt.png`、`.svg`
- 时间对齐图组：`outputs/ppt_case/time_alignment/`
- 空间映射图组：`outputs/ppt_case/spatial_mapping/`
- 补偿、可信度和一致性图组：`outputs/ppt_case/dsta_validation/`

## 4. 启动命令

```bash
.venv/bin/streamlit run apps/dsta_cube.py
```

原事件分析页面：

```bash
.venv/bin/streamlit run app.py
```
