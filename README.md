# DSTA 动态时空同步与城市时空数据立方体

本项目以 DAIR-V2X-Seq SPD-Example 真实车路协同观测为输入，实现“动态时空同步（Dynamic Spatio-Temporal Alignment，DSTA）”技术链，并构建统一的城市时空数据立方体。以局部通行异常候选事件 `EVT2_006` 为下游验证场景，演示从多源异构感知到统一时空数据流、再到事件态势认知的完整链路。

> **核心结论**：DSTA 将异步、异构、质量不一致的城市感知观测组织为可追溯、可校验、可交互的统一时空数据流。

---

## 研究背景与挑战

城市道路中的车端与路侧设备持续产生观测，但不同来源在以下方面存在显著差异：

- **采样节奏不一致**：帧率、采样周期、上报延迟不同；
- **空间表达不一致**：观测范围、坐标系、覆盖区域差异大；
- **数据质量不一致**：时间戳漂移、轨迹断点、网络抖动、缺失与乱序并存；
- **跨源关联困难**：同一目标在不同模态中没有统一身份标识。

因此，需要在上层事件认知之前，先建立**统一事件时间**和**统一世界坐标**的高质量时空数据底座。

---

## 技术路线：五项核心算法

| 步骤 | 算法 | 功能说明 | 关键指标 |
|---|---|---|---|
| 1 | 自适应时间对齐 | 基于数据到达延迟与采样频率动态调整事件时间窗口 | 正确配对率 67.33% → 98.02%；P90 时间误差 103.60 → 5.70 ms |
| 2 | 统一空间映射与多尺度网格 | 将多源观测映射到统一城市网格，支持质量感知自适应划分 | 固定/密度/质量感知平均量化误差 12.40/11.33/8.60 m |
| 3 | 乱序与缺失补偿 | 事件时间重排序 + 时空邻域插值 + 轨迹连续性约束 | 40% 缺失时 RMSE 1.332 → 0.179 m；恢复覆盖率 100% |
| 4 | 模态可信度动态加权 | 根据模态质量、历史可靠性动态计算融合权重 | 退化阶段等权/动态加权 RMSE 2.150/0.508 m |
| 5 | 时空一致性联合校验 | 时间、空间、跨模态三维一致性联合校验 | OR：FPR 0.130；联合方法：FPR 0.022 |

最终输出：**双粒度统一时空数据流**——观测级记录 + 网格窗口级状态，保留来源、质量标记与数据溯源信息。

---

## 项目结构

```
.
├── app.py                          # 城市交通态势回放（EVT2_006 交互回放）
├── apps/
│   └── dsta_cube.py                # DSTA 统一数据立方体与五项算法验证演示
├── data/
│   ├── raw/                        # 原始 DAIR-V2X-Seq 数据（未提交，需自行下载）
│   └── processed/                  # 中间与最终数据产物
│       ├── spatiotemporal_events.parquet
│       ├── city_state.parquet
│       ├── world_tracks.parquet
│       ├── optimized_grid_cells.csv
│       └── ...
├── scripts/                        # 数据构建、训练、评估、可视化脚本
│   ├── build_frame_pairs.py        # 车/路帧对构建
│   ├── build_spatiotemporal_events.py
│   ├── build_city_state.py         # 网格窗口状态聚合
│   ├── build_quality_grid.py       # 质量感知网格
│   ├── detect_anomalies.py         # 异常状态检测
│   ├── detect_and_group_events.py  # 事件发现与聚合
│   ├── train_state_autoencoder.py  # 状态自编码器
│   └── ...
├── src/dsta/                       # 核心算法包
│   ├── cube_data.py                # 统一数据立方体构建
│   ├── grid_evaluation.py          # 网格评估
│   ├── quality_quadtree.py         # 质量感知四叉树
│   └── ...
├── outputs/
│   ├── figures/                    # 分析图表（未提交）
│   ├── models/                     # 训练好的模型（未提交）
│   ├── reports/                    # 评估报告与 CSV
│   └── ppt_case/                   # PPT 案例素材、验收报告、Storyboard
├── experiments/                    # 受控扰动与验证实验
├── requirements.txt
└── README.md
```

---

## 数据产物规模

- **真实统一观测记录**：8,327 条
- **网格窗口状态**：830 条
- **自适应网格数**：186 个
- **时间切片数**：13 个
- **官方协同帧对**：249 组
- **下游验证候选事件**：`EVT2_006`（7 个异常状态 / 2 个网格 / 112 条观测记录）

---

## 环境依赖与启动方式

```bash
# 1. 创建并激活虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 DSTA 统一数据立方体与算法验证演示
streamlit run apps/dsta_cube.py

# 4. 启动 EVT2_006 城市交通态势回放
streamlit run app.py
```

依赖清单：`numpy`、`pandas`、`scipy`、`pyproj`、`h3`、`matplotlib`、`plotly`、`pyarrow`、`pytest`、`streamlit`。

---

## 关键可交付物

| 类型 | 路径 | 说明 |
|---|---|---|
| 交互演示 | `apps/dsta_cube.py` | 真实数据立方体 + 四项受控扰动实验 + 五项指标概览 |
| 事件回放 | `app.py` | EVT2_006 局部/全局视角动态回放、证据回溯 |
| 图片索引 | `outputs/ppt_case/final_figure_index.md` | 冻结的图片编号、来源、口径、指标与图注 |
| 汇报规划 | `outputs/ppt_case/ppt_storyboard.md` | 8 页 PPT 结构、讲述词与真实性说明 |
| 验收报告 | `outputs/ppt_case/final_acceptance_report.md` | 最终验收结论与限制项 |
| 立方体主图 | `outputs/ppt_case/cube/dsta_cube_ppt.png/svg` | 统一时空数据立方体 PPT 主图 |
| 算法图组 | `outputs/ppt_case/time_alignment/`、`spatial_mapping/`、`dsta_validation/` | 五项算法的 PPT 图组 |

---

## 主要脚本速查

```bash
# 构建帧对与世界坐标轨迹
python scripts/build_frame_pairs.py
python scripts/build_world_tracks.py

# 构建统一时空事件与城市网格状态
python scripts/build_spatiotemporal_events.py
python scripts/build_city_state.py

# 训练状态自编码器并检测异常
python scripts/train_state_autoencoder.py
python scripts/detect_anomalies.py

# 事件发现与认知
python scripts/detect_and_group_events.py
python scripts/build_event_cognition.py

# 可视化
python scripts/plot_world_tracks.py
python scripts/plot_temporal_analysis.py
python scripts/plot_city_state.py
```

---

## 真实性与局限性说明

- **受控仿真**：时间对齐中的网络延迟与抖动、乱序缺失补偿中的缺失/乱序、可信度加权中的双通道噪声、一致性校验中的异常标签均为明确标注的受控扰动，不是现场网络或真实事故标签。
- **候选关联**：一致性校验中的 815 条匹配与 EVT2_006 的下游回溯均为候选关联，不是跨源身份真值。
- **事件定位**：`EVT2_006` 是候选交通模式异常，不是已确认事故或拥堵；相对优先级也不是事故概率。
- **坐标说明**：可视化中的 X/Y 为数据集世界坐标（非地理经纬度），T 为相对事件时间（非地理高度）。

---

## 作者与致谢

本项目围绕“多源异构感知数据动态时空同步与城市时空数据立方体构建”技术攻关展开，实验数据来自 [DAIR-V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq)。
