# DSTA 统一时空数据流 Schema

## 1. 粒度原则

统一数据流严格分为两个层级，不能混用：

1. **观测级 `unified_observations.parquet`**：一行代表某一来源、某一序列和帧中的一条目标观测，共 8,327 行。
2. **网格窗口级 `unified_cube_states.parquet`**：一行代表一个 `window_id × grid_id` 城市状态，共 830 行，主键唯一。

两表通过 `window_id + grid_id` 关联。观测表中的记录数不是独立车辆数，`track_id` 也只有在 `(source, sequence_id)` 内才具有轨迹身份意义。

## 2. 观测级 Schema

| 字段 | 类型/单位 | 来源分类 | 说明 |
|---|---|---|---|
| `event_time_us` | int64，μs | 真实原始字段重命名 | 等于 `timestamp_us`，事件时间 |
| `event_time_s` | float，s | 严格计算 | 相对窗口 0 起点的时间，用于立方体 T 轴 |
| `window_id` | int | 算法计算 | 按 `city_state.window_start_us` 映射，范围 0–12 |
| `grid_id` | string | 已有空间映射结果 | 质量感知自适应网格标识 |
| `source` | string | 真实原始字段 | `vehicle-side` 或 `infrastructure-side` |
| `sequence_id` / `frame_id` / `track_id` | string | 真实原始字段 | 来源内数据与轨迹标识 |
| `world_x` / `world_y` / `world_z` | m | 真实处理结果 | DAIR 数据集世界坐标，不是经纬度 |
| `arrival_timestamp_us` | int64，μs | 真实原始字段 | 当前文件中与事件时间一致，不能视作实测网络延迟 |
| `observation_count` | int | 严格计算 | 每行恒为 1，用于统一聚合接口 |
| `quality_flags` | string | 严格派生 | 由三个真实布尔质量标记组合，均为假时为 `observed_ok` |
| `is_missing` | bool | 真实处理结果 | 上游缺失标记 |
| `is_out_of_order` | bool | 真实处理结果 | 上游乱序标记 |
| `is_interpolated` | bool | 真实处理结果 | 上游插值标记 |
| `provenance` | string | 真实原始字段重命名 | 等于 `coordinate_provenance` |
| `record_granularity` | string | 严格计算 | 恒为 `observation` |

其余保留字段包括 `object_type`、网格中心坐标和原观测 ID。

## 3. 网格窗口级 Schema

| 字段 | 类型/单位 | 来源分类 | 说明 |
|---|---|---|---|
| `event_time_us` | int64，μs | 已有状态字段重命名 | 等于 `window_start_us` |
| `event_time_s` | float，s | 严格计算 | 相对窗口 0 的时间，当前为 0、2、…、24 s |
| `window_id` + `grid_id` | 复合主键 | 已有算法结果 | 830 行中唯一 |
| `world_x` / `world_y` | m | 已有聚合结果 | 对应 `mean_world_x` / `mean_world_y` |
| `x0` / `y0` / `size` | m | 已有空间映射结果 | 自适应网格真实范围 |
| `observation_count` | int | 已有聚合结果 | 窗口网格内观测记录数，不是车辆数 |
| `unique_local_tracks` | int | 已有聚合结果 | 按本地轨迹键统计，未跨源去重 |
| `mean_speed_mps` | m/s | 已有聚合结果 | 窗口网格平均速度 |
| `vehicle_observations` / `infrastructure_observations` | int | 已有聚合结果 | 两个真实来源的记录数 |
| `source_coverage_count` | int | 严格计算 | 当前网格窗口有数据的真实来源数量，取 1 或 2 |
| `both_sources_present` | bool | 已有聚合结果 | 两个来源是否均出现 |
| `quality_flags` | string | 严格聚合 | 观测级质量标记及数量的汇总 |
| `missing_count` / `out_of_order_count` / `interpolated_count` | int | 严格聚合 | 对应窗口网格内各类质量记录数 |
| `provenance` | string | 严格聚合 | 观测坐标来源集合 |
| `raw_is_anomaly` / `ae_is_anomaly` 等 | bool | 下游已有结果 | 仅作为下游应用验证属性，不属于技术攻关 1 核心输出 |
| `event_id` / `is_event_member` | string/bool | 下游已有结果 | 用于突出 `EVT2_006`，不是同步算法标签 |
| `record_granularity` | string | 严格计算 | 恒为 `grid_window_state` |

## 4. 真实、计算与仿真字段边界

- **真实或已有上游结果**：时间戳、来源、世界坐标、轨迹/帧标识、网格标识、质量布尔标记、城市状态统计。
- **本轮严格计算**：相对时间、窗口映射、来源覆盖数量、质量标记文本、粒度标签。
- **下游验证字段**：异常分支标记和事件成员 ID，仅用于证明统一数据可支撑事件感知。
- **仿真验证字段**：时间对齐实验中的延迟和时钟抖动只存在于 `outputs/ppt_case/time_alignment/`，没有混入统一真实数据表。
- **尚不存在**：模态可信度动态权重、补偿置信度、联合一致性分数。第一版 Schema 不虚构这些字段；待对应实验完成后再扩展。

## 5. 立方体轴定义

- X：`mean_world_x`，真实世界坐标米制 X。
- Y：`mean_world_y`，真实世界坐标米制 Y。
- T：`event_time_s`，相对窗口 0 的事件时间，不是地理高度。
- 基本体素：一条 `window_id × grid_id` 状态记录。
- 体素属性：观测数、平均速度、来源覆盖、质量汇总，以及仅供下游验证的异常/事件标记。

当前交互页面保留全部 830 个状态；局部视图只是视觉过滤，不改变或抽样交付数据。
