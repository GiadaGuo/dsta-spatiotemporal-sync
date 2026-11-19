"""Build the frozen six-slide DSTA validation presentation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ppt_case/final_presentation"
PPTX_PATH = OUT / "DSTA_动态时空同步算法验证汇报.pptx"

BG = "071426"; PANEL = "0C2038"; PANEL_2 = "102B48"; CYAN = "00C8FF"
ORANGE = "FF9738"; PURPLE = "A58BFF"; WHITE = "F4F8FC"; MUTED = "AFC2D5"
GRID = "244766"; DARK = "06101E"; YELLOW = "F7C948"
# PostScript name is used because LibreOffice headless does not reliably resolve
# the macOS display-family name during preview rendering.
FONT = "HiraginoSansGB-W3"
W, H = 13.333, 7.5


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


def set_bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid(); fill.fore_color.rgb = rgb(color)


def add_rect(slide, x, y, w, h, fill=PANEL, line=GRID, radius=True, transparency=0):
    shape_type = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    s = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill); s.fill.transparency = transparency
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(.8)
    else:
        s.line.fill.background()
    return s


def add_text(slide, text, x, y, w, h, size=18, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE, font=FONT,
             margin=.05, fit=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = text
    run.font.name = font; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = rgb(color)
    if fit:
        tf.fit_text(font_family=font, max_size=size)
    return box


def add_rich_text(slide, parts, x, y, w, h, size=16, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(.04); tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = align
    for text, color, bold, pt in parts:
        r = p.add_run(); r.text = text; r.font.name = FONT; r.font.color.rgb = rgb(color)
        r.font.bold = bold; r.font.size = Pt(pt or size)
    return box


def add_title(slide, number, title, kicker="DSTA 动态时空同步算法 · 能力验证"):
    add_text(slide, kicker, .45, .16, 7.2, .26, 9.5, CYAN, True)
    add_text(slide, title, .45, .46, 11.9, .55, 27, WHITE, True, valign=MSO_ANCHOR.TOP)
    add_text(slide, f"P{number:02d}", 12.2, .20, .65, .28, 10, MUTED, True, PP_ALIGN.RIGHT)
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(.45), Inches(1.04), Inches(12.42), Inches(.018))
    line.fill.solid(); line.fill.fore_color.rgb = rgb(GRID); line.line.fill.background()


def add_footer(slide, text):
    add_text(slide, text, .48, 7.18, 12.35, .18, 8.5, MUTED, False, valign=MSO_ANCHOR.MIDDLE)


def add_conclusion(slide, text, y=6.55, color=CYAN):
    add_rect(slide, .45, y, 12.42, .48, fill="0A2941", line=color)
    add_text(slide, text, .68, y + .05, 11.95, .36, 14, WHITE, True, PP_ALIGN.CENTER)


def add_picture_contain(slide, path: Path, x, y, w, h, line=GRID):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih); pw, ph = iw * scale, ih * scale
    left, top = x + (w - pw) / 2, y + (h - ph) / 2
    add_rect(slide, x, y, w, h, fill=PANEL, line=line, radius=True)
    pic = slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(pw), Inches(ph))
    return pic


def add_picture_cover(slide, path: Path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    target = w / h; source = iw / ih
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    if source > target:
        crop = (1 - target / source) / 2; pic.crop_left = pic.crop_right = crop
    elif source < target:
        crop = (1 - source / target) / 2; pic.crop_top = pic.crop_bottom = crop
    return pic


def label(slide, text, x, y, w, color=CYAN):
    add_rect(slide, x, y, w, .28, fill=color, line=None)
    add_text(slide, text, x + .05, y + .015, w - .10, .23, 10, DARK, True, PP_ALIGN.CENTER)


def metric(slide, value, caption, x, y, w, accent=CYAN, value_size=31):
    add_text(slide, value, x, y, w, .46, value_size, accent, True, PP_ALIGN.CENTER)
    add_text(slide, caption, x, y + .44, w, .25, 10.5, MUTED, False, PP_ALIGN.CENTER)


def load_metrics():
    timing = pd.read_csv(ROOT / "outputs/ppt_case/time_alignment/time_alignment_metrics.csv").set_index("method")
    spatial = pd.read_csv(ROOT / "outputs/ppt_case/spatial_mapping/spatial_mapping_metrics.csv").set_index("display_name")
    missing = pd.read_csv(ROOT / "outputs/ppt_case/dsta_validation/missing_compensation/missing_compensation_metrics.csv")
    reliability = pd.read_csv(ROOT / "outputs/ppt_case/dsta_validation/reliability_weighting/reliability_metrics_summary.csv")
    consistency = pd.read_csv(ROOT / "outputs/ppt_case/dsta_validation/consistency_validation/consistency_metrics.csv")
    return timing, spatial, missing, reliability, consistency


def slide_1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    add_title(slide, 1, "多源感知时空同步：从城市高风险需求到可复现算法验证")
    # Left: target scenes
    add_rect(slide, .45, 1.27, 3.05, 5.08)
    label(slide, "目标场景", .67, 1.48, .92, CYAN)
    add_text(slide, "城市高风险感知", .68, 1.86, 2.55, .34, 18, WHITE, True)
    scenes = [("01", "消防应急", "视频流 · 通信告警"), ("02", "防汛排涝", "水情 IoT · 北斗轨迹"), ("03", "人群管控", "视频 · 信令 · 定位")]
    for i, (n, title, sub) in enumerate(scenes):
        yy = 2.38 + i * .93
        add_rect(slide, .68, yy, 2.58, .73, fill=PANEL_2, line=GRID)
        add_text(slide, n, .80, yy + .11, .40, .30, 15, CYAN, True, PP_ALIGN.CENTER)
        add_text(slide, title, 1.28, yy + .07, 1.70, .26, 15, WHITE, True)
        add_text(slide, sub, 1.28, yy + .36, 1.72, .20, 9.5, MUTED)
    add_text(slide, "目标数据：视频 / 通信 / 北斗 / IoT", .68, 5.53, 2.56, .42, 11, ORANGE, True, PP_ALIGN.CENTER)
    # Middle: common problems
    add_rect(slide, 3.70, 1.27, 3.15, 5.08)
    label(slide, "共性问题", 3.92, 1.48, .92, ORANGE)
    add_text(slide, "跨模态差异，归结为五类同步挑战", 3.93, 1.86, 2.66, .38, 14.5, WHITE, True)
    issues = [("T", "时间异步"), ("XY", "空间异构"), ("↕", "乱序缺失"), ("Q", "质量波动"), ("✓", "一致性风险")]
    for i, (icon, text) in enumerate(issues):
        yy = 2.36 + i * .67
        circle = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(3.96), Inches(yy), Inches(.43), Inches(.43))
        circle.fill.solid(); circle.fill.fore_color.rgb = rgb("143B58"); circle.line.color.rgb = rgb(CYAN)
        add_text(slide, icon, 3.96, yy + .02, .43, .36, 11, CYAN, True, PP_ALIGN.CENTER)
        add_text(slide, text, 4.55, yy + .02, 1.78, .36, 14, WHITE, True)
    add_text(slide, "机制可复用 ≠ 场景效果可直接外推", 3.96, 5.81, 2.60, .28, 10.5, ORANGE, True, PP_ALIGN.CENTER)
    # Right: real DAIR frame pair
    add_rect(slide, 7.05, 1.27, 5.82, 5.08)
    label(slide, "真实验证 CASE", 7.28, 1.48, 1.30, PURPLE)
    add_text(slide, "DAIR-V2X-Seq · 官方协同帧对", 7.30, 1.82, 5.20, .34, 16, WHITE, True)
    veh = ROOT / "data/raw/V2X-Seq-SPD-Example/vehicle-side/image/010482.jpg"
    inf = ROOT / "data/raw/V2X-Seq-SPD-Example/infrastructure-side/image/010699.jpg"
    add_picture_cover(slide, veh, 7.30, 2.25, 2.52, 1.42); label(slide, "车端 010482", 7.39, 3.36, .94, CYAN)
    add_picture_cover(slide, inf, 10.02, 2.25, 2.52, 1.42); label(slide, "路侧 010699", 10.11, 3.36, .94, ORANGE)
    metric(slide, "8,327", "真实观测", 7.28, 4.08, 1.55, CYAN)
    metric(slide, "249", "官方协同帧对", 9.08, 4.08, 1.55, ORANGE)
    metric(slide, "2", "真实来源", 10.88, 4.08, 1.55, PURPLE)
    add_text(slide, "真实时间戳 · 世界坐标 · 轨迹信息", 7.40, 5.28, 5.03, .30, 12.5, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, "DAIR 为算法验证数据集，非消防、防汛等目标场景现场数据", 7.32, 5.72, 5.18, .30, 9.5, MUTED, False, PP_ALIGN.CENTER)
    add_conclusion(slide, "应用场景不同，但底层时空同步问题具有共性；以真实车路数据与受控扰动验证 DSTA 关键机制。")
    add_footer(slide, "数据：DAIR-V2X-Seq SPD-Example｜公开可复现验证载体")


def slide_2(prs, timing, spatial):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    add_title(slide, 2, "动态时空对齐：提升异步观测匹配与空间表达精度")
    # Two large evidence panels
    for x, title, tag, color in [(.45, "自适应时间对齐", "真实帧对 + 受控延迟", CYAN), (6.72, "质量感知空间映射", "真实冻结测试集", PURPLE)]:
        add_rect(slide, x, 1.26, 6.15, 5.14)
        label(slide, tag, x + .20, 1.44, 1.45, color)
        add_text(slide, title, x + .20, 1.77, 3.50, .34, 17, WHITE, True)
    add_picture_contain(slide, ROOT / "outputs/ppt_case/time_alignment/time_alignment_ppt.png", .67, 2.15, 5.72, 3.22)
    add_picture_contain(slide, ROOT / "outputs/ppt_case/spatial_mapping/spatial_mapping_ppt.png", 6.94, 2.15, 5.72, 3.22)
    fixed = timing.loc["Fixed arrival-time window"]; improved = timing.loc["Adaptive event-time window"]
    metric(slide, f"{fixed.correct_pair_rate*100:.2f}% → {improved.correct_pair_rate*100:.2f}%", "正确配对率", .82, 5.49, 2.42, CYAN, 24)
    metric(slide, f"{fixed.p90_time_error_ms:.2f} → {improved.p90_time_error_ms:.2f} ms", "P90 时间误差", 3.50, 5.49, 2.62, ORANGE, 22)
    base = spatial.loc["Fixed 32 m"]; quality = spatial.loc["Quality-aware"]
    metric(slide, f"{base.mean_quantization_error_m:.2f} → {quality.mean_quantization_error_m:.2f} m", "平均量化误差", 7.11, 5.49, 2.63, PURPLE, 22)
    metric(slide, f"{int(base.n_cells)} / {int(quality.n_cells)}", "固定 / 质量感知网格数", 10.01, 5.49, 2.43, CYAN, 24)
    add_conclusion(slide, "在选定数据与实验条件下，DSTA 改善动态时间错位下的配对效果，并自适应组织空间观测。")
    add_footer(slide, "时间：101 组真实官方帧对，延迟与抖动为受控注入｜空间：同一 ROI、统一 ≤200 叶网格预算；两项指标不可合成为综合精度")


def slide_3(prs, missing, reliability):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    add_title(slide, 3, "复杂扰动下的鲁棒同步：缺失可恢复，质量可感知")
    add_rect(slide, .45, 1.25, 6.15, 5.20); label(slide, "乱序 / 缺失", .67, 1.44, 1.05, ORANGE)
    add_text(slide, "观测中断时：恢复轨迹连续性", .68, 1.78, 4.8, .34, 17, WHITE, True)
    add_picture_contain(slide, ROOT / "outputs/ppt_case/dsta_validation/missing_compensation/missing_compensation_ppt.png", .67, 2.15, 5.72, 3.22)
    r40 = missing[(missing.missing_rate.eq(.4))].set_index("method")
    metric(slide, f"{r40.loc['baseline_arrival_time','position_rmse_m']:.3f} m", "基线 RMSE", .85, 5.49, 2.10, ORANGE, 27)
    metric(slide, f"{r40.loc['improved_event_time','position_rmse_m']:.3f} m", "DSTA RMSE", 3.25, 5.49, 2.10, CYAN, 27)
    add_rect(slide, 6.72, 1.25, 6.15, 5.20); label(slide, "质量退化", 6.94, 1.44, 1.05, PURPLE)
    add_text(slide, "单通道退化时：降低异常影响", 6.95, 1.78, 4.8, .34, 17, WHITE, True)
    add_picture_contain(slide, ROOT / "outputs/ppt_case/dsta_validation/reliability_weighting/reliability_weighting_ppt.png", 6.94, 2.15, 5.72, 3.22)
    rr = reliability[reliability.phase.eq("overall")].set_index("method")
    metric(slide, f"{rr.loc['equal_weight','rmse_mean']:.3f} m", "整体等权 RMSE", 7.14, 5.49, 2.10, ORANGE, 27)
    metric(slide, f"{rr.loc['dynamic_reliability','rmse_mean']:.3f} m", "动态加权 RMSE", 9.63, 5.49, 2.20, PURPLE, 27)
    add_conclusion(slide, "真实轨迹提供参考：中断观测可补偿，质量下降通道可降权。")
    add_footer(slide, "缺失、乱序及双通道噪声均为受控仿真｜96 点真实补充轨迹不属于 EVT2_006｜双通道不代表真实车路融合")


def slide_4(prs, consistency):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    add_title(slide, 4, "多维时空一致性校验：显著降低误报，保留质量溯源")
    add_rect(slide, .45, 1.25, 8.25, 5.30)
    add_picture_contain(slide, ROOT / "outputs/ppt_case/dsta_validation/consistency_validation/consistency_validation_ppt.png", .62, 1.43, 7.91, 4.80)
    med = consistency[consistency.strength.eq("medium")].set_index("method")
    orow, joint = med.loc["simple_or"], med.loc["joint_multidim"]
    add_rect(slide, 8.92, 1.25, 3.95, 5.30)
    label(slide, "质量把关", 9.15, 1.45, 1.00, CYAN)
    add_text(slide, "更低误报，代价是更低召回", 9.15, 1.80, 3.35, .45, 17, WHITE, True)
    metric(slide, f"{orow.fpr*100:.1f}% → {joint.fpr*100:.1f}%", "FPR · 误报率", 9.18, 2.35, 3.38, CYAN, 27)
    add_text(slide, f"Recall  {orow.recall*100:.1f}%  →  {joint.recall*100:.1f}%", 9.20, 3.30, 3.30, .33, 17, ORANGE, True, PP_ALIGN.CENTER)
    add_text(slide, f"F1         {orow.f1*100:.1f}%  →  {joint.f1*100:.1f}%", 9.20, 3.72, 3.30, .33, 17, PURPLE, True, PP_ALIGN.CENTER)
    stages = [("异常候选", CYAN), ("联合校验", PURPLE), ("质量标记", ORANGE), ("来源追溯", CYAN)]
    for i, (txt, col) in enumerate(stages):
        xx = 9.12 + (i % 2) * 1.72; yy = 4.45 + (i // 2) * .63
        add_rect(slide, xx, yy, 1.48, .43, fill=PANEL_2, line=col)
        add_text(slide, txt, xx, yy + .04, 1.48, .30, 11.5, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, "部署需按场景风险选择阈值", 9.18, 5.83, 3.40, .30, 11, MUTED, True, PP_ALIGN.CENTER)
    add_conclusion(slide, "联合校验提供低误报的保守质量控制能力；实际部署需权衡召回率与误报率。")
    add_footer(slide, "815 条真实候选匹配并非身份真值｜异常为测试集受控注入｜358 条正常校准与 457 条独立测试严格分离")


def slide_5(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    add_title(slide, 5, "从异构观测到统一时空数据：可组织、可追溯、可交互")
    add_rect(slide, .45, 1.25, 9.20, 5.52)
    add_picture_contain(slide, ROOT / "outputs/ppt_case/cube/dsta_cube_ppt.png", .62, 1.43, 8.86, 4.75)
    add_text(slide, "X / Y：真实世界坐标   ·   Z：相对事件时间 T（非地理高度）", .75, 6.21, 8.56, .30, 11, CYAN, True, PP_ALIGN.CENTER)
    add_rect(slide, 9.87, 1.25, 3.00, 5.52)
    label(slide, "真实数据产物", 10.10, 1.46, 1.16, PURPLE)
    for i, (val, cap, col) in enumerate([("8,327", "真实观测", CYAN), ("830", "网格窗口状态", ORANGE), ("186", "自适应网格", PURPLE), ("13", "时间切片", YELLOW)]):
        yy = 1.89 + i * .91
        add_text(slide, val, 10.13, yy, 2.48, .42, 28, col, True, PP_ALIGN.CENTER)
        add_text(slide, cap, 10.13, yy + .40, 2.48, .22, 10, MUTED, False, PP_ALIGN.CENTER)
    add_text(slide, "双粒度 Schema", 10.18, 5.63, 2.33, .25, 12, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, "event_time · window_id · grid_id\nsource · X/Y · quality_flags · provenance", 10.08, 5.92, 2.54, .52, 9.5, MUTED, False, PP_ALIGN.CENTER)
    add_footer(slide, "立方体由真实标准化观测与严格聚合数据构建；实验扰动层独立保存，不宣称五项实验已端到端串联生成该立方体")


def slide_6(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(slide)
    add_title(slide, 6, "五项核心能力得到验证，具备向城市多源感知场景适配的技术基础")
    # Evidence matrix
    add_rect(slide, .45, 1.22, 12.42, 3.52)
    headers = ["共性问题", "验证载体", "关键证据", "可迁移能力"]
    widths = [2.00, 3.55, 3.15, 3.12]; xs = [.62, 2.62, 6.17, 9.32]
    for x, w, h in zip(xs, widths, headers):
        add_rect(slide, x, 1.42, w, .42, fill="123552", line=GRID, radius=False)
        add_text(slide, h, x, 1.47, w, .28, 11.5, CYAN, True, PP_ALIGN.CENTER)
    rows = [
        ("时间异步", "101 组真实帧对 + 扰动", "配对率 98.02%", "事件时间对齐"),
        ("空间异构", "1,235 条真实坐标", "平均误差 8.60 m", "自适应空间组织"),
        ("乱序缺失", "96 点真实轨迹 + 扰动", "40% 缺失 RMSE 0.179 m", "时序补偿"),
        ("质量退化", "真实轨迹 + 仿真通道", "整体 RMSE 0.450 m", "动态可靠性加权"),
        ("一致性风险", "815 条候选 + 异常注入", "FPR 2.2% · Recall 52.9%", "多维质量校验"),
    ]
    for i, row in enumerate(rows):
        yy = 1.85 + i * .53
        if i % 2 == 0: add_rect(slide, .62, yy, 11.82, .49, fill="0D263F", line=None, radius=False)
        for j, (x, w, text) in enumerate(zip(xs, widths, row)):
            col = [WHITE, MUTED, [CYAN, PURPLE, ORANGE, PURPLE, CYAN][i], WHITE][j]
            add_text(slide, text, x + .04, yy + .05, w - .08, .34, 10.7, col, j in (0, 2), PP_ALIGN.CENTER)
    # Transfer logic
    add_text(slide, "跨场景适配逻辑", .55, 4.94, 2.25, .30, 15, WHITE, True)
    add_rect(slide, .62, 5.37, 2.55, .83, fill="0D2F49", line=CYAN)
    add_text(slide, "已验证的共性机制", .62, 5.52, 2.55, .28, 15, CYAN, True, PP_ALIGN.CENTER)
    add_text(slide, "→", 3.25, 5.51, .42, .30, 22, MUTED, True, PP_ALIGN.CENTER)
    add_rect(slide, 3.72, 5.18, 5.77, 1.22, fill=PANEL_2, line=PURPLE)
    add_text(slide, "目标数据适配层", 3.91, 5.28, 1.55, .28, 13, PURPLE, True)
    add_text(slide, "时间戳语义/时钟基准  ·  坐标系/定位精度  ·  模态映射/关联条件\n缺失乱序/质量指标  ·  现场延迟/吞吐/可靠性测试", 3.90, 5.61, 5.28, .55, 10.5, WHITE, False, PP_ALIGN.CENTER)
    add_text(slide, "→", 9.55, 5.51, .42, .30, 22, MUTED, True, PP_ALIGN.CENTER)
    add_rect(slide, 10.02, 5.18, 2.58, 1.22, fill="142B47", line=ORANGE)
    add_text(slide, "消防应急\n防汛排涝\n人群管控", 10.02, 5.29, 2.58, .88, 13.5, WHITE, True, PP_ALIGN.CENTER)
    add_conclusion(slide, "本次 Case 证明关键同步机制在真实数据参考及受控扰动条件下有效，并明确了推广所需接口与验证路径。", y=6.57)
    add_footer(slide, "能力可迁移，指标不可直接外推｜尚未完成视频、通信、北斗、IoT 四模态现场融合或生产级实时性验证")


def write_manifest():
    text = """# DSTA 6 页汇报｜素材与指标来源

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
"""
    (OUT / "ppt_asset_manifest.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timing, spatial, missing, reliability, consistency = load_metrics()
    prs = Presentation(); prs.slide_width = Inches(W); prs.slide_height = Inches(H)
    prs.core_properties.title = "DSTA 动态时空同步算法验证汇报"
    prs.core_properties.subject = "真实数据与受控扰动下的关键算法能力验证"
    prs.core_properties.author = "DSTA 项目组"
    slide_1(prs); slide_2(prs, timing, spatial); slide_3(prs, missing, reliability)
    slide_4(prs, consistency); slide_5(prs); slide_6(prs)
    prs.save(PPTX_PATH); write_manifest()
    print(json.dumps({"pptx": str(PPTX_PATH), "slides": len(prs.slides)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
