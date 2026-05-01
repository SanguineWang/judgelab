from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FONT = "/System/Library/Fonts/PingFang.ttc"
BASE_ARCH = Path("/Users/mac/.codex/generated_images/019ddeb7-f8a0-7982-a000-82ff88bbda09/ig_0bf554d1b5c701bd0169f3832fc9e08191bd53a3d6dafa0241.png")
BASE_FLOW = Path("/Users/mac/.codex/generated_images/019ddeb7-f8a0-7982-a000-82ff88bbda09/ig_0bf554d1b5c701bd0169f383757f2c81918adddb446f258471.png")


def font(size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size=size, index=index)


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, size: int, min_size: int = 18) -> ImageFont.FreeTypeFont:
    current = size
    while current > min_size:
        fnt = font(current)
        if draw.textbbox((0, 0), text, font=fnt)[2] <= max_width:
            return fnt
        current -= 1
    return font(min_size)


def wrap_label(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    for part in text.split("\n"):
        if not part:
            lines.append("")
            continue
        if any(ch.isspace() for ch in part):
            wrapped = wrap(part, max_chars)
            lines.extend(wrapped or [""])
            continue
        while len(part) > max_chars:
            lines.append(part[:max_chars])
            part = part[max_chars:]
        lines.append(part)
    return lines


def draw_center_text(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill="#1f2937", line_gap=6) -> None:
    x1, y1, x2, y2 = box
    max_chars = max(4, int((x2 - x1) / (fnt.size * 0.62)))
    lines = wrap_label(text, max_chars)
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line, h in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        x = x1 + ((x2 - x1) - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += h + line_gap


def rounded_box(draw, box, fill, outline="#d7deea", width=2, radius=20):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill="#3b82f6", width=4):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    if x2 >= x1:
        points = [(x2, y2), (x2 - 14, y2 - 8), (x2 - 14, y2 + 8)]
    else:
        points = [(x2, y2), (x2 + 14, y2 - 8), (x2 + 14, y2 + 8)]
    draw.polygon(points, fill=fill)


def render_architecture() -> Path:
    bg = Image.open(BASE_ARCH).convert("RGBA").resize((1600, 900))
    overlay = Image.new("RGBA", bg.size, (255, 255, 255, 70))
    img = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(img)

    title_font = font(40)
    sub_font = font(23)
    label_font = font(24)
    small_font = font(18)
    rounded_box(draw, (50, 30, 1180, 135), "#fffffff0", "#e5e7eb", width=1, radius=18)
    draw.text((70, 42), "BigSample JudgeLab 大样本文本判定平台架构", font=title_font, fill="#111827")
    draw.text((72, 94), "本地工作流 · 多 Excel 导入 · DuckDB 数据底座 · LLM 标注 · 模型训练 · 全量预测", font=sub_font, fill="#4b5563")

    layers = [
        ((80, 165, 1520, 275), "Streamlit 工作台 UI", "数据集管理 / 流程导航 / 分页预览 / 进度反馈 / 参数配置"),
        ((80, 320, 1520, 455), "Application Services", "导入预检 · 抽样 · LLM 初标 · 标签质控 · 数据集划分 · 训练调度 · 批量预测"),
        ((80, 505, 1520, 650), "Storage Layer", "workspace.db 管理元数据与历史；每个数据集独立 data.duckdb 承载百万级表格数据"),
        ((80, 700, 1520, 820), "Model & Output Layer", "hfl/chinese-macbert-base / 模型版本 / predictions / Excel、CSV、Parquet、报告导出"),
    ]
    colors = ["#f8fbff", "#f5f9ff", "#f2fbf7", "#fbf7ff"]
    outlines = ["#93c5fd", "#60a5fa", "#34d399", "#a78bfa"]
    for idx, (box, heading, desc) in enumerate(layers):
        rounded_box(draw, box, colors[idx], outlines[idx], width=3, radius=24)
        draw.text((box[0] + 34, box[1] + 24), heading, font=label_font, fill="#111827")
        draw.text((box[0] + 34, box[1] + 64), desc, font=small_font, fill="#4b5563")
        if idx < len(layers) - 1:
            arrow(draw, ((box[0] + box[2]) // 2, box[3] + 8), ((box[0] + box[2]) // 2, layers[idx + 1][0][1] - 14), fill="#64748b", width=3)

    side_boxes = [
        ((110, 330, 340, 440), "多文件导入\n表头一致性检测"),
        ((405, 330, 635, 440), "工作流状态\n数据资产追踪"),
        ((700, 330, 930, 440), "LLM Schema\n结构化字段展开"),
        ((995, 330, 1225, 440), "Job 化任务\n进度与断点"),
        ((1290, 330, 1490, 440), "模型版本\n低置信度回流"),
    ]
    for box, text in side_boxes:
        rounded_box(draw, box, "#ffffffd8", "#bfdbfe", width=2, radius=16)
        draw_center_text(draw, box, text, small_font)

    out = ASSETS / "judgelab-architecture.png"
    img.convert("RGB").save(out, quality=96)
    return out


def render_workflow() -> Path:
    bg = Image.open(BASE_FLOW).convert("RGBA").resize((1600, 900))
    overlay = Image.new("RGBA", bg.size, (255, 255, 255, 72))
    img = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(img)
    title_font = font(42)
    sub_font = font(22)
    step_font = font(21)
    small_font = font(17)

    draw.text((70, 52), "BigSample JudgeLab 数据工作流转图", font=title_font, fill="#111827")
    draw.text((72, 106), "从多源 Excel 到全量预测结果，每一步都沉淀为可追溯的数据资产", font=sub_font, fill="#4b5563")

    steps = [
        ("1", "创建数据集", "workspace"),
        ("2", "Excel 导入", "raw_records"),
        ("3", "智能抽样", "sampled"),
        ("4", "LLM 初标", "labeled"),
        ("5", "标签质控", "reviewed"),
        ("6", "数据划分", "train / val / test"),
        ("7", "模型训练", "model v1"),
        ("8", "全量预测", "predictions"),
    ]
    xs = [115, 310, 505, 700, 895, 1090, 1285, 1480]
    y = 470
    for i, (num, title, asset) in enumerate(steps):
        box = (xs[i] - 78, y - 78, xs[i] + 78, y + 78)
        fill = "#eff6ff" if i == 3 else "#ffffffdd"
        outline = "#3b82f6" if i == 3 else "#93c5fd"
        rounded_box(draw, box, fill, outline, width=3, radius=22)
        draw.ellipse((xs[i] - 21, y - 58, xs[i] + 21, y - 16), fill="#1d4ed8" if i == 3 else "#3b82f6")
        draw_center_text(draw, (xs[i] - 21, y - 58, xs[i] + 21, y - 16), num, font(20), fill="#ffffff")
        draw_center_text(draw, (box[0] + 10, y - 4, box[2] - 10, y + 36), title, step_font, fill="#111827")
        draw_center_text(draw, (box[0] + 10, y + 42, box[2] - 10, box[3] - 8), asset, small_font, fill="#64748b", line_gap=3)
        if i < len(steps) - 1:
            arrow(draw, (box[2] + 8, y), (xs[i + 1] - 88, y), fill="#3b82f6", width=3)

    bottom = (110, 720, 1490, 820)
    rounded_box(draw, bottom, "#f8fafcdd", "#cbd5e1", width=2, radius=22)
    draw_center_text(
        draw,
        bottom,
        "闭环优化：低置信度样本 → 人工/LLM 复核 → 难例库 → Prompt 优化 / 模型再训练 → 新一轮全量预测",
        font(25),
        fill="#1f2937",
    )

    out = ASSETS / "judgelab-data-workflow.png"
    img.convert("RGB").save(out, quality=96)
    return out


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(render_architecture())
    print(render_workflow())


if __name__ == "__main__":
    main()
