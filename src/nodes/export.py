"""
=================================================================
导出节点（阶段6.B-1）—— 需求④⑤：把成果落地成文件
=================================================================
在 outputs/<标题_创意分析>/ 下生成：
  · storyboard.md   分镜表(Markdown) + 元数据 + 分析报告  （需求②③⑤）
  · storyboard.csv  分镜表(CSV)                          （需求⑤）
  · 台词.txt        完整无时间轴台词                      （需求④）
  · 全部文字.txt    视频内全部文字汇总                    （需求④）
  · frames/         各镜头关键帧                          （需求④）
纯 Python 写文件，中文没问题（不经过 cv2/ffmpeg）。
=================================================================
"""
import csv
import shutil
from pathlib import Path

from src.state import GraphState
from src.tools.library import add_record
from config import OUTPUT_DIR


def safe_name(name: str) -> str:
    """去掉文件名里的非法字符（Windows 不允许 \\ / : * ? \" < > |）。"""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip() or "未命名"


def build_markdown(state: GraphState) -> str:
    meta = state["metadata"]
    storyboard = state.get("storyboard", [])
    report = state.get("report", {})

    lines = []
    # --- 标题与元数据（需求①）---
    lines.append(f"# 《{meta.title}》创意分析\n")
    lines.append(f"- 平台：{meta.platform}　作者：{meta.author or '—'}　时长：{meta.duration}s")
    if meta.tags:
        lines.append(f"- 话题标签：{'、'.join(meta.tags)}")
    if meta.description:
        lines.append(f"- 简介：{meta.description}")
    lines.append("")

    # --- 作品概览（全局总览产出）---
    ov = report.get("overview", {})
    if ov:
        chars = "、".join(c.get("label", "") for c in ov.get("characters", []))
        lines.append("## 作品概览\n")
        lines.append(f"- 广告产品/品牌：{ov.get('product', '未知')}")
        lines.append(f"- 一句话主题：{ov.get('theme', '')}")
        lines.append(f"- 主要人物：{chars or '—'}")
        lines.append(f"- 主要场景：{'、'.join(ov.get('scenes', [])) or '—'}")
        lines.append(f"- 核心卖点：{'、'.join(ov.get('selling_points', [])) or '—'}")
        lines.append(f"- 目标受众：{ov.get('audience', '—')}")
        lines.append("")

    # --- 分镜脚本表（需求②）---
    lines.append("## 分镜脚本表\n")
    lines.append("| 镜号 | 时间段 | 镜头语言 | 画面内容 | 口播台词 | 屏幕贴纸/字幕 | 叙事作用 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in storyboard:
        # 表格里不能有换行符和竖线，替换掉
        def cell(x): return str(x).replace("\n", " ").replace("|", "／")
        lines.append(f"| {r.index} | {cell(r.time_range)} | {cell(r.camera)} | {cell(r.visual)} "
                     f"| {cell(r.voiceover) or '—'} | {cell(r.on_screen_text) or '—'} | {cell(r.narrative)} |")
    lines.append("")

    # --- 分析报告（需求③）---
    stats = report.get("shot_stats", {})
    audio = report.get("audio_summary", {})
    lines.append("## 分析报告\n")
    lines.append(f"- 镜头统计：共 {stats.get('count',0)} 个镜头，"
                 f"平均时长 {stats.get('avg_duration',0)}s")
    lines.append(f"- 音频：语速 {audio.get('speech_rate',0)} {audio.get('rate_unit','字/分')}，"
                 f"关键词 {'、'.join(audio.get('keywords',[])) or '（无口播）'}")
    lines.append(f"- 画面风格：{report.get('style_summary','')}")
    lines.append(f"- 创意套路：{report.get('creative_summary','')}")
    lines.append(f"- 图文匹配评分：{report.get('image_text_score',0)} / 5")
    lines.append(f"- 分析耗时：{report.get('elapsed_sec',0)} 秒")
    lines.append("")
    return "\n".join(lines)


def collect_clean_texts(state: GraphState) -> list[str]:
    """
    汇总"清洗后"的画面文字：用分镜表里已被 LLM 清洗合并的 on_screen_text，
    按镜头整段去重（保留完整短语，不打散）。干净、无水印、无碎片（需求④）。
    """
    seen, result = set(), []
    for r in state.get("storyboard", []):
        txt = str(r.on_screen_text).strip()
        if txt and txt not in ("—", "") and txt not in seen:
            seen.add(txt)
            result.append(txt)
    return result


def export_node(state: GraphState) -> dict:
    meta = state["metadata"]
    storyboard = state.get("storyboard", [])
    audio = state.get("audio_result", {})
    ocr = state.get("ocr_result", {})

    # 输出目录：outputs/<标题_创意分析>/
    out_dir = OUTPUT_DIR / safe_name(f"{meta.title}_创意分析")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Markdown
    md = build_markdown(state)
    (out_dir / "storyboard.md").write_text(md, encoding="utf-8")

    # 2) CSV（分镜表）
    with open(out_dir / "storyboard.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["镜号", "时间段", "镜头语言", "画面内容", "口播台词", "屏幕贴纸/字幕", "叙事作用"])
        for r in storyboard:
            w.writerow([r.index, r.time_range, r.camera, r.visual,
                        r.voiceover, r.on_screen_text, r.narrative])

    # 3) 完整无时间轴台词（需求④）—— 带说话人标注（AI推测）
    dialogue = state.get("dialogue", [])
    if dialogue:
        lines_txt = [f"[{d['speaker']}] {d['text']}" for d in dialogue]
        (out_dir / "台词.txt").write_text("\n".join(lines_txt), encoding="utf-8")
    else:
        (out_dir / "台词.txt").write_text(
            audio.get("full_text", "") or "（本视频无口播）", encoding="utf-8")

    # 4) 视频内全部文字汇总（需求④）—— 用清洗后的文字，干净无水印无碎片
    clean_texts = collect_clean_texts(state)
    (out_dir / "全部文字.txt").write_text(
        "\n".join(clean_texts) or "（画面无有效文字）", encoding="utf-8")
    # 原始 OCR 识别结果另存一份，供溯源/核对（含水印和噪声）
    (out_dir / "画面文字_原始识别.txt").write_text(
        "\n".join(ocr.get("all_texts", [])), encoding="utf-8")

    # 5) 复制关键帧到输出目录（需求④）
    frame_out = out_dir / "frames"
    frame_out.mkdir(exist_ok=True)
    for r in storyboard:
        if r.evidence_frame and Path(r.evidence_frame).exists():
            shutil.copy(r.evidence_frame, frame_out / f"shot_{r.index}.jpg")

    # 登记进知识库索引（需求⑤：支持查看历史 + 定时清理）
    add_record(meta.title, meta.platform, str(out_dir))

    print(f"[导出] 已导出到: {out_dir}（已入库）")
    print("\n" + "=" * 60 + "\n" + md + "=" * 60)   # 需求⑤：终端也展示 Markdown

    return {"exports": {"dir": str(out_dir)}}
