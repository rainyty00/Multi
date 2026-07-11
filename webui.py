"""
=================================================================
webui.py —— 网页界面（gradio）
=================================================================
它不重写任何分析逻辑，只是给现有的 LangGraph 流程套一层网页：
  · 分析标签页：上传视频/填链接 → 实时逐节点进度 → 展示分镜表/报告/关键帧 → 下载
  · 历史标签页：浏览过往成品（读 library.db）→ 查看 + 下载

运行：
  PYTHONUTF8=1 E:/Miniconda/envs/Multi/python.exe webui.py
  然后浏览器打开终端里显示的地址（默认 http://127.0.0.1:7860）
=================================================================
"""
import os
import re
import time
import shutil
import datetime
from glob import glob
from pathlib import Path

import gradio as gr

from src.graph import build_graph
from src.tools.library import list_records, auto_cleanup

auto_cleanup()          # ★启动时自动清理超过 7 天的过期成品

# 流程图只构建一次（每次分析用不同 thread_id 隔离）
APP = build_graph()

# ★给用户看的步骤 → 背后对应哪些内部节点
#   （视觉理解/OCR/时序对齐/分镜合成 这些专业术语不暴露，统一收进"分析中"）
STEP_NODES = {
    "采集视频": ["ingest"],
    "预处理": ["preprocess"],
    "分析中": ["overview", "audio", "ocr", "visual", "align", "compose"],
    "质量评估": ["evaluate"],
    "分析报告": ["speaker", "report"],
    "导出入库": ["export"],
}

SB_HEADERS = ["镜号", "时间段", "镜头语言", "画面内容", "口播台词", "屏幕文字", "叙事作用"]


# ---------- 一些小工具 ----------
def render_progress(done_nodes: set) -> str:
    """
    把进度画成用户看得懂的清单。
    一个步骤：底下的节点【全部】跑完 → ✅；否则（没跑或只跑了一部分）→ ⬜
    """
    lines = ["### 分析进度"]
    for step, nodes in STEP_NODES.items():
        all_done = all(n in done_nodes for n in nodes)
        lines.append(f"- ✅ {step}" if all_done else f"- ⬜ {step}")
    return "\n".join(lines)


def frame_num(path: str) -> int:
    m = re.search(r"shot_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def list_frames(out_dir: str) -> list:
    """按镜号顺序列出关键帧，并给每张配上"镜N"的图注（不改动图片本身）。"""
    files = sorted(glob(os.path.join(out_dir, "frames", "*.jpg")), key=frame_num)
    return [(f, f"镜{frame_num(f)}") for f in files]


def make_zip(out_dir: str):
    """把成品目录打包成 zip，供下载。"""
    if not out_dir or not os.path.isdir(out_dir):
        return None
    return shutil.make_archive(out_dir, "zip", out_dir)


def format_report_md(report: dict) -> str:
    if not report:
        return ""
    stats = report.get("shot_stats", {})
    audio = report.get("audio_summary", {})

    # 作品概览（全局总览产出）
    ov = report.get("overview", {})
    ov_md = ""
    if ov:
        chars = "、".join(c.get("label", "") for c in ov.get("characters", []))
        ov_md = (
            "### 作品概览\n"
            f"- 广告产品/品牌：**{ov.get('product', '未知')}**\n"
            f"- 一句话主题：{ov.get('theme', '')}\n"
            f"- 主要人物：{chars or '—'}\n"
            f"- 主要场景：{'、'.join(ov.get('scenes', [])) or '—'}\n"
            f"- 核心卖点：{'、'.join(ov.get('selling_points', [])) or '—'}\n"
            f"- 目标受众：{ov.get('audience', '—')}\n\n"
        )

    return (
        ov_md +
        "### 分析报告\n"
        f"- 镜头统计：{stats.get('count',0)} 个，平均 {stats.get('avg_duration',0)}s\n"
        f"- 音频：语速 {audio.get('speech_rate',0)} {audio.get('rate_unit','字/分')}，"
        f"关键词 {'、'.join(audio.get('keywords',[])) or '（无口播）'}\n"
        f"- 画面风格：{report.get('style_summary','')}\n"
        f"- 创意套路：{report.get('creative_summary','')}\n"
        f"- 图文匹配评分：{report.get('image_text_score',0)} / 5\n"
        f"- 分析耗时：{report.get('elapsed_sec',0)} 秒"
    )


def strip_dialogue_note(text: str) -> str:
    """
    去掉台词文件开头的"（说话人为 AI 根据画面推测…）"说明行。
    界面上台词框下方已有灰色提示，不必重复；下载的台词.txt 里仍保留该说明。
    """
    lines = text.splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("（说话人为")):
        lines.pop(0)
    return "\n".join(lines)


def rows_to_md_table(rows) -> str:
    """把分镜行渲染成 Markdown 表格（表头居中靠 CSS，正文左对齐，列宽自适应不挤字）。"""
    if not rows:
        return "*（暂无分镜数据）*"
    head = "| " + " | ".join(SB_HEADERS) + " |"
    sep = "|" + "|".join(["---"] * len(SB_HEADERS)) + "|"
    body = []
    for r in rows:
        cells = [str(c).replace("\n", " ").replace("|", "／") for c in r]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join(["### 📋 分镜脚本表", head, sep, *body])


def build_outputs(state: dict):
    """从最终状态提取要展示的各部分。"""
    meta = state.get("metadata")
    meta_md = (f"## 《{meta.title}》\n平台：{meta.platform}　·　时长：{meta.duration}s　·　"
               f"作者：{meta.author or '—'}") if meta else ""

    sb_md = rows_to_md_table([
        [r.index, r.time_range, r.camera, r.visual,
         r.voiceover or "—", r.on_screen_text or "—", r.narrative]
        for r in state.get("storyboard", [])
    ])

    report_md = format_report_md(state.get("report", {}))
    out_dir = state.get("exports", {}).get("dir", "")
    gallery = list_frames(out_dir) if out_dir else []

    dialogue = ""
    tpath = Path(out_dir) / "台词.txt" if out_dir else None
    if tpath and tpath.exists():
        dialogue = strip_dialogue_note(tpath.read_text(encoding="utf-8"))

    return meta_md, sb_md, gallery, report_md, dialogue, make_zip(out_dir)


# ---------- 分析（生成器：边跑边更新进度）----------
def analyze(video_path, url):
    source = video_path or (url.strip() if url else "")
    if not source:
        yield "❌ 请先上传视频或填入链接", "", None, None, "", "", None
        return

    thread_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {"configurable": {"thread_id": thread_id}}
    done = set()

    # 先给个初始进度
    yield render_progress(done), "", None, None, "", "", None

    try:
        # ★app.stream 每跑完一个节点就吐一次结果，我们据此更新进度
        for chunk in APP.stream({"source": source, "start_ts": time.time()},
                                config, stream_mode="updates"):
            for node in chunk:
                done.add(node)   # 收集"已跑完的节点名"，由 render_progress 换算成用户步骤
            # 只更新进度，其余结果保持不变（gr.update()）
            yield (render_progress(done), gr.update(), gr.update(),
                   gr.update(), gr.update(), gr.update(), gr.update())
    except Exception as e:
        yield f"❌ 分析出错：{e}", "", None, None, "", "", None
        return

    # 跑完了，取最终状态，展示全部结果
    state = APP.get_state(config).values
    meta_md, rows, gallery, report_md, dialogue, zip_path = build_outputs(state)
    yield ("✅ 分析完成！\n\n" + render_progress(done),
           meta_md, rows, gallery, report_md, dialogue, zip_path)


# ---------- 历史记录 ----------
def history_choices():
    """下拉框选项：[(显示文字, 成品目录), ...]"""
    return [(f"《{r[1]}》  {r[3]}  ({r[2]})", r[4]) for r in list_records()]


def view_history(out_dir):
    """查看某条历史成品：展示它的 Markdown + 关键帧 + zip 下载。"""
    if not out_dir or not os.path.isdir(out_dir):
        return "（请选择一条记录）", None, None
    md_path = Path(out_dir) / "storyboard.md"
    md = md_path.read_text(encoding="utf-8") if md_path.exists() else "（找不到成品文件）"
    return md, list_frames(out_dir), make_zip(out_dir)


# ---------- 界面样式 ----------
CSS = """
.gradio-container {max-width: 2300px !important; margin: auto !important;}
#hero {text-align:center; padding: 8px 0 4px 0;}
#hero h1 {margin-bottom: 4px;}
#hero p {color: #6b7280; margin-top: 0;}
.card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 14px;}
#ttl-banner {background:#eef2ff; border:1px solid #c7d2fe; border-radius:10px;
             padding:10px 14px; color:#3730a3; font-weight:600; text-align:center;}
#refresh-btn {width:100% !important;}
/* 分镜表表头居中、不换行（正文仍左对齐） */
thead th, table th {text-align:center !important; white-space:nowrap !important;}
footer {display:none !important;}
/* 统一两个Tab内部行的边距、宽度 */
[data-testid="tabitem"] .gr-row {width:100% !important; padding-inline: 12px !important;}
[data-testid="tabitem"] {padding: 0 6px;}
/* 强制Radio历史条目铺满整行，和刷新按钮同宽 */
.radio-group label {
    width: 100% !important;
    box-sizing: border-box !important;
}
.radio-group label span {
    width: 100% !important;
    display: block !important;
}
.radio-group {
    padding-inline: 0 !important;
}
"""


# Gradio 6：主题在 launch() 里传
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", radius_size="lg")


def build_ui():
    with gr.Blocks(title="广告创意拆解 Agent", fill_width=True) as demo:
        gr.HTML(
            "<div id='hero'><h1>🎬 多模态广告创意拆解 Agent</h1>"
            "<p>上传广告视频 → 自动还原 <b>分镜脚本表</b> · <b>创意分析报告</b> · <b>素材文件</b></p></div>"
        )

        with gr.Tabs():
            # ===== 分析标签页：左 1/3 输入区，右 2/3 结果区 =====
            with gr.Tab("🎬 分析"):
                with gr.Row():
                    # 左栏（占 1/3）：上传文件 → 链接 → 按钮 → 进度
                    with gr.Column(scale=4, min_width=180):
                        gr.Markdown("#### 上传与分析")
                        video_in = gr.File(label="① 上传视频", type="filepath",
                                           file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm"])
                        url_in = gr.Textbox(label="② 或 视频链接（B站等）",
                                            placeholder="https://www.bilibili.com/video/...")
                        analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
                        status = gr.Markdown("### 分析进度\n等待开始…", elem_classes="card")

                    # 右栏（占 2/3）：元数据/报告 → 分镜表 → 关键帧 → 台词 → 下载
                    with gr.Column(scale=11, min_width=320):
                        gr.HTML("<div id='ttl-banner'>🎬 分析结果将在下方展示，完成后可打包下载</div>")
                        meta_out = gr.Markdown("### 分析结果",elem_classes="card")
                        report_out = gr.Markdown("### 分析报告", elem_classes="card")
                        sb_out = gr.Markdown("### 分镜脚本表",elem_classes="card")
                        gallery_out = gr.Gallery(label="🖼️ 关键帧", columns=5, height=280)
                        dialogue_out = gr.Textbox(
                            label="📝 台词", lines=6,
                            info="说话人为 AI 推测，(?) 表示不确定，仅供参考",
                        )
                        download_out = gr.File(label="⬇️ 下载全部（zip）")

                analyze_btn.click(
                    analyze, [video_in, url_in],
                    [status, meta_out, sb_out, gallery_out, report_out, dialogue_out, download_out],
                )

            # ===== 历史记录标签页：左 1（点击列表+刷新），右 2（提示+内容+关键帧+下载）=====
            with gr.Tab("📚 历史记录"):
                with gr.Row():
                    # 左栏（缩窄为原来的 4/5）：过往成品做成"点击即选"的列表 + 刷新
                    with gr.Column(scale=4, min_width=180):
                        gr.Markdown("#### 选择过往成品")
                        hist_radio = gr.Radio(choices=history_choices(), show_label=False)
                        refresh_btn = gr.Button("🔄 刷新", elem_id="refresh-btn")

                    # 右栏：顶部保存提示 → 成品内容 → 关键帧 → 下载
                    with gr.Column(scale=11, min_width=320):
                        gr.HTML("<div id='ttl-banner'>⏳ 历史成品请及时下载！</div>")
                        hist_md = gr.Markdown("### 分析结果", elem_classes="card")
                        hist_gallery = gr.Gallery(label="🖼️ 关键帧", columns=5, height=280)
                        hist_zip = gr.File(label="⬇️ 下载（zip）")

                # 点击列表某项即查看；刷新则重新拉取列表
                refresh_btn.click(lambda: gr.update(choices=history_choices()), None, hist_radio)
                hist_radio.change(view_history, hist_radio, [hist_md, hist_gallery, hist_zip])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS)
