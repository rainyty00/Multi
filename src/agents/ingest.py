"""
=================================================================
采集解析 Agent（阶段1）
=================================================================
职责：
  1. 识别输入是「本地文件」还是「平台链接」，是哪个平台
  2. ★为本次任务分配一个"作业目录" data/jobs/<job_id>/（全英文路径）
  3. 把视频复制/下载进作业目录，统一命名成 source.xxx（英文）
  4. 平台链接 → 用 yt-dlp 抓元数据 + 下载
  5. 中文原标题只存进 metadata（用于显示和最终导出命名），不作为处理路径

★为什么要作业目录：内部处理全用英文路径，一次性躲开 cv2/ffmpeg 等库
  在中文路径上的各种坑；每个任务产物自成一夹，也方便"查看历史"和"定时清理"。
=================================================================
"""
import os
import json
import shutil
import datetime
from pathlib import Path

import yt_dlp

from src.state import GraphState, VideoMeta
from config import DATA_DIR, UPLOAD_DIR, FFMPEG_EXE

# 所有任务的作业目录都放在 data/jobs/ 下
JOBS_DIR = DATA_DIR / "jobs"


# ---------- 解析本地视频的真实路径 ----------
def resolve_local_path(source: str):
    """
    找到用户指定的本地视频到底在哪。依次尝试：
      1) source 本身就是存在的路径
      2) data/upload/ 下的同名文件（用户把视频放上传目录时，可只传文件名）
    找到返回 Path，找不到返回 None。
    """
    p = Path(source)
    if p.exists():
        return p
    for cand in (UPLOAD_DIR / source, UPLOAD_DIR / p.name):
        if cand.exists():
            return cand
    return None


# ---------- 为本次任务建作业目录 ----------
def make_job_dir() -> tuple[str, Path]:
    """用当前时间生成一个唯一 job_id，并创建对应目录。返回 (job_id, 目录Path)。"""
    job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


# ---------- 写"说明文件"：双保险（每目录一份 + 一个总索引）----------
def write_job_manifest(job_id: str, job_dir: Path, meta, source: str):
    """
    记录"英文文件夹 ↔ 原始视频"的对应关系，方便以后查。
      1. 每个 job 目录里放一份 meta.json（自带身份，可单独搬走）
      2. data/jobs/index.json 总索引里追加一条（一眼查全部）
    """
    created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = {"标题": meta.title, "平台": meta.platform, "来源": source, "创建时间": created}

    # 1) 当前目录的说明
    (job_dir / "meta.json").write_text(
        json.dumps({"job_id": job_id, **info}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) 总索引（读出来→加一条→写回去）
    index_path = JOBS_DIR / "index.json"
    index = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    index[job_id] = info
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 从 URL 判断平台 ----------
def detect_platform(source: str) -> str:
    """返回：bilibili / xiaohongshu / douyin / local / unknown"""
    s = source.lower()
    if resolve_local_path(source) is not None:   # 本地能找到（含 data/upload/）
        return "local"
    if "bilibili.com" in s or "b23.tv" in s:
        return "bilibili"
    if "xiaohongshu.com" in s or "xhslink" in s:
        return "xiaohongshu"
    if "douyin.com" in s or "iesdouyin" in s:
        return "douyin"
    return "unknown"


# ---------- 本地文件：复制进作业目录，重命名成英文 source.xxx ----------
def handle_local_file(path: str, job_dir: Path) -> tuple[str, VideoMeta]:
    src = resolve_local_path(path)            # 解析真实路径（可能在 data/upload/）
    dest = job_dir / f"source{src.suffix}"   # 例如 source.mp4
    shutil.copy(src, dest)                    # 复制一份进作业目录
    meta = VideoMeta(
        platform="local",
        title=src.stem,                       # 中文原标题只留作元数据（显示/命名用）
    )
    return str(dest), meta


# ---------- 平台链接：用 yt-dlp 下载进作业目录 ----------
def download_with_ytdlp(url: str, job_dir: Path) -> tuple[str, VideoMeta]:
    ydl_opts = {
        # ★下载到作业目录，统一命名 source.xxx（英文），不用中文标题当文件名
        "outtmpl": str(job_dir / "source.%(ext)s"),
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG_EXE,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    meta = VideoMeta(
        platform=detect_platform(url),
        title=info.get("title", ""),
        description=info.get("description", ""),
        tags=info.get("tags", []) or [],
        duration=float(info.get("duration", 0) or 0),
        author=info.get("uploader", "") or info.get("uploader_id", ""),
        cover_path=info.get("thumbnail", ""),
    )
    # 合并后是 source.mp4
    video_path = str(job_dir / "source.mp4")
    return video_path, meta


# ---------- 节点主函数 ----------
def ingest_node(state: GraphState) -> dict:
    source = state["source"]
    platform = detect_platform(source)

    # 先建作业目录
    job_id, job_dir = make_job_dir()
    print(f"[采集 Agent] 识别平台: {platform}，作业目录: {job_dir}")

    if platform == "local":
        video_path, meta = handle_local_file(source, job_dir)
    elif platform in ("bilibili", "xiaohongshu", "douyin"):
        print("[采集 Agent] 开始用 yt-dlp 下载...")
        video_path, meta = download_with_ytdlp(source, job_dir)
    else:
        raise ValueError(f"无法识别的输入来源：{source}（既不是本地文件，也不是支持的平台链接）")

    # 写说明文件（双保险：本目录 meta.json + 总索引 index.json）
    write_job_manifest(job_id, job_dir, meta, source)

    print(f"[采集 Agent] 完成。视频: {video_path}")
    print(f"[采集 Agent] 标题: {meta.title} | 作者: {meta.author} | 时长: {meta.duration}s")

    return {
        "video_path": video_path,
        "metadata": meta,
        "job_id": job_id,
        "job_dir": str(job_dir),
    }
