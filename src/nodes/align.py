"""
=================================================================
时序对齐节点（阶段4.A · 三支线汇合点）—— 确定性，不花钱
=================================================================
作用：
  1. 作为三条并发支线（音频/视觉/OCR）的"汇合点"(fan-in)。
     LangGraph 会等三条都跑完，才执行这个节点。
  2. 按镜头的时间区间，把三路信号"对齐打包"成每镜一个证据包：
       {镜号, 时间段, 画面, 镜头语言, 风格, 口播分段, 屏幕文字}
     这个证据包就是下一步「分镜合成」的输入。

★为什么先对齐再合成：让分镜合成 Agent 拿到"已经按镜头凑齐的材料"，
  它只需专注"理解和写表"，不用自己去翻时间戳对应关系。
=================================================================
"""
from pathlib import Path

import cv2

from src.state import GraphState, Shot


def sec_to_mmss(sec: float) -> str:
    """把秒数转成 mm:ss 格式，如 6.87 → 00:06"""
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def best_text_time(shot: Shot, ocr_timeline: list[dict]):
    """
    ★用 OCR 时间戳，找出这一镜里"屏幕文字最清晰"的时刻（置信度最高那条）。
    没有文字就返回 None。
    例：镜15(32.4~39.3s) → OCR 在 33.0s 识别到"点点就好"(置信度1.0) → 返回 33.0
    """
    in_shot = [t for t in ocr_timeline if shot.start <= t["time"] <= shot.end]
    if not in_shot:
        return None
    return max(in_shot, key=lambda x: x["score"])["time"]


def extract_text_frames(video_path: str, shots: list[Shot],
                        ocr_timeline: list[dict], job_dir: Path) -> dict:
    """
    为"有屏幕文字"的镜头，按 OCR 时间戳额外抽一张【文字证据帧】。
    返回 {镜号: 帧路径}；没有文字的镜头不在里面。

    ★为什么需要它：抽关键帧发生在预处理阶段，那时 OCR 还没跑、不知道文字在哪一秒，
      只能取中间帧。而长镜头里字幕常只出现两三秒，中间帧往往拍不到。
      到了本节点，OCR 时间戳已就绪，才能"回头"把那一刻的帧抽出来。
    """
    text_times = {s.index: best_text_time(s, ocr_timeline) for s in shots}
    if not any(t is not None for t in text_times.values()):
        return {}

    frame_dir = job_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    result = {}
    for idx, t in text_times.items():
        if t is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))   # 跳到文字出现那一帧
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = frame_dir / f"shot_{idx}_text.jpg"     # 与代表帧分开存
        cv2.imwrite(str(out_path), frame)                 # job 目录是英文路径，可正常写
        result[idx] = str(out_path)
    cap.release()
    return result


def assign_segments_to_shots(audio_segments: list[dict], shots: list[Shot]) -> dict:
    """
    ★把每句台词【只】归给一个镜头：取"重叠时长最大"的那个。
    修复原来"只要时间有重叠就算"导致的——一句横跨边界的台词被重复计入两镜。
    """
    mapping = {s.index: [] for s in shots}
    for seg in audio_segments:
        best_idx, best_overlap = None, 0.0
        for s in shots:
            overlap = min(seg["end"], s.end) - max(seg["start"], s.start)
            if overlap > best_overlap:
                best_overlap, best_idx = overlap, s.index
        if best_idx is None and shots:          # 完全没重叠（极少见）→ 归第一个镜头
            best_idx = shots[0].index
        if best_idx is not None:
            mapping[best_idx].append(seg["text"])
    return mapping


def align_node(state: GraphState) -> dict:
    shots: list[Shot] = state["shots"]
    audio = state.get("audio_result", {})
    visual = state.get("visual_result", {})
    ocr = state.get("ocr_result", {})

    audio_segments = audio.get("segments", [])
    ocr_by_shot = ocr.get("by_shot", {})

    # 台词按"重叠最大"归属到唯一镜头
    voiceover_map = assign_segments_to_shots(audio_segments, shots)
    # 英文台词分句之间要加空格，中文不加
    sep = " " if audio.get("lang") == "en" else ""

    # ★用 OCR 时间戳，为"有屏幕文字"的镜头额外抽一张【文字证据帧】
    text_frames = extract_text_frames(
        state["video_path"], shots, ocr.get("timeline", []), Path(state["job_dir"]))

    aligned = []
    for shot in shots:
        # 1) 该镜头独占的口播分段
        voiceover_segs = voiceover_map.get(shot.index, [])

        # 2) 该镜头的视觉分析
        v = visual.get(shot.index, {})

        # 3) 该镜头的屏幕文字（OCR 原始结果，含噪声，交给分镜合成去清洗）
        ocr_texts = ocr_by_shot.get(shot.index, [])

        aligned.append({
            "index": shot.index,
            "time_range": f"{sec_to_mmss(shot.start)}-{sec_to_mmss(shot.end)}",
            "visual": v.get("visual", ""),
            "camera": v.get("camera", ""),
            "style": v.get("style", ""),
            "voiceover": sep.join(voiceover_segs),  # 口播（英文句间加空格；可能为空）
            "ocr_texts": ocr_texts,                  # 原始屏幕文字列表
            # ★双证据帧：代表帧核对"画面/镜头语言"，文字帧核对"屏幕文字"
            "keyframe": shot.keyframes[0] if shot.keyframes else "",
            "text_frame": text_frames.get(shot.index, ""),   # 无文字的镜头为空
        })

    audio_ok = "有" if audio.get("full_text") else "无口播"
    print(f"[时序对齐] 三支线汇合完成 → {len(aligned)}个镜头证据包 "
          f"(音频:{audio_ok} | 视觉:{len(visual)}镜 | OCR:{len(ocr.get('all_texts', []))}条 | "
          f"文字证据帧:{len(text_frames)}张)")

    return {"aligned": aligned}
