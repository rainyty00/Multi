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
from src.state import GraphState, Shot


def sec_to_mmss(sec: float) -> str:
    """把秒数转成 mm:ss 格式，如 6.87 → 00:06"""
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


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
            # ★证据帧：这个镜头的代表关键帧，阶段5评估要用它来核对
            "keyframe": shot.keyframes[0] if shot.keyframes else "",
        })

    audio_ok = "有" if audio.get("full_text") else "无口播"
    print(f"[时序对齐] 三支线汇合完成 → {len(aligned)}个镜头证据包 "
          f"(音频:{audio_ok} | 视觉:{len(visual)}镜 | OCR:{len(ocr.get('all_texts', []))}条)")

    return {"aligned": aligned}
