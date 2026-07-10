"""
=================================================================
OCR Agent（阶段3 · 支线二）—— 本地、免费
=================================================================
职责：识别画面里的文字（slogan、字幕、贴纸），过滤明显噪声，
      去重后按镜头归类。产出写进 state["ocr_result"]：
  {
    "by_shot":   {镜号: [该镜出现的文字...]},
    "all_texts": [全片去重后的所有文字...],   # 需求④"视频内全部文字汇总"
    "timeline":  [{time, text, score}...],    # 明细，供后面时序对齐参考
  }

★定位：工具重(RapidOCR)、逻辑轻(去重/归类)。用独立的"密集抽帧"策略，
  和视觉 Agent 的"稀疏抽帧"分开，互不影响。
=================================================================
"""
import re
import cv2
from rapidocr_onnxruntime import RapidOCR

from src.state import GraphState, Shot
from config import FRAME_SAMPLE_INTERVAL


# ---------- 规则清洗：干掉明显的噪声 ----------
def is_noise(text: str) -> bool:
    """
    判断一段 OCR 文字是不是明显噪声，是就丢弃。
    ★这是"轻规则"清洗；更聪明的"slogan 变体合并 / 水印识别"要靠后面的 LLM 清洗层。
    """
    t = text.strip()
    if len(t) <= 1:
        return True                       # 单个字符，多半是误识别
    if re.fullmatch(r"[\d\W_]+", t):
        return True                       # 纯数字/纯符号（如 000、28）
    return False


# RapidOCR 引擎全局只建一次（初始化要加载模型，有点慢）
_engine = None
def get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        print("[OCR Agent] 初始化 RapidOCR 引擎...")
        _engine = RapidOCR()
    return _engine


# ---------- 判断某个时间点属于哪个镜头 ----------
def find_shot_index(time: float, shots: list[Shot]) -> int:
    """给一个时间点（秒），返回它落在第几个镜号；找不到就归到最后一个镜头。"""
    for s in shots:
        if s.start <= time <= s.end:
            return s.index
    return shots[-1].index if shots else 1


# ---------- 对单帧做 OCR ----------
def ocr_one_frame(frame) -> list[tuple[str, float]]:
    """
    对一帧图片做 OCR，返回 [(文字, 置信度), ...]。
    RapidOCR 返回结构是 [[框坐标, 文字, 置信度], ...]，可能为 None。
    """
    engine = get_engine()
    result, _ = engine(frame)
    if not result:
        return []
    out = []
    for item in result:
        text = item[1].strip()
        score = float(item[2])
        if text:
            out.append((text, score))
    return out


# ---------- 节点主函数 ----------
def ocr_node(state: GraphState) -> dict:
    video_path = state["video_path"]
    shots: list[Shot] = state["shots"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = state["metadata"].duration

    print(f"[OCR Agent] 每 {FRAME_SAMPLE_INTERVAL}s 抽一帧做识别，总时长 {duration}s")

    timeline = []      # 每条识别到的文字明细
    seen_texts = set() # 用来去重（置信度低于阈值的丢弃）
    # 置信度门槛：低于它多半是误识别/噪声。
    # ★调高到 0.7，掐掉像 "T&"/"AZVI" 这种 OCR 没把握的字符碎片
    SCORE_MIN = 0.7

    # 从 0 秒开始，每隔 FRAME_SAMPLE_INTERVAL 秒抽一帧
    t = 0.0
    while t <= duration:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            for text, score in ocr_one_frame(frame):
                # 置信度够高 且 不是明显噪声，才保留
                if score >= SCORE_MIN and not is_noise(text):
                    timeline.append({"time": round(t, 2), "text": text, "score": round(score, 2)})
        t += FRAME_SAMPLE_INTERVAL
    cap.release()

    # 按镜头归类 + 全片去重
    by_shot: dict[int, list[str]] = {}
    all_texts: list[str] = []
    for item in timeline:
        text = item["text"]
        idx = find_shot_index(item["time"], shots)
        # 逐镜去重
        by_shot.setdefault(idx, [])
        if text not in by_shot[idx]:
            by_shot[idx].append(text)
        # 全片去重
        if text not in seen_texts:
            seen_texts.add(text)
            all_texts.append(text)

    print(f"[OCR Agent] 识别到 {len(timeline)} 处文字，去重后 {len(all_texts)} 条:")
    for txt in all_texts:
        print(f"    · {txt}")

    return {"ocr_result": {
        "by_shot": by_shot,
        "all_texts": all_texts,
        "timeline": timeline,
    }}
