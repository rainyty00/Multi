"""
=================================================================
音频理解 Agent（阶段3 · 支线一）—— 本地、免费
=================================================================
职责：
  1. 用 faster-whisper 把 .wav 转成文字（带时间戳，★自动识别中/英文）
  2. 算语速（中文按"字/分"，英文按"词/分"）
  3. 提关键词（中文用 jieba，英文用词频法）
产出写进 state["audio_result"]：
  { full_text, segments, speech_rate, rate_unit, keywords, lang }
=================================================================
"""
import re
from collections import Counter

from faster_whisper import WhisperModel
import jieba.analyse

from src.state import GraphState
from config import ASR_MODEL

# 常见英文停用词（提关键词时过滤掉这些没信息量的词）
_EN_STOP = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have",
    "has", "had", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "s", "t", "m", "re", "ll", "ve", "d", "not", "no", "yes", "okay", "ok", "just", "so",
    "can", "will", "would", "should", "could", "let", "go", "get", "got", "up", "out",
    "please", "here", "there", "now", "then", "cmon", "mon", "gonna", "wanna",
}


_model = None
def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"[音频 Agent] 加载本地 whisper 模型: {ASR_MODEL}")
        _model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    return _model


# ---------- 语音转写（自动识别语言）----------
def transcribe(audio_path: str) -> tuple[list[dict], str]:
    """
    转写音频。返回：(分段列表, 语言代码如'zh'/'en')
    ★language 不再写死 'zh'，改成自动识别，中英文广告都能处理。
    ★vad_filter 跳过纯音乐，避免无人声时幻觉。
    """
    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    seg_list = []
    for seg in segments:   # segments 是生成器，遍历才真正转写
        seg_list.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
    return seg_list, info.language


def join_text(seg_list: list[dict], lang: str) -> str:
    """拼完整台词：英文分段间加空格，中文不加。"""
    texts = [s["text"] for s in seg_list]
    sep = " " if lang == "en" else ""
    return sep.join(texts).strip()


# ---------- 算语速 ----------
def calc_speech_rate(seg_list: list[dict], lang: str) -> tuple[float, str]:
    """中文按'字/分'，英文按'词/分'。返回 (数值, 单位)。"""
    speaking = sum(s["end"] - s["start"] for s in seg_list)
    if speaking <= 0:
        return 0.0, ("词/分" if lang == "en" else "字/分")
    if lang == "en":
        units = sum(len(re.findall(r"[A-Za-z']+", s["text"])) for s in seg_list)  # 词数
        return round(units / speaking * 60, 1), "词/分"
    units = sum(len(s["text"]) for s in seg_list)  # 字数
    return round(units / speaking * 60, 1), "字/分"


# ---------- 提关键词 ----------
def extract_keywords(full_text: str, lang: str, topk: int = 10) -> list[str]:
    """中文用 jieba TF-IDF；英文用去停用词后的词频统计。"""
    if not full_text.strip():
        return []
    if lang == "en":
        words = re.findall(r"[a-z']{3,}", full_text.lower())         # 取长度≥3的英文词
        words = [w.strip("'") for w in words if w not in _EN_STOP]   # 去停用词
        return [w for w, _ in Counter(words).most_common(topk)]
    return jieba.analyse.extract_tags(full_text, topK=topk)


# ---------- 节点主函数 ----------
def audio_node(state: GraphState) -> dict:
    audio_path = state["audio_path"]
    print(f"[音频 Agent] 开始转写: {audio_path}")

    seg_list, lang = transcribe(audio_path)
    full_text = join_text(seg_list, lang)
    speech_rate, rate_unit = calc_speech_rate(seg_list, lang)
    keywords = extract_keywords(full_text, lang)

    print(f"[音频 Agent] 语言:{lang} | 台词: {full_text[:60]}{'...' if len(full_text) > 60 else ''}")
    print(f"[音频 Agent] 分段:{len(seg_list)} | 语速:{speech_rate}{rate_unit} | 关键词:{keywords}")

    return {"audio_result": {
        "full_text": full_text,
        "segments": seg_list,
        "speech_rate": speech_rate,
        "rate_unit": rate_unit,
        "keywords": keywords,
        "lang": lang,
    }}
