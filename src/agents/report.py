"""
=================================================================
分析报告 Agent（阶段6.A）—— 需求③配套分析报告
=================================================================
产出写进 state["report"]：
  · shot_stats     镜头统计（数量/总时长/平均时长）
  · audio_summary  音频：语速 + 关键词
  · styles         各镜画面风格标签
  · image_text_score 图文匹配分（取自评估的全片均分）
  · style_summary  全片画面风格总结（LLM 生成）
  · creative_summary 创意套路总结（LLM 基于各镜叙事作用归纳）

统计类是确定性计算（不花钱）；两段总结用 DeepSeek 生成。
=================================================================
"""
import json
import time

from openai import OpenAI

from src.state import GraphState
from config import TEXT_MODEL, TEXT_API_KEY, TEXT_BASE_URL

_client = OpenAI(api_key=TEXT_API_KEY, base_url=TEXT_BASE_URL)


def parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        text = text[s:e + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}


def report_node(state: GraphState) -> dict:
    shots = state["shots"]
    audio = state.get("audio_result", {})
    visual = state.get("visual_result", {})
    ocr = state.get("ocr_result", {})
    storyboard = state.get("storyboard", [])
    eval_report = state.get("eval_report", {})
    overview = state.get("overview", {})   # 全局总览：这里主要用它的"角色清单"
    meta = state["metadata"]

    print("[报告 Agent] 生成分析报告...")

    # 1) 镜头统计（确定性）
    count = len(shots)
    total = meta.duration
    shot_stats = {
        "count": count,
        "total_duration": total,
        "avg_duration": round(total / count, 2) if count else 0.0,
    }

    # 2) 音频摘要 + 各镜风格标签（现成数据）
    audio_summary = {
        "speech_rate": audio.get("speech_rate", 0.0),
        "rate_unit": audio.get("rate_unit", "字/分"),
        "keywords": audio.get("keywords", []),
    }
    styles = [v.get("style", "") for v in visual.values()]

    # 3) 图文匹配分（取评估的全片均分）
    image_text_score = eval_report.get("avg_score", 0.0)

    # 4) 作品概览 + 两段总结（LLM）：★放在最后做，此时"画面全部文字"已就绪，
    #    里面白纸黑字有品牌名（如 Skittles彩虹糖），能纠正"只看图猜错产品"的问题。
    brief = [
        {"镜": r.index, "画面": r.visual, "口播": r.voiceover,
         "屏幕文字": r.on_screen_text, "叙事作用": r.narrative}
        for r in storyboard
    ]
    all_texts = ocr.get("all_texts", [])
    roster = overview.get("characters", [])

    prompt = (
        f"下面是广告视频《{meta.title}》的完整拆解信息。\n"
        f"【逐镜信息】{json.dumps(brief, ensure_ascii=False)}\n"
        f"【画面里出现的全部文字】{json.dumps(all_texts, ensure_ascii=False)}\n"
        f"【已识别的主要角色】{json.dumps(roster, ensure_ascii=False)}\n\n"
        "请综合判断，用 JSON 返回（客观、看不出就写\"未知\"、不要脑补）：\n"
        "{\n"
        '  "product": "广告推广的真实产品/品牌。★重点看【画面里出现的全部文字】中的品牌名/logo'
        '来判断；不要被剧情表象带偏——广告常用隐喻（比如用给宠物美容作比喻，产品其实是某零食）",\n'
        '  "theme": "一句话概括广告的故事/主题",\n'
        '  "scenes": ["主要场景1", "主要场景2"],\n'
        '  "selling_points": ["核心卖点1", "核心卖点2"],\n'
        '  "audience": "目标受众",\n'
        '  "style_summary": "全片画面风格总结(1~2句)",\n'
        '  "creative_summary": "创意套路总结：叙事结构/钩子/卖点节奏(1~3句)"\n'
        "}\n只输出 JSON。"
    )
    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    summary = parse_json(resp.choices[0].message.content)

    # 作品概览：产品/主题/场景/卖点/受众来自这次综合；角色沿用视觉产出的清单
    overview_final = {
        "product": summary.get("product", "未知"),
        "theme": summary.get("theme", ""),
        "characters": roster,
        "scenes": summary.get("scenes", []),
        "selling_points": summary.get("selling_points", []),
        "audience": summary.get("audience", ""),
    }

    # 计时：从流程开始到现在的耗时（秒）
    start_ts = state.get("start_ts", 0)
    elapsed = round(time.time() - start_ts, 1) if start_ts else 0.0

    report = {
        "overview": overview_final,   # 作品概览（最后综合，产品判断更准）
        "shot_stats": shot_stats,
        "audio_summary": audio_summary,
        "styles": styles,
        "image_text_score": image_text_score,
        "style_summary": summary.get("style_summary", ""),
        "creative_summary": summary.get("creative_summary", ""),
        "elapsed_sec": elapsed,
    }

    print(f"[报告 Agent] 产品:{overview_final['product']} | 镜头{count}个 | "
          f"图文匹配分{image_text_score}")

    return {"report": report}
