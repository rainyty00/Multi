"""
=================================================================
全局总览 Agent（"略读"）—— 一次看几张代表帧，建立"整片认知"
=================================================================
作用（对应"先总览略读，再精读"里的略读）：
  在逐镜精读之前，先让 Gemini 一次性看全片的几张代表帧，产出：
    · 作品概览：产品/品牌、主题、主要人物、主要场景、核心卖点、目标受众
    · 角色清单：给反复出现的人物起统一标签（老奶奶/医生…）
  这份结果会喂给：
    · 视觉 Agent（精读时用统一称呼，解决"同一人被描述成不同人"）
    · 报告 Agent（把作品概览显示出来）

★只调用 1 次（看几张图），不是每镜一次，所以不会让费用翻倍。
=================================================================
"""
import base64
import json

import httpx
from openai import OpenAI

from src.state import GraphState, Shot
from config import VISION_MODEL, VISION_API_KEY, VISION_BASE_URL, VISION_PROXY

# 和视觉 Agent 用同一个模型/代理（Gemini）
_http = httpx.Client(proxy=VISION_PROXY, timeout=90) if VISION_PROXY else None
_client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL, http_client=_http)


def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


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


def sample_frames(shots: list[Shot], max_n: int = 8) -> list[str]:
    """从所有镜头的代表帧里，均匀挑最多 max_n 张给略读用（够看懂全片即可）。"""
    frames = [s.keyframes[0] for s in shots if s.keyframes]
    if len(frames) <= max_n:
        return frames
    step = len(frames) / max_n
    return [frames[int(i * step)] for i in range(max_n)]


def overview_node(state: GraphState) -> dict:
    shots: list[Shot] = state["shots"]
    frames = sample_frames(shots)
    if not frames:
        print("[全局总览] 无可用帧，跳过")
        return {"overview": {}}

    print(f"[全局总览] 一次性看 {len(frames)} 张代表帧，建立整片认知（调用 {VISION_MODEL}）...")

    prompt = (
        "下面是同一条广告视频里、按时间顺序抽取的若干代表帧。请通读这些画面，"
        "综合判断这条广告的整体信息。用 JSON 返回（看不出就写\"未知\"，不要脑补）：\n"
        "{\n"
        '  "product": "广告推广的产品/品牌（结合画面文字、logo、主体判断）",\n'
        '  "theme": "一句话概括广告的主题/故事",\n'
        '  "characters": [{"label":"统一称呼(如 老奶奶)", "desc":"外形特征(白发/遮阳帽/青绿运动服)"}],\n'
        '  "scenes": ["主要场景1", "主要场景2"],\n'
        '  "selling_points": ["核心卖点1", "核心卖点2"],\n'
        '  "audience": "目标受众"\n'
        "}\n"
        "characters 只列反复出现的主要人物，给它们固定统一的称呼。只输出 JSON。"
    )

    content = [{"type": "text", "text": prompt}]
    for f in frames:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(f)}})

    try:
        resp = _client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
        )
        data = parse_json(resp.choices[0].message.content)
    except Exception as e:
        # 略读失败不阻断流程，降级为空（精读就退回原来的独立分析）
        print(f"[全局总览] 失败，降级跳过：{e}")
        return {"overview": {}}

    chars = data.get("characters", [])
    print(f"[全局总览] 产品:{data.get('product','?')} | 主要人物:{[c.get('label') for c in chars]}")
    return {"overview": data}
