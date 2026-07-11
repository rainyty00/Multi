"""
=================================================================
视觉理解 Agent（阶段3 · 支线三）—— 调用 Gemini 多模态，看图说话
=================================================================
职责：对每个镜头的关键帧，调 VISION_MODEL（Gemini）分析出：
  - visual：画面内容（客观描述看到了什么）
  - camera：镜头语言（景别/运镜/角度）
  - style ：画面风格标签
产出写进 state["visual_result"] = { 镜号: {visual, camera, style} }

★这是三支线里唯一花钱的（调云 API）。★核心原则：只做"忠实描述"，
  不要脑补画面里没有的东西。
=================================================================
"""
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI

from src.state import GraphState, Shot
from config import (VISION_MODEL, VISION_API_KEY, VISION_BASE_URL,
                    VISION_PROXY, VL_CONCURRENCY)


# ★给 Gemini 客户端单独配代理（只有这一路走代理，国内接口不受影响）
_http = httpx.Client(proxy=VISION_PROXY, timeout=60) if VISION_PROXY else None
_client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL, http_client=_http)


# ---------- 把本地图片转成 base64 的 data URL ----------
def image_to_data_url(image_path: str) -> str:
    """
    多模态接口要求图片以 URL 形式传入。本地图片没有网址，
    就编码成 base64 的 data URL（data:image/jpeg;base64,....）直接内嵌。
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------- 从模型返回里抽出 JSON ----------
def parse_json(text: str) -> dict:
    """
    模型有时会用 ```json ... ``` 包裹，或前后带多余文字。
    这里尽量稳地把中间的 JSON 抠出来解析。
    """
    text = text.strip()
    # 去掉可能的 ```json 包裹
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    # 找到第一个 { 和最后一个 }，取中间
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        # 解析失败就原样返回，避免整个流程崩掉
        return {"visual": text, "camera": "", "style": ""}


# ---------- 分析单个镜头 ----------
def analyze_shot(shot: Shot, roster_text: str = "") -> dict:
    """
    对一个镜头的关键帧调用 Gemini，返回 {visual, camera, style}。
    roster_text：全局总览产出的"角色清单"，让描述用统一称呼（保证跨镜头一致性）。
    """
    if not shot.keyframes:
        return {"visual": "（无关键帧）", "camera": "", "style": ""}

    # 有角色清单就作为上下文加进去
    roster_hint = ""
    if roster_text:
        roster_hint = (
            f"\n【已知本片主要角色（请用这些统一称呼）】{roster_text}\n"
            "描述画面时，若出现这些角色，务必用上面的统一称呼（如\"老奶奶\"），"
            "不要每次换一种说法。"
        )

    prompt = (
        "你是专业的广告分镜分析师。请仔细观察这张广告视频截图，"
        "只做客观描述，不要脑补画面里没有的内容。"
        + roster_hint +
        "\n用 JSON 返回，字段如下：\n"
        '{\n'
        '  "visual": "画面内容：主体是谁/什么、在做什么、场景环境（一两句话）",\n'
        '  "camera": "镜头语言：景别(特写/近景/中景/全景)、机位角度、有无运镜",\n'
        '  "style": "画面风格：如 手绘动画/写实实拍/扁平插画 等标签"\n'
        '}\n'
        "只输出 JSON，不要多余文字。"
    )

    resp = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(shot.keyframes[0])}},
            ],
        }],
        temperature=0.2,   # 低温度：描述更稳定、少发挥
    )
    answer = resp.choices[0].message.content
    return parse_json(answer)


# ---------- 节点主函数 ----------
def visual_node(state: GraphState) -> dict:
    shots: list[Shot] = state["shots"]

    # 从全局总览拿"角色清单"，拼成一段文字，喂给每镜精读
    overview = state.get("overview", {})
    chars = overview.get("characters", [])
    roster_text = "；".join(f"{c.get('label','')}={c.get('desc','')}" for c in chars)
    print(f"[视觉 Agent] 开始逐镜精读 {len(shots)} 个镜头"
          f"{'（带角色清单）' if roster_text else ''}"
          f"（{VISION_MODEL}，并发 {VL_CONCURRENCY}）...")

    # ★逐镜并发：每镜的上下文都来自同一份"角色清单"、互不依赖，所以能安全并行。
    #   用线程池限流（VL_CONCURRENCY），避免无脑并发撞 API 限流(429)。
    #   ★结果按【镜号】归位到字典，不依赖返回顺序（谁先回来先收谁）。
    visual_result: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=VL_CONCURRENCY) as pool:
        future_to_shot = {pool.submit(analyze_shot, s, roster_text): s for s in shots}
        for future in as_completed(future_to_shot):
            shot = future_to_shot[future]
            try:
                info = future.result()
            except Exception as e:      # 单镜失败不拖垮整条支线
                print(f"    ⚠️ 镜{shot.index} 分析失败：{e}")
                info = {"visual": "（分析失败）", "camera": "", "style": ""}
            visual_result[shot.index] = info

    # 按镜号排序打印，便于阅读
    for idx in sorted(visual_result):
        info = visual_result[idx]
        print(f"    镜{idx}: [{info.get('style','')}] {info.get('visual','')[:40]}...")

    return {"visual_result": visual_result}
