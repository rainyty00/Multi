"""
=================================================================
说话人标注 Agent（方案③ LLM 推测）—— 给台词加"人物/旁白"标注
=================================================================
思路（对应"确保准确度"的手段）：
  1. 每句台词按时间戳定位到所属镜头 → 附上"该镜画面里有谁"（视觉描述）
  2. 让 LLM 先梳理固定角色表，再从表里给每句选说话人
  3. 不确定的加 (?)，画外音标"旁白"
产出写进 state["dialogue"] = [{time, speaker, text}]。

★这是"推测"，不是真值。台词文件里会明确注明。
=================================================================
"""
import json

from openai import OpenAI

from src.state import GraphState, Shot
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


def find_shot_index(t: float, shots: list[Shot]) -> int:
    for s in shots:
        if s.start <= t <= s.end:
            return s.index
    return shots[-1].index if shots else 1


def speaker_node(state: GraphState) -> dict:
    audio = state.get("audio_result", {})
    segments = audio.get("segments", [])

    # 没有口播就跳过（比如纯音乐广告）
    if not segments:
        print("[说话人] 无口播，跳过标注")
        return {"dialogue": []}

    shots: list[Shot] = state["shots"]
    visual = state.get("visual_result", {})

    # 1) 每镜"画面里有谁"（给 LLM 做锚定的上下文）
    shot_ctx = {s.index: visual.get(s.index, {}).get("visual", "") for s in shots}

    # 2) 台词逐句，标上它所属的镜号
    lines = []
    for i, seg in enumerate(segments):
        sid = find_shot_index(seg["start"], shots)
        lines.append({"行号": i, "镜号": sid, "台词": seg["text"]})

    print(f"[说话人] 用 {TEXT_MODEL} 推测 {len(lines)} 句台词的说话人...")

    prompt = (
        "你是影视对白归属分析师。下面是一条广告的资料。\n\n"
        f"【各镜头画面里出现了谁】\n{json.dumps(shot_ctx, ensure_ascii=False)}\n\n"
        f"【对白列表（按时间顺序，已标所属镜号）】\n{json.dumps(lines, ensure_ascii=False)}\n\n"
        "请完成：\n"
        "1. 先根据画面梳理出角色清单（如 年长女性/年轻男性/医生/旁白 等）。\n"
        "2. 为每句台词推断最可能的说话人，依据：该句所在镜头出现的人物、台词内容与逻辑、上下文连贯。\n"
        "3. 拿不准的说话人后面加 (?)。画外音/无明确出镜说话人标“旁白”。\n\n"
        "只输出 JSON：\n"
        '{"roster": ["角色1","角色2",...], '
        '"lines": [{"行号": 数字, "speaker": "说话人", "text": "台词"}]}'
    )

    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    data = parse_json(resp.choices[0].message.content)

    # 3) 把说话人结果配回时间戳
    dialogue = []
    for rl in data.get("lines", []):
        idx = rl.get("行号")
        seg = segments[idx] if isinstance(idx, int) and 0 <= idx < len(segments) else None
        dialogue.append({
            "time": f"{seg['start']}-{seg['end']}" if seg else "",
            "speaker": rl.get("speaker", "?"),
            "text": rl.get("text", ""),
        })

    print(f"[说话人] 角色清单: {data.get('roster', [])}")
    return {"dialogue": dialogue}
