"""
=================================================================
分镜合成 Agent（阶段4.B + 5.B）★核心
=================================================================
两种工作模式：
  · 首次合成：把所有镜头的证据包 → 生成完整分镜表
  · 定向重写：评估不合格时，只重写"不合格的镜头"，并带上评估反馈，
             通过的镜头保持不动。每重写一次给该镜的 retry_count +1。

★核心原则：忠实还原，不脑补。证据里没有的画面/台词，不许编。
=================================================================
"""
import json

from openai import OpenAI

from src.state import GraphState, StoryboardRow
from config import TEXT_MODEL, TEXT_API_KEY, TEXT_BASE_URL, COMPOSE_BATCH_SIZE

_client = OpenAI(api_key=TEXT_API_KEY, base_url=TEXT_BASE_URL)


def parse_json_array(text: str) -> list:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return []


# ---------- 构造提示词（feedback 非空时是"定向重写"）----------
def build_prompt(aligned: list, feedback: dict | None = None) -> str:
    evidence = json.dumps(aligned, ensure_ascii=False, indent=2)

    # 定向重写时，把上一版的扣分原因附上，让模型针对性改进
    fb_text = ""
    if feedback:
        lines = [f"  镜{idx}: {reason}" for idx, reason in feedback.items()]
        fb_text = ("\n【上一版评估发现的问题，请针对性改进这些镜头】\n"
                   + "\n".join(lines) + "\n")

    return f"""你是专业的广告分镜脚本还原师。下面是从广告视频里逐镜头提取的原始素材（证据包）：

{evidence}
{fb_text}
请为每个镜头还原出一行标准分镜脚本。要求：
1. **忠实还原，严禁脑补**：证据里没有的画面、台词，一律不许编造。
2. **清洗屏幕文字 on_screen_text**：从 ocr_texts 里挑出真正的广告文字（slogan、字幕）。规则：
   - 合并被 OCR 打散的**同一句话**的碎片（如 "JUSTDOIT"/"JUS" → "JUST DO IT."）；
   - **删掉水印**（如 "LAZY SQUARE"）和背景装饰噪声（挂画、电脑屏幕上的无关字）；
   - ★**删掉孤立的、无意义的字符碎片**（如 "T&"、"AZVI"、"SOHAL"、"inota" 这类 OCR 误识的乱码）。
     它们是识别错误，画面里并不存在——**绝对不要把它们拼接到 slogan 前后**
     （例如不许输出 "T& Skittles彩虹糖"，正确的是 "Skittles彩虹糖"）；
   - 宁可少写，也不要写画面里没有的文字。
3. **narrative 叙事作用**：一句话推断这一镜的作用（钩子/痛点/卖点/使用场景/情绪渲染/行动号召CTA）。
4. **camera 镜头语言：原样照搬证据里的 camera，不许精简、不许删字段**。
   它包含【景别 + 机位角度 + 色调】三要素（如"中景，平视，色调明亮偏冷"），
   ★三个都要保留，尤其不要把"色调"砍掉。
5. visual 用画面内容（可精简，不可加新信息），voiceover 用口播（没有留空）。

只输出一个 JSON 数组，每元素对应一个镜头，字段：
[{{"index":镜号, "camera":"", "visual":"", "voiceover":"", "on_screen_text":"", "narrative":""}}]
不要输出 JSON 以外的任何文字。"""


def _row_from_data(row: dict, aligned: list, fallback_index: int) -> StoryboardRow:
    """把模型返回的一条数据，套进 StoryboardRow（补上时间段、证据帧）。"""
    idx = row.get("index", fallback_index)
    ev = next((a for a in aligned if a["index"] == idx), {})
    return StoryboardRow(
        index=idx,
        time_range=ev.get("time_range", ""),
        camera=row.get("camera", ""),
        visual=row.get("visual", ""),
        voiceover=row.get("voiceover", ""),
        on_screen_text=row.get("on_screen_text", ""),
        narrative=row.get("narrative", ""),
        # ★双证据帧：代表帧核对画面/镜头语言，文字帧核对屏幕文字
        evidence_frame=ev.get("keyframe", ""),
        text_frame=ev.get("text_frame", ""),
    )


def _generate_once(batch: list, feedback: dict | None) -> list[dict]:
    """调一次 LLM，处理一批镜头。"""
    resp = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": build_prompt(batch, feedback)}],
        temperature=0.3,
    )
    return parse_json_array(resp.choices[0].message.content)


def _generate(aligned_subset: list, feedback: dict | None) -> list[dict]:
    """
    生成分镜行。★镜头很多时自动分批——每批 COMPOSE_BATCH_SIZE 个，
    避免一次性塞太多把模型上下文撑爆（导致返回截断、JSON 解析失败、空表）。
    分批调用后把各批结果拼起来。
    """
    n = len(aligned_subset)
    if n <= COMPOSE_BATCH_SIZE:
        return _generate_once(aligned_subset, feedback)

    # 分批
    all_rows = []
    total_batches = (n + COMPOSE_BATCH_SIZE - 1) // COMPOSE_BATCH_SIZE
    for bi, i in enumerate(range(0, n, COMPOSE_BATCH_SIZE), start=1):
        batch = aligned_subset[i:i + COMPOSE_BATCH_SIZE]
        print(f"    分批合成 {bi}/{total_batches}（本批 {len(batch)} 镜）...")
        try:
            all_rows.extend(_generate_once(batch, feedback))
        except Exception as e:
            print(f"    ⚠️ 第 {bi} 批合成失败：{e}")   # 单批失败不拖垮整体
    return all_rows


# ---------- 节点主函数 ----------
def compose_node(state: GraphState) -> dict:
    aligned = state["aligned"]
    eval_report = state.get("eval_report")
    prev_storyboard = state.get("storyboard")
    retry_count = dict(state.get("retry_count", {}))

    # 判断是不是"定向重写"模式：有评估结果、没通过、且已有上一版分镜
    is_retry = bool(eval_report) and not eval_report.get("passed") and bool(prev_storyboard)

    if is_retry:
        failed = eval_report["failed_shots"]
        feedback = eval_report["feedback"]
        targets = [a for a in aligned if a["index"] in failed]   # 只重写不合格的镜头
        print(f"[分镜合成] 🔁定向重写 {len(targets)} 个不合格镜头 {failed}（带评估反馈）")

        rows_data = _generate(targets, feedback)

        # 合并：从上一版出发，替换掉被重写的那几镜
        storyboard = list(prev_storyboard)
        pos = {r.index: i for i, r in enumerate(storyboard)}
        for rd in rows_data:
            new_row = _row_from_data(rd, aligned, rd.get("index"))
            if new_row.index in pos:
                storyboard[pos[new_row.index]] = new_row
        # 给这些镜头的重写次数 +1
        for idx in failed:
            retry_count[idx] = retry_count.get(idx, 0) + 1
    else:
        print(f"[分镜合成] 用 {TEXT_MODEL} 首次合成 {len(aligned)} 个镜头的分镜表...")
        rows_data = _generate(aligned, None)
        storyboard = [_row_from_data(rd, aligned, i + 1) for i, rd in enumerate(rows_data)]

    print(f"[分镜合成] 完成，共 {len(storyboard)} 行:")
    for r in storyboard:
        print(f"    镜{r.index} [{r.narrative}] 文字:{r.on_screen_text}")

    return {"storyboard": storyboard, "retry_count": retry_count}
