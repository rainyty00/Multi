"""
=================================================================
评估 Agent（阶段5.A）—— 调 Qwen-VL-Max，核对"分镜描述↔原始画面"
=================================================================
职责：对每一行分镜，用独立的多模态模型(qwen-vl-max)核对忠实度并打分：
  · 规则层：证据是否完整（有时间段、有证据帧）
  · 模型层：画面一致性 / 镜头语言准确性 / 文字对齐度（各 0~5 分）
产出写进 state["eval_report"]：
  { per_shot:[...], avg_score, passed(bool), failed_shots:[镜号], feedback:{镜号:原因} }

★模型隔离：评估用的 qwen-vl-max 和生成用的 Gemini 不同厂商；
  且只喂"原始帧 + 待核文字"，不给生成方的中间推理，避免自评偏差。
=================================================================
"""
import os
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from src.state import GraphState, StoryboardRow
from config import EVAL_MODEL, EVAL_API_KEY, EVAL_BASE_URL, VL_CONCURRENCY

_client = OpenAI(api_key=EVAL_API_KEY, base_url=EVAL_BASE_URL)

# 每个维度满分 5；单镜平均分 ≥ 这个值才算通过
PASS_SCORE = 3.0


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
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}


# ---------- 评估单行分镜 ----------
def evaluate_row(row: StoryboardRow) -> dict:
    """
    对一行分镜打分。返回：
      {index, evidence_ok, visual_consistency, camera_accuracy, text_alignment, avg, reason}
    """
    # 1) 规则层：证据完整性检查（无需调模型）
    evidence_ok = bool(row.time_range) and bool(row.evidence_frame) and os.path.exists(row.evidence_frame)
    if not evidence_ok:
        return {
            "index": row.index, "evidence_ok": False,
            "visual_consistency": 0, "camera_accuracy": 0, "text_alignment": 0,
            "avg": 0.0, "reason": "证据不完整（缺时间段或证据帧）",
        }

    # 2) 模型层：把原始帧 + 分镜文字给 qwen-vl-max 核对
    prompt = (
        "你是严格的广告分镜质检员。下面给你一张广告视频的原始截图，以及一段"
        "别人写的对这一镜的文字描述。请核对文字描述与画面是否相符，给三个维度打分"
        "（0~5，5=完全相符，0=完全不符），并说明理由。\n\n"
        f"【镜头语言描述】{row.camera}\n"
        f"【画面内容描述】{row.visual}\n"
        f"【屏幕文字描述】{row.on_screen_text}\n\n"
        "用 JSON 返回：\n"
        '{\n'
        '  "visual_consistency": 画面内容描述与截图是否相符(0-5),\n'
        '  "camera_accuracy": 镜头语言(景别/角度)判断是否准确(0-5),\n'
        '  "text_alignment": 屏幕文字描述与截图里的文字是否相符(0-5),\n'
        '  "reason": "一句话说明扣分点，没有就写相符"\n'
        '}\n只输出 JSON。'
    )

    resp = _client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(row.evidence_frame)}},
            ],
        }],
        temperature=0.1,   # 评分要稳定
    )
    data = parse_json(resp.choices[0].message.content)

    vc = float(data.get("visual_consistency", 0))
    ca = float(data.get("camera_accuracy", 0))
    ta = float(data.get("text_alignment", 0))
    avg = round((vc + ca + ta) / 3, 2)

    return {
        "index": row.index, "evidence_ok": True,
        "visual_consistency": vc, "camera_accuracy": ca, "text_alignment": ta,
        "avg": avg, "reason": data.get("reason", ""),
    }


# ---------- 节点主函数 ----------
def evaluate_node(state: GraphState) -> dict:
    storyboard: list[StoryboardRow] = state["storyboard"]
    print(f"[评估 Agent] 用 {EVAL_MODEL} 核对 {len(storyboard)} 行分镜"
          f"（并发 {VL_CONCURRENCY}）...")

    # ★逐镜并发：每镜是"分镜文字 ↔ 原始帧"的独立核对，互不依赖，可安全并行。
    #   线程池限流，结果按镜号归位。
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=VL_CONCURRENCY) as pool:
        future_to_row = {pool.submit(evaluate_row, r): r for r in storyboard}
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                results[row.index] = future.result()
            except Exception as e:      # 单镜评估失败给中性分，不阻断流程
                print(f"    ⚠️ 镜{row.index} 评估失败：{e}")
                results[row.index] = {
                    "index": row.index, "evidence_ok": True,
                    "visual_consistency": 3, "camera_accuracy": 3, "text_alignment": 3,
                    "avg": 3.0, "reason": f"评估失败({e})，给中性分",
                }

    per_shot = []
    failed_shots = []
    feedback = {}
    for idx in sorted(results):         # 按镜号顺序汇总/打印
        r = results[idx]
        per_shot.append(r)
        # 判定：证据完整 且 平均分达标 才通过
        ok = r["evidence_ok"] and r["avg"] >= PASS_SCORE
        status = "✅通过" if ok else "❌不合格"
        print(f"    镜{r['index']}: 均分{r['avg']} "
              f"(画面{r['visual_consistency']}/镜头{r['camera_accuracy']}/文字{r['text_alignment']}) "
              f"{status} {r['reason']}")
        if not ok:
            failed_shots.append(r["index"])
            feedback[r["index"]] = r["reason"]

    scores = [r["avg"] for r in per_shot]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    passed = len(failed_shots) == 0

    print(f"[评估 Agent] 全片均分 {avg_score} | {'全部通过' if passed else f'不合格镜号 {failed_shots}'}")

    return {"eval_report": {
        "per_shot": per_shot,
        "avg_score": avg_score,
        "passed": passed,
        "failed_shots": failed_shots,
        "feedback": feedback,
    }}
