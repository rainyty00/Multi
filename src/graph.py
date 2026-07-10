"""
=================================================================
graph.py —— 用 LangGraph 把各个节点连成「流程图」
=================================================================
LangGraph 的三个核心动作（★理解这三步就懂 LangGraph 了）：
  1. 定义"节点"(node)：就是一个普通函数，输入 state，返回要更新到 state 的字段
  2. 用 add_node / add_edge 把节点连起来，决定"谁先谁后"
  3. compile() 编译成可运行的 app，然后 app.invoke(初始状态) 就跑起来

阶段0：我们先放两个"占位节点"，不干真活，只打印和写状态，
        让你看清楚"状态在节点间流动"。后面阶段再把真正的 Agent 换进来。
=================================================================
"""
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from src.state import GraphState, VideoMeta, Shot, StoryboardRow
from config import MAX_RETRY, CHECKPOINT_DB
from src.agents.ingest import ingest_node        # 阶段1：采集 Agent
from src.nodes.preprocess import preprocess_node  # 阶段2：预处理节点
from src.agents.overview import overview_node      # 全局总览（略读）
from src.agents.audio import audio_node           # 阶段3：音频 Agent
from src.agents.ocr import ocr_node                # 阶段3：OCR Agent
from src.agents.visual import visual_node          # 阶段3：视觉 Agent
from src.nodes.align import align_node              # 阶段4：时序对齐（三支线汇合点）
from src.agents.compose import compose_node          # 阶段4：分镜合成 Agent
from src.agents.evaluate import evaluate_node         # 阶段5：评估 Agent
from src.agents.speaker import speaker_node           # 阶段6：说话人标注 Agent
from src.agents.report import report_node             # 阶段6：分析报告 Agent
from src.nodes.export import export_node               # 阶段6：导出节点


# ---------- 条件路由：评估之后往哪走 ----------
def route_after_eval(state: GraphState) -> str:
    """
    评估后的分岔判断（这就是"反馈循环+条件路由"的核心）：
      · 全部通过           → "done"（继续/结束）
      · 有不合格 且 还能重试 → "retry"（退回分镜合成定向重写）
      · 有不合格 但 重试用完 → "done"（降级出稿，不死循环）
    """
    report = state["eval_report"]
    if report["passed"]:
        return "done"
    retry_count = state.get("retry_count", {})
    # 只要还有任一不合格镜头没到重试上限，就继续重试
    can_retry = any(retry_count.get(idx, 0) < MAX_RETRY for idx in report["failed_shots"])
    if can_retry:
        return "retry"
    print(f"[路由] 重试已达上限({MAX_RETRY}次)，剩余不合格镜头降级出稿：{report['failed_shots']}")
    return "done"


# ---------- 把节点连成图 ----------
def build_graph():
    """
    组装并编译流程图。
    START 是内置的起点，END 是内置的终点。
    """
    # 1) 新建一个图，告诉它状态的类型是 GraphState
    workflow = StateGraph(GraphState)

    # 2) 注册节点：add_node("节点名字", 对应的函数)
    workflow.add_node("ingest", ingest_node)          # 阶段1：采集 Agent
    workflow.add_node("preprocess", preprocess_node)  # 阶段2：预处理节点
    workflow.add_node("overview", overview_node)      # 全局总览（略读）
    workflow.add_node("audio", audio_node)            # 阶段3：音频 Agent（暂时串行测试）
    workflow.add_node("ocr", ocr_node)                # 阶段3：OCR Agent
    workflow.add_node("visual", visual_node)          # 阶段3：视觉 Agent
    workflow.add_node("align", align_node)            # 阶段4：三支线汇合点
    workflow.add_node("compose", compose_node)        # 阶段4：分镜合成 Agent
    workflow.add_node("evaluate", evaluate_node)      # 阶段5：评估 Agent
    workflow.add_node("speaker", speaker_node)        # 阶段6：说话人标注 Agent
    workflow.add_node("report", report_node)          # 阶段6：分析报告 Agent
    workflow.add_node("export", export_node)          # 阶段6：导出节点

    # 3) 连边
    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "preprocess")

    # 预处理后先做"全局总览"（略读），再扇出三支线
    workflow.add_edge("preprocess", "overview")

    # ★并发扇出(fan-out)：全局总览之后，三条支线同时开跑
    workflow.add_edge("overview", "audio")
    workflow.add_edge("overview", "ocr")
    workflow.add_edge("overview", "visual")

    # ★并发扇入(fan-in)：三条支线都汇合到 align；
    #   LangGraph 会自动等三条全部完成，才执行 align
    workflow.add_edge("audio", "align")
    workflow.add_edge("ocr", "align")
    workflow.add_edge("visual", "align")

    workflow.add_edge("align", "compose")
    workflow.add_edge("compose", "evaluate")

    # ★条件边：评估后按 route_after_eval 的返回值决定去向
    #   "retry" → 回 compose 定向重写；"done" → 去分析报告
    workflow.add_conditional_edges("evaluate", route_after_eval, {
        "retry": "compose",
        "done": "speaker",
    })
    workflow.add_edge("speaker", "report")
    workflow.add_edge("report", "export")
    workflow.add_edge("export", END)

    # 4) 编译成可运行的 app，★挂上持久化 checkpointer（断点恢复）
    #    check_same_thread=False：因为三支线并发会用到多线程，连接要允许跨线程
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    # ★把我们自定义的数据类型显式登记，消除 langgraph 的"未注册类型"反序列化警告。
    #   LangGraph 的内置安全类型不受影响，仍照常放行。
    serde = JsonPlusSerializer(allowed_msgpack_modules=[VideoMeta, Shot, StoryboardRow])
    checkpointer = SqliteSaver(conn, serde=serde)
    return workflow.compile(checkpointer=checkpointer)