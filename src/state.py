"""
=================================================================
GraphState —— 整个项目的「数据主干」
=================================================================
核心概念（★必须理解）：
  LangGraph 里各个 Agent 不直接互相调用，而是共享这一个"状态字典"。
  每个节点：读取自己需要的字段 → 干活 → 把产出写回状态 → 传给下一个节点。

  所以这个文件定义的，就是"整个流程里会流动哪些数据"。
  现在很多字段是空的，我们会在后面每个阶段把它们逐步填满。
=================================================================
"""
from typing import TypedDict, List, Dict, Optional
from pydantic import BaseModel


# ---------- 一些结构化的小数据模型（用 pydantic 定义，带类型校验） ----------

class VideoMeta(BaseModel):
    """需求①：基础视频元数据"""
    platform: str = ""       # 平台：bilibili / xiaohongshu / douyin / local
    title: str = ""          # 标题
    description: str = ""     # 简介
    tags: List[str] = []      # 话题标签
    duration: float = 0.0     # 总时长（秒）
    author: str = ""          # 作者
    cover_path: str = ""      # 封面图本地路径


class Shot(BaseModel):
    """一个镜头（由预处理节点的镜头切分产生）"""
    index: int                    # 镜号，从 1 开始
    start: float                  # 起始时间（秒）
    end: float                    # 结束时间（秒）
    keyframes: List[str] = []     # 该镜头的关键帧图片路径（给视觉 Agent 用）


class StoryboardRow(BaseModel):
    """需求②：标准化分镜脚本表的「一行」"""
    index: int                    # 镜号
    time_range: str = ""          # 时间段，如 "00:03-00:07"
    camera: str = ""              # 镜头语言（景别/运镜/角度）
    visual: str = ""              # 画面内容
    voiceover: str = ""           # 口播台词
    on_screen_text: str = ""      # 屏幕贴纸/字幕
    narrative: str = ""           # 镜头叙事作用（hook/痛点/卖点/CTA...）
    # ★证据完整性：每行挂上关键帧路径，供评估核对与人工复核
    evidence_frame: str = ""      # 代表帧（镜头中间时刻）→ 核对"画面/镜头语言"
    # ★文字证据帧：该镜"屏幕文字最清晰时刻"的帧（由 OCR 时间戳定位）
    #   长镜头里字幕常只出现几秒，中间帧未必拍得到；用它来核对"屏幕文字"才准。
    text_frame: str = ""          # 无文字的镜头留空


# ---------- 主状态：在所有节点之间流动的那个大字典 ----------

class GraphState(TypedDict, total=False):
    # total=False 表示：这些字段都是"可选"的，允许一开始为空、后续逐步填充

    # === 输入（阶段1 用户给的）===
    source: str                        # 视频来源：URL 或本地文件路径
    start_ts: float                    # 流程开始时间戳（用于计时）

    # === 作业目录（阶段1 采集时分配，全英文路径，避免中文路径的各种坑）===
    job_id: str                        # 本次任务的唯一编号，如 20260706_143512
    job_dir: str                       # 本次任务的工作目录 data/jobs/<job_id>/

    # === 需求①：元数据（阶段1 采集解析 Agent 填）===
    metadata: VideoMeta
    video_path: str                    # 复制/下载到 job 目录后的英文视频路径

    # === 阶段2 预处理节点填 ===
    shots: List[Shot]                  # 镜头列表（切分结果）
    audio_path: str                    # 分离出的音轨文件路径

    # === 全局总览（"略读"：产品/主题/主要人物/场景 + 角色清单）===
    overview: dict

    # === 阶段3 三条并发支线各自填（注意：写不同的字段，才能安全并发）===
    audio_result: dict                 # 音频 Agent：台词、语速、关键词
    visual_result: dict                # 视觉 Agent：逐镜画面/镜头语言 {镜号: {...}}
    ocr_result: dict                   # OCR Agent：逐镜屏幕文字 + 全片文字汇总

    # === 阶段4 时序对齐节点填：每个镜头的"证据包"（画面+口播+屏幕文字）===
    aligned: List[dict]

    # === 说话人标注（LLM 推测）：[{time, speaker, text}] ===
    dialogue: List[dict]

    # === 阶段4 分镜合成 Agent 填（需求②）===
    storyboard: List[StoryboardRow]

    # === 阶段5 评估 Agent 填 ===
    eval_report: dict                  # 各维度分数 + 失败镜号 + 原因
    retry_count: Dict[int, int]        # 每个镜头重写了几次（★控制最多2次的关键）

    # === 阶段6 报告 Agent 填（需求③）===
    report: dict

    # === 阶段6 导出节点填（需求④⑤）===
    exports: dict                      # 导出文件路径、入库记录id 等