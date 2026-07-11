# 🎬 多模态广告创意拆解 Agent

基于 **LangGraph** 的多模态多 Agent 系统：输入一条广告视频，自动**还原完整分镜脚本表**，并产出创意分析报告与配套素材。替代人工逐帧看视频做拆解笔记的重复劳动。

---

## 功能特性

- **标准分镜脚本表**：镜号 / 时间段 / 镜头语言 / 画面内容 / 口播台词 / 屏幕文字 / 叙事作用
- **作品概览**：自动识别广告产品/品牌、主题、主要人物、场景、核心卖点、目标受众
- **分析报告**：镜头统计、语速/关键词、画面风格、创意套路、图文匹配评分
- **素材文件**：各镜头关键帧、完整台词（含说话人推测）、视频内全部文字
- **多种导出**：终端/网页 Markdown 表格、CSV、打包 zip；SQLite 知识库入库、历史查看、7 天 TTL
- **Web 界面**：gradio，上传视频 → 实时逐节点进度 → 结果展示 → 下载

---

## 技术亮点

| 亮点 | 说明 |
|---|---|
| **LangGraph 多 Agent** | 状态图编排 10+ 个 Agent/节点，条件路由 + 反馈循环 |
| **双层异步并发** | ① 支线间：音频 ∥ OCR ∥ 视觉 fan-out/fan-in；② 支线内：逐镜多模态调用限流线程池并发 |
| **评估 + 定向回退** | Qwen-VL 逐镜多维打分，不合格只重写失败镜（≤2 次）后降级出稿 |
| **模型隔离** | 生成用 Gemini、评估用 Qwen-VL，避免自评偏差 |
| **断点恢复** | SqliteSaver 持久化，崩溃后从上个成功节点续跑，不重跑耗时步骤 |
| **全局总览"略读→精读"** | 先建角色清单再逐镜精读，保证跨镜头人物描述一致 |
| **ReAct / CoT 双推理** | 采集用 ReAct，合成/评估用 CoT |

---

## 架构

```
用户上传视频
  → 采集(下载/本地) → 预处理(镜头切分/抽帧/分音轨)
  → 全局总览(略读：建角色清单)
  → ┌ 音频理解(Whisper)   ┐
    ├ OCR(RapidOCR)      ├ 三支线异步并发
    └ 视觉理解(Gemini)   ┘
  → 时序对齐 → 分镜合成(DeepSeek) ⇄ 评估(Qwen-VL, 回退≤2次)
  → 说话人标注 → 分析报告 → 导出入库 + WebUI 展示
```

**模型分工**（均走 OpenAI 兼容接口，换模型只改 `config.py` 三行）：

| 环节 | 模型 |
|---|---|
| 视觉理解 / 全局总览 | Gemini 2.5 Flash |
| 分镜合成 / 说话人 / 报告 | DeepSeek-V3 |
| 评估 | Qwen-VL-Max（阿里 DashScope）|
| 音频转写 | faster-whisper（本地）|
| 画面文字 | RapidOCR（本地）|

---

## 环境要求

- Python 3.12

## 安装

```bash
# 1. 创建并激活 conda 环境
conda create -n Multi python=3.12 -y
conda activate Multi

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载本地语音模型（faster-whisper small，约 460M）

# 4. 配置密钥：复制模板并填入你的 key
cp .env.example .env
#   需要 GEMINI_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY
```

> 大模型文件（`models/`）和运行数据（`data/`、`outputs/`）不在仓库内，需自行下载/生成。

## 运行

**命令行：**
```bash
# 分析视频（视频放 data/upload/）
python main.py "彩虹糖广告.mp4"
python main.py --resume <任务号>   # 断点恢复
python main.py --list              # 查看历史
python main.py --jobs              # 查看作业目录对应
```

**Web 界面：**
```bash
python webui.py
# 浏览器打开 http://127.0.0.1:7860
```

---

## 目录结构

```
config.py            全局配置（路径/模型/参数）
main.py              命令行入口
webui.py             gradio 网页界面
src/
  state.py           数据主干 GraphState
  graph.py           LangGraph 流程图组装
  agents/            采集/全局总览/音频/OCR/视觉/合成/评估/说话人/报告
  nodes/             预处理/时序对齐/导出
  tools/             知识库(SQLite)
models/              本地模型（whisper / FunASR）
data/                运行数据（上传/作业目录/checkpoints/库）
outputs/             成品
```

## 技术栈

LangGraph · LangChain · gradio · faster-whisper · RapidOCR · scenedetect · OpenAI SDK（Gemini/DeepSeek/Qwen-VL 兼容接口）· SQLite

## 已知局限

- 镜头切分偶尔漏极短/柔和转场片段
- 每镜取 1 张代表帧，长镜头内细节可能漏（下一步：自适应多帧）
- 说话人为 LLM 推测（下一步：接 FunASR 声纹分离交叉验证）
- 平台下载受反爬限制，目前主用本地上传
