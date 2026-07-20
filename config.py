"""
全局配置文件。
把所有「路径、模型名、可调参数」集中放这里，方便以后统一修改，
避免这些设置散落在各处代码里。
"""
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

# ---------- 路径配置 ----------
# BASE_DIR = 本项目根目录（config.py 所在的目录）
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # 存中间产物（作业目录 jobs/ 等）
UPLOAD_DIR = DATA_DIR / "upload"      # 用户上传的视频放这里
OUTPUT_DIR = BASE_DIR / "outputs"     # 存最终导出的 Markdown / CSV / 报告

# 确保这些目录存在（不存在就创建）
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- 模型配置 ----------
# 视觉理解模型（全局总览 / 逐镜精读）：Gemini 2.5 Flash（走 OpenAI 兼容接口）
VISION_MODEL = "gemini-2.5-flash"
VISION_API_KEY = os.getenv("GEMINI_API_KEY")
VISION_BASE_URL = os.getenv("GEMINI_BASE_URL")

#   Gemini 需走代理。默认复用系统 HTTP_PROXY
VISION_PROXY = os.getenv("VISION_PROXY") or os.getenv("HTTP_PROXY") or None

# 文本推理模型（分镜合成 / 报告，纯文字）：DeepSeek-V3
TEXT_MODEL = "deepseek-chat"
TEXT_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TEXT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 评估模型：Qwen-VL-Max（阿里 DashScope）
# ★和生成隔离：视觉生成用 Gemini，评估用 Qwen-VL-Max 
EVAL_MODEL = "qwen-vl-max"
EVAL_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EVAL_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

# ---------- ffmpeg 路径 ----------
# 我们不依赖系统 PATH 里的 ffmpeg，而是用 pip 装的 imageio-ffmpeg 自带的那个二进制。
# 这样最省心，不用你手动配环境变量。yt-dlp、抽帧、分音轨都会用到它。
import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# ---------- 语音识别配置 ----------
# ★我们已经把 small 模型下载到 models/small/ 了。
# faster-whisper 支持直接传"本地模型目录"来加载，这样就不会再联网下载。
# 如果以后想换档位，可以改成 "base"/"medium" 等名字（那样会触发联网下载）。
ASR_MODEL = str(BASE_DIR / "models" / "small")

# ---------- 可调参数 ----------
MAX_RETRY = 2               # 分镜评估不合格时，最多重写几次

#   逐镜多模态调用的并发数（视觉精读 / 评估质检）
#   逐镜分析彼此独立（上下文来自"略读"的角色清单，不依赖上一镜），所以可以安全并发。
#   ★必须限流：无脑全开会撞 API 每分钟请求上限(429)。撞限流就把这个值调小。
VL_CONCURRENCY = 5

# 分镜合成的分批大小：镜头很多时（如高帧率长视频切出上百镜），
# 一次性塞进一个 prompt 会超模型上下文导致返回被截断。分批合成、每批 N 镜，再拼起来。
COMPOSE_BATCH_SIZE = 20

# 断点恢复：每步状态快照存这里，单独放一个文件夹便于管理
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DB = str(CHECKPOINT_DIR / "checkpoints.db")
# 镜头切分灵敏度（AdaptiveDetector）：越小越敏感、切得越碎。
# 用自适应检测器，对"柔和转场/镜头运动"比 ContentDetector 更鲁棒，少漏刀。
ADAPTIVE_THRESHOLD = 3.0
FRAME_SAMPLE_INTERVAL = 0.5  # OCR 密集抽帧间隔（秒）
FILE_TTL_DAYS = 7           # 导出文件保留天数，超过自动清理