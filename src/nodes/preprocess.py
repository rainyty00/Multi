"""
=================================================================
预处理节点（阶段2）—— 纯工程处理，不用 LLM、不花钱
=================================================================
要做三件事，我们分小块写：
  2.1 探测时长 + 分离音轨   （本文件当前部分）
  2.2 镜头切分             （稍后加）
  2.3 抽帧                 （稍后加）

★为什么它是"节点"不是"Agent"：这些都是确定性的机械处理，
  用固定算法就能完成，不需要大模型推理判断。
=================================================================
"""
import subprocess
from pathlib import Path

import cv2  # OpenCV，用来读视频信息
from scenedetect import detect, AdaptiveDetector  # 镜头切分（自适应，少漏刀）

from src.state import GraphState, Shot
from config import FFMPEG_EXE, ADAPTIVE_THRESHOLD


# ---------- 2.1-A：用 OpenCV 探测视频基本信息 ----------
def probe_video(video_path: str) -> dict:
    """
    打开视频，读出 fps（帧率）、总帧数、宽高，算出时长（秒）。
    返回一个字典。
    """
    cap = cv2.VideoCapture(video_path)          # 打开视频文件
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 打不开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)             # 帧率，如 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()                              # ★用完一定要释放，否则文件被占用

    # 时长 = 总帧数 / 帧率。防止 fps 为 0 导致除零
    duration = frame_count / fps if fps else 0.0

    return {
        "fps": fps,
        "frame_count": int(frame_count),
        "width": width,
        "height": height,
        "duration": round(duration, 2),
    }


# ---------- 2.1-B：用 ffmpeg 把音轨分离成 wav ----------
def extract_audio(video_path: str, job_dir: Path) -> str:
    """
    从视频里抽出音频，存成 16kHz 单声道 wav（语音识别 ASR 的标准输入格式）。
    存到作业目录 job_dir/audio.wav。返回 wav 文件路径。
    """
    audio_path = job_dir / "audio.wav"

    # 拼 ffmpeg 命令（★每个参数的意思写在注释里）
    cmd = [
        FFMPEG_EXE,
        "-y",                       # 已存在就覆盖，不询问
        "-i", str(video_path),      # 输入视频
        "-vn",                      # 不要视频流（video none）
        "-acodec", "pcm_s16le",     # 音频编码：16 位 PCM（标准 wav）
        "-ar", "16000",             # 采样率 16kHz
        "-ac", "1",                 # 单声道（1 个声道）
        str(audio_path),
    ]
    # 运行命令。capture_output=True 把输出抓起来，不刷屏
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        # ★出错时把 ffmpeg 的报错打出来，方便调试
        raise RuntimeError(f"ffmpeg 抽音频失败:\n{result.stderr[-500:]}")

    return str(audio_path)


# ---------- 2.2：镜头切分 ----------
def detect_shots(video_path: str, total_duration: float) -> list[Shot]:
    """
    用 scenedetect 检测镜头剪切点，切成一个个镜头。
    返回 Shot 列表（此时每个 Shot 的 keyframes 还是空的，2.3 再填）。
    """
    # ★AdaptiveDetector：自适应检测，对柔和转场/镜头运动更鲁棒，
    #   比 ContentDetector 更不容易漏掉剪辑点。阈值越小越敏感。
    scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=ADAPTIVE_THRESHOLD))

    shots: list[Shot] = []
    if not scene_list:
        # ★没检测到任何剪切点（比如整段是一个长镜头）→ 把整条视频当作 1 个镜头
        shots.append(Shot(index=1, start=0.0, end=total_duration))
    else:
        # scene_list 是 [(起始时间码, 结束时间码), ...]
        for i, (start_tc, end_tc) in enumerate(scene_list, start=1):
            shots.append(Shot(
                index=i,
                start=round(start_tc.get_seconds(), 2),  # 时间码转成"秒"
                end=round(end_tc.get_seconds(), 2),
            ))
    return shots


# ---------- 2.3：为每个镜头抽代表关键帧 ----------
def extract_keyframes(video_path: str, shots: list[Shot], job_dir: Path) -> list[Shot]:
    """
    每个镜头抽 1 张"代表帧"（取镜头正中间那一帧，最能代表该镜头内容）。
    图片存到 job_dir/frames/shot_<镜号>.jpg，路径写回 shot.keyframes。
    ★因为 job 目录是英文路径，cv2.imwrite 就能正常写盘了（不再踩中文路径的坑）。
    ★这些帧就是后面「视觉 Agent」和「评估 Agent」要看的图。
    """
    # 建一个专门放帧的文件夹（在作业目录下）
    frame_dir = job_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for shot in shots:
        # 取镜头中间时刻，换算成第几帧
        mid_time = (shot.start + shot.end) / 2
        frame_no = int(mid_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # 跳到那一帧
        ok, frame = cap.read()                       # 读出这一帧（一张图）
        if not ok:
            continue
        out_path = frame_dir / f"shot_{shot.index}.jpg"
        cv2.imwrite(str(out_path), frame)            # 存成 jpg
        shot.keyframes = [str(out_path)]             # 记录路径到该镜头

    cap.release()
    return shots


# ---------- 节点主函数 ----------
def preprocess_node(state: GraphState) -> dict:
    video_path = state["video_path"]
    job_dir = Path(state["job_dir"])   # 作业目录（英文路径）
    print(f"[预处理] 开始处理: {video_path}")

    # 1) 探测视频信息
    info = probe_video(video_path)
    print(f"[预处理] 帧率={info['fps']} 总帧数={info['frame_count']} "
          f"分辨率={info['width']}x{info['height']} 时长={info['duration']}s")

    # 2) 把探测到的真实时长补进 metadata
    meta = state["metadata"]
    meta.duration = info["duration"]

    # 3) 分离音轨
    audio_path = extract_audio(video_path, job_dir)
    print(f"[预处理] 音轨已分离: {audio_path}")

    # 4) 镜头切分
    shots = detect_shots(video_path, info["duration"])
    print(f"[预处理] 切分出 {len(shots)} 个镜头:")
    for s in shots:
        print(f"    镜{s.index}: {s.start}s ~ {s.end}s  (时长 {round(s.end - s.start, 2)}s)")

    # 5) 为每个镜头抽代表关键帧
    shots = extract_keyframes(video_path, shots, job_dir)
    print(f"[预处理] 关键帧已抽取到 {job_dir / 'frames'}")

    return {"metadata": meta, "audio_path": audio_path, "shots": shots}
