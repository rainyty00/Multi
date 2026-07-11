"""
=================================================================
main.py —— 程序入口
=================================================================
用法（在项目根目录 f:/project/Multi 下，用环境 python 运行）：
  分析视频：  python main.py "耐克广告.mp4"      （文件放 data/upload/）
  断点恢复：  python main.py --resume <任务号>    （崩溃后从上个成功节点继续）
  查看历史：  python main.py --list
  查看作业：  python main.py --jobs     （英文文件夹 ↔ 原始视频 对应表）
  清理过期：  python main.py --clean
=================================================================
"""
import sys
import time
import json
import datetime
from src.graph import build_graph
from src.tools.library import list_records, cleanup_expired, auto_cleanup
from src.agents.ingest import JOBS_DIR


def run(source: str):
    auto_cleanup()          # ★启动时自动清理超过 7 天的过期成品
    app = build_graph()
    # ★给这次任务一个唯一编号(thread_id)，作为断点恢复的钥匙
    thread_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 50)
    print(f"开始运行流程... 任务号(thread_id) = {thread_id}")
    print("★如果中途崩溃，用  python main.py --resume " + thread_id + "  从断点继续")
    print("=" * 50)
    t0 = time.time()   # 计时开始
    final_state = app.invoke({"source": source, "start_ts": t0}, config)
    print(f"流程结束，总耗时 {round(time.time() - t0, 1)} 秒。"
          f"导出目录：{final_state.get('exports', {}).get('dir', '（无）')}")


def resume(thread_id: str):
    """崩溃后从断点继续：用同一个 thread_id 再跑一次，传入 None 表示"接着之前的来"。"""
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    print(f"从断点恢复任务 {thread_id} ...（已完成的节点会跳过，不重跑）")
    # ★重置计时起点：否则"分析耗时"会把崩溃到恢复之间的等待也算进去
    app.update_state(config, {"start_ts": time.time()})
    final_state = app.invoke(None, config)   # None = 不给新输入，接着已存的状态继续
    print("恢复完成。导出目录：", final_state.get("exports", {}).get("dir", "（无）"))


def show_history():
    records = list_records()
    if not records:
        print("（暂无历史记录）")
        return
    print(f"共 {len(records)} 条历史记录：")
    for rid, title, platform, created, out_dir in records:
        print(f"  #{rid} [{created}] 《{title}》({platform}) → {out_dir}")


def show_jobs():
    """列出 data/jobs 里"英文文件夹 ↔ 原始视频"的对应关系（读总索引）。"""
    index_path = JOBS_DIR / "index.json"
    if not index_path.exists():
        print("（暂无 jobs 记录）")
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"共 {len(index)} 个作业目录：")
    for jid, info in sorted(index.items(), reverse=True):
        print(f"  {jid}  →  《{info.get('标题')}》({info.get('平台')})  {info.get('创建时间')}")


def clean():
    removed = cleanup_expired()
    print(f"已清理 {len(removed)} 条过期记录" if removed else "没有过期记录需要清理")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--list":
        show_history()
    elif arg == "--jobs":
        show_jobs()
    elif arg == "--clean":
        clean()
    elif arg == "--resume":
        if len(sys.argv) < 3:
            print("用法：python main.py --resume <任务号>")
        else:
            resume(sys.argv[2])
    elif arg:
        run(arg)
    else:
        print("用法：python main.py \"视频文件名或链接\" | --resume <任务号> | --list | --clean")
