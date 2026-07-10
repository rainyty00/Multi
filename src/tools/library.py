"""
=================================================================
知识库（阶段6.B-2）—— 需求⑤：入库 / 查看历史 / 一周清理
=================================================================
用一个 SQLite 小数据库(data/library.db) 记录每次生成的索引：
  id / 标题 / 平台 / 输出目录 / 生成时间
配套三个功能：
  · add_record   导出后登记一条
  · list_records 查看过往生成记录
  · cleanup_expired 删除超过 N 天的记录和文件（需求⑤"一周后清除"）
=================================================================
"""
import sqlite3
import shutil
import datetime
from pathlib import Path

from config import DATA_DIR, FILE_TTL_DAYS

DB_PATH = DATA_DIR / "library.db"


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT,
                platform   TEXT,
                out_dir    TEXT,
                created_at TEXT
            )
        """)


def add_record(title: str, platform: str, out_dir: str):
    """导出成功后登记一条记录。"""
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO records(title, platform, out_dir, created_at) VALUES(?,?,?,?)",
            (title, platform, out_dir, datetime.datetime.now().isoformat(timespec="seconds")),
        )


def list_records() -> list:
    """返回所有历史记录（最新在前）。"""
    init_db()
    with _conn() as c:
        return c.execute(
            "SELECT id, title, platform, created_at, out_dir FROM records ORDER BY id DESC"
        ).fetchall()


def cleanup_expired(ttl_days: int = FILE_TTL_DAYS) -> list:
    """删除超过 ttl_days 天的记录及其输出文件。返回被删的目录列表。"""
    init_db()
    cutoff = datetime.datetime.now() - datetime.timedelta(days=ttl_days)
    removed = []
    with _conn() as c:
        rows = c.execute("SELECT id, out_dir, created_at FROM records").fetchall()
        for rid, out_dir, created in rows:
            try:
                created_dt = datetime.datetime.fromisoformat(created)
            except Exception:
                continue
            if created_dt < cutoff:
                shutil.rmtree(out_dir, ignore_errors=True)   # 删文件夹
                c.execute("DELETE FROM records WHERE id=?", (rid,))  # 删记录
                removed.append(out_dir)
    return removed
