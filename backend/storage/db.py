"""SQLite 持久化存储 — 替代内存 dict

表:
  documents     — 文档记录（含 metadata/structured/chunks JSON）
  reports       — 阅读报告
  mindmaps      — 思维导图
  chat_history  — 对话历史
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "app.db"


def _now() -> str:
    return datetime.now().isoformat()


def _get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """初始化数据库表（幂等）"""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'uploaded',
            format TEXT,
            pages INTEGER NOT NULL DEFAULT 0,
            file_size INTEGER NOT NULL DEFAULT 0,
            filename TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            structured_json TEXT,
            chunks_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reports (
            doc_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
            content TEXT NOT NULL DEFAULT '',
            generated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mindmaps (
            doc_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
            root_json TEXT NOT NULL DEFAULT '{}',
            generated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_doc ON chat_history(doc_id, id);
    """)
    # 老库迁移: 补充管线错误字段
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "error_stage" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN error_stage TEXT NOT NULL DEFAULT ''")
    if "error_msg" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN error_msg TEXT NOT NULL DEFAULT ''")
    if "retry_count" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    conn.close()


# ═══ Document CRUD ═══

def insert_document(doc_id: str, filename: str, fmt: str, file_size: int) -> None:
    conn = _get_conn()
    now = _now()
    conn.execute(
        """INSERT INTO documents (id, title, status, format, pages, file_size, filename,
           metadata_json, created_at, updated_at)
           VALUES (?, '', 'uploaded', ?, 0, ?, ?, '{}', ?, ?)""",
        (doc_id, fmt, file_size, filename, now, now),
    )
    conn.commit()
    conn.close()


def update_document(doc_id: str, **fields) -> None:
    """更新文档字段。特殊处理: metadata_json 需要 JSON 序列化"""
    conn = _get_conn()
    sets = ["updated_at = ?"]
    values: list[Any] = [_now()]

    for k, v in fields.items():
        if k == "metadata_json":
            sets.append("metadata_json = ?")
            values.append(json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
        elif k in ("structured", "structured_json"):
            sets.append("structured_json = ?")
            values.append(v.model_dump_json() if hasattr(v, 'model_dump_json') else json.dumps(v, ensure_ascii=False))
        elif k == "chunks":
            sets.append("chunks_json = ?")
            values.append(json.dumps(v, ensure_ascii=False))
        elif k == "status":
            sets.append("status = ?")
            values.append(v.value if hasattr(v, 'value') else str(v))
        elif k == "title":
            sets.append("title = ?")
            values.append(str(v))
        elif k in ("format", "filename"):
            sets.append(f"{k} = ?")
            values.append(v.value if hasattr(v, 'value') else str(v))
        elif k in ("pages", "file_size"):
            sets.append(f"{k} = ?")
            values.append(int(v))
        elif k in ("error_stage", "error_msg"):
            sets.append(f"{k} = ?")
            values.append(str(v))
        elif k == "retry_count":
            sets.append("retry_count = ?")
            values.append(int(v))

    values.append(doc_id)
    conn.execute(f"UPDATE documents SET {', '.join(sets)} WHERE id = ?", values)
    conn.commit()
    conn.close()


def claim_queued_document() -> dict | None:
    """原子认领一个 queued 任务(多 worker 安全): status queued→parsing"""
    conn = _get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id, filename, retry_count FROM documents WHERE status = 'queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if not row:
            conn.commit()
            return None
        conn.execute(
            "UPDATE documents SET status = 'parsing', updated_at = ? WHERE id = ?",
            (_now(), row["id"]),
        )
        conn.commit()
        return dict(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_document(doc_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document(doc_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ═══ Reports ═══

def save_report(doc_id: str, content: str, generated_at: str | None = None) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO reports (doc_id, content, generated_at)
           VALUES (?, ?, ?)""",
        (doc_id, content, generated_at or _now()),
    )
    conn.commit()
    conn.close()


def get_report(doc_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM reports WHERE doc_id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def append_report(doc_id: str, extra_content: str) -> None:
    """在已有报告末尾追加内容"""
    existing = get_report(doc_id)
    if existing:
        save_report(doc_id, existing["content"] + "\n\n---\n\n" + extra_content)
    else:
        save_report(doc_id, extra_content)


# ═══ Mindmaps ═══

def save_mindmap(doc_id: str, root_json: str) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO mindmaps (doc_id, root_json, generated_at)
           VALUES (?, ?, ?)""",
        (doc_id, root_json, _now()),
    )
    conn.commit()
    conn.close()


def get_mindmap(doc_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM mindmaps WHERE doc_id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ═══ Chat History ═══

def add_chat_message(doc_id: str, role: str, content: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO chat_history (doc_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (doc_id, role, content, _now()),
    )
    conn.commit()
    conn.close()


def get_chat_history(doc_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE doc_id = ? ORDER BY id ASC",
        (doc_id,),
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def clear_chat_history(doc_id: str) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM chat_history WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()


# ═══ Startup ═══

init_db()
