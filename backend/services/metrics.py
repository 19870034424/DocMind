"""可观测性 — request_id 贯穿 + JSONL 指标打点

- request_id: 中间件生成, 存 ContextVar, 一次请求的全部日志/打点携带同一 ID
- log_event: 结构化事件追加写入 data/logs/metrics.jsonl (线程安全)
- 常用事件: qa_request / llm_usage / pipeline_stage / pipeline_result
"""
import json
import logging
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = LOG_DIR / "metrics.jsonl"

_lock = threading.Lock()


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(kind: str, **fields) -> None:
    """追加一条结构化事件(自动附时间戳和 request_id)"""
    record = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "request_id": request_id_var.get(),
        "event": kind,
        **fields,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock, open(METRICS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        logging.getLogger(__name__).exception("metrics log_event failed")


def setup_logging() -> None:
    """结构化日志: 文件(带 request_id) + 控制台"""
    fmt = "%(asctime)s [%(levelname)s] [rid=%(request_id)s] %(name)s: %(message)s"

    class _RIDFilter(logging.Filter):
        def filter(self, record):
            record.request_id = request_id_var.get()
            return True

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    filt = _RIDFilter()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    console.addFilter(filt)
    file_h = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_h.setFormatter(logging.Formatter(fmt))
    file_h.addFilter(filt)
    root.addHandler(console)
    root.addHandler(file_h)
    # 压掉三方库噪音
    for noisy in ("httpx", "openai", "uvicorn.access", "sentence_transformers", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
