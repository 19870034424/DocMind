"""文档处理 Worker — 从 SQLite 队列认领任务, 限流/超时/重试

两种运行方式:
1. 内嵌线程: main.py 启动时调用 start_inline_worker()(开发/单机默认)
2. 独立进程: python worker.py(生产可多开, 认领是原子操作, 多进程安全)

任务生命周期: uploaded → queued → parsing → (chunking) → ready
失败: retry_count < pipeline_max_retries 时回 queued 重试, 否则 error(记录 error_stage/error_msg)
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout

from config import settings
from services import metrics
from services.metrics import log_event
from services.pipeline import process_document
from storage.db import claim_queued_document, get_document, update_document

logger = logging.getLogger(__name__)


def _requeue(doc_id: str, retry_count: int) -> None:
    update_document(doc_id, status="queued", retry_count=retry_count + 1,
                    error_stage="", error_msg="")


def _mark_error(doc_id: str, stage: str, msg: str) -> None:
    update_document(doc_id, status="error", error_stage=stage, error_msg=msg[:1000])


def run_claimed_task(doc_row: dict) -> None:
    """处理一个已认领的任务: 限流由线程池上限保证, 超时/重试在此实现"""
    doc_id = doc_row["id"]
    retry_count = doc_row.get("retry_count", 0)
    filename = doc_row["filename"]
    from storage.file_store import file_store

    file_path = file_store.get_path(doc_id, filename)
    if not file_path.exists():
        _mark_error(doc_id, "worker", f"原始文件缺失: {file_path}")
        log_event("pipeline_result", doc_id=doc_id, ok=False, reason="file_missing")
        return

    max_retry = settings.pipeline_max_retries
    log_event("pipeline_start", doc_id=doc_id, attempt=retry_count + 1)

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"pipeline-{doc_id}")
    try:
        fut = pool.submit(process_document, doc_id, file_path, filename)
        fut.result(timeout=settings.pipeline_timeout_seconds)
        row = get_document(doc_id)
        ok = row and row["status"] == "ready"
        log_event("pipeline_result", doc_id=doc_id, ok=bool(ok))
        if not ok:
            # process_document 内部已写入 error 状态, 这里只考虑是否值得重试
            err = (row or {}).get("error_stage", "?")
            if retry_count < max_retry:
                logger.warning("doc %s 第 %s 次失败(%s), 重新入队", doc_id, retry_count + 1, err)
                _requeue(doc_id, retry_count)
    except FutTimeout:
        logger.error("doc %s 管线超时(>%ss)", doc_id, settings.pipeline_timeout_seconds)
        if retry_count < max_retry:
            _requeue(doc_id, retry_count)
        else:
            _mark_error(doc_id, "timeout", f"处理超时(>{settings.pipeline_timeout_seconds}s)")
        log_event("pipeline_result", doc_id=doc_id, ok=False, reason="timeout")
    except Exception:
        logger.exception("doc %s worker 异常", doc_id)
        if retry_count < max_retry:
            _requeue(doc_id, retry_count)
        else:
            _mark_error(doc_id, "worker", "worker 内部异常")
        log_event("pipeline_result", doc_id=doc_id, ok=False, reason="exception")
    finally:
        pool.shutdown(wait=False)


class PipelineWorker(threading.Thread):
    """轮询 queued 任务的常驻线程, 内部线程池限制并发"""

    def __init__(self):
        super().__init__(daemon=True, name="pipeline-worker")
        self._sem = threading.Semaphore(settings.pipeline_max_concurrent)
        self._stop = threading.Event()

    def run(self):
        logger.info("pipeline worker 启动 (并发上限=%s, 轮询=%ss)",
                    settings.pipeline_max_concurrent, settings.worker_poll_seconds)
        pool = ThreadPoolExecutor(max_workers=settings.pipeline_max_concurrent,
                                  thread_name_prefix="worker-task")
        while not self._stop.is_set():
            if self._sem.acquire(blocking=False):
                row = claim_queued_document()
                if row:
                    # 信号量由 _guarded_run 在任务结束后释放, 保证并发上限真实生效
                    pool.submit(self._guarded_run, row)
                else:
                    self._sem.release()
                    self._stop.wait(settings.worker_poll_seconds)
            else:
                self._stop.wait(settings.worker_poll_seconds)

    def _guarded_run(self, row: dict):
        try:
            run_claimed_task(row)
        except Exception:
            logger.exception("task %s escaped guard", row.get("id"))
        finally:
            self._sem.release()

    def stop(self):
        self._stop.set()


def start_inline_worker() -> PipelineWorker | None:
    """后端进程内启动一个 worker 线程(开发/单机模式)"""
    if not settings.pipeline_inline_worker:
        return None
    w = PipelineWorker()
    w.start()
    return w


if __name__ == "__main__":
    metrics.setup_logging()
    logger = logging.getLogger("worker")
    logger.info("独立 worker 进程启动, Ctrl+C 退出")
    w = PipelineWorker()
    w.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        w.stop()
        logger.info("worker 退出")
