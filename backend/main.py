from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from services.metrics import log_event, new_request_id, request_id_var, setup_logging

setup_logging()
from routers import knowledge_graph
from routers.chat import router as chat_router
from routers.documents import router as documents_router

app = FastAPI(
    title="DocMind — AI 文档阅读助手",
    version="0.2.0",
    description="""
基于 **RAG(检索增强生成)** 的 AI 教学辅助平台 API。

上传教材/课件后,系统自动完成:

- **文档解析与索引** — MinerU 解析 → langchain 递归分块 → 本地 BGE(bge-small-zh-v1.5)嵌入 → 存入 Milvus
- **智能问答** — BM25 + 向量语义混合检索,RRF 融合排序,LLM 基于原文回答
- **阅读报告** — 评估 → 拆解 → 批判 → 读者指南 四步分析 + 可信度评分,支持按章节深度分析
- **知识图谱** — LLM 抽取文档概念与关系,前端 D3-force 可视化
- **ReAct Agent 深度问答** — Agent 自主多步检索与推理,SSE 流式输出步骤与答案

## 使用约定

- 所有业务接口前缀为 `/api/documents`
- 文档必须处于 `ready` 状态才能触发问答、报告、知识图谱
- `/chat/stream`、`/chat/deep`、`/chat/deep/resume` 返回 **SSE 流式响应**(text/event-stream)
""",
    openapi_tags=[
        {
            "name": "documents",
            "description": "文档管理:上传(含解析/分块/嵌入完整管线)、列表、详情、删除,以及基于文档的阅读报告与普通问答。",
        },
        {
            "name": "knowledge-graph",
            "description": "知识图谱:LLM 抽取文档概念与关系并建图,支持图读取、节点详情与搜索。前端以 D3 力导向图展示。",
        },
        {
            "name": "chat",
            "description": "流式对话:快速模式(单轮 RAG 问答)与深度模式(ReAct Agent 多步推理),均为 SSE 流式响应,支持会话断点续传。",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """每次请求生成 request_id, 贯穿日志与指标; 响应头回传便于对账"""
    rid = request.headers.get("X-Request-ID") or new_request_id()
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response

app.include_router(documents_router)
app.include_router(knowledge_graph.router)
app.include_router(chat_router)

# MinerU 抽取的文档图片: /api/assets/{doc_id}/images/xxx.jpg
from fastapi.staticfiles import StaticFiles

_UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/assets", StaticFiles(directory=str(_UPLOADS_DIR)), name="assets")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    """进程内启动处理管线 worker(生产可设 PIPELINE_INLINE_WORKER=false 用独立进程)"""
    from worker import start_inline_worker
    if start_inline_worker():
        log_event("worker_started", mode="inline")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
