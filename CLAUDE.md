# DocMind — AI 文档阅读助手

基于 RAG 的文档问答平台:上传教材/课件,自动解析建库,提供智能问答、阅读报告、知识图谱。

## 怎么跑

```bash
# 1. Milvus 数据库(Docker)
docker compose -p docmind -f docker-compose.milvus.yml up -d
# 2. 后端(conda pytorch 环境,GPU 加速嵌入/精排) → http://localhost:8000
cd backend && C:\Users\Lenovo\.conda\envs\pytorch\python.exe main.py
# 3. 前端 → http://localhost:5173
cd frontend && npx vite --host
```

## 技术栈

- 后端:FastAPI + SQLite(data/app.db)+ Milvus(19531,docmind 容器栈)+ BM25(内存,语料持久化在 data/bm25/)
- 检索:bge-m3 嵌入(1024 维)→ 向量+BM25 双路召回 → RRF 融合 → bge-reranker-v2-m3 精排(全 GPU)
- 生成:DeepSeek(qwen 可切),统一走 services/llm_client.py(超时/重试/用量打点)
- 解析:MinerU 子进程(用 .venv-mineru 的 python,不要用 conda 环境跑 mineru)
- 前端:React 18 + TS + Vite(5173,/api 代理到 8000)

## 目录与约定

- routers/ 路由;services/ 业务(检索策略在 retrieval.py,嵌入在 embedder.py,管线在 pipeline.py);storage/ Milvus/SQLite/文件
- worker.py:文档处理任务队列(独立进程 `python worker.py`,默认后端内嵌线程)
- 可观测性:data/logs/metrics.jsonl(JSONL 事件)+ app.log(request_id 贯穿)
- 上传文件存 uploads/{doc_id}.{ext},状态机 uploaded→queued→parsing→chunking→ready/error
- 后端 lint:`ruff check .`(配置在 backend/ruff.toml,零告警基线);测试 `pytest tests/`(125 个)

## 当前状态(2026-09-08)

- 检索/问答/报告/知识图谱/Agent 深度问答均可用;原文预览已移除(仅 AI 解析视图+分块预览)
- 未提交改动较多(检索重构、任务队列、可观测性),待 git 提交
- 下一步候选:parent-child 检索、BM25 重建任务化、用户鉴权
