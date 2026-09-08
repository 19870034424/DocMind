import os

# 项目专属 HF 模型缓存(D 盘): C 盘空间不足, 且系统 HF_HUB_CACHE 指向其他项目目录。
# 强制覆盖(仅影响本进程), 必须在 import transformers/huggingface_hub 之前执行。
os.environ["HF_HUB_CACHE"] = "D:\\hf_models\\hub"
os.environ["HF_HOME"] = "D:\\hf_models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 模型已全部缓存, 强制离线, 避免任何库启动时对 hf.co 发 HEAD 请求卡住几十秒
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.makedirs("D:\\hf_models\\hub", exist_ok=True)

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM
    llm_provider: str = "qwen"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # Embedding: "qwen" | "local"
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # Rerank(混合检索召回后的精排)
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_pool_multiplier: int = 3  # 召回 pool = top_k * 该倍数, 精排后取 top_k

    # 检索配置
    retrieval_nprobe: int = 32          # IVF 每次查询探测的簇数
    retrieval_rrf_k: int = 60           # RRF 融合常数
    retrieval_top_k_qa: int = 15        # 问答召回块数
    retrieval_top_k_report: int = 30    # 报告/深读召回块数
    context_max_tokens_qa: int = 6000
    context_max_tokens_report: int = 12000

    # Storage
    upload_dir: str = "../uploads"
    chroma_persist_dir: str = "../chroma_data"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Evaluation
    eval_datasets_dir: str = "../eval_datasets"

    # Web Search
    web_search_enabled: bool = True
    web_search_max_results: int = 5

    # LLM 调用容错
    llm_timeout_seconds: int = 60        # 单次 LLM 调用超时
    llm_max_retries: int = 2             # SDK 层自动重试次数(超时/限流/连接错误)
    llm_daily_token_budget: int = 0      # 每日 token 预算, 0 = 不限
    llm_rewrite_timeout_seconds: int = 12  # 查询改写专用超时(不拖死问答)

    # 处理管线 worker
    pipeline_inline_worker: bool = True  # 后端进程内启动 worker 线程(生产可关掉用独立进程)
    pipeline_max_concurrent: int = 2     # 同时处理的文档数上限
    pipeline_timeout_seconds: int = 900  # 单文档整条管线超时
    pipeline_max_retries: int = 2        # 失败自动重试次数
    worker_poll_seconds: float = 2.0     # 轮询 queued 任务间隔

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
