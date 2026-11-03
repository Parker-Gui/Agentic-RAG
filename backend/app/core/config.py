import secrets
import warnings
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AnyUrl,
    BeforeValidator,
    EmailStr,
    HttpUrl,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",") if i.strip()]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 两个 .env,都相对 ./backend/ 解析（所有入口都从这里启动:uvicorn / pytest /
        # langgraph dev / alembic）。列表里靠后的优先级更高。
        #   ../.env    仓库根,进了 git 的共享默认值(DB、SMTP、占位密钥)
        #   ./.env     本机私有,不进 git —— 大模型与外部服务的真实凭据写这里
        #              (LLM_PROVIDER / OLLAMA_* / OPENAI_* / TAVILY_* / LANGSMITH_*)
        # 只写 ../.env 时,唯一能读到 backend/.env 的是 langgraph dev(靠 langgraph.json
        # 的 "env": ".env" 把它灌进 os.environ),uvicorn 和 pytest 拿到的是默认值 ——
        # 表现为调 ollama.com 报 401 Unauthorized。
        env_file=("../.env", ".env"),
        env_ignore_empty=True,
        extra="ignore",
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    # Lifetime of a login session (`sessions` row); rotated on every refresh.
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    FRONTEND_HOST: str = "http://localhost:5173"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    BACKEND_CORS_ORIGINS: Annotated[list[AnyUrl] | str, BeforeValidator(parse_cors)] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [self.FRONTEND_HOST]

    """ Logging """

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    """ Middleware """

    # Header carrying the request ID, both inbound (trusted, sanitized) and out.
    REQUEST_ID_HEADER: str = "X-Request-ID"

    # Blanket per-IP cap, counted per worker process. See RateLimitMiddleware.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 300
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rate_limit_exempt_paths(self) -> list[str]:
        """Prefixes the rate limit ignores: docs, schema and health probes.

        Uptime checks poll far above any sane per-IP budget, and a throttled
        ``/docs`` is just an annoyance -- neither is worth protecting.
        """
        return [
            "/docs",
            "/redoc",
            f"{self.API_V1_STR}/openapi.json",
            f"{self.API_V1_STR}/utils/health-check",
        ]

    # Signed-cookie sessions, off by default: auth is stateless JWT today. Turn
    # on when a flow needs server-side state across redirects (OAuth `state`).
    SESSION_ENABLED: bool = False
    SESSION_COOKIE_NAME: str = "session"
    SESSION_MAX_AGE_SECONDS: int = 60 * 60 * 24 * 14  # 14 days

    @computed_field  # type: ignore[prop-decorator]
    @property
    def session_https_only(self) -> bool:
        """Mark the session cookie ``Secure`` everywhere but local http dev."""
        return self.ENVIRONMENT != "local"

    """ Reserved integrations """

    # Unset until a Redis-backed component lands (cache, shared rate limit,
    # pub/sub). The app lifespan connects it when this is set.
    REDIS_URL: str | None = None

    DB_ECHO: bool = False  # Set DB_ECHO=true to log SQL queries (latency + log-noise drain by default)

    """ RAG (语义检索) """

    # 嵌入模型（ModelScope 上的模型 id）。向量维度由模型决定，是 article_chunks.embedding
    # 的列宽（见 db.models.article_chunk.EMBED_DIM）；换模型必须同时写迁移改列类型。
    RAG_EMBED_MODEL: str = "BAAI/bge-m3"
    # 向量存在 Postgres（pgvector）里，与业务库同一个连接，故无独立的向量库配置。
    # 切块参数与检索返回条数。
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 5
    # 嵌入 worker 每批处理的文章数。
    RAG_INDEX_BATCH: int = 100
    # 是否在应用启动时预热 RAG（加载嵌入模型）。默认关：模型未下载时不拖慢启动。
    RAG_PRELOAD: bool = False

    # ---- Phase 2: hybrid 检索 + rerank + 阈值 ----
    RAG_HYBRID_ENABLED: bool = False  # 稠密+稀疏(BGE-m3 lexical) 融合
    RAG_RERANK_ENABLED: bool = False  # 交叉编码器重排(已定采用;实现 _hybrid_retrieve 后置 True 启用)
    RAG_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RAG_MIN_SCORE: float = 0.0  # 低于此分丢弃(口径随流水线段变化)
    RAG_FETCH_K: int = 30  # rerank 前的 over-fetch 数

    # ---- Phase 4: 评估三元组的裁判(LLM-judge) ----
    # 中立裁判：避免"生成=评判同模型"的自评偏差。留空则退回生成用的 LLM_PROVIDER。
    RAG_EVAL_JUDGE_PROVIDER: Literal["ollama", "openai", "bedrock"] | None = None
    # 裁判模型 id；留空用该 provider 的默认模型。bedrock 用 inference profile，如 us.anthropic.claude-...
    RAG_EVAL_JUDGE_MODEL: str | None = None

    PROJECT_NAME: str
    SENTRY_DSN: HttpUrl | None = None
    POSTGRES_SERVER: str
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_URL(self) -> str:
        """Build async PostgreSQL connection URL.

        psycopg 3 serves both the async engine and Alembic, so no separate
        asyncpg driver is needed.
        """
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Build sync PostgreSQL connection URL (for Alembic)."""
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_DSN(self) -> str:
        """Driverless DSN for libraries that parse it themselves (e.g. asyncpg)."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    SMTP_PORT: int = 587
    SMTP_HOST: str | None = None
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    EMAILS_FROM_EMAIL: EmailStr | None = None
    EMAILS_FROM_NAME: str | None = None

    @model_validator(mode="after")
    def _set_default_emails_from(self) -> Self:
        if not self.EMAILS_FROM_NAME:
            self.EMAILS_FROM_NAME = self.PROJECT_NAME
        return self

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48

    @computed_field  # type: ignore[prop-decorator]
    @property
    def emails_enabled(self) -> bool:
        return bool(self.SMTP_HOST and self.EMAILS_FROM_EMAIL)

    EMAIL_TEST_USER: EmailStr = "test@example.com"
    FIRST_SUPERUSER: EmailStr
    FIRST_SUPERUSER_PASSWORD: str

    def _check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == "changethis":
            message = (
                f'The value of {var_name} is "changethis", for security, please change it, at least for deployments.'
            )
            if self.ENVIRONMENT == "local":
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        self._check_default_secret("SECRET_KEY", self.SECRET_KEY)
        self._check_default_secret("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD)
        self._check_default_secret("FIRST_SUPERUSER_PASSWORD", self.FIRST_SUPERUSER_PASSWORD)

        return self

    # llm_providers
    LLM_PROVIDER: Literal["ollama", "openai"] = "ollama"

    OLLAMA_MODEL: str = "ollama-model"
    OLLAMA_AVAILABLE_MODELS: list[str] = [OLLAMA_MODEL]
    OLLAMA_API_KEY: str | None = None
    OLLAMA_BASE_URL: str | None = None

    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_AVAILABLE_MODELS: list[str] = [OPENAI_MODEL]
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None

    # AWS Bedrock（目前仅评估裁判用）。凭证走 boto3 默认链：env AWS_ACCESS_KEY_ID/... 或 ~/.aws。
    BEDROCK_REGION: str = "us-east-1"
    BEDROCK_MODEL: str | None = None  # inference profile id，如 us.anthropic.claude-...
    BEDROCK_AVAILABLE_MODELS: list[str] = []

    """ Tavily (外部联网检索) """

    # 缺 key 时工具不抛异常，返回 results=[] 的错误 JSON —— 原因见
    # agents/tools/tavily_tool.py 的模块 docstring（报错文案不能变成证据）。
    TAVILY_API_KEY: str | None = None
    # basic 约 1 credit/次，advanced 约 2：正文更全但更慢更贵。search 与 extract 共用。
    TAVILY_SEARCH_DEPTH: Literal["basic", "advanced"] = "basic"
    # 单次检索返回条数的硬上限，模型给的 max_results 会被夹到这个值。
    TAVILY_MAX_RESULTS: int = 5
    # 单条证据保留的正文字符数。整页正文进了 messages，react / critic 每轮都要重付一遍。
    TAVILY_MAX_CONTENT_CHARS: int = 2000
    TAVILY_TIMEOUT: float = 30.0
    # True 时搜索直接带回整页 markdown 正文（替代 Tavily 的短摘要）：更全，token 也更多。
    # 默认关，需要全文时让模型调 tavily_extract，按需付费。
    TAVILY_INCLUDE_RAW_CONTENT: bool = False


settings = Settings()  # type: ignore
