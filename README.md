# Intelligence Analysis · 企业智能情报与知识检索系统

> 面向企业私域的**情报采集 → 向量化入库 → 混合检索 → Agent 分析**全链路系统。
> 爬虫定时抓取竞品与法规站点的新文章，BGE-m3 双向量入库 pgvector，由 LangGraph 手工编排的
> ReAct Agent 结合本地知识库与联网检索产出带出处的情报分析。

**技术栈**：FastAPI · Pydantic v2 · PostgreSQL + pgvector · SQLAlchemy 2.0 (async) · Alembic ·
LangGraph · LangChain · BGE-m3 / BGE-reranker-v2-m3 · Selenium + undetected-chromedriver ·
Vite + React + TanStack Router · uv workspace · Docker Compose

文档索引：[docs/README.md](./docs/README.md) ｜ 评估层专篇：[README-EVAL.md](./README-EVAL.md)

---

## 目录

1. [这是什么 · 解决什么问题](#1-这是什么--解决什么问题)
2. [系统全貌与数据流](#2-系统全貌与数据流)
3. [仓库结构](#3-仓库结构)
4. [核心链路详解](#4-核心链路详解)
5. [数据模型](#5-数据模型)
6. [质量保障：评估与测试](#6-质量保障评估与测试)
7. [快速开始](#7-快速开始)
8. [配置说明](#8-配置说明)
9. [已知短板与路线图](#9-已知短板与路线图)

---

## 1. 这是什么 · 解决什么问题

业务场景：企业需要持续跟踪**竞争对手动态**与**法规变化**，信息散落在几十个官网新闻页、
公告 API、以及内部的非结构化文档里。人工每天逐站点翻页既不可持续，也无法回答
「过去半年 XX 公司在甜味剂上有什么动作」这类需要跨文档聚合的问题。

系统分三层解决：

| 层 | 解决的问题 | 关键设计 |
| --- | --- | --- |
| **采集** | 站点结构千差万别、有反爬、需要增量 | 配置驱动的多策略爬虫；**列表页时间戳**增量而非关键词匹配 |
| **检索** | 语义检索对专有名词弱，纯词面检索对同义表述弱 | BGE-m3 一次前向产出稠密 + 稀疏双向量，RRF 融合 + 交叉编码器精排 |
| **分析** | 单次检索答不了需要多步推理的问题 | LangGraph 手工编排 ReAct，检索/读全文/联网三类工具 + 轮次预算门 |

「不编造」是硬要求：检索层有最低分阈值兜底（库内无相关 → 返回空而不强答），
Agent 的每条结论都回标来源 URL / 文档，供人工核验。

---

## 2. 系统全貌与数据流

```
                    ┌─────────────────────── 采集（离线，cron / 手动）────────────────────────┐
                    │                                                                        │
  config/           │  python -m app.services.crawler                                        │
  monitor_sites     │       │                                                                │
  .json ────────────┼──▶ pipeline.scan_all_sites()                                           │
  (站点 + 选择器)    │       │                                                                │
                    │       ├─ 列表页：extract.parse_list_page  →  候选 + 列表页日期（不开文章）│
                    │       ├─ 增量筛：> monitor_last_times.json 里该站点的时间戳             │
                    │       ├─ 详情页：detail.parse_article（Selenium）                       │
                    │       │            或 json_api.fetch_article_api（SPA 的 JSON 接口）    │
                    │       └─ persistence.save_hits_to_db  ──▶  ┌──────────────┐            │
                    │                                            │  articles    │            │
                    └────────────────────────────────────────────│  (Postgres)  │────────────┘
                                                                 └──────┬───────┘
                    ┌───────────────── 索引（离线 worker）──────────────┤ embedding_done=False
                    │                                                   ▼
                    │  python -m app.services.rag
                    │       │
                    │       ├─ chunking.chunk_text        递归字符分块（中英双语分隔符 + 重叠）
                    │       ├─ embedding.encode_with_sparse   BGE-m3 一次前向 → dense + sparse
                    │       └─ chunk_repo.add            ──▶  ┌────────────────────────────┐
                    │                                         │ article_chunks             │
                    └─────────────────────────────────────────│ embedding vector(1024)     │
                                                              │ sparse_embedding           │
                                                              │   sparsevec(250002)        │
                                                              │ 两路 HNSW 索引             │
                                                              └─────────────┬──────────────┘
   ┌────────────────────────── 检索 + 分析（在线）────────────────────────────┤
   │                                                                         ▼
   │  POST /api/v1/rag/query ──▶ RagService.query()
   │                                 ├─ 稠密召回 RAG_FETCH_K=30 条
   │                                 ├─ 稀疏召回 + retrieval.rrf_fuse   （RAG_HYBRID_ENABLED）
   │                                 ├─ rerank.get_reranker().rerank    （RAG_RERANK_ENABLED）
   │                                 └─ retrieval.apply_min_score       （no-answer 兜底）
   │
   │  LangGraph Agent（langgraph dev / eval harness）
   │      START → reasoning → action ──tools_condition──▶ tools ──_budget_gate──▶ reasoning
   │                  ▲                      │                        │
   │                  └──────────────────────┘                        └──▶ finalize → END
   │      工具：search_rag_tool · read_article_tool · tavily_search · tavily_extract
   └─────────────────────────────────────────────────────────────────────────────────────────
```

三段是**解耦**的：爬虫可以只抓不写（`--scrape-only`），索引 worker 只认
`embedding_done=False` 的行，检索层不关心内容怎么来的。向量与 `embedding_done` 标志同在
Postgres、同一事务提交，不存在「向量写了但文章没标记」的漂移。

---

## 3. 仓库结构

```
Intelligence_analysis/
├── backend/
│   ├── app/
│   │   ├── main.py                    # ASGI 装配：中间件 → 异常处理 → 路由 → lifespan
│   │   ├── api/
│   │   │   ├── deps.py                # Annotated 依赖别名（CurrentUser / RagServiceDep / …）
│   │   │   ├── middleware.py          # RequestID + 速率限制
│   │   │   ├── exception_handlers.py  # AppException → 统一 JSON 错误体
│   │   │   └── routes/v1/             # users / utils / articles / rag
│   │   ├── services/                  # 业务逻辑（路由只调服务，不碰仓储）
│   │   │   ├── user.py  article.py    # 薄领域 → 扁平模块
│   │   │   ├── rag/                   # 厚领域子包，对外只暴露 RagService
│   │   │   │   ├── facade.py          #   检索用例（dense / hybrid / rerank / 阈值）
│   │   │   │   ├── retrieval.py       #   RRF 融合 + 阈值过滤（纯函数，好单测）
│   │   │   │   ├── rerank.py          #   BGE-reranker 交叉编码器
│   │   │   │   ├── embedding.py       #   BGE-m3 单例（dense + sparse 一次前向）
│   │   │   │   ├── model_store.py     #   模型下载/缓存（ModelScope 镜像）
│   │   │   │   ├── chunking.py        #   递归字符分块
│   │   │   │   ├── indexer.py         #   嵌入 worker：切块 → 编码 → 写库
│   │   │   │   └── __main__.py        #   python -m app.services.rag
│   │   │   └── crawler/               # 厚领域子包，按职责分层
│   │   │       ├── config/            #   sites.py 站点表 · state.py 时间戳 · config_builder.py 由 xlsx 生成
│   │   │       ├── fetch/             #   browser · extract(列表页) · detail(正文) · json_api · dates · titles · urls
│   │   │       ├── orchestration/     #   pipeline.py 单站点/全站点扫描
│   │   │       ├── delivery/          #   cli.py · persistence.py 落库 · reporting.py 运行汇总
│   │   │       └── __main__.py        #   python -m app.services.crawler
│   │   ├── agents/                    # LangGraph ReAct 单智能体
│   │   │   ├── graph.py               #   StateGraph 装配 + _budget_gate 轮次预算门
│   │   │   ├── nodes.py               #   reasoning / action / tool / finalize 四节点 + prompt
│   │   │   ├── state.py               #   ToolUseState（messages + tool_rounds）
│   │   │   ├── providers.py           #   Ollama / OpenAI / Bedrock 三个 provider
│   │   │   └── agent_tools/           #   rag_tool（内部库）· tavily_tool（外网）· errors
│   │   ├── repository/                # 纯数据访问：无状态函数，flush+refresh，从不 commit
│   │   ├── db/models/                 # article · article_chunk · user · session
│   │   ├── schemas/                   # Pydantic v2：*Create / *Update / *Read / *List
│   │   └── core/                      # config(Settings) · security(JWT) · exceptions · logging
│   ├── alembic/versions/              # 4 条迁移，含 pgvector + sparsevec + HNSW
│   ├── eval/                          # RAG 评估层 → 详见 README-EVAL.md
│   │   ├── dataset.jsonl              #   33 题标注集（question + gold article）
│   │   ├── retrieval_eval.py          #   recall@k / MRR / nDCG，三配置对比
│   │   ├── triad_eval.py              #   RAG 三元组编排
│   │   ├── faithfulness_eval.py       #   有据性（逐句核对证据）
│   │   ├── relevance_eval.py          #   上下文相关性 + 答案相关性
│   │   ├── judge_base.py              #   中立裁判（生成 ≠ 评判模型）
│   │   ├── report.py  reports/        #   报告产物化（json + md）
│   │   └── run_eval.py                #   入口
│   ├── tests/                         # 结构镜像 app/，anyio + AsyncMock，全程离线
│   ├── bootstrap/                     # backend_pre_start · initial_data · tests_pre_start
│   └── langgraph.json                 # langgraph dev 的图注册（agent → graph.py:build_graph）
├── frontend/src/
│   ├── routes/                        # TanStack Router：index · articles · settings · admin · 登录注册
│   ├── components/  hooks/  lib/
│   └── client/                        # 由后端 OpenAPI 生成的 SDK（勿手改）
├── config/                            # 爬虫配置（本地文件，仓库内为空占位）
│   ├── monitor_sites.json             #   站点表：新闻页 URL / group / fetch_mode / api 块
│   ├── list_selectors.json            #   列表容器选择器（status=ready/fixed 才生效）
│   ├── monitor_last_times.json        #   每站点已处理到的时间戳 ← 增量的唯一依据
│   └── recipients.json
├── crawlers/logs/                     # 每次运行的日志与站点汇总
├── docs/
│   ├── en/  zh/                       # 双语长文档（架构 / 开发 / 部署 / 贡献）
│   └── services/                      # 专题分析
│       ├── ingestion-markdown-pipeline.md   # 摄取管道改造方案（未实施）
│       ├── rag-indexing-report.md
│       ├── rag-blocking-call-fix.md
│       └── user-service-flow.md
├── compose.yml  compose.override.yml  compose.traefik.yml
└── CLAUDE.md  AGENTS.md  .claude/rules/     # 给 AI 协作者的项目约定
```

**分层铁律**（详见 [`.claude/rules/architecture.md`](./.claude/rules/architecture.md)）：

- 路由只调服务，**从不**导入仓储；
- 仓储只 `flush()` + `refresh()`，**从不** `commit()`（会话由 `get_db_session` 自动提交）；
- 路由返回 `-> Any`，序列化交给 `response_model`；
- 薄领域用扁平模块（`services/user.py`），厚领域用子包并只从 `__init__.py` 暴露门面
  （`services/rag/` 只对外给 `RagService`）。

---

## 4. 核心链路详解

### 4.1 采集层 `app/services/crawler/`

**配置驱动而非硬编码。** 新增一个站点通常只改 `config/monitor_sites.json`，不写代码。
站点表可由监测名单 xlsx 生成：`python -m app.services.crawler.config.config_builder`。

**三种抓取模式**，由 `fetch_mode` 决定：

| 模式 | 适用 | 实现 |
| --- | --- | --- |
| 默认（浏览器） | 服务端渲染的新闻页 | `fetch/browser.py` Selenium + undetected-chromedriver |
| `json_api` | 前端渲染、无服务端 HTML 的 SPA | `fetch/json_api.py` 直接打站点的 list/detail 接口 |
| 固定选择器 | 列表容器已人工标注 | `config/list_selectors.json`，`status` 为 `ready`/`fixed` 才启用 |

**增量的关键设计**：`extract.list_date_for_anchor` **从列表页读日期，不打开文章**。
先在列表页筛出「比上次时间戳更新」的候选，再只打开这些。这样导航栏、页脚里的垃圾链接
不会消耗「最多打开 N 篇」的预算，导致真文章被漏掉。时间戳只推进到**实际处理完**的位置；
日期在任何地方都读不到的条目直接跳过，而不是留着每次重跑都再出现一遍。

**日期抽取是多源 fallback**（`fetch/dates.py`）：meta 标签 → JSON-LD → `<time>` →
页面顶部文本 → 列表页日期 → 正文头部。顺序是刻意的：结构化标记先行，正文启发式最后。

**正文抽取**（`fetch/detail.py`）：先剥离 `script/style/nav/header/footer/aside`、
cookie 同意弹窗、以及短的垃圾 `<form>`（长表单不动 —— ASP.NET WebForms 会把整页包在一个
`<form>` 里），再在 21 个候选容器里按文本长度选正文，全部落空时退回「文本减链接文本」
的 prose 打分。这一层的已知缺陷与改造方案见
[docs/services/ingestion-markdown-pipeline.md](./docs/services/ingestion-markdown-pipeline.md)。

### 4.2 索引层 `app/services/rag/indexer.py`

```
articles(embedding_done=False)
  → chunk_text(chunk_size=500, overlap=100)      # 递归字符分块，中英双语分隔符
  → embedder.encode_with_sparse(chunks)          # BGE-m3 一次前向 → dense + sparse
  → delete_by_article + add                      # 先清旧块，重跑安全（唯一约束不撞）
  → mark_embedding_done
```

**为什么是递归字符分块**：按段落 → 换行 → 中英句末标点 → 分句符 → 逗号逐级降级，
按**字符数**而非某个词表的 token 数计长度 —— embedder 是 BGE-m3（XLM-R 词表，
即 `SPARSE_DIM = 250002` 的来源），用 tiktoken 口径算 `chunk_size` 在中文上会偏得很远。

**阻塞调用一律甩线程池**：模型加载（import torch + 读几 GB 权重）和编码都是同步阻塞，
`asyncio.to_thread` 包起来；`RagService` 的构造函数刻意不碰模型，改在 `query` 里取单例
（`get_embedder_async`）。这个坑的完整记录见
[docs/services/rag-blocking-call-fix.md](./docs/services/rag-blocking-call-fix.md)。

### 4.3 检索层 `app/services/rag/facade.py`

```
question
  → encode                                      dense(1024) + sparse(250002)
  → chunk_repo.search        余弦 `<=>`  ─┐
  → chunk_repo.search_sparse 内积 `<#>`  ─┴→ retrieval.rrf_fuse(k=60)
  → rerank.rerank                             BGE-reranker-v2-m3 交叉编码器
  → retrieval.apply_min_score                  低于阈值丢弃 → 支持返回空
```

两路都建了 **HNSW 索引**（`vector_cosine_ops` / `sparsevec_ip_ops`）。
融合用 **RRF**（只看名次不看原始分）—— 稠密的余弦距离和稀疏的内积量纲根本不可比，
按分加权会引入一个没法调对的魔法系数。

`apply_min_score` 的阈值口径**随流水线位置变化**（纯稠密是 `1-cosine`、RRF 后是很小的
正数、rerank 后是 sigmoid 分），所以不能拿一个常数跨段套用 —— 代码注释里写明了这点。

开关：`RAG_HYBRID_ENABLED` / `RAG_RERANK_ENABLED`，两者任一为真即走完整流水线。

### 4.4 Agent 层 `app/agents/`

`StateGraph` 手工编排的 ReAct 单智能体，**推理与行动拆成独立节点**：

| 节点 | 职责 |
| --- | --- |
| `reasoning` | 产出 Thought，写回 `state["messages"]` |
| `action` | 依据刚才的 Thought 决定调工具还是直接作答 |
| `tools` | 执行工具，累加 `tool_rounds` |
| `finalize` | 预算用尽时收口作答 |

**轮次预算门 `_budget_gate` 放在 `tools` 之后而不是 `action` 之后**，这是个刻意的取舍：
`action` 跑完时那条带 `tool_calls` 的 AIMessage 已经进了 state，此时改判去 finalize 会在
消息历史里留下一个没有 ToolMessage 应答的悬空 tool_call（部分 provider 直接 400）。
放在 tools 之后，每个 tool_call 都拿到了结果，历史始终自洽，而且允许用满 `MAX_TOOL_ROUNDS`
轮 —— 最后一轮的结果照样能进最终答案。

**工具**（`agent_tools/__init__.py` 只暴露一个 `tools` 列表，`bind_tools` 与 `ToolNode`
共用它，杜绝「绑的工具」和「执行的工具」错配）：

| 工具 | 作用 |
| --- | --- |
| `search_rag_tool` | 内部情报库语义检索（薄适配层，重逻辑在 `services/rag`） |
| `read_article_tool` | 按 `article_id` 读全文 —— 命中的是片段，模型要读全文只能靠它 |
| `tavily_search` | 外网检索 |
| `tavily_extract` | 抓取指定 URL 正文 |

四个工具**输出同一形状**的 JSON（`{"results":[{title,url,published_at,content,score}]}`），
异常一律兜成 `{"results": [], "error": ...}`，避免报错文本被当成检索内容喂给模型。

**Provider 抽象**：`OllamaProvider` / `OpenAIProvider` / `BedrockProvider` 共用
`LLMProvider` 接口。`get_llm_provider(name)` 支持显式覆盖 `LLM_PROVIDER` —— 评估的中立裁判
就靠这个换到另一家模型。

Agent 目前**不经 HTTP 暴露**，通过 `langgraph dev`（图注册在 `backend/langgraph.json`）
或评估入口 `eval/run_eval.py` 驱动；`POST /api/v1/rag/query` 只做检索、不生成答案。

### 4.5 API 层

| 端点 | 说明 |
| --- | --- |
| `POST /api/v1/rag/query` | 语义检索，返回带出处的文章块 |
| `GET  /api/v1/articles/today` | 今日文章，可按 `group_type` 过滤 |
| `GET  /api/v1/articles/history` | 分页历史，支持分组/站点/日期区间/关键词 |
| `GET  /api/v1/articles/{id}` | 单篇详情（含 `body_text`） |
| `POST /api/v1/users/signup` `login/access-token` `login/refresh-token` `logout` `logout-all` | JWT 认证（access + refresh，refresh 落 `sessions` 表可撤销） |
| `GET/PATCH/DELETE /api/v1/users/me` `password-recovery` `reset-password` | 账户自助 |
| `GET/POST/PATCH/DELETE /api/v1/users/…` | 管理员用户管理（`RoleChecker`） |
| `GET  /api/v1/utils/health-check/` | 健康检查 |

中间件从外到内：`RequestID → CORS → RateLimit → Session → 异常处理 → 路由`。
顺序是设计过的：RequestID 最外层，被拒绝或崩溃的请求也可追踪；CORS 在限流之外，
浏览器才读得到 429，且预检不算进调用方额度。

交互式文档：`http://localhost:8848/docs`。

---

## 5. 数据模型

```
users ──1:N──▶ sessions            （refresh token 会话，可撤销）

articles ──1:N──▶ article_chunks
  id            BigInteger PK      ← 必须显式 BigInteger：裸 int 会让 autogenerate
  site_name     索引                  把它收窄回 int4，外键随即失败
  title / url   url 唯一 = 天然去重键
  publish_date  索引
  body_text     正文（BODY_CHAR_LIMIT 截断）
  summary
  group_type    competitor / regulatory，索引
  is_external
  embedding_done  ← 索引 worker 的唯一队列信号
                       │
                       ▼
                  article_chunks
                    article_id + chunk_index  唯一
                    text
                    group_type                     从文章冗余下来，检索时省一次 JOIN 过滤
                    embedding         vector(1024)      HNSW / vector_cosine_ops
                    sparse_embedding  sparsevec(250002) HNSW / sparsevec_ip_ops，可空
```

**迁移链**（`backend/alembic/versions/`）：

| revision | 内容 |
| --- | --- |
| `a1b2c3d4e5f6` | users + sessions |
| `b2c3d4e5f6a7` | articles |
| `c3d4e5f6a7b8` | article_chunks + pgvector 扩展 + 稠密 HNSW |
| `d4e5f6a7b8c9` | sparse_embedding + 稀疏 HNSW |

> 拉取代码后**必须** `uv run alembic upgrade head`，否则稀疏列不存在，hybrid 检索直接报错。

`repository/article.py` 的一个陷阱值得留意：`embedding_done` 的重置条件绑在 `body_text`
的变化上（`case(Article.body_text != literal(body_text), ...)`）—— 正文变了才重新索引。
将来新增内容列时必须同步这段，否则新内容永远不会触发重建。

---

## 6. 质量保障：评估与测试

### 6.1 RAG 评估层 `backend/eval/`

自建 **33 题标注集**（`dataset.jsonl`，每题一个 gold article），两个维度：

**检索质量（IR 指标）** —— 三配置对比：

| 配置 | recall@5 | MRR | nDCG@5 | 平均去重后文章数 |
| --- | --- | --- | --- | --- |
| dense | **0.985** | 0.970 | 0.971 | 2.12 |
| hybrid | 0.919 | 0.779 | 0.800 | 3.12 |
| hybrid + rerank | 0.957 | **1.000** | **0.972** | 1.97 |

**这里最有价值的发现是一个负结果**：hybrid 相比纯稠密**反而退化**。定位原因是稀疏路径用
内积（`sparsevec_ip_ops`）、**没有 BM25 式的文档长度归一化**，语料里约 20% 的样板文章
（页脚被当正文抽出来，正文仅 131~259 字）单 token 权重被抬高，RRF 融合后挤掉了 gold。
交叉编码器精排能完全滤除这类噪声（MRR 回到 1.000），但那是在浪费候选池 —— 根因在摄取层，
改造方案见 [ingestion-markdown-pipeline.md](./docs/services/ingestion-markdown-pipeline.md)。

**生成质量（RAG 三元组，LLM-judge）**，n=33：

| 指标 | 得分 | 判什么 |
| --- | --- | --- |
| faithfulness | 0.970 | 答案每句是否有检索证据支撑（有据性） |
| context_relevance | 0.603 | 召回的上下文有多少真的用得上 |
| answer_relevance | 1.000 | 答案是否直接回答了问题（只判切题） |

**中立裁判**是这里的关键设计：生成用 `us.anthropic.claude-sonnet-4-5`，评判用
`us.anthropic.claude-opus-4-5`（`RAG_EVAL_JUDGE_PROVIDER` / `RAG_EVAL_JUDGE_MODEL`），
规避「生成=评判同模型」的自评偏差。裁判走**离散判定 + 结构化输出**，
而非让模型打 0~1 的连续分。

完整原理、跑法、以及已知的评估盲区见 **[README-EVAL.md](./README-EVAL.md)**。

### 6.2 单元测试 `backend/tests/`

- 目录**镜像** `app/` 结构；`anyio` 而非 pytest-asyncio（`@pytest.mark.anyio`）；
- 全程**离线**：无数据库、无 Redis、无网络。`mock_db` 是 `AsyncMock`，测试各自 stub 所需的仓储函数；
- 测试替身（`FakeUser` / `FakeArticle` / `auth_headers_for`）集中在 `tests/helpers.py`；
- API 测试用 `httpx.AsyncClient` + `create_app()`，只覆盖 `get_db_session` ——
  路由、认证、RBAC、`response_model` 序列化全部真跑；
- `addopts` 开了 `--strict-markers`，marker 打错是**报错**而不是静默跳过。

约定详见 [`.claude/rules/testing.md`](./.claude/rules/testing.md) 与
[`backend/tests/README.md`](./backend/tests/README.md)。

---

## 7. 快速开始

### 7.1 Docker Compose

仓库根需要一个 `.env`（compose 通过 `env_file` 读它，未提交）。必填项见
[§8 配置说明](#8-配置说明)，另需 `DOCKER_IMAGE_BACKEND` / `DOCKER_IMAGE_FRONTEND` / `DOMAIN`；
LLM 与工具相关的键可参照 [`backend/.env.example`](./backend/.env.example)。缺变量时
compose 会直接报 `Variable not set` 而不是静默启动。

```bash
docker compose up -d
```

| 服务 | 地址 | 说明 |
| --- | --- | --- |
| backend | http://localhost:8000/docs | FastAPI + 交互式文档 |
| frontend | http://localhost:5173 | 构建后的前端 |
| adminer | http://localhost:8080 | 数据库 GUI |
| db | localhost:5432 | Postgres + pgvector；数据在 `app-db-data` 卷 |

> ⚠️ 不要用 Docker Desktop 的 “Reset to factory defaults” —— 会删掉 `app-db-data` 卷，
> 语料和向量全部丢失。

`prestart` 服务会等数据库就绪、跑迁移、建初始超级用户（`backend/bootstrap/`）。

### 7.2 本地开发

```bash
# 后端
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8848

# 前端
cd frontend
bun install
bun dev
```

### 7.3 跑各条链路

```bash
cd backend

# 爬虫：抓取 + 写库（--scrape-only 只抓不写，用于排查站点）
uv run python -m app.services.crawler
uv run python -m app.services.crawler --scrape-only

# 由 xlsx 生成站点表
uv run python -m app.services.crawler.config.config_builder

# 索引 worker：把 embedding_done=False 的文章切块编码入库
uv run python -m app.services.rag

# Agent 交互调试（LangGraph Studio）
uv run langgraph dev

# RAG 评估（需要 LLM 凭证；报告落 eval/reports/）
uv run python -m eval.run_eval
```

### 7.4 常用开发命令

```bash
cd backend
uv run pytest                                  # 全量测试
uv run pytest tests/services/rag -v            # 单目录
uv run pytest -k "register" -v                 # 按名筛选
uv run ruff check . --fix && uv run ruff format .
uv run ty check
uv run alembic revision --autogenerate -m "描述"

cd frontend
bun test && bun run lint
../scripts/generate-client.sh                  # 由后端 OpenAPI 重新生成 SDK
```

> 环境提示：仓库根的 `.python-version` 是 `3.13`。若 `uv sync` 因 Python 版本约束反复
> 改动环境，用 `uv run --no-sync <cmd>` 跳过同步。

---

## 8. 配置说明

配置集中在 `backend/app/core/config.py`（Pydantic `BaseSettings`），从 `.env` 读取。

**必填**：`PROJECT_NAME` · `POSTGRES_SERVER` · `POSTGRES_USER` · `FIRST_SUPERUSER` ·
`FIRST_SUPERUSER_PASSWORD`。部署前至少改 `SECRET_KEY` · `POSTGRES_PASSWORD` ·
`FIRST_SUPERUSER_PASSWORD`（生成：`uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`）。

**RAG**：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_EMBED_MODEL` | `BAAI/bge-m3` | 稠密 1024 维 + 稀疏，一次前向双出 |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | 500 / 100 | 字符数口径 |
| `RAG_TOP_K` / `RAG_FETCH_K` | 5 / 30 | 返回条数 / rerank 前的 over-fetch |
| `RAG_HYBRID_ENABLED` | `false` | 稠密 + 稀疏 RRF 融合 |
| `RAG_RERANK_ENABLED` | `false` | 交叉编码器精排 |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | |
| `RAG_MIN_SCORE` | `0.0` | 低于此分丢弃（口径随流水线段变化） |
| `RAG_PRELOAD` | `false` | 启动时预热嵌入模型；失败只告警不阻断启动 |
| `RAG_INDEX_BATCH` | 100 | 索引 worker 每批文章数 |

**LLM / 工具**：

| 变量 | 说明 |
| --- | --- |
| `LLM_PROVIDER` | `ollama`（默认）或 `openai` |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` | |
| `OPENAI_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` | |
| `BEDROCK_REGION` / `BEDROCK_MODEL` | 凭证走 boto3 默认链，不在配置里读 |
| `RAG_EVAL_JUDGE_PROVIDER` / `RAG_EVAL_JUDGE_MODEL` | 评估的中立裁判；留空则退回生成用的 provider |
| `TAVILY_API_KEY` / `TAVILY_SEARCH_DEPTH` / `TAVILY_MAX_RESULTS` | 联网检索 |
| `LANGSMITH_TRACING` / `LANGSMITH_PROJECT` / `LANGSMITH_API_KEY` | 全链路追踪（靠环境变量自动生效） |

**其他**：`RATE_LIMIT_*`（默认 300 req/60s）· `SESSION_*`（默认关）· `SENTRY_DSN`
（仅非 local 环境上报）· `DB_ECHO` · `DB_POOL_*` · `SMTP_*`。

**爬虫配置**在仓库根的 `config/`（不在 `backend/` 下，由 `crawler/common/paths.py` 解析；
可用 `MONITOR_DATA_DIR` 把它挪出仓库）。仓库内这些 json 是**空占位**，实际内容是本地文件。

---

## 9. 已知短板与路线图

诚实记录当前状态，按优先级：

**摄取层（影响最大）** —— 详见
[ingestion-markdown-pipeline.md](./docs/services/ingestion-markdown-pipeline.md)（方案已定，未实施）：

1. **抽取质量门缺失**：`pipeline.py` 只检查正文长度是否为 0，131 字的页脚顺利通过并打成功日志。
   加最小正文长度门 → 抽取失败**可见**而非静默。这一步独立、成本最低，应该先做。
2. **正文抽取换 trafilatura**：现有 21 选择器 + 「最长者胜」判别式有结构缺陷
   （通配选择器会命中祖先容器，而祖先文本必然更长）。现有启发式降为 fallback。
3. **保留结构，归一到 Markdown**：`get_text()` 把标题层级、列表、表格全拍平了，
   任何结构感知分块都无从施展。改为输出 Markdown → `MarkdownNodeParser` 按标题切语义段
   → 现有 `chunk_text` 按尺寸切小块。
4. **接入 PDF**（`pymupdf4llm`）：`fetch/urls.py` 的 `is_document_url` 已在识别 PDF 链接，
   路由钩子半个就位。企业内部文档是既定的下一批数据源。

**检索与评估**：

5. `RAG_HYBRID_ENABLED` / `RAG_RERANK_ENABLED` 目前默认 `false`，等摄取层的样板污染修掉再翻开。
6. `recall@k` 口径：命中的是 chunk，去重成文章后实际返回数少于 k，需要统一口径。
7. **现有 33 题在单 gold 题上已饱和**（dense 0.985，29/33 满分）且全是短新闻 ——
   换任何分块器都测不出差异。需要补长文档、跨 heading 聚合、表格数字这类题，
   加了 PDF 之后自然就有。
8. `build_report` 不存答案原文，导致「分数满分但输出有问题」的 case 事后无法追溯。

**Agent**：

9. `ACT_INSTRUCTION` / `FINAL_INSTRUCTION` 的措辞会让最终答案泄露内部脚手架
   （出现「根据我刚刚的 Thought…」）。`answer_relevance` 拿满分也抓不到 ——
   它按设计只判切题、不判语气详略。属 prompt 问题，非指标问题。

**基础设施**：

10. `torch` 当前是 CPU build，GPU 闲置；全量重建索引前应先装 CUDA build。
11. `lifespan` 里 Redis 与 Bot polling 是预留槽位，尚未实现。

---

## 致谢

后端骨架源自 [full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)
（@tiangolo），分层约定参考 [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)（@zhanymkanov）。

## License

MIT，见 [LICENSE](./LICENSE)。
