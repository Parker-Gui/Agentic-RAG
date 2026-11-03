# RAG 评估层 (`backend/eval/`)

> 本文讲清楚 RAG 评估层**是什么、每一块的技术原理、怎么实现的,以及未来可以怎么优化**。
> 面向想理解或接手这套评估体系的人。

---

## 1. 这是什么 · 为什么需要

`eval/` 是本 RAG 系统的**质量度量层**——把"这套检索/回答到底好不好"从**感觉**变成**可量化、可对比、可追溯的数字**。它回答三类问题:

- **检索找得准吗?** hybrid / rerank 相比纯稠密,到底提升了多少?
- **答案在编吗?** agent 给的事实,是否都被检索到的上下文支撑?
- **答非所问吗?** 召回的上下文、生成的答案,和问题相关吗?

它是**离线批处理**工具:不在 agent 运行时、不在 API 请求路径里,而是开发者主动跑 `python -m eval.run_eval` 的一次性评测(类比索引 worker `python -m app.services.rag`)。

它的价值有三层:
1. **调参决策依据**——该不该开 hybrid、rerank 值不值那 2GB 模型 + 延迟、`chunk_size` 取多少,用数字拍板而非拍脑袋。
2. **回归防线**——改了检索/prompt/模型后重跑,防止"优化了 A 悄悄搞崩了 B"(是 RAG 的"基准测试",与单元测试互补:单测保证代码不报错,eval 保证质量不退化)。
3. **可展示的成果**——一张 `dense → hybrid → +rerank` 的指标对比表,证明"评估驱动开发",而非功能堆砌。

---

## 2. 目录结构 · 文件地图

```
backend/eval/
├── metrics.py            # IR 指标纯函数：recall@k / MRR / nDCG@k
├── retrieval_eval.py     # 检索评测：注入 query_fn，逐题跑→聚合平均
├── judge_base.py         # 三个 LLM-judge 共用的「中立裁判」构造器
├── faithfulness_eval.py  # Faithfulness judge（答案 vs 上下文）+ 聚合
├── relevance_eval.py     # Context / Answer Relevance 两个 judge
├── triad_eval.py         # 三元组编排：一次 agent 运行喂三个 judge（并发）
├── report.py             # 报告产物化：build_report / render_markdown / save_report
├── run_eval.py           # CLI 入口：python -m eval.run_eval
├── dataset.jsonl         # 评估集：每行 {question, expected_article_ids}
└── reports/              # 生成的报告（.gitignore 忽略内容）

backend/tests/eval/test_eval.py   # 全部注入 fake，不连库/模型
```

评估分**两大维度**:检索质量(§3,不用 LLM)和生成质量(§4,用 LLM-judge)。

---

## 3. 维度一:检索质量 — IR 指标

**目标**:检索返回的文章,和"标准答案文章"比,找得全不全、排得靠不靠前。**客观、可复现、便宜(不用 LLM)**,只需要一份人工标注的评估集。

### 3.1 数据契约

评估集 `dataset.jsonl` 每行:

```json
{"question": "有哪些无可可巧克力竞品？", "expected_article_ids": [2, 96]}
```

- `gold` = `expected_article_ids`(该命中的正确文章 id 集合,人工标注)
- `retrieved` = 检索返回命中的 `article_id` 列表(**去重保序**:同一篇文章的多个 chunk 只算一次,保留首次名次)

### 3.2 三个指标(`metrics.py`,纯函数)

| 指标 | 公式(直观) | 衡量 |
|---|---|---|
| **recall@k** | `\|retrieved[:k] ∩ gold\| / \|gold\|` | 前 k 个里**找全**了多少 |
| **MRR** | 第一个命中的 `1/rank`(没命中=0) | 正确结果**排得多靠前** |
| **nDCG@k** | `DCG / IDCG`,`DCG = Σ 1/log2(i+1)`(命中项累加) | 综合**排序质量**(越靠前得分越高) |

三者互补:recall 只看"在不在前 k",MRR 只看第一个命中,nDCG 综合考虑所有命中的位置。

### 3.3 聚合与三配置对比(`retrieval_eval.py`)

`evaluate_retrieval(dataset, *, query_fn, k)` 对每题跑 `query_fn` → 去重成文章 id → 和 gold 比 → 三个指标在数据集上取平均。

**精髓在 `query_fn` 可注入**:`run_eval` 用它注入三种检索配置,于是能量化对比:

```
dense          （纯稠密向量）
hybrid         （稠密 + 稀疏 BGE-m3 lexical，RRF 融合）
hybrid+rerank  （再加 bge-reranker 交叉编码精排）
```

### 3.4 被评估对象的原理(检索侧简述)

评估的检索链路本身(实现在 `app/services/rag/`)值得一提,因为指标好坏由它决定:

- **Dense(1024 维)**:BGE-m3 输出的语义向量(模型 hidden size),比 **cosine** 相似度。懂同义/改写,但对精确专名/术语不敏感。
- **Sparse(250002 维)**:BGE-m3 的 lexical_weights,维度=词表大小,只存非零 `{token_id: weight}`,比**内积**。本质是"加权关键词匹配",精确但不懂同义。
- **RRF 融合**:dense(cosine ~0.6)与 sparse(内积,量纲不同)分数**不可比**,所以用倒数排名融合(Reciprocal Rank Fusion)——只看各自排第几名,不看绝对分。
- **Rerank**:交叉编码器(bge-reranker-v2-m3)把 `(query, chunk)` 成对喂模型,输出校准的相关性分 `[0,1]`,只对召回的 top-N 精排。

---

## 4. 维度二:生成质量 — RAG 评估三元组(LLM-judge)

**目标**:检索指标管不到"生成"这一侧——答案是否有据、是否切题。这三项合称 **RAG triad**:

| 指标 | judge 输入 | 判什么 | 聚合成 |
|---|---|---|---|
| **Faithfulness** | 答案 + 上下文 | 答案的事实都被上下文支撑吗(有没有编) | 有据率 |
| **Context Relevance** | 问题 + 上下文 | 召回的每条上下文与问题相关吗 | 信噪比(相关数/总数) |
| **Answer Relevance** | 问题 + 答案 | 答案切题回答了问题吗(≠有据性) | 切题率 |

### 4.1 LLM-judge 的核心原理:**离散判定 + 结构化输出 + 数据集聚合**

关键认知——**不是让 LLM 直接吐一个 0.87 的分数**(那样不可靠、不可解释)。而是三步:

1. **对单个样本做一次离散的、有明确标准的判断**(LLM 擅长的二元语义判断),例如"这条上下文相关吗:是/否"。
2. **用结构化输出强制返回定型结果**——`with_structured_output(PydanticVerdict)`。它把 pydantic schema 转成 JSON schema,按 provider 能力用 tool-calling 或 JSON mode **约束模型输出符合 schema 的 JSON**,再解析回 verdict 实例。**"保证拿到 bool"靠的是 schema 约束 + 解析,不是在 prompt 里央求模型。**
3. **在整个数据集上把这些离散判定聚合成比率**——"量化"交给统计,不交给 LLM。

一个 judge = 4 个部件(以 Faithfulness 为例):

```python
# ① Verdict：判定结果的形状
class GroundednessVerdict(BaseModel):
    grounded: bool
    reason: str = ""

# ② System prompt：把判定规则写死（哪种情况 false）
JUDGE_PROMPT = "你是事实核查员。只依据<上下文>判断<答案>里的事实性陈述是否都能被支撑……"

# ③④ 构造裁判 model（结构化输出）+ 拼输入调用
model = build_judge_model(GroundednessVerdict)   # 见 §4.2
result = await model.ainvoke([SystemMessage(JUDGE_PROMPT), HumanMessage(payload)])
```

三个 judge 复用同一套模板(`faithfulness_eval.py` / `relevance_eval.py`),只换 Verdict + prompt + 输入。

**为什么用 LLM 而非规则/embedding**:相关性/有据性是语义判断,规则(关键词)判不了(同义、改写、数字换算);embedding 相似度能凑合做 Answer Relevance,但判不了有据性(相似 ≠ 被支撑——方向反了、否定、数字错都可能相似度很高)。代价:慢、花钱、**裁判本身可能判错**(见 §4.3、§7)。

### 4.2 中立裁判:治"自评偏差"(`judge_base.py` + `BedrockProvider`)

**问题**:若裁判 = 生成答案的同一个模型,它会**自我偏袒**(self-enhancement bias,倾向认为自己没编)。

**方案**:裁判用**跨家族**的中立模型——本项目用 **AWS Bedrock 上的 Claude**(生成侧默认 ollama/openai,天然跨家族)。三个 judge 共用一个构造器:

```python
# judge_base.py
def build_judge_model(output_schema):
    provider = get_llm_provider(settings.RAG_EVAL_JUDGE_PROVIDER)   # 可指定裁判 provider
    return provider.build_model(
        settings.RAG_EVAL_JUDGE_MODEL, temperature=0                # temp=0 求判定可复现
    ).with_structured_output(output_schema)
```

配套改动(见 `app/agents/providers.py`):
- 新增 `BedrockProvider`(`langchain_aws.ChatBedrockConverse`;凭证走 boto3 默认链,不进代码);
- `get_llm_provider(name)` 支持指定 provider(裁判可与生成不同源);
- `build_model(**kwargs)` 透传 `temperature=0`。

配置(`RAG_EVAL_JUDGE_PROVIDER` 留空则退回生成 provider,行为向后兼容):

```
RAG_EVAL_JUDGE_PROVIDER=bedrock
RAG_EVAL_JUDGE_MODEL=us.anthropic.claude-...   # Bedrock inference profile id
BEDROCK_REGION=us-east-1
```

> **裁判选型原则**:① 跨家族(与生成不同源)② 能力 ≥ 生成模型(judge 质量 = 评估可信度上限,别用弱模型当裁判)③ 结构化输出稳 + 便宜 + 低温。

### 4.3 编排:一次 agent 运行喂三个 judge(`triad_eval.py`)

agent 端到端很贵(LLM 生成 + 多轮检索),所以**每题只跑一次 agent**,产出的 `{answer, contexts}` 同时喂三个 judge,**三次评判并发**(`asyncio.gather`):

```
run_agent(question) → {answer, contexts}
   ├─ Context Relevance : judge(question, contexts)
   ├─ Faithfulness      : judge(answer, contexts)
   └─ Answer Relevance  : judge(question, answer)
→ 每题三个分 → 数据集聚合成三个 rate + 逐题 details（含每个 judge 的 reason）
```

三个 judge 均可注入,便于单测(见 §7)。

---

## 5. 报告产物化(`report.py`)

否则 `run_eval` 只在终端打印汇总 rate,**逐题明细与 judge 的 reason 算了就丢**——指标低时无法追溯是哪一题、裁判给了什么理由。

- `build_report(retrieval, triad)` / `render_markdown(report)` — **纯函数**(可单测)
- `save_report(report)` — 写 `eval/reports/report_<UTC时间戳>.{json,md}`(IO)

报告含:**检索三配置对比表 + 三元组 + 逐题明细 + 低分明细(带 judge reason,可追溯)**。样例(节选):

```markdown
## 检索指标 (recall@5 / MRR / nDCG@5)
| 配置 | recall@5 | MRR | nDCG@5 |
| dense | 0.500 | 0.500 | 0.420 |
| hybrid+rerank | 0.800 | 0.780 | 0.740 |

## 低分明细（可追溯 reason）
- **甜味剂出口同比增速**
  - faithfulness=0.0 — 答案里的12.3%在检索上下文中找不到
```

---

## 6. 怎么跑 · 前提

```bash
cd backend
uv run python -m eval.run_eval          # 或 uv run --no-sync python -m eval.run_eval
```

流程:加载 `dataset.jsonl` → 三配置各跑 IR 指标并打印 → 跑三元组 → 落 JSON+Markdown 报告。

**真跑出有意义数字的前提**(缺一不可):
1. **DB 迁移 + 有索引数据**:`alembic upgrade head`(含 sparse 列迁移 `d4e5f6a7b8c9`)+ 已索引的 `article_chunks`;
2. **`dataset.jsonl` 标注好**:20–50 条 `问题 → expected_article_ids`(现仅 1 行示例);
3. **生成侧 LLM 环境**:三元组要跑整张 agent 图(检索指标不需要 LLM,纯本地);
4. **中立裁判凭证**:Bedrock model id + AWS 凭证。

> Windows 注意:入口用 `run_compatible_async`(而非裸 `asyncio.run`),否则 psycopg 在默认 ProactorEventLoop 下报 `InterfaceError`。

**关键配置项**(`app/core/config.py`):`RAG_TOP_K`、`RAG_FETCH_K`(rerank over-fetch)、`RAG_MIN_SCORE`(阈值)、`RAG_HYBRID_ENABLED` / `RAG_RERANK_ENABLED`、`RAG_EVAL_JUDGE_PROVIDER/MODEL`、`BEDROCK_*`。

---

## 7. 测试策略(`tests/eval/test_eval.py`)

全部**注入 fake、不连库/模型**(anyio):

- **IR 指标**:注入假 `query_fn`,验证 recall/MRR/nDCG 聚合、article_id 去重。
- **纯函数边界**:`context_relevance_score` 的空上下文 / 越界(幻觉)编号过滤 / 去重。
- **三元组编排**:注入假 `run_agent` + 三个假 judge,验证三 rate 聚合、逐题 details 带 reason、空数据集短路(不构造裁判)。
- **报告**:`build_report` 结构、`render_markdown` 关键段落 + 低分 reason 可追溯、`save_report` 落两文件(`tmp_path`)。

> `app/repository` 的 SQL 无法离线测,IR 指标通过纯函数 + 注入 query_fn 覆盖聚合逻辑,真实检索质量靠真跑 `run_eval` 验证。

---

## 8. 关键设计决策与踩坑

- **RAGAS 撤掉,改手写三元组**:RAGAS(0.3.1)与本项目 langchain 1.x 不兼容(import `langchain_community.chat_models.vertexai` 失败),已撤依赖。手写 LLM-judge 反而更可控、更贴合项目栈。
- **`audit.py` 随 ReAct 重构删除**:早期 faithfulness 想复用 `audit.py` 的**确定性机械核对**(数字/日期/引用逐字对),重构后 audit 没了,改为纯 LLM-judge。代价:失去了比 LLM 更硬的那层核对(见 §9)。
- **LLM 不打分,做离散判定 + 聚合**:可靠性的来源(§4.1)。
- **中立裁判 + `temperature=0`**:治自评偏差 + 保判定可复现。
- **`run_eval` 允许 `print`**:它是评测报告 CLI,打印结果是本职(pyproject 对它豁免 `T201`);测试的 fake 回调允许未用参数(豁免 `ARG001`)。

---

## 9. 未来优化路线

按"该先做 → 锦上添花"排序:

### 🔴 该修的洞(影响可信度)
- **contexts 清洗**:`run_eval._run_agent` 目前把工具的**原始 JSON**(`{"results":[...]}` 整段,含 tavily/rag 混合)直接当 contexts 喂 judge → judge 要从噪声里读,三元组数字失真。应先解析出干净的检索片段(title/content)再喂。**这是当前最该先补的一块。**

### 🟡 值得加(提升可信度/实用性)
- **裁判 self-consistency(多次投票)**:LLM-judge 单次判定有噪声,对每题判 N 次取多数,降方差。
- **no-answer / 拒答评估**:情报 agent 的关键能力——库里没有时该说"没有"或走外网,而非编。评估集加"无答案"题,看 agent 是否正确拒答。与本项目"可信度"卖点高度契合。
- **成本 / 延迟记录**:评估时记录每配置的耗时/token,把 rerank 的 trade-off("提升 nDCG 但每查增加 X ms + 2GB 模型")**用数字坐实**。
- **进度日志**:评估过程目前几乎无日志,大数据集跑起来看不到进度;加逐题 `logger.info`。

### 🟢 锦上添花
- **确定性核对 + LLM-judge 结合**:把旧 `audit` 那套数字/引用机械核对加回来,与 LLM-judge 双层,做更可信的 faithfulness(机械核对确定性部分,LLM 判语义部分)。
- **per-config 三元组**:像 IR 那样对 dense/hybrid/rerank 分别跑三元组,看检索配置对生成质量的影响(但 agent 端到端贵)。
- **CI gate / 阈值**:指标低于阈值 fail 构建。
- **历史趋势对比**:多次报告 diff(现每次独立 timestamp,无对比)。

### ⚪ 前提(非功能,但没有就空转)
- **`dataset.jsonl` 标注**(20–50 条)+ **生成侧 LLM 环境** + **Bedrock 凭证**。

---

## 附:一句话总结

**检索侧(IR 三指标)客观便宜、能量化 hybrid/rerank 提升;生成侧(三元组 + 中立裁判)覆盖 IR 管不到的"有据、切题";报告让指标可追溯。当前代码完整、测试齐(注入 fake),真正的瓶颈是喂数据(标注 dataset)与打通 LLM 环境,而非继续堆功能。**
