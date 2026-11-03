# 摄取管道改造：保留结构的 Markdown 中间表示

> 状态：**方案已定，未实施**。
> 目标：把「爬虫 → 正文 → 分块」这一段从"抽取即拍平"改成"去样板 + 保结构"，并让 HTML 与 PDF 走同一条下游管道。
> 检索栈（BGE-m3 编码、`article_chunks` 存储、hybrid/RRF/rerank）**不在改动范围内**。

---

## 1. 背景：为什么要改

### 1.1 实测证据

RAG 评估（见 [README-EVAL.md](../../README-EVAL.md)，报告 `backend/eval/reports/`）暴露出两个问题：

**约 20% 的语料是不可用内容。** 27 篇样板文章（Layn 24-33、浩天 114-125、Symrise 栏目 86-90），语料总量约 130 篇。浩天那批正文只有 131~259 字——**页脚被当成正文抽出来了**。

它带来的直接指标退化：

| 配置 | recall@5 | MRR | nDCG@5 | 受污染题数 |
| --- | --- | --- | --- | --- |
| dense | **0.985** | 0.970 | 0.971 | 0/33 |
| hybrid | 0.919 | 0.779 | 0.800 | **5/33** |
| hybrid+rerank | 0.957 | **1.000** | **0.972** | 0/33 |

机制：稀疏路径用内积（`sparsevec_ip_ops`），**没有 BM25 式的文档长度归一化**，短样板文本单 token 权重被抬高，RRF 融合后挤掉 gold。重排能完全滤掉，但那是在浪费候选池。

> 注：这 27 篇中有多少是**抽取错**（选中页脚）、多少是**候选筛选错**（栏目索引页被当成文章爬进来），需要看原始 HTML 才能确定。后者任何抽取库都救不了，需要在候选筛选层修。

**这 20% 是靠下游检索指标才被发现的。** 抽取阶段完全静默：`orchestration/pipeline.py` 的质量判断只看 `len(body_text.strip())` 是否为 0，131 字的页脚顺利通过并打了成功日志。

### 1.2 结构信息在爬取阶段就被销毁

`fetch/detail.py`：

```python
text = _find_content_container(soup).get_text(separator="\n", strip=True)
#      ↑ 去样板（正确，要保留）      ↑ 拍平结构（要改掉）
```

一行里焊了两件事。到 `indexer.py` 拿到 `article.body_text` 时，只剩扁平字符串——没有标题层级、列表、表格。这导致：

- 任何**结构感知分块**都无从施展（喂扁平文本给结构化 parser，它退化成普通字符分块）
- `BODY_CHAR_LIMIT = 8000` 的截断本是为摘要器的上下文窗口设的，却连累 RAG——**长文章 8000 字之后的内容从未进过索引**

### 1.3 手写抽取的结构缺陷

`_find_content_container` 的判别式是「21 个选择器的匹配中，`get_text()` 最长者胜」。两个问题：

1. **通配选择器会命中祖先容器。** `[class*='article']`、`[class*='content']` 可能匹配到 `<div class="page-content">`（包着 nav + 正文 + footer），祖先文本长度**必然**大于正文，所以一旦命中祖先就系统性夺冠。这不是调阈值能修的。
2. **更好的判别式被更差的挡在后面。** `_prose_len`（总文本 − 链接文本，即 link density）质量明显更高，但它**只在 21 个 named selector 全部失败时才跑**。正确的结构应该是所有候选都过 link-density 打分，选择器只提供候选。

另外选择器列表是无上界的维护负担——现在 21 条，每接一个新站点就可能 +1~2（`.w-richtext` 那条注释即实证）。

---

## 2. 方案

### 2.1 管道

```
HTML → trafilatura.extract(output_format="markdown")  ┐
                                                      ├→ Markdown
PDF  → pymupdf4llm.to_markdown()                      ┘     │
                                                            ▼
                                     MarkdownNodeParser（按标题切语义大块）
                                                            │
                                                            ▼
                                     chunk_text（节内按尺寸切小块）
                                                            │
                                          爬虫的 title/date/url 作为前缀拼入
                                                            ▼
                                          之后不变：BGE-m3 编码 → article_chunks
                                                   → hybrid + RRF + rerank
```

**核心思想：解析层 source-specific，归一到 Markdown；分块层 source-agnostic。**

两个解析器**都直接输出 markdown**，没有额外的「HTML → Markdown」转换步骤。

### 2.2 分层职责

| 层 | 做什么 | 用什么 | 手写值不值 |
| --- | --- | --- | --- |
| 列表页候选提取 | 从列表页找出文章链接 + 日期 | 现有 `fetch/extract.py` | **值，是资产** |
| 日期抽取 | meta / JSON-LD / `<time>` / 列表页 / 正文头部多源 fallback | 现有 `fetch/dates.py` | **值** |
| 正文抽取（HTML） | 判断哪部分是正文，丢掉页眉/导航/广告/页脚，**保留结构** | **trafilatura**（现有启发式作 fallback） | **不值，换库** |
| 正文抽取（PDF） | 版面与阅读顺序 → markdown | **pymupdf4llm** | 新增能力 |
| 结构分块 | 按标题切语义段 | **LlamaIndex `MarkdownNodeParser`** | 换库 |
| 尺寸分块 | 节内递归字符分块 | 现有 `rag/chunking.py` | **值，保留** |

`extract.py` 中 `list_date_for_anchor` **从列表页读日期、不打开文章**——这省下了打开文章的预算，也保证垃圾链接不会占额度导致漏掉真文章。这是没有库能替代的设计决策。

---

## 3. 技术选型与理由

### 3.1 为什么是 trafilatura，而不是继续手写

trafilatura 的核心不是选择器白名单，是 **link density + 文本块打分 + DOM 结构判别**：

- 打的是**相对分数**，能拒绝"最长但不是正文"的容器——正好对应 §1.3 的两个缺陷
- **站点无关**，不用维护选择器表
- 在公开的正文抽取 benchmark 上长期是 F1 第一档

**保留现有启发式作 fallback**：那 21 个选择器 + prose-score 是对这批站点的真实知识（cookie-consent 清除、ASP.NET 整页 `<form>` 的长度守卫、Modernizr `generatedcontent` 类导致 `<html>` 夺冠的排除），trafilatura 未必都覆盖。库先跑，库失败或输出过短时落回。

### 3.2 为什么不用 LlamaIndex 的 HTML reader

- **`TrafilaturaWebReader`** 就是 trafilatura 的一层包装，但返回 `Document(text=...)`——**纯文本，且丢掉 metadata**。不如直接调 `trafilatura.extract()`。
- **`SimpleWebPageReader(html_to_text=True)`** 用 html2text 转**整页**，零样板去除。
- **`HTMLTagReader`** 要你指定抽哪个 tag——就是 `_CONTENT_SELECTORS` 的问题搬过来，且只有一个选择器、无 fallback、无 prose-score。**比现有实现更差。**
- **unstructured `partition_html`** 把文档切成 `Title`/`NarrativeText`/`ListItem` 元素，但**不判断哪部分是文章**，导航菜单会作为 `ListItem` 与正文并列。

结论：LlamaIndex 对这个问题的"好答案"本身就是调 trafilatura。直接调库，保住 metadata 和 fallback 的控制权。

### 3.3 为什么 metadata 不用 trafilatura 的

title / date 爬虫已经拿到了，而且质量更高：日期走 `extract_publish_date` 的多源 fallback + 列表页日期，trafilatura 只能从文章页猜。**结构走 trafilatura，metadata 走现有爬虫。**

### 3.4 为什么 `MarkdownNodeParser` 不能单独用

它只按标题切、**不管尺寸**。两个失败情形：

- 新闻文章常常一个标题都没有 → 整篇变成一个 node
- 某一节特别长 → 单个 node 超出 BGE-m3 的窗口

所以必须两级：结构切分 → 尺寸二次切分。

### 3.5 为什么尺寸层保留手写的 `chunk_text`，不用 `SentenceSplitter`

- **`SentenceSplitter` 默认用 tiktoken（GPT 词表）计长度**，而 embedder 是 BGE-m3（XLM-R 词表，即 `article_chunk.py` 的 `SPARSE_DIM = 250002`）。中文上两者比例差很远，`chunk_size` 口径是错的。用它必须显式替换 tokenizer。
- `chunk_text` 按**字符数**算，不依赖任何词表，天然无此问题。
- 分隔符表是**中英双语**的（`_SEPARATORS`），`SentenceSplitter` 的断句偏英文。
- 少一个需要解释的配置项。

### 3.6 为什么不"跳过抽取，直接把 raw HTML 丢给框架"

这个方案被明确否决，三条理由任一都足够：

1. **样板会变成 chunk。** 同一站点几十篇文章共享同一套 nav，索引里出现几十条近乎重复的 chunk。§1.1 那个 20% 的污染 bug 会扩大到接近全量：稀疏路径被淹没、`RAG_FETCH_K=30` 的候选池被垃圾占满、`context_relevance`（现 0.603）直接塌。而且各站点样板文本不同，dedup 和 RRF 救不了。
2. **成本放大 5~10 倍。** 正文约占页面可见文本的 5~20%、占字节的 1~3%。`page_source` 里还有 inline script、`__NEXT_DATA__` 这类几百 KB 的 JSON blob、SVG path、base64 data URI——不处理就是在给 base64 做 embedding。一旦开始处理，就已经在做抽取了。
3. **HTML 与 PDF 的解析难点不同。** 网页的难点是"这一页里哪 5~20% 是文章"；PDF 的难点是版面与阅读顺序，它没有导航菜单、cookie 弹窗、相关阅读栏，约 95% 是内容。**统一工具等于放弃网页那侧的难点。** 能统一的是输出格式，不是解析器。

---

## 4. 需要改动的位置

### 4.1 数据模型

**新增一列（如 `content_md`），不覆盖 `body_text`。** 理由：

- 摘要器那条路和它的 8000 字截断是调过的，别去动
- 留两列可直接 A/B（老 chunk vs 新 chunk），**不用重爬**

代价：一次 migration + 全量重建索引。

### 4.2 代码触点

| 位置 | 要改什么 |
| --- | --- |
| `fetch/detail.py` | `.get_text()` → trafilatura markdown 输出；现有启发式降为 fallback |
| `fetch/json_api.py` | **第二条内容路径**（走 detail API 取 `data.content`）也要改，否则一半文章有结构一半没有 |
| `rag/chunking.py` | **`_normalize` 在 markdown 路径上必须跳过**——`re.sub(r"(?<!\n)\n(?!\n)", " ", text)` 会把列表塌成一行、表格塌成一行、标题粘上正文。它存在的理由（修 HTML 抽取的假换行）在 markdown 输出下已消失 |
| `repository/article.py` | `embedding_done` 的重置条件绑在 `body_text` 上（`case(Article.body_text != literal(body_text), ...)`），管不到新列；`list_pending_embedding` 同样过滤 `body_text IS NOT NULL` |
| `common/constants.py` | `BODY_CHAR_LIMIT = 8000` 拆开：摘要用截断纯文本，RAG 用完整 markdown |
| `orchestration/pipeline.py` | 质量门从"非零"改成最小正文长度（如 300 字），让抽取失败**可见**而非静默 |
| `rag/indexer.py` | 读 `content_md`；插入两级分块 |

### 4.3 顺带的小问题

- `chunk_text` 的 `_merge` overlap 是盲切字符（`current[-overlap:]`），在 markdown 上可能把半行表格或悬空的 `- ` 复制到下一块开头。不致命，但产生垃圾前缀。
- 表格行理想情况下不该被切开——对 PDF 的财务表格比对新闻更重要。

---

## 5. 执行顺序

**质量门（§4.2 最后一行）可以独立先做。** 它比换库小一个数量级，却能当场挡住那 27 篇里的大部分——不管用哪个抽取器。换库解决"抽得对不对"，质量门解决"抽错了别扩散"，**防线该先有**，因为抽取永远会在新站点上失败。

然后：

1. **HTML 路径改造 + 新列 + migration。** `detail.py` 与 `json_api.py` 一起改，`chunking.py` 跳过 `_normalize`，repo 逻辑跟上，拆开 `BODY_CHAR_LIMIT`。全量重抽取 + 重建索引。
   → 可测收益：hybrid 的 0.919 能否回到 dense 之上。
2. **接 PDF。** `pymupdf4llm`。`fetch/urls.py` 的 `is_document_url` 已经在识别 PDF 链接并标 `external`，路由钩子半个已就位。补 5~10 篇真实 PDF（GRAS notice、专利、年报）进语料。
3. **补难题。** 现有 33 题**在单 gold 题上已饱和**（dense recall 0.985，29/33 满分），且全是短新闻——**换任何分块器都测不出差异**。需要能吃到结构的题：长文档、跨 heading 聚合、表格里的数字。这类题在加了 PDF 之后自然就有了。
4. **分块器 A/B。** 手写 `chunk_text` vs LangChain `RecursiveCharacterTextSplitter` vs LlamaIndex `MarkdownNodeParser` 组合，在新语料上跑 recall@5 / MRR / nDCG。

步骤 1、2 共享一次全量重抽取 + 重建索引，**不要拆成两次跑**。

---

## 6. 风险与遗留

- **全量重建索引的代价。** 当前 `torch` 是 `2.13.0+cpu`，RTX 4060 8GB 闲置。重建前先装 CUDA build，否则重排与编码都在 CPU 上跑。
- **依赖。** `llama-index-core` 自带 pydantic 约束和 `Settings` 全局单例；本项目 `.python-version=3.13` 对 `requires-python>=3.14`，必须 `uv run --no-sync`，环境本就脆弱。若最终只用 `MarkdownNodeParser`（按标题切分），也可评估自行实现以避免整个依赖树。
- **trafilatura 未必全胜。** 那些站点特调的坑（ASP.NET 整页 `<form>`）它靠通用算法绕过而非专门处理，结果可能好也可能只是"不同"。因此 fallback 必须保留，且改造前应先做离线对比（存一批原始 HTML，两种抽取器比正文长度与样板残留），**拿到数据再开 migration**。
- **候选筛选层未处理。** §1.1 提到栏目索引页被当成文章爬进来的问题，属于 `fetch/urls.py` 的 `is_probable_article`，不在本方案范围内，需单独处理。
- **API 名称需核对。** 文中的 `output_format="markdown"`、`to_markdown()`、reader 类名等按实际安装版本核对，这些 API 会变。

---

## 7. 一句话总结

`_find_content_container(soup).get_text()` 里焊了两件事：**去样板**（正确）和**拍平结构**（要改）。拆开之后，去样板换成 trafilatura（更准、免维护选择器表），保结构改为输出 Markdown；PDF 用 pymupdf4llm 归一到同一格式；下游一套两级分块（结构 → 尺寸）。检索栈一行不动。
