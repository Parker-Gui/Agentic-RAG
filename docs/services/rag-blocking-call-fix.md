# RAG 工具在 langgraph dev 下的阻塞调用排查

日期：2026-08-28 · 触发入口：agent 调用 `search_rag_tool` · 修复提交：`fix(rag): 模型加载移出事件循环`

## 现象

`langgraph dev` 里 agent 每次走到内部知识库检索都失败，工具兜底返回 0 条结果：

```
blockbuster.blockbuster.BlockingError: Blocking call to os.getcwd
[warning] [search_rag] 检索失败 query='甜味剂 主要公司 发明 改造': Blocking call to os.getcwd
```

栈顶是 `tempfile.gettempdir()` → `os.getcwd()`，中间经过 `dill/session.py`、
`torch/utils/data/datapipes/`。检索链路本身（DB、向量、SQL）一行都没跑到。

## 背景：blockbuster 在管什么

事件循环是单线程的：跑在它上面的任何同步慢活儿，都会让同一进程里所有请求一起等
（健康检查、其他 run 全部受影响）。`langgraph dev` 因此挂了
[blockbuster](https://pypi.org/project/blockbuster/)，给 `os.getcwd` / `io.*.read` /
`os.read` 这类底层调用打桩，一旦发现它们**在事件循环线程上**被调用就抛
`BlockingError`。

关键推论：**报警点不是罪魁**。`os.getcwd()` 本身是纳秒级的，它只是那条阻塞链上第一个
被打了桩的函数。排查要顺着栈往上找是谁触发了这条链——这里是 `import torch`。

## 原因

三处阻塞调用都发生在事件循环线程上，且是层层叠住的：修掉一处才暴露下一处。

### 1. 模型加载藏在构造函数里

```python
class RagService:
    def __init__(self, db):
        self.embedder = get_embedder()   # ← 同步加载 2.2G 权重
```

`RagService(db)` 看起来是廉价构造，实际会 `import FlagEmbedding` → `import torch` →
`dill` → `tempfile.gettempdir()` → `os.getcwd()`。而调用点在协程里，既没 `await`
也没 `to_thread`：

```python
async with get_worker_db_context() as db:
    hits = await RagService(db).query(...)
```

`facade.query` 里的 `to_thread(self.embedder.encode, ...)` 只包住了**编码**，**加载**
一直裸在循环上——库里向量早就建好了，所以这个 bug 只在首次调用时炸，且炸在检索之前。

### 2. `to_thread` 只包了一半

```python
cand = await asyncio.to_thread(get_reranker().rerank, question, cand, top_k)
#                              ^^^^^^^^^^^^^^^ 实参先求值，仍在循环上
```

Python 要先算完实参才能调 `to_thread`。于是 `get_reranker()`（加载权重 + import torch）
在循环上跑完，进线程池的只有已经加载好的 `.rerank`。

### 3. 函数内懒 import 也是磁盘 I/O

```python
async def search_rag_tool(...):
    from app.db.session import get_worker_db_context   # 首次执行 = import psycopg
```

import 要去磁盘找 psycopg 的 C 扩展并读 `.so`，同样被判阻塞。更麻烦的是 psycopg 内部
用 `try/except` 逐个尝试三种实现，把 `BlockingError` 吞掉后伪装成依赖缺失：

```
ImportError: no pq wrapper available.
- couldn't import psycopg 'binary' implementation: Blocking call to io.TextIOWrapper.read
- couldn't import psycopg 'python' implementation: Blocking call to os.read
```

只看第一行会以为环境没装好，实际 psycopg 装得好好的。

## 解决方案

原则：**阻塞的不只是"用模型"，还有"拿到模型"**；协程里一律走异步 getter，import 一律
放模块级（模块在建图时导入，那时还没进事件循环，读磁盘没人管）。

| 位置 | 改动 |
|---|---|
| `services/rag/embedding.py` | 拆 `_load_embedder`(lru_cache) + `get_embedder()` + 新增 `get_embedder_async()` |
| `services/rag/rerank.py` | 同上，新增 `get_reranker_async()` |
| `services/rag/facade.py` | 构造函数不再碰模型；`query` / `_hybrid_retrieve` 里 `await` 异步 getter |
| `agents/agent_tools/rag_tool.py` | 函数内懒 import 提到模块级 |
| `main.py` | `_preload_rag` 改 `async` + `await`（原来在 lifespan 里同步加载，堵启动的循环） |
| `services/rag/indexer.py` | `get_embedder` / `encode_with_sparse` 进线程池 |

异步 getter 的形状：

```python
async def get_embedder_async() -> Embedder:
    """协程里取嵌入器：把阻塞的首次加载甩到线程池，不占事件循环。"""
    return await asyncio.to_thread(get_embedder)
```

rerank 那行的正确写法——先 `await` 拿实例，再把纯计算丢线程池：

```python
reranker = await get_reranker_async()
cand = await asyncio.to_thread(reranker.rerank, question, cand, top_k)
```

### 顺带补的锁

```python
_load_lock = threading.Lock()

def get_embedder() -> Embedder:
    with _load_lock:
        return _load_embedder()   # lru_cache
```

`lru_cache` 不是原子的。改走 `to_thread` 之后，并发首调会落在**不同线程**上，两边都发现
缓存为空、各加载一份 2.2G 权重——4.4G 内存换一个单例的错觉。加锁让第一个进去装，其余
排队等一次缓存命中。

## 验证

在 blockbuster 上下文里跑完整工具链（模拟 `langgraph dev` 的运行条件）：

```python
import asyncio, blockbuster
from app.agents.agent_tools.rag_tool import search_rag_tool   # 建图时已导入

async def main():
    with blockbuster.blockbuster_ctx():
        print(await search_rag_tool.ainvoke({"query": "甜味剂 主要公司 发明 改造", "top_k": 3}))

asyncio.run(main())
```

- `search_rag_tool` 返回 3 条真实命中，无 `BlockingError`。
- 嵌入器单例复用确认：连续两次 `await get_embedder_async()` 得到同一实例。
- `uv run pytest tests --ignore=tests/agents` → 106 passed。
  （`tests/agents/` 5 个模块收集即失败，引用了已删除的 `app.agents.audit`、`Evidence`、
  `JudgeVerdict`、`MAX_REVISIONS`，是 ReACT 重构的存量问题，本次未动。）

## 沿用规则

- 协程里要用重模型，走 `get_embedder_async()` / `get_reranker_async()`；同步 `get_*()`
  只留给 CLI 和已经在线程池里的代码。
- 服务的 `__init__` 不做 I/O、不加载模型——构造要廉价，慢活儿放显式的 `await`。
- `asyncio.to_thread(f(), ...)` 是错的，`asyncio.to_thread(f, ...)` 才对；实参位置出现
  函数调用就要警觉。
- agent 工具模块里不做函数内懒 import：这些模块在建图时导入，放模块级即可，放函数里等于
  把 import 的磁盘 I/O 推到事件循环上。
- 排查 `BlockingError` 时先看栈的**上游**，别盯着报警的那个纳秒级调用；同时留意被
  `try/except` 吞掉后伪装成 `ImportError` 的情况。
- 临时绕过（仅本地）：`langgraph dev --allow-blocking`。这只关掉检测，不解决卡循环。
