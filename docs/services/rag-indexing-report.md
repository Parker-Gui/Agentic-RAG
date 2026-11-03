# RAG 向量化入库报告

日期：2026-08-28 · 执行入口：`uv run python -m app.services.rag`

## 结果

| 项 | 值 |
|---|---|
| 处理文章 | 59 / 59（`embedding_done=true`，无残留 pending） |
| 生成块 | 949（平均 405 字符） |
| dense 向量 | 949 条，1024 维（BGE-m3） |
| sparse 向量 | 949 条（BGE-m3 lexical，250002 维 sparsevec） |
| group_type | competitor 949 |
| 耗时 | 约 8 分钟（CPU，torch 2.13.0+cpu，无 CUDA） |

参数：`RAG_CHUNK_SIZE=500`，`RAG_CHUNK_OVERLAP=100`，`RAG_INDEX_BATCH=100`，模型来自本地 `models/bge-m3`。

## 过程记录

1. 首次运行失败：`UndefinedColumn: column "sparse_embedding" of relation "article_chunks" does not exist`。
   原因是迁移 `d4e5f6a7b8c9_add_sparse_embedding` 未应用（DB 停在 `c3d4e5f6a7b8`）。
   模型加载与编码本身正常，失败发生在写库，事务回滚，无脏数据。
2. `uv run alembic upgrade head` 应用迁移（加列 + 稀疏 HNSW 索引 `sparsevec_ip_ops`）。
3. 重跑索引，59 篇全部成功，退出码 0。

## 检索冒烟测试

`RagService.query`（当前 `RAG_HYBRID_ENABLED=False`、`RAG_RERANK_ENABLED=False`，走纯 dense）：

- `plant-based protein innovation` → 3 命中，top 相似度 0.5750（IFF 系列文章）
- `sugar reduction sweetener` → 3 命中，top 相似度 0.6495（MycoTechnology / 甜味剂类文章）

出处字段（title / url / site_name / publish_date）均正常返回。

## 待办

- sparse 列已回填，可以把 `RAG_HYBRID_ENABLED` / `RAG_RERANK_ENABLED` 置 `True` 验证 hybrid + rerank 链路。
- 新文章入库后需再次触发本 worker（cron 或手动），当前无自动调度。
