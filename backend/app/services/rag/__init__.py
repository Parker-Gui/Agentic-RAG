"""RAG 服务包。

厚领域子包：对外只暴露门面 RagService，嵌入 / 向量库 / 切块等子模块均为包内实现，
外部（路由、worker）不应直接导入。
"""

from app.services.rag.facade import RagService
from app.services.rag.ingestion import KnowledgeService
from app.services.rag.parsers import MAX_FILE_BYTES as MAX_DOCUMENT_BYTES
from app.services.rag.parsers import parse_html as parse_web_document

__all__ = ["MAX_DOCUMENT_BYTES", "KnowledgeService", "RagService", "parse_web_document"]
