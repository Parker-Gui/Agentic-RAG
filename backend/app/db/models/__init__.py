"""SQLAlchemy models.

Importing this package registers every mapper, so ``Base.metadata`` is complete
for Alembic and string-based relationships (``User.sessions``) can resolve.
"""

from app.db.base import Base
from app.db.models.article import Article
from app.db.models.article_chunk import EMBED_DIM, ArticleChunk
from app.db.models.artifact import Artifact
from app.db.models.conversation import Conversation, ConversationTurn
from app.db.models.knowledge import KnowledgeChunk, KnowledgeDocument, KnowledgeRevision
from app.db.models.session import Session
from app.db.models.user import User, UserRole

__all__ = [
    "EMBED_DIM",
    "Article",
    "ArticleChunk",
    "Artifact",
    "Base",
    "Conversation",
    "ConversationTurn",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "KnowledgeRevision",
    "Session",
    "User",
    "UserRole",
]
