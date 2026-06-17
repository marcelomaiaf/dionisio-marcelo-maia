from __future__ import annotations

EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_PROPERTY = "embedding"
EMBEDDING_TEXT_PROPERTY = "embedding_text"
KNOWLEDGE_NODE_LABEL = "KnowledgeNode"
VECTOR_INDEX_NAME = "dionisio_knowledge_embedding"

SEARCHABLE_NODE_LABELS = {
    "Domain",
    "Entity",
    "Field",
    "Operation",
    "Parameter",
    "RiskPolicy",
    "Workflow",
    "WorkflowStep",
}
