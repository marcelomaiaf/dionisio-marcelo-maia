from __future__ import annotations

from typing import Any

from dionisio_agent.config import Settings
from dionisio_agent.graphrag.constants import EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL


def create_openai_embedder(settings: Settings) -> Any:
    """Create a neo4j-graphrag-compatible embedder.

    OpenRouter is the default provider because this project already uses
    OpenRouter for model access. Direct OpenAI remains a fallback for local
    experiments where only OPENAI_API_KEY is configured.
    """
    try:
        from neo4j_graphrag.embeddings import OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on optional package install
        raise RuntimeError(
            "neo4j-graphrag and openai are required for GraphRAG embeddings."
        ) from exc

    return OpenAIEmbeddings(**embedding_client_kwargs(settings))


def embedding_client_kwargs(settings: Settings) -> dict[str, Any]:
    if settings.openrouter_api_key:
        headers: dict[str, str] = {}
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_title:
            headers["X-OpenRouter-Title"] = settings.openrouter_app_title
        kwargs: dict[str, Any] = {
            "model": EMBEDDING_MODEL,
            "api_key": settings.require_openrouter_key(),
            "base_url": settings.openrouter_base_url,
        }
        if headers:
            kwargs["default_headers"] = headers
        return kwargs

    if settings.openai_api_key:
        return {
            "model": OPENAI_EMBEDDING_MODEL,
            "api_key": settings.openai_api_key,
        }

    raise RuntimeError(
        "OPENROUTER_API_KEY is required to create or query GraphRAG embeddings. "
        "OPENAI_API_KEY is supported only as a fallback."
    )
