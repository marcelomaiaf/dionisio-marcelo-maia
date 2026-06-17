from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = None

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    openrouter_site_url: str | None = None
    openrouter_app_title: str = "Dionisio RevOps Agent"
    openai_agents_tracing: bool = False

    dionisio_api_key: str | None = None
    dionisio_api_base_url: str = "https://dionisio-crm.web.app"
    dionisio_openapi_url: str = "https://app.odionisio.com/api/case-mock/docs.json"

    neo4j_uri: str | None = None
    neo4j_username: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    graphrag_enabled: bool = False

    request_timeout_seconds: float = 12.0
    approval_policy: str = "destructive"
    interactive_approval: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_if_present()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", cls.openrouter_base_url),
            openrouter_model=os.getenv("OPENROUTER_MODEL", cls.openrouter_model),
            openrouter_site_url=os.getenv("OPENROUTER_SITE_URL"),
            openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE", cls.openrouter_app_title),
            openai_agents_tracing=os.getenv("OPENAI_AGENTS_TRACING", "false").lower()
            in {"1", "true", "yes", "y"},
            dionisio_api_key=os.getenv("DIONISIO_API_KEY"),
            dionisio_api_base_url=os.getenv("DIONISIO_API_BASE_URL", cls.dionisio_api_base_url),
            dionisio_openapi_url=os.getenv("DIONISIO_OPENAPI_URL", cls.dionisio_openapi_url),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username=os.getenv("NEO4J_USERNAME", cls.neo4j_username),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", cls.neo4j_database),
            graphrag_enabled=os.getenv("DIONISIO_GRAPHRAG_ENABLED", "").lower()
            in {"1", "true", "yes", "y"},
            request_timeout_seconds=float(os.getenv("DIONISIO_TIMEOUT_SECONDS", "12")),
            approval_policy=os.getenv("DIONISIO_APPROVAL_POLICY", "destructive").lower(),
            interactive_approval=os.getenv("DIONISIO_INTERACTIVE_APPROVAL", "").lower()
            in {"1", "true", "yes", "y"},
        )

    def require_openrouter_key(self) -> str:
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to run the agent.")
        return self.openrouter_api_key

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        return self.openai_api_key

    def require_dionisio_key(self) -> str:
        if not self.dionisio_api_key:
            raise RuntimeError("DIONISIO_API_KEY is required to call the Dionisio API.")
        return self.dionisio_api_key

    def require_neo4j_uri(self) -> str:
        if not self.neo4j_uri:
            raise RuntimeError("NEO4J_URI is required to use the GraphRAG knowledge base.")
        return self.neo4j_uri

    def require_neo4j_password(self) -> str:
        if not self.neo4j_password:
            raise RuntimeError("NEO4J_PASSWORD is required to use the GraphRAG knowledge base.")
        return self.neo4j_password


def load_dotenv_if_present(start_dir: str | Path | None = None) -> Path | None:
    """Load a local .env file without overriding real environment variables."""
    current = Path(start_dir or os.getcwd()).resolve()
    candidates = [current, *current.parents]
    for directory in candidates:
        env_path = directory / ".env"
        if env_path.exists():
            _load_dotenv_file(env_path)
            return env_path
    return None


def _load_dotenv_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
