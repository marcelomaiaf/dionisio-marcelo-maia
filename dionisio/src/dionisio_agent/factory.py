from __future__ import annotations

from dionisio_agent.agent import create_revops_agent
from dionisio_agent.api_client import DionisioAPIClient
from dionisio_agent.approval import ApprovalStore
from dionisio_agent.config import Settings
from dionisio_agent.graphrag.retriever import Neo4jKnowledgeBase
from dionisio_agent.operation_catalog import OperationCatalog
from dionisio_agent.tools import ToolRuntime, create_tools


async def create_runtime(settings: Settings):
    catalog = await OperationCatalog.from_url(
        settings.dionisio_openapi_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    approval_store = ApprovalStore()
    api_client = DionisioAPIClient(
        api_key=settings.require_dionisio_key(),
        base_url=settings.dionisio_api_base_url,
        catalog=catalog,
        approval_store=approval_store,
        approval_policy=settings.approval_policy,
        timeout_seconds=settings.request_timeout_seconds,
    )
    runtime = ToolRuntime(
        catalog=catalog,
        api_client=api_client,
        approval_store=approval_store,
        interactive_approval=settings.interactive_approval,
        knowledge_base=Neo4jKnowledgeBase(settings) if settings.graphrag_enabled else None,
    )
    tools = create_tools(runtime)
    agent = create_revops_agent(settings, tools)
    return agent, runtime
