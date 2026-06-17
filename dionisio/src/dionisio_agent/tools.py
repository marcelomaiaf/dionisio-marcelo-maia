from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents import function_tool

from dionisio_agent.api_client import DionisioAPIClient
from dionisio_agent.approval import ApprovalStore, call_fingerprint
from dionisio_agent.graphrag.retriever import Neo4jKnowledgeBase
from dionisio_agent.operation_catalog import OperationCatalog, canonical_domain


@dataclass
class ToolRuntime:
    catalog: OperationCatalog
    api_client: DionisioAPIClient
    approval_store: ApprovalStore
    interactive_approval: bool = False
    knowledge_base: Neo4jKnowledgeBase | None = None
    discovered_operation_ids: set[str] = field(default_factory=set)


def create_tools(runtime: ToolRuntime) -> list[Any]:
    @function_tool
    async def search_api_operations(
        query: str,
        domain: str | None = None,
        destructive: bool | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Search the Dionisio OpenAPI catalog for operations relevant to a user request.

        Use this after search_api_knowledge when the user asks for a Dionisio
        API action or lookup. Prefer leaving domain and destructive unset for
        discovery workflows, because reservation tasks often require operations
        from both Clientes and Reservas. If domain is used, pass an exact
        catalog domain or a known alias such as reservations, clients, orders,
        coupons, promotions, delivery, store, analytics, ifood, or system.
        """
        normalized_domain = canonical_domain(domain)
        matches = runtime.catalog.search(
            query,
            domain=normalized_domain,
            destructive=destructive,
            limit=max(1, min(limit, 20)),
        )
        runtime.discovered_operation_ids.update(op.operation_id for op in matches)
        return {
            "applied_filters": {
                "domain": normalized_domain,
                "destructive": destructive,
            },
            "items": [
                {
                    "operation_id": op.operation_id,
                    "method": op.method,
                    "path": op.path,
                    "domain": op.domain,
                    "summary": op.summary,
                    "destructive": op.destructive,
                    "requires_body": op.request_schema is not None,
                    "request_schema": runtime.catalog.resolved_request_schema(op.operation_id),
                    "parameters": [
                        {
                            "name": parameter.name,
                            "location": parameter.location,
                            "required": parameter.required,
                            "description": parameter.description,
                        }
                        for parameter in op.parameters
                    ],
                }
                for op in matches
            ]
        }

    @function_tool
    async def call_dionisio_operation(
        operation_id: str,
        path_params_json: str = "{}",
        query_json: str = "{}",
        body_json: str = "null",
        dry_run: bool = False,
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute an allowlisted Dionisio API operation by operation_id after validation.

        Args:
            operation_id: Exact OpenAPI operationId, for example "reservations.list".
            path_params_json: JSON object string for path params, for example {"id":"res_123"}.
            query_json: JSON object string for query params, for example {"date":"2026-06-14"}.
            body_json: JSON object string for request body, or "null" when there is no body.
            dry_run: When true, validates and previews the call without sending it.
            approval_id: Human approval id required for risky operations.
        """
        parsed = _parse_tool_json(path_params_json, query_json, body_json)
        if "error" in parsed:
            return parsed
        discovery_error = _operation_discovery_error(runtime, operation_id)
        if discovery_error:
            return discovery_error
        result = await runtime.api_client.execute(
            operation_id=operation_id,
            path_params=parsed["path_params"],
            query=parsed["query"],
            body=parsed["body"],
            dry_run=dry_run,
            approval_id=approval_id,
        )
        return result.model_dump(mode="json")

    @function_tool
    async def approve_human_approval(approval_id: str) -> dict[str, Any]:
        """Mark a pending approval as approved after the human user explicitly confirms it.

        Use this only when the user clearly says they approve the pending action.
        Args:
            approval_id: The exact approval id previously returned by request_human_approval.
        """
        approval = runtime.approval_store.get(approval_id)
        if not approval:
            return {
                "ok": False,
                "error": {
                    "code": "approval_not_found",
                    "message": "No pending approval exists with this approval_id in the current runtime.",
                },
            }
        approved = runtime.approval_store.approve(approval_id)
        return {
            "ok": True,
            "approval_id": approved.approval_id,
            "approved": approved.approved,
            "operation_id": approved.operation_id,
            "risk_level": approved.risk_level,
            "action_summary": approved.action_summary,
            "message": "Approval recorded. The agent may now execute the exact approved operation with this approval_id.",
        }

    @function_tool
    async def request_human_approval(
        operation_id: str,
        action_summary: str,
        path_params_json: str = "{}",
        query_json: str = "{}",
        body_json: str = "null",
        risk_level: str = "high",
    ) -> dict[str, Any]:
        """Create a human approval request for a risky Dionisio API operation.

        Args:
            operation_id: Exact OpenAPI operationId that will be executed after approval.
            action_summary: Human-readable summary of the intended operation and risk.
            path_params_json: JSON object string for path params.
            query_json: JSON object string for query params.
            body_json: JSON object string for request body, or "null" when there is no body.
            risk_level: Risk label such as "medium", "high", or "critical".
        """
        parsed = _parse_tool_json(path_params_json, query_json, body_json)
        if "error" in parsed:
            return parsed
        discovery_error = _operation_discovery_error(runtime, operation_id)
        if discovery_error:
            return discovery_error
        fingerprint = call_fingerprint(
            operation_id,
            {"path_params": parsed["path_params"], "query": parsed["query"]},
            parsed["body"],
        )
        approval = runtime.approval_store.request(
            fingerprint=fingerprint,
            action_summary=action_summary,
            operation_id=operation_id,
            risk_level=risk_level,
        )

        if runtime.interactive_approval:
            answer = input(
                f"Approve {operation_id}? Type the approval id {approval.approval_id} to confirm: "
            ).strip()
            if answer == approval.approval_id:
                runtime.approval_store.approve(approval.approval_id)

        return {
            "approval_id": approval.approval_id,
            "approved": runtime.approval_store.is_approved(approval.approval_id, fingerprint),
            "operation_id": operation_id,
            "risk_level": risk_level,
            "action_summary": action_summary,
            "message": "Use this approval_id in call_dionisio_operation only after human approval.",
        }

    tools = [
        search_api_operations,
        call_dionisio_operation,
        request_human_approval,
        approve_human_approval,
    ]

    if runtime.knowledge_base is not None:
        @function_tool
        async def search_api_knowledge(
            query: str,
            domain: str | None = None,
            operation_id: str | None = None,
            limit: int = 5,
        ) -> dict[str, Any]:
            """Search the Neo4j GraphRAG knowledge base for Dionisio API concepts.

            Call this before search_api_operations for Dionisio API requests. Use
            it to understand semantic API context, ordered workflows, related
            operations, required entities, fields, parameters, and risk policies.
            This tool is read-only and best-effort; it never executes Dionisio
            API operations.
            """
            normalized_operation_id = _none_if_nullish(operation_id)
            return runtime.knowledge_base.search(
                query=query,
                domain=canonical_domain(domain),
                operation_id=normalized_operation_id,
                limit=limit,
            )

        tools.append(search_api_knowledge)

    return tools


def _parse_tool_json(path_params_json: str, query_json: str, body_json: str) -> dict[str, Any]:
    try:
        path_params = json.loads(path_params_json or "{}")
        query = json.loads(query_json or "{}")
        body = json.loads(body_json or "null")
    except json.JSONDecodeError as exc:
        return {
            "error": {
                "code": "invalid_json",
                "message": f"Tool JSON argument is invalid: {exc.msg}",
            }
        }

    if not isinstance(path_params, dict):
        return {
            "error": {
                "code": "invalid_path_params_json",
                "message": "path_params_json must decode to a JSON object.",
            }
        }
    if not isinstance(query, dict):
        return {
            "error": {
                "code": "invalid_query_json",
                "message": "query_json must decode to a JSON object.",
            }
        }
    if body is not None and not isinstance(body, dict):
        return {
            "error": {
                "code": "invalid_body_json",
                "message": "body_json must decode to a JSON object or null.",
            }
        }

    return {"path_params": path_params, "query": query, "body": body}


def _none_if_nullish(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null", "undefined"}:
        return None
    return value


def _operation_discovery_error(runtime: ToolRuntime, operation_id: str) -> dict[str, Any] | None:
    if operation_id in runtime.discovered_operation_ids:
        return None
    try:
        runtime.catalog.get(operation_id)
    except KeyError:
        return {
            "ok": False,
            "error": {
                "code": "unknown_operation_id",
                "message": f"Unknown operation_id: {operation_id}",
            },
        }
    return {
        "ok": False,
        "error": {
            "code": "operation_not_discovered",
            "message": (
                "Call search_api_operations for this user request before executing or requesting "
                "approval for this operation_id."
            ),
            "operation_id": operation_id,
        },
    }
