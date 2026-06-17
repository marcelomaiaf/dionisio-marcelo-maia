from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from dionisio_agent.approval import ApprovalStore, call_fingerprint
from dionisio_agent.operation_catalog import Operation, OperationCatalog

logger = logging.getLogger(__name__)


class ApiError(BaseModel):
    code: str
    message: str
    details: Any = None


class ApiExecutionResult(BaseModel):
    ok: bool
    operation_id: str
    method: str | None = None
    path: str | None = None
    status_code: int | None = None
    data: Any = None
    error: ApiError | None = None
    approval_required: bool = False
    destructive: bool = False
    audit_id: str = Field(default_factory=lambda: f"audit_{uuid.uuid4().hex[:12]}")
    latency_seconds: float | None = None


@dataclass(frozen=True)
class RetryPolicy:
    max_get_attempts: int = 3
    backoff_seconds: float = 0.25


class DionisioAPIClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        catalog: OperationCatalog,
        approval_store: ApprovalStore | None = None,
        approval_policy: str = "destructive",
        timeout_seconds: float = 12.0,
        retry_policy: RetryPolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.catalog = catalog
        self.approval_store = approval_store or ApprovalStore()
        self.approval_policy = approval_policy
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self._http_client = http_client

    async def execute(
        self,
        *,
        operation_id: str,
        path_params: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        body: Any = None,
        dry_run: bool = False,
        approval_id: str | None = None,
    ) -> ApiExecutionResult:
        audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        path_params = path_params or {}
        query = query or {}

        try:
            operation = self.catalog.get(operation_id)
        except KeyError as exc:
            return self._error(operation_id, audit_id, "validation_error", str(exc), started)

        issues = self.catalog.validate_call(
            operation_id,
            path_params=path_params,
            query=query,
            body=body,
        )
        if issues:
            return self._error(
                operation_id,
                audit_id,
                "validation_error",
                "The operation call does not match the OpenAPI contract.",
                started,
                details=[issue.__dict__ for issue in issues],
                operation=operation,
            )

        url_path = self._render_path(operation.path, path_params)
        fingerprint = call_fingerprint(operation_id, {"path_params": path_params, "query": query}, body)

        if dry_run:
            return ApiExecutionResult(
                ok=True,
                operation_id=operation_id,
                method=operation.method,
                path=url_path,
                data={
                    "dry_run": True,
                    "operation_id": operation_id,
                    "method": operation.method,
                    "path": url_path,
                    "query": query,
                    "body": body,
                    "destructive": operation.destructive,
                    "requires_approval": self._requires_approval(operation),
                    "fingerprint": fingerprint,
                },
                destructive=operation.destructive,
                audit_id=audit_id,
                latency_seconds=round(time.perf_counter() - started, 4),
            )

        if self._requires_approval(operation) and not self.approval_store.is_approved(
            approval_id, fingerprint
        ):
            return ApiExecutionResult(
                ok=False,
                operation_id=operation_id,
                method=operation.method,
                path=url_path,
                error=ApiError(
                    code="approval_required",
                    message="Human approval is required before executing this operation.",
                    details={
                        "operation_id": operation_id,
                        "destructive": operation.destructive,
                        "approval_policy": self.approval_policy,
                        "fingerprint": fingerprint,
                    },
                ),
                approval_required=True,
                destructive=operation.destructive,
                audit_id=audit_id,
                latency_seconds=round(time.perf_counter() - started, 4),
            )

        result = await self._send(operation, url_path, query, body, audit_id, started)
        self._audit(result)
        return result

    async def _send(
        self,
        operation: Operation,
        path: str,
        query: dict[str, Any],
        body: Any,
        audit_id: str,
        started: float,
    ) -> ApiExecutionResult:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        attempts = self.retry_policy.max_get_attempts if operation.method == "GET" else 1

        client = self._http_client or httpx.AsyncClient(timeout=self.timeout_seconds)
        close_client = self._http_client is None
        try:
            for attempt in range(1, attempts + 1):
                try:
                    response = await client.request(
                        operation.method,
                        url,
                        params=query or None,
                        json=body if body is not None else None,
                        headers=headers,
                    )
                    data = self._decode_response(response)
                    if response.status_code < 400:
                        return ApiExecutionResult(
                            ok=True,
                            operation_id=operation.operation_id,
                            method=operation.method,
                            path=path,
                            status_code=response.status_code,
                            data=data,
                            destructive=operation.destructive,
                            audit_id=audit_id,
                            latency_seconds=round(time.perf_counter() - started, 4),
                        )
                    if operation.method == "GET" and response.status_code in {408, 429, 500, 502, 503, 504}:
                        if attempt < attempts:
                            await self._sleep_backoff(attempt)
                            continue
                    return self._error(
                        operation.operation_id,
                        audit_id,
                        "http_error",
                        f"Dionisio API returned HTTP {response.status_code}.",
                        started,
                        details=data,
                        operation=operation,
                        status_code=response.status_code,
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if operation.method == "GET" and attempt < attempts:
                        await self._sleep_backoff(attempt)
                        continue
                    return self._error(
                        operation.operation_id,
                        audit_id,
                        "network_error",
                        str(exc),
                        started,
                        operation=operation,
                    )
        finally:
            if close_client:
                await client.aclose()

        return self._error(operation.operation_id, audit_id, "unexpected_error", "No response.", started)

    async def _sleep_backoff(self, attempt: int) -> None:
        import asyncio

        await asyncio.sleep(self.retry_policy.backoff_seconds * attempt)

    def _requires_approval(self, operation: Operation) -> bool:
        if operation.destructive:
            return True
        if self.approval_policy == "mutations" and operation.is_mutation:
            return True
        return False

    @staticmethod
    def _render_path(path_template: str, path_params: dict[str, Any]) -> str:
        path = path_template
        for name, value in path_params.items():
            path = path.replace("{" + name + "}", quote(str(value), safe=""))
        return path

    @staticmethod
    def _decode_response(response: httpx.Response) -> Any:
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    def _error(
        self,
        operation_id: str,
        audit_id: str,
        code: str,
        message: str,
        started: float,
        *,
        details: Any = None,
        operation: Operation | None = None,
        status_code: int | None = None,
    ) -> ApiExecutionResult:
        result = ApiExecutionResult(
            ok=False,
            operation_id=operation_id,
            method=operation.method if operation else None,
            path=operation.path if operation else None,
            status_code=status_code,
            error=ApiError(code=code, message=message, details=details),
            destructive=operation.destructive if operation else False,
            audit_id=audit_id,
            latency_seconds=round(time.perf_counter() - started, 4),
        )
        self._audit(result)
        return result

    @staticmethod
    def _audit(result: ApiExecutionResult) -> None:
        logger.info(
            "dionisio_api_call",
            extra={
                "audit_id": result.audit_id,
                "operation_id": result.operation_id,
                "method": result.method,
                "path": result.path,
                "status_code": result.status_code,
                "ok": result.ok,
                "destructive": result.destructive,
                "latency_seconds": result.latency_seconds,
                "error_code": result.error.code if result.error else None,
            },
        )

