from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft7Validator, ValidationError

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
PATH_PARAM_RE = re.compile(r"{([^}]+)}")
DOMAIN_ALIASES = {
    "analytics": "Analytics",
    "client": "Clientes",
    "clients": "Clientes",
    "clientes": "Clientes",
    "coupon": "Cupons",
    "coupons": "Cupons",
    "cupons": "Cupons",
    "delivery": "Delivery",
    "ifood": "iFood",
    "order": "Pedidos",
    "orders": "Pedidos",
    "pedido": "Pedidos",
    "pedidos": "Pedidos",
    "promotion": "Promoções",
    "promotions": "Promoções",
    "promocao": "Promoções",
    "promocoes": "Promoções",
    "reserva": "Reservas",
    "reservas": "Reservas",
    "reservation": "Reservas",
    "reservations": "Reservas",
    "store": "Loja",
    "loja": "Loja",
    "system": "Sistema",
    "sistema": "Sistema",
}


@dataclass(frozen=True)
class OperationParameter:
    name: str
    location: str
    required: bool
    schema: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True)
class Operation:
    operation_id: str
    method: str
    path: str
    summary: str
    tags: tuple[str, ...]
    destructive: bool
    parameters: tuple[OperationParameter, ...] = ()
    request_schema: dict[str, Any] | None = None

    @property
    def domain(self) -> str:
        return self.tags[0] if self.tags else "Sistema"

    @property
    def is_mutation(self) -> bool:
        return self.method in {"POST", "PUT", "PATCH", "DELETE"}


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str


class OperationCatalog:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.operations = self._parse_operations(spec)
        self._by_id = {op.operation_id: op for op in self.operations}

    @classmethod
    async def from_url(cls, url: str, timeout_seconds: float = 12.0) -> "OperationCatalog":
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()
        return cls(response.json())

    @classmethod
    def from_file(cls, path: str | Path) -> "OperationCatalog":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_file(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.spec, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, operation_id: str) -> Operation:
        try:
            return self._by_id[operation_id]
        except KeyError as exc:
            raise KeyError(f"Unknown operation_id: {operation_id}") from exc

    def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        destructive: bool | None = None,
        limit: int = 8,
    ) -> list[Operation]:
        normalized_query = query.lower().strip()
        terms = [term for term in normalized_query.replace("/", " ").replace(".", " ").split() if term]
        scored: list[tuple[int, Operation]] = []

        for op in self.operations:
            if domain and not domain_matches(op.domain, domain, op.operation_id):
                continue
            if destructive is not None and op.destructive != destructive:
                continue

            haystack = " ".join(
                [
                    op.operation_id,
                    op.method,
                    op.path,
                    op.summary,
                    op.domain,
                    *op.tags,
                    self._schema_search_text(op.request_schema),
                ]
            ).lower()
            score = 0
            if not terms:
                score = 1
            for term in terms:
                if term in haystack:
                    score += 3 if term in op.operation_id.lower() else 1
            if score:
                scored.append((score, op))

        scored.sort(key=lambda item: (-item[0], item[1].operation_id))
        return [op for _, op in scored[:limit]]

    def validate_call(
        self,
        operation_id: str,
        *,
        path_params: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        body: Any = None,
    ) -> list[ValidationIssue]:
        op = self.get(operation_id)
        issues: list[ValidationIssue] = []
        path_params = path_params or {}
        query = query or {}

        allowed_query = {p.name for p in op.parameters if p.location == "query"}
        allowed_path = {p.name for p in op.parameters if p.location == "path"}

        for parameter in op.parameters:
            source = path_params if parameter.location == "path" else query
            if parameter.required and parameter.name not in source:
                issues.append(
                    ValidationIssue(
                        field=f"{parameter.location}.{parameter.name}",
                        message="Required parameter is missing.",
                    )
                )

        for name in path_params:
            if name not in allowed_path:
                issues.append(ValidationIssue(field=f"path.{name}", message="Unknown path parameter."))

        for name in query:
            if name not in allowed_query:
                issues.append(ValidationIssue(field=f"query.{name}", message="Unknown query parameter."))

        if op.request_schema is not None:
            if body is None:
                issues.append(ValidationIssue(field="body", message="Request body is required."))
            else:
                try:
                    Draft7Validator(self._resolve_refs(op.request_schema)).validate(body)
                except ValidationError as exc:
                    field = ".".join(str(part) for part in exc.absolute_path) or "body"
                    issues.append(ValidationIssue(field=field, message=exc.message))
        elif body not in (None, {}) and not op.is_mutation:
            issues.append(ValidationIssue(field="body", message="This operation does not accept a body."))

        return issues

    def _parse_operations(self, spec: dict[str, Any]) -> list[Operation]:
        operations: list[Operation] = []
        for path, path_item in spec.get("paths", {}).items():
            for method, definition in path_item.items():
                if method.lower() not in HTTP_METHODS:
                    continue

                parameters = [
                    OperationParameter(
                        name=item["name"],
                        location=item["in"],
                        required=bool(item.get("required", False)),
                        schema=item.get("schema", {}),
                        description=item.get("description"),
                    )
                    for item in definition.get("parameters", [])
                ]
                documented_path_params = {parameter.name for parameter in parameters if parameter.location == "path"}
                for path_param in PATH_PARAM_RE.findall(path):
                    if path_param not in documented_path_params:
                        parameters.append(
                            OperationParameter(
                                name=path_param,
                                location="path",
                                required=True,
                                schema={"type": "string"},
                                description="Inferred from OpenAPI path template.",
                            )
                        )
                request_schema = self._extract_request_schema(definition)
                operations.append(
                    Operation(
                        operation_id=definition.get("operationId") or f"{method}.{path}",
                        method=method.upper(),
                        path=path,
                        summary=definition.get("summary", ""),
                        tags=tuple(definition.get("tags", [])),
                        destructive=bool(definition.get("x-destructive", False)),
                        parameters=tuple(parameters),
                        request_schema=request_schema,
                    )
                )
        return operations

    @staticmethod
    def _extract_request_schema(definition: dict[str, Any]) -> dict[str, Any] | None:
        content = definition.get("requestBody", {}).get("content", {})
        json_content = content.get("application/json")
        if not json_content:
            return None
        return json_content.get("schema")

    def _resolve_refs(self, value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                resolved = self._resolve_json_pointer(value["$ref"])
                merged = {key: val for key, val in value.items() if key != "$ref"}
                if merged and isinstance(resolved, dict):
                    resolved = {**resolved, **merged}
                return self._resolve_refs(resolved)
            return {key: self._resolve_refs(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._resolve_refs(item) for item in value]
        return value

    def _resolve_json_pointer(self, ref: str) -> Any:
        if not ref.startswith("#/"):
            raise ValueError(f"Only local OpenAPI refs are supported: {ref}")
        node: Any = self.spec
        for raw_part in ref[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            node = node[part]
        return node

    def resolved_request_schema(self, operation_id: str) -> dict[str, Any] | None:
        schema = self.get(operation_id).request_schema
        if schema is None:
            return None
        return self._resolve_refs(schema)

    def _schema_search_text(self, schema: dict[str, Any] | None) -> str:
        if schema is None:
            return ""
        resolved = self._resolve_refs(schema)
        parts: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"properties", "required"}:
                        parts.append(str(item))
                    elif key in {"description", "enum", "type"}:
                        parts.append(str(item))
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(resolved)
        return " ".join(parts)


def canonical_domain(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_token(value)
    if normalized in {"", "none", "null", "undefined"}:
        return None
    return DOMAIN_ALIASES.get(normalized, value.strip())


def domain_matches(actual_domain: str, requested_domain: str | None, operation_id: str | None = None) -> bool:
    canonical = canonical_domain(requested_domain)
    if canonical is None:
        return True
    actual_normalized = normalize_token(actual_domain)
    requested_normalized = normalize_token(canonical)
    if actual_normalized == requested_normalized:
        return True
    if operation_id and "." in operation_id:
        return normalize_token(operation_id.split(".", 1)[0]) == normalize_token(requested_domain)
    return False


def normalize_token(value: str) -> str:
    without_accents = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value).strip())
        if not unicodedata.combining(char)
    )
    return without_accents.lower()
