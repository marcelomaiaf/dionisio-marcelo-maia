from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from dionisio_agent.config import Settings
from dionisio_agent.graphrag.constants import (
    EMBEDDING_PROPERTY,
    EMBEDDING_TEXT_PROPERTY,
    VECTOR_INDEX_NAME,
)
from dionisio_agent.graphrag.embeddings import create_openai_embedder
from dionisio_agent.operation_catalog import domain_matches

logger = logging.getLogger(__name__)


@dataclass
class Neo4jKnowledgeBase:
    settings: Settings
    _neo4j_driver: Any = field(default=None, init=False, repr=False)
    _vector_retriever: Any = field(default=None, init=False, repr=False)
    _embedder: Any = field(default=None, init=False, repr=False)
    _resource_lock: Any = field(default_factory=RLock, init=False, repr=False)
    _search_cache: dict[tuple[str, str | None, str | None, int], tuple[float, dict[str, Any]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def search(
        self,
        *,
        query: str,
        domain: str | None = None,
        operation_id: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        try:
            clamped_limit = max(1, min(limit, 20))
            cache_key = _cache_key(
                query=query,
                domain=domain,
                operation_id=operation_id,
                limit=clamped_limit,
            )
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.info(
                    "dionisio_graphrag_search_cache_hit %s",
                    json.dumps(
                        {
                            "query": query,
                            "domain": domain,
                            "operation_id": operation_id,
                            "limit": clamped_limit,
                            "item_count": len(cached.get("items", [])),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    extra={
                        "query": query,
                        "domain": domain,
                        "operation_id": operation_id,
                        "limit": clamped_limit,
                        "item_count": len(cached.get("items", [])),
                    },
                )
                return cached
            logger.info(
                "dionisio_graphrag_search_started %s",
                json.dumps(
                    {
                        "query": query,
                        "domain": domain,
                        "operation_id": operation_id,
                        "limit": clamped_limit,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                extra={
                    "query": query,
                    "domain": domain,
                    "operation_id": operation_id,
                    "limit": clamped_limit,
                },
            )
            keys_by_score = self._semantic_keys(query=query, limit=clamped_limit)
            if not keys_by_score:
                logger.info(
                    "dionisio_graphrag_search_completed %s",
                    json.dumps(
                        {"item_count": 0, "matched_keys": []},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    extra={"item_count": 0, "matched_keys": []},
                )
                result = _empty_result(degraded=False)
                self._set_cached(cache_key, result)
                return result
            items = self._hydrate_items(
                keys_by_score=keys_by_score,
                domain=domain,
                operation_id=operation_id,
            )
            logger.info(
                "dionisio_graphrag_search_completed %s",
                json.dumps(
                    {
                        "item_count": len(items),
                        "matched_keys": list(keys_by_score),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                extra={
                    "item_count": len(items),
                    "matched_keys": list(keys_by_score),
                },
            )
            result = {
                "items": items,
                "meta": {"degraded": False, "error_logged": False, "cache_hit": False},
            }
            self._set_cached(cache_key, result)
            return result
        except Exception as exc:  # pragma: no cover - exercised by integration/fallback tests
            logger.warning(
                "dionisio_graphrag_search_degraded %s",
                json.dumps(
                    {
                        "error_class": exc.__class__.__name__,
                        "error_message": str(exc),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                extra={
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc),
                },
            )
            return _empty_result(degraded=True)

    def _semantic_keys(self, *, query: str, limit: int) -> dict[str, float]:
        try:
            from neo4j_graphrag.retrievers import VectorCypherRetriever
        except ImportError as exc:
            raise RuntimeError(
                "neo4j-graphrag, openai, and neo4j are required for semantic GraphRAG search."
            ) from exc

        retriever = self._get_vector_retriever(VectorCypherRetriever)
        result = retriever.get_search_results(
            query_text=query,
            top_k=limit,
            effective_search_ratio=self.settings.graphrag_effective_search_ratio,
        )
        scores: dict[str, float] = {}
        for record in result.records:
            key = record["key"]
            score = record["score"]
            if key:
                scores[str(key)] = float(score or 0.0)
        return scores

    def _hydrate_items(
        self,
        *,
        keys_by_score: dict[str, float],
        domain: str | None,
        operation_id: str | None,
    ) -> list[dict[str, Any]]:
        keys = list(keys_by_score)
        driver = self._get_driver()
        nodes = _fetch_nodes(driver, self.settings, keys)
        operations = _fetch_related_operations(driver, self.settings, keys)
        workflow_keys_by_operation = _fetch_workflow_keys_for_operations(
            driver,
            self.settings,
            _operation_keys(operations),
        )
        _attach_workflow_keys(operations, workflow_keys_by_operation)
        workflow_keys = _workflow_keys(nodes, operations)
        workflows = _fetch_workflows(driver, self.settings, sorted(workflow_keys))
        related = _fetch_operation_related(driver, self.settings, _operation_keys(operations))

        items: list[dict[str, Any]] = []
        for key in keys:
            node = nodes.get(key)
            if not node:
                continue
            node_operations = operations.get(key, [])
            if operation_id:
                node_operations = [op for op in node_operations if op.get("operation_id") == operation_id]
                if not node_operations and node.get("operation_id") != operation_id:
                    continue
            if domain:
                node_operations = [
                    op
                    for op in node_operations
                    if domain_matches(str(op.get("domain", "")), domain, str(op.get("operation_id", "")))
                ]
                node_matches_domain = domain_matches(str(node.get("domain", "")), domain) or domain_matches(
                    str(node.get("name", "")),
                    domain,
                )
                if not node_operations and not node_matches_domain:
                    continue

            primary_operation = node_operations[0] if node_operations else None
            matched_workflows = [
                workflows[workflow_key]
                for workflow_key in _node_workflow_keys(node, node_operations)
                if workflow_key in workflows
            ]
            op_key = primary_operation.get("operation_id") if primary_operation else None
            items.append(
                {
                    "score": keys_by_score[key],
                    "node": node,
                    "operation": primary_operation,
                    "workflow": matched_workflows[0] if matched_workflows else None,
                    "workflows": matched_workflows,
                    "related": related.get(op_key, _empty_related()) if op_key else _empty_related(),
                }
            )
        return items

    def close(self) -> None:
        with self._resource_lock:
            if self._neo4j_driver is not None:
                self._neo4j_driver.close()
            self._neo4j_driver = None
            self._vector_retriever = None
            self._embedder = None
            self._search_cache.clear()

    def warmup(self) -> None:
        try:
            from neo4j_graphrag.retrievers import VectorCypherRetriever
        except ImportError:
            return
        try:
            started = time.perf_counter()
            self._get_vector_retriever(VectorCypherRetriever)
            logger.info(
                "dionisio_graphrag_warmup_completed",
                extra={"latency_seconds": round(time.perf_counter() - started, 4)},
            )
        except Exception as exc:  # pragma: no cover - best-effort startup optimization
            logger.warning(
                "dionisio_graphrag_warmup_degraded",
                extra={
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc),
                },
            )

    def _get_driver(self) -> Any:
        if self._neo4j_driver is not None:
            return self._neo4j_driver
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j package is required for GraphRAG hydration.") from exc

        with self._resource_lock:
            if self._neo4j_driver is None:
                self._neo4j_driver = GraphDatabase.driver(
                    self.settings.require_neo4j_uri(),
                    auth=(self.settings.neo4j_username, self.settings.require_neo4j_password()),
                    connection_timeout=self.settings.graphrag_neo4j_timeout_seconds,
                    max_connection_lifetime=300,
                )
        return self._neo4j_driver

    def _get_vector_retriever(self, retriever_cls: Any) -> Any:
        if self._vector_retriever is not None:
            return self._vector_retriever

        with self._resource_lock:
            if self._vector_retriever is None:
                if self._embedder is None:
                    self._embedder = create_openai_embedder(self.settings)
                self._vector_retriever = retriever_cls(
                    self._get_driver(),
                    VECTOR_INDEX_NAME,
                    _retrieval_query(),
                    embedder=self._embedder,
                    neo4j_database=self.settings.neo4j_database,
                )
        return self._vector_retriever

    def _get_cached(
        self,
        cache_key: tuple[str, str | None, str | None, int],
    ) -> dict[str, Any] | None:
        ttl = self.settings.graphrag_cache_ttl_seconds
        if ttl <= 0 or self.settings.graphrag_cache_max_entries <= 0:
            return None
        with self._resource_lock:
            cached = self._search_cache.get(cache_key)
            if cached is None:
                return None
            cached_at, result = cached
            if time.monotonic() - cached_at > ttl:
                self._search_cache.pop(cache_key, None)
                return None
            value = copy.deepcopy(result)
        value.setdefault("meta", {})["cache_hit"] = True
        return value

    def _set_cached(
        self,
        cache_key: tuple[str, str | None, str | None, int],
        result: dict[str, Any],
    ) -> None:
        if (
            self.settings.graphrag_cache_ttl_seconds <= 0
            or self.settings.graphrag_cache_max_entries <= 0
        ):
            return
        with self._resource_lock:
            if len(self._search_cache) >= self.settings.graphrag_cache_max_entries:
                oldest_key = min(self._search_cache, key=lambda key: self._search_cache[key][0])
                self._search_cache.pop(oldest_key, None)
            self._search_cache[cache_key] = (time.monotonic(), copy.deepcopy(result))


def _retrieval_query() -> str:
    return f"""
    RETURN node.key AS key,
           labels(node) AS labels,
           node {{.*, {EMBEDDING_PROPERTY}: null, {EMBEDDING_TEXT_PROPERTY}: null}} AS properties,
           score
    """


def _cache_key(
    *,
    query: str,
    domain: str | None,
    operation_id: str | None,
    limit: int,
) -> tuple[str, str | None, str | None, int]:
    normalized_query = " ".join(query.strip().lower().split())
    normalized_domain = domain.strip().lower() if domain else None
    normalized_operation_id = operation_id.strip() if operation_id else None
    return (normalized_query, normalized_domain, normalized_operation_id, limit)


def _fetch_nodes(driver: Any, settings: Settings, keys: list[str]) -> dict[str, dict[str, Any]]:
    records, _, _ = driver.execute_query(
        f"""
        MATCH (n)
        WHERE n.key IN $keys
        RETURN n.key AS key,
               labels(n) AS labels,
               n {{.*, {EMBEDDING_PROPERTY}: null, {EMBEDDING_TEXT_PROPERTY}: null}} AS properties
        """,
        keys=keys,
        database_=settings.neo4j_database,
        timeout=settings.graphrag_neo4j_timeout_seconds,
    )
    return {
        record["key"]: _format_node(record["labels"], record["properties"])
        for record in records
    }


def _fetch_related_operations(
    driver: Any,
    settings: Settings,
    keys: list[str],
) -> dict[str, list[dict[str, Any]]]:
    records, _, _ = driver.execute_query(
        """
        MATCH (n)
        WHERE n.key IN $keys
        OPTIONAL MATCH (op_self:Operation)
        WHERE op_self.key = n.key
        OPTIONAL MATCH (n)-[:HAS_STEP]->(op_workflow:Operation)
        OPTIONAL MATCH (n)-[:USES_OPERATION]->(op_step:Operation)
        OPTIONAL MATCH (op_ref:Operation)-[
            :REQUIRES_FIELD|ACCEPTS_FIELD|RETURNS_FIELD|
            REQUIRES_PARAMETER|ACCEPTS_PARAMETER|HAS_RISK_POLICY|
            REQUIRES_ENTITY|AFFECTS_ENTITY|READS_ENTITY|MUTATES_ENTITY
        ]->(n)
        OPTIONAL MATCH (n)-[:HAS_OPERATION]->(op_domain:Operation)
        WITH n, [op IN collect(DISTINCT op_self) + collect(DISTINCT op_workflow) +
                    collect(DISTINCT op_step) + collect(DISTINCT op_ref) +
                    collect(DISTINCT op_domain)
                 WHERE op IS NOT NULL] AS ops
        UNWIND ops AS op
        WITH n, op
        ORDER BY op.operation_id
        RETURN n.key AS key,
               collect(DISTINCT op {
                   .operation_id,
                   .method,
                   .path,
                   .summary,
                   .domain,
                   .destructive,
                   .is_mutation
               }) AS operations
        """,
        keys=keys,
        database_=settings.neo4j_database,
        timeout=settings.graphrag_neo4j_timeout_seconds,
    )
    return {record["key"]: list(record["operations"]) for record in records}


def _fetch_workflows(driver: Any, settings: Settings, workflow_keys: list[str]) -> dict[str, dict[str, Any]]:
    if not workflow_keys:
        return {}
    records, _, _ = driver.execute_query(
        """
        MATCH (w:Workflow)
        WHERE w.key IN $workflow_keys
        OPTIONAL MATCH (w)-[:HAS_WORKFLOW_STEP]->(step:WorkflowStep)
        OPTIONAL MATCH (step)-[:USES_OPERATION]->(op:Operation)
        WITH w, step, op
        ORDER BY step.order
        RETURN w.key AS key,
               w {
                   .key,
                   .name,
                   .description,
                   .supported,
                   .requires_clarification,
                   .missing_capabilities,
                   .decision_rule
               } AS workflow,
               collect({
                   order: step.order,
                   operation_id: coalesce(step.operation_id, op.operation_id),
                   purpose: step.purpose,
                   condition: step.condition,
                   output: step.output,
                   query_hint_json: step.query_hint_json,
                   path_hint_json: step.path_hint_json,
                   body_hint_json: step.body_hint_json,
                   approval_required: step.approval_required,
                   requires_user_confirmation: step.requires_user_confirmation,
                   missing_capability: step.missing_capability,
                   status: step.status,
                   operation: op {
                       .operation_id,
                       .method,
                       .path,
                       .domain,
                       .destructive,
                       .is_mutation
                   }
               }) AS steps
        """,
        workflow_keys=workflow_keys,
        database_=settings.neo4j_database,
        timeout=settings.graphrag_neo4j_timeout_seconds,
    )
    return {
        record["key"]: {
            **record["workflow"],
            "steps": [_format_step(step) for step in record["steps"] if step.get("order") is not None],
        }
        for record in records
    }


def _fetch_workflow_keys_for_operations(
    driver: Any,
    settings: Settings,
    operation_keys: list[str],
) -> dict[str, list[str]]:
    if not operation_keys:
        return {}
    records, _, _ = driver.execute_query(
        """
        MATCH (w:Workflow)-[:HAS_WORKFLOW_STEP]->(step:WorkflowStep)-[:USES_OPERATION]->(op:Operation)
        WHERE op.key IN $operation_keys
        RETURN op.key AS operation_id,
               collect(DISTINCT w.key) AS workflow_keys
        """,
        operation_keys=operation_keys,
        database_=settings.neo4j_database,
        timeout=settings.graphrag_neo4j_timeout_seconds,
    )
    return {
        record["operation_id"]: [key for key in record["workflow_keys"] if key]
        for record in records
    }


def _attach_workflow_keys(
    operations: dict[str, list[dict[str, Any]]],
    workflow_keys_by_operation: dict[str, list[str]],
) -> None:
    for operation_list in operations.values():
        for operation in operation_list:
            operation["workflow_keys"] = workflow_keys_by_operation.get(
                str(operation.get("operation_id")),
                [],
            )


def _fetch_operation_related(
    driver: Any,
    settings: Settings,
    operation_keys: list[str],
) -> dict[str, dict[str, Any]]:
    if not operation_keys:
        return {}
    records, _, _ = driver.execute_query(
        """
        MATCH (op:Operation)
        WHERE op.key IN $operation_keys
        OPTIONAL MATCH (op)-[:REQUIRES_ENTITY|AFFECTS_ENTITY|READS_ENTITY|MUTATES_ENTITY]->(entity:Entity)
        OPTIONAL MATCH (op)-[:REQUIRES_PARAMETER|ACCEPTS_PARAMETER]->(param:Parameter)
        OPTIONAL MATCH (op)-[:REQUIRES_FIELD|ACCEPTS_FIELD|RETURNS_FIELD]->(field:Field)
        OPTIONAL MATCH (op)-[:HAS_RISK_POLICY]->(risk:RiskPolicy)
        RETURN op.key AS operation_id,
               collect(DISTINCT entity.name) AS entities,
               collect(DISTINCT param.name) AS parameters,
               collect(DISTINCT field.name) AS fields,
               risk.name AS risk_policy
        """,
        operation_keys=operation_keys,
        database_=settings.neo4j_database,
        timeout=settings.graphrag_neo4j_timeout_seconds,
    )
    return {
        record["operation_id"]: {
            "entities": [item for item in record["entities"] if item],
            "parameters": [item for item in record["parameters"] if item],
            "fields": [item for item in record["fields"] if item],
            "risk_policy": record["risk_policy"],
        }
        for record in records
    }


def _workflow_keys(nodes: dict[str, dict[str, Any]], operations: dict[str, list[dict[str, Any]]]) -> set[str]:
    keys: set[str] = set()
    for node in nodes.values():
        keys.update(_node_workflow_keys(node, operations.get(str(node.get("key")), [])))
    keys.update(_workflow_keys_for_operations(nodes, operations))
    return keys


def _workflow_keys_for_operations(
    nodes: dict[str, dict[str, Any]],
    operations: dict[str, list[dict[str, Any]]],
) -> set[str]:
    keys: set[str] = set()
    for node in nodes.values():
        if node.get("label") == "Workflow":
            keys.add(str(node["key"]))
        workflow_id = node.get("workflow_id")
        if workflow_id:
            keys.add(str(workflow_id))
    for operation_list in operations.values():
        for operation in operation_list:
            for workflow_key in operation.get("workflow_keys", []) or []:
                keys.add(str(workflow_key))
    return keys


def _node_workflow_keys(node: dict[str, Any], operations: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    if node.get("label") == "Workflow":
        keys.append(str(node["key"]))
    if node.get("workflow_id"):
        keys.append(str(node["workflow_id"]))
    for operation in operations:
        for workflow_key in operation.get("workflow_keys", []) or []:
            keys.append(str(workflow_key))
    return list(dict.fromkeys(keys))


def _operation_keys(operations: dict[str, list[dict[str, Any]]]) -> list[str]:
    keys = [
        operation["operation_id"]
        for operation_list in operations.values()
        for operation in operation_list
        if operation.get("operation_id")
    ]
    return list(dict.fromkeys(keys))


def _format_node(labels: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    clean = _clean_properties(properties)
    clean["labels"] = labels
    clean["label"] = next((label for label in labels if label != "KnowledgeNode"), labels[0] if labels else "Node")
    return clean


def _format_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "order": step.get("order"),
        "operation_id": step.get("operation_id"),
        "purpose": step.get("purpose"),
        "condition": step.get("condition"),
        "output": step.get("output"),
        "query_hint": _json_or_none(step.get("query_hint_json")),
        "path_hint": _json_or_none(step.get("path_hint_json")),
        "body_hint": _json_or_none(step.get("body_hint_json")),
        "approval_required": bool(step.get("approval_required", False)),
        "requires_user_confirmation": bool(step.get("requires_user_confirmation", False)),
        "missing_capability": step.get("missing_capability"),
        "status": step.get("status"),
        "operation": step.get("operation"),
    }


def _clean_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(properties).items()
        if key not in {EMBEDDING_PROPERTY, EMBEDDING_TEXT_PROPERTY, "schema_json"}
    }


def _json_or_none(value: Any) -> Any:
    if not value:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _empty_related() -> dict[str, Any]:
    return {"entities": [], "parameters": [], "fields": [], "risk_policy": None}


def _empty_result(*, degraded: bool) -> dict[str, Any]:
    return {"items": [], "meta": {"degraded": degraded, "error_logged": degraded}}
