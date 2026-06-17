from __future__ import annotations

import argparse
import re
from collections import defaultdict
from typing import Any

from dionisio_agent.config import Settings
from dionisio_agent.graphrag.builder import build_openapi_graph
from dionisio_agent.graphrag.constants import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_PROPERTY,
    EMBEDDING_TEXT_PROPERTY,
    KNOWLEDGE_NODE_LABEL,
    SEARCHABLE_NODE_LABELS,
    VECTOR_INDEX_NAME,
)
from dionisio_agent.graphrag.embeddings import create_openai_embedder
from dionisio_agent.graphrag.models import GraphDocument, NodeRecord, RelationshipRecord
from dionisio_agent.operation_catalog import OperationCatalog

SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


async def build_graph_from_settings(settings: Settings) -> GraphDocument:
    catalog = await OperationCatalog.from_url(
        settings.dionisio_openapi_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return build_openapi_graph(catalog)


async def ingest_graph_from_settings(settings: Settings, *, clear: bool = False) -> dict[str, Any]:
    graph = await build_graph_from_settings(settings)
    write_graph_to_neo4j(graph, settings, clear=clear)
    return {
        "nodes": len(graph.nodes),
        "relationships": len(graph.relationships),
        "node_labels": graph.node_count_by_label(),
        "relationship_types": graph.relationship_count_by_type(),
        "neo4j_uri": settings.neo4j_uri,
        "database": settings.neo4j_database,
    }


def write_graph_to_neo4j(graph: GraphDocument, settings: Settings, *, clear: bool = False) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover - depends on optional package install
        raise RuntimeError("neo4j package is required. Install project dependencies first.") from exc

    driver = GraphDatabase.driver(
        settings.require_neo4j_uri(),
        auth=(settings.neo4j_username, settings.require_neo4j_password()),
    )
    try:
        with driver.session(database=settings.neo4j_database) as session:
            if clear:
                session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n").consume())
            session.execute_write(_create_constraints, graph.nodes)
            for label, nodes in _nodes_by_label(graph.nodes).items():
                session.execute_write(_upsert_nodes, label, nodes)
            for rel_key, relationships in _relationships_by_pattern(graph.relationships).items():
                session.execute_write(_upsert_relationships, rel_key, relationships)
            session.execute_write(_mark_knowledge_nodes)
            _create_knowledge_vector_index(driver, settings)
            _embed_knowledge_nodes(driver, settings)
    finally:
        driver.close()


def _create_constraints(tx: Any, nodes: tuple[NodeRecord, ...]) -> None:
    labels = sorted({node.label for node in nodes})
    for label in labels:
        _validate_identifier(label)
        tx.run(
            f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.key IS UNIQUE"
        ).consume()


def _upsert_nodes(tx: Any, label: str, nodes: list[NodeRecord]) -> None:
    _validate_identifier(label)
    rows = [
        {
            "key": node.key,
            "properties": {**node.properties, "key": node.key},
        }
        for node in nodes
    ]
    tx.run(
        f"UNWIND $rows AS row "
        f"MERGE (n:{label} {{key: row.key}}) "
        "SET n += row.properties",
        rows=rows,
    ).consume()


def _upsert_relationships(
    tx: Any,
    rel_key: tuple[str, str, str],
    relationships: list[RelationshipRecord],
) -> None:
    start_label, rel_type, end_label = rel_key
    _validate_identifier(start_label)
    _validate_identifier(end_label)
    _validate_identifier(rel_type)
    rows = [
        {
            "start_key": relationship.start_key,
            "end_key": relationship.end_key,
            "properties": relationship.properties,
        }
        for relationship in relationships
    ]
    tx.run(
        f"UNWIND $rows AS row "
        f"MATCH (a:{start_label} {{key: row.start_key}}) "
        f"MATCH (b:{end_label} {{key: row.end_key}}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        "SET r += row.properties",
        rows=rows,
    ).consume()


def _mark_knowledge_nodes(tx: Any) -> None:
    _validate_identifier(KNOWLEDGE_NODE_LABEL)
    for label in sorted(SEARCHABLE_NODE_LABELS):
        _validate_identifier(label)
        tx.run(f"MATCH (n:{label}) SET n:{KNOWLEDGE_NODE_LABEL}").consume()


def _create_knowledge_vector_index(driver: Any, settings: Settings) -> None:
    try:
        from neo4j_graphrag.indexes import create_vector_index
    except ImportError as exc:  # pragma: no cover - depends on optional package install
        raise RuntimeError(
            "neo4j-graphrag is required. Install project dependencies first."
        ) from exc

    create_vector_index(
        driver,
        VECTOR_INDEX_NAME,
        label=KNOWLEDGE_NODE_LABEL,
        embedding_property=EMBEDDING_PROPERTY,
        dimensions=EMBEDDING_DIMENSIONS,
        similarity_fn="cosine",
        fail_if_exists=False,
        neo4j_database=settings.neo4j_database,
    )


def _embed_knowledge_nodes(driver: Any, settings: Settings, *, batch_size: int = 25) -> None:
    embedder = create_openai_embedder(settings)
    records, _, _ = driver.execute_query(
        f"""
        MATCH (n:{KNOWLEDGE_NODE_LABEL})
        WHERE n.{EMBEDDING_TEXT_PROPERTY} IS NOT NULL
        RETURN n.key AS key, n.{EMBEDDING_TEXT_PROPERTY} AS text
        ORDER BY n.key
        """,
        database_=settings.neo4j_database,
    )

    pending = list(records)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [str(record["text"]) for record in batch]
        embeddings = _embed_batch(embedder, texts)
        rows = [
            {"key": record["key"], "embedding": embedding}
            for record, embedding in zip(batch, embeddings, strict=True)
        ]
        _write_embeddings(driver, settings, rows)


def _embed_batch(embedder: Any, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = getattr(embedder, "client", None)
    model = getattr(embedder, "model", None)
    if client is None or model is None:
        return [embedder.embed_query(text) for text in texts]
    response = client.embeddings.create(input=texts, model=model)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def _write_embeddings(driver: Any, settings: Settings, rows: list[dict[str, Any]]) -> None:
    driver.execute_query(
        f"""
        UNWIND $rows AS row
        MATCH (n:{KNOWLEDGE_NODE_LABEL} {{key: row.key}})
        SET n.{EMBEDDING_PROPERTY} = row.embedding
        """,
        rows=rows,
        database_=settings.neo4j_database,
    )


def _nodes_by_label(nodes: tuple[NodeRecord, ...]) -> dict[str, list[NodeRecord]]:
    grouped: dict[str, list[NodeRecord]] = defaultdict(list)
    for node in nodes:
        grouped[node.label].append(node)
    return dict(grouped)


def _relationships_by_pattern(
    relationships: tuple[RelationshipRecord, ...],
) -> dict[tuple[str, str, str], list[RelationshipRecord]]:
    grouped: dict[tuple[str, str, str], list[RelationshipRecord]] = defaultdict(list)
    for relationship in relationships:
        grouped[(relationship.start_label, relationship.rel_type, relationship.end_label)].append(relationship)
    return dict(grouped)


def _validate_identifier(value: str) -> None:
    if not SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe Neo4j identifier: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="dionisio-graphrag-ingest")
    parser.add_argument("--clear", action="store_true", help="Delete current graph before ingesting.")
    args = parser.parse_args()

    import asyncio
    import json

    settings = Settings.from_env()
    result = asyncio.run(ingest_graph_from_settings(settings, clear=args.clear))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
