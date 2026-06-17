from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeRecord:
    label: str
    key: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationshipRecord:
    start_label: str
    start_key: str
    rel_type: str
    end_label: str
    end_key: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphDocument:
    nodes: tuple[NodeRecord, ...]
    relationships: tuple[RelationshipRecord, ...]

    def node_count_by_label(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.label] = counts.get(node.label, 0) + 1
        return counts

    def relationship_count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for relationship in self.relationships:
            counts[relationship.rel_type] = counts.get(relationship.rel_type, 0) + 1
        return counts
