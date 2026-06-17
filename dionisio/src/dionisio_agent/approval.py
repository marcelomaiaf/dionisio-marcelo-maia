from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def call_fingerprint(operation_id: str, target: dict[str, Any], body: Any) -> str:
    payload = {
        "operation_id": operation_id,
        "target": target,
        "body": body,
    }
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class PendingApproval:
    approval_id: str
    fingerprint: str
    action_summary: str
    operation_id: str
    risk_level: str
    approved: bool
    created_at: str


class ApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, PendingApproval] = {}

    def request(
        self,
        *,
        fingerprint: str,
        action_summary: str,
        operation_id: str,
        risk_level: str,
    ) -> PendingApproval:
        approval = PendingApproval(
            approval_id=f"appr_{uuid.uuid4().hex[:12]}",
            fingerprint=fingerprint,
            action_summary=action_summary,
            operation_id=operation_id,
            risk_level=risk_level,
            approved=False,
            created_at=datetime.now(tz=UTC).isoformat(),
        )
        self._items[approval.approval_id] = approval
        return approval

    def approve(self, approval_id: str) -> PendingApproval:
        approval = self._items[approval_id]
        approval.approved = True
        return approval

    def get(self, approval_id: str) -> PendingApproval | None:
        return self._items.get(approval_id)

    def is_approved(self, approval_id: str | None, fingerprint: str) -> bool:
        if not approval_id:
            return False
        approval = self._items.get(approval_id)
        return bool(approval and approval.approved and approval.fingerprint == fingerprint)
