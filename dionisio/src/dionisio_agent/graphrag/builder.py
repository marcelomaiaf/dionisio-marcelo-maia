from __future__ import annotations

import json
from typing import Any

from dionisio_agent.graphrag.constants import EMBEDDING_TEXT_PROPERTY, SEARCHABLE_NODE_LABELS
from dionisio_agent.graphrag.models import GraphDocument, NodeRecord, RelationshipRecord
from dionisio_agent.operation_catalog import Operation, OperationCatalog

DOMAIN_ENTITY = {
    "Analytics": "Metric",
    "Clientes": "Client",
    "Cupons": "Coupon",
    "Delivery": "Delivery",
    "iFood": "IfoodOrder",
    "Loja": "Store",
    "Pedidos": "Order",
    "Promoções": "Promotion",
    "Reservas": "Reservation",
    "Sistema": "System",
}

DOMAIN_DEPENDENCIES = {
    "Analytics": ("Clientes", "Cupons", "Delivery", "iFood", "Loja", "Pedidos", "Promoções", "Reservas"),
    "Cupons": ("Clientes", "Promoções"),
    "Delivery": ("Loja", "Pedidos"),
    "iFood": ("Loja", "Pedidos", "Delivery"),
    "Pedidos": ("Clientes", "Cupons", "Delivery", "Loja", "Promoções"),
    "Promoções": ("Cupons", "Loja", "Pedidos"),
    "Reservas": ("Clientes", "Loja"),
}

OPERATION_ENTITY_DEPENDENCIES = {
    "analytics.conversations": ("Metric",),
    "analytics.couponReturns": ("Metric", "Coupon"),
    "analytics.orders": ("Metric", "Order"),
    "analytics.reservations": ("Metric", "Reservation"),
    "analytics.revenue": ("Metric", "Order"),
    "analytics.topItems": ("Metric", "Product", "Order"),
    "clients.addGroupMembers": ("ClientGroup", "Client"),
    "clients.create": ("Client",),
    "clients.createGroup": ("ClientGroup", "Client"),
    "clients.get": ("Client",),
    "clients.inactive": ("Client", "Reservation", "Metric"),
    "clients.insights": ("Client", "Reservation", "Order", "Coupon", "Metric"),
    "clients.list": ("Client",),
    "clients.listGroups": ("ClientGroup",),
    "clients.reservations": ("Client", "Reservation"),
    "clients.search": ("Client",),
    "clients.topSpenders": ("Client", "Order", "Metric"),
    "clients.update": ("Client",),
    "coupons.analytics": ("Coupon", "CouponInstance", "Metric"),
    "coupons.assignClient": ("Coupon", "Client"),
    "coupons.assignGroup": ("Coupon", "ClientGroup", "Client"),
    "coupons.assignInstance": ("Coupon", "CouponInstance", "Client"),
    "coupons.create": ("Coupon",),
    "coupons.deactivate": ("Coupon",),
    "coupons.get": ("Coupon",),
    "coupons.instances": ("Coupon", "CouponInstance"),
    "coupons.list": ("Coupon",),
    "coupons.update": ("Coupon",),
    "delivery.createPause": ("Delivery", "DeliveryPause", "Store"),
    "delivery.currentPause": ("Delivery", "DeliveryPause", "Store"),
    "delivery.endPause": ("Delivery", "DeliveryPause", "Store"),
    "delivery.getConfig": ("Delivery", "Store"),
    "delivery.neighborhoods": ("Delivery", "Neighborhood", "Store"),
    "delivery.updateConfig": ("Delivery", "Store"),
    "ifood.cancel": ("IfoodOrder",),
    "ifood.confirm": ("IfoodOrder",),
    "ifood.dispatch": ("IfoodOrder", "Delivery"),
    "ifood.get": ("IfoodOrder",),
    "ifood.list": ("IfoodOrder",),
    "ifood.merchants": ("IfoodMerchant", "Store"),
    "orders.cancel": ("Order",),
    "orders.create": ("Order", "Client", "Store", "Product"),
    "orders.get": ("Order",),
    "orders.list": ("Order", "Client"),
    "orders.stats": ("Order", "Metric"),
    "orders.updateStatus": ("Order",),
    "promotions.analytics": ("Promotion", "Metric"),
    "promotions.create": ("Promotion",),
    "promotions.delete": ("Promotion",),
    "promotions.get": ("Promotion",),
    "promotions.list": ("Promotion",),
    "promotions.update": ("Promotion",),
    "reservations.availability": ("Store", "Reservation", "Availability"),
    "reservations.cancel": ("Reservation",),
    "reservations.confirm": ("Reservation",),
    "reservations.create": ("Client", "Store", "Availability", "Reservation"),
    "reservations.get": ("Reservation",),
    "reservations.list": ("Reservation", "Client"),
    "reservations.reschedule": ("Reservation", "Store", "Availability"),
    "reservations.update": ("Reservation",),
    "store.features": ("Store", "Integration"),
    "store.get": ("Store",),
    "store.getHours": ("Store",),
    "store.members": ("Store", "StaffMember"),
    "store.update": ("Store",),
    "store.updateHours": ("Store",),
    "system.reset": ("System",),
}

# The public OpenAPI has some request/query details only in summaries. These
# hints are explicitly marked as summary-derived so the agent can distinguish
# them from schema-validated fields.
SUMMARY_BODY_FIELD_HINTS = {
    "orders.create": {
        "items": "Array of order items documented in the summary.",
        "items[].productId": "Product id documented in items[{productId,quantity}].",
        "items[].quantity": "Quantity documented in items[{productId,quantity}].",
        "type": "Order type documented in the summary.",
        "clientId": "Client id documented in the summary.",
        "paymentMethod": "Payment method documented in the summary.",
    },
    "orders.updateStatus": {"status": "New order status documented in the summary."},
    "coupons.create": {
        "name": "Coupon name documented in the summary.",
        "type": "Coupon type documented in the summary.",
        "benefitText": "Human-readable coupon benefit documented in the summary.",
    },
    "coupons.assignGroup": {"groupId": "Client group id documented as body: groupId."},
    "coupons.assignClient": {"clientId": "Client id documented as body: clientId."},
    "coupons.assignInstance": {"clientId": "Client id documented as body: clientId."},
    "promotions.create": {
        "name": "Promotion name documented in the summary.",
        "discountType": "Discount type documented in the summary.",
        "discountValue": "Discount value documented in the summary.",
        "validFrom": "Promotion start date/time documented in the summary.",
        "validUntil": "Promotion end date/time documented in the summary.",
    },
    "store.updateHours": {"workingHours": "Store working hours documented as body: workingHours."},
}

SUMMARY_QUERY_PARAMETER_HINTS = {
    "analytics.couponReturns": ("periodStart", "periodEnd"),
    "analytics.conversations": ("periodStart", "periodEnd"),
    "analytics.orders": ("periodStart", "periodEnd"),
    "analytics.reservations": ("periodStart", "periodEnd"),
    "analytics.revenue": ("periodStart", "periodEnd"),
    "analytics.topItems": ("periodStart", "periodEnd", "limit"),
    "coupons.instances": ("state",),
    "coupons.list": ("status",),
    "ifood.list": ("date", "status", "limit", "offset"),
    "orders.list": ("date", "status", "type", "clientId", "limit", "offset"),
    "orders.stats": ("periodStart", "periodEnd"),
    "promotions.list": ("status", "discountType"),
}

WORKFLOW_DEFINITIONS = {
    "workflow.reservation_tonight_capacity": {
        "name": "reservation_tonight_capacity",
        "description": "Count tonight's reservations and calculate remaining availability for tonight.",
        "trigger_examples": ["Quantas reservas temos pra hoje à noite e quantos lugares ainda sobram?"],
        "steps": (
            {
                "operation_id": "reservations.list",
                "purpose": "List reservations for today's date and filter the returned reservations to the evening/night period.",
                "query_hint": {"date": "today"},
                "output": "Reservation count for tonight.",
            },
            {
                "operation_id": "reservations.availability",
                "purpose": "Fetch availability for today's date and inspect evening/night slots.",
                "query_hint": {"date": "today"},
                "output": "Free seats/slots remaining tonight.",
            },
        ),
    },
    "workflow.reservation_reschedule_by_client_name": {
        "name": "reservation_reschedule_by_client_name",
        "description": "Find a client by name, choose the correct reservation, check availability, reschedule, and verify.",
        "trigger_examples": [
            "remarca a reserva da Ana pra sexta às 20h",
            "Remarca a reserva do João de quinta pra sábado no mesmo horário e confirma se tem mesa.",
        ],
        "steps": (
            {
                "operation_id": "clients.search",
                "purpose": "Search clients by the provided name.",
                "query_hint": {"name": "client name from user"},
                "output": "Candidate clients.",
                "condition": "If more than one client matches, ask the user which client they mean and list returned names.",
                "requires_user_confirmation": True,
            },
            {
                "operation_id": "reservations.list",
                "purpose": "List reservations for the confirmed client.",
                "query_hint": {"clientId": "confirmed client id", "date": "original date if user provided one"},
                "output": "Candidate reservations.",
                "condition": "If more than one active reservation matches, ask which reservation should be rescheduled. If the user says same time, preserve the original reservation time and only change the date.",
                "requires_user_confirmation": True,
            },
            {
                "operation_id": "reservations.availability",
                "purpose": "Check availability for the requested new date and time before mutating the reservation.",
                "query_hint": {"date": "requested target date"},
                "output": "Availability at requested time.",
                "condition": "If the requested slot is unavailable, stop and explain alternatives instead of rescheduling.",
            },
            {
                "operation_id": "reservations.reschedule",
                "purpose": "Reschedule the chosen reservation to the requested start timestamp.",
                "path_hint": {"id": "reservation id"},
                "body_hint": {"start": "requested date plus original or requested time as timestamp ms"},
                "approval_required": True,
                "output": "Reschedule result.",
            },
            {
                "operation_id": "reservations.get",
                "purpose": "Fetch the reservation after reschedule to confirm the new time.",
                "path_hint": {"id": "reservation id"},
                "output": "Verified updated reservation.",
            },
        ),
    },
    "workflow.inactive_client_reactivation_campaign_60_days": {
        "name": "inactive_client_reactivation_campaign_60_days",
        "description": "Create a reactivation campaign for clients inactive for 60 days and assign a 15 percent coupon to that group.",
        "trigger_examples": [
            "Cria uma campanha de reativação para clientes inativos há 60 dias, com um cupom de 15%.",
            "cria campanha de reativacao para clientes inativos ha 60 dias",
            "campanha reativacao 60 dias cupom 15%",
        ],
        "steps": (
            {
                "operation_id": "clients.inactive",
                "purpose": "Check whether there are clients inactive for the requested number of days.",
                "query_hint": {"days": 60},
                "condition": "If total is zero, do not create a group or coupon.",
                "output": "Inactive clients and total count.",
            },
            {
                "operation_id": "clients.createGroup",
                "purpose": "Create a dynamic group for clients inactive for 60 days.",
                "body_hint": {
                    "name": "clientes_60_dias",
                    "description": "grupo de clientes que não voltam há 60 dias",
                    "rules": {"inactiveDays": 60},
                },
                "condition": "Only run if clients.inactive returned one or more clients.",
                "output": "Created client group id.",
            },
            {
                "operation_id": "coupons.create",
                "purpose": "Create a coupon representing the requested 15 percent benefit.",
                "body_hint": {"name": "reativacao_60_dias_15", "type": "percent", "benefitText": "15%"},
                "output": "Created coupon id.",
            },
            {
                "operation_id": "coupons.assignGroup",
                "purpose": "Assign the new coupon to every client in the created group.",
                "path_hint": {"id": "coupon id"},
                "body_hint": {"groupId": "created client group id"},
                "output": "Coupon assignment result.",
            },
        ),
    },
    "workflow.reservation_create_by_client_name": {
        "name": "reservation_create_by_client_name",
        "description": "Create a reservation after identifying the client and checking availability.",
        "steps": (
            {"operation_id": "clients.search", "purpose": "Identify the client by name or phone.", "requires_user_confirmation": True},
            {"operation_id": "reservations.availability", "purpose": "Check date/time availability before creating the reservation."},
            {"operation_id": "reservations.create", "purpose": "Create the reservation with clientId, start, party size and optional area/description."},
            {"operation_id": "reservations.get", "purpose": "Verify the created reservation."},
        ),
    },
    "workflow.reservation_confirm_or_update": {
        "name": "reservation_confirm_or_update",
        "description": "Confirm or update an existing reservation after locating it and checking constraints.",
        "steps": (
            {"operation_id": "clients.search", "purpose": "Identify client if the reservation id is not known.", "requires_user_confirmation": True},
            {"operation_id": "reservations.list", "purpose": "Find candidate reservations by date, status or clientId.", "requires_user_confirmation": True},
            {"operation_id": "reservations.get", "purpose": "Inspect selected reservation before changes."},
            {"operation_id": "reservations.availability", "purpose": "Check availability before changing area, party size or table when relevant."},
            {"operation_id": "reservations.update", "purpose": "Update non-time reservation fields such as areaId, adults, children, tableId or description."},
            {"operation_id": "reservations.confirm", "purpose": "Confirm a pending reservation when the user asks to confirm it."},
            {"operation_id": "reservations.get", "purpose": "Verify final reservation status and fields."},
        ),
    },
    "workflow.reservation_cancel_by_client_name": {
        "name": "reservation_cancel_by_client_name",
        "description": "Cancel a reservation only after identifying the client and reservation.",
        "steps": (
            {"operation_id": "clients.search", "purpose": "Identify the client by name or phone.", "requires_user_confirmation": True},
            {"operation_id": "reservations.list", "purpose": "Find active reservations for the confirmed client.", "requires_user_confirmation": True},
            {"operation_id": "reservations.cancel", "purpose": "Cancel the selected reservation.", "approval_required": True},
            {"operation_id": "reservations.get", "purpose": "Verify cancellation status."},
        ),
    },
    "workflow.client_lookup_profile_history": {
        "name": "client_lookup_profile_history",
        "description": "Find clients, inspect their profile, derived metrics and reservation history.",
        "trigger_examples": ["quais clientes não voltam há 60 dias?"],
        "steps": (
            {"operation_id": "clients.list", "purpose": "List clients using q, groupId, limit or offset when the user asks for a broad list."},
            {"operation_id": "clients.search", "purpose": "Search exact clients by name or phone when the user gives an identifier."},
            {"operation_id": "clients.get", "purpose": "Fetch full client details for a selected client."},
            {"operation_id": "clients.insights", "purpose": "Read visits, no-show count, spend, coupon usage and lastVisitAt for a selected client."},
            {"operation_id": "clients.reservations", "purpose": "Read reservation history for a selected client."},
            {"operation_id": "clients.inactive", "purpose": "List clients inactive for the requested number of days.", "query_hint": {"days": "inactive days from user"}},
        ),
    },
    "workflow.client_create_update": {
        "name": "client_create_update",
        "description": "Create or update a client and verify the resulting profile.",
        "steps": (
            {"operation_id": "clients.search", "purpose": "Check for an existing client by phone or name before creating duplicates."},
            {"operation_id": "clients.create", "purpose": "Create a client when required name and phone are known."},
            {"operation_id": "clients.get", "purpose": "Verify created client."},
            {"operation_id": "clients.update", "purpose": "Update allowed client fields such as name, phone, email, cpf, gender or notes."},
            {"operation_id": "clients.get", "purpose": "Verify updated client details."},
        ),
    },
    "workflow.client_group_management": {
        "name": "client_group_management",
        "description": "List, create and populate client groups.",
        "steps": (
            {"operation_id": "clients.listGroups", "purpose": "List existing client groups before reusing or creating a group."},
            {"operation_id": "clients.list", "purpose": "Find members for a static group when criteria are client attributes."},
            {"operation_id": "clients.createGroup", "purpose": "Create a static or dynamic group. Use rules for dynamic membership such as inactiveDays."},
            {"operation_id": "clients.addGroupMembers", "purpose": "Add explicit clientIds to a static group when needed."},
            {"operation_id": "clients.listGroups", "purpose": "Verify the group exists and member count changed."},
        ),
    },
    "workflow.high_spenders_without_coupon_usage": {
        "name": "high_spenders_without_coupon_usage",
        "description": "Find clients above a spending threshold in a period and filter to those who never used coupons.",
        "trigger_examples": ["Lista os clientes que gastaram mais de R$500 no último mês e nunca usaram cupom."],
        "steps": (
            {
                "operation_id": "clients.topSpenders",
                "purpose": "List clients above the requested spend threshold.",
                "query_hint": {"period": "last_month", "minSpent": "threshold in cents, e.g. R$500 -> 50000"},
                "output": "Candidate high-spending clients.",
            },
            {
                "operation_id": "clients.insights",
                "purpose": "For each candidate, inspect couponsUsed and retain clients with couponsUsed equal to zero.",
                "path_hint": {"clientId": "candidate client id"},
                "output": "Filtered clients who never used coupons.",
            },
        ),
    },
    "workflow.order_cancel": {
        "name": "order_cancel",
        "description": "Cancel an order after locating and verifying it.",
        "steps": (
            {"operation_id": "orders.list", "purpose": "Find the order when the id is not known."},
            {"operation_id": "orders.get", "purpose": "Verify selected order details before cancellation."},
            {"operation_id": "orders.cancel", "purpose": "Cancel the selected order.", "approval_required": True},
            {"operation_id": "orders.get", "purpose": "Verify cancellation result."},
        ),
    },
    "workflow.order_create_for_client": {
        "name": "order_create_for_client",
        "description": "Create an order for a known or searched client.",
        "steps": (
            {"operation_id": "clients.search", "purpose": "Identify client if clientId is not already known."},
            {"operation_id": "orders.create", "purpose": "Create order using items, type, clientId and paymentMethod."},
            {"operation_id": "orders.get", "purpose": "Verify created order."},
        ),
    },
    "workflow.order_status_management": {
        "name": "order_status_management",
        "description": "Inspect an order, update its status, and verify the change.",
        "steps": (
            {"operation_id": "orders.list", "purpose": "Find orders by date, status, type, clientId, limit or offset."},
            {"operation_id": "orders.get", "purpose": "Inspect selected order."},
            {"operation_id": "orders.updateStatus", "purpose": "Update order status when the target status is clear."},
            {"operation_id": "orders.get", "purpose": "Verify updated order status."},
        ),
    },
    "workflow.order_metrics": {
        "name": "order_metrics",
        "description": "Answer order volume/status questions and compare order metrics over a period.",
        "steps": (
            {"operation_id": "orders.list", "purpose": "List raw orders when the user needs examples or records."},
            {"operation_id": "orders.stats", "purpose": "Read aggregate order statistics for periodStart and periodEnd."},
            {"operation_id": "analytics.orders", "purpose": "Read analytics-grade order metrics for the requested period."},
            {"operation_id": "analytics.revenue", "purpose": "Combine order metrics with revenue when the question involves value."},
        ),
    },
    "workflow.coupon_assign_to_client": {
        "name": "coupon_assign_to_client",
        "description": "Create or choose a coupon and assign it to one client.",
        "steps": (
            {"operation_id": "clients.search", "purpose": "Identify target client.", "requires_user_confirmation": True},
            {"operation_id": "coupons.list", "purpose": "Find an existing coupon if the user references one."},
            {"operation_id": "coupons.create", "purpose": "Create a coupon if the user asks for a new one."},
            {"operation_id": "coupons.assignClient", "purpose": "Assign coupon to the confirmed client."},
            {"operation_id": "coupons.instances", "purpose": "Verify generated coupon instance/code."},
        ),
    },
    "workflow.coupon_performance": {
        "name": "coupon_performance",
        "description": "Analyze coupon performance and overall coupon returns.",
        "steps": (
            {"operation_id": "coupons.list", "purpose": "Find the coupon when id is not known."},
            {"operation_id": "coupons.analytics", "purpose": "Read generated/used/usage-rate metrics for a coupon."},
            {"operation_id": "analytics.couponReturns", "purpose": "Compare coupon returns for the requested period."},
        ),
    },
    "workflow.coupon_lifecycle_management": {
        "name": "coupon_lifecycle_management",
        "description": "Inspect, update or deactivate coupons with verification.",
        "steps": (
            {"operation_id": "coupons.list", "purpose": "Find coupons by status or list candidates."},
            {"operation_id": "coupons.get", "purpose": "Inspect selected coupon details."},
            {"operation_id": "coupons.update", "purpose": "Update allowed coupon fields when requested."},
            {"operation_id": "coupons.deactivate", "purpose": "Deactivate selected coupon.", "approval_required": True},
            {"operation_id": "coupons.get", "purpose": "Verify final coupon state."},
        ),
    },
    "workflow.coupon_instance_assignment": {
        "name": "coupon_instance_assignment",
        "description": "Find coupon instances and assign an available instance to a client.",
        "steps": (
            {"operation_id": "coupons.list", "purpose": "Find the coupon if id is unknown."},
            {"operation_id": "coupons.instances", "purpose": "Find available coupon instances using state when needed."},
            {"operation_id": "clients.search", "purpose": "Identify target client by name or phone.", "requires_user_confirmation": True},
            {"operation_id": "coupons.assignInstance", "purpose": "Associate an available coupon instance to the confirmed client."},
            {"operation_id": "coupons.instances", "purpose": "Verify instance assignment."},
        ),
    },
    "workflow.delivery_pause": {
        "name": "delivery_pause",
        "description": "Pause delivery after checking current pause/configuration.",
        "steps": (
            {"operation_id": "delivery.currentPause", "purpose": "Check whether delivery is already paused."},
            {"operation_id": "delivery.getConfig", "purpose": "Read delivery configuration before pausing."},
            {"operation_id": "delivery.createPause", "purpose": "Create a delivery pause.", "approval_required": True},
            {"operation_id": "delivery.currentPause", "purpose": "Verify active pause."},
        ),
    },
    "workflow.delivery_resume": {
        "name": "delivery_resume",
        "description": "End an active delivery pause and verify delivery state.",
        "steps": (
            {"operation_id": "delivery.currentPause", "purpose": "Find the active pause id."},
            {"operation_id": "delivery.endPause", "purpose": "End the active pause."},
            {"operation_id": "delivery.currentPause", "purpose": "Verify there is no active pause."},
        ),
    },
    "workflow.delivery_configuration_management": {
        "name": "delivery_configuration_management",
        "description": "Inspect and update delivery configuration, neighborhoods and fees.",
        "steps": (
            {"operation_id": "delivery.getConfig", "purpose": "Read delivery fees, minimum and scheduling config."},
            {"operation_id": "delivery.neighborhoods", "purpose": "List served neighborhoods and fees."},
            {"operation_id": "delivery.updateConfig", "purpose": "Update delivery configuration when requested."},
            {"operation_id": "delivery.getConfig", "purpose": "Verify updated delivery config."},
        ),
    },
    "workflow.ifood_order_lifecycle": {
        "name": "ifood_order_lifecycle",
        "description": "Confirm and dispatch iFood orders through their operational lifecycle.",
        "steps": (
            {"operation_id": "ifood.list", "purpose": "Find pending iFood orders."},
            {"operation_id": "ifood.get", "purpose": "Inspect selected iFood order."},
            {"operation_id": "ifood.confirm", "purpose": "Confirm pending iFood order."},
            {"operation_id": "ifood.dispatch", "purpose": "Dispatch confirmed iFood order."},
            {"operation_id": "ifood.get", "purpose": "Verify final iFood order state."},
        ),
    },
    "workflow.ifood_cancel_order": {
        "name": "ifood_cancel_order",
        "description": "Cancel an iFood order after locating and verifying it.",
        "steps": (
            {"operation_id": "ifood.merchants", "purpose": "List iFood merchants if the order context depends on merchant identity."},
            {"operation_id": "ifood.list", "purpose": "Find iFood orders by date, status, limit or offset."},
            {"operation_id": "ifood.get", "purpose": "Inspect selected iFood order before cancellation."},
            {"operation_id": "ifood.cancel", "purpose": "Request cancellation of the selected iFood order.", "approval_required": True},
            {"operation_id": "ifood.get", "purpose": "Verify final iFood order state."},
        ),
    },
    "workflow.store_profile_update": {
        "name": "store_profile_update",
        "description": "Inspect and update store profile/configuration fields.",
        "steps": (
            {"operation_id": "store.get", "purpose": "Read current store data such as name, address, plan and payments."},
            {"operation_id": "store.features", "purpose": "Read active feature flags and integrations before suggesting actions that depend on integrations."},
            {"operation_id": "store.members", "purpose": "List team members and roles when the request involves staff or responsibility."},
            {"operation_id": "store.update", "purpose": "Update store data when requested."},
            {"operation_id": "store.get", "purpose": "Verify updated store data."},
        ),
    },
    "workflow.store_hours_update": {
        "name": "store_hours_update",
        "description": "Update store working hours and verify the change.",
        "steps": (
            {"operation_id": "store.getHours", "purpose": "Read current working hours."},
            {"operation_id": "store.updateHours", "purpose": "Update workingHours."},
            {"operation_id": "store.getHours", "purpose": "Verify updated working hours."},
        ),
    },
    "workflow.analytics_revenue_and_top_items": {
        "name": "analytics_revenue_and_top_items",
        "description": "Answer business performance questions by combining revenue, orders and top item analytics.",
        "steps": (
            {"operation_id": "analytics.revenue", "purpose": "Read revenue for the requested period."},
            {"operation_id": "analytics.orders", "purpose": "Read order metrics for the requested period."},
            {"operation_id": "analytics.topItems", "purpose": "Read best-selling items for the requested period."},
        ),
    },
    "workflow.analytics_reservations_and_conversations": {
        "name": "analytics_reservations_and_conversations",
        "description": "Answer reservation and conversation performance questions for a period.",
        "steps": (
            {"operation_id": "analytics.reservations", "purpose": "Read reservation metrics such as no-show rate for the requested period."},
            {"operation_id": "analytics.conversations", "purpose": "Read IA vs human conversation metrics for the requested period."},
            {"operation_id": "reservations.list", "purpose": "Inspect raw reservations when aggregate metrics need examples or filtering."},
        ),
    },
    "workflow.promotion_create_measure": {
        "name": "promotion_create_measure",
        "description": "Create a promotion and later inspect its performance.",
        "steps": (
            {"operation_id": "promotions.create", "purpose": "Create promotion with documented discount fields."},
            {"operation_id": "promotions.get", "purpose": "Verify promotion details."},
            {"operation_id": "promotions.analytics", "purpose": "Inspect promotion usage metrics."},
        ),
    },
    "workflow.promotion_lifecycle_management": {
        "name": "promotion_lifecycle_management",
        "description": "List, inspect, update or delete promotions with verification.",
        "steps": (
            {"operation_id": "promotions.list", "purpose": "Find promotions by status or discountType."},
            {"operation_id": "promotions.get", "purpose": "Inspect selected promotion."},
            {"operation_id": "promotions.update", "purpose": "Update promotion fields when requested."},
            {"operation_id": "promotions.delete", "purpose": "Delete selected promotion.", "approval_required": True},
            {"operation_id": "promotions.get", "purpose": "Verify deleted promotion remains accessible by GET when relevant."},
        ),
    },
    "workflow.contextual_campaign_for_existing_group": {
        "name": "contextual_campaign_for_existing_group",
        "description": "Create a coupon campaign for an already identified client group; ask for clarification when no group is available in context.",
        "trigger_examples": ["cria uma campanha de reativação pra esse grupo"],
        "requires_clarification": True,
        "decision_rule": "If the prior conversation does not contain a concrete groupId or unambiguous group reference, ask which group should receive the campaign before creating a coupon.",
        "steps": (
            {"operation_id": "clients.listGroups", "purpose": "Resolve the referenced group when groupId is not in context.", "requires_user_confirmation": True},
            {"operation_id": "coupons.create", "purpose": "Create the requested coupon once the group is known."},
            {"operation_id": "coupons.assignGroup", "purpose": "Assign the coupon to the confirmed group."},
            {"operation_id": "coupons.instances", "purpose": "Verify coupon instances generated for the group."},
        ),
    },
    "workflow.reschedule_and_notify_client_unsupported": {
        "name": "reschedule_and_notify_client_unsupported",
        "description": "Rescheduling is supported, but notifying the client is not available in the documented API.",
        "trigger_examples": [
            "remarca a reserva da Ana pra sexta às 20h e avisa ela",
            "remarca reserva e avisa cliente",
            "remarcar reserva com notificacao ao cliente",
        ],
        "supported": False,
        "missing_capabilities": ["client notification", "message sending", "messaging endpoint"],
        "decision_rule": "Offer to reschedule using reservation_reschedule_by_client_name, but explicitly say the API has no endpoint to notify the client.",
        "steps": (
            {"operation_id": "clients.search", "purpose": "Supported part: identify the client.", "requires_user_confirmation": True},
            {"operation_id": "reservations.list", "purpose": "Supported part: find the reservation.", "requires_user_confirmation": True},
            {"operation_id": "reservations.availability", "purpose": "Supported part: check availability."},
            {"operation_id": "reservations.reschedule", "purpose": "Supported part: reschedule after approval.", "approval_required": True},
            {"purpose": "Unsupported part: notify the client.", "missing_capability": "No documented notification endpoint exists.", "status": "unsupported"},
        ),
    },
    "workflow.menu_item_remove_and_notify_unsupported": {
        "name": "menu_item_remove_and_notify_unsupported",
        "description": "Removing a menu item and notifying customers is not supported by the documented API.",
        "trigger_examples": [
            "O prato ‘Risoto de Funghi’ saiu do cardápio — remove ele e avisa quem tinha pedido ele nos últimos 7 dias.",
            "prato saiu do cardapio remove e avisa clientes",
            "remove item do cardapio e notifica clientes",
            "cardapio avisa clientes",
        ],
        "supported": False,
        "missing_capabilities": ["menu item catalog", "product lookup by dish name", "menu update/delete endpoint", "client notification endpoint"],
        "decision_rule": "Do not invent menu or notification endpoints. Explain that the API can inspect recent orders only if enough order/product data is returned, but cannot remove the dish or notify customers.",
        "steps": (
            {"operation_id": "orders.list", "purpose": "Supported diagnostic part: inspect orders from the last 7 days if product ids/items are present in returned data.", "query_hint": {"date": "last 7 days"}},
            {"purpose": "Unsupported part: map dish name to product id.", "missing_capability": "No menu/product lookup endpoint exists.", "status": "unsupported"},
            {"purpose": "Unsupported part: remove dish from menu.", "missing_capability": "No menu update/delete endpoint exists.", "status": "unsupported"},
            {"purpose": "Unsupported part: notify affected clients.", "missing_capability": "No notification endpoint exists.", "status": "unsupported"},
        ),
    },
    "workflow.system_reset": {
        "name": "system_reset",
        "description": "Reset the case mock world only after explicit human approval.",
        "steps": (
            {"operation_id": "system.reset", "purpose": "Reset all case mock data to the seeded original state.", "approval_required": True},
        ),
    },
}

WORKFLOW_STEPS = {
    workflow_id: tuple(
        step["operation_id"]
        for step in workflow["steps"]
        if step.get("operation_id")
    )
    for workflow_id, workflow in WORKFLOW_DEFINITIONS.items()
}


def build_openapi_graph(catalog: OperationCatalog) -> GraphDocument:
    nodes: dict[tuple[str, str], NodeRecord] = {}
    relationships: dict[tuple[str, str, str, str, str], RelationshipRecord] = {}

    def add_node(label: str, key: str, **properties: Any) -> None:
        existing = nodes.get((label, key))
        merged = {**(existing.properties if existing else {}), **_neo4j_properties(properties), "key": key}
        if label in SEARCHABLE_NODE_LABELS:
            merged[EMBEDDING_TEXT_PROPERTY] = _embedding_text(label, key, merged)
        nodes[(label, key)] = NodeRecord(label=label, key=key, properties=merged)

    def add_rel(
        start_label: str,
        start_key: str,
        rel_type: str,
        end_label: str,
        end_key: str,
        **properties: Any,
    ) -> None:
        rel_key = (start_label, start_key, rel_type, end_label, end_key)
        existing = relationships.get(rel_key)
        merged = {**(existing.properties if existing else {}), **_neo4j_properties(properties)}
        relationships[rel_key] = RelationshipRecord(
            start_label=start_label,
            start_key=start_key,
            rel_type=rel_type,
            end_label=end_label,
            end_key=end_key,
            properties=merged,
        )

    add_node(
        "SourceDocument",
        "openapi.docs_json",
        name="Dionisio OpenAPI docs.json",
        source_type="openapi",
        openapi_version=catalog.spec.get("openapi"),
        title=catalog.spec.get("info", {}).get("title"),
        version=catalog.spec.get("info", {}).get("version"),
    )

    for domain, entity in DOMAIN_ENTITY.items():
        add_node("Domain", domain, name=domain)
        add_node("Entity", entity, name=entity)
        add_rel("Domain", domain, "PRIMARY_ENTITY", "Entity", entity)
        for dependency in DOMAIN_DEPENDENCIES.get(domain, ()):
            add_node("Domain", dependency, name=dependency)
            add_rel("Domain", domain, "DEPENDS_ON_DOMAIN", "Domain", dependency)

    for operation in catalog.operations:
        _add_operation_graph(operation, catalog, add_node, add_rel)

    _add_workflow_graph(catalog, add_node, add_rel)

    return GraphDocument(
        nodes=tuple(nodes.values()),
        relationships=tuple(relationships.values()),
    )


def _add_operation_graph(
    operation: Operation,
    catalog: OperationCatalog,
    add_node: Any,
    add_rel: Any,
) -> None:
    endpoint_key = f"{operation.method} {operation.path}"
    operation_definition = _operation_definition(catalog, operation)
    documented_responses = _response_schemas(operation_definition)
    body_status = _body_documentation_status(operation)

    add_node(
        "Operation",
        operation.operation_id,
        operation_id=operation.operation_id,
        method=operation.method,
        path=operation.path,
        summary=operation.summary,
        description=operation_definition.get("description"),
        domain=operation.domain,
        destructive=operation.destructive,
        is_mutation=operation.is_mutation,
        has_request_schema=operation.request_schema is not None,
        body_documentation_status=body_status,
        response_statuses=list(documented_responses),
    )
    add_node("Endpoint", endpoint_key, method=operation.method, path=operation.path)
    add_node("RiskPolicy", _risk_key(operation), name=_risk_key(operation))
    add_rel("Domain", operation.domain, "HAS_OPERATION", "Operation", operation.operation_id)
    add_rel("Operation", operation.operation_id, "USES_ENDPOINT", "Endpoint", endpoint_key)
    add_rel("SourceDocument", "openapi.docs_json", "DOCUMENTS", "Operation", operation.operation_id)
    add_rel("Operation", operation.operation_id, "HAS_RISK_POLICY", "RiskPolicy", _risk_key(operation))

    produced_entity = DOMAIN_ENTITY.get(operation.domain)
    if produced_entity:
        add_rel("Operation", operation.operation_id, "AFFECTS_ENTITY", "Entity", produced_entity)
        if operation.is_mutation:
            add_rel("Operation", operation.operation_id, "MUTATES_ENTITY", "Entity", produced_entity)
        else:
            add_rel("Operation", operation.operation_id, "READS_ENTITY", "Entity", produced_entity)

    for entity in OPERATION_ENTITY_DEPENDENCIES.get(operation.operation_id, ()):
        add_node("Entity", entity, name=entity)
        add_rel("Operation", operation.operation_id, "REQUIRES_ENTITY", "Entity", entity)

    for parameter in operation.parameters:
        key = f"{operation.operation_id}.{parameter.location}.{parameter.name}"
        add_node(
            "Parameter",
            key,
            name=parameter.name,
            location=parameter.location,
            required=parameter.required,
            schema_type=parameter.schema.get("type"),
            schema_ref=parameter.schema.get("$ref"),
            schema_json=_json_dumps(parameter.schema),
            description=parameter.description,
            documentation_source="openapi_parameter",
        )
        add_rel(
            "Operation",
            operation.operation_id,
            "REQUIRES_PARAMETER" if parameter.required else "ACCEPTS_PARAMETER",
            "Parameter",
            key,
        )
        referenced = _field_entity_reference(parameter.name)
        if referenced:
            add_node("Entity", referenced, name=referenced)
            add_rel("Parameter", key, "REFERENCES_ENTITY", "Entity", referenced)

    for parameter_name in SUMMARY_QUERY_PARAMETER_HINTS.get(operation.operation_id, ()):
        if _has_parameter(operation, "query", parameter_name):
            continue
        key = f"{operation.operation_id}.query.{parameter_name}"
        add_node(
            "Parameter",
            key,
            name=parameter_name,
            location="query",
            required=False,
            documentation_source="summary_hint",
            description="Mentioned in the OpenAPI summary, but not declared in formal parameters.",
        )
        add_rel("Operation", operation.operation_id, "ACCEPTS_PARAMETER", "Parameter", key)
        referenced = _field_entity_reference(parameter_name)
        if referenced:
            add_node("Entity", referenced, name=referenced)
            add_rel("Parameter", key, "REFERENCES_ENTITY", "Entity", referenced)

    schema = catalog.resolved_request_schema(operation.operation_id)
    for field_path, field_schema, required in _iter_schema_fields(schema):
        _add_field(
            add_node,
            add_rel,
            operation,
            field_path,
            field_schema,
            required,
            location="body",
            documentation_source="openapi_request_schema",
            rel_type="REQUIRES_FIELD" if required else "ACCEPTS_FIELD",
        )

    for field_path, description in SUMMARY_BODY_FIELD_HINTS.get(operation.operation_id, {}).items():
        if _schema_contains_field(schema, field_path):
            continue
        _add_field(
            add_node,
            add_rel,
            operation,
            field_path,
            {"description": description},
            required=False,
            location="body",
            documentation_source="summary_hint",
            rel_type="ACCEPTS_FIELD",
        )

    for status, response_schema in documented_responses.items():
        schema_key = f"{operation.operation_id}.response.{status}"
        add_node(
            "ResponseSchema",
            schema_key,
            operation_id=operation.operation_id,
            status=status,
            description=_response_description(operation_definition, status),
            schema_json=_json_dumps(response_schema),
        )
        add_rel("Operation", operation.operation_id, "RETURNS_SCHEMA", "ResponseSchema", schema_key)
        resolved = catalog._resolve_refs(response_schema)
        for field_path, field_schema, required in _iter_schema_fields(resolved):
            field_key = _add_field(
                add_node,
                add_rel,
                operation,
                field_path,
                field_schema,
                required,
                location=f"response.{status}",
                documentation_source="openapi_response_schema",
                rel_type="RETURNS_FIELD",
            )
            add_rel("ResponseSchema", schema_key, "HAS_FIELD", "Field", field_key)


def _add_workflow_graph(catalog: OperationCatalog, add_node: Any, add_rel: Any) -> None:
    for workflow_id, workflow in WORKFLOW_DEFINITIONS.items():
        add_node(
            "Workflow",
            workflow_id,
            name=workflow["name"],
            description=workflow["description"],
            trigger_examples=list(workflow.get("trigger_examples", ())),
            supported=workflow.get("supported", True),
            requires_clarification=workflow.get("requires_clarification", False),
            missing_capabilities=list(workflow.get("missing_capabilities", ())),
            decision_rule=workflow.get("decision_rule"),
        )
        previous_step_key: str | None = None
        previous_operation_id: str | None = None
        for order, step in enumerate(workflow["steps"], start=1):
            operation_id = step.get("operation_id")
            step_key = f"{workflow_id}.step.{order:02d}"
            add_node(
                "WorkflowStep",
                step_key,
                workflow_id=workflow_id,
                order=order,
                operation_id=operation_id,
                purpose=step.get("purpose"),
                condition=step.get("condition"),
                output=step.get("output"),
                query_hint_json=_json_dumps(step.get("query_hint")),
                path_hint_json=_json_dumps(step.get("path_hint")),
                body_hint_json=_json_dumps(step.get("body_hint")),
                approval_required=bool(step.get("approval_required", False)),
                requires_user_confirmation=bool(step.get("requires_user_confirmation", False)),
                missing_capability=step.get("missing_capability"),
                status=step.get("status"),
            )
            add_rel("Workflow", workflow_id, "HAS_WORKFLOW_STEP", "WorkflowStep", step_key, order=order)
            if previous_step_key:
                add_rel("WorkflowStep", previous_step_key, "NEXT_STEP", "WorkflowStep", step_key, workflow=workflow_id)
            previous_step_key = step_key

            if operation_id and _has_operation(catalog, operation_id):
                add_rel(
                    "Workflow",
                    workflow_id,
                    "HAS_STEP",
                    "Operation",
                    operation_id,
                    order=order,
                    purpose=step.get("purpose"),
                    condition=step.get("condition"),
                    approval_required=bool(step.get("approval_required", False)),
                    requires_user_confirmation=bool(step.get("requires_user_confirmation", False)),
                )
                add_rel("WorkflowStep", step_key, "USES_OPERATION", "Operation", operation_id, order=order)
                if previous_operation_id:
                    add_rel(
                        "Operation",
                        previous_operation_id,
                        "NEXT_OPERATION",
                        "Operation",
                        operation_id,
                        workflow=workflow_id,
                    )
                previous_operation_id = operation_id


def _add_field(
    add_node: Any,
    add_rel: Any,
    operation: Operation,
    field_path: str,
    field_schema: dict[str, Any],
    required: bool,
    *,
    location: str,
    documentation_source: str,
    rel_type: str,
) -> str:
    key = f"{operation.operation_id}.{location}.{field_path}"
    add_node(
        "Field",
        key,
        name=field_path,
        location=location,
        required=required,
        schema_type=field_schema.get("type"),
        schema_ref=field_schema.get("$ref"),
        enum=field_schema.get("enum"),
        description=field_schema.get("description"),
        documentation_source=documentation_source,
        schema_json=_json_dumps(field_schema),
    )
    add_rel("Operation", operation.operation_id, rel_type, "Field", key)
    referenced = _field_entity_reference(field_path)
    if referenced:
        add_node("Entity", referenced, name=referenced)
        add_rel("Field", key, "REFERENCES_ENTITY", "Entity", referenced)
    return key


def _risk_key(operation: Operation) -> str:
    if operation.destructive:
        return "destructive_requires_human_approval"
    if operation.is_mutation:
        return "mutation_requires_validation"
    return "read_only"


def _operation_definition(catalog: OperationCatalog, operation: Operation) -> dict[str, Any]:
    return catalog.spec.get("paths", {}).get(operation.path, {}).get(operation.method.lower(), {})


def _response_schemas(operation_definition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for status, response in operation_definition.get("responses", {}).items():
        schema = response.get("content", {}).get("application/json", {}).get("schema")
        if schema:
            schemas[str(status)] = schema
    return schemas


def _response_description(operation_definition: dict[str, Any], status: str) -> str | None:
    response = operation_definition.get("responses", {}).get(status, {})
    return response.get("description")


def _body_documentation_status(operation: Operation) -> str:
    if operation.request_schema is not None:
        return "openapi_request_schema"
    if operation.operation_id in SUMMARY_BODY_FIELD_HINTS:
        return "summary_hint_only"
    if operation.is_mutation:
        return "not_documented_in_openapi"
    return "no_body_expected"


def _has_operation(catalog: OperationCatalog, operation_id: str) -> bool:
    try:
        catalog.get(operation_id)
        return True
    except KeyError:
        return False


def _has_parameter(operation: Operation, location: str, name: str) -> bool:
    return any(parameter.location == location and parameter.name == name for parameter in operation.parameters)


def _iter_schema_fields(
    schema: dict[str, Any] | None,
    prefix: str = "",
) -> list[tuple[str, dict[str, Any], bool]]:
    if not isinstance(schema, dict):
        return []

    fields: list[tuple[str, dict[str, Any], bool]] = []
    schema_type = schema.get("type")

    if schema_type == "array":
        item_prefix = f"{prefix}[]" if prefix else "items[]"
        fields.extend(_iter_schema_fields(schema.get("items"), item_prefix))
        return fields

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return fields

    required_fields = set(schema.get("required", []))
    for name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue
        field_path = f"{prefix}.{name}" if prefix else name
        required = name in required_fields
        fields.append((field_path, field_schema, required))
        if field_schema.get("type") == "array":
            fields.extend(_iter_schema_fields(field_schema.get("items"), f"{field_path}[]"))
        else:
            fields.extend(_iter_schema_fields(field_schema, field_path))
    return fields


def _schema_contains_field(schema: dict[str, Any] | None, field_path: str) -> bool:
    return any(path == field_path for path, _, _ in _iter_schema_fields(schema))


def _field_entity_reference(field_path: str) -> str | None:
    normalized = field_path.lower().replace("[]", "").replace(".", "_").replace("-", "_")
    compact = normalized.replace("_", "")
    if compact.endswith("clientid") or normalized in {"client", "client_id"}:
        return "Client"
    if compact.endswith("reservationid") or normalized in {"reservation", "reservation_id"}:
        return "Reservation"
    if compact.endswith("orderid") or normalized in {"order", "order_id"}:
        return "Order"
    if compact.endswith("couponid") or normalized in {"coupon", "coupon_id"}:
        return "Coupon"
    if compact.endswith("groupid") or normalized in {"clientgroup", "client_group", "group_id"}:
        return "ClientGroup"
    if compact.endswith("storeid") or normalized in {"store", "store_id"}:
        return "Store"
    if compact.endswith("productid") or normalized in {"product", "product_id"}:
        return "Product"
    if compact.endswith("promotionid") or normalized in {"promotion", "promotion_id"}:
        return "Promotion"
    if compact.endswith("instanceid") or normalized in {"couponinstance", "coupon_instance", "instance_id"}:
        return "CouponInstance"
    if compact.endswith("areaid") or normalized in {"area", "area_id"}:
        return "Area"
    if compact.endswith("tableid") or normalized in {"table", "table_id"}:
        return "Table"
    return None


def _neo4j_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {key: _neo4j_value(value) for key, value in properties.items() if value is not None}


def _embedding_text(label: str, key: str, properties: dict[str, Any]) -> str:
    fields_by_label = {
        "Domain": ("name",),
        "Entity": ("name",),
        "Field": (
            "name",
            "location",
            "required",
            "schema_type",
            "enum",
            "description",
            "documentation_source",
        ),
        "Operation": (
            "operation_id",
            "method",
            "path",
            "summary",
            "description",
            "domain",
            "destructive",
            "is_mutation",
            "body_documentation_status",
            "response_statuses",
        ),
        "Parameter": (
            "name",
            "location",
            "required",
            "schema_type",
            "description",
            "documentation_source",
        ),
        "RiskPolicy": ("name",),
        "Workflow": (
            "name",
            "description",
            "trigger_examples",
            "supported",
            "missing_capabilities",
            "decision_rule",
        ),
        "WorkflowStep": (
            "workflow_id",
            "order",
            "operation_id",
            "purpose",
            "condition",
            "output",
            "query_hint_json",
            "path_hint_json",
            "body_hint_json",
            "approval_required",
            "requires_user_confirmation",
            "missing_capability",
            "status",
        ),
    }
    parts = [f"label: {label}", f"key: {key}"]
    for field in fields_by_label.get(label, ()):
        value = properties.get(field)
        if value not in (None, "", [], ()):
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def _neo4j_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
        return value
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
