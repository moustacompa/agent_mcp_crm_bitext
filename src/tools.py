"""Demo CRM tools used by the agent.

These methods are functional stubs. Replace them with real CRM integrations
when connecting the agent to production systems.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any


class MCPTools:
    """Simple dispatcher and CRM tool stubs."""

    @staticmethod
    def cancel_order(order_id: str = "unknown") -> dict[str, Any]:
        return {
            "status": "success",
            "message": f"Commande {order_id} annulee avec succes.",
            "refund_delay": "5-10 jours ouvres",
        }

    @staticmethod
    def get_order_status(order_id: str = "unknown") -> dict[str, Any]:
        statuses = ["En preparation", "Expedie", "En transit", "Livre"]
        return {
            "order_id": order_id,
            "status": random.choice(statuses),
            "estimated_delivery": (datetime.now() + timedelta(days=random.randint(1, 5))).strftime("%d/%m/%Y"),
        }

    @staticmethod
    def modify_order(order_id: str = "unknown", changes: str = "") -> dict[str, Any]:
        return {"status": "success", "order_id": order_id, "changes": changes or "changements demandes"}

    @staticmethod
    def create_order(product_id: str = "demo-product", quantity: int | str = 1) -> dict[str, Any]:
        order_id = f"ORD-{random.randint(10000, 99999)}"
        return {"status": "created", "order_id": order_id, "product_id": product_id, "quantity": quantity}

    @staticmethod
    def get_delivery_options(address: str = "") -> dict[str, Any]:
        return {
            "address": address or "adresse non precisee",
            "options": [
                {"name": "standard", "delay": "3-5 jours", "price": "gratuit"},
                {"name": "express", "delay": "24-48h", "price": "9.99 EUR"},
            ],
        }

    @staticmethod
    def get_delivery_estimate(order_id: str = "unknown") -> dict[str, Any]:
        return {
            "order_id": order_id,
            "estimated_delivery": (datetime.now() + timedelta(days=3)).strftime("%d/%m/%Y"),
        }

    @staticmethod
    def process_refund(order_id: str = "unknown", reason: str = "") -> dict[str, Any]:
        return {
            "status": "initiated",
            "order_id": order_id,
            "reason": reason or "raison non precisee",
            "refund_amount": "49.99 EUR",
            "delay": "5-10 jours ouvres",
        }

    @staticmethod
    def initiate_return(order_id: str = "unknown", items: str = "") -> dict[str, Any]:
        return {"status": "created", "order_id": order_id, "items": items or "articles a confirmer"}

    @staticmethod
    def update_shipping_address(order_id: str = "unknown", new_address: str = "") -> dict[str, Any]:
        return {
            "status": "success",
            "order_id": order_id,
            "message": "Adresse de livraison mise a jour.",
            "new_address": new_address or "adresse a confirmer",
        }

    @staticmethod
    def get_refund_policy() -> dict[str, Any]:
        return {
            "policy": "Retours acceptes sous 30 jours apres reception. Remboursement sous 5-10 jours ouvres."
        }

    @staticmethod
    def create_account(email: str = "", name: str = "") -> dict[str, Any]:
        account_id = f"ACC-{random.randint(10000, 99999)}"
        return {"status": "created", "account_id": account_id, "email": email, "name": name}

    @staticmethod
    def delete_account(account_id: str = "unknown") -> dict[str, Any]:
        return {"status": "scheduled", "account_id": account_id, "delay": "24h"}

    @staticmethod
    def update_account(account_id: str = "unknown", fields: str = "") -> dict[str, Any]:
        return {"status": "success", "account_id": account_id, "updated_fields": fields or "a confirmer"}

    @staticmethod
    def switch_account_type(account_id: str = "unknown", new_type: str = "standard") -> dict[str, Any]:
        return {"status": "success", "account_id": account_id, "new_type": new_type}

    @staticmethod
    def reset_password(email: str = "") -> dict[str, Any]:
        return {"status": "sent", "message": f"Email de reinitialisation envoye a {email or 'adresse a confirmer'}."}

    @staticmethod
    def get_payment_methods() -> dict[str, Any]:
        return {"methods": ["card", "paypal", "bank_transfer"]}

    @staticmethod
    def report_payment_issue(order_id: str = "unknown", issue_type: str = "") -> dict[str, Any]:
        ticket_id = f"PAY-{random.randint(10000, 99999)}"
        return {"status": "created", "ticket_id": ticket_id, "order_id": order_id, "issue_type": issue_type}

    @staticmethod
    def report_wrong_item(order_id: str = "unknown", description: str = "") -> dict[str, Any]:
        ticket_id = f"WRG-{random.randint(10000, 99999)}"
        return {"status": "created", "ticket_id": ticket_id, "order_id": order_id, "description": description}

    @staticmethod
    def get_support_contact() -> dict[str, Any]:
        return {
            "phone": "+1-800-555-0100",
            "hours": "9h-18h lun-ven",
            "email": "support@example.com",
            "chat": "www.example.com/chat",
        }

    @staticmethod
    def escalate_to_human(reason: str = "") -> dict[str, Any]:
        return {
            "status": "transferred",
            "message": "Vous allez etre mis en relation avec un agent humain.",
            "reason": reason or "demande utilisateur",
            "wait_time": "~5 minutes",
        }

    @staticmethod
    def submit_complaint(subject: str = "", description: str = "") -> dict[str, Any]:
        ticket_id = f"TKT-{random.randint(10000, 99999)}"
        return {
            "status": "created",
            "ticket_id": ticket_id,
            "subject": subject,
            "description": description,
            "message": f"Reclamation enregistree sous le numero {ticket_id}.",
        }

    @staticmethod
    def submit_review(order_id: str = "unknown", rating: int | str = "", comment: str = "") -> dict[str, Any]:
        return {"status": "published", "order_id": order_id, "rating": rating, "comment": comment}

    @staticmethod
    def manage_newsletter(email: str = "", action: str = "subscribe") -> dict[str, Any]:
        return {"status": "success", "email": email, "action": action}

    @staticmethod
    def get_cancellation_fee(order_id: str = "unknown") -> dict[str, Any]:
        return {"order_id": order_id, "fee": "0 EUR", "message": "Aucun frais detecte pour cette commande."}

    @staticmethod
    def get_invoice(order_id: str = "unknown") -> dict[str, Any]:
        return {
            "order_id": order_id,
            "invoice_url": f"https://example.com/invoices/{order_id}.pdf",
            "date": datetime.now().strftime("%d/%m/%Y"),
        }

    @staticmethod
    def track_refund_status(order_id: str = "unknown") -> dict[str, Any]:
        return {"order_id": order_id, "status": "processing", "estimated_payment": "5 jours ouvres"}

    @staticmethod
    def execute(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        method = getattr(MCPTools, tool_name, None)
        if method is None:
            return {"status": "error", "message": f"Tool {tool_name} non trouve."}
        return method(**kwargs)
