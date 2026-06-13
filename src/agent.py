"""Agent orchestration: intent classification, MCP tool call and LLM answer."""

from __future__ import annotations

import re
from typing import Any

from .mcp_mapping import INTENT_TO_TOOL
from .tools import MCPTools


SYSTEM_PROMPT = """Tu es un agent CRM specialise dans le support client.
Lorsque tu recois un message, tu dois :
1. Identifier clairement la demande de l'utilisateur
2. Utiliser les informations du tool appele pour repondre precisement
3. Repondre de maniere professionnelle, empathique et concise
4. Proposer d'autres actions si pertinent

Contexte du tool execute : {tool_result}"""


class CRMAgent:
    """CRM agent: intent classification -> MCP tool -> optional LLM response."""

    def __init__(
        self,
        llm: Any,
        intent_classifier: Any,
        tools: MCPTools | None = None,
        intent_to_tool: dict[str, dict[str, Any]] | None = None,
        confidence_threshold: float = 0.4,
    ) -> None:
        self.llm = llm
        self.intent_classifier = intent_classifier
        self.tools = tools or MCPTools()
        self.intent_to_tool = intent_to_tool or INTENT_TO_TOOL
        self.confidence_threshold = confidence_threshold
        self.conversation_history: list[dict[str, str]] = []

    def classify_intent(self, message: str) -> tuple[str, float]:
        intent = self.intent_classifier.predict([message])[0]
        confidence = float(self.intent_classifier.predict_proba([message]).max())
        return intent, confidence

    def execute_tool(self, intent: str, message: str) -> dict[str, Any]:
        tool_config = self.intent_to_tool.get(intent)
        if not tool_config:
            return {"status": "no_tool", "message": "Aucun tool associe a cet intent."}

        kwargs = self._extract_params(message, tool_config.get("params", []))
        return self.tools.execute(tool_config["tool"], **kwargs)

    def generate_response(self, message: str, intent: str, tool_result: dict[str, Any]) -> str:
        if self.llm is None:
            return self._fallback_response(intent, tool_result)

        prompt = f"""{SYSTEM_PROMPT.format(tool_result=tool_result)}

Historique de la conversation :
{self._format_history()}

Utilisateur : {message}
Intent detecte : {intent}

Agent CRM :"""
        return self.llm.invoke(prompt)

    def chat(self, message: str) -> dict[str, Any]:
        intent, confidence = self.classify_intent(message)
        if confidence < self.confidence_threshold:
            intent = "contact_human_agent"

        tool_result = self.execute_tool(intent, message)
        response = self.generate_response(message, intent, tool_result)
        self.conversation_history.append({"user": message, "agent": response})

        return {
            "response": response,
            "intent": intent,
            "confidence": round(confidence, 3),
            "tool_called": self.intent_to_tool.get(intent, {}).get("tool", "none"),
            "tool_result": tool_result,
        }

    def reset(self) -> None:
        self.conversation_history = []

    def _format_history(self) -> str:
        if not self.conversation_history:
            return "(nouvelle conversation)"
        lines: list[str] = []
        for turn in self.conversation_history[-4:]:
            lines.append(f"Utilisateur : {turn['user']}")
            lines.append(f"Agent : {turn['agent']}")
        return "\n".join(lines)

    def _extract_params(self, message: str, params: list[str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        order_match = re.search(r"#?(ORD-?\d+|\d{4,})", message, re.IGNORECASE)
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", message)

        if "order_id" in params:
            kwargs["order_id"] = order_match.group(0) if order_match else "unknown"
        if "account_id" in params:
            kwargs["account_id"] = order_match.group(0) if order_match else "unknown"
        if "email" in params:
            kwargs["email"] = email_match.group(0) if email_match else ""
        if "reason" in params:
            kwargs["reason"] = message
        if "description" in params:
            kwargs["description"] = message
        if "subject" in params:
            kwargs["subject"] = "Demande client"
        if "new_address" in params:
            kwargs["new_address"] = message
        if "address" in params:
            kwargs["address"] = message
        if "changes" in params:
            kwargs["changes"] = message
        if "items" in params:
            kwargs["items"] = "a confirmer"
        if "product_id" in params:
            kwargs["product_id"] = "demo-product"
        if "quantity" in params:
            kwargs["quantity"] = 1
        if "fields" in params:
            kwargs["fields"] = message
        if "new_type" in params:
            kwargs["new_type"] = "standard"
        if "issue_type" in params:
            kwargs["issue_type"] = "payment_issue"
        if "rating" in params:
            kwargs["rating"] = 5
        if "comment" in params:
            kwargs["comment"] = message
        if "action" in params:
            kwargs["action"] = "subscribe"

        return kwargs

    @staticmethod
    def _fallback_response(intent: str, tool_result: dict[str, Any]) -> str:
        status = tool_result.get("status", "ok")
        return (
            f"Votre demande a ete identifiee comme '{intent}'. "
            f"Le tool CRM a retourne le statut '{status}'. "
            "Un conseiller peut completer la reponse si necessaire."
        )
