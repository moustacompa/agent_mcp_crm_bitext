"""FastAPI entrypoint for the CRM MCP agent."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.crm_agent import CRMAgent, INTENT_TO_TOOL, MCPTools, load_classifier
from src.crm_agent.config import CLASSIFIER_PATH, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL
from src.crm_agent.ollama import create_ollama_llm


app = FastAPI(
    title="Agent MCP CRM",
    description="API de l'agent CRM base sur MCP et le dataset Bitext.",
    version="1.0.0",
)

logging.basicConfig(
    filename="agent_conversations.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    tool_called: str
    tool_result: dict
    session_id: str


sessions: dict[str, CRMAgent] = {}
classifier = load_classifier(CLASSIFIER_PATH)


def build_agent() -> CRMAgent:
    use_ollama = os.getenv("CRM_AGENT_USE_OLLAMA", "true").lower() == "true"
    llm = None
    if use_ollama:
        llm = create_ollama_llm(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        )
    return CRMAgent(llm=llm, intent_classifier=classifier, tools=MCPTools(), intent_to_tool=INTENT_TO_TOOL)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message vide")

    session_id = request.session_id or "default"
    agent = sessions.setdefault(session_id, build_agent())
    result = agent.chat(request.message)

    logging.info(
        json.dumps(
            {
                "session": session_id,
                "message": request.message,
                "intent": result["intent"],
                "confidence": result["confidence"],
                "tool_called": result["tool_called"],
            },
            ensure_ascii=False,
        )
    )

    return ChatResponse(**result, session_id=session_id)


@app.delete("/session/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    sessions.pop(session_id, None)
    return {"status": "session reinitialisee", "session_id": session_id}


@app.get("/intents")
def list_intents() -> dict[str, object]:
    return {"intents": list(classifier.classes_), "total": len(classifier.classes_)}
