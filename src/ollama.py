"""Ollama helpers."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from .config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL


def check_ollama_running(base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout: int = 3) -> tuple[bool, list[str]]:
    try:
        response = urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout)
        data = json.loads(response.read())
        models = [model["name"] for model in data.get("models", [])]
        return True, models
    except Exception:
        return False, []


def create_ollama_llm(
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = 0.3,
    num_ctx: int = 2048,
) -> Any:
    from langchain_ollama import OllamaLLM

    return OllamaLLM(model=model, base_url=base_url, temperature=temperature, num_ctx=num_ctx)
