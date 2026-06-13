from __future__ import annotations

import argparse

import bootstrap  # noqa: F401

from src import CRMAgent, INTENT_TO_TOOL, MCPTools, load_classifier
from src.config import DEFAULT_OLLAMA_BASE_URL, FALLBACK_OLLAMA_MODEL
from src.ollama import create_ollama_llm


MESSAGES = [
    "I need to cancel my order #12345",
    "Where is my package? I placed order ORD-67890 a week ago",
    "I received the wrong item, I want a refund",
    "Can I speak to a human please?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local CRM agent demo.")
    parser.add_argument("--no-ollama", action="store_true", help="Use fallback responses without LLM.")
    parser.add_argument("--model", default=FALLBACK_OLLAMA_MODEL, help="Ollama model name.")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Ollama base URL.")
    parser.add_argument("--classifier-backend", choices=["tfidf", "transformer"], default="tfidf")
    args = parser.parse_args()

    llm = None if args.no_ollama else create_ollama_llm(model=args.model, base_url=args.base_url)
    if args.classifier_backend == "transformer":
        from src.transformer_classifier import load_transformer_classifier

        classifier = load_transformer_classifier()
    else:
        classifier = load_classifier()
    agent = CRMAgent(llm=llm, intent_classifier=classifier, tools=MCPTools(), intent_to_tool=INTENT_TO_TOOL)

    for message in MESSAGES:
        print(f"Utilisateur: {message}")
        try:
            result = agent.chat(message)
        except Exception as exc:
            print(f"Erreur LLM: {exc}")
            agent.llm = None
            result = agent.chat(message)
        print(f"Intent: {result['intent']} ({result['confidence']:.1%})")
        print(f"Tool: {result['tool_called']}")
        print(f"Resultat tool: {result['tool_result']}")
        print(f"Reponse: {result['response']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
