from __future__ import annotations

from src.crm_agent.ollama import check_ollama_running


def main() -> None:
    ok, models = check_ollama_running()
    if ok:
        print("Ollama est actif.")
        print(f"Modeles disponibles: {models}")
    else:
        print("Ollama non accessible. Lance: ollama serve")


if __name__ == "__main__":
    main()
