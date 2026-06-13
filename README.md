# Agent MCP CRM Bitext

Ce projet est une version modulaire du notebook `diallo_agent_mcp_crm_bitext.ipynb`.
Il prépare le dataset Bitext Customer Support, entraîne un classificateur
d'intentions, mappe les intentions vers des tools MCP CRM, puis expose un agent
conversationnel via FastAPI.

## Structure

- `src/crm_agent/data.py` : chargement, nettoyage, split et format JSONL.
- `src/crm_agent/classifier.py` : modèle TF-IDF + LogisticRegression.
- `src/crm_agent/mcp_mapping.py` : mapping intent Bitext vers tools MCP.
- `src/crm_agent/tools.py` : stubs fonctionnels des tools CRM.
- `src/crm_agent/agent.py` : orchestration classification -> tool -> réponse LLM.
- `src/crm_agent/ollama.py` : helpers de connexion Ollama.
- `scripts/` : scripts de préparation, entraînement, évaluation et démo.
- `api.py` : API FastAPI.
- `Dockerfile` : conteneurisation de l'API.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place le fichier CSV Bitext à la racine du projet sous le nom :

```text
Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
```

Tu peux aussi fournir son chemin explicitement au script de préparation.

## Préparer les données

```bash
python scripts/prepare_data.py --csv-path "chemin\vers\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
```

Fichiers générés :

- `bitext_train.csv`
- `bitext_val.csv`
- `bitext_test.csv`
- `bitext_train_formatted.jsonl`
- `bitext_val_formatted.jsonl`

## Entraîner le classificateur léger

```bash
python scripts/train_classifier.py
```

Le modèle est sauvegardé dans `intent_classifier.pkl`.

## Évaluer

```bash
python scripts/evaluate_classifier.py --plot
```

Avec `--plot`, une matrice de confusion `confusion_matrix.png` est générée.

## Tester l'agent

Sans Ollama :

```bash
python scripts/demo_agent.py --no-ollama
```

Avec Ollama :

```bash
ollama serve
python scripts/check_ollama.py
python scripts/demo_agent.py --model qwen2.5:3b
```

## Lancer l'API

En mode LLM Ollama :

```bash
uvicorn api:app --reload --port 8000
```

En mode fallback sans Ollama :

```bash
$env:CRM_AGENT_USE_OLLAMA="false"
uvicorn api:app --reload --port 8000
```

Endpoints utiles :

- `GET /health`
- `POST /chat`
- `DELETE /session/{session_id}`
- `GET /intents`

Exemple :

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"I need to cancel my order #12345\",\"session_id\":\"demo\"}"
```

## Modèle Transformer optionnel

Le notebook contenait aussi une alternative DistilBERT. Elle est conservée dans :

```bash
python scripts/train_transformer_classifier.py
```

Cette étape est plus lourde que le classificateur TF-IDF et peut nécessiter un
environnement GPU ou plus de temps CPU.

## Docker

Après avoir généré `intent_classifier.pkl` :

```bash
docker build -t agent-crm .
docker run -p 8000:8000 -e CRM_AGENT_USE_OLLAMA=false agent-crm
```
