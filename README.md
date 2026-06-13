# Agent MCP CRM Bitext

Ce projet est une version modulaire du notebook `diallo_agent_mcp_crm_bitext.ipynb`.
Il prépare le dataset Bitext Customer Support, entraîne un classificateur
d'intentions, mappe les intentions vers des tools MCP CRM, puis expose un agent
conversationnel via FastAPI.

## Structure

- `data/` : dataset source et artefacts générés.
- `src/data.py` : chargement, nettoyage, split et format JSONL.
- `src/classifier.py` : modèle TF-IDF + LogisticRegression.
- `src/mcp_mapping.py` : mapping intent Bitext vers tools MCP.
- `src/tools.py` : stubs fonctionnels des tools CRM.
- `src/agent.py` : orchestration classification -> tool -> réponse LLM.
- `src/ollama.py` : helpers de connexion Ollama.
- `scripts/` : scripts de préparation, entraînement, évaluation et démo.
- `api.py` : API FastAPI.
- `Dockerfile` : conteneurisation de l'API.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Le CSV Bitext est attendu dans `data/` sous le nom :

```text
data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11_1.csv
```

Vous pouvez aussi fournir son chemin explicitement au script de préparation.

## Préparer les données

```bash
python scripts/prepare_data.py --csv-path "data\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11_1.csv"
```

Fichiers générés dans `data/` :

- `data/bitext_train.csv`
- `data/bitext_val.csv`
- `data/bitext_test.csv`
- `data/bitext_train_formatted.jsonl`
- `data/bitext_val_formatted.jsonl`

## Entraîner le classificateur léger

```bash
python scripts/train_classifier.py
```

Le modèle est sauvegardé dans `data/intent_classifier.pkl`.

Cette option est recommandée sur CPU pour ce projet : elle s'entraîne très vite
et donne une baseline forte pour la classification d'intentions.

## Évaluer

```bash
python scripts/evaluate_classifier.py --plot
```

Avec `--plot`, une matrice de confusion `confusion_matrix.png` est générée.

Pour évaluer le classificateur Transformer après entraînement :

```bash
python scripts/evaluate_classifier.py --classifier-backend transformer
```

## Tester l'agent

Sans Ollama :

```bash
python scripts/demo_agent.py --no-ollama
```

Avec le classificateur Transformer :

```bash
python scripts/demo_agent.py --no-ollama --classifier-backend transformer
```

Avec Ollama :

```bash
ollama serve
ollama pull qwen2.5:1.5b
python scripts/check_ollama.py
python scripts/demo_agent.py --model qwen2.5:1.5b
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

Pour utiliser le classificateur Transformer dans l'API :

```bash
$env:CRM_CLASSIFIER_BACKEND="transformer"
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

Le notebook contenait aussi une alternative Transformer. Le script utilise par
défaut un mini-BERT (`google/bert_uncased_L-2_H-128_A-2`) pour rester viable sur
CPU :

```bash
python scripts/train_transformer_classifier.py
```

Par défaut, le modèle Transformer est sauvegardé dans `data/bert_intent_model`.

Commandes utiles pour CPU :

```bash
# Test très rapide, faible qualité, utile pour vérifier l'installation
python scripts/train_transformer_classifier.py --sample-size 300 --epochs 0.1 --batch-size 32 --max-length 64

# Entraînement CPU plus sérieux
python scripts/train_transformer_classifier.py --epochs 1 --batch-size 16 --max-length 96

# Meilleure qualité mais plus lent
python scripts/train_transformer_classifier.py --epochs 2 --batch-size 8 --max-length 128 --fine-tune-base
```

Conclusion pratique : BERT/mini-BERT réduit fortement le coût par rapport au
fine-tuning d'un LLM génératif pour Ollama. En revanche, sur CPU, il reste plus
lourd et souvent moins intéressant que `scripts/train_classifier.py` pour une
tâche simple de classification d'intentions. La solution recommandée est donc :
TF-IDF + LogisticRegression pour l'intention, tools MCP pour les actions, et
Ollama uniquement en inférence optionnelle pour reformuler les réponses.

## Docker

Après avoir généré `data/intent_classifier.pkl` :

```bash
docker build -t agent-crm .
docker run -p 8000:8000 -e CRM_AGENT_USE_OLLAMA=false agent-crm
```
