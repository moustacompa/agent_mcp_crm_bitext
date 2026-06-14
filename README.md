# Agent MCP CRM Bitext

Ce projet est une version modulaire du notebook `diallo_agent_mcp_crm_bitext.ipynb`.
Il prépare le dataset Bitext Customer Support, entraîne un classificateur
d'intentions, mappe les intentions vers des tools MCP CRM, puis expose un agent
conversationnel via FastAPI.

## Processus

### Étape 1 : préparer l'environnement

Créer un environnement virtuel et installer les dépendances :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Étape 2 : placer le dataset

Le CSV Bitext est attendu dans `data/` sous le nom :

```text
data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11_1.csv
```

Vous pouvez aussi fournir un autre chemin avec l'option `--csv-path` à l'étape suivante.

### Étape 3 : préparer les données

Nettoyer les templates, créer les splits train/validation/test et générer les fichiers JSONL :

```bash
python scripts/prepare_data.py --csv-path "data\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11_1.csv"
```

Fichiers générés dans `data/` :

- `data/bitext_train.csv`
- `data/bitext_val.csv`
- `data/bitext_test.csv`
- `data/bitext_train_formatted.jsonl`
- `data/bitext_val_formatted.jsonl`

### Étape 4 : entraîner le classificateur d'intentions

Entraîner le modèle léger recommandé pour CPU :

```bash
python scripts/train_classifier.py
```

Le modèle est sauvegardé dans :

```text
data/intent_classifier.pkl
```

Cette option est recommandée sur CPU pour ce projet : elle s'entraîne très vite
et donne une baseline forte pour la classification d'intentions.

### Étape 5 : évaluer le classificateur

```bash
python scripts/evaluate_classifier.py --plot
```

Avec `--plot`, une matrice de confusion `confusion_matrix.png` est générée.

### Étape 6 : installer le modèle Ollama léger

Le modèle Ollama recommandé pour cet environnement est `qwen2.5:1.5b`.

```bash
ollama serve
ollama pull qwen2.5:1.5b
python scripts/check_ollama.py
```

### Étape 7 : tester l'agent en local

Sans Ollama, avec une réponse fallback :

```bash
python scripts/demo_agent.py --no-ollama
```

Avec Ollama :

```bash
python scripts/demo_agent.py --model qwen2.5:1.5b
```

### Étape 8 : lancer l'API FastAPI

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

### Étape 9 : conteneuriser l'API

Après avoir généré `data/intent_classifier.pkl` :

```bash
docker build -t agent-crm .
docker run -p 8000:8000 -e CRM_AGENT_USE_OLLAMA=false agent-crm
```

## Variante Transformer optionnelle

Le projet contient aussi un classificateur Transformer léger basé sur un mini-BERT :
`google/bert_uncased_L-2_H-128_A-2`.

Cette variante est disponible, mais elle est moins recommandée que TF-IDF sur CPU
pour ce dataset.

### Entraîner le Transformer

```bash
python scripts/train_transformer_classifier.py
```

Par défaut, le modèle Transformer est sauvegardé dans :

```text
data/bert_intent_model
```

Commandes utiles pour CPU :

```bash
# Test très rapide, faible qualité, utile pour vérifier l'installation
python scripts/train_transformer_classifier.py --sample-size 300 --epochs 0.1 --batch-size 32 --max-length 64

# Entraînement CPU plus sérieux
python scripts/train_transformer_classifier.py --epochs 1 --batch-size 16 --max-length 96

# Meilleure qualité mais plus lent
python scripts/train_transformer_classifier.py --epochs 2 --batch-size 8 --max-length 128 --fine-tune-base
```

### Évaluer le Transformer

```bash
python scripts/evaluate_classifier.py --classifier-backend transformer
```

### Utiliser le Transformer dans la démo

```bash
python scripts/demo_agent.py --no-ollama --classifier-backend transformer
```

### Utiliser le Transformer dans l'API

```bash
$env:CRM_CLASSIFIER_BACKEND="transformer"
$env:CRM_AGENT_USE_OLLAMA="false"
uvicorn api:app --reload --port 8000
```

## Structure du projet

- `data/` : dataset source et artefacts générés.
- `src/data.py` : chargement, nettoyage, split et format JSONL.
- `src/classifier.py` : modèle TF-IDF + LogisticRegression.
- `src/transformer_classifier.py` : wrapper du classificateur Transformer.
- `src/mcp_mapping.py` : mapping intent Bitext vers tools MCP.
- `src/tools.py` : stubs fonctionnels des tools CRM.
- `src/agent.py` : orchestration classification -> tool -> réponse LLM.
- `src/ollama.py` : helpers de connexion Ollama.
- `scripts/` : scripts de préparation, entraînement, évaluation et démo.
- `api.py` : API FastAPI.
- `Dockerfile` : conteneurisation de l'API.

## Recommandation

Sur CPU, la solution recommandée est :

```text
TF-IDF + LogisticRegression pour l'intention
        ↓
Tools MCP pour les actions CRM
        ↓
qwen2.5:1.5b avec Ollama pour reformuler les réponses
```

BERT/mini-BERT réduit fortement le coût par rapport au fine-tuning d'un LLM
génératif pour Ollama, mais il reste plus lourd que le classificateur TF-IDF
pour une tâche simple de classification d'intentions.
