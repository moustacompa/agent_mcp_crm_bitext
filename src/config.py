"""Shared project configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11_1.csv"
TRAIN_CSV = DATA_DIR / "bitext_train.csv"
VAL_CSV = DATA_DIR / "bitext_val.csv"
TEST_CSV = DATA_DIR / "bitext_test.csv"
TRAIN_JSONL = DATA_DIR / "bitext_train_formatted.jsonl"
VAL_JSONL = DATA_DIR / "bitext_val_formatted.jsonl"
CLASSIFIER_PATH = DATA_DIR / "intent_classifier.pkl"
TRANSFORMER_CLASSIFIER_DIR = DATA_DIR / "bert_intent_model"

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
FALLBACK_OLLAMA_MODEL = "qwen2.5:1.5b"
