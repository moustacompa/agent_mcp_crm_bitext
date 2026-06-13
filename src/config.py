"""Shared project configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11_1.csv"
TRAIN_CSV = PROJECT_ROOT / "bitext_train.csv"
VAL_CSV = PROJECT_ROOT / "bitext_val.csv"
TEST_CSV = PROJECT_ROOT / "bitext_test.csv"
TRAIN_JSONL = PROJECT_ROOT / "bitext_train_formatted.jsonl"
VAL_JSONL = PROJECT_ROOT / "bitext_val_formatted.jsonl"
CLASSIFIER_PATH = PROJECT_ROOT / "intent_classifier.pkl"

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "agent-crm"
FALLBACK_OLLAMA_MODEL = "qwen2.5:3b"
