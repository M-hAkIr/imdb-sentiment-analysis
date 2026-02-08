import os
from enum import StrEnum
from pathlib import Path


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


LABEL_MAPPING: dict[str, Sentiment] = {
    "POSITIVE": Sentiment.POSITIVE,
    "POS": Sentiment.POSITIVE,
    "NEGATIVE": Sentiment.NEGATIVE,
    "NEG": Sentiment.NEGATIVE,
}

# Text limits
MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = 5000
DEFAULT_BATCH_SIZE = 8
MAX_TOKEN_LENGTH = 512
TEXT_TRUNCATE_LENGTH = 100

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Models
MODELS: dict[str, str] = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta": "siebert/sentiment-roberta-large-english",
}
DEFAULT_MODEL = "distilbert"

# Runtime (overridable via env)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))

# Data
DATA_FILE = DATA_DIR / "IMDB-movie-reviews.csv"
CSV_ENCODING = "ISO-8859-1"
CSV_DELIMITER = ";"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ALLOW_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:7860"
).split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

# Demo
DEMO_HOST = os.getenv("DEMO_HOST", "0.0.0.0")
DEMO_PORT = int(os.getenv("DEMO_PORT", "7860"))
DEMO_SHARE = os.getenv("DEMO_SHARE", "false").lower() == "true"

def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)
