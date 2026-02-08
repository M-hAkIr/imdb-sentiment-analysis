import re
from pathlib import Path

import pandas as pd

from src.config import CSV_DELIMITER, CSV_ENCODING, MAX_TEXT_LENGTH, MIN_TEXT_LENGTH


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load IMDB reviews from a semicolon-delimited CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=CSV_DELIMITER,
        encoding=CSV_ENCODING,
        engine='python',
        skipinitialspace=True,
    )

    required = {'review', 'sentiment'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}, got: {list(df.columns)}")

    return df


def clean_html_tags(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return clean_html_tags(text)


def validate_text(text: str, min_length: int = MIN_TEXT_LENGTH,
                  max_length: int = MAX_TEXT_LENGTH) -> bool:
    return isinstance(text, str) and min_length <= len(text.strip()) <= max_length


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean reviews and flag valid rows."""
    df = df.copy()
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    df['is_valid'] = df['cleaned_review'].apply(validate_text)
    return df
