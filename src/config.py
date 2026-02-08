from pathlib import Path

# Text limits
MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = 5000

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Data
DATA_FILE = DATA_DIR / "IMDB-movie-reviews.csv"
CSV_ENCODING = "ISO-8859-1"
CSV_DELIMITER = ";"
