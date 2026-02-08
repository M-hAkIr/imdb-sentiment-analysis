import logging
import sys

import pandas as pd

from src.benchmarking import BenchmarkReport, compare_models
from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    DATA_FILE,
    MODELS,
    OUTPUT_DIR,
    ensure_directories,
)
from src.models import PredictionResult, create_analyzer, normalize_label
from src.preprocessing import load_data, preprocess_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_and_preprocess() -> pd.DataFrame:
    logger.info("Loading dataset: %s", DATA_FILE)
    df = load_data(DATA_FILE)
    logger.info("Loaded %d reviews", len(df))

    counts = df['sentiment'].value_counts()
    logger.info("Distribution: %s", dict(counts))

    df = preprocess_dataset(df)
    valid = df['is_valid'].sum()
    logger.info("Valid reviews: %d/%d", valid, len(df))

    df = df[df['is_valid']].reset_index(drop=True)
    return df


def run_predictions(model_name: str, texts: list[str],
                    ground_truth: pd.Series) -> BenchmarkReport:
    logger.info("Running %s...", model_name)

    analyzer = create_analyzer(model_name, batch_size=BATCH_SIZE)
    predictions = analyzer.predict_batch(texts)

    report = BenchmarkReport(model_name, predictions, ground_truth)
    metrics = report.calculate_metrics()

    logger.info(
        "%s -- acc: %.4f, f1: %.4f, avg: %.2fms",
        model_name, metrics['accuracy'],
        metrics['f1_weighted'], metrics['avg_inference_time_ms'],
    )

    report.generate_report(output_dir=str(OUTPUT_DIR))
    return report


def save_predictions(df: pd.DataFrame, predictions: list[PredictionResult],
                     model_name: str) -> None:
    output_df = pd.DataFrame({
        'review': df['cleaned_review'],
        'true_sentiment': df['sentiment'],
        'predicted_sentiment': [normalize_label(p.label).value for p in predictions],
        'confidence': [p.confidence for p in predictions],
        'avg_inference_time_ms': [p.inference_time * 1000 for p in predictions],
    })

    path = DATA_DIR / f"predictions_{model_name}.csv"
    output_df.to_csv(path, index=False, encoding='utf-8')
    logger.info("Saved predictions: %s", path)


def main():
    try:
        logger.info("Starting IMDB Sentiment Analysis Benchmarking")
        ensure_directories()

        df = load_and_preprocess()

        reports = []
        for model_name in MODELS:
            try:
                report = run_predictions(
                    model_name=model_name,
                    texts=df['cleaned_review'].tolist(),
                    ground_truth=df['sentiment'],
                )
                save_predictions(df, report.predictions, model_name)
                reports.append(report)
            except Exception as e:
                logger.error("Error with %s: %s", model_name, e)
                continue

        if len(reports) >= 2:
            comparison = compare_models(reports)
            logger.info("Model comparison:\n%s", comparison.to_string(index=False))

            comp_path = OUTPUT_DIR / "model_comparison.csv"
            comparison.to_csv(comp_path, index=False)

        logger.info("Done. Results in %s", OUTPUT_DIR)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
