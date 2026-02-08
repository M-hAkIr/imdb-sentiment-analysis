import json
import logging
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import Sentiment
from src.models import PredictionResult, normalize_label

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


class BenchmarkReport:
    """Collects predictions + ground truth and generates metrics/reports."""

    def __init__(self, model_name: str, predictions: list[PredictionResult],
                 ground_truth: pd.Series):
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(ground_truth)} ground truth"
            )

        self.model_name = model_name
        self.predictions = predictions
        self.ground_truth = ground_truth

        self.pred_labels = [normalize_label(p.label).value for p in predictions]
        self.true_labels = [normalize_label(str(label)).value
                           for label in ground_truth.tolist()]
        self._cached_metrics = None

    def calculate_metrics(self) -> dict:
        if self._cached_metrics is not None:
            return self._cached_metrics

        labels = [Sentiment.POSITIVE.value, Sentiment.NEGATIVE.value]

        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.pred_labels,
            labels=labels, average=None, zero_division=0,
        )

        f1_w = f1_score(
            self.true_labels, self.pred_labels,
            labels=labels, average='weighted', zero_division=0,
        )

        accuracy = accuracy_score(self.true_labels, self.pred_labels)

        times = [p.inference_time for p in self.predictions]
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        throughput = len(self.predictions) / total_time if total_time > 0 else 0

        self._cached_metrics = {
            'model_name': self.model_name,
            'accuracy': float(accuracy),
            'f1_weighted': float(f1_w),
            'f1_positive': float(f1[0]),
            'f1_negative': float(f1[1]),
            'precision_positive': float(precision[0]),
            'precision_negative': float(precision[1]),
            'recall_positive': float(recall[0]),
            'recall_negative': float(recall[1]),
            'avg_inference_time_ms': float(avg_time * 1000),
            'total_inference_time_sec': float(total_time),
            'throughput_per_sec': float(throughput),
            'support': {
                'positive': int(support[0]),
                'negative': int(support[1]),
                'total': int(sum(support)),
            },
        }
        return self._cached_metrics

    def plot_confusion_matrix(self, save_path: str) -> None:
        labels = [Sentiment.POSITIVE.value, Sentiment.NEGATIVE.value]
        cm = confusion_matrix(self.true_labels, self.pred_labels, labels=labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive', 'Negative'],
            yticklabels=['Positive', 'Negative'],
            square=True, linewidths=1, linecolor='gray', ax=ax,
        )
        ax.set_title(f'Confusion Matrix - {self.model_name}', fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        fig.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved confusion matrix: %s", save_path)

    def generate_report(self, output_dir: str = 'outputs/') -> dict:
        metrics = self.calculate_metrics()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        safe_name = self.model_name.replace('/', '_')

        # JSON
        with open(out / f"benchmark_{safe_name}.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Markdown
        with open(out / f"benchmark_{safe_name}.md", 'w') as f:
            f.write(self._make_markdown(metrics))

        # Confusion matrix
        cm_dir = out / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        self.plot_confusion_matrix(str(cm_dir / f"{safe_name}.png"))

        logger.info("Reports saved for %s", self.model_name)
        return metrics

    def _make_markdown(self, metrics: dict) -> str:
        name = metrics['model_name']
        safe = name.replace('/', '_')

        return f"""# Benchmark Report: {name}

## Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) |
| F1 Score (Weighted) | {metrics['f1_weighted']:.4f} |
| Total Samples | {metrics['support']['total']} |

## Per-Class Metrics

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Positive | {metrics['precision_positive']:.4f} | {metrics['recall_positive']:.4f} | {metrics['f1_positive']:.4f} | {metrics['support']['positive']} |
| Negative | {metrics['precision_negative']:.4f} | {metrics['recall_negative']:.4f} | {metrics['f1_negative']:.4f} | {metrics['support']['negative']} |

## Performance

| Metric | Value |
|--------|-------|
| Avg Inference Time | {metrics['avg_inference_time_ms']:.2f} ms |
| Total Time | {metrics['total_inference_time_sec']:.2f} s |
| Throughput | {metrics['throughput_per_sec']:.1f} predictions/sec |

## Confusion Matrix

![Confusion Matrix](confusion_matrices/{safe}.png)
"""


def compare_models(reports: list[BenchmarkReport]) -> pd.DataFrame:
    """Build a comparison table across benchmark reports."""
    rows = []
    for report in reports:
        m = report.calculate_metrics()
        rows.append({
            'Model': m['model_name'],
            'Accuracy': round(m['accuracy'], 4),
            'F1 (Weighted)': round(m['f1_weighted'], 4),
            'F1 (Positive)': round(m['f1_positive'], 4),
            'F1 (Negative)': round(m['f1_negative'], 4),
            'Avg Time (ms)': round(m['avg_inference_time_ms'], 2),
            'Throughput (pred/s)': round(m['throughput_per_sec'], 1),
        })
    return pd.DataFrame(rows)
